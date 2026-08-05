"""cp_12 · 场景用例：v0 验收标准的可执行版本。

S1/S2/S3 就是 v0 的三条验收标准（见 functional_overview.md）：
  S1 OOM 能救回来（不是"重启了"就算，必须跑完全部 epoch）
  S2 kill -9 能续训
  S3 代码错误 0 次重启
"""

from __future__ import annotations

import sys

import pytest
from conftest import make_watchdog, train_cmd


# ---------- S1: OOM 救回 ----------

def test_s1_oom_recovered_and_completes(workdir, contract_file):
    """条件性 OOM：guardian 减 batch 后训练必须真的跑完 5 epoch。"""
    wd = make_watchdog(workdir, contract_file())
    cmd = train_cmd(
        workdir, "--epochs", "5", "--batch_size", "64",
        "--fail-mode", "oom_if_batch_gt", "--fail-threshold", "16",
    )

    result = wd.run(cmd)

    assert result["status"] == "completed", f"训练未跑完: {result}"
    assert wd.restarts, "应至少重启一次"
    # 每次重启都真的降了 batch
    for rec in wd.restarts:
        assert rec.trigger == "crash"
        assert "dataloader.batch_size" in rec.applied, f"未调整 batch: {rec.to_dict()}"
    # 最终 batch 降到阈值以下
    final_bs = int(wd.restarts[-1].applied["dataloader.batch_size"]["to"])
    assert final_bs <= 16, f"最终 batch={final_bs}，未降到阈值以下"
    # 训练确实跑到了最后一个 epoch
    log = (workdir / "logs" / "train.log").read_text(encoding="utf-8")
    assert "epoch 4" in log, "最后一个 epoch 没有跑到"


def test_s1_restart_records_cmd_diff(workdir, contract_file):
    """改写 diff 必须落盘，人能一眼看出改了什么。

    用 --fail-at 1 让 epoch 0 先产出 cp_0，这样才有回滚点可断言。
    """
    wd = make_watchdog(workdir, contract_file())
    wd.run(train_cmd(workdir, "--epochs", "4", "--batch_size", "64", "--fail-at", "1",
                     "--fail-mode", "oom_if_batch_gt", "--fail-threshold", "16"))
    rec = wd.restarts[0].to_dict()
    assert "--batch_size 64" in rec["cmd_before"]
    assert "--batch_size 32" in rec["cmd_after"]
    assert rec["resumed_from"] is not None, "epoch 0 已存过 cp_0，应能回滚"
    assert "cp_0" in rec["resumed_from"]


def test_s1_no_checkpoint_yet_restarts_from_scratch(workdir, contract_file):
    """epoch 0 就失败、还没有任何 checkpoint 时，从头重启而非报错。"""
    wd = make_watchdog(workdir, contract_file())
    result = wd.run(train_cmd(workdir, "--epochs", "3", "--batch_size", "64",
                              "--fail-mode", "oom_if_batch_gt", "--fail-threshold", "16"))
    assert result["status"] == "completed"
    first = wd.restarts[0]
    assert first.resumed_from is None, "无 ckpt 时应为 None"
    assert "--resume" not in first.cmd_after, "无 ckpt 时不应加 --resume"


# ---------- S2: kill -9 续训 ----------

def test_s2_sigkill_resumes_unchanged(workdir, contract_file):
    """被强杀后参数不变从最近 ckpt 续训，最终跑完。"""
    wd = make_watchdog(workdir, contract_file())
    cmd = train_cmd(workdir, "--epochs", "5", "--batch_size", "64", "--self-kill-at", "2")

    result = wd.run(cmd)

    assert result["status"] == "completed", f"未能续训跑完: {result}"
    assert len(wd.restarts) >= 1
    rec = wd.restarts[0]
    assert rec.trigger == "crash"
    # 关键：不应出现 batch 调整（kill 不是 OOM）
    assert rec.applied == {}, f"kill -9 不应调整参数，实际: {rec.applied}"
    assert "--resume" in rec.cmd_after


# ---------- S3: 代码错误不反复重启 ----------

def test_s3_code_error_zero_restarts(workdir, contract_file):
    """TypeError 必须 0 次重启，直接判不可恢复。"""
    wd = make_watchdog(workdir, contract_file())
    cmd = train_cmd(workdir, "--epochs", "5", "--fail-at", "2", "--fail-mode", "type_error")

    result = wd.run(cmd)

    assert result["status"] == "failed"
    assert result["crash"] == "code"
    assert wd.restarts == [], f"代码错误不应重启，实际重启 {len(wd.restarts)} 次"
    assert wd.retry_count == 0


# ---------- S4: 未知错误保守停止 ----------

def test_s4_unknown_error_stops(workdir, contract_file):
    wd = make_watchdog(workdir, contract_file())
    result = wd.run(train_cmd(workdir, "--epochs", "5", "--fail-at", "1",
                              "--fail-mode", "unknown_error"))
    assert result["status"] == "failed"
    assert result["crash"] == "unknown"
    assert wd.restarts == [], "未知错误应保守停止而非反复重启"


# ---------- S5: batch 下限保护 ----------

def test_s5_batch_floor_stops_retrying(workdir, contract_file):
    """无条件 OOM：减到下限后不再减，达 max_retries 后停止，不无限重启。"""
    wd = make_watchdog(workdir, contract_file(), max_retries=4, min_batch_size=8)
    result = wd.run(train_cmd(workdir, "--epochs", "5", "--batch_size", "16",
                              "--fail-at", "0", "--fail-mode", "oom"))

    assert result["status"] == "failed"
    assert wd.retry_count <= 4, "重启次数应受 max_retries 限制"
    # 至少有一次因为触到下限而放弃调整
    skipped = [r.skipped for r in wd.restarts if r.skipped]
    assert any("下限" in s for s in skipped), f"应记录触及下限: {skipped}"


# ---------- S8: 契约缺失降级 ----------

def test_s8_no_resumable_means_no_restart(workdir, contract_file):
    """契约无 resumable：可恢复中断也不重启，只告警。"""
    path = contract_file(resumable=None)
    wd = make_watchdog(workdir, path)
    assert wd.can_restart() is False

    result = wd.run(train_cmd(workdir, "--epochs", "5", "--fail-at", "1", "--fail-mode", "oom"))

    assert result["status"] == "failed"
    assert wd.restarts == [], "无可续训入口时不应盲目重启"


# ---------- S10: 命令行改写正确性 ----------

def test_s10_rewrite_replaces_and_passes_through(workdir, contract_file):
    """--batch_size 被替换（只出现一次），无关参数原样保留。"""
    wd = make_watchdog(workdir, contract_file())
    wd.run(train_cmd(workdir, "--epochs", "3", "--batch_size", "64",
                     "--num_workers", "3", "--epoch_seconds", "0.01",
                     "--fail-mode", "oom_if_batch_gt", "--fail-threshold", "16"))

    after = wd.restarts[0].cmd_after
    assert after.count("--batch_size") == 1, "flag 必须原地替换，不能追加"
    # 无关参数一字未改
    assert "--num_workers" in after
    assert after[after.index("--num_workers") + 1] == "3"
    assert "--epoch_seconds" in after


# ---------- 正常路径不触发任何机制 ----------

def test_normal_run_no_restarts(workdir, contract_file):
    wd = make_watchdog(workdir, contract_file())
    result = wd.run(train_cmd(workdir, "--epochs", "3"))
    assert result["status"] == "completed"
    assert wd.restarts == []
    assert wd.retry_count == 0
