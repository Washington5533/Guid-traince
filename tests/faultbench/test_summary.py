"""cp_5 · 摘要：重启轨迹、作废算力、契约缺失时的降级。

`wasted_epochs` 是 sidecar 形态下最关键的一个数字——它直接回答"这次干预
值不值"。曾经有过一个 bug：watchdog 拿不到训练进度，这个字段恒为 None
但摘要照常渲染成"作废约 0 epoch"，看起来像是免费的。此处回归。
"""

from __future__ import annotations

import json

from conftest import make_watchdog, train_cmd

from guardian.checkpoint_analyzer import CheckpointAnalyzer
from guardian.monitor import TrainingMonitor
from guardian.notifier import Notifier
from guardian.summary import SummaryGenerator
from guardian.task_contract import TaskContract


def _stack(workdir, contract_path, **wd_cfg):
    """按 run.py 的方式装好整套 v0 组件。"""
    contract = TaskContract({}, contract_path)
    notifier = Notifier({"channels": []})
    monitor = TrainingMonitor({"sliding_window": 20}, notifier, contract=contract)
    analyzer = CheckpointAnalyzer(
        {"stability_checks": 1, "save_top_k": 10, "keep_recent": 2},
        ckpt_dir=workdir / "checkpoints", contract=contract,
    )
    wd = make_watchdog(workdir, contract_path, **wd_cfg)
    wd.progress_fn = monitor.current_step
    gen = SummaryGenerator({"name": "test-run"}, monitor, analyzer, wd)
    return monitor, analyzer, wd, gen


def _run(workdir, contract_path, *extra, **wd_cfg):
    monitor, analyzer, wd, gen = _stack(workdir, contract_path, **wd_cfg)

    def on_tick(_wd, _proc):
        monitor.poll_metrics()
        analyzer.poll()

    result = wd.run(train_cmd(workdir, *extra), on_tick=on_tick)
    monitor.poll_metrics()
    analyzer.poll()
    return gen.generate(result), gen


# ---------- wasted_epochs 回归 ----------

def test_wasted_epochs_is_populated(workdir, contract_file):
    """OOM 在 epoch 2 触发、从 cp_1 恢复 -> 作废的 epoch 数必须是真实数字。"""
    summary, _ = _run(
        workdir, contract_file(),
        "--epochs", "6", "--batch_size", "64", "--epoch_seconds", "0.15",
        "--fail-at", "2", "--fail-mode", "oom_if_batch_gt", "--fail-threshold", "16",
    )
    assert summary["status"] == "completed"
    restarts = summary["restarts"]
    assert restarts, "应有重启记录"
    for rec in restarts:
        assert rec["wasted_epochs"] is not None, \
            "有指标通道时必须能算出作废的 epoch 数，不能是 None"
        assert rec["wasted_epochs"] >= 1, \
            f"从 {rec['resumed_from']} 回滚必然作废至少 1 个 epoch，实际 {rec['wasted_epochs']}"


def test_wasted_epochs_none_without_metrics_channel(workdir, contract_file):
    """无指标通道时拿不到进度，应记 None 而不是假装 0。"""
    path = contract_file(metrics_channel=None)
    monitor = TrainingMonitor({}, Notifier({"channels": []}), contract=TaskContract({}, path))
    assert monitor.enabled is False

    wd = make_watchdog(workdir, path)
    wd.progress_fn = monitor.current_step
    wd.run(train_cmd(workdir, "--epochs", "5", "--batch_size", "64", "--epoch_seconds", "0.1",
                     "--fail-at", "2", "--fail-mode", "oom_if_batch_gt", "--fail-threshold", "16"))
    assert wd.restarts
    assert all(r.wasted_epochs is None for r in wd.restarts), \
        "拿不到进度时不应猜一个数字"


# ---------- 摘要结构与渲染 ----------

def test_summary_records_restart_trajectory(workdir, contract_file):
    summary, gen = _run(
        workdir, contract_file(),
        "--epochs", "5", "--batch_size", "64", "--epoch_seconds", "0.1",
        "--fail-mode", "oom_if_batch_gt", "--fail-threshold", "16",
    )
    rec = summary["restarts"][0]
    assert rec["trigger"] == "crash"
    assert "--batch_size 64" in rec["cmd_before"]
    assert "--batch_size 32" in rec["cmd_after"]
    assert rec["applied"]["dataloader.batch_size"] == {"from": 64, "to": 32}

    text = gen.render(summary)
    assert "重启" in text and "batch_size 64->32" in text


def test_summary_on_unrecoverable_failure(workdir, contract_file):
    summary, gen = _run(
        workdir, contract_file(),
        "--epochs", "5", "--epoch_seconds", "0.05", "--fail-at", "2", "--fail-mode", "type_error",
    )
    assert summary["status"] == "failed"
    assert summary["failure"]["kind"] == "code"
    assert summary["restarts"] == [], "代码错误不该有重启记录"
    assert "失败原因" in gen.render(summary)


def test_summary_includes_anomalies(workdir, contract_file):
    summary, _ = _run(
        workdir, contract_file(),
        "--epochs", "10", "--epoch_seconds", "0.15",
        "--loss-spike-at", "7", "--loss-spike-ratio", "4.0",
    )
    types = [e["type"] for e in summary["anomaly_events"]]
    assert "loss_spike" in types, f"应记录检出的异常，实际 {types}"
    # v0：只告警，不重启
    assert summary["restarts"] == []
    assert all(e["response"]["action"] == "alert_only" for e in summary["anomaly_events"])


def test_summary_no_ai_narrative_without_advisor(workdir, contract_file):
    summary, _ = _run(workdir, contract_file(), "--epochs", "2", "--epoch_seconds", "0.05")
    assert "ai_narrative" not in summary, "v0 无 advisor，该字段应省略"


def test_summary_saves_json_and_txt(workdir, contract_file):
    summary, gen = _run(workdir, contract_file(), "--epochs", "2", "--epoch_seconds", "0.05")
    jpath, tpath = gen.save_summary(summary, workdir / "out")
    assert jpath.exists() and tpath.exists()
    reloaded = json.loads(jpath.read_text(encoding="utf-8"))
    assert reloaded["status"] == summary["status"]
    assert reloaded["checkpoints"]["total"] >= 1


def test_summary_survives_empty_everything():
    """monitor/analyzer/watchdog 全为 None 时也能生成摘要，不崩溃。"""
    gen = SummaryGenerator({"name": "empty"})
    summary = gen.generate({"status": "completed"})
    assert summary["restarts"] == []
    assert summary["anomaly_events"] == []
    assert isinstance(gen.render(summary), str)
