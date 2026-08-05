"""cp_2 · 监控与检测：增量读取、规则检测、通道缺失降级。"""

from __future__ import annotations

import math

from conftest import make_watchdog, train_cmd

from guardian.monitor import LogFileChannel, TrainingMonitor
from guardian.notifier import Notifier
from guardian.task_contract import TaskContract

PATTERN = r"epoch (\d+) loss ([\d.naN]+)"


def _monitor(contract_path, **cfg):
    return TrainingMonitor(cfg, Notifier({"channels": []}), TaskContract({}, contract_path))


# ---------- 指标通道 ----------

def test_channel_incremental_read(tmp_path):
    """连续两次 poll，只处理新增行，不重复处理旧行。"""
    log = tmp_path / "t.log"
    log.write_text("epoch 0 loss 1.0\nepoch 1 loss 0.5\n", encoding="utf-8")
    ch = LogFileChannel(log, PATTERN)

    first = ch.poll()
    assert len(first) == 2
    assert first[0]["step"] == 0 and first[0]["loss"] == 1.0

    assert ch.poll() == [], "无新增内容时应返回空"

    with log.open("a", encoding="utf-8") as fh:
        fh.write("epoch 2 loss 0.33\nepoch 3 loss 0.25\nepoch 4 loss 0.2\n")
    second = ch.poll()
    assert len(second) == 3, "只应处理新增的 3 行"
    assert [r["step"] for r in second] == [2, 3, 4]


def test_channel_missing_file_is_safe(tmp_path):
    ch = LogFileChannel(tmp_path / "nope.log", PATTERN)
    assert ch.poll() == []


def test_channel_extracts_extra_metrics(tmp_path):
    log = tmp_path / "t.log"
    log.write_text("epoch 3 loss 0.25 val_acc 0.9100 lr 0.0005\n", encoding="utf-8")
    rec = LogFileChannel(log, PATTERN).poll()[0]
    assert rec["val_acc"] == 0.91
    assert rec["lr"] == 0.0005


# ---------- 规则检测 ----------

def test_detects_loss_spike(workdir, contract_file):
    mon = _monitor(contract_file(), sliding_window=10, loss_spike_ratio=0.5)
    log = workdir / "logs" / "train.log"
    log.write_text("".join(f"epoch {i} loss 0.5\n" for i in range(5)), encoding="utf-8")
    assert mon.poll_metrics() == [], "平稳 loss 不应告警"

    with log.open("a", encoding="utf-8") as fh:
        fh.write("epoch 5 loss 2.0\n")
    events = mon.poll_metrics()
    assert [e.type for e in events] == ["loss_spike"]
    assert "突增" in events[0].detail


def test_detects_nan(workdir, contract_file):
    mon = _monitor(contract_file())
    log = workdir / "logs" / "train.log"
    log.write_text("epoch 0 loss 1.0\nepoch 1 loss nan\n", encoding="utf-8")

    events = mon.poll_metrics()
    assert [e.type for e in events] == ["nan_inf"]
    assert events[0].level == "error", "NaN 必须是 error 级"


def test_nan_does_not_poison_window(workdir, contract_file):
    """NaN 不进滑动窗口，否则后续均值全被污染。"""
    mon = _monitor(contract_file(), sliding_window=10)
    log = workdir / "logs" / "train.log"
    log.write_text("epoch 0 loss 1.0\nepoch 1 loss nan\nepoch 2 loss 0.9\n", encoding="utf-8")
    mon.poll_metrics()
    assert all(not math.isnan(v) for v in mon.window)


def test_monotonic_loss_no_alerts(workdir, contract_file):
    mon = _monitor(contract_file(), sliding_window=20)
    log = workdir / "logs" / "train.log"
    log.write_text("".join(f"epoch {i} loss {1.0 / (i + 1):.4f}\n" for i in range(15)),
                   encoding="utf-8")
    assert mon.poll_metrics() == [], "单调递减 loss 不应有任何告警"


# ---------- 契约缺失降级 ----------

def test_no_metrics_channel_degrades(workdir, contract_file):
    """契约无 metrics_channel：退化为进程级看护，不报错。"""
    mon = _monitor(contract_file(metrics_channel=None))
    assert mon.enabled is False
    assert mon.poll_metrics() == []


def test_log_file_without_pattern_degrades(workdir, contract_file):
    """type=log_file 但没有 log_pattern：无法解析，降级。"""
    mon = _monitor(contract_file(
        metrics_channel={"type": "log_file", "path": "./logs/train.log"}
    ))
    assert mon.enabled is False


def test_wandb_channel_is_supported(workdir, contract_file):
    """wandb 通道已支持——读本地 run 目录。"""
    import shutil
    wdir = workdir / "wandb" / "run-test"
    (wdir / "files").mkdir(parents=True)
    (wdir / "files" / "wandb-events.jsonl").write_text(
        '{"_timestamp": 1, "loss": 0.5, "_step": 0}\n', encoding="utf-8")
    (wdir / "files" / "wandb-summary.json").write_text(
        '{"loss": 0.5, "accuracy": 0.9, "_runtime": 10}', encoding="utf-8")
    mon = _monitor(contract_file(metrics_channel={"type": "wandb", "path": str(wdir)}))
    assert mon.enabled is True
    results = mon.poll_metrics()
    assert len(results) >= 0  # 至少不报错
    assert len(mon.history) >= 1  # 从 events 读到了数据


# ---------- 端到端：监控与守护同时工作 ----------

def test_monitor_observes_live_training(workdir, contract_file):
    """guardian 看护训练的同时，从进程外读到指标并检出注入的 loss 突增。"""
    path = contract_file()
    mon = _monitor(path, sliding_window=10, loss_spike_ratio=0.5)
    wd = make_watchdog(workdir, path)

    result = wd.run(
        train_cmd(workdir, "--epochs", "8", "--epoch_seconds", "0.2",
                  "--loss-spike-at", "6", "--loss-spike-ratio", "3.0"),
        on_tick=lambda _wd, _proc: mon.poll_metrics(),
    )
    mon.poll_metrics()  # 收尾读取

    assert result["status"] == "completed"
    assert wd.restarts == [], "v0 下 loss 突增只告警，不应重启训练"
    assert mon.history, "应从进程外读到指标"
    assert any(e.type == "loss_spike" for e in mon.anomalies), \
        f"应检出注入的突增，实际: {[e.type for e in mon.anomalies]}"
