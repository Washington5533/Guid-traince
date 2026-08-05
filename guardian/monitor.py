"""cp_2 · 训练监控：进程外读指标 + 规则异常检测。

sidecar 形态下 monitor 跑在训练进程之外，指标从训练脚本已有的输出通道
读取（tail 日志 / wandb / tensorboard）。因此检测粒度取决于脚本的输出频率，
且决策不占用训练时间。详见 checkpoint/cp_2.md

v0 只到 alert_only：检测 + 告警，不做任何重启式干预（那属于 v1 的 agent 决策）。
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AnomalyEvent:
    """一次异常检测结果。"""

    type: str
    detail: str
    level: str = "warning"
    step: int | None = None
    epoch: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    response: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type, "detail": self.detail, "severity": self.level,
            "step": self.step, "epoch": self.epoch,
            "response": self.response or {"source": "rule_default", "action": "alert_only",
                                          "restart": False},
        }


class LogFileChannel:
    """按正则增量 tail 一个日志文件。

    只处理新增内容（记住上次读到的字节偏移），不重复处理旧行。
    """

    def __init__(self, path: str | Path, pattern: str):
        self.path = Path(path)
        self.regex = re.compile(pattern)
        self._offset = 0

    def poll(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        size = self.path.stat().st_size
        if size < self._offset:      # 文件被截断/轮转，从头再来
            self._offset = 0
        if size == self._offset:
            return []
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self._offset)
            chunk = fh.read()
            self._offset = fh.tell()

        out: list[dict[str, Any]] = []
        for line in chunk.splitlines():
            m = self.regex.search(line)
            if not m:
                continue
            rec: dict[str, Any] = {"raw": line}
            groups = m.groups()
            if len(groups) >= 1:
                rec["step"] = _to_int(groups[0])
            if len(groups) >= 2:
                rec["loss"] = _to_float(groups[1])
            # 尽量多抓一些常见指标，供摘要使用
            for key, pat in (("val_acc", r"val_acc[= ]([\d.]+)"), ("lr", r"lr[= ]([\d.eE+-]+)")):
                extra = re.search(pat, line)
                if extra:
                    rec[key] = _to_float(extra.group(1))
            out.append(rec)
        return out


def _to_int(raw: Any) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _to_float(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def build_channel(metrics_channel: dict) -> LogFileChannel | None:
    """按契约声明构造指标通道。不支持的类型返回 None（降级为进程级看护）。"""
    ch_type = (metrics_channel or {}).get("type")
    path = (metrics_channel or {}).get("path")
    if not ch_type or not path:
        return None
    if ch_type in ("log_file", "metrics_json"):
        pattern = metrics_channel.get("log_pattern")
        if not pattern:
            return None
        return LogFileChannel(path, pattern)
    # wandb / tensorboard 属于 v0 之后：需要各自的读取实现
    return None


class TrainingMonitor:
    """进程外指标监控与规则异常检测。

    v0 的应对动作恒为 alert_only——检测到就告警，不打断训练。
    重启式干预属于 v1（由 agent 在有限动作集里选择后交给 cp_3 执行）。
    """

    def __init__(
        self,
        config: dict | None = None,
        notifier: Any = None,
        contract: Any = None,
        advisor: Any = None,
    ):
        self.cfg = config or {}
        self.notifier = notifier
        self.contract = contract
        self.advisor = advisor      # v1；v0 恒为 None

        self.window = deque(maxlen=int(self.cfg.get("sliding_window", 50)))
        self.spike_ratio = float(self.cfg.get("loss_spike_ratio", 0.5))
        self.stagnation_steps = int(self.cfg.get("loss_stagnation_steps", 500))
        self.stagnation_threshold = float(self.cfg.get("loss_stagnation_threshold", 0.001))

        self.channel = build_channel(contract.metrics_channel()) if contract else None
        self.history: list[dict[str, Any]] = []
        self.anomalies: list[AnomalyEvent] = []
        self._last_record: dict[str, Any] | None = None
        self._cooldown_seen: set[str] = set()

    @property
    def enabled(self) -> bool:
        """没有可用指标通道时，退化为进程级看护（存活 + GPU）。"""
        return self.channel is not None

    def poll_metrics(self) -> list[AnomalyEvent]:
        """sidecar 主路径：读取新增指标，逐条送入检测。"""
        if self.channel is None:
            return []
        found: list[AnomalyEvent] = []
        for rec in self.channel.poll():
            self.history.append(rec)
            self._last_record = rec
            found.extend(self._check(rec))
        return found

    def _check(self, rec: dict[str, Any]) -> list[AnomalyEvent]:
        """全部规则检测（纯规则，判定"是不是异常"，不受 agent 影响）。"""
        events: list[AnomalyEvent] = []
        loss = rec.get("loss")
        step = rec.get("step")

        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            events.append(AnomalyEvent(
                "nan_inf", f"loss 为 {loss}（NaN/Inf）", level="error", step=step, metrics=rec,
            ))
            # NaN 不进滑动窗口，否则后续均值全被污染
            return self._emit(events)

        if loss is not None:
            spike = self._check_loss_spike(loss)
            if spike:
                events.append(AnomalyEvent("loss_spike", spike, step=step, metrics=rec))
            self.window.append(loss)

            stag = self._check_stagnation()
            if stag:
                events.append(AnomalyEvent("loss_stagnation", stag, step=step, metrics=rec))

        return self._emit(events)

    def _check_loss_spike(self, loss: float) -> str | None:
        if len(self.window) < 3:
            return None
        mean = sum(self.window) / len(self.window)
        if mean <= 0:
            return None
        if loss > mean * (1 + self.spike_ratio):
            pct = (loss / mean - 1) * 100
            return f"Loss 突增 +{pct:.0f}%，当前 {loss:.4f}，窗口均值 {mean:.4f}"
        return None

    def _check_stagnation(self) -> str | None:
        if len(self.window) < min(self.stagnation_steps, self.window.maxlen or 0):
            return None
        recent = list(self.window)[-self.stagnation_steps:]
        if len(recent) < 2:
            return None
        delta = recent[0] - recent[-1]
        if delta < self.stagnation_threshold:
            return f"Loss 停滞 {len(recent)} 步，下降仅 {delta:.6f}"
        return None

    def _emit(self, events: list[AnomalyEvent]) -> list[AnomalyEvent]:
        """v0：应对动作恒为 alert_only，只告警不打断训练。"""
        for ev in events:
            ev.response = {"source": "rule_default", "action": "alert_only", "restart": False}
            self.anomalies.append(ev)
            if self.notifier is not None:
                self.notifier.send(
                    f"检测到 {ev.type}", ev.detail,
                    alert_type=ev.type, level=ev.level, response=ev.response,
                )
        return events

    def last_progress(self) -> dict[str, Any] | None:
        """最近一条指标记录，供挂起检测判断"是否还在前进"。"""
        return self._last_record

    def current_step(self) -> int | None:
        """训练已跑到的 step/epoch。供 cp_3 计算重启作废了多少算力。

        取历史最大值而非最后一条：重启后新进程会从较小的 epoch 重新往上走，
        用最大值才代表"曾经到达过的进度"。
        """
        steps = [r["step"] for r in self.history if r.get("step") is not None]
        return max(steps) if steps else None

    def get_metrics_history(self) -> list[dict[str, Any]]:
        return list(self.history)

    def get_anomaly_history(self) -> list[dict[str, Any]]:
        return [ev.to_dict() for ev in self.anomalies]
