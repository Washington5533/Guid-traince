"""cp_2 · 训练监控：进程外读指标 + 规则异常检测 + GPU 硬件轮询。

sidecar 形态下 monitor 跑在训练进程之外，指标从训练脚本已有的输出通道
读取（tail 日志 / wandb / tensorboard）。因此检测粒度取决于脚本的输出频率，
且决策不占用训练时间。GPU 硬件指标独立轮询 nvidia-smi，与训练进程完全解耦。

v0 只到 alert_only：检测 + 告警，不做任何重启式干预（那属于 v1 的 agent 决策）。
详见 checkpoint/cp_2.md
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
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


# ---------------------------------------------------------------------------
# 指标通道：增量 tail 日志文件
# ---------------------------------------------------------------------------

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
            for key, pat in (("val_acc", r"val_acc[= ]([\d.]+)"),
                             ("lr", r"lr[= ]([\d.eE+-]+)")):
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


# ---------------------------------------------------------------------------
# GPU 硬件轮询（独立于训练进程，经 nvidia-smi）
# ---------------------------------------------------------------------------

_NVIDIA_SMI = shutil.which("nvidia-smi")

# nvidia-smi 查询模板：利用率、温度、显存、功耗
_GPU_QUERY = (
    "index,temperature.gpu,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,power.draw,power.limit"
)


@dataclass
class GpuSnapshot:
    """一次 nvidia-smi 采样结果（单卡）。"""

    index: int
    temperature_c: float | None = None
    util_pct: float | None = None
    mem_util_pct: float | None = None
    mem_used_mb: float | None = None
    mem_total_mb: float | None = None
    power_w: float | None = None
    power_limit_w: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"index": self.index}
        if self.error:
            d["error"] = self.error
            return d
        for key in ("temperature_c", "util_pct", "mem_util_pct",
                     "mem_used_mb", "mem_total_mb", "power_w", "power_limit_w"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        return d


def poll_gpu() -> list[GpuSnapshot]:
    """运行一次 nvidia-smi 并解析所有 GPU 的当前状态。

    没有 nvidia-smi（CPU-only 机器或未安装驱动）时返回空列表，不报错。
    """
    if _NVIDIA_SMI is None:
        return []
    try:
        proc = subprocess.run(
            [_NVIDIA_SMI, f"--query-gpu={_GPU_QUERY}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    if proc.returncode != 0:
        return []

    snapshots: list[GpuSnapshot] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            snapshots.append(GpuSnapshot(
                index=int(parts[0]) if parts[0] else 0,
                temperature_c=_try_float(parts[1]),
                util_pct=_try_float(parts[2]),
                mem_util_pct=_try_float(parts[3]),
                mem_used_mb=_try_float(parts[4]),
                mem_total_mb=_try_float(parts[5]),
                power_w=_try_float(parts[6]),
                power_limit_w=_try_float(parts[7]),
            ))
        except (ValueError, IndexError):
            continue
    return snapshots


def _try_float(raw: str) -> float | None:
    """[N/A] / [Not Supported] 等不可用标记 → None。"""
    v = raw.strip()
    if v == "" or v.startswith("["):
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 训练监控主类
# ---------------------------------------------------------------------------

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
        on_intervention: Any = None,
    ):
        self.cfg = config or {}
        self.notifier = notifier
        self.contract = contract
        self.advisor = advisor      # v1；v0 恒为 None
        self.on_intervention = on_intervention  # v1: callable(action, param, reason) -> 通知 cp_3

        # --- 指标异常检测参数 ---
        self.window = deque(maxlen=int(self.cfg.get("sliding_window", 50)))
        self.spike_ratio = float(self.cfg.get("loss_spike_ratio", 0.5))
        self.stagnation_steps = int(self.cfg.get("loss_stagnation_steps", 500))
        self.stagnation_threshold = float(self.cfg.get("loss_stagnation_threshold", 0.001))

        # --- GPU 硬件检测参数 ---
        self.hw_interval = float(self.cfg.get("hardware_poll_interval", 30))
        self.gpu_idle_threshold = float(self.cfg.get("gpu_idle_threshold", 20))
        self.gpu_temp_threshold = float(self.cfg.get("gpu_temp_threshold", 85))
        self._last_hw_poll = 0.0
        self._gpu_idle_streak: dict[int, int] = {}   # gpu_index -> 连续低利用率次数

        self.channel = build_channel(contract.metrics_channel()) if contract else None
        self.history: list[dict[str, Any]] = []
        self.anomalies: list[AnomalyEvent] = []
        self.gpu_history: list[dict[str, Any]] = []
        self._last_record: dict[str, Any] | None = None
        self._cooldown_seen: set[str] = set()

    @property
    def enabled(self) -> bool:
        """没有可用指标通道时，退化为进程级看护（存活 + GPU）。"""
        return self.channel is not None

    # ------------------------------------------------------------------
    # sidecar 主路径
    # ------------------------------------------------------------------

    def poll_metrics(self) -> list[AnomalyEvent]:
        """sidecar 主路径：读取新增指标 + 按间隔轮询 GPU，逐条送入检测。"""
        events: list[AnomalyEvent] = []

        # 指标通道
        if self.channel is not None:
            for rec in self.channel.poll():
                self.history.append(rec)
                self._last_record = rec
                events.extend(self._check(rec))

        # GPU 硬件（独立于指标通道，按间隔轮询）
        now = time.monotonic()
        if now - self._last_hw_poll >= self.hw_interval:
            self._last_hw_poll = now
            events.extend(self._poll_hardware())

        return events

    # ------------------------------------------------------------------
    # 规则检测（纯规则，判定"是不是异常"）
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # GPU 硬件检测
    # ------------------------------------------------------------------

    def _poll_hardware(self) -> list[AnomalyEvent]:
        """轮询 nvidia-smi，执行 GPU 空转 / 温度检测。"""
        events: list[AnomalyEvent] = []
        for snap in poll_gpu():
            self.gpu_history.append(snap.to_dict())
            events.extend(self._check_gpu(snap))
        return events

    def _check_gpu(self, snap: GpuSnapshot) -> list[AnomalyEvent]:
        """对单张 GPU 的一次采样执行全部硬件检测规则。"""
        events: list[AnomalyEvent] = []

        # GPU 空转：利用率持续低于阈值（需连续 5 次采样）
        util = snap.util_pct
        if util is not None:
            if util < self.gpu_idle_threshold:
                streak = self._gpu_idle_streak.get(snap.index, 0) + 1
                self._gpu_idle_streak[snap.index] = streak
                if streak >= 5:
                    events.append(AnomalyEvent(
                        "gpu_idle",
                        f"GPU {snap.index} 利用率持续偏低（{util:.0f}%，连续 {streak} 次），"
                        "可能存在数据加载瓶颈",
                        step=None, metrics={"gpu_index": snap.index, "utilization": util},
                    ))
            else:
                self._gpu_idle_streak[snap.index] = 0

        # GPU 温度（无自动降频，硬件安全不交给 agent）
        temp = snap.temperature_c
        if temp is not None and temp > self.gpu_temp_threshold:
            events.append(AnomalyEvent(
                "gpu_temp",
                f"GPU {snap.index} 温度过高（{temp:.0f}°C > {self.gpu_temp_threshold}°C）",
                level="warning", metrics={"gpu_index": snap.index, "temperature": temp},
            ))

        return self._emit(events)

    # ------------------------------------------------------------------
    # 告警发出
    # ------------------------------------------------------------------

    def _emit(self, events: list[AnomalyEvent]) -> list[AnomalyEvent]:
        """v1：有 advisor 时由 agent 选择应对动作，否则走规则默认（alert_only）。"""
        for ev in events:
            action_space = self._action_space_for(ev.type)
            default = "alert_only"
            response = self._decide_response(ev, action_space, default)
            ev.response = response
            self.anomalies.append(ev)
            if self.notifier is not None:
                self.notifier.send(
                    f"检测到 {ev.type}", ev.detail,
                    alert_type=ev.type, level=ev.level, response=ev.response,
                )
            # 重启式动作通知 cp_3
            if response.get("restart") and self.on_intervention is not None:
                self.on_intervention(
                    response["action"], response.get("param"),
                    f"{ev.type}: {ev.detail}",
                )
        return events

    # -- 动作空间定义（cp_2.md 有限动作集） --

    @staticmethod
    def _action_space_for(anomaly_type: str) -> list:
        if anomaly_type == "loss_spike":
            return [
                "ignore", "alert_only",
                {"action": "restart_with_lower_lr",
                 "ratio": {"min": 0.1, "max": 0.9}},
            ]
        if anomaly_type == "loss_stagnation":
            return [
                "ignore", "alert_only",
                {"action": "suggest_lr_increase",
                 "ratio": {"min": 1.1, "max": 3.0}},
            ]
        if anomaly_type == "nan_inf":
            return [
                "rollback_to_last_ckpt", "alert_only",
                {"action": "restart_with_lower_lr",
                 "ratio": {"min": 0.1, "max": 0.9}},
            ]
        if anomaly_type == "gpu_idle":
            return ["ignore", "alert_only"]
        if anomaly_type == "gpu_temp":
            return ["alert_only"]       # 硬件安全不交给 agent
        return ["alert_only"]

    def _decide_response(self, ev: AnomalyEvent, action_space: list, default: str) -> dict:
        """异常确认后决定'怎么应对'：advisor 可用时问 agent，否则/超时走默认。"""
        if self.advisor is None:
            return {"source": "rule_default", "action": "alert_only", "restart": False}

        context = {
            "anomaly_type": ev.type,
            "detail": ev.detail,
            "step": ev.step,
            "metrics": ev.metrics,
            "history_count": len(self.anomalies),
        }
        result = self.advisor.decide("monitor_response", context, action_space, default)
        action = result.get("action", default)
        restart = action not in ("ignore", "alert_only", "suggest_lr_increase")
        # suggest_lr_increase 是纯建议，不自动执行
        param = None
        if isinstance(action, dict):
            param = {k: v for k, v in action.items() if k != "action"}
            action = action.get("action", default)
        return {
            "source": result.get("source", "rule_default"),
            "action": action,
            "restart": restart,
            "param": param,
        }

    # ------------------------------------------------------------------
    # 查询接口（供 cp_3 / cp_5 / cp_10）
    # ------------------------------------------------------------------

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

    def get_gpu_history(self) -> list[dict[str, Any]]:
        """GPU 硬件采样历史（供 summary 统计资源用量）。"""
        return list(self.gpu_history)
