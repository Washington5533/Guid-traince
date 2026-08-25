"""Sub-agent 决策上下文记忆。

滚动窗口记忆：保留最近 N 条决策/事件，训练阶段摘要，
供 LLM prompt 使用。轻量、无外部依赖。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["RollingMemory", "DecisionRecord", "TrainingPhase"]


class TrainingPhase:
    """训练阶段枚举。"""
    INIT = "init"
    EARLY = "early"          # 前 10%
    MID = "mid"              # 10% ~ 80%
    LATE = "late"            # 80% ~ 95%
    CONVERGED = "converged"  # 最后 5%
    FINISHED = "finished"


@dataclass
class DecisionRecord:
    """单条决策记录。"""
    timestamp: float
    event_type: str          # anomaly / crash / hang / tick / intervention
    description: str
    action_taken: str = ""
    action_params: dict = field(default_factory=dict)
    source: str = "sub_agent"  # sub_agent / rule_default / pc_override
    outcome: str = ""          # success / failed / pending / rejected
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


class RollingMemory:
    """滚动窗口决策记忆。

    保留最近 max_size 条记录，超出自动淘汰最旧的。
    同时维护训练阶段摘要和关键统计。
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._records: list[DecisionRecord] = []
        self._phase = TrainingPhase.INIT
        self._start_time = time.time()
        self._total_epochs: int = 0
        self._current_epoch: int = 0
        self._anomaly_count: int = 0
        self._intervention_count: int = 0
        self._crash_count: int = 0
        self._best_metric_value: float | None = None
        self._best_metric_name: str = ""
        self._consecutive_failures: int = 0

    # ── 记录 ─────────────────────────────────────────────────────────

    def record(self, record: DecisionRecord) -> None:
        """追加一条决策记录。"""
        self._records.append(record)
        if len(self._records) > self.max_size:
            self._records = self._records[-self.max_size:]
        # 更新统计
        if record.event_type == "anomaly":
            self._anomaly_count += 1
        elif record.event_type == "crash":
            self._crash_count += 1
        elif record.event_type == "intervention":
            self._intervention_count += 1
            if record.outcome == "failed":
                self._consecutive_failures += 1
            else:
                self._consecutive_failures = 0

    def record_decision(self, event_type: str, description: str, action: str = "",
                        params: dict | None = None, source: str = "sub_agent",
                        confidence: float = 1.0, **kwargs) -> DecisionRecord:
        """便捷方法：创建并记录一条决策。"""
        rec = DecisionRecord(
            timestamp=time.time(),
            event_type=event_type,
            description=description,
            action_taken=action,
            action_params=params or {},
            source=source,
            confidence=confidence,
            metadata=kwargs,
        )
        self.record(rec)
        return rec

    # ── 训练进度 ─────────────────────────────────────────────────────

    def update_progress(self, current_epoch: int, total_epochs: int,
                        current_metric_value: float | None = None,
                        metric_name: str = "") -> None:
        """更新训练进度，自动推断阶段。"""
        self._current_epoch = current_epoch
        self._total_epochs = total_epochs
        if total_epochs > 0:
            progress = current_epoch / total_epochs
            if progress < 0.1:
                self._phase = TrainingPhase.EARLY
            elif progress < 0.8:
                self._phase = TrainingPhase.MID
            elif progress < 0.95:
                self._phase = TrainingPhase.LATE
            else:
                self._phase = TrainingPhase.CONVERGED
        if current_metric_value is not None and metric_name:
            if self._best_metric_value is None or (
                metric_name.startswith("loss") and current_metric_value < self._best_metric_value
            ) or (
                not metric_name.startswith("loss") and current_metric_value > self._best_metric_value
            ):
                self._best_metric_value = current_metric_value
                self._best_metric_name = metric_name

    def mark_finished(self) -> None:
        self._phase = TrainingPhase.FINISHED

    # ── 查询 ─────────────────────────────────────────────────────────

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def progress(self) -> float:
        if self._total_epochs <= 0:
            return 0.0
        return min(self._current_epoch / self._total_epochs, 1.0)

    def get_recent(self, n: int = 10) -> list[DecisionRecord]:
        return self._records[-n:]

    def get_summary(self) -> dict[str, Any]:
        """生成记忆摘要，供 LLM prompt 使用。"""
        recent = self.get_recent(10)
        return {
            "phase": self._phase,
            "progress": round(self.progress, 3),
            "current_epoch": self._current_epoch,
            "total_epochs": self._total_epochs,
            "anomaly_count": self._anomaly_count,
            "crash_count": self._crash_count,
            "intervention_count": self._intervention_count,
            "consecutive_failures": self._consecutive_failures,
            "best_metric": {
                "name": self._best_metric_name,
                "value": self._best_metric_value,
            } if self._best_metric_name else None,
            "recent_decisions": [
                {
                    "time": r.timestamp,
                    "type": r.event_type,
                    "desc": r.description[:100],
                    "action": r.action_taken,
                    "source": r.source,
                    "outcome": r.outcome,
                    "confidence": r.confidence,
                }
                for r in recent
            ],
        }

    def get_context_for_llm(self) -> str:
        """生成 LLM prompt 用的文本上下文。"""
        summary = self.get_summary()
        lines = [
            f"训练阶段: {summary['phase']} ({summary['progress']:.0%})",
            f"Epoch: {summary['current_epoch']}/{summary['total_epochs']}",
            f"异常次数: {summary['anomaly_count']} | 崩溃: {summary['crash_count']} | 干预: {summary['intervention_count']}",
        ]
        if summary["best_metric"]:
            lines.append(f"最佳指标: {summary['best_metric']['name']}={summary['best_metric']['value']}")
        if summary["consecutive_failures"] > 0:
            lines.append(f"连续失败: {summary['consecutive_failures']} 次")
        if summary["recent_decisions"]:
            lines.append("近期决策:")
            for d in summary["recent_decisions"][-5:]:
                lines.append(f"  [{d['type']}] {d['desc']} → {d['action']} ({d['source']}, {d['outcome']})")
        return "\n".join(lines)

    # ── 序列化（持久化 / 断点续守） ──────────────────────────────────

    def to_dict(self) -> dict:
        """导出为可 JSON 序列化的字典。"""
        return {
            "max_size": self.max_size,
            "phase": self._phase,
            "records": [asdict(r) for r in self._records],
            "anomaly_count": self._anomaly_count,
            "intervention_count": self._intervention_count,
            "crash_count": self._crash_count,
            "consecutive_failures": self._consecutive_failures,
            "best_metric_value": self._best_metric_value,
            "best_metric_name": self._best_metric_name,
            "current_epoch": self._current_epoch,
            "total_epochs": self._total_epochs,
            "start_time": self._start_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RollingMemory":
        """从字典恢复（断点续守）。"""
        mem = cls(max_size=data.get("max_size", 100))
        mem._phase = data.get("phase", TrainingPhase.INIT)
        mem._anomaly_count = data.get("anomaly_count", 0)
        mem._intervention_count = data.get("intervention_count", 0)
        mem._crash_count = data.get("crash_count", 0)
        mem._consecutive_failures = data.get("consecutive_failures", 0)
        mem._best_metric_value = data.get("best_metric_value")
        mem._best_metric_name = data.get("best_metric_name", "")
        mem._current_epoch = data.get("current_epoch", 0)
        mem._total_epochs = data.get("total_epochs", 0)
        mem._start_time = data.get("start_time", time.time())
        for r in data.get("records", []):
            try:
                mem._records.append(DecisionRecord(**r))
            except Exception:
                pass  # 跳过损坏的记录
        return mem

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:
        return f"RollingMemory(phase={self._phase}, records={len(self._records)}, anomalies={self._anomaly_count})"
