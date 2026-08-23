"""Sub-agent 模块：算力服务器端自主智能体。

不依赖 Claude Code 或任何外部 agent 平台。Sub-agent 在 guardian 进程内
以 LLM 驱动的决策循环运行，拥有记忆、工具调用能力，在训练全程自主决策。

生命周期：
    spawn() → on_tick() × N → approve()/reject() × M → shutdown()

权限分级：
    supervised  → 只自动执行告警，高风险动作需 PC 端审批
    auto        → 自动执行告警 + 参数调整，停止仍需确认
    full        → 全部自动执行
"""

from guardian.sub_agent.core import SubAgent
from guardian.sub_agent.memory import RollingMemory
from guardian.sub_agent.prompts import (
    SYSTEM_ANOMALY_RESPONSE,
    SYSTEM_CRASH_RECOVERY,
    SYSTEM_TRAINING_SUMMARY,
    build_anomaly_prompt,
    build_crash_prompt,
    build_summary_prompt,
)
from guardian.sub_agent.tool_registry import (
    Action,
    ActionResult,
    ToolSpec,
    ToolRegistry,
    default_registry,
)

__all__ = [
    "SubAgent",
    "ToolRegistry",
    "ToolSpec",
    "Action",
    "ActionResult",
    "RollingMemory",
    "default_registry",
]
