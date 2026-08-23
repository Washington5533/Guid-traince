"""Sub-agent 工具注册表 + 权限策略。

每个工具定义：可调用函数、风险等级、是否需要审批、参数 schema。
权限策略根据 autonomy level 控制哪些工具需要 PC 端审批。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "Action",
    "ActionResult",
    "ToolSpec",
    "ToolRegistry",
    "default_registry",
]


class RiskLevel(Enum):
    """动作风险等级。"""
    NONE = "none"          # 只读查询，零风险
    LOW = "low"            # 低风险（如更新 dashboard 配置）
    MEDIUM = "medium"       # 中等风险（如启用梯度累积）
    HIGH = "high"           # 高风险（如重启训练、停止训练）


@dataclass
class Action:
    """Sub-agent 发出的一个动作。"""
    tool_name: str
    params: dict = field(default_factory=dict)
    reason: str = ""
    confidence: float = 1.0      # 0.0 ~ 1.0，LLM 对决策的信心
    context: dict = field(default_factory=dict)


@dataclass
class ActionResult:
    """动作执行结果。"""
    action_id: str
    tool_name: str
    success: bool
    result: Any = None
    error: str = ""
    executed_at: float = field(default_factory=time.time)
    # 如果动作被 PC 端驳回
    rejected: bool = False
    rejection_reason: str = ""


@dataclass
class ToolSpec:
    """工具规格。"""
    name: str
    fn: Callable[..., Any]
    description: str
    risk: RiskLevel
    requires_confirmation: bool = True
    param_schema: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def call(self, params: dict, context: dict | None = None) -> ActionResult:
        """调用工具，返回结果。"""
        action_id = f"act_{uuid.uuid4().hex[:12]}"
        try:
            result = self.fn(params, context or {})
            return ActionResult(action_id=action_id, tool_name=self.name, success=True, result=result)
        except Exception as exc:
            return ActionResult(action_id=action_id, tool_name=self.name, success=False, error=str(exc))


class ToolRegistry:
    """工具注册表：管理 sub-agent 可调用的所有动作。"""

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}
        self._history: list[ActionResult] = []
        self._register_defaults()

    def _register_defaults(self) -> None:
        """注册默认工具集。"""
        # 只读查询工具
        self.register(ToolSpec(
            name="query_status",
            fn=self._noop_fn,
            description="查询当前训练状态（epoch/step/loss/GPU）",
            risk=RiskLevel.NONE,
            requires_confirmation=False,
            tags=["read", "status"],
        ))
        self.register(ToolSpec(
            name="query_gpu",
            fn=self._noop_fn,
            description="查询 GPU 设备状态（利用率/温度/显存）",
            risk=RiskLevel.NONE,
            requires_confirmation=False,
            tags=["read", "gpu"],
        ))

        # 告警工具
        self.register(ToolSpec(
            name="alert",
            fn=self._noop_fn,
            description="发送告警（终端 + webhook + PC 弹窗）",
            risk=RiskLevel.NONE,
            requires_confirmation=False,
            tags=["notify"],
            param_schema={"level": {"type": "string", "enum": ["info", "warning", "error"]}},
        ))

        # 训练控制工具
        self.register(ToolSpec(
            name="restart_with_lower_lr",
            fn=self._noop_fn,
            description="降低学习率并重启训练（从最近 checkpoint 续训）",
            risk=RiskLevel.HIGH,
            requires_confirmation=True,
            tags=["control", "restart", "lr"],
            param_schema={"ratio": {"type": "number", "min": 0.01, "max": 1.0}},
        ))
        self.register(ToolSpec(
            name="reduce_batch",
            fn=self._noop_fn,
            description="减小 batch_size 并重启训练",
            risk=RiskLevel.HIGH,
            requires_confirmation=True,
            tags=["control", "restart", "batch"],
            param_schema={"ratio": {"type": "number", "min": 0.1, "max": 1.0}},
        ))
        self.register(ToolSpec(
            name="enable_grad_accum",
            fn=self._noop_fn,
            description="启用梯度累积（不中断训练，等效增大 batch）",
            risk=RiskLevel.MEDIUM,
            requires_confirmation=True,
            tags=["control", "grad_accum"],
            param_schema={"steps": {"type": "integer", "min": 1, "max": 64}},
        ))
        self.register(ToolSpec(
            name="stop_training",
            fn=self._noop_fn,
            description="停止训练（不可逆）",
            risk=RiskLevel.HIGH,
            requires_confirmation=True,
            tags=["control", "stop"],
        ))

    def register(self, spec: ToolSpec) -> None:
        """注册一个新工具。"""
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """获取工具规格。"""
        return self._tools.get(name)

    def list_tools(self, tags: list[str] | None = None) -> list[ToolSpec]:
        """列出所有工具，可按标签过滤。"""
        tools = list(self._tools.values())
        if tags:
            tools = [t for t in tools if any(tag in t.tags for tag in tags)]
        return tools

    def get_tools_description(self) -> str:
        """生成工具描述文本，供 LLM prompt 使用。"""
        lines = []
        for tool in self._tools.values():
            schema = ", ".join(f"{k}: {v}" for k, v in tool.param_schema.items()) if tool.param_schema else "无参数"
            lines.append(f"- {tool.name} (risk={tool.risk.value}, confirm={tool.requires_confirmation}): {tool.description} [参数: {schema}]")
        return "\n".join(lines)

    def requires_approval(self, tool_name: str, autonomy: str) -> bool:
        """根据 autonomy level 判断一个工具是否需要 PC 审批。"""
        spec = self._tools.get(tool_name)
        if spec is None:
            return True  # 未知工具默认需要审批
        if not spec.requires_confirmation:
            return False

        policy = AUTONOMY_POLICY.get(autonomy, AUTONOMY_POLICY["supervised"])
        return policy.get(tool_name, "confirm") == "confirm"

    def execute(self, action: Action, autonomy: str) -> ActionResult:
        """执行一个动作（仅当不需要审批时直接执行）。"""
        spec = self._tools.get(action.tool_name)
        if spec is None:
            return ActionResult(
                action_id=f"act_{uuid.uuid4().hex[:12]}",
                tool_name=action.tool_name,
                success=False,
                error=f"未知工具: {action.tool_name}",
            )
        if self.requires_approval(action.tool_name, autonomy):
            return ActionResult(
                action_id=f"act_{uuid.uuid4().hex[:12]}",
                tool_name=action.tool_name,
                success=False,
                error=f"工具 {action.tool_name} 需要 PC 端审批（autonomy={autonomy}）",
            )
        result = spec.call(action.params, action.context)
        self._history.append(result)
        return result

    def record_result(self, result: ActionResult) -> None:
        """记录执行结果（由审批通过后的外部调用触发）。"""
        self._history.append(result)

    @staticmethod
    def _noop_fn(params: dict, context: dict) -> dict:
        """默认 no-op 实现，工具的实际逻辑由外部注入。"""
        return {"status": "not_implemented", "params": params}

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())


# ── 权限策略 ──────────────────────────────────────────────────────────

AUTONOMY_POLICY: dict[str, dict[str, str]] = {
    "supervised": {
        "query_status": "auto",
        "query_gpu": "auto",
        "alert": "auto",
        "restart_with_lower_lr": "confirm",
        "reduce_batch": "confirm",
        "enable_grad_accum": "confirm",
        "stop_training": "confirm",
    },
    "auto": {
        "query_status": "auto",
        "query_gpu": "auto",
        "alert": "auto",
        "restart_with_lower_lr": "auto",
        "reduce_batch": "auto",
        "enable_grad_accum": "auto",
        "stop_training": "confirm",
    },
    "full": {
        "query_status": "auto",
        "query_gpu": "auto",
        "alert": "auto",
        "restart_with_lower_lr": "auto",
        "reduce_batch": "auto",
        "enable_grad_accum": "auto",
        "stop_training": "auto",
    },
}


def default_registry() -> ToolRegistry:
    """创建默认工具注册表。"""
    return ToolRegistry()
