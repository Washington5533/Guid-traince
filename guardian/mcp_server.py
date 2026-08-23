"""cp_10 · MCP 工具层 (GuardianMCPServer)。

把 guardian 的观测与操作能力暴露为标准 MCP 工具，供 Claude Code / OpenClaw
等外部 agent 客户端接入。详见 docs/IMPLEMENTATION_REPORT.md。

非阻塞保证（三条硬性约束）：
1. mcp 包未安装 → 只打印说明，训练不受影响
2. 端口/资源绑定失败 → warning 日志，watchdog 循环不受影响
3. MCP server 运行时崩溃 → 不影响 guardian 看护与训练子进程

运行方式：
    # watch 内同进程后台线程（状态实时共享）
    python run.py watch --with-mcp -- python train.py

    # 独立进程（跨进程读盘），可对着已在跑的 watch 补挂
    python run.py serve --transport stdio
"""

from __future__ import annotations

import enum
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from guardian.logging_config import get_logger

logger = get_logger(__name__)


class GuardianMode(enum.Enum):
    STANDALONE = "standalone"       # agent 自主决策
    MCP_DELEGATED = "mcp"           # 外部 Claude Code 决策，agent 让位

# ---------------------------------------------------------------------------
# MCP SDK 可选导入
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
_MCP_ERROR: str | None = None

try:
    import mcp  # noqa: F401
    from mcp.server.mcpserver import MCPServer
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    _MCP_AVAILABLE = True
except ImportError:
    try:
        # 兼容 mcp<2.0（旧版 SDK）
        from mcp.server import Server as MCPServer  # type: ignore
        from mcp.server.stdio import stdio_server
        from mcp.types import Tool, TextContent
        _MCP_AVAILABLE = True
    except ImportError as exc:
        _MCP_ERROR = f"mcp 包未安装（{exc}）。MCP 能力不可用，其余功能照常。"
        MCPServer = object  # type: ignore
        Tool = dict     # type: ignore
        TextContent = str  # type: ignore


# ---------------------------------------------------------------------------
# 幂等保证
# ---------------------------------------------------------------------------

class IdempotencyGuard:
    """相同 request_id 在 dedup_window 秒内重复到达 → 返回首次结果。"""

    def __init__(self, window: float = 300):
        self.window = window
        self._seen: dict[str, tuple[float, Any]] = {}

    def check(self, request_id: str | None) -> Any | None:
        if not request_id:
            return None
        now = time.monotonic()
        if request_id in self._seen:
            ts, result = self._seen[request_id]
            if now - ts < self.window:
                return result
            del self._seen[request_id]
        return None

    def record(self, request_id: str | None, result: Any) -> None:
        if request_id:
            self._seen[request_id] = (time.monotonic(), result)

    def cleanup(self) -> None:
        """清理过期条目，避免内存泄漏。"""
        now = time.monotonic()
        expired = [rid for rid, (ts, _) in self._seen.items()
                   if now - ts >= self.window]
        for rid in expired:
            del self._seen[rid]


# ---------------------------------------------------------------------------
# 工具定义
# ---------------------------------------------------------------------------

READONLY_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_training_status",
        "description": (
            "返回当前训练状态：最新 epoch/step、loss/accuracy、GPU 状态。"
            "只读，无副作用。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "get_metrics_history",
        "description": (
            "返回完整指标时间序列。支持 limit/cursor 分页——默认返回最近 200 条"
            "加聚合统计，完整序列需分页拉取，避免塞爆 agent 上下文。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "返回条数上限，默认 200"},
                "cursor": {"type": "integer", "description": "偏移量，0=最新"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "list_checkpoints",
        "description": (
            "列出所有 checkpoint：路径、指标、是否 best/top_k。只读。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "description": "排序指标，默认 val/accuracy"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "compare_checkpoints",
        "description": (
            "对比两个 checkpoint 的指标差异。传入 cp_a 和 cp_b 的 epoch 编号。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "epoch_a": {"type": "integer", "description": "第一个 checkpoint 的 epoch"},
                "epoch_b": {"type": "integer", "description": "第二个 checkpoint 的 epoch"},
            },
            "required": ["epoch_a", "epoch_b"],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "get_anomaly_history",
        "description": (
            "全部异常事件 + 每次事件的应对来源（agent/rule_default）。只读。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "get_recovery_history",
        "description": (
            "全部重启记录：trigger（crash/intervention/hang）、恢复起点、"
            "作废 epoch 数、参数变更。只读。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "get_summary",
        "description": (
            "已生成的训练摘要（结构化 + AI 解读）。只读。"
            "已生成的摘要也可作为 MCP resource 直接引用。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "get_agent_decision_log",
        "description": (
            "全部 agent 调用记录：decision_point、动作、来源（agent/timeout/error/"
            "disabled）、延迟、上下文摘要。只读。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "get_contract_status",
        "description": (
            "cp_11 契约四项各自的开启/降级状态。只读。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "list_contract_proposals",
        "description": (
            "全部 agent 提议记录（pending/approved/rejected）及依据。只读。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string",
                           "description": "筛选状态：pending / approved / rejected"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "analyze_architecture",
        "description": (
            "分析模型架构：解析模块结构、计算 FLOPs/参数量、检测瓶颈层、"
            "生成 D3 可渲染的架构树数据。"
            "返回包含 tree/bottlenecks/stats 的 JSON，"
            "可直接用于 D3 treemap 或 backbone flow 可视化。只读。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_entry": {"type": "string",
                                "description": "模型入口（module:function），如 train:build_model"},
                "project_dir": {"type": "string",
                                "description": "项目目录（Python 路径），用于导入模型模块"},
            },
            "required": ["model_entry"],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False,
                        "idempotentHint": True},
    },
]


# ---- v2 新增只读工具 (F3/F4/F7/F10) ----

READONLY_TOOLS_V2: list[dict[str, Any]] = [
    {
        "name": "list_experiments",
        "description": "列出所有历史实验摘要（experiment_id, status, best_metric, timestamp）。只读。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "返回条数上限，默认 50"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "query_experiment",
        "description": (
            "自然语言查询实验记录。例如：'上次 mAP 最高的那次，lr 和 batch_size 是多少'。"
            "只读推理，不修改任何数据。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "自然语言问题"},
            },
            "required": ["question"],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "compare_experiments",
        "description": "对比两个实验的指标、参数、异常事件。只读。",
        "inputSchema": {
            "type": "object",
            "properties": {
                "id_a": {"type": "string", "description": "第一个实验 ID"},
                "id_b": {"type": "string", "description": "第二个实验 ID"},
            },
            "required": ["id_a", "id_b"],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "get_model_structure",
        "description": (
            "返回模型结构 JSON：节点列表（含 type/params/FLOPs/input_shape/output_shape）、"
            "边列表、总参数量、总 FLOPs。供外部 agent 分析模型瓶颈使用。只读。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_entry": {"type": "string", "description": "模型入口，如 'train:build_model'"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "get_guardian_mode",
        "description": (
            "当前 guardian 模式：standalone（agent 自主决策）或 "
            "mcp_delegated（外部 Claude Code 决策，内置 agent 已让位）。只读。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "get_gallery_config",
        "description": "当前图片筛选策略配置（如已生成）。只读。",
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "get_import_format",
        "description": (
            "返回 Guardian 导入格式规范（JSON Schema）。"
            "外部训练数据（WandB/TensorBoard/CSV 等）转换时参考此格式。只读。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "inspect_source",
        "description": (
            "读取外部数据文件的前 N 行，返回采样数据供分析列结构。"
            "用于导入外部训练数据前的格式探测。只读。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "外部数据文件路径"},
                "lines": {"type": "integer", "description": "采样行数，默认 20，上限 100"},
            },
            "required": ["file_path"],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "get_training_log",
        "description": (
            "读取训练日志文件的尾部内容。支持指定行数和偏移量。"
            "用于排查训练错误、查看崩溃前的日志、检查输出。只读。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lines": {"type": "integer", "description": "返回行数，默认 100，上限 1000"},
                "offset": {"type": "integer", "description": "偏移量（从末尾倒数），0=最新"},
                "grep": {"type": "string", "description": "过滤关键字（可选），如 'Error'、'epoch'"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "get_post_training_checklist",
        "description": (
            "训练结束后的待办清单。列出：哪些 checkpoint 可用、可以生成什么（可视化/推理/图库/摘要）、"
            "每个操作的推荐命令和参数。训练结束后应优先调用此工具，然后按清单逐项执行。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "get_pending_decisions",
        "description": (
            "获取所有待处理的 provisional 决策（MCP 模式下 agent 继续做决策，但标记为可覆盖）。"
            "每条决策含 id、决策点、临时动作、超时剩余秒数。"
            "外部 agent 审核后调用 resolve_decision 批准或覆盖。超时未处理自动转为 approved。"
            "只读。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "get_dashboard_config",
        "description": (
            "获取 Dashboard 当前配置：启用的图表组、面板显隐、平滑开关、布局模板。"
            "只读，无副作用。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "process_id": {"type": "string", "description": "训练进程 ID，默认当前活动进程"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "recommend_charts",
        "description": (
            "让 AI agent 分析当前训练状态（指标趋势、异常数量、训练阶段），"
            "推荐 Dashboard 应重点关注的图表组和显示配置（是否开平滑等）。只读。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "process_id": {"type": "string", "description": "训练进程 ID，默认当前活动进程"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": False},
    },
    {
        "name": "list_dashboard_templates",
        "description": (
            "列出可用的 Dashboard 布局模板。training=训练监控（图表+日志），"
            "comparison=实验对比，minimal=最小面板。只读。"
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
        "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    },
]

# ---- v2 新增写工具（训练后可写） ----

WRITE_TOOLS_V2: list[dict[str, Any]] = [
    {
        "name": "run_visualization",
        "description": (
            "触发生成模型管线可视化 HTML（交互式 D3.js 可折叠树，含 FLOPs/瓶颈/改进建议）。"
            "【仅在训练结束后可用】。生成类工具，不修改训练状态，无需 token。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_entry": {"type": "string", "description": "模型入口，如 'train:build_model'"},
                "output_path": {"type": "string", "description": "输出 HTML 路径，默认 ./logs/model_viz.html"},
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "set_gallery_config",
        "description": (
            "更新图片筛选策略配��，触发重新筛选（多策略：汇报精选/难样本/边界案例）。"
            "【仅在训练结束后可用】。生成类工具，无需 token。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "strategies": {"description": "筛选策略 JSON（与 propose_strategies 输出格式一致）"},
                "checkpoint_epoch": {"type": "integer", "description": "用于推理的 checkpoint epoch"},
                "data_source": {"type": "string", "description": "数据源路径"},
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": ["strategies", "checkpoint_epoch", "data_source"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    },
    {
        "name": "run_inference",
        "description": (
            "使用指定 checkpoint 对输入数据跑推理（分类/检测/分割，固定脚本）。"
            "【仅在训练结束后可用】。生成类工具，无需 token。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "checkpoint_epoch": {"type": "integer", "description": "checkpoint epoch"},
                "task_type": {"type": "string", "description": "classification | detection | segmentation"},
                "inputs": {"type": "string", "description": "输入数据路径"},
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": ["checkpoint_epoch", "task_type", "inputs"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    },
    {
        "name": "submit_import",
        "description": (
            "提交外部训练数据导入。Agent 自行转换后调用此工具，Guardian 校验格式并入库到 Dashboard 可见。"
            "支持两种方式：\n"
            "A) metrics_path: 指向本地 JSONL 文件路径（大数据）\n"
            "B) metrics: 直接传指标列表（小数据）\n"
            "需要 write_token 鉴权。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "meta": {
                    "type": "object",
                    "description": "元信息，必须含 name 字段",
                    "properties": {
                        "name": {"type": "string", "description": "实验名称"},
                        "command": {"type": "string", "description": "原始训练命令（可空）"},
                        "source": {"type": "string", "description": "数据来源标识（如 wandb, tensorboard, csv_export）"},
                    },
                    "required": ["name"],
                },
                "metrics_path": {"type": "string", "description": "本地 JSONL 文件路径（与 metrics 二选一）"},
                "metrics": {
                    "type": "array",
                    "description": "指标列表（与 metrics_path 二选一）",
                    "items": {"type": "object"},
                },
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": ["meta"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    },
    {
        "name": "resolve_decision",
        "description": (
            "处理一条待定的 provisional 决策（来自 get_pending_decisions）。\n"
            "override=false: 认可当前 provisional 决策，标记为 approved。\n"
            "override=true:  用新的 action 覆盖。如果覆盖的动作是 restart_with_lower_lr / "
            "reduce_batch / enable_grad_accum，会立即执行重启式干预（kill 训练进程 + 回滚 checkpoint）。\n"
            "【注意】override=true 且 action=stop_training 会停止训练。\n"
            "需要 write_token 鉴权。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "decision_id": {"type": "string", "description": "待处理决策 ID（从 get_pending_decisions 获取）"},
                "override": {"type": "boolean", "description": "是否覆盖（false=批准, true=覆盖）", "default": False},
                "action": {"type": "string", "description": "覆盖时的动作名（override=true 时必填）"},
                "param": {"description": "动作参数（ratio 或 steps，视动作类型而定）"},
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": ["decision_id"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False},
    },
]

WRITE_TOOLS: list[dict[str, Any]] = [
    {
        "name": "trigger_recovery",
        "description": (
            "手动触发重启恢复流程。【风险】会 kill 训练子进程并回滚到最近 checkpoint，"
            "作废其后全部算力。仅在显式授权后方可调用。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "description": "幂等键，5 分钟内重复调用返回首次结果"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True,
                        "idempotentHint": False},
    },
    # trigger_full_validate 暂未实现（依赖 buildable_entry 契约 v1），
    # 从工具列表中隐藏，待实现后恢复。handler 代码保留在 _handle_trigger_full_validate 中。
    {
        "name": "restart_with_params",
        "description": (
            "用调整后的参数重启训练（batch_size / lr 等，受 cp_11 白名单与 "
            "cli_mappings 约束）。【风险】会 kill 训练子进程并回滚到最近 checkpoint，"
            "作废其后全部算力。参数越界会被拒绝。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["reduce_batch", "restart_with_lower_lr", "enable_grad_accum"],
                    "description": "restart_with_lower_lr: 降低学习率 | reduce_batch: 减半 batch_size | enable_grad_accum: 梯度累积"
                },
                "param": {
                    "description": "动作参数：restart_with_lower_lr/reduce_batch 传 ratio(float, 如 0.5)；enable_grad_accum 传 steps(int)"
                },
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": ["action"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True,
                        "idempotentHint": False},
    },
    {
        "name": "stop_training",
        "description": (
            "停止训练子进程并终止看护。训练中止，需人工重新拉起。"
            "已停止时重复调用无副作用。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": True,
                        "idempotentHint": True},
    },
    {
        "name": "approve_contract_proposal",
        "description": (
            "批准一条 agent 的契约扩展提议，写入正式注册表/白名单。"
            "批准后扩大 agent 后续可自主选择的空间，需连同 evidence 审阅。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string", "description": "提议 ID"},
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": ["proposal_id"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "reject_contract_proposal",
        "description": (
            "拒绝并归档一条 agent 的契约扩展提议。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "proposal_id": {"type": "string", "description": "提议 ID"},
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": ["proposal_id"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False,
                        "idempotentHint": True},
    },
    {
        "name": "set_dashboard_config",
        "description": (
            "设置 Dashboard 配置：图表组选择、面板显隐、平滑开关、布局模板。"
            "【需 write token】。外部 agent 可通过此工具调整 Dashboard 展示，"
            "Dashboard 前端会通过 WebSocket 实时收到变更。用户手动操作（checkbox/滑块）"
            "不受此工具覆盖——用户的本地操作优先级始终最高。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "process_id": {"type": "string", "description": "训练进程 ID，默认当前活动进程"},
                "charts": {"description": "图表配置: {\"default_groups\": [\"loss\",\"accuracy\"], \"smoothing\": true, \"range_mode\": \"auto\"}"},
                "panels": {"description": "面板显隐: {\"cursor_info\": true, \"logs\": true, \"ai_chat\": false}"},
                "template": {"type": "string", "description": "布局模板: training | comparison | minimal"},
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": [],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    },
]


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

class GuardianMCPServer:
    """把 guardian 能力暴露为 MCP 工具。

    mode="shared": --with-mcp，与 guardian 同进程，直接共享内存中的模块实例。
    mode="standalone": run.py serve，跨进程定期读盘刷新状态。
    """

    def __init__(
        self,
        config: dict | None = None,
        monitor: Any = None,
        ckpt_analyzer: Any = None,
        watchdog: Any = None,
        summary_gen: Any = None,
        advisor: Any = None,
        task_contract: Any = None,
        *,
        mode: str = "shared",
        state_dir: str | Path | None = None,
        dash_url: str | None = None,
    ):
        self.cfg = config or {}
        self.mcp_cfg = self.cfg.get("mcp") or {}

        self.monitor = monitor
        self.ckpt_analyzer = ckpt_analyzer
        self.watchdog = watchdog
        self.summary_gen = summary_gen
        self.advisor = advisor
        self.task_contract = task_contract

        self.mode = mode
        self.state_dir = Path(state_dir) if state_dir else Path(
            self.cfg.get("project", {}).get("log_dir", "./logs")
        )

        self.write_enabled = bool(self.mcp_cfg.get("enable_write_tools", False))
        self.write_token = os.environ.get(
            str(self.mcp_cfg.get("write_token_env", "GUARDIAN_MCP_TOKEN")) or ""
        )
        self.default_limit = int(self.mcp_cfg.get("default_result_limit", 200))
        self.dedup_window = float(self.mcp_cfg.get("dedup_window", 300))
        self.refresh_interval = float(self.mcp_cfg.get("state_refresh_interval", 5))

        self.idem = IdempotencyGuard(self.dedup_window)
        self.access_log_path = self.state_dir / "mcp_access_log.json"
        self._last_snapshot = 0.0
        self._snapshot_cache: dict[str, Any] = {}

        # Dashboard 通信（MCP 工具通过 HTTP 与 Dashboard 交互）
        self.dash_url = dash_url

        # 双模式架构
        self.mode_state = GuardianMode.STANDALONE
        self._training_active = True  # 默认训练进行中
        self._gallery_config: dict | None = None  # 缓存的图集配置
        self._transport: str = "stdio"  # 当前传输方式，stdio 模式跳过 token 鉴权

    # ------------------------------------------------------------------
    # 可用性检查
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> tuple[bool, str | None]:
        """mcp 包是否可用。不可用时返回 (False, 原因)。"""
        if _MCP_AVAILABLE:
            return True, None
        return False, _MCP_ERROR

    # ------------------------------------------------------------------
    # 授权
    # ------------------------------------------------------------------

    def _authorize(self, tool_name: str, token: str | None = None) -> tuple[bool, str]:
        # stdio 模式：本地进程间通信，OS 进程隔离即为安全边界，跳过 token 鉴权
        if self._transport == "stdio":
            return True, "ok"
        # SSE / HTTP 模式：网络可达，需要 token 鉴权
        if not self.write_enabled:
            return False, f"写工具 {tool_name!r} 未启用（enable_write_tools=false）"
        if not self.write_token:
            return False, "未配置 write_token_env 环境变量"
        if token != self.write_token:
            return False, "鉴权失败：token 不匹配"
        return True, "ok"

    # ------------------------------------------------------------------
    # 双模式管理
    # ------------------------------------------------------------------

    def set_mode(self, new_mode: GuardianMode) -> None:
        """切换 guardian 模式，同步通知 advisor 让位/恢复。"""
        prev = self.mode_state
        self.mode_state = new_mode
        if prev != new_mode:
            if self.advisor is not None and hasattr(self.advisor, "set_delegated"):
                self.advisor.set_delegated(new_mode == GuardianMode.MCP_DELEGATED)
            self._log_access("set_guardian_mode", None,
                           {"from": prev.value, "to": new_mode.value},
                           {"status": "ok"}, True)

    def on_client_connect(self) -> None:
        """MCP 客户端连接时调用：切换为让位模式。"""
        self.set_mode(GuardianMode.MCP_DELEGATED)

    def on_client_disconnect(self) -> None:
        """MCP 客户端断开时调用：恢复自主决策。"""
        self.set_mode(GuardianMode.STANDALONE)

    def get_mode(self) -> dict[str, str]:
        """返回当前模式信息。"""
        return {
            "mode": self.mode_state.value,
            "description": (
                "外部 Claude Code 决策中，内置 agent 已让位"
                if self.mode_state == GuardianMode.MCP_DELEGATED
                else "agent 自主决策中"
            ),
        }

    # ------------------------------------------------------------------
    # 训练阶段感知
    # ------------------------------------------------------------------

    @property
    def training_active(self) -> bool:
        return self._training_active

    @training_active.setter
    def training_active(self, value: bool) -> None:
        self._training_active = bool(value)

    def _check_post_training(self, tool_name: str) -> tuple[bool, str]:
        """检查工具是否仅在训练后可用的约束。"""
        if self._training_active:
            return False, (
                f"工具 {tool_name!r} 仅在训练结束后可用。"
                f"当前训练仍在进行中。"
            )
        return True, "ok"

    # ------------------------------------------------------------------
    # 授权（扩展版：阶段 + token 双重检查）
    # ------------------------------------------------------------------

    def _authorize_post_training(
        self, tool_name: str, token: str | None = None,
        require_token: bool = False,
    ) -> tuple[bool, str]:
        """训练后检查 + 可选 token 鉴权。

        require_token=False: 生成类工具（viz/gallery/inference），不破坏训练状态，免 token。
        require_token=True:  导入类工具（submit_import），修改持久化数据，需 token。
        """
        if require_token:
            ok, msg = self._authorize(tool_name, token)
            if not ok:
                return False, msg
        ok, msg = self._check_post_training(tool_name)
        if not ok:
            return False, msg
        return True, "ok"

    # ------------------------------------------------------------------
    # 访问日志
    # ------------------------------------------------------------------

    def _log_access(self, tool_name: str, client_id: str | None,
                    params: dict, result: Any, success: bool,
                    deduplicated: bool = False) -> None:
        entry = {
            "tool": tool_name,
            "client_id": client_id,
            "params": _safe_serialize(params),
            "success": success,
            "deduplicated": deduplicated,
            "timestamp": time.time(),
        }
        try:
            self.access_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.access_log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # 跨进程状态快照（standalone 模式）
    # ------------------------------------------------------------------

    def _snapshot_state(self) -> None:
        """从磁盘刷新内存快照（standalone 模式定期调用）。"""
        now = time.monotonic()
        if now - self._last_snapshot < self.refresh_interval:
            return
        self._last_snapshot = now
        # 读盘逻辑：standalone 模式下 guardian 模块把状态写到 state_dir 下
        for fname in ("metrics_history.json", "anomaly_history.json",
                       "restart_history.json", "summary.json"):
            path = self.state_dir / fname
            if path.exists():
                try:
                    self._snapshot_cache[fname] = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                except (ValueError, OSError):
                    pass
        # 扫描 checkpoint 目录
        self._snapshot_checkpoints()

    def _snapshot_checkpoints(self) -> None:
        """扫描 state_dir 上级目录的 checkpoints/ 目录（standalone 回退用）。"""
        ckpt_root = self.state_dir.parent / "checkpoints"
        if not ckpt_root.is_dir():
            # 也尝试 state_dir 同级的 checkpoints
            ckpt_root = self.state_dir / "checkpoints"
        if not ckpt_root.is_dir():
            return
        ckpts = []
        for d in sorted(ckpt_root.iterdir()):
            if not d.is_dir() or not d.name.startswith("cp_"):
                continue
            try:
                epoch = int(d.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            metrics = {}
            metrics_file = d / "metrics.json"
            if metrics_file.is_file():
                try:
                    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    pass
            ckpts.append({"epoch": epoch, "path": str(d), "metrics": metrics})
        self._snapshot_cache["checkpoints"] = ckpts

    # ------------------------------------------------------------------
    # 工具处理器 —— 只读
    # ------------------------------------------------------------------

    def _handle_training_status(self, **kwargs) -> str:
        if self.monitor is not None:
            hist = self.monitor.get_metrics_history()
            latest = hist[-1] if hist else {}
            gpu_hist = getattr(self.monitor, "get_gpu_history", lambda: [])()
            latest_gpu = gpu_hist[-1] if gpu_hist else {}
            anomaly_count = len(self.monitor.get_anomaly_history())
        else:
            # standalone 模式：从磁盘快照重建
            hist = self._snapshot_cache.get("metrics_history.json", [])
            latest = hist[-1] if hist else {}
            latest_gpu = {}
            anomalies = self._snapshot_cache.get("anomaly_history.json", [])
            anomaly_count = len(anomalies)
        return json.dumps({
            "latest_metrics": latest,
            "total_records": len(hist),
            "latest_gpu": latest_gpu,
            "anomaly_count": anomaly_count,
            "source": "live" if self.monitor else "snapshot",
        }, ensure_ascii=False, indent=2)

    def _handle_metrics_history(self, limit: int = 200, cursor: int = 0, **kwargs) -> str:
        if self.monitor is not None:
            hist = self.monitor.get_metrics_history()
        else:
            hist = self._snapshot_cache.get("metrics_history.json", [])
        total = len(hist)
        start = max(0, total - limit - cursor)
        end = total - cursor if cursor else total
        window = hist[max(0, start):end]
        losses = [r["loss"] for r in hist if isinstance(r.get("loss"), (int, float))]
        agg = {}
        if losses:
            agg["loss_min"] = min(losses)
            agg["loss_max"] = max(losses)
            agg["loss_avg"] = round(sum(losses) / len(losses), 6)
        return json.dumps({
            "total": total, "returned": len(window),
            "cursor": cursor, "limit": limit,
            "aggregates": agg,
            "metrics": window,
            "source": "live" if self.monitor else "snapshot",
        }, ensure_ascii=False, indent=2)

    def _handle_list_checkpoints(self, metric: str = "val/accuracy", **kwargs) -> str:
        if self.ckpt_analyzer is not None:
            report = self.ckpt_analyzer.report(metric)
            return json.dumps(report, ensure_ascii=False, indent=2)
        # standalone 模式：从磁盘快照构建
        ckpts = self._snapshot_cache.get("checkpoints", [])
        if not ckpts:
            return json.dumps({"error": "checkpoint analyzer 未绑定，且无磁盘快照"}, ensure_ascii=False)
        best = None
        if ckpts:
            best_val = None
            for c in ckpts:
                v = c["metrics"].get(metric)
                if isinstance(v, (int, float)):
                    if best_val is None or v > best_val:
                        best_val = v
                        best = c
        return json.dumps({
            "total": len(ckpts),
            "latest": ckpts[-1] if ckpts else None,
            "best": {"epoch": best["epoch"], "metric_val": best_val, "metric": metric} if best else None,
            "checkpoints": ckpts,
        }, ensure_ascii=False, indent=2)

    def _handle_compare_checkpoints(self, epoch_a: int, epoch_b: int, **kwargs) -> str:
        if self.ckpt_analyzer is not None:
            info_a = self.ckpt_analyzer.known.get(epoch_a)
            info_b = self.ckpt_analyzer.known.get(epoch_b)
            if not info_a or not info_b:
                return json.dumps({"error": "一个或两个 epoch 不存在"}, ensure_ascii=False)
            a = info_a.metrics
            b = info_b.metrics
        else:
            # standalone：从快照中查找
            ckpts = {c["epoch"]: c["metrics"] for c in self._snapshot_cache.get("checkpoints", [])}
            a = ckpts.get(epoch_a)
            b = ckpts.get(epoch_b)
            if not a or not b:
                return json.dumps({"error": "一个或两个 epoch 不存在"}, ensure_ascii=False)
        all_keys = set(a.keys()) | set(b.keys())
        diffs = {}
        for k in sorted(all_keys):
            va, vb = a.get(k), b.get(k)
            if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                diffs[k] = {"cp_a": va, "cp_b": vb, "delta": round(vb - va, 6)}
        return json.dumps({
            "epoch_a": epoch_a, "epoch_b": epoch_b,
            "diffs": diffs,
        }, ensure_ascii=False, indent=2)

    def _handle_anomaly_history(self, **kwargs) -> str:
        if self.monitor is not None:
            return json.dumps(self.monitor.get_anomaly_history(), ensure_ascii=False, indent=2)
        data = self._snapshot_cache.get("anomaly_history.json", [])
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _handle_recovery_history(self, **kwargs) -> str:
        if self.watchdog is not None:
            return json.dumps([r.to_dict() for r in self.watchdog.restarts],
                              ensure_ascii=False, indent=2)
        data = self._snapshot_cache.get("restart_history.json", [])
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _handle_summary(self, **kwargs) -> str:
        if self.summary_gen is not None:
            return json.dumps(self.summary_gen.generate(), ensure_ascii=False, indent=2)
        data = self._snapshot_cache.get("summary.json")
        if data is None:
            return json.dumps({"error": "summary generator 未绑定，且无历史摘要"}, ensure_ascii=False)
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _handle_agent_decision_log(self, **kwargs) -> str:
        # 优先内存（共享模式），回退到持久化 JSONL（standalone / 跨进程）
        if self.advisor is not None:
            mem = list(self.advisor.decision_log)
            if mem:
                return json.dumps(mem, ensure_ascii=False, indent=2)
            # 内存为空但可能有持久化文件
            log_path = getattr(self.advisor, "_log_path", None)
            if log_path:
                from guardian.agent_advisor import AgentAdvisor
                disk = AgentAdvisor.load_log(log_path)
                if disk:
                    return json.dumps(disk, ensure_ascii=False, indent=2)
        # standalone 模式：从 summary JSON 中读取 agent_decisions
        try:
            state_dir = Path(self._state_dir or ".")
            summaries = sorted(state_dir.glob("summary_*.json"), reverse=True)
            if summaries:
                data = json.loads(summaries[0].read_text(encoding="utf-8"))
                decisions = data.get("agent_decisions", [])
                if decisions:
                    return json.dumps(decisions, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return json.dumps([], ensure_ascii=False)

    def _handle_contract_status(self, **kwargs) -> str:
        if self.task_contract is None:
            return json.dumps({"error": "task_contract 未绑定"}, ensure_ascii=False)
        status = self.task_contract.get_capability_status()
        return json.dumps({
            "capabilities": status,
            "missing": [k for k, v in status.items() if not v],
        }, ensure_ascii=False, indent=2)

    def _handle_contract_proposals(self, status: str | None = None, **kwargs) -> str:
        if self.task_contract is None:
            return json.dumps([], ensure_ascii=False)
        proposals = getattr(self.task_contract, "list_proposals", lambda s: [])(
            status
        )
        return json.dumps(proposals, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 工具处理器 —— 受限写
    # ------------------------------------------------------------------

    def _handle_trigger_recovery(self, request_id: str | None = None, **kwargs) -> str:
        ok, msg = self._authorize("trigger_recovery", kwargs.get("_token"))
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        dup = self.idem.check(request_id)
        if dup is not None:
            return json.dumps(dup, ensure_ascii=False)
        if self.watchdog is None:
            result = {"error": "watchdog 未绑定"}
        else:
            self.watchdog.request_intervention("resume_unchanged", reason="MCP 手动触发恢复")
            result = {"status": "requested", "action": "resume_unchanged"}
        self.idem.record(request_id, result)
        return json.dumps(result, ensure_ascii=False)

    def _handle_restart_with_params(self, action: str, param: Any = None,
                                     request_id: str | None = None, **kwargs) -> str:
        ok, msg = self._authorize("restart_with_params", kwargs.get("_token"))
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        dup = self.idem.check(request_id)
        if dup is not None:
            return json.dumps({"deduplicated": True, **dup}, ensure_ascii=False)
        if self.watchdog is None:
            result = {"error": "watchdog 未绑定"}
        else:
            self.watchdog.request_intervention(
                action, param=param, reason=f"MCP 触发 {action}"
            )
            result = {"status": "requested", "action": action, "param": param}
        self.idem.record(request_id, result)
        return json.dumps(result, ensure_ascii=False)

    def _handle_stop_training(self, request_id: str | None = None, **kwargs) -> str:
        ok, msg = self._authorize("stop_training", kwargs.get("_token"))
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        dup = self.idem.check(request_id)
        if dup is not None:
            return json.dumps({"deduplicated": True, **dup}, ensure_ascii=False)
        if self.watchdog is None:
            result = {"error": "watchdog 未绑定"}
        else:
            self.watchdog.stop()
            result = {"status": "stopped"}
        self.idem.record(request_id, result)
        return json.dumps(result, ensure_ascii=False)

    def _handle_trigger_full_validate(self, epoch: int,
                                       request_id: str | None = None, **kwargs) -> str:
        ok, msg = self._authorize("trigger_full_validate", kwargs.get("_token"))
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        dup = self.idem.check(request_id)
        if dup is not None:
            return json.dumps({"deduplicated": True, **dup}, ensure_ascii=False)
        result = {"status": "not_implemented",
                  "note": "full_validate 依赖 buildable_entry 契约（v1），当前版本仅支持 metrics.json 读取"}
        self.idem.record(request_id, result)
        return json.dumps(result, ensure_ascii=False)

    def _handle_approve_proposal(self, proposal_id: str,
                                  request_id: str | None = None, **kwargs) -> str:
        ok, msg = self._authorize("approve_contract_proposal", kwargs.get("_token"))
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        if self.task_contract is None or not hasattr(self.task_contract, "approve_proposal"):
            result = {"error": "task_contract 未绑定或 v0 不支持提议审核"}
        else:
            result = self.task_contract.approve_proposal(proposal_id)
        return json.dumps(result, ensure_ascii=False)

    def _handle_reject_proposal(self, proposal_id: str,
                                 request_id: str | None = None, **kwargs) -> str:
        ok, msg = self._authorize("reject_contract_proposal", kwargs.get("_token"))
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        if self.task_contract is None or not hasattr(self.task_contract, "reject_proposal"):
            result = {"error": "task_contract 未绑定或 v0 不支持提议审核"}
        else:
            result = self.task_contract.reject_proposal(proposal_id)
        return json.dumps(result, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 工具处理器 —— v2 只读 (F3/F4/F7/F10)
    # ------------------------------------------------------------------

    def _handle_list_experiments(self, limit: int = 50, **kwargs) -> str:
        from .experiment_query import ExperimentQuery
        eq = ExperimentQuery({"log_dir": str(self.state_dir)}, advisor=self.advisor)
        exps = eq.list_experiments(limit=limit)
        return json.dumps({
            "total": len(exps),
            "experiments": exps,
        }, ensure_ascii=False, indent=2)

    def _handle_query_experiment(self, question: str, **kwargs) -> str:
        from .experiment_query import ExperimentQuery
        eq = ExperimentQuery({"log_dir": str(self.state_dir)}, advisor=self.advisor)
        result = eq.query(question)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _handle_compare_experiments(self, id_a: str, id_b: str, **kwargs) -> str:
        from .experiment_query import ExperimentQuery
        eq = ExperimentQuery({"log_dir": str(self.state_dir)}, advisor=self.advisor)
        result = eq.compare(id_a, id_b)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _handle_get_model_structure(self, model_entry: str | None = None, **kwargs) -> str:
        from .model_viz import ModelVisualizer
        mv = ModelVisualizer(advisor=self.advisor)

        model_fn = None
        # 尝试从合约获取 model_entry
        if not model_entry and self.task_contract:
            model_entry = self.task_contract.script.get("buildable_entry", {}).get("model_fn")
        if model_entry:
            try:
                # 加入项目目录到 sys.path（log_dir 可能在项目根下或 logs/ 子目录）
                proj_dir = str(self.state_dir)
                if proj_dir.endswith(("logs", "log")):
                    proj_dir = str(Path(proj_dir).parent)
                if proj_dir and proj_dir not in sys.path:
                    sys.path.insert(0, proj_dir)
                mod_path, fn_name = model_entry.split(":", 1)
                import importlib
                mod = importlib.import_module(mod_path)
                model_fn = getattr(mod, fn_name)
            except Exception as exc:
                return json.dumps({"error": f"无法 import {model_entry}: {exc}"}, ensure_ascii=False)

        if model_fn is None:
            return json.dumps({"error": "需要 model_entry 参数，或在 contract 中声明 buildable_entry.model_fn"}, ensure_ascii=False)

        graph = mv.parse_model(model_fn)
        if "error" in graph:
            return json.dumps(graph, ensure_ascii=False)
        stats = mv.compute_stats(graph)
        return json.dumps({**graph, "layer_stats": stats.get("layer_stats", [])},
                         ensure_ascii=False, indent=2)

    def _handle_analyze_architecture(
        self, model_entry: str | None = None, project_dir: str = "", **kwargs
    ) -> str:
        """分析模型架构：FLOPs / 参数 / 瓶颈 / D3 tree data。"""
        from .arch_analyzer import ArchAnalyzer

        analyzer = ArchAnalyzer()

        # 从合约获取 model_entry
        if not model_entry and self.task_contract:
            model_entry = self.task_contract.script.get("buildable_entry", {}).get(
                "model_fn", ""
            )
        if not model_entry:
            return json.dumps(
                {"error": "需要 model_entry 参数，或在 contract 中声明 buildable_entry.model_fn"},
                ensure_ascii=False,
            )

        try:
            proj_dir = project_dir or str(self.state_dir)
            if proj_dir.endswith(("logs", "log")):
                proj_dir = str(Path(proj_dir).parent)
            if proj_dir not in sys.path:
                sys.path.insert(0, proj_dir)
            result = _mcp_run_arch_analyzer(analyzer, model_entry, proj_dir)
        except Exception as exc:
            return json.dumps({"error": f"架构分析失败: {exc}"}, ensure_ascii=False)

        return json.dumps(result, ensure_ascii=False, indent=2)

    def _handle_get_guardian_mode(self, **kwargs) -> str:
        mode_info = self.get_mode()

        # ---- 使用指南（Agent 连接后首先调用本工具，在此附带上下文）----
        readonly_tools = [
            "get_training_status", "get_metrics_history",
            "list_checkpoints", "compare_checkpoints",
            "get_anomaly_history", "get_recovery_history",
            "get_summary", "get_agent_decision_log",
            "get_contract_status", "list_contract_proposals",
            "list_experiments", "query_experiment", "compare_experiments",
            "get_model_structure", "get_guardian_mode",
            "get_gallery_config", "get_import_format", "inspect_source",
            "get_training_log", "get_post_training_checklist",
            "get_pending_decisions", "analyze_architecture",
        ]
        write_tools_during = [
            "trigger_recovery", "restart_with_params", "stop_training",
            "approve_contract_proposal", "reject_contract_proposal",
            "submit_import", "resolve_decision",
        ]
        write_tools_post = [
            "run_visualization", "set_gallery_config", "run_inference",
        ]

        mode_info["usage_guide"] = {
            "overview": (
                "Guardian MCP 提供 22 个只读工具 + 10 个写工具，"
                "覆盖训练全生命周期（监控、恢复、分析、可视化、推理）。"
            ),
            "tools": {
                "read_only": readonly_tools,
                "write_during_training": write_tools_during,
                "write_after_training": write_tools_post,
            },
            "write_tools": {
                "enabled": self.write_enabled,
                "token_configured": bool(self.write_token),
                "usage": (
                    "写工具需要 write_token 参数鉴权。"
                    "token 通过环境变量 GUARDIAN_MCP_TOKEN 配置。"
                    "如未启用，仅可使用只读工具。"
                ),
            },
            "training_phase": {
                "active": self._training_active,
                "note": (
                    "run_visualization / set_gallery_config / run_inference "
                    "仅在训练结束后可用，训练中调用会返回错误。"
                    "训练结束后可调用 get_post_training_checklist 获取待办清单。"
                ),
            },
            "recommended_workflow": [
                "1. 调用 get_training_status 了解当前状态",
                "2. 训练中: 用 get_metrics_history / get_anomaly_history 监控",
                "3. 训练中: 如需干预, 用 restart_with_params (需 token)",
                "4. 训练后: 先调 get_post_training_checklist 获取待办",
                "5. 训练后: 调 get_summary 获取摘要 + AI 解读",
                "6. 训练后: 调 run_visualization / run_inference 分析和测试",
                "7. 跨实验: 用 list_experiments + query_experiment 探索历史",
            ],
            "mcp_mode": (
                "当前为 MCP delegated 模式: 你（外部 Agent）拥有完整决策权，"
                "内置 agent 已让位。断开连接后自动恢复 standalone 模式。"
                if self.mode_state == GuardianMode.MCP_DELEGATED
                else "当前为 standalone 模式: 内置 agent 自主决策。"
                      "MCP 连接后可切换为 delegated 模式。"
            ),
        }

        return json.dumps(mode_info, ensure_ascii=False, indent=2)

    def _handle_get_gallery_config(self, **kwargs) -> str:
        if self._gallery_config is None:
            return json.dumps({"error": "尚未生成图集配置"}, ensure_ascii=False)
        return json.dumps(self._gallery_config, ensure_ascii=False, indent=2)

    def _handle_get_import_format(self, **kwargs) -> str:
        """返回 Guardian 导入格式规范。"""
        spec = {
            "meta": {
                "required": ["name"],
                "optional": ["command", "source", "project_dir"],
                "description": "元信息，name 为必填字符串",
            },
            "metrics": {
                "format": "JSONL（每行一个 JSON 对象）",
                "required": "至少含一个数值类型的 key",
                "optional_index": ["step", "epoch"],
                "suggested_keys": ["loss", "acc", "lr", "mAP"],
                "example": {"step": 0, "loss": 2.1, "acc": 0.12, "lr": 0.001},
                "description": (
                    "step/epoch 为可选序号字段，Dashboard 自动分配索引。"
                    "key 命名建议用标准名，Dashboard 自动分组。"
                ),
            },
            "submit_import": {
                "方式A_文件路径": {
                    "meta": {"name": "实验名", "source": "wandb"},
                    "metrics_path": "./path/to/metrics.jsonl",
                },
                "方式B_直接传内容": {
                    "meta": {"name": "实验名", "source": "csv"},
                    "metrics": [{"step": 0, "loss": 2.1}, {"step": 1, "loss": 1.8}],
                },
                "校验规则": [
                    "meta 必须含 name（字符串）",
                    "metrics 每条必须为 dict，至少含一个数值字段",
                    "metrics_path 存在且为合法 JSONL",
                    "单次指标上限 100000 条",
                ],
            },
        }
        return json.dumps(spec, ensure_ascii=False, indent=2)

    def _handle_inspect_source(self, file_path: str | None = None,
                               lines: int = 20, **kwargs) -> str:
        """采样外部数据文件的前 N 行（仅限 state_dir 及其子目录）。"""
        if not file_path:
            return json.dumps({"error": "缺少 file_path 参数"}, ensure_ascii=False)
        lines = min(max(int(lines or 20), 1), 100)
        p = Path(file_path).resolve()
        # 安全检查：仅允许读取 state_dir 及其子目录内的文件
        try:
            p.relative_to(self.state_dir.resolve())
        except ValueError:
            return json.dumps({
                "error": f"路径超出允许范围",
                "detail": f"仅允许读取 {self.state_dir} 目录内的文件",
            }, ensure_ascii=False)
        if not p.is_file():
            return json.dumps({"error": f"文件不存在: {file_path}"}, ensure_ascii=False)
        try:
            sampled = []
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f):
                    if i >= lines:
                        break
                    sampled.append(line.rstrip("\n\r"))
            # 自动检测格式提示
            hints = []
            if sampled:
                first = sampled[0]
                if "\t" in first:
                    hints.append("检测到 TSV 格式（tab 分隔）")
                elif "," in first:
                    hints.append("检测到 CSV 格式（逗号分隔）")
                if first.startswith("{"):
                    hints.append("检测到 JSON/JSONL 格式")
                if first.startswith("#") or first.startswith("//"):
                    hints.append("首行为注释")
            return json.dumps({
                "file_path": str(p),
                "total_lines_sampled": len(sampled),
                "lines": sampled,
                "format_hints": hints,
            }, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps({"error": f"读取失败: {exc}"}, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 工具处理器 —— v2 受限写 (F3/F10/F7)
    # ------------------------------------------------------------------

    def _handle_run_visualization(self, model_entry: str | None = None,
                                  output_path: str | None = None,
                                  request_id: str | None = None, **kwargs) -> str:
        ok, msg = self._authorize_post_training("run_visualization", require_token=False)
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        dup = self.idem.check(request_id)
        if dup is not None:
            return json.dumps(dup, ensure_ascii=False)

        from .model_viz import ModelVisualizer
        mv = ModelVisualizer(advisor=self.advisor)

        model_fn = None
        if model_entry:
            try:
                mod_path, fn_name = model_entry.split(":", 1)
                import importlib
                mod = importlib.import_module(mod_path)
                model_fn = getattr(mod, fn_name)
            except Exception as exc:
                result = {"error": f"无法 import {model_entry}: {exc}"}
                self.idem.record(request_id, result)
                return json.dumps(result, ensure_ascii=False)

        if model_fn is None:
            result = {"error": "需要 model_entry 参数，如 'train:build_model'"}
            self.idem.record(request_id, result)
            return json.dumps(result, ensure_ascii=False)

        out = output_path or str(self.state_dir / "model_viz.html")
        viz_result = mv.visualize(model_fn, output_path=out)
        if "error" in viz_result:
            result = viz_result
        else:
            result = {
                "status": "completed",
                "output_path": viz_result["output_path"],
                "total_params": viz_result["stats"].get("total_params"),
                "total_flops": viz_result["stats"].get("total_flops"),
                "bottleneck_count": len(viz_result["viz_config"].get("bottlenecks", [])),
            }
        self.idem.record(request_id, result)
        return json.dumps(result, ensure_ascii=False)

    def _handle_set_gallery_config(self, strategies: Any,
                                   checkpoint_epoch: int,
                                   data_source: str,
                                   request_id: str | None = None, **kwargs) -> str:
        ok, msg = self._authorize_post_training("set_gallery_config", require_token=False)
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        dup = self.idem.check(request_id)
        if dup is not None:
            return json.dumps(dup, ensure_ascii=False)

        if isinstance(strategies, str):
            try:
                strategies = json.loads(strategies)
            except json.JSONDecodeError:
                result = {"error": "strategies 必须为 JSON 对象"}
                self.idem.record(request_id, result)
                return json.dumps(result, ensure_ascii=False)

        self._gallery_config = strategies

        from .gallery import GalleryManager
        from .inference import InferenceRunner

        gm = GalleryManager(advisor=self.advisor, ckpt_analyzer=self.ckpt_analyzer)
        ir = InferenceRunner()

        ckpt_dir = self.cfg.get("project", {}).get("ckpt_dir", "./checkpoints")
        ckpt_path = Path(ckpt_dir) / f"cp_{checkpoint_epoch}" / "model.pth"
        if not ckpt_path.exists():
            result = {"error": f"checkpoint 不存在: {ckpt_path}"}
            self.idem.record(request_id, result)
            return json.dumps(result, ensure_ascii=False)

        try:
            gallery_result = gm.execute(ckpt_path, strategies, data_source, inference_runner=ir)
        except Exception as exc:
            result = {"error": f"执行失败: {exc}"}
            self.idem.record(request_id, result)
            return json.dumps(result, ensure_ascii=False)

        if "error" in gallery_result:
            result = gallery_result
        else:
            result = {
                "status": "completed",
                "galleries": {name: len(imgs) for name, imgs in gallery_result.items()},
            }
        self.idem.record(request_id, result)
        return json.dumps(result, ensure_ascii=False)

    def _handle_run_inference(self, checkpoint_epoch: int, task_type: str, inputs: str,
                              request_id: str | None = None, **kwargs) -> str:
        ok, msg = self._authorize_post_training("run_inference", require_token=False)
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        dup = self.idem.check(request_id)
        if dup is not None:
            return json.dumps(dup, ensure_ascii=False)

        from .inference import InferenceRunner

        ir = InferenceRunner()
        ckpt_dir = self.cfg.get("project", {}).get("ckpt_dir", "./checkpoints")
        ckpt_path = Path(ckpt_dir) / f"cp_{checkpoint_epoch}" / "model.pth"
        if not ckpt_path.exists():
            result = {"error": f"checkpoint 不存在: {ckpt_path}"}
            self.idem.record(request_id, result)
            return json.dumps(result, ensure_ascii=False)

        try:
            infer_result = ir.run(
                checkpoint_path=ckpt_path,
                task_type=task_type,
                inputs=inputs,
                output_dir=str(self.state_dir / "inference"),
            )
        except Exception as exc:
            result = {"error": f"推理失败: {exc}"}
            self.idem.record(request_id, result)
            return json.dumps(result, ensure_ascii=False)

        self.idem.record(request_id, infer_result)
        return json.dumps(infer_result, ensure_ascii=False, indent=2)

    def _handle_submit_import(self, meta: dict | None = None,
                             metrics_path: str | None = None,
                             metrics: list | None = None,
                             request_id: str | None = None, **kwargs) -> str:
        """提交外部训练数据导入。"""
        ok, msg = self._authorize("submit_import", kwargs.get("_token"))
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        dup = self.idem.check(request_id)
        if dup is not None:
            return json.dumps(dup, ensure_ascii=False)

        # 校验 meta
        if not isinstance(meta, dict) or not meta.get("name"):
            result = {"error": "meta 必须含 name 字段", "detail": "meta 必须含 name（字符串）"}
            return json.dumps(result, ensure_ascii=False)

        # 加载 metrics：优先 metrics_path，其次 metrics
        loaded_metrics: list[dict] | None = None
        if metrics_path:
            p = Path(metrics_path)
            if not p.is_file():
                result = {"error": f"metrics_path 文件不存在: {metrics_path}"}
                return json.dumps(result, ensure_ascii=False)
            try:
                loaded_metrics = []
                with open(p, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            loaded_metrics.append(json.loads(line))
            except Exception as exc:
                result = {"error": f"metrics_path 解析失败: {exc}"}
                return json.dumps(result, ensure_ascii=False)
        elif metrics is not None:
            loaded_metrics = metrics
        else:
            result = {"error": "缺少 metrics_path 或 metrics 参数（二选一）"}
            return json.dumps(result, ensure_ascii=False)

        # 校验 metrics
        if not isinstance(loaded_metrics, list) or len(loaded_metrics) == 0:
            result = {"error": "metrics 不能为空", "detail": "metrics 必须为非空列表"}
            return json.dumps(result, ensure_ascii=False)
        if len(loaded_metrics) > 100000:
            result = {"error": "指标数量超限", "detail": f"单次上限 100000 条，当前 {len(loaded_metrics)} 条"}
            return json.dumps(result, ensure_ascii=False)
        for i, m in enumerate(loaded_metrics):
            if not isinstance(m, dict):
                result = {"error": f"metrics[{i}] 格式错误", "detail": "每条必须为 dict"}
                return json.dumps(result, ensure_ascii=False)
            if not any(isinstance(v, (int, float)) for v in m.values()):
                result = {"error": f"metrics[{i}] 无数值", "detail": "每条至少含一个数值字段"}
                return json.dumps(result, ensure_ascii=False)

        # 生成唯一 process_id
        process_id = f"import_{uuid.uuid4().hex[:8]}"
        state = {
            "process_id": process_id,
            "name": meta["name"],
            "command": meta.get("command", ""),
            "project_dir": meta.get("project_dir", ""),
            "status": "imported",
            "registered_at": time.time(),
            "finished_at": time.time(),
            "config": {"source": meta.get("source", "external")},
            "model_entry": "",
            "log_file": "",
        }

        # 写入磁盘（与 dashboard 相同的格式）
        d = self.state_dir / process_id
        try:
            d.mkdir(parents=True, exist_ok=True)
            # 写 meta.json
            (d / "meta.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            # 写 metrics.jsonl
            with (d / "metrics.jsonl").open("w", encoding="utf-8") as f:
                for m in loaded_metrics:
                    f.write(json.dumps(m, ensure_ascii=False) + "\n")
        except Exception as exc:
            result = {"error": f"写入失败: {exc}"}
            self.idem.record(request_id, result)
            return json.dumps(result, ensure_ascii=False)

        result = {"process_id": process_id, "records": len(loaded_metrics), "status": "imported"}
        self.idem.record(request_id, result)
        return json.dumps(result, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # 待处理决策（MCP 模式下 agent 继续决策但标记为可覆盖）
    # ------------------------------------------------------------------

    def _handle_get_pending_decisions(self, **kwargs) -> str:
        """返回所有 pending 状态的 provisional 决策。"""
        if self.advisor is None:
            return json.dumps({
                "pending": [],
                "note": "agent 未启用，无 provisional 决策。",
            }, ensure_ascii=False, indent=2)
        pending = self.advisor.get_pending_decisions()
        return json.dumps({
            "mode": self.advisor.mode,
            "count": len(pending),
            "pending": pending,
        }, ensure_ascii=False, indent=2)

    def _handle_resolve_decision(
        self, decision_id: str,
        override: bool = False,
        action: str | None = None,
        param: Any = None,
        request_id: str | None = None,
        **kwargs,
    ) -> str:
        """外部 agent 处理一条 provisional 决策。"""
        ok, msg = self._authorize("resolve_decision", kwargs.get("_token"))
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)

        if self.advisor is None:
            return json.dumps({"error": "agent 未启用，无待处理决策"}, ensure_ascii=False)

        if override and not action:
            return json.dumps({
                "error": "override=true 时必须提供 action 参数",
            }, ensure_ascii=False)

        result = self.advisor.resolve_decision(
            decision_id, action=action, param=param, override=override,
        )

        if result["status"] in ("not_found", "already_resolved"):
            return json.dumps(result, ensure_ascii=False)

        # 如果外部 agent 覆盖了决策且需要补救操作，通过 watchdog 执行
        if result.get("corrective_needed") and self.watchdog is not None:
            corrective = result["corrective"]
            target_action = corrective["action"]
            target_param = corrective.get("param")

            # 映射到 watchdog 可执行的动作
            if target_action in ("restart_with_lower_lr", "reduce_batch",
                                "enable_grad_accum", "resume_unchanged"):
                self.watchdog.request_intervention(
                    target_action, param=target_param,
                    reason=f"MCP resolve_decision 覆盖 {decision_id}：{target_action}",
                )
                result["intervention"] = "requested"
            elif target_action == "stop_training":
                self.watchdog.stop()
                result["intervention"] = "stopped"
            else:
                result["intervention"] = "skipped"
                result["note"] = (
                    f"动作 {target_action!r} 不支持自动执行。"
                    f"请用 restart_with_params 或 trigger_recovery 手动干预。"
                )

        self.idem.record(request_id, result)
        return json.dumps(result, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Dashboard 配置工具
    # ------------------------------------------------------------------

    def _dash_request(self, method: str, path: str, data: dict | None = None) -> dict:
        """向 Dashboard HTTP API 发请求，失败返回 error dict。"""
        if not self.dash_url:
            return {"error": "Dashboard 未启用（--with-dashboard），此工具不可用"}
        try:
            import urllib.request as _ur
            url = f"{self.dash_url}{path}"
            req = _ur.Request(url, method=method)
            if data:
                req.add_header("Content-Type", "application/json")
                req.data = json.dumps(data).encode()
            import time as _time
            resp = _ur.urlopen(req, timeout=5)
            result = json.loads(resp.read().decode())
            resp.close()
            return result
        except Exception as exc:
            return {"error": f"Dashboard 请求失败: {exc}"}

    def _handle_get_dashboard_config(self, process_id: str | None = None, **kwargs) -> str:
        """读取 Dashboard 当前配置。"""
        pid = process_id or self._current_process_id()
        result = self._dash_request("GET", f"/api/process/{pid}/dashboard-config")
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _handle_set_dashboard_config(self, process_id: str | None = None,
                                      charts: dict | None = None,
                                      panels: dict | None = None,
                                      template: str | None = None,
                                      request_id: str | None = None,
                                      **kwargs) -> str:
        """设置 Dashboard 配置（需 write token）。"""
        ok, msg = self._authorize("set_dashboard_config", kwargs.get("_token"))
        if not ok:
            return json.dumps({"error": msg}, ensure_ascii=False)
        pid = process_id or self._current_process_id()
        payload = {"_source": "mcp_agent"}
        if charts is not None:
            payload["charts"] = charts
        if panels is not None:
            payload["panels"] = panels
        if template is not None:
            payload["template"] = template
        result = self._dash_request("POST", f"/api/process/{pid}/dashboard-config", payload)
        self.idem.record(request_id, result)
        return json.dumps(result, ensure_ascii=False, indent=2)

    def _handle_recommend_charts(self, process_id: str | None = None, **kwargs) -> str:
        """AI 推荐 Dashboard 图表配置。"""
        pid = process_id or self._current_process_id()
        if self.advisor is None or not self.advisor.is_enabled("chart_selection"):
            return json.dumps({
                "error": "agent 未启用（需 --agent 且配置 API key），chart_selection 决策点不可用",
                "fallback": {"groups": ["loss", "accuracy"], "smoothing": False},
            }, ensure_ascii=False, indent=2)

        # 从 Dashboard 获取当前状态
        dash = self._dash_request("GET", f"/api/process/{pid}/dashboard-config")
        # 获取指标摘要
        try:
            import urllib.request as _ur
            url = f"{self.dash_url}/api/process/{pid}/metrics?limit=200"
            resp = _ur.urlopen(url, timeout=5)
            mdata = json.loads(resp.read().decode())
            resp.close()
            hist = mdata.get("metrics", [])
        except Exception:
            hist = []

        # 构建指标摘要
        summary = {}
        if hist:
            for k in hist[-1]:
                if k in ("step", "epoch", "timestamp", "_group", "_ts"):
                    continue
                vals = [m[k] for m in hist[-50:] if k in m and isinstance(m[k], (int, float))]
                if vals:
                    summary[k] = {"last": vals[-1], "min": min(vals), "max": max(vals),
                                  "trend": "rising" if len(vals) > 5 and vals[-1] > vals[-5] else "falling"}

        # 训练阶段推测
        total = len(hist)
        if total < 50:
            phase = "early"
        elif total < 500:
            phase = "mid"
        else:
            phase = "late"

        # 调用 agent
        available = dash.get("charts", {}).get("default_groups", ["loss", "accuracy", "lr", "gpu", "custom"])
        result = self.advisor.recommend_charts(
            process_id=pid,
            metrics_summary=summary,
            chart_groups=available,
            anomaly_count=0,
            training_phase=phase,
        )
        if result is None:
            return json.dumps({
                "error": "agent 推荐失败，使用默认配置",
                "fallback": {"groups": ["loss", "accuracy"], "smoothing": False},
            }, ensure_ascii=False, indent=2)
        return json.dumps({"recommendation": result, "source": "agent"}, ensure_ascii=False, indent=2)

    def _handle_list_dashboard_templates(self, **kwargs) -> str:
        """列出可用 Dashboard 布局模板。"""
        templates = {
            "templates": [
                {
                    "name": "training",
                    "description": "训练监控：图表区（loss/accuracy/lr/gpu）+ 坐标信息 + 日志 + AI 对话",
                    "panels": {"cursor_info": True, "logs": True, "ai_chat": True},
                },
                {
                    "name": "comparison",
                    "description": "实验对比：多个进程的图表并列 + 指标对比表格",
                    "panels": {"cursor_info": False, "logs": False, "ai_chat": True},
                },
                {
                    "name": "minimal",
                    "description": "最小面板：仅图表区，适合嵌入或低带宽环境",
                    "panels": {"cursor_info": False, "logs": False, "ai_chat": False},
                },
            ],
            "default": "training",
        }
        return json.dumps(templates, ensure_ascii=False, indent=2)

    def _current_process_id(self) -> str:
        """获取当前活动进程 ID。"""
        if self.monitor is not None and hasattr(self.monitor, "process_id"):
            return self.monitor.process_id
        return "guardian-run"

    # ------------------------------------------------------------------
    # 工具路由
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 训练日志读取
    # ------------------------------------------------------------------

    def _handle_get_training_log(self, lines: int = 100, offset: int = 0,
                                  grep: str | None = None, **kwargs) -> str:
        """读取训练日志尾部。"""
        lines = min(max(int(lines or 100), 1), 1000)
        offset = max(int(offset or 0), 0)

        # 尝试从 task_contract 获取日志路径
        log_path = None
        if self.task_contract is not None:
            ch = self.task_contract.metrics_channel()
            if ch and isinstance(ch, dict):
                log_path = ch.get("path")

        # 回退：扫描 state_dir 父目录下的 logs/
        if not log_path or not Path(log_path).is_file():
            for candidate in [
                self.state_dir / "train.log",
                self.state_dir.parent / "logs" / "train.log",
                self.state_dir / ".." / "logs" / "train.log",
            ]:
                p = Path(candidate).resolve()
                if p.is_file():
                    log_path = str(p)
                    break

        if not log_path:
            return json.dumps({
                "error": "未找到训练日志文件",
                "detail": "请确认 contract.yaml 中 metrics_channel.path 配置正确，或日志文件在 logs/ 目录下",
            }, ensure_ascii=False)

        try:
            text = Path(log_path).read_text(encoding="utf-8", errors="replace")
            all_lines = text.splitlines()
            total = len(all_lines)
            start = max(0, total - lines - offset)
            end = total - offset if offset else total
            window = all_lines[start:end]

            # 可选过滤
            if grep:
                window = [l for l in window if grep.lower() in l.lower()]

            return json.dumps({
                "log_file": log_path,
                "total_lines": total,
                "returned": len(window),
                "lines": window,
                "offset": offset,
                "grep": grep,
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return json.dumps({"error": f"读取日志失败: {e}"}, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 训练后待办清单（主动提示 Claude Code 下一步做什么）
    # ------------------------------------------------------------------

    def _handle_post_training_checklist(self, **kwargs) -> str:
        """扫描可用资源，生成训练后待办清单。"""
        items = []
        training_done = not self._training_active

        # 1. 检查 checkpoint
        ckpts = self._snapshot_cache.get("checkpoints", [])
        if not ckpts and self.ckpt_analyzer is not None:
            try:
                report = self.ckpt_analyzer.report()
                ckpts = report.get("checkpoints", [])
            except Exception:
                pass
        if ckpts:
            best = max(ckpts, key=lambda c: c.get("metrics", {}).get("val/accuracy", 0)) if ckpts else None
            items.append({
                "category": "checkpoint",
                "title": "最佳 Checkpoint 分析",
                "available": True,
                "detail": f"共 {len(ckpts)} 个 checkpoint" + (f"，最佳 epoch={best['epoch']}" if best else ""),
                "suggested_tool": "list_checkpoints",
                "suggested_args": {},
            })

        # 2. 模型可视化
        items.append({
            "category": "analysis",
            "title": "模型结构可视化",
            "available": training_done,
            "detail": "生成交互式 D3.js 模型管线图（FLOPs + 瓶颈标注 + 改进建议）" if training_done else "训练结束后可用",
            "suggested_tool": "run_visualization",
            "suggested_args": {"model_entry": "train:build_model（需根据实际模块名调整）"},
        })

        # 3. 推理
        if ckpts:
            best_epoch = best["epoch"] if best else ckpts[-1]["epoch"]
            items.append({
                "category": "evaluation",
                "title": f"对最佳 Checkpoint (epoch={best_epoch}) 跑推理",
                "available": training_done,
                "detail": "分类/检测/分割推理，生成结果 JSON" if training_done else "训练结束后可用",
                "suggested_tool": "run_inference",
                "suggested_args": {"checkpoint_epoch": best_epoch, "task_type": "classification"},
            })

        # 4. 图片筛选
        if ckpts:
            best_epoch = best["epoch"] if best else ckpts[-1]["epoch"]
            items.append({
                "category": "evaluation",
                "title": "图片筛选与展示",
                "available": training_done,
                "detail": "多策略筛选（汇报精选/难样本/边界案例），可选 Streamlit 展示" if training_done else "训练结束后可用",
                "suggested_tool": "set_gallery_config",
                "suggested_args": {"checkpoint_epoch": best_epoch, "data_source": "./data/test"},
            })

        # 5. 摘要
        summary = self._snapshot_cache.get("summary.json")
        items.append({
            "category": "report",
            "title": "训练摘要",
            "available": summary is not None,
            "detail": "结构化摘要 + AI 解读" if summary else "尚未生成",
            "suggested_tool": "get_summary",
            "suggested_args": {},
        })

        # 6. 实验查询
        items.append({
            "category": "cross-experiment",
            "title": "跨实验对比",
            "available": True,
            "detail": "查询历史实验、对比指标、自然语言问答",
            "suggested_tool": "list_experiments",
            "suggested_args": {},
        })

        available_count = sum(1 for it in items if it["available"])
        return json.dumps({
            "training_active": not training_done,
            "total_items": len(items),
            "available_now": available_count,
            "checklist": items,
            "hint": "训练已结束，建议按顺序执行以上待办项。先调用 get_summary 获取概览。" if training_done else "训练进行中，部分项目需要等训练结束后才能执行。",
        }, ensure_ascii=False, indent=2)

    _READ_HANDLERS: dict[str, Any]
    _WRITE_HANDLERS: dict[str, Any]

    def __init_handlers__(self) -> None:
        self._READ_HANDLERS = {
            "get_training_status": self._handle_training_status,
            "get_metrics_history": self._handle_metrics_history,
            "list_checkpoints": self._handle_list_checkpoints,
            "compare_checkpoints": self._handle_compare_checkpoints,
            "get_anomaly_history": self._handle_anomaly_history,
            "get_recovery_history": self._handle_recovery_history,
            "get_summary": self._handle_summary,
            "get_agent_decision_log": self._handle_agent_decision_log,
            "get_contract_status": self._handle_contract_status,
            "list_contract_proposals": self._handle_contract_proposals,
            # v2
            "list_experiments": self._handle_list_experiments,
            "query_experiment": self._handle_query_experiment,
            "compare_experiments": self._handle_compare_experiments,
            "get_model_structure": self._handle_get_model_structure,
            "get_guardian_mode": self._handle_get_guardian_mode,
            "get_gallery_config": self._handle_get_gallery_config,
            # import
            "get_import_format": self._handle_get_import_format,
            "inspect_source": self._handle_inspect_source,
            # 日志
            "get_training_log": self._handle_get_training_log,
            # 训练后主动提示
            "get_post_training_checklist": self._handle_post_training_checklist,
            # 待处理决策（MCP 模式下 agent 继续决策但标记为可覆盖）
            "get_pending_decisions": self._handle_get_pending_decisions,
            # Dashboard 配置
            "get_dashboard_config": self._handle_get_dashboard_config,
            "recommend_charts": self._handle_recommend_charts,
            "list_dashboard_templates": self._handle_list_dashboard_templates,
            # 架构分析
            "analyze_architecture": self._handle_analyze_architecture,
        }
        self._WRITE_HANDLERS = {
            "trigger_recovery": self._handle_trigger_recovery,
            "restart_with_params": self._handle_restart_with_params,
            "stop_training": self._handle_stop_training,
            # trigger_full_validate 暂未实现，handler 保留待 v1 契约就绪后恢复
            "approve_contract_proposal": self._handle_approve_proposal,
            "reject_contract_proposal": self._handle_reject_proposal,
            # v2
            "run_visualization": self._handle_run_visualization,
            "set_gallery_config": self._handle_set_gallery_config,
            "run_inference": self._handle_run_inference,
            # import
            "submit_import": self._handle_submit_import,
            # 待处理决策覆盖
            "resolve_decision": self._handle_resolve_decision,
            # Dashboard 配置
            "set_dashboard_config": self._handle_set_dashboard_config,
        }

    def call_tool(self, name: str, arguments: dict) -> str:
        """直接调用工具（供 MCP handler 或测试使用）。"""
        # 跨进程模式下先刷新状态快照
        if self.mode == "standalone":
            self._snapshot_state()

        if name in getattr(self, "_READ_HANDLERS", {}):
            handler = self._READ_HANDLERS[name]
            result = handler(**arguments)
            self._log_access(name, arguments.get("_client_id"), arguments, result, True)
            return result

        if name in getattr(self, "_WRITE_HANDLERS", {}):
            handler = self._WRITE_HANDLERS[name]
            result = handler(**arguments)
            success = "error" not in (json.loads(result) if isinstance(result, str) else result)
            self._log_access(name, arguments.get("_client_id"), arguments, result, success)
            return result

        return json.dumps({"error": f"未知工具 {name!r}"}, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 启动
    # ------------------------------------------------------------------

    def start(self, transport: str = "stdio", *,
              host: str = "127.0.0.1", port: int | None = None) -> str:
        """启动 MCP server（阻塞）。

        transport="stdio" : 标准输入输出（供 Claude Code 子进程接入）
        transport="sse"   : SSE over HTTP（供远程 agent 通过 SSH 隧道接入）
        transport="http"  : Streamable HTTP（同 sse，走新协议）
        transport="tcp"   : SSE 的旧名别名，等同 "sse"
        """
        self._transport = transport
        available, err = self.is_available()
        if not available:
            return f"[MCP] {err}\n[MCP] 请 pip install -r requirements-mcp.txt 后重试。"

        self.__init_handlers__()
        return self._start_mcp_server(transport, host=host, port=port)

    def start_in_background(self, transport: str = "stdio", *,
                            host: str = "127.0.0.1",
                            port: int | None = None) -> threading.Thread | None:
        """--with-mcp 专用：在独立线程启动，绝不阻塞 guardian 看护循环。"""
        self._transport = transport
        available, err = self.is_available()
        if not available:
            logger.warning("MCP 不可用: %s", err)
            logger.info("MCP 训练照常进行，仅外部 agent 接入不可用")
            return None

        self.__init_handlers__()
        t = threading.Thread(
            target=self._start_mcp_server,
            args=(transport,),
            kwargs={"host": host, "port": port},
            daemon=True,
            name="mcp-server",
        )
        t.start()
        return t

    def _start_mcp_server(self, transport: str, *,
                          host: str = "127.0.0.1",
                          port: int | None = None) -> str:  # pragma: no cover
        """实际启动 MCP SDK server。需要 mcp 包已安装。"""
        if not _MCP_AVAILABLE:
            return f"MCP 不可用: {_MCP_ERROR}"

        # 收集全部工具定义 → name→schema 映射（供 list_tools 注回）
        all_tool_defs: dict[str, dict[str, Any]] = {}
        for tdef in (READONLY_TOOLS + READONLY_TOOLS_V2 +
                     WRITE_TOOLS + WRITE_TOOLS_V2):
            all_tool_defs[tdef["name"]] = tdef

        # SSE/HTTP 模式需要 uvicorn（MCP SDK 内部懒加载）
        if transport in ("sse", "http", "tcp"):
            try:
                import uvicorn  # noqa: F401
            except ImportError:
                msg = "[MCP] SSE/HTTP 传输需要 uvicorn。请 pip install -r requirements-dashboard.txt"
                logger.warning(msg)
                return msg

        try:
            import asyncio

            async def _serve():
                # SDK v2：input_schema 由类型注解推断，不支持作为参数传入。
                # 因此用 MCPServer 子类接管 list_tools()，把预定义的 schema 注回。
                srv = _SchemaInjectedMCPServer("guardian", all_tool_defs)

                _register_tools_v2(srv, READONLY_TOOLS, self.call_tool)
                _register_tools_v2(srv, READONLY_TOOLS_V2, self.call_tool)
                _register_tools_v2(srv, WRITE_TOOLS, self.call_tool)
                _register_tools_v2(srv, WRITE_TOOLS_V2, self.call_tool)

                if transport in ("sse", "tcp", "http"):
                    # SSE / Streamable HTTP — 网络可访问，适合远程 agent 通过 SSH 隧道接入
                    # "tcp" 保留为旧名别名，等同 "sse"
                    _port = port or int(self.mcp_cfg.get("tcp_port", 8766))
                    _host = host or self.mcp_cfg.get("tcp_host", "127.0.0.1")
                    logger.info("MCP 监听 http://%s:%s/sse （%s 传输）", _host, _port, transport)
                    await srv.run_sse_async(host=_host, port=_port)
                else:
                    await srv.run_stdio_async()

            asyncio.run(_serve())
            return "MCP server 已退出"
        except Exception as exc:
            msg = f"[MCP] server 启动/运行失败: {exc}"
            logger.error("MCP server 异常: %s", msg, exc_info=True)
            return msg


# -----------------------------------------------------------------------
# 工具
# -----------------------------------------------------------------------

if _MCP_AVAILABLE:

    class _SchemaInjectedMCPServer(MCPServer):  # type: ignore[misc, valid-type]
        """MCPServer 子类：list_tools() 时把预定义的 input_schema 注回工具对象。

        MCP SDK v2 中 @server.tool() 不接受 input_schema 参数，schema 由
        函数类型注解自动推断。但 guardian 的工具 schema 以字典形式定义，
        因此通过此类覆盖 list_tools() 来还原显式 schema。
        """

        def __init__(self, name: str, tool_schemas: dict[str, dict[str, Any]]):
            super().__init__(name)
            self._guardian_schemas = tool_schemas

        async def list_tools(self) -> list:  # 返回 list[MCPTool]
            tools = await super().list_tools()
            for t in tools:
                schema = self._guardian_schemas.get(t.name)
                if schema:
                    input_schema = schema.get("inputSchema")
                    if input_schema:
                        # MCPTool 是 Pydantic model，可直接赋值
                        t.input_schema = input_schema
            return tools


    def _register_tools_v2(
        srv: MCPServer,
        tool_defs: list[dict[str, Any]],
        call_tool_fn,
    ) -> None:
        """在给定 MCPServer 上批量注册工具（SDK v2 兼容）。

        SDK v2 的 @server.tool() 不接受 input_schema，这里只传 name /
        description，schema 由 _SchemaInjectedMCPServer.list_tools() 补回。
        """
        for tdef in tool_defs:
            name: str = tdef["name"]
            desc: str = tdef.get("description", "")

            # 用闭包变量 _tname 固定当前工具名（避免 Python 循环闭包晚绑定）
            def _make_handler(_tname: str):
                async def _handler(**kwargs: Any) -> str:
                    return call_tool_fn(_tname, kwargs)
                # 保留原函数名便于调试
                _handler.__name__ = _tname
                return _handler

            handler = _make_handler(name)
            srv.add_tool(handler, name=name, description=desc)

def _safe_serialize(obj: Any) -> Any:
    """确保可 JSON 序列化，不抛异常。"""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def _mcp_run_arch_analyzer(analyzer, model_entry: str, project_dir: str) -> dict:
    """MCP handler 用：动态导入模型并执行架构分析。"""
    import importlib

    if project_dir not in sys.path:
        sys.path.insert(0, project_dir)
    mod_parts = model_entry.split(":", 1)
    if len(mod_parts) != 2:
        return {"error": f"invalid model_entry: {model_entry}"}
    mod = importlib.import_module(mod_parts[0])
    model_fn = getattr(mod, mod_parts[1])
    return analyzer.analyze(model_fn)
