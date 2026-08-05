"""cp_10 · MCP 工具层 (GuardianMCPServer)。

把 guardian 的观测与操作能力暴露为标准 MCP 工具，供 Claude Code / OpenClaw
等外部 agent 客户端接入。详见 checkpoint/cp_10.md

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

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# MCP SDK 可选导入
# ---------------------------------------------------------------------------

_MCP_AVAILABLE = False
_MCP_ERROR: str | None = None

try:
    import mcp  # noqa: F401
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    _MCP_AVAILABLE = True
except ImportError as exc:
    _MCP_ERROR = f"mcp 包未安装（{exc}）。MCP 能力不可用，其余功能照常。"
    Server = object  # type: ignore
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
                "action": {"type": "string", "description": "reduce_batch / restart_with_lower_lr / enable_grad_accum"},
                "param": {"description": "动作参数：ratio / steps 等"},
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
        "name": "trigger_full_validate",
        "description": (
            "手动触发指定 checkpoint 的完整校验。【风险】占用算力，"
            "可能与训练争抢 GPU。重复调用只是重算，结果相同。"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "epoch": {"type": "integer", "description": "要校验的 checkpoint epoch"},
                "request_id": {"type": "string", "description": "幂等键"},
            },
            "required": ["epoch"],
        },
        "annotations": {"readOnlyHint": False, "destructiveHint": False,
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
        if not self.write_enabled:
            return False, f"写工具 {tool_name!r} 未启用（enable_write_tools=false）"
        if not self.write_token:
            return False, "未配置 write_token_env 环境变量"
        if token != self.write_token:
            return False, "鉴权失败：token 不匹配"
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

    # ------------------------------------------------------------------
    # 工具处理器 —— 只读
    # ------------------------------------------------------------------

    def _handle_training_status(self, **kwargs) -> str:
        if self.monitor is None:
            return json.dumps({"error": "monitor 未绑定"}, ensure_ascii=False)
        hist = self.monitor.get_metrics_history()
        latest = hist[-1] if hist else {}
        gpu_hist = getattr(self.monitor, "get_gpu_history", lambda: [])()
        latest_gpu = gpu_hist[-1] if gpu_hist else {}
        return json.dumps({
            "latest_metrics": latest,
            "total_records": len(hist),
            "latest_gpu": latest_gpu,
            "anomaly_count": len(self.monitor.get_anomaly_history()),
        }, ensure_ascii=False, indent=2)

    def _handle_metrics_history(self, limit: int = 200, cursor: int = 0, **kwargs) -> str:
        if self.monitor is None:
            return json.dumps({"error": "monitor 未绑定"}, ensure_ascii=False)
        hist = self.monitor.get_metrics_history()
        total = len(hist)
        start = max(0, total - limit - cursor)
        end = total - cursor if cursor else total
        window = hist[max(0, start):end]
        # 聚合统计
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
        }, ensure_ascii=False, indent=2)

    def _handle_list_checkpoints(self, metric: str = "val/accuracy", **kwargs) -> str:
        if self.ckpt_analyzer is None:
            return json.dumps({"error": "checkpoint analyzer 未绑定"}, ensure_ascii=False)
        report = self.ckpt_analyzer.report(metric)
        return json.dumps(report, ensure_ascii=False, indent=2)

    def _handle_compare_checkpoints(self, epoch_a: int, epoch_b: int, **kwargs) -> str:
        if self.ckpt_analyzer is None:
            return json.dumps({"error": "checkpoint analyzer 未绑定"}, ensure_ascii=False)
        info_a = self.ckpt_analyzer.known.get(epoch_a)
        info_b = self.ckpt_analyzer.known.get(epoch_b)
        if not info_a or not info_b:
            return json.dumps({"error": "一个或两个 epoch 不存在"}, ensure_ascii=False)
        a = info_a.metrics
        b = info_b.metrics
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
        if self.monitor is None:
            return json.dumps([], ensure_ascii=False)
        return json.dumps(self.monitor.get_anomaly_history(), ensure_ascii=False, indent=2)

    def _handle_recovery_history(self, **kwargs) -> str:
        if self.watchdog is None:
            return json.dumps([], ensure_ascii=False)
        return json.dumps([r.to_dict() for r in self.watchdog.restarts],
                          ensure_ascii=False, indent=2)

    def _handle_summary(self, **kwargs) -> str:
        if self.summary_gen is None:
            return json.dumps({"error": "summary generator 未绑定"}, ensure_ascii=False)
        return json.dumps(self.summary_gen.generate(), ensure_ascii=False, indent=2)

    def _handle_agent_decision_log(self, **kwargs) -> str:
        if self.advisor is None:
            return json.dumps([], ensure_ascii=False)
        return json.dumps(list(self.advisor.decision_log), ensure_ascii=False, indent=2)

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
    # 工具路由
    # ------------------------------------------------------------------

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
        }
        self._WRITE_HANDLERS = {
            "trigger_recovery": self._handle_trigger_recovery,
            "restart_with_params": self._handle_restart_with_params,
            "stop_training": self._handle_stop_training,
            "trigger_full_validate": self._handle_trigger_full_validate,
            "approve_contract_proposal": self._handle_approve_proposal,
            "reject_contract_proposal": self._handle_reject_proposal,
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

    def start(self, transport: str = "stdio") -> str:
        """启动 MCP server（阻塞）。"""
        available, err = self.is_available()
        if not available:
            return f"[MCP] {err}\n[MCP] 请 pip install -r requirements-mcp.txt 后重试。"

        self.__init_handlers__()
        return self._start_mcp_server(transport)

    def start_in_background(self, transport: str = "stdio") -> threading.Thread | None:
        """--with-mcp 专用：在独立线程启动，绝不阻塞 guardian 看护循环。"""
        available, err = self.is_available()
        if not available:
            print(f"[MCP] {err}", flush=True)
            print("[MCP] 训练照常进行，仅外部 agent 接入不可用。", flush=True)
            return None

        self.__init_handlers__()
        t = threading.Thread(
            target=self._start_mcp_server,
            args=(transport,),
            daemon=True,
            name="mcp-server",
        )
        t.start()
        return t

    def _start_mcp_server(self, transport: str) -> str:  # pragma: no cover
        """实际启动 MCP SDK server。需要 mcp 包已安装。"""
        if not _MCP_AVAILABLE:
            return f"MCP 不可用: {_MCP_ERROR}"

        # 构建工具注册表
        def _make_tool_handler(name: str):
            def handler(**arguments):
                return self.call_tool(name, arguments)
            return handler

        try:
            # 使用 mcp SDK 的标准模式注册工具
            import mcp
            import asyncio

            async def _serve():
                server = Server("guardian")

                # 注册只读工具
                for tdef in READONLY_TOOLS:
                    name = tdef["name"]
                    @server.tool(
                        name=name,
                        description=tdef["description"],
                        input_schema=tdef.get("inputSchema"),
                    )
                    async def _ro_tool(**kwargs, _name=name):
                        result = self.call_tool(_name, kwargs)
                        return result

                # 注册写工具（即使未启用也注册，调用时才拒绝）
                for tdef in WRITE_TOOLS:
                    name = tdef["name"]
                    @server.tool(
                        name=name,
                        description=tdef["description"],
                        input_schema=tdef.get("inputSchema"),
                    )
                    async def _rw_tool(**kwargs, _name=name):
                        result = self.call_tool(_name, kwargs)
                        return result

                if transport == "stdio":
                    async with stdio_server() as (read_stream, write_stream):
                        await server.run(
                            read_stream, write_stream,
                            server.create_initialization_options(),
                        )

            asyncio.run(_serve())
            return "MCP server 已退出"
        except Exception as exc:
            msg = f"[MCP] server 启动/运行失败: {exc}"
            print(msg, flush=True)
            return msg


# -----------------------------------------------------------------------
# 工具
# -----------------------------------------------------------------------

def _safe_serialize(obj: Any) -> Any:
    """确保可 JSON 序列化，不抛异常。"""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)
