"""SubAgent 主类：guardian 内置的自主智能体。

不依赖 Claude Code 或任何外部 agent 平台。通过 LLM 驱动决策循环，
拥有记忆 (RollingMemory) 和工具调用能力 (ToolRegistry)。

生命周期：
    spawn()  → 初始化 LLM + tools + memory
    on_tick() → 每个训练 tick 调用，返回待执行动作列表
    approve() → PC 端审批通过后执行动作
    reject()  → PC 端驳回后记录并调整策略
    shutdown() → 训练结束，生成总结
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from guardian.logging_config import get_logger
from guardian.sub_agent.memory import RollingMemory
from guardian.sub_agent.prompts import (
    SYSTEM_ANOMALY_RESPONSE,
    SYSTEM_CRASH_RECOVERY,
    SYSTEM_HEALTH_CHECK,
    SYSTEM_TRAINING_SUMMARY,
    build_anomaly_prompt,
    build_crash_prompt,
    build_health_prompt,
    build_summary_prompt,
)
from guardian.sub_agent.tool_registry import Action, ActionResult, ToolRegistry

logger = get_logger(__name__)


class SubAgent:
    """guardian 内置自主智能体。"""

    def __init__(
        self,
        config: dict | None = None,
        tool_registry: ToolRegistry | None = None,
        llm_callback: callable | None = None,
    ):
        """
        Args:
            config: 配置字典，支持 keys:
                - autonomy: "supervised" | "auto" | "full"
                - decision_timeout: LLM 调用超时（秒）
                - memory_window: 记忆窗口大小
                - circuit_breaker_threshold: 连续失败熔断阈值
                - circuit_breaker_cooldown: 熔断冷却时间（秒）
            tool_registry: 工具注册表（外部注入实际调用函数）
            llm_callback: LLM 调用函数 (system_prompt, user_message, timeout) -> str
                          外部注入，便于测试和替换 LLM 后端
        """
        self.cfg = config or {}
        self.autonomy = self.cfg.get("autonomy", "supervised")
        self.decision_timeout = float(self.cfg.get("decision_timeout", 8))
        self.memory_window = int(self.cfg.get("memory_window", 100))

        self.tools = tool_registry or ToolRegistry()
        self.memory = RollingMemory(max_size=self.memory_window)
        self._llm = llm_callback or self._default_llm_call

        # 熔断器
        self._consecutive_failures = 0
        self._breaker_until: float | None = None
        self._failure_threshold = int(self.cfg.get("circuit_breaker_threshold", 5))
        self._breaker_cooldown = float(self.cfg.get("circuit_breaker_cooldown", 300))

        # 线程池（LLM 调用非阻塞）
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sub_agent")

        # 待审批队列
        self._pending_actions: list[dict] = []
        self._max_pending = int(self.cfg.get("max_pending_actions", 50))

        # 状态
        self._spawned = False
        self._shutdown = False

    # ── 生命周期 ─────────────────────────────────────────────────────

    def spawn(self, training_context: dict) -> dict:
        """训练开始时调用，初始化 sub-agent。

        Args:
            training_context: 训练上下文，包含:
                - command: 训练命令
                - total_epochs: 总 epoch 数
                - model_entry: 模型入口
                - project_dir: 项目目录
                - log_file: 日志文件路径
        """
        self._spawned = True
        self._shutdown = False
        self.memory = RollingMemory(max_size=self.memory_window)

        # 注入工具注册表的实际调用函数
        self._inject_tool_handlers(training_context)

        self.memory.record_decision(
            event_type="spawn",
            description=f"Sub-agent spawned for training: {training_context.get('command', 'unknown')}",
            source="system",
        )
        logger.info(
            "Sub-agent spawned: autonomy=%s, tools=%d",
            self.autonomy, len(self.tools.tool_names),
        )
        return {"status": "spawned", "autonomy": self.autonomy}

    def on_tick(self, metrics: dict, gpu_stats: dict | None = None) -> list[dict]:
        """每个训练 tick 调用。

        Args:
            metrics: 当前训练指标 {epoch, step, loss, val_acc, lr, ...}
            gpu_stats: GPU 状态 [{gpu_id, utilization, temperature, memory_used, ...}]

        Returns:
            待执行动作列表，每个元素:
            {
                "action": Action 对象,
                "needs_approval": bool,
                "priority": "high" | "normal" | "low",
            }
        """
        if not self._spawned or self._shutdown:
            return []

        # 更新记忆中的训练进度
        self.memory.update_progress(
            current_epoch=metrics.get("epoch", 0),
            total_epochs=metrics.get("total_epochs", 0),
            current_metric_value=metrics.get("val_acc") or metrics.get("loss"),
            metric_name="val_acc" if metrics.get("val_acc") else "loss",
        )

        # 检查是否有异常（外部检测 + sub-agent 自主判断）
        actions = []
        anomalies = self._detect_anomalies(metrics, gpu_stats)
        for anomaly in anomalies:
            action = self._decide_anomaly_response(anomaly, metrics, gpu_stats)
            if action:
                needs_approval = self.tools.requires_approval(action.tool_name, self.autonomy)
                actions.append({
                    "action": action,
                    "needs_approval": needs_approval,
                    "priority": "high" if anomaly.get("severity") == "critical" else "normal",
                })

        # 定期健康检查（每 50 个 step）
        if metrics.get("step", 0) % 50 == 0:
            health = self._check_health(metrics, gpu_stats)
            if health and not health.get("healthy", True):
                logger.info("Sub-agent health check: concerns=%s", health.get("concerns", []))

        return actions

    def approve(self, action_id: str) -> ActionResult:
        """PC 端审批通过后执行动作。"""
        for pending in self._pending_actions:
            if pending.get("action_id") == action_id:
                action = pending["action"]
                result = self.tools.execute(action, self.autonomy)
                self._pending_actions = [p for p in self._pending_actions if p.get("action_id") != action_id]
                self.memory.record_decision(
                    event_type="intervention",
                    description=f"PC approved: {action.tool_name}",
                    action_taken=action.tool_name,
                    action_params=action.params,
                    source="pc_approved",
                    outcome="success" if result.success else "failed",
                )
                logger.info("Sub-agent action approved & executed: %s → %s", action_id, result.success)
                return result

        # 尝试从 pending_decisions 中查找（由外部 populate）
        return ActionResult(
            action_id=action_id,
            tool_name="unknown",
            success=False,
            error=f"Action {action_id} not found in pending queue",
        )

    def reject(self, action_id: str, reason: str = "") -> ActionResult:
        """PC 端驳回动作。"""
        for pending in self._pending_actions:
            if pending.get("action_id") == action_id:
                action = pending["action"]
                self._pending_actions = [p for p in self._pending_actions if p.get("action_id") != action_id]
                self.memory.record_decision(
                    event_type="intervention",
                    description=f"PC rejected: {action.tool_name} — {reason}",
                    action_taken=action.tool_name,
                    source="pc_rejected",
                    outcome="rejected",
                )
                logger.info("Sub-agent action rejected: %s — %s", action_id, reason)
                return ActionResult(
                    action_id=action_id,
                    tool_name=action.tool_name,
                    success=False,
                    rejected=True,
                    rejection_reason=reason,
                )
        return ActionResult(
            action_id=action_id,
            tool_name="unknown",
            success=False,
            error=f"Action {action_id} not found in pending queue",
        )

    def shutdown(self, final_metrics: dict | None = None) -> dict:
        """训练结束时调用，生成总结并清理。"""
        self._shutdown = True
        self.memory.mark_finished()

        # 生成 AI 总结
        summary_data = self.memory.get_summary()
        if final_metrics:
            summary_data["final_metrics"] = final_metrics

        narrative = self._generate_summary(summary_data)

        result = {
            "status": "shutdown",
            "narrative": narrative,
            "memory_summary": summary_data,
            "decision_log": [
                {
                    "timestamp": r.timestamp,
                    "type": r.event_type,
                    "description": r.description,
                    "action": r.action_taken,
                    "source": r.source,
                    "outcome": r.outcome,
                }
                for r in self.memory.get_recent(50)
            ],
            "stats": {
                "total_decisions": len(self.memory),
                "anomalies_detected": self.memory._anomaly_count,
                "interventions": self.memory._intervention_count,
                "crashes": self.memory._crash_count,
            },
        }
        logger.info("Sub-agent shutdown: %d decisions recorded", len(self.memory))
        return result

    # ── 内部：决策逻辑 ────────────────────────────────────────────────

    def _decide_anomaly_response(self, anomaly: dict, metrics: dict,
                                  gpu_stats: dict | None) -> Action | None:
        """对单个异常做 LLM 决策。"""
        if self._is_breaker_open():
            logger.debug("Circuit breaker open, skipping LLM decision")
            return Action(tool_name="alert", params={"level": "warning"},
                          reason="LLM 熔断中，降级为告警", confidence=0.5)

        context = {
            "decision_point": "monitor_response",
            "anomaly_type": anomaly.get("type", "unknown"),
            "description": anomaly.get("description", ""),
            "severity": anomaly.get("severity", "medium"),
            "current_metrics": metrics,
            "training_phase": self.memory.phase,
            "anomaly_count": self.memory._anomaly_count,
        }
        memory_ctx = self.memory.get_context_for_llm()
        action_space = self._build_action_space(anomaly.get("type"))

        user_message = build_anomaly_prompt(context, memory_ctx, action_space)

        try:
            raw = self._call_llm(SYSTEM_ANOMALY_RESPONSE, user_message, self.decision_timeout)
            action = self._parse_action(raw, action_space)
            if action:
                action.context = {"anomaly": anomaly, "metrics": metrics}
                self.memory.record_decision(
                    event_type="anomaly",
                    description=anomaly.get("description", ""),
                    action_taken=action.tool_name,
                    action_params=action.params,
                    source="sub_agent",
                    confidence=action.confidence,
                    anomaly_type=anomaly.get("type"),
                )
                self._consecutive_failures = 0
                return action
        except Exception as exc:
            logger.warning("Sub-agent LLM 决策失败: %s", exc)
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._failure_threshold:
                self._breaker_until = time.time() + self._breaker_cooldown
                logger.warning("Sub-agent circuit breaker triggered")

        # 降级：规则默认
        fallback = self._rule_default_action(anomaly)
        if fallback:
            self.memory.record_decision(
                event_type="anomaly",
                description=anomaly.get("description", ""),
                action_taken=fallback.tool_name,
                source="rule_default",
                outcome="pending",
            )
        return fallback

    def _decide_crash_recovery(self, crash_info: dict) -> Action | None:
        """LLM 决策：崩溃恢复策略。"""
        if self._is_breaker_open():
            return Action(tool_name="resume_unchanged",
                          reason="LLM 熔断中，降级为原样续训", confidence=0.5)

        context = crash_info
        memory_ctx = self.memory.get_context_for_llm()
        action_space = self._build_crash_action_space(crash_info.get("crash_type", "unknown"))

        user_message = build_crash_prompt(context, memory_ctx, action_space)

        try:
            raw = self._call_llm(SYSTEM_CRASH_RECOVERY, user_message, self.decision_timeout)
            action = self._parse_action(raw, action_space)
            if action:
                action.context = {"crash": crash_info}
                self.memory.record_decision(
                    event_type="crash",
                    description=f"Crash recovery: {crash_info.get('crash_type', 'unknown')}",
                    action_taken=action.tool_name,
                    action_params=action.params,
                    source="sub_agent",
                )
                self._consecutive_failures = 0
                return action
        except Exception as exc:
            logger.warning("Sub-agent crash recovery 决策失败: %s", exc)
            self._consecutive_failures += 1

        return Action(tool_name="resume_unchanged", reason="LLM 失败，默认原样续训",
                      confidence=0.3)

    def _check_health(self, metrics: dict, gpu_stats: dict | None) -> dict | None:
        """定期健康检查（LLM 评估，非阻塞）。"""
        if not metrics or self._is_breaker_open():
            return None

        try:
            future = self._executor.submit(
                self._call_llm,
                SYSTEM_HEALTH_CHECK,
                build_health_prompt(metrics, gpu_stats or {}),
                5.0,
            )
            raw = future.result(timeout=6.0)
            result = self._parse_json(raw)
            if isinstance(result, dict):
                return result
        except Exception:
            pass
        return None

    def _generate_summary(self, summary_data: dict) -> str | None:
        """生成训练总结（AI 解读）。"""
        try:
            future = self._executor.submit(
                self._call_llm,
                SYSTEM_TRAINING_SUMMARY,
                build_summary_prompt(summary_data),
                self.decision_timeout * 3,
            )
            return future.result(timeout=self.decision_timeout * 3 + 2)
        except Exception as exc:
            logger.warning("Sub-agent 总结生成失败: %s", exc)
            return None

    # ── 内部：异常检测辅助 ────────────────────────────────────────────

    def _detect_anomalies(self, metrics: dict, gpu_stats: dict | None) -> list[dict]:
        """简单的规则化异常检测（辅助 LLM 决策，非主要检测手段）。"""
        anomalies = []
        loss = metrics.get("loss")
        if loss is not None and self.memory._best_metric_value is not None:
            best_loss = self.memory._best_metric_value if self.memory._best_metric_name == "loss" else None
            if best_loss and loss > best_loss * 2.0:
                anomalies.append({
                    "type": "loss_spike",
                    "description": f"Loss 突增: current={loss:.4f}, best={best_loss:.4f}",
                    "severity": "high",
                })

        if gpu_stats:
            for gpu in gpu_stats:
                temp = gpu.get("temperature", 0)
                if temp > 90:
                    anomalies.append({
                        "type": "gpu_overheat",
                        "description": f"GPU{gpu.get('gpu_id', '?')} 温度过高: {temp}°C",
                        "severity": "critical",
                    })
                elif temp > 85:
                    anomalies.append({
                        "type": "gpu_hot",
                        "description": f"GPU{gpu.get('gpu_id', '?')} 温度偏高: {temp}°C",
                        "severity": "medium",
                    })
                util = gpu.get("utilization", 100)
                if util < 10:
                    anomalies.append({
                        "type": "gpu_idle",
                        "description": f"GPU{gpu.get('gpu_id', '?')} 利用率过低: {util}%",
                        "severity": "low",
                    })

        return anomalies

    # ── 内部：LLM 调用 ────────────────────────────────────────────────

    def _call_llm(self, system_prompt: str, user_message: str, timeout: float) -> str:
        """调用 LLM。默认使用 OpenAI 兼容接口（可通过 config 切换）。"""
        # 优先使用外部注入的回调
        if self._llm is not None:
            return self._llm(system_prompt, user_message, timeout)

        # 内置 fallback：尝试 OpenAI SDK
        try:
            import openai
            client = openai.OpenAI(
                api_key="not-needed",  # 使用自定义 base_url 时可为空
                base_url=self.cfg.get("llm_base_url", "http://localhost:8000/v1"),
            )
            resp = client.chat.completions.create(
                model=self.cfg.get("llm_model", "deepseek-v4-pro"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=256,
                timeout=min(timeout, 60),
            )
            return resp.choices[0].message.content or ""
        except ImportError:
            raise RuntimeError("未配置 LLM 回调，且 openai SDK 未安装")

    @staticmethod
    def _default_llm_call(system_prompt: str, user_message: str, timeout: float) -> str:
        """默认 LLM 调用实现（需要外部通过 config 注入）。"""
        raise RuntimeError(
            "SubAgent 需要 LLM 回调。请通过 config['llm_callback'] 注入，"
            "或在 spawn() 时设置环境变量 ANTHROPIC_API_KEY / OPENAI_API_KEY。"
        )

    # ── 内部：工具注入 ────────────────────────────────────────────────

    def _inject_tool_handlers(self, training_context: dict) -> None:
        """将训练上下文中的实际调用函数注入工具注册表。"""
        # 这些函数由 watch 主循环在 spawn 时注入
        handlers = training_context.get("tool_handlers", {})
        for tool_name, handler_fn in handlers.items():
            spec = self.tools.get(tool_name)
            if spec:
                spec.fn = handler_fn

    def set_tool_handler(self, tool_name: str, handler: callable) -> None:
        """运行时注入/替换某个工具的实际调用函数。"""
        spec = self.tools.get(tool_name)
        if spec:
            spec.fn = handler

    # ── 内部：解析辅助 ────────────────────────────────────────────────

    def _parse_action(self, raw: Any, action_space: list) -> Action | None:
        """解析 LLM 返回为 Action 对象。"""
        if isinstance(raw, str):
            text = raw.strip().strip("\"'")
            for candidate in action_space:
                if isinstance(candidate, str) and candidate == text:
                    return Action(tool_name=text, confidence=1.0)
                elif isinstance(candidate, dict) and candidate.get("action") == text:
                    return Action(tool_name=text, confidence=0.9)
            return None

        if isinstance(raw, dict) and "action" in raw:
            name = raw["action"]
            params = {k: v for k, v in raw.items() if k != "action"}
            for candidate in action_space:
                if isinstance(candidate, dict) and candidate.get("action") == name:
                    # 验证参数范围
                    if self._params_valid(params, candidate):
                        return Action(tool_name=name, params=params, confidence=0.9)
            # 即使不在 action_space 中也尝试返回（由外部验证）
            return Action(tool_name=name, params=params, confidence=0.7)

        return None

    @staticmethod
    def _params_valid(params: dict, candidate: dict) -> bool:
        """检查参数是否在合法范围内。"""
        allowed = {k for k in candidate if k != "action"}
        if set(params.keys()) - allowed:
            return False
        for key, value in params.items():
            bounds = candidate.get(key)
            if isinstance(bounds, dict):
                if "min" in bounds and value < bounds["min"]:
                    return False
                if "max" in bounds and value > bounds["max"]:
                    return False
        return True

    @staticmethod
    def _parse_json(text: str) -> Any:
        """从文本中提取 JSON。"""
        import json as _json
        text = (text or "").strip()
        try:
            return _json.loads(text)
        except Exception:
            pass
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return _json.loads(text[start:end + 1])
            except Exception:
                pass
        return None

    # ── 内部：Action Space 构建 ───────────────────────────────────────

    def _build_action_space(self, anomaly_type: str) -> list:
        """根据异常类型构建可用动作集。"""
        base = [
            "ignore",
            "alert",
        ]
        if anomaly_type in ("loss_spike", "nan_inf", "loss_stagnation"):
            base.extend([
                {"action": "restart_with_lower_lr", "ratio": {"min": 0.1, "max": 1.0}},
                {"action": "reduce_batch", "ratio": {"min": 0.25, "max": 1.0}},
                {"action": "enable_grad_accum", "steps": {"min": 1, "max": 32}},
            ])
        if anomaly_type in ("gpu_overheat", "gpu_hot"):
            base.append("alert")
        return base

    def _build_crash_action_space(self, crash_type: str) -> list:
        """根据崩溃类型构建恢复动作集。"""
        if crash_type == "OOM":
            return [
                "resume_unchanged",
                {"action": "reduce_batch", "ratio": {"min": 0.25, "max": 0.75}},
                {"action": "enable_grad_accum", "steps": {"min": 2, "max": 32}},
            ]
        if crash_type in ("sigkill", "network", "unknown"):
            return ["resume_unchanged", "stop_training"]
        return ["resume_unchanged", "stop_training"]

    @staticmethod
    def _rule_default_action(anomaly: dict) -> Action | None:
        """规则默认动作（LLM 不可用时的 fallback）。"""
        atype = anomaly.get("type", "")
        if atype in ("loss_spike", "loss_stagnation", "nan_inf"):
            return Action(tool_name="alert", params={"level": "warning"},
                          reason="规则默认：告警", confidence=0.5)
        if atype in ("gpu_overheat", "gpu_hot"):
            return Action(tool_name="alert", params={"level": "error"},
                          reason="规则默认：高温告警", confidence=0.8)
        return Action(tool_name="ignore", reason="规则默认：忽略", confidence=0.5)

    # ── 内部：熔断器 ──────────────────────────────────────────────────

    def _is_breaker_open(self) -> bool:
        """检查熔断器是否打开。"""
        if self._breaker_until is not None:
            if time.monotonic() < self._breaker_until:
                return True
            # 冷却结束，自动恢复
            self._breaker_until = None
            self._consecutive_failures = 0
        return False

    # ── 状态查询 ──────────────────────────────────────────────────────

    @property
    def is_spawned(self) -> bool:
        return self._spawned and not self._shutdown

    @property
    def pending_count(self) -> int:
        return len(self._pending_actions)

    def get_pending_actions(self) -> list[dict]:
        """获取待审批动作列表（供 PC 端查询）。"""
        return [
            {
                "action_id": p.get("action_id"),
                "tool_name": p["action"].tool_name,
                "params": p["action"].params,
                "reason": p["action"].reason,
                "confidence": p["action"].confidence,
                "priority": p.get("priority", "normal"),
                "created_at": p.get("created_at", time.time()),
            }
            for p in self._pending_actions
            if p.get("action_id") and p.get("status", "pending") == "pending"
        ]

    def __repr__(self) -> str:
        status = "spawned" if self._spawned else "not_spawned"
        if self._shutdown:
            status = "shutdown"
        return f"SubAgent(status={status}, autonomy={self.autonomy}, memory={len(self.memory)})"
