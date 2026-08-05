"""cp_9 · Agent 决策封装 (AgentAdvisor)。

所有"中层 agent 化"决策点的统一出入口：调用 LLM、裁剪动作空间、强制超时
降级，确保规则引擎兜底不被破坏。advisor 跑在 guardian 进程里，不在训练
进程内——sidecar 下 decision_timeout 不占用训练时间。详见 checkpoint/cp_9.md

`agent.enabled: false` 或未配置 api_key_env 对应的环境变量时，`is_enabled()`
恒为 False，整层零成本降级为原有规则引擎行为。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

from .config import resolve_secret


class AgentAdvisor:
    """decide() / narrate() / suggest() 统一入口。"""

    def __init__(self, config: dict | None = None):
        self.cfg = config or {}
        self.enabled_cfg = bool(self.cfg.get("enabled", False))
        self.provider = self.cfg.get("provider", "anthropic")
        self.model = self.cfg.get("model")
        self.api_key_env = self.cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        self.decision_timeout = float(self.cfg.get("decision_timeout", 8))
        self.failure_threshold = int(self.cfg.get("consecutive_failure_threshold", 5))
        self.cooldown = float(self.cfg.get("circuit_breaker_cooldown", 600))
        self.decision_points = dict(self.cfg.get("decision_points") or {})

        self._consecutive_failures = 0
        self._breaker_until: float | None = None  # time.monotonic() 时间戳
        self.decision_log: list[dict[str, Any]] = []
        # 每次 decide()/narrate() 各自起一个线程等待自己的 timeout，互不阻塞
        # 彼此的调用方（完整校验表"并发调用安全"）。
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="advisor")

    # --- 开关状态 -----------------------------------------------------

    def is_enabled(self, decision_point: str | None = None) -> bool:
        """配置检查 + 熔断状态 + 单点开关。"""
        if not self.enabled_cfg:
            return False
        if not resolve_secret(self.cfg, "api_key_env"):
            return False
        if decision_point is not None and self.decision_points.get(decision_point, True) is False:
            return False
        if self._breaker_until is not None:
            if time.monotonic() < self._breaker_until:
                return False
            # cooldown 结束，自动恢复尝试
            self._breaker_until = None
            self._consecutive_failures = 0
        return True

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._breaker_until = time.monotonic() + self.cooldown

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._breaker_until = None

    # --- 主入口 ---------------------------------------------------------

    def decide(
        self,
        decision_point: str,
        context: dict[str, Any],
        action_space: list[Any],
        default_action: Any,
    ) -> dict[str, Any]:
        """所有决策点的统一入口，见 cp_9.md 调用契约。

        永远返回一个合法动作：
          {"action": ..., "source": "agent"|"rule_default"|"disabled"|
                                     "timeout"|"error"|"invalid_output",
           "latency_ms": float}
        """
        if not self.is_enabled(decision_point):
            return self._log_decision(decision_point, context, default_action,
                                       "disabled", 0.0)

        start = time.monotonic()
        prompt = self._build_prompt(decision_point, context, action_space)
        try:
            future = self._executor.submit(self._call_llm, prompt, self.decision_timeout)
            raw_output = future.result(timeout=self.decision_timeout)
        except FutureTimeoutError:
            self._record_failure()
            latency_ms = (time.monotonic() - start) * 1000
            return self._log_decision(decision_point, context, default_action,
                                       "timeout", latency_ms)
        except Exception:
            self._record_failure()
            latency_ms = (time.monotonic() - start) * 1000
            return self._log_decision(decision_point, context, default_action,
                                       "error", latency_ms)

        latency_ms = (time.monotonic() - start) * 1000
        chosen = self._validate_action(raw_output, action_space)
        if chosen is None:
            self._record_failure()
            return self._log_decision(decision_point, context, default_action,
                                       "invalid_output", latency_ms)
        self._record_success()
        return self._log_decision(decision_point, context, chosen, "agent", latency_ms)

    def narrate(self, structured_data: dict[str, Any], prompt_template: str | None = None) -> str | None:
        """纯文本生成：summary 的自然语言解读，无动作约束，失败返回 None。"""
        if not self.is_enabled("summary_narrative"):
            return None
        try:
            future = self._executor.submit(
                self._call_llm_text, structured_data, prompt_template, self.decision_timeout,
            )
            text = future.result(timeout=self.decision_timeout)
        except Exception:
            self._record_failure()
            return None
        self._record_success()
        return text

    def suggest(
        self,
        kind: str,
        context: dict[str, Any],
        registry_snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """只读式建议生成：cp_11 propose_registry_entry 用。

        不生效、不走超时降级语义（找不到就是找不到，不影响当次训练）。
        失败返回 None。
        """
        if not self.enabled_cfg or not resolve_secret(self.cfg, "api_key_env"):
            return None
        try:
            return self._call_llm_suggest(kind, context, registry_snapshot)
        except Exception:
            return None

    # --- 内部实现 ---------------------------------------------------------

    def _build_prompt(
        self, decision_point: str, context: dict[str, Any], action_space: list[Any],
    ) -> dict[str, Any]:
        """拼接结构化上下文和动作集为 LLM 可理解的 prompt。"""
        return {
            "decision_point": decision_point,
            "context": context,
            "action_space": action_space,
            "model": self.model,
        }

    def _call_llm(self, prompt: dict[str, Any], timeout: float) -> Any:
        """实际网络调用，测试中总是被 mock 掉。

        真实实现按 provider 分发；未接入具体 SDK 时抛错触发降级，
        而不是静默返回假数据。
        """
        raise NotImplementedError(
            f"provider={self.provider} 的真实 LLM 调用尚未接入，"
            "请在测试中 mock AgentAdvisor._call_llm"
        )

    def _call_llm_text(
        self, structured_data: dict[str, Any], prompt_template: str | None, timeout: float,
    ) -> str:
        """narrate() 的真实网络调用，测试中总是被 mock 掉。"""
        raise NotImplementedError(
            f"provider={self.provider} 的真实 LLM 调用尚未接入，"
            "请在测试中 mock AgentAdvisor._call_llm_text"
        )

    def _call_llm_suggest(
        self, kind: str, context: dict[str, Any], registry_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """suggest() 的真实网络调用，测试中总是被 mock 掉。"""
        raise NotImplementedError(
            f"provider={self.provider} 的真实 LLM 调用尚未接入，"
            "请在测试中 mock AgentAdvisor._call_llm_suggest"
        )

    @staticmethod
    def _validate_action(raw_output: Any, action_space: list[Any]) -> Any | None:
        """校验 LLM 返回是否为动作集内的合法动作（含参数范围检查）。

        action_space 里的元素可以是：
          - 字符串（如 "ignore"）：raw_output 必须逐字相等
          - dict {"action": name, "min"/"max"/...: 范围}：raw_output 需是同名
            dict，且数值参数落在声明的范围内
        raw_output 形状：str，或 {"action": name, **params}
        """
        if isinstance(raw_output, str):
            name, params = raw_output, {}
        elif isinstance(raw_output, dict) and "action" in raw_output:
            name = raw_output["action"]
            params = {k: v for k, v in raw_output.items() if k != "action"}
        else:
            return None

        for candidate in action_space:
            if isinstance(candidate, str):
                if candidate == name and not params:
                    return name
                continue
            if isinstance(candidate, dict) and candidate.get("action") == name:
                if AgentAdvisor._params_in_range(params, candidate):
                    return {"action": name, **params} if params else name
                return None
        return None

    @staticmethod
    def _params_in_range(params: dict[str, Any], candidate: dict[str, Any]) -> bool:
        """逐个参数检查是否落在 candidate 声明的 min/max 范围内。

        candidate 形如 {"action": "reduce_batch", "ratio": {"min": 0.1, "max": 0.9}}。
        candidate 里没声明范围的参数键，只要求 params 不带未声明的键。
        """
        allowed_keys = {k for k in candidate.keys() if k != "action"}
        if set(params.keys()) - allowed_keys:
            return False
        for key, value in params.items():
            bounds = candidate.get(key)
            if not isinstance(bounds, dict):
                continue
            if "min" in bounds and value < bounds["min"]:
                return False
            if "max" in bounds and value > bounds["max"]:
                return False
        return True

    def _log_decision(
        self,
        decision_point: str,
        context: dict[str, Any],
        chosen_action: Any,
        source: str,
        latency_ms: float,
    ) -> dict[str, Any]:
        """记录每次决策：来源、耗时、上下文摘要。供 summary / MCP 工具查询。"""
        entry = {
            "decision_point": decision_point,
            "action": chosen_action,
            "source": source,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
            "context_summary": _summarize_context(context),
        }
        self.decision_log.append(entry)
        return entry

    def close(self) -> None:
        """释放线程池，测试/主流程结束时可选调用。"""
        self._executor.shutdown(wait=False, cancel_futures=True)


def _summarize_context(context: dict[str, Any], max_len: int = 200) -> str:
    """把上下文压成一行摘要，避免决策日志无限膨胀。"""
    try:
        text = ", ".join(f"{k}={v}" for k, v in context.items())
    except Exception:
        text = str(context)
    return text[:max_len]
