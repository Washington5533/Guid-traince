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

import json
import os
import sys
from importlib import import_module

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

    # -- prompt 构建器

    SYSTEM_DECIDE = (
        "你是一个训练守护 agent，负责在预设的有限动作集里做出操作决策。"
        "你收到的上下文来自训练监控系统（cp_2 monitor），异常检测本身已由"
        "规则引擎确认——你只需要选择'怎么应对'。\n\n"
        "规则：\n"
        "1. 你必须从提供的 action_space 中选择一个动作，不能自己发明\n"
        "2. 如果动作带有参数（如 ratio），参数必须在声明的范围内\n"
        "3. 返回 JSON 格式：{\"action\": \"动作名\", ...参数} 或纯字符串动作名\n"
        "4. 只返回 JSON 或动作名，不要任何解释文字"
    )

    SYSTEM_NARRATE = (
        "你是一个训练分析师。根据结构化训练摘要，用中文生成一段简洁的"
        "自然语言总结（100-300 字）。内容包括：训练是否顺利完成、关键指标、"
        "异常事件与处理结果、整体评价。不要编造数据。"
    )

    SYSTEM_SUGGEST = (
        "你是一个训练系统专家。根据当前注册表和任务上下文，生成一条"
        "新的注册表扩展提议。输出必须为 JSON："
        "{\"name\": \"指标/路径名\", \"direction\": \"max\"|\"min\", "
        "\"kind\": \"metric\"|\"adjustable_path\", \"evidence\": \"依据说明\", "
        "\"entry\": {...具体条目...}}。\n"
        "evidence 必须引用上下文中的具体数据点或趋势，不能写'感觉'或'可能'。"
    )

    @staticmethod
    def _parse_llm_response(text: str) -> Any:
        """从 LLM 返回文本中提取动作：优先找 JSON 对象，否则取纯文本。"""
        text = (text or "").strip()
        # 尝试整体 JSON 解析
        try:
            obj = json.loads(text)
            return obj
        except (json.JSONDecodeError, ValueError):
            pass
        # 尝试用 {} 包裹的 JSON 片段
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
        # 回退：取第一行非空的纯文本作为动作名
        for line in text.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(("//", "#", "--")):
                # 去掉可能的引号
                return stripped.strip("'\"")
        return text

    # -- SDK dispatch

    def _get_api_key(self) -> str | None:
        return resolve_secret(self.cfg, "api_key_env")

    def _get_model_id(self) -> str:
        if self.model:
            return self.model
        if self.provider == "anthropic":
            return "claude-haiku-4-5-20251001"
        if self.provider == "openai":
            return "gpt-4.1-mini"
        return "claude-haiku-4-5-20251001"

    def _check_sdk(self, pkg: str) -> None:
        """SDK 未安装时抛出明确的 RuntimeError，由 decide() 捕获走降级。"""
        try:
            import_module(pkg)
        except ImportError:
            raise RuntimeError(
                f"{pkg} SDK 未安装，请 pip install {pkg} 后重试。"
                f"当前 provider={self.provider}，训练将以规则默认动作继续。"
            ) from None

    def _call_anthropic(self, system_prompt: str, user_message: str, timeout: float,
                        max_tokens: int = 512) -> str:
        self._check_sdk("anthropic")
        import anthropic
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError("未配置 ANTHROPIC_API_KEY 环境变量")
        # per-request timeout + no auto-retry：decide() 自己控制超时与重试
        client = anthropic.Anthropic(api_key=api_key).with_options(
            timeout=min(timeout, 60), max_retries=0,
        )
        response = client.messages.create(
            model=self._get_model_id(),
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        # 安全提取文本：响应可能含多个 content block
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        if not text and getattr(response, "stop_reason", "") == "refusal":
            raise RuntimeError("模型拒绝响应 (refusal)，可能是 prompt 被安全过滤")
        return text

    def _call_openai(self, system_prompt: str, user_message: str, timeout: float,
                     max_tokens: int = 512) -> str:
        self._check_sdk("openai")
        import openai
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError("未配置 OPENAI_API_KEY 环境变量")
        client = openai.OpenAI(api_key=api_key).with_options(
            timeout=min(timeout, 60), max_retries=0,
        )
        response = client.chat.completions.create(
            model=self._get_model_id(),
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content or ""

    def _build_prompt(
        self, decision_point: str, context: dict[str, Any], action_space: list[Any],
    ) -> dict[str, Any]:
        """组装结构化 prompt（供 decide() 传给 _call_llm）。"""
        return {
            "decision_point": decision_point,
            "context": context,
            "action_space": action_space,
            "model": self._get_model_id(),
        }

    def _build_decision_prompt(
        self, decision_point: str, context: dict[str, Any], action_space: list[Any],
    ) -> str:
        """构造 decide() 的 user message。"""
        import json
        parts = [
            f"决策点: {decision_point}",
            f"上下文: {json.dumps(context, ensure_ascii=False, indent=2)}",
            "",
            "可选动作（必须从以下列表中选一个）：",
        ]
        for item in action_space:
            if isinstance(item, str):
                parts.append(f"  - \"{item}\"")
            elif isinstance(item, dict):
                parts.append(f"  - {json.dumps(item, ensure_ascii=False)}")
        parts.append("")
        parts.append("请返回你的选择：")
        return "\n".join(parts)

    def _build_narrative_prompt(
        self, structured_data: dict[str, Any], prompt_template: str | None,
    ) -> str:
        """构造 narrate() 的 user message。"""
        import json
        data_text = json.dumps(structured_data, ensure_ascii=False, indent=2)
        if prompt_template:
            return prompt_template.format(summary=data_text)
        return f"请基于以下训练摘要生成自然语言解读：\n\n{data_text}"

    def _build_suggest_prompt(
        self, kind: str, context: dict[str, Any], registry_snapshot: dict[str, Any] | None,
    ) -> str:
        """构造 suggest() 的 user message。"""
        import json
        parts = [
            f"提议类型: {kind}",
            f"任务上下文: {json.dumps(context, ensure_ascii=False, indent=2)}",
        ]
        if registry_snapshot:
            parts.append(f"当前注册表: {json.dumps(registry_snapshot, ensure_ascii=False, indent=2)}")
        parts.append("")
        parts.append("请生成一条新的注册表扩展提议（JSON 格式）：")
        return "\n".join(parts)

    # -- 三个 LLM 调用入口（替换 NotImplementedError stubs）

    def _call_llm(self, prompt: dict[str, Any], timeout: float) -> Any:
        """decide() 的 LLM 调用：发送决策 prompt，返回动作名或 dict。"""
        decision_point = prompt.get("decision_point", "unknown")
        context = prompt.get("context") or {}
        action_space = list(prompt.get("action_space") or [])

        user_message = self._build_decision_prompt(decision_point, context, action_space)

        if self.provider == "openai":
            raw = self._call_openai(self.SYSTEM_DECIDE, user_message, timeout)
        else:
            raw = self._call_anthropic(self.SYSTEM_DECIDE, user_message, timeout)
        return self._parse_llm_response(raw)

    def _call_llm_text(
        self, structured_data: dict[str, Any], prompt_template: str | None, timeout: float,
    ) -> str:
        """narrate() 的 LLM 调用：返回自然语言文本。"""
        user_message = self._build_narrative_prompt(structured_data, prompt_template)

        if self.provider == "openai":
            return self._call_openai(self.SYSTEM_NARRATE, user_message, timeout, max_tokens=1024)
        return self._call_anthropic(self.SYSTEM_NARRATE, user_message, timeout, max_tokens=1024)

    def _call_llm_suggest(
        self, kind: str, context: dict[str, Any], registry_snapshot: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """suggest() 的 LLM 调用：返回提议 dict。失败抛异常，外层返回 None。"""
        user_message = self._build_suggest_prompt(kind, context, registry_snapshot)

        if self.provider == "openai":
            raw = self._call_openai(self.SYSTEM_SUGGEST, user_message, timeout=15)
        else:
            raw = self._call_anthropic(self.SYSTEM_SUGGEST, user_message, timeout=15)
        result = self._parse_llm_response(raw)
        if not isinstance(result, dict):
            raise ValueError(f"LLM 返回了非 JSON 的提议：{str(result)[:200]}")
        return result

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
