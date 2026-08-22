"""cp_9 · Agent 决策封装 (AgentAdvisor)。

所有"中层 agent 化"决策点的统一出入口：调用 LLM、裁剪动作空间、强制超时
降级，确保规则引擎兜底不被破坏。advisor 跑在 guardian 进程里，不在训练
进程内——sidecar 下 decision_timeout 不占用训练时间。详见 checkpoint/cp_9.md

`agent.enabled: false` 或未配置 api_key_env 对应的环境变量时，`is_enabled()`
恒为 False，整层零成本降级为原有规则引擎行为。
"""

from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any

import json
import os
import sys
from importlib import import_module

from .config import resolve_secret

from guardian.logging_config import get_logger

logger = get_logger(__name__)


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

        # MCP 让位机制：外部 Claude Code 接入时，内置 agent 进入"临时决策"模式
        # autonomous = 自主决策（无 MCP 客户端）
        # provisional = 临时决策（MCP 客户端在线，决策仍然执行但可被覆盖）
        self._mode: str = "autonomous"
        self._delegation_since: float | None = None

        # 待处理决策队列（provisional 模式下由 MCP server 暴露给外部 agent）
        self.pending_decisions: list[dict[str, Any]] = []
        self._max_pending = int(self.cfg.get("max_pending_decisions", 200))

        # 持久化：decision_log 同时写入 JSONL 文件（跨进程可读）
        self._log_path: str | None = self.cfg.get("decision_log_path")

    # --- 开关状态 -----------------------------------------------------

    def is_enabled(self, decision_point: str | None = None) -> bool:
        """配置检查 + 熔断状态 + 单点开关。

        MCP 模式下不再完全让位：agent 继续决策，但标记为 provisional
        （可被外部 agent 覆盖）。这样外部 agent 不在线时也不会丢失智能决策。
        """
        if not self.enabled_cfg:
            return False
        if not self._has_credentials():
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

    def set_delegated(self, mcp_active: bool) -> None:
        """MCP 客户端接入/断开时切换模式。

        mcp_active=True:  外部 Claude Code 在线，agent 进入 provisional 模式。
                          决策仍然执行，但标记为可被覆盖，并推入待处理队列。
        mcp_active=False: 恢复自主决策，清空待处理队列中未被覆盖的条目。
        """
        prev_mode = self._mode
        self._mode = "provisional" if mcp_active else "autonomous"

        if mcp_active and prev_mode != "provisional":
            self._delegation_since = time.time()
            self.decision_log.append({
                "decision_point": "mcp_delegation",
                "action": "provisional_mode",
                "source": "system",
                "latency_ms": 0,
                "timestamp": self._delegation_since,
                "context_summary": "MCP 客户端已连接，agent 进入 provisional 模式（决策仍执行，可被覆盖）",
            })
        elif not mcp_active and prev_mode == "provisional":
            duration = (time.time() - (self._delegation_since or time.time()))
            self._delegation_since = None
            # 清理未被覆盖的待处理决策
            stale_count = sum(1 for d in self.pending_decisions if d["status"] == "pending")
            if stale_count:
                self.decision_log.append({
                    "decision_point": "mcp_delegation",
                    "action": "auto_approved",
                    "source": "system",
                    "latency_ms": 0,
                    "timestamp": time.time(),
                    "context_summary": f"MCP 客户端已断开，{stale_count} 条未覆盖的 provisional 决策自动转为 approved",
                })
            # 将所有 pending 的自动标记为 approved
            for d in self.pending_decisions:
                if d["status"] == "pending":
                    d["status"] = "approved"
                    d["resolved_at"] = time.time()
            self.decision_log.append({
                "decision_point": "mcp_delegation",
                "action": "autonomous_mode",
                "source": "system",
                "latency_ms": 0,
                "timestamp": time.time(),
                "context_summary": f"MCP 客户端已断开，恢复自主决策（让位持续 {duration:.0f}s）",
            })

    @property
    def mode(self) -> str:
        """当前决策模式：autonomous | provisional。"""
        return self._mode

    def _has_credentials(self) -> bool:
        """有任一形式的 API 凭据即返回 True。
        支持：配置指定的 env var、ANTHROPIC_API_KEY、ANTHROPIC_AUTH_TOKEN。
        """
        if resolve_secret(self.cfg, "api_key_env"):
            return True
        # 直接查环境变量（兼容 ANTHROPIC_AUTH_TOKEN 等 OAuth / 第三方兼容 API）
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                     "OPENAI_API_KEY"):
            if os.environ.get(name):
                return True
        return False

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
            logger.error("decide() 调用 LLM 失败，降级为规则默认动作 (decision_point=%s)",
                         decision_point, exc_info=True)
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
            # narrate 需要生成较长文本，给 2x decision_timeout
            _narrate_timeout = max(self.decision_timeout * 2, 20)
            future = self._executor.submit(
                self._call_llm_text, structured_data, prompt_template, _narrate_timeout,
            )
            text = future.result(timeout=_narrate_timeout)
        except FutureTimeoutError:
            self._record_failure()
            return None
        except Exception:
            logger.warning("narrate() 生成自然语言解读失败，返回 None", exc_info=True)
            self._record_failure()
            return None
        self._record_success()
        return text

    def recommend_charts(
        self,
        process_id: str,
        metrics_summary: dict[str, Any],
        chart_groups: list[str] | None = None,
        anomaly_count: int = 0,
        training_phase: str = "mid",
    ) -> dict[str, Any] | None:
        """图表推荐：根据当前训练状态推荐 Dashboard 应关注的图表组。

        供 MCP 工具 recommend_charts 调用，失败返回 None（降级为当前配置不变）。
        """
        if not self.is_enabled("chart_selection"):
            return None

        available = chart_groups or ["loss", "accuracy", "lr", "gpu", "custom"]
        context = {
            "process_id": process_id,
            "metrics_summary": metrics_summary,
            "available_groups": available,
            "anomaly_count": anomaly_count,
            "training_phase": training_phase,
        }
        try:
            future = self._executor.submit(
                self._call_llm_chart_recommend, context, self.decision_timeout,
            )
            result = future.result(timeout=self.decision_timeout)
        except FutureTimeoutError:
            self._record_failure()
            return None
        except Exception:
            logger.warning("recommend_charts() 调用 LLM 失败", exc_info=True)
            self._record_failure()
            return None

        self._record_success()
        return result

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
        if not self.enabled_cfg or not self._has_credentials():
            return None
        try:
            return self._call_llm_suggest(kind, context, registry_snapshot)
        except Exception:
            logger.warning("suggest() 生成注册表建议失败 (kind=%s)，返回 None", kind, exc_info=True)
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

    SYSTEM_CHART_RECOMMEND = (
        "你是一个训练监控专家。根据当前训练状态（指标趋势、异常数量、训练阶段），"
        "推荐 Dashboard 应重点关注的图表组和显示配置。\n"
        "规则：\n"
        "- loss 异常或训练早期 → 必选 loss\n"
        "- 训练中后期、有 accuracy/metric 数据 → 加选 accuracy\n"
        "- GPU 温度异常、利用率异常 → 加选 gpu\n"
        "- lr 只在 warmup 结束/decay 阶段有意义\n"
        "- 训练接近结束时建议开 smoothing 看趋势\n"
        "返回 JSON：{\"groups\": [\"loss\", \"accuracy\", ...], \"smoothing\": true|false, "
        "\"reason\": \"一句话推荐理由（中文）\"}\n"
        "groups 必须是 available_groups 的子集，smoothing 是布尔值。"
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
            # LLM 输出非 JSON 属预期情况，继续尝试后续解析策略
            pass
        # 尝试用 {} 包裹的 JSON 片段
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except (json.JSONDecodeError, ValueError):
                # 片段解析失败同样属预期情况，继续回退纯文本解析
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
        """取 API 凭据：配置的 env var → ANTHROPIC_API_KEY → ANTHROPIC_AUTH_TOKEN。"""
        key = resolve_secret(self.cfg, "api_key_env")
        if key:
            return key
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"):
            val = os.environ.get(name)
            if val:
                return val
        return None

    def _get_model_id(self) -> str:
        if self.model:
            return self.model
        # 环境变量覆盖（兼容 ANTHROPIC_MODEL / ANTHROPIC_DEFAULT_HAIKU_MODEL）
        for env_name in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
            val = os.environ.get(env_name)
            if val:
                return val
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
            raise RuntimeError("未配置 ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN 环境变量")

        # 构造 client：支持自定义 base_url（第三方 Anthropic 兼容 API）
        client_kwargs: dict[str, Any] = {"max_retries": 0}
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url

        # 区分 api_key 与 auth_token（OAuth / 第三方兼容 API 用 token）
        if os.environ.get("ANTHROPIC_AUTH_TOKEN") and not os.environ.get("ANTHROPIC_API_KEY"):
            client_kwargs["auth_token"] = api_key
        else:
            client_kwargs["api_key"] = api_key

        client = anthropic.Anthropic(**client_kwargs).with_options(
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
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs).with_options(
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

    def _call_llm_chart_recommend(self, context: dict[str, Any], timeout: float) -> dict[str, Any]:
        """recommend_charts() 的 LLM 调用：返回推荐配置 dict。"""
        user = (
            f"训练状态:\n"
            f"  指标摘要: {json.dumps(context.get('metrics_summary', {}), ensure_ascii=False)}\n"
            f"  可用图表组: {context.get('available_groups', [])}\n"
            f"  异常计数: {context.get('anomaly_count', 0)}\n"
            f"  训练阶段: {context.get('training_phase', 'mid')}\n"
            f"\n请推荐图表配置（JSON）。"
        )
        if self.provider == "openai":
            raw = self._call_openai(self.SYSTEM_CHART_RECOMMEND, user, timeout)
        else:
            raw = self._call_anthropic(self.SYSTEM_CHART_RECOMMEND, user, timeout)
        result = self._parse_llm_response(raw)
        if not isinstance(result, dict):
            raise ValueError(f"LLM 返回了非 JSON：{str(result)[:200]}")
        # 确保 groups 是 available 的子集
        available = set(context.get("available_groups", []))
        result["groups"] = [g for g in result.get("groups", []) if g in available]
        result.setdefault("smoothing", False)
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
        """记录每次决策：来源、耗时、上下文摘要。供 summary / MCP 工具查询。

        provisional 模式下：自动推入 pending_decisions 队列，标记为可被外部覆盖。
        """
        # 在 provisional 模式下，将 agent 来源改写为 agent_provisional
        if source == "agent" and self._mode == "provisional":
            source = "agent_provisional"

        entry = {
            "decision_point": decision_point,
            "action": chosen_action,
            "source": source,
            "latency_ms": round(latency_ms, 2),
            "timestamp": time.time(),
            "context_summary": _summarize_context(context),
        }
        self.decision_log.append(entry)
        self._persist_decision(entry)

        # provisional 模式的决策推入待处理队列，供外部 agent 查阅/覆盖
        if source == "agent_provisional":
            pending = {
                "id": f"pd_{uuid.uuid4().hex[:12]}",
                "decision_point": decision_point,
                "context": context,
                "provisional_action": chosen_action,
                "status": "pending",          # pending | approved | overridden
                "created_at": entry["timestamp"],
                "ttl": float(self.cfg.get("pending_decision_ttl", 120)),
                "resolved_by": None,
                "resolved_action": None,
                "resolved_at": None,
            }
            self.pending_decisions.append(pending)
            # 限制队列长度，防止内存泄漏
            if len(self.pending_decisions) > self._max_pending:
                self.pending_decisions = self.pending_decisions[-self._max_pending:]
            # 清理过期条目
            self._expire_pending()

        return entry

    def _persist_decision(self, entry: dict[str, Any]) -> None:
        """将单条决策追加到 JSONL 文件（非阻塞，失败不影响训练流程）。"""
        if not self._log_path:
            return
        try:
            from pathlib import Path
            line = json.dumps(entry, ensure_ascii=False, default=str)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            logger.warning("决策日志写入失败: %s", self._log_path, exc_info=True)

    @staticmethod
    def load_log(log_path: str) -> list[dict[str, Any]]:
        """从 JSONL 文件加载历史决策日志（供 MCP 跨进程读取）。"""
        entries: list[dict[str, Any]] = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning("读取决策日志失败: %s", log_path, exc_info=True)
        return entries

    def _expire_pending(self) -> None:
        """清理已过期的待处理决策：超时自动转为 approved。"""
        now = time.time()
        for d in self.pending_decisions:
            if d["status"] == "pending" and (now - d["created_at"]) > d["ttl"]:
                d["status"] = "approved"
                d["resolved_at"] = now
                self.decision_log.append({
                    "decision_point": d["decision_point"],
                    "action": "auto_approved",
                    "source": "system",
                    "latency_ms": 0,
                    "timestamp": now,
                    "context_summary": f"provisional 决策 {d['id']} 超时未覆盖，自动转为 approved",
                })

    # --- 待处理决策查询与覆盖（MCP 工具调用入口） ---

    def get_pending_decisions(self) -> list[dict[str, Any]]:
        """返回当前所有 pending 状态的待处理决策（供 MCP 工具 get_pending_decisions）。

        返回的列表不含内部 context 冗余字段，改为摘要形式。
        """
        self._expire_pending()
        result = []
        for d in self.pending_decisions:
            if d["status"] != "pending":
                continue
            result.append({
                "id": d["id"],
                "decision_point": d["decision_point"],
                "provisional_action": d["provisional_action"],
                "context_summary": _summarize_context(d.get("context", {})),
                "created_at": d["created_at"],
                "ttl": d["ttl"],
                "remaining_seconds": round(max(0, d["ttl"] - (time.time() - d["created_at"])), 1),
            })
        return result

    def resolve_decision(
        self,
        decision_id: str,
        action: str | None = None,
        param: Any = None,
        override: bool = False,
    ) -> dict[str, Any]:
        """外部 agent 处理一条待处理决策（供 MCP 工具 resolve_decision）。

        override=False: 认可当前 provisional 决策，标记为 approved。
        override=True:  用新的 action 覆盖。返回的 dict 包含 corrective_info，
                        供 MCP handler 判断是否需要执行补救操作。

        返回 {"status": "approved"|"overridden"|"not_found"|"already_resolved", ...}
        """
        self._expire_pending()
        for d in self.pending_decisions:
            if d["id"] != decision_id:
                continue
            if d["status"] != "pending":
                return {
                    "status": "already_resolved",
                    "id": decision_id,
                    "current_status": d["status"],
                    "resolved_by": d.get("resolved_by"),
                    "resolved_at": d.get("resolved_at"),
                }

            now = time.time()
            if not override:
                d["status"] = "approved"
                d["resolved_by"] = "mcp_agent"
                d["resolved_at"] = now
                self.decision_log.append({
                    "decision_point": d["decision_point"],
                    "action": "mcp_approved",
                    "source": "mcp_agent",
                    "latency_ms": 0,
                    "timestamp": now,
                    "context_summary": f"外部 agent 认可 provisional 决策 {decision_id}：{d['provisional_action']}",
                })
                return {
                    "status": "approved",
                    "id": decision_id,
                    "provisional_action": d["provisional_action"],
                    "corrective_needed": False,
                }

            # override: 外部 agent 选择了不同的动作
            d["status"] = "overridden"
            d["resolved_by"] = "mcp_agent"
            d["resolved_action"] = {"action": action, "param": param}
            d["resolved_at"] = now

            # 生成补救信息，供 MCP handler 执行
            corrective = {
                "action": action,
                "param": param,
                "original_action": d["provisional_action"],
                "decision_point": d["decision_point"],
            }
            self.decision_log.append({
                "decision_point": d["decision_point"],
                "action": "mcp_overridden",
                "source": "mcp_agent",
                "latency_ms": 0,
                "timestamp": now,
                "context_summary": (
                    f"外部 agent 覆盖 provisional 决策 {decision_id}："
                    f"{d['provisional_action']} → {action}"
                    + (f"（param={param}）" if param is not None else "")
                ),
            })
            return {
                "status": "overridden",
                "id": decision_id,
                "original_action": d["provisional_action"],
                "corrective_needed": True,
                "corrective": corrective,
            }

        return {"status": "not_found", "id": decision_id}

    def close(self) -> None:
        """释放线程池，测试/主流程结束时可选调用。"""
        # 清理未处理的 pending 决策
        self._expire_pending()
        for d in self.pending_decisions:
            if d["status"] == "pending":
                d["status"] = "approved"
                d["resolved_at"] = time.time()
        # 确保内存中剩余决策也持久化
        for entry in self.decision_log:
            self._persist_decision(entry)
        self._executor.shutdown(wait=False, cancel_futures=True)


def _summarize_context(context: dict[str, Any], max_len: int = 200) -> str:
    """把上下文压成一行摘要，避免决策日志无限膨胀。"""
    try:
        text = ", ".join(f"{k}={v}" for k, v in context.items())
    except Exception:
        logger.warning("上下文摘要生成失败，回退为 str(context)", exc_info=True)
        text = str(context)
    return text[:max_len]
