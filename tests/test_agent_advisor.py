"""cp_9 · AgentAdvisor 单元测试（mock LLM，不需要真实 API key）。

覆盖 checkpoint/cp_9.md 的快速校验 + 完整校验表。
"""

from __future__ import annotations

import time

import pytest

from guardian.agent_advisor import AgentAdvisor

API_KEY_ENV = "TEST_ADVISOR_API_KEY"


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    """所有测试默认视为"已配置 API key"，未配置降级单独测。"""
    monkeypatch.setenv(API_KEY_ENV, "sk-fake-for-tests")


def make_advisor(**overrides) -> AgentAdvisor:
    cfg = {
        "enabled": True,
        "provider": "anthropic",
        "api_key_env": API_KEY_ENV,
        "decision_timeout": 0.2,
        "consecutive_failure_threshold": 5,
        "circuit_breaker_cooldown": 0.3,
        "decision_points": {},
    }
    cfg.update(overrides)
    return AgentAdvisor(cfg)


# ---------- 快速校验 ----------

def test_instantiable():
    adv = AgentAdvisor({"enabled": False})
    assert adv is not None
    adv.close()


def test_disabled_by_config_returns_default():
    adv = AgentAdvisor({"enabled": False})
    result = adv.decide("monitor_response", {"loss": 1.0}, ["ignore", "alert_only"], "alert_only")
    assert result["action"] == "alert_only"
    assert result["source"] == "disabled"
    adv.close()


def test_normal_decision_returns_agent_choice(monkeypatch):
    adv = make_advisor()
    monkeypatch.setattr(adv, "_call_llm", lambda prompt, timeout: "ignore")
    result = adv.decide("monitor_response", {"loss": 1.0}, ["ignore", "alert_only"], "alert_only")
    assert result["action"] == "ignore"
    assert result["source"] == "agent"
    adv.close()


def test_invalid_output_falls_back_to_default(monkeypatch):
    adv = make_advisor()
    monkeypatch.setattr(adv, "_call_llm", lambda prompt, timeout: "skip_batch")
    result = adv.decide("monitor_response", {"loss": 1.0}, ["ignore", "alert_only"], "alert_only")
    assert result["action"] == "alert_only"
    assert result["source"] == "invalid_output"
    adv.close()


def test_narrate_basic(monkeypatch):
    adv = make_advisor()
    monkeypatch.setattr(adv, "_call_llm_text", lambda data, tmpl, timeout: "本次训练总体平稳。")
    text = adv.narrate({"status": "completed"})
    assert isinstance(text, str) and text
    adv.close()


# ---------- 完整校验 ----------

def test_timeout_degrades_immediately(monkeypatch):
    adv = make_advisor(decision_timeout=0.05)

    def slow_call(prompt, timeout):
        time.sleep(1.0)
        return "ignore"

    monkeypatch.setattr(adv, "_call_llm", slow_call)
    start = time.monotonic()
    result = adv.decide("monitor_response", {}, ["ignore", "alert_only"], "alert_only")
    elapsed = time.monotonic() - start
    assert result["action"] == "alert_only"
    assert result["source"] == "timeout"
    assert elapsed < 0.5, "必须在 decision_timeout 附近立即返回，不能等满慢调用"
    adv.close()


def test_circuit_breaker_trips_after_threshold(monkeypatch):
    adv = make_advisor(consecutive_failure_threshold=5, circuit_breaker_cooldown=10)
    calls = {"n": 0}

    def failing_call(prompt, timeout):
        calls["n"] += 1
        raise RuntimeError("network down")

    monkeypatch.setattr(adv, "_call_llm", failing_call)
    for _ in range(5):
        r = adv.decide("monitor_response", {}, ["ignore", "alert_only"], "alert_only")
        assert r["source"] == "error"
    assert calls["n"] == 5

    # 第 6 次：熔断已触发，不应再发起网络请求
    r6 = adv.decide("monitor_response", {}, ["ignore", "alert_only"], "alert_only")
    assert r6["source"] == "disabled"
    assert calls["n"] == 5, "熔断后不应再调用 _call_llm"
    adv.close()


def test_circuit_breaker_recovers_after_cooldown(monkeypatch):
    adv = make_advisor(consecutive_failure_threshold=2, circuit_breaker_cooldown=0.1)

    def failing_call(prompt, timeout):
        raise RuntimeError("boom")

    monkeypatch.setattr(adv, "_call_llm", failing_call)
    adv.decide("monitor_response", {}, ["ignore", "alert_only"], "alert_only")
    adv.decide("monitor_response", {}, ["ignore", "alert_only"], "alert_only")
    assert not adv.is_enabled("monitor_response")

    time.sleep(0.15)  # 等 cooldown 结束
    assert adv.is_enabled("monitor_response")

    monkeypatch.setattr(adv, "_call_llm", lambda prompt, timeout: "ignore")
    r = adv.decide("monitor_response", {}, ["ignore", "alert_only"], "alert_only")
    assert r["source"] == "agent"
    adv.close()


def test_param_out_of_range_is_invalid(monkeypatch):
    adv = make_advisor()
    action_space = [
        "alert_only",
        {"action": "restart_with_lower_lr", "ratio": {"min": 0.1, "max": 0.9}},
    ]
    monkeypatch.setattr(
        adv, "_call_llm",
        lambda prompt, timeout: {"action": "restart_with_lower_lr", "ratio": 5.0},
    )
    result = adv.decide("monitor_response", {}, action_space, "alert_only")
    assert result["source"] == "invalid_output"
    assert result["action"] == "alert_only"
    adv.close()


def test_param_in_range_is_accepted(monkeypatch):
    adv = make_advisor()
    action_space = [
        "alert_only",
        {"action": "restart_with_lower_lr", "ratio": {"min": 0.1, "max": 0.9}},
    ]
    monkeypatch.setattr(
        adv, "_call_llm",
        lambda prompt, timeout: {"action": "restart_with_lower_lr", "ratio": 0.5},
    )
    result = adv.decide("monitor_response", {}, action_space, "alert_only")
    assert result["source"] == "agent"
    assert result["action"] == {"action": "restart_with_lower_lr", "ratio": 0.5}
    adv.close()


def test_decision_log_records_every_call(monkeypatch):
    adv = make_advisor()
    monkeypatch.setattr(adv, "_call_llm", lambda prompt, timeout: "ignore")
    for i in range(10):
        adv.decide(f"point_{i}", {"i": i}, ["ignore", "alert_only"], "alert_only")
    assert len(adv.decision_log) == 10
    for i, entry in enumerate(adv.decision_log):
        assert entry["decision_point"] == f"point_{i}"
        assert entry["source"] == "agent"
        assert "latency_ms" in entry
        assert "context_summary" in entry
    adv.close()


def test_narrate_failure_returns_none(monkeypatch):
    adv = make_advisor()
    monkeypatch.setattr(adv, "_call_llm_text", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    result = adv.narrate({"status": "failed"})
    assert result is None
    adv.close()


def test_concurrent_decide_calls_do_not_block_each_other(monkeypatch):
    adv = make_advisor(decision_timeout=0.3)

    def slow_call(prompt, timeout):
        time.sleep(0.15)
        return "ignore"

    monkeypatch.setattr(adv, "_call_llm", slow_call)

    from concurrent.futures import ThreadPoolExecutor
    caller_pool = ThreadPoolExecutor(max_workers=2)
    start = time.monotonic()
    f1 = caller_pool.submit(adv.decide, "monitor_response", {}, ["ignore"], "alert_only")
    f2 = caller_pool.submit(adv.decide, "watchdog_recovery", {}, ["ignore"], "alert_only")
    r1, r2 = f1.result(), f2.result()
    elapsed = time.monotonic() - start
    assert r1["source"] == "agent" and r2["source"] == "agent"
    # 并发执行应远小于两次串行耗时（2 * 0.15s）
    assert elapsed < 0.3
    caller_pool.shutdown()
    adv.close()


def test_single_decision_point_switch(monkeypatch):
    adv = make_advisor(decision_points={"watchdog_recovery": False})
    monkeypatch.setattr(adv, "_call_llm", lambda prompt, timeout: "ignore")

    r_watchdog = adv.decide("watchdog_recovery", {}, ["ignore", "resume_unchanged"], "resume_unchanged")
    assert r_watchdog["source"] == "disabled"

    r_monitor = adv.decide("monitor_response", {}, ["ignore", "alert_only"], "alert_only")
    assert r_monitor["source"] == "agent", "关闭 watchdog_recovery 不应影响 monitor_response"
    adv.close()


def test_suggest_exception_returns_none(monkeypatch):
    adv = make_advisor()
    monkeypatch.setattr(adv, "_call_llm_suggest", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    result = adv.suggest("metric_registry", {"task_type": "classification"}, {})
    assert result is None
    adv.close()


def test_suggest_success_returns_dict(monkeypatch):
    adv = make_advisor()
    monkeypatch.setattr(
        adv, "_call_llm_suggest",
        lambda kind, context, snapshot: {"name": "val/f1", "direction": "max", "evidence": "..."},
    )
    result = adv.suggest("metric_registry", {"task_type": "classification"}, {})
    assert result["name"] == "val/f1"
    adv.close()


def test_no_api_key_disables(monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    adv = AgentAdvisor({"enabled": True, "api_key_env": API_KEY_ENV})
    assert adv.is_enabled() is False
    result = adv.decide("monitor_response", {}, ["ignore", "alert_only"], "alert_only")
    assert result["source"] == "disabled"
    adv.close()


def test_sidecar_action_space_filtered_by_caller(monkeypatch):
    """sidecar 下调用方传入不含 skip_batch 的 action_space，advisor 仍拒绝越界动作。"""
    adv = make_advisor()
    sidecar_action_space = ["ignore", "restart_with_lower_lr", "alert_only"]  # 不含 skip_batch
    monkeypatch.setattr(adv, "_call_llm", lambda prompt, timeout: "skip_batch")
    result = adv.decide("monitor_response", {}, sidecar_action_space, "alert_only")
    assert result["source"] == "invalid_output"
    assert result["action"] == "alert_only"
    adv.close()
