"""History API 集成测试（remote/server.py 的 9 个历史端点）。

使用 starlette TestClient + 临时 persist_dir 中的 fake events.jsonl。
不需要真实 LLM 凭据——AI 端点测试降级路径（summary fallback）。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from guardian.remote.server import RemoteServer, RemoteHandler


# ── Fixtures ─────────────────────────────────────────────────────────

def _make_handler() -> RemoteHandler:
    """创建一个 mock RemoteHandler，所有方法返回安全默认值。"""
    h = MagicMock()
    h.get_training_status.return_value = {"status": "idle"}
    h.get_metrics_history.return_value = {"metrics": []}
    h.get_gpu_status.return_value = {}
    h.approve_action.return_value = {"status": "approved"}
    h.reject_action.return_value = {"status": "rejected"}
    h.get_decision_log.return_value = []
    h.get_anomaly_history.return_value = []
    h.get_training_log.return_value = []
    h.get_device_info.return_value = {}
    return h


def _write_events(persist_dir: Path, session_id: str, events: list[dict]) -> None:
    """写入 events.jsonl 文件。"""
    d = persist_dir / session_id
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "events.jsonl", "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")


@pytest.fixture()
def history_client():
    """创建带 3 个假历史会话的 TestClient。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        persist = Path(tmpdir)

        # Session A: 完整训练（metrics + anomaly + decision + crash）
        _write_events(persist, "session-a", [
            {"type": "training_start", "timestamp": 1000.0, "data": {}},
            {"type": "metrics", "timestamp": 1001.0, "data": {"step": 1, "loss": 2.5, "accuracy": 0.1}},
            {"type": "metrics", "timestamp": 1002.0, "data": {"step": 2, "loss": 1.8, "accuracy": 0.3}},
            {"type": "metrics", "timestamp": 1003.0, "data": {"step": 3, "loss": 0.9, "accuracy": 0.7}},
            {"type": "anomaly", "timestamp": 1004.0, "data": {"type": "loss_spike", "severity": "medium"}},
            {"type": "decision", "timestamp": 1005.0, "data": {"action": "reduce_lr", "source": "agent"}},
            {"type": "crash", "timestamp": 1006.0, "data": {"type": "oom", "detail": "CUDA OOM"}},
            {"type": "metrics", "timestamp": 1007.0, "data": {"step": 4, "loss": 0.5, "accuracy": 0.85}},
            {"type": "training_end", "timestamp": 1008.0, "data": {"status": "completed"}},
        ])

        # Session B: 短训练（只有 metrics）
        _write_events(persist, "session-b", [
            {"type": "metrics", "timestamp": 2000.0, "data": {"step": 1, "loss": 3.0, "accuracy": 0.05}},
            {"type": "metrics", "timestamp": 2001.0, "data": {"step": 2, "loss": 2.0, "accuracy": 0.2}},
        ])

        # Session C: 空目录（无 events.jsonl，应被跳过）
        (persist / "session-empty").mkdir()

        handler = _make_handler()
        server = RemoteServer(handler, port=0, persist_dir=persist)
        app = server._build_app()

        from starlette.testclient import TestClient
        with TestClient(app) as client:
            yield client


# ── 测试: GET /api/history/sessions ─────────────────────────────────

class TestHistorySessions:
    def test_lists_sessions(self, history_client):
        resp = history_client.get("/api/history/sessions")
        assert resp.status_code == 200
        data = resp.json()
        ids = [s["session_id"] for s in data["sessions"]]
        assert "session-a" in ids
        assert "session-b" in ids
        # session-empty 没有 events.jsonl，不应出现
        assert "session-empty" not in ids

    def test_session_has_correct_event_count(self, history_client):
        resp = history_client.get("/api/history/sessions")
        sessions = resp.json()["sessions"]
        a = next(s for s in sessions if s["session_id"] == "session-a")
        assert a["event_count"] == 9

    def test_session_has_type_counts(self, history_client):
        resp = history_client.get("/api/history/sessions")
        sessions = resp.json()["sessions"]
        a = next(s for s in sessions if s["session_id"] == "session-a")
        tc = a["type_counts"]
        assert tc["metrics"] == 4
        assert tc["anomaly"] == 1
        assert tc["decision"] == 1
        assert tc["crash"] == 1

    def test_session_has_last_metrics(self, history_client):
        resp = history_client.get("/api/history/sessions")
        sessions = resp.json()["sessions"]
        a = next(s for s in sessions if s["session_id"] == "session-a")
        assert a["last_metrics"]["loss"] == 0.5
        assert a["last_metrics"]["accuracy"] == 0.85

    def test_session_duration(self, history_client):
        resp = history_client.get("/api/history/sessions")
        sessions = resp.json()["sessions"]
        a = next(s for s in sessions if s["session_id"] == "session-a")
        assert a["duration_seconds"] == 8.0  # 1008 - 1000


# ── 测试: GET /api/history/{session_id}/summary ─────────────────────

class TestHistorySummary:
    def test_summary_found(self, history_client):
        resp = history_client.get("/api/history/session-a/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "session-a"
        assert data["event_count"] == 9
        assert data["metrics_count"] == 4
        assert data["anomaly_count"] == 1
        assert data["decision_count"] == 1
        assert data["crash_count"] == 1

    def test_summary_not_found(self, history_client):
        resp = history_client.get("/api/history/nonexistent/summary")
        assert resp.status_code == 404

    def test_summary_last_metrics(self, history_client):
        resp = history_client.get("/api/history/session-a/summary")
        data = resp.json()
        assert data["last_metrics"]["loss"] == 0.5


# ── 测试: GET /api/history/{session_id}/metrics ─────────────────────

class TestHistoryMetrics:
    def test_metrics_returned(self, history_client):
        resp = history_client.get("/api/history/session-a/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["metrics"]) == 4

    def test_metrics_have_timestamp(self, history_client):
        resp = history_client.get("/api/history/session-a/metrics")
        metrics = resp.json()["metrics"]
        assert all("timestamp" in m for m in metrics)

    def test_metrics_loss_values(self, history_client):
        resp = history_client.get("/api/history/session-a/metrics")
        metrics = resp.json()["metrics"]
        losses = [m["loss"] for m in metrics]
        assert losses == [2.5, 1.8, 0.9, 0.5]

    def test_metrics_limit(self, history_client):
        resp = history_client.get("/api/history/session-a/metrics?limit=2")
        metrics = resp.json()["metrics"]
        assert len(metrics) == 2


# ── 测试: GET /api/history/{session_id}/anomalies ───────────────────

class TestHistoryAnomalies:
    def test_anomalies_returned(self, history_client):
        resp = history_client.get("/api/history/session-a/anomalies")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["anomalies"]) == 1
        assert data["anomalies"][0]["type"] == "loss_spike"

    def test_anomalies_empty_for_session_b(self, history_client):
        resp = history_client.get("/api/history/session-b/anomalies")
        assert resp.json()["anomalies"] == []


# ── 测试: GET /api/history/{session_id}/decisions ───────────────────

class TestHistoryDecisions:
    def test_decisions_returned(self, history_client):
        resp = history_client.get("/api/history/session-a/decisions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["decisions"]) == 1
        assert data["decisions"][0]["action"] == "reduce_lr"


# ── 测试: GET /api/history/{session_id}/crashes ─────────────────────

class TestHistoryCrashes:
    def test_crashes_returned(self, history_client):
        resp = history_client.get("/api/history/session-a/crashes")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["crashes"]) == 1
        assert data["crashes"][0]["type"] == "oom"
        assert "CUDA OOM" in data["crashes"][0]["detail"]

    def test_crashes_empty_for_session_b(self, history_client):
        resp = history_client.get("/api/history/session-b/crashes")
        assert resp.json()["crashes"] == []


# ── 测试: POST /api/history/{session_id}/ai/analyze ─────────────────

class TestHistoryAiAnalyze:
    def test_ai_analyze_fallback_to_summary(self, history_client):
        """无 AI 凭据时降级为 summary 文本。"""
        resp = history_client.post("/api/history/session-a/ai/analyze")
        assert resp.status_code == 200
        data = resp.json()
        assert "analysis" in data
        assert "session-a" in data["analysis"]
        # 降级路径返回 source=summary
        assert data["source"] == "summary"

    def test_ai_analyze_not_found(self, history_client):
        resp = history_client.post("/api/history/nonexistent/ai/analyze")
        assert resp.status_code == 404


# ── 测试: POST /api/history/{session_id}/ai/chat ────────────────────

class TestHistoryAiChat:
    def test_ai_chat_empty_question(self, history_client):
        resp = history_client.post("/api/history/session-a/ai/chat",
                                   json={"question": ""})
        assert resp.status_code == 400

    def test_ai_chat_fallback(self, history_client):
        """无 AI 凭据时返回失败提示。"""
        resp = history_client.post("/api/history/session-a/ai/chat",
                                   json={"question": "这次训练怎么样？"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data

    def test_ai_chat_not_found(self, history_client):
        resp = history_client.post("/api/history/nonexistent/ai/chat",
                                   json={"question": "test"})
        assert resp.status_code == 404


# ── 测试: POST /api/history/compare ─────────────────────────────────

class TestHistoryCompare:
    def test_compare_two_sessions(self, history_client):
        resp = history_client.post("/api/history/compare",
                                   json={"session_ids": ["session-a", "session-b"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 2
        a = next(s for s in data["sessions"] if s["session_id"] == "session-a")
        b = next(s for s in data["sessions"] if s["session_id"] == "session-b")
        assert a["loss_final"] == 0.5
        assert b["loss_final"] == 2.0
        assert a["acc_best"] == 0.85
        assert b["acc_best"] == 0.2

    def test_compare_needs_at_least_two(self, history_client):
        resp = history_client.post("/api/history/compare",
                                   json={"session_ids": ["session-a"]})
        assert resp.status_code == 400

    def test_compare_handles_missing_session(self, history_client):
        resp = history_client.post("/api/history/compare",
                                   json={"session_ids": ["session-a", "nonexistent"]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["sessions"]) == 2
        missing = next(s for s in data["sessions"] if s["session_id"] == "nonexistent")
        assert "error" in missing

    def test_compare_duration(self, history_client):
        resp = history_client.post("/api/history/compare",
                                   json={"session_ids": ["session-a", "session-b"]})
        data = resp.json()
        a = next(s for s in data["sessions"] if s["session_id"] == "session-a")
        assert a["duration_seconds"] == 8.0
        b = next(s for s in data["sessions"] if s["session_id"] == "session-b")
        assert b["duration_seconds"] == 1.0
