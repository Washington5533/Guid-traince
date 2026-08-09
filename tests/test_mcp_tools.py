"""mcp_server.py 工具路由与鉴权测试（不依赖 MCP SDK）。"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from guardian.mcp_server import (
    GuardianMCPServer,
    IdempotencyGuard,
    READONLY_TOOLS,
    READONLY_TOOLS_V2,
    WRITE_TOOLS,
    WRITE_TOOLS_V2,
)


# ---------------------------------------------------------------------------
# IdempotencyGuard
# ---------------------------------------------------------------------------

class TestIdempotencyGuard:
    def test_first_call_not_duplicate(self):
        g = IdempotencyGuard(window=300)
        assert g.check("req-1") is None

    def test_duplicate_within_window(self):
        g = IdempotencyGuard(window=300)
        g.record("req-1", {"status": "ok"})
        assert g.check("req-1") == {"status": "ok"}

    def test_none_request_id_always_none(self):
        g = IdempotencyGuard(window=300)
        assert g.check(None) is None
        g.record(None, {"x": 1})
        assert g.check(None) is None

    def test_no_duplicate_after_window(self):
        # 用很短的窗口，手动将记录时间改到过期
        g = IdempotencyGuard(window=300)
        g.record("req-1", {"status": "ok"})
        # 模拟时间推进：把记录的 timestamp 改到 600s 之前
        g._seen["req-1"] = (g._seen["req-1"][0] - 600, g._seen["req-1"][1])
        assert g.check("req-1") is None  # 过期

    def test_cleanup_removes_expired(self):
        g = IdempotencyGuard(window=300)
        g.record("old", {"x": 1})
        # 模拟过期
        g._seen["old"] = (g._seen["old"][0] - 600, g._seen["old"][1])
        g.cleanup()
        assert "old" not in g._seen


# ---------------------------------------------------------------------------
# MCP Server 工具数量
# ---------------------------------------------------------------------------

class TestToolCounts:
    def test_total_tools(self):
        total = (len(READONLY_TOOLS) + len(READONLY_TOOLS_V2) +
                 len(WRITE_TOOLS) + len(WRITE_TOOLS_V2))
        # trigger_full_validate 已移除，
        # 新增 get_post_training_checklist + get_training_log +
        # get_pending_decisions + resolve_decision，
        # v0.2 再新增 4 个 Dashboard 工具
        # （get/set_dashboard_config + recommend_charts + list_dashboard_templates）
        assert total == 35

    def test_readonly_tool_names_unique(self):
        all_names = [t["name"] for t in READONLY_TOOLS + READONLY_TOOLS_V2 +
                     WRITE_TOOLS + WRITE_TOOLS_V2]
        assert len(all_names) == len(set(all_names)), "工具名重复"

    def test_all_have_input_schema(self):
        for t in READONLY_TOOLS + READONLY_TOOLS_V2 + WRITE_TOOLS + WRITE_TOOLS_V2:
            assert "inputSchema" in t, f"{t['name']} 缺少 inputSchema"


# ---------------------------------------------------------------------------
# MCP Server 创建与基本属性
# ---------------------------------------------------------------------------

@pytest.fixture
def server():
    """创建一个空配置的 stdio 模式 GuardianMCPServer。"""
    srv = GuardianMCPServer(
        config={
            "project": {"log_dir": str(Path(tempfile.mkdtemp()) / "logs")},
            "mcp": {"enable_write_tools": True},
        },
        mode="shared",
    )
    srv._transport = "stdio"
    srv.__init_handlers__()
    return srv


class TestServerBasics:
    def test_is_available(self):
        ok, err = GuardianMCPServer.is_available()
        # mcp 包可能已安装也可能未安装，两种都合法
        if ok:
            assert err is None
        else:
            assert err is not None
            assert "未安装" in err

    def test_default_transport(self):
        srv = GuardianMCPServer()
        assert srv._transport == "stdio"

    def test_stdio_auth_always_passes(self, server):
        ok, msg = server._authorize("any_tool")
        assert ok is True
        assert msg == "ok"

    def test_default_write_disabled(self):
        srv = GuardianMCPServer()
        assert srv.write_enabled is False


# ---------------------------------------------------------------------------
# 鉴权：非 stdio 模式（模拟 SSE/HTTP）
# ---------------------------------------------------------------------------

class TestAuthNonStdio:
    def test_write_disabled_blocks(self):
        srv = GuardianMCPServer(
            config={"mcp": {"enable_write_tools": False}},
        )
        srv._transport = "sse"
        ok, msg = srv._authorize("trigger_recovery")
        assert ok is False
        assert "未启用" in msg

    def test_no_token_env_blocks(self):
        srv = GuardianMCPServer(
            config={"mcp": {"enable_write_tools": True}},
        )
        srv._transport = "sse"
        srv.write_token = ""  # 环境变量未设置
        ok, msg = srv._authorize("trigger_recovery")
        assert ok is False
        assert "未配置" in msg

    def test_wrong_token_blocks(self):
        srv = GuardianMCPServer(
            config={"mcp": {"enable_write_tools": True}},
        )
        srv._transport = "sse"
        srv.write_token = "correct"
        ok, msg = srv._authorize("trigger_recovery", token="wrong")
        assert ok is False
        assert "不匹配" in msg

    def test_correct_token_passes(self):
        srv = GuardianMCPServer(
            config={"mcp": {"enable_write_tools": True}},
        )
        srv._transport = "sse"
        srv.write_token = "correct"
        ok, msg = srv._authorize("trigger_recovery", token="correct")
        assert ok is True


# ---------------------------------------------------------------------------
# 工具调用：只读
# ---------------------------------------------------------------------------

class TestReadonlyTools:
    def test_get_guardian_mode(self, server):
        result = server.call_tool("get_guardian_mode", {})
        data = json.loads(result)
        assert data["mode"] in ("standalone", "mcp_delegated")

    def test_get_contract_status_empty(self, server):
        result = server.call_tool("get_contract_status", {})
        data = json.loads(result)
        assert "error" in data  # task_contract 未绑定

    def test_get_anomaly_history_empty(self, server):
        result = server.call_tool("get_anomaly_history", {})
        data = json.loads(result)
        assert data == []  # monitor 未绑定 → 空列表

    def test_get_recovery_history_empty(self, server):
        result = server.call_tool("get_recovery_history", {})
        data = json.loads(result)
        assert data == []

    def test_get_agent_decision_log_empty(self, server):
        result = server.call_tool("get_agent_decision_log", {})
        data = json.loads(result)
        assert data == []

    def test_unknown_tool(self, server):
        result = server.call_tool("nonexistent_tool", {})
        data = json.loads(result)
        assert "未知" in data["error"]

    def test_get_import_format(self, server):
        result = server.call_tool("get_import_format", {})
        data = json.loads(result)
        assert "meta" in data
        assert "submit_import" in data

    def test_inspect_source_path_blocked(self, server):
        # /nonexistent 路径不在 state_dir 内 → 被安全限制拒绝
        result = server.call_tool("inspect_source", {"file_path": "/nonexistent/file.csv"})
        data = json.loads(result)
        assert "超出" in data.get("error", "")


# ---------------------------------------------------------------------------
# 工具调用：写（stdio 模式自动放行）
# ---------------------------------------------------------------------------

class TestWriteToolsStdio:
    def test_trigger_recovery_no_watchdog(self, server):
        result = server.call_tool("trigger_recovery", {"request_id": "t1"})
        data = json.loads(result)
        assert "watchdog 未绑定" in data.get("error", "")

    def test_stop_training_no_watchdog(self, server):
        result = server.call_tool("stop_training", {"request_id": "t1"})
        data = json.loads(result)
        assert "watchdog 未绑定" in data.get("error", "")

    def test_idempotency_dedup(self, server):
        result1 = server.call_tool("stop_training", {"request_id": "dup-1"})
        result2 = server.call_tool("stop_training", {"request_id": "dup-1"})
        data2 = json.loads(result2)
        assert data2.get("deduplicated") is True

    def test_approve_proposal_no_contract(self, server):
        result = server.call_tool("approve_contract_proposal", {"proposal_id": "p1"})
        data = json.loads(result)
        assert "未绑定" in data.get("error", "")


# ---------------------------------------------------------------------------
# 训练阶段保护
# ---------------------------------------------------------------------------

class TestPostTrainingGate:
    @pytest.fixture
    def srv_with_write(self):
        srv = GuardianMCPServer(
            config={
                "project": {"log_dir": str(Path(tempfile.mkdtemp()) / "logs")},
                "mcp": {"enable_write_tools": True},
            },
            mode="shared",
        )
        srv._transport = "stdio"
        srv.__init_handlers__()
        return srv

    def test_training_active_blocks_post_training_tool(self, srv_with_write):
        srv_with_write._training_active = True
        result = srv_with_write.call_tool("run_visualization",
                                          {"model_entry": "foo:bar"})
        data = json.loads(result)
        assert "训练结束后" in data.get("error", "")

    def test_training_finished_allows_post_training_tool(self, srv_with_write):
        srv_with_write._training_active = False
        result = srv_with_write.call_tool("run_visualization",
                                          {"model_entry": "foo:bar"})
        data = json.loads(result)
        # 会因为 model_entry 不可导入而报错，但不是 "训练中" 错误
        assert "训练结束后" not in data.get("error", "")


# ---------------------------------------------------------------------------
# 双模式
# ---------------------------------------------------------------------------

class TestDualMode:
    def test_initial_mode_standalone(self, server):
        data = json.loads(server.call_tool("get_guardian_mode", {}))
        assert data["mode"] == "standalone"

    def test_on_client_connect_switches_mode(self, server):
        server.on_client_connect()
        data = json.loads(server.call_tool("get_guardian_mode", {}))
        assert data["mode"] == "mcp"  # GuardianMode.MCP_DELEGATED.value

    def test_on_client_disconnect_restores(self, server):
        server.on_client_connect()
        server.on_client_disconnect()
        data = json.loads(server.call_tool("get_guardian_mode", {}))
        assert data["mode"] == "standalone"


# ---------------------------------------------------------------------------
# inspect_source 路径限制
# ---------------------------------------------------------------------------

class TestInspectSourceSecurity:
    def test_rejects_path_outside_state_dir(self, server):
        result = server.call_tool("inspect_source",
                                  {"file_path": "/etc/passwd"})
        data = json.loads(result)
        assert "超出" in data.get("error", "")

    def test_rejects_parent_directory_traversal(self, server):
        result = server.call_tool("inspect_source",
                                  {"file_path": "../../../etc/passwd"})
        data = json.loads(result)
        assert "超出" in data.get("error", "")


# ---------------------------------------------------------------------------
# restart_with_params action enum
# ---------------------------------------------------------------------------

class TestRestartWithParams:
    def test_schema_has_action_enum(self):
        tool = [t for t in WRITE_TOOLS if t["name"] == "restart_with_params"][0]
        action_schema = tool["inputSchema"]["properties"]["action"]
        assert "enum" in action_schema
        assert "reduce_batch" in action_schema["enum"]
        assert "restart_with_lower_lr" in action_schema["enum"]
        assert "enable_grad_accum" in action_schema["enum"]
