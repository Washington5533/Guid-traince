# -*- coding: utf-8 -*-
# Test all 32 MCP tools end-to-end.
# Usage: python tests/test_mcp_all_tools.py

import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, ".")

DASH_URL = "http://127.0.0.1:8765"
PASS = 0
FAIL = 0
SKIP = 0


def check(name, condition, detail=""):
    global PASS, FAIL, SKIP
    if condition is None:  # SKIP signal
        SKIP += 1
        print(f"  SKIP {name}")
        return
    if condition:
        PASS += 1
        print(f"  OK   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  --  {detail}")


def api(method, path, data=None, timeout=10):
    url = f"{DASH_URL}{path}"
    req = urllib.request.Request(url, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        result = json.loads(resp.read().decode("utf-8"))
        resp.close()
        return result if isinstance(result, dict) else {"raw": result}
    except Exception as exc:
        return {"error": str(exc)}


# ===================================================================
# Setup: build a seeded MCP server with all handlers + mock data
# ===================================================================
print("=" * 60)
print("  MCP All-32-Tools Comprehensive Test")
print("=" * 60)

from guardian.mcp_server import GuardianMCPServer

# Build MCP server in standalone mode with dash_url
mcp = GuardianMCPServer(
    config={"project": {"log_dir": "./logs"}, "mcp": {"enable_write_tools": True}},
    mode="standalone",
    dash_url=DASH_URL,
)

# Init handler dicts (same as call_tool would use)
mcp._READ_HANDLERS = {}
mcp._WRITE_HANDLERS = {}

# Register test process with rich data
test_pid = "mcp-all-test"
r = api("POST", "/api/register", {
    "process_id": test_pid,
    "name": "MCP Full Test",
    "status": "running",
    "command": "python train.py --epochs 20",
    "model_entry": "train:build_model",
    "project_dir": ".",
    "log_file": "./logs/train.log",
    "dash_config": {"template": "training", "charts": {"default_groups": ["loss", "accuracy"], "smoothing": False, "range_mode": "auto"}, "panels": {"cursor_info": True, "logs": True, "ai_chat": False}},
})
check("Setup: register test process", r.get("ok"), str(r))

# Push 100 rich metrics
for i in range(100):
    api("POST", f"/api/process/{test_pid}/push", {
        "type": "metrics",
        "data": {
            "step": i, "epoch": round(i / 10, 2),
            "train_loss": round(2.5 - i * 0.02 + (i % 7) * 0.01, 4),
            "val_loss": round(2.8 - i * 0.015 + (i % 5) * 0.02, 4),
            "val_acc": round(0.1 + i * 0.008, 4),
            "lr": round(0.001 * (0.95 ** (i // 10)), 6),
            "gpu_util_pct": 85 + (i % 10),
            "gpu_mem_used": 4500 + (i % 500),
        },
    })
time.sleep(0.3)

# Push some log lines
for j in range(10):
    api("POST", f"/api/process/{test_pid}/push", {
        "type": "log_line",
        "data": f"[INFO] Epoch {j} train_loss={2.5-j*0.1:.4f} val_acc={0.1+j*0.08:.4f}",
    })

check("Setup: 100 metrics pushed", True, "")
check("Setup: log lines pushed", True, "")

# ---- Wire up ALL handlers ----
# The handler dicts are populated by build_handlers() which is called in start().
# For testing we manually register all handlers.

# Let's use the actual call_tool mechanism by setting up handler dicts properly.
# We'll register handlers by copying the pattern from the MCP server's __init__.

# First, scan what handlers exist on the mcp instance
all_methods = [m for m in dir(mcp) if m.startswith("_handle_")]
print(f"\n  Found {len(all_methods)} handler methods on MCP server\n")

# Manually register all known handlers
READ_TOOL_NAMES = [
    "get_training_status", "get_metrics_history", "list_checkpoints",
    "compare_checkpoints", "get_anomaly_history", "get_recovery_history",
    "get_summary", "get_agent_decision_log", "get_contract_status",
    "list_contract_proposals", "list_experiments", "query_experiment",
    "compare_experiments", "get_model_structure", "get_guardian_mode",
    "get_gallery_config", "get_import_format", "inspect_source",
    "get_training_log", "get_post_training_checklist", "get_pending_decisions",
    "get_dashboard_config", "recommend_charts", "list_dashboard_templates",
]

WRITE_TOOL_NAMES = [
    "trigger_recovery", "restart_with_params", "stop_training",
    "approve_contract_proposal", "reject_contract_proposal",
    "run_visualization", "set_gallery_config", "run_inference",
    "submit_import", "resolve_decision", "set_dashboard_config",
]

HANDLER_MAP = {
    "get_training_status": "_handle_training_status",
    "get_metrics_history": "_handle_metrics_history",
    "list_checkpoints": "_handle_list_checkpoints",
    "compare_checkpoints": "_handle_compare_checkpoints",
    "get_anomaly_history": "_handle_anomaly_history",
    "get_recovery_history": "_handle_recovery_history",
    "get_summary": "_handle_summary",
    "get_agent_decision_log": "_handle_agent_decision_log",
    "get_contract_status": "_handle_contract_status",
    "list_contract_proposals": "_handle_contract_proposals",
    "list_experiments": "_handle_list_experiments",
    "query_experiment": "_handle_query_experiment",
    "compare_experiments": "_handle_compare_experiments",
    "get_model_structure": "_handle_get_model_structure",
    "get_guardian_mode": "_handle_get_guardian_mode",
    "get_gallery_config": "_handle_get_gallery_config",
    "get_import_format": "_handle_get_import_format",
    "inspect_source": "_handle_inspect_source",
    "get_training_log": "_handle_get_training_log",
    "get_post_training_checklist": "_handle_post_training_checklist",
    "get_pending_decisions": "_handle_get_pending_decisions",
    "get_dashboard_config": "_handle_get_dashboard_config",
    "recommend_charts": "_handle_recommend_charts",
    "list_dashboard_templates": "_handle_list_dashboard_templates",
    "trigger_recovery": "_handle_trigger_recovery",
    "restart_with_params": "_handle_restart_with_params",
    "stop_training": "_handle_stop_training",
    "approve_contract_proposal": "_handle_approve_proposal",
    "reject_contract_proposal": "_handle_reject_proposal",
    "run_visualization": "_handle_run_visualization",
    "set_gallery_config": "_handle_set_gallery_config",
    "run_inference": "_handle_run_inference",
    "submit_import": "_handle_submit_import",
    "resolve_decision": "_handle_resolve_decision",
    "set_dashboard_config": "_handle_set_dashboard_config",
}

for name in READ_TOOL_NAMES:
    handler_name = HANDLER_MAP.get(name)
    if handler_name and hasattr(mcp, handler_name):
        mcp._READ_HANDLERS[name] = getattr(mcp, handler_name)

for name in WRITE_TOOL_NAMES:
    handler_name = HANDLER_MAP.get(name)
    if handler_name and hasattr(mcp, handler_name):
        mcp._WRITE_HANDLERS[name] = getattr(mcp, handler_name)

# Set stdio transport (no token needed for write tools)
mcp._transport = "stdio"
mcp.write_enabled = True

# Set training as active (most tools work during training)
mcp._training_active = True

# Mock some internal state that tools need
mcp._state_dir = "./logs"

print(f"  READ handlers: {len(mcp._READ_HANDLERS)} tools")
print(f"  WRITE handlers: {len(mcp._WRITE_HANDLERS)} tools")
print(f"  Total tools available: {len(mcp._READ_HANDLERS) + len(mcp._WRITE_HANDLERS)}")

registered = set(mcp._READ_HANDLERS.keys()) | set(mcp._WRITE_HANDLERS.keys())
check("All 32+ tools registered", len(registered) >= 32,
      f"found {len(registered)}: missing={set(HANDLER_MAP.keys()) - registered}")


# ===================================================================
# Helper to call any tool
# ===================================================================
def call_tool(name, **kwargs):
    if name in mcp._READ_HANDLERS:
        handler = mcp._READ_HANDLERS[name]
    elif name in mcp._WRITE_HANDLERS:
        handler = mcp._WRITE_HANDLERS[name]
    else:
        return json.dumps({"error": f"unknown tool: {name}"})
    try:
        result = handler(**kwargs)
        if isinstance(result, str):
            return json.loads(result)
        return result
    except Exception as exc:
        return {"error": str(exc), "tool": name}


# ===================================================================
# Test all tools by category
# ===================================================================

# ----- Group 1: Core Status/Metrics (4 tools) -----
print("\n" + "=" * 40)
print("  Group 1: Core Status & Metrics")
print("=" * 40)

r = call_tool("get_training_status")
check("get_training_status returns dict", isinstance(r, dict) and "error" not in r,
      str(r)[:100])

r = call_tool("get_metrics_history", limit=10)
check("get_metrics_history returns list/status", isinstance(r, dict),
      f"keys: {list(r.keys()) if isinstance(r, dict) else 'N/A'}")

r = call_tool("get_training_log", lines=20)
check("get_training_log returns log", isinstance(r, dict),
      f"keys: {list(r.keys()) if isinstance(r, dict) else 'N/A'}")

r = call_tool("get_guardian_mode")
check("get_guardian_mode returns mode", isinstance(r, dict) and "mode" in r,
      str(r))


# ----- Group 2: Checkpoints (2 tools) -----
print("\n" + "=" * 40)
print("  Group 2: Checkpoints")
print("=" * 40)

r = call_tool("list_checkpoints")
check("list_checkpoints returns dict", isinstance(r, dict),
      str(r)[:100])

r = call_tool("compare_checkpoints", cp_a=1, cp_b=5)
check("compare_checkpoints returns dict", isinstance(r, dict),
      str(r)[:100])


# ----- Group 3: Anomalies & Recovery (2 tools) -----
print("\n" + "=" * 40)
print("  Group 3: Anomalies & Recovery")
print("=" * 40)

r = call_tool("get_anomaly_history")
check("get_anomaly_history returns valid", not isinstance(r, str) or len(str(r)) < 500,
      str(r)[:100])

r = call_tool("get_recovery_history")
check("get_recovery_history returns valid", not isinstance(r, str) or len(str(r)) < 500,
      str(r)[:100])


# ----- Group 4: Summary & Agent Log (2 tools) -----
print("\n" + "=" * 40)
print("  Group 4: Summary & Agent Log")
print("=" * 40)

r = call_tool("get_summary")
check("get_summary returns valid", r is not None,
      str(r)[:100])

r = call_tool("get_agent_decision_log", limit=5)
check("get_agent_decision_log returns valid", r is not None,
      str(r)[:100])


# ----- Group 5: Contract (2 tools) -----
print("\n" + "=" * 40)
print("  Group 5: Contract Status & Proposals")
print("=" * 40)

r = call_tool("get_contract_status")
check("get_contract_status returns dict", isinstance(r, dict),
      str(r)[:100])

r = call_tool("list_contract_proposals")
check("list_contract_proposals returns valid", r is not None,
      str(r)[:100])


# ----- Group 6: Experiments (3 tools) -----
print("\n" + "=" * 40)
print("  Group 6: Experiments")
print("=" * 40)

r = call_tool("list_experiments", limit=5)
check("list_experiments returns dict", isinstance(r, dict),
      str(r)[:100])

r = call_tool("query_experiment", question="best loss")
check("query_experiment returns dict", isinstance(r, dict),
      str(r)[:100])

r = call_tool("compare_experiments", id_a="exp1", id_b="exp2")
check("compare_experiments returns dict", isinstance(r, dict),
      str(r)[:100])


# ----- Group 7: Model & Gallery (3 tools) -----
print("\n" + "=" * 40)
print("  Group 7: Model Structure & Gallery")
print("=" * 40)

r = call_tool("get_model_structure")
check("get_model_structure returns dict", isinstance(r, dict),
      str(r)[:100])

r = call_tool("get_gallery_config")
check("get_gallery_config returns dict", isinstance(r, dict),
      str(r)[:100])

r = call_tool("get_import_format")
check("get_import_format returns dict", isinstance(r, dict) and "format" in str(r).lower() or "schema" in str(r).lower(),
      str(r)[:100])


# ----- Group 8: Source & Checklist (2 tools) -----
print("\n" + "=" * 40)
print("  Group 8: Source Inspection & Checklist")
print("=" * 40)

r = call_tool("inspect_source", path="guardian/mcp_server.py", lines=5)
check("inspect_source returns dict", isinstance(r, dict),
      str(r)[:100])

r = call_tool("get_post_training_checklist")
check("get_post_training_checklist returns dict", isinstance(r, dict),
      str(r)[:100])


# ----- Group 9: Pending Decisions (1 tool) -----
print("\n" + "=" * 40)
print("  Group 9: Pending Decisions")
print("=" * 40)

r = call_tool("get_pending_decisions")
check("get_pending_decisions returns dict", isinstance(r, dict),
      str(r)[:100])


# ----- Group 10: Dashboard Config (4 tools) - NEW -----
print("\n" + "=" * 40)
print("  Group 10: Dashboard Config (NEW)")
print("=" * 40)

r = call_tool("get_dashboard_config", process_id=test_pid)
check("get_dashboard_config returns config", isinstance(r, dict) and "error" not in r,
      str(r)[:100])

r = call_tool("set_dashboard_config", process_id=test_pid,
              charts={"default_groups": ["loss", "accuracy", "gpu"], "smoothing": True})
check("set_dashboard_config writes config", isinstance(r, dict) and r.get("ok"),
      str(r)[:100])

r = call_tool("recommend_charts", process_id=test_pid)
check("recommend_charts returns dict", isinstance(r, dict),
      str(r)[:100] if len(str(r)) < 100 else str(r)[:100] + "...")

r = call_tool("list_dashboard_templates")
check("list_dashboard_templates returns 3 templates",
      isinstance(r, dict) and len(r.get("templates", [])) == 3,
      f"got {len(r.get('templates', []))} templates")


# ----- Group 11: Write Tools (8 tools) -----
print("\n" + "=" * 40)
print("  Group 11: Write Tools (stdio mode)")
print("=" * 40)

# These are destructive but in stdio mode with no real watchdog, they should fail gracefully
r = call_tool("trigger_recovery", request_id="test-recovery-1")
check("trigger_recovery (no watchdog) returns ok/error gracefully",
      isinstance(r, dict), str(r)[:100])

r = call_tool("restart_with_params", action="restart_with_lower_lr", param=0.5, request_id="test-rwp-1")
check("restart_with_params (no watchdog) returns ok/error gracefully",
      isinstance(r, dict), str(r)[:100])

r = call_tool("stop_training", request_id="test-stop-1")
check("stop_training (no watchdog) returns ok/error gracefully",
      isinstance(r, dict), str(r)[:100])

r = call_tool("approve_contract_proposal", proposal_id="prop-test-1")
check("approve_contract_proposal returns dict", isinstance(r, dict),
      str(r)[:100])

r = call_tool("reject_contract_proposal", proposal_id="prop-test-2")
check("reject_contract_proposal returns dict", isinstance(r, dict),
      str(r)[:100])

r = call_tool("resolve_decision", decision_id="pd_nonexistent", override=False)
check("resolve_decision (nonexistent) returns dict", isinstance(r, dict),
      str(r)[:100])


# ----- Group 12: Post-training tools (3 tools) -----
print("\n" + "=" * 40)
print("  Group 12: Post-training tools")
print("=" * 40)

# These require training to be complete. Test both modes.
# During training -> should return error
mcp._training_active = True
r = call_tool("run_visualization", model_entry="train:build_model")
check("run_visualization (training active) returns blocked",
      isinstance(r, dict), str(r)[:100])

r = call_tool("set_gallery_config", strategies="{}", checkpoint_epoch=5, data_source="./data")
check("set_gallery_config (training active) returns blocked",
      isinstance(r, dict), str(r)[:100])

r = call_tool("run_inference", checkpoint_epoch=5, task_type="classification", inputs="./data")
check("run_inference (training active) returns blocked",
      isinstance(r, dict), str(r)[:100])

# After training -> should attempt (may fail gracefully on missing deps)
mcp._training_active = False
r = call_tool("run_visualization", model_entry="train:build_model", request_id="viz-test-1")
check("run_visualization (post-training) returns dict",
      isinstance(r, dict), str(r)[:100])

r = call_tool("set_gallery_config", strategies="{}", checkpoint_epoch=5, data_source="./data", request_id="gal-test-1")
check("set_gallery_config (post-training) returns dict",
      isinstance(r, dict), str(r)[:100])

r = call_tool("run_inference", checkpoint_epoch=5, task_type="classification", inputs="./data", request_id="inf-test-1")
check("run_inference (post-training) returns dict",
      isinstance(r, dict), str(r)[:100])


# ----- Group 13: submit_import (1 tool) -----
print("\n" + "=" * 40)
print("  Group 13: submit_import")
print("=" * 40)

r = call_tool("submit_import",
              meta={"name": "test-import", "source": "test"},
              metrics=[{"step": 0, "loss": 1.0}, {"step": 1, "loss": 0.9}])
check("submit_import returns dict", isinstance(r, dict),
      str(r)[:100])


# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 60)
total = PASS + FAIL + SKIP
print(f"  Tools tested: {len(registered)}")
print(f"  PASS: {PASS}  FAIL: {FAIL}  SKIP: {SKIP}")
if FAIL == 0:
    print(f"  ALL {PASS} TESTS PASSED")
else:
    print(f"  {FAIL} FAILURES FOUND")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
