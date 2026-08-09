# -*- coding: utf-8 -*-
"""Dashboard config + MCP tools test."""

import json
import sys
import time
import os
import urllib.request

DASH_URL = "http://127.0.0.1:8765"
PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  OK  {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}  --  {detail}")


def api(method, path, data=None):
    url = f"{DASH_URL}{path}"
    req = urllib.request.Request(url, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode("utf-8")
    try:
        resp = urllib.request.urlopen(req, timeout=10)
        result = json.loads(resp.read().decode("utf-8"))
        resp.close()
        return result if isinstance(result, dict) else {"raw": result}
    except Exception as exc:
        return {"error": str(exc)}


print("=" * 60)
print("  Dashboard Config + MCP Tools Test")
print("=" * 60)

# ===== 1. Register test process =====
print("\n[1] Register test process")
test_pid = "test-mcp-dash"
r = api("POST", "/api/register", {
    "process_id": test_pid,
    "name": "MCP Test",
    "status": "running",
    "command": "python test.py",
    "dash_config": {
        "template": "training",
        "charts": {"default_groups": ["loss", "accuracy"], "smoothing": False, "range_mode": "auto"},
        "panels": {"cursor_info": True, "logs": True, "ai_chat": False},
    },
})
check("register ok", r.get("ok"), str(r))

# Push mock metrics
for i in range(30):
    api("POST", f"/api/process/{test_pid}/push", {
        "type": "metrics",
        "data": {
            "step": i,
            "epoch": round(i / 10, 2),
            "train_loss": round(2.5 - i * 0.05 + (i % 7) * 0.02, 4),
            "val_loss": round(2.8 - i * 0.04 + (i % 5) * 0.03, 4),
            "val_acc": round(0.1 + i * 0.025, 4),
            "lr": round(0.001 * (0.95 ** (i // 5)), 6),
            "gpu_util_pct": 85 + (i % 10),
        },
    })
time.sleep(0.3)

r = api("GET", f"/api/process/{test_pid}")
check("process exists", "error" not in r, str(r.get("error", "")))
check("metrics > 0", r.get("metrics_count", 0) > 0, f"count={r.get('metrics_count', 0)}")

# ===== 2. GET dashboard-config =====
print("\n[2] GET /dashboard-config")
r = api("GET", f"/api/process/{test_pid}/dashboard-config")
check("get config ok", "error" not in r, str(r.get("error", "")))
check("template=training", r.get("template") == "training", f"got: {r.get('template')}")
check("groups=[loss,accuracy]", r.get("charts", {}).get("default_groups") == ["loss", "accuracy"],
      f"got: {r.get('charts', {}).get('default_groups')}")
check("cursor_info=true", r.get("panels", {}).get("cursor_info") is True)
print(f"  config: {json.dumps(r, ensure_ascii=False)}")

# ===== 3. POST dashboard-config =====
print("\n[3] POST /dashboard-config (mimics MCP set_dashboard_config)")
r = api("POST", f"/api/process/{test_pid}/dashboard-config", {
    "charts": {"default_groups": ["loss", "accuracy", "gpu"], "smoothing": True, "range_mode": "last_100"},
    "panels": {"cursor_info": True, "logs": True, "ai_chat": True},
    "template": "training",
    "_source": "mcp_agent",
})
check("set config ok", r.get("ok"), str(r))
check("gpu group added", "gpu" in r.get("config", {}).get("charts", {}).get("default_groups", []))
check("smoothing=true", r.get("config", {}).get("charts", {}).get("smoothing") is True)

r2 = api("GET", f"/api/process/{test_pid}/dashboard-config")
check("GET confirms gpu added", "gpu" in r2.get("charts", {}).get("default_groups", []))

# ===== 4. MCP list_dashboard_templates =====
print("\n[4] MCP list_dashboard_templates")
from guardian.mcp_server import GuardianMCPServer

mcp = GuardianMCPServer(
    config={"project": {"log_dir": "./logs"}},
    mode="standalone",
    dash_url=DASH_URL,
)
mcp._READ_HANDLERS = {
    "get_dashboard_config": mcp._handle_get_dashboard_config,
    "recommend_charts": mcp._handle_recommend_charts,
    "list_dashboard_templates": mcp._handle_list_dashboard_templates,
}
mcp._WRITE_HANDLERS = {"set_dashboard_config": mcp._handle_set_dashboard_config}

r = json.loads(mcp._handle_list_dashboard_templates())
templates = r.get("templates", [])
check("3 templates", len(templates) == 3, f"got {len(templates)}: {[t['name'] for t in templates]}")
check("has training", any(t["name"] == "training" for t in templates))
check("has comparison", any(t["name"] == "comparison" for t in templates))
check("has minimal", any(t["name"] == "minimal" for t in templates))
print(f"  templates: {json.dumps(r, ensure_ascii=False)}")

# ===== 5. MCP get_dashboard_config =====
print("\n[5] MCP get_dashboard_config")
r = json.loads(mcp._handle_get_dashboard_config(process_id=test_pid))
check("MCP get config ok", "error" not in r, str(r.get("error", "")))
check("has gpu group", "gpu" in r.get("charts", {}).get("default_groups", []))
print(f"  MCP result: {json.dumps(r, ensure_ascii=False)}")

# ===== 6. MCP set_dashboard_config (auth) =====
print("\n[6] MCP set_dashboard_config (auth)")
r = json.loads(mcp._handle_set_dashboard_config(
    process_id=test_pid,
    charts={"default_groups": ["loss"], "smoothing": False},
    _token="bad-token",
))
check("no token rejected", "error" in r, str(r))

os.environ["GUARDIAN_MCP_TOKEN"] = "test-token-456"
mcp.write_token = "test-token-456"

r = json.loads(mcp._handle_set_dashboard_config(
    process_id=test_pid,
    charts={"default_groups": ["loss"], "smoothing": False},
    _token="test-token-456",
))
check("with token ok", r.get("ok"), str(r))
check("groups changed to [loss]", r.get("config", {}).get("charts", {}).get("default_groups") == ["loss"])

# Restore
r = json.loads(mcp._handle_set_dashboard_config(
    process_id=test_pid,
    charts={"default_groups": ["loss", "accuracy"], "smoothing": False},
    _token="test-token-456",
))
check("restore ok", r.get("ok"))

# ===== 7. MCP recommend_charts (no agent fallback) =====
print("\n[7] MCP recommend_charts (no agent fallback)")
r = json.loads(mcp._handle_recommend_charts(process_id=test_pid))
if "error" in r:
    check("no agent -> correct fallback", "fallback" in r, str(r))
    print(f"  fallback: {r.get('fallback')}")
else:
    check("agent recommendation", True, "agent available")

# ===== 8. Edge cases =====
print("\n[8] Edge cases")
r = mcp._dash_request("GET", "/api/process/nonexistent/dashboard-config")
check("404 returns error", "error" in r, str(r))

mcp2 = GuardianMCPServer(config={"project": {"log_dir": "./logs"}}, mode="standalone")
r = mcp2._dash_request("GET", "/api/process/x/dashboard-config")
check("no dash_url returns error", "error" in r, str(r))

# ===== Summary =====
print("\n" + "=" * 60)
total = PASS + FAIL
if FAIL == 0:
    print(f"  Result: {PASS}/{total} ALL PASS")
else:
    print(f"  Result: {PASS}/{total} ({FAIL} FAILED)")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
