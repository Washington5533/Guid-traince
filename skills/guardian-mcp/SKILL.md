---
name: guardian-mcp
description: Use Guardian MCP tools to monitor, intervene, and analyze ML training. Covers 35 tools across monitoring, checkpoint analysis, recovery, dashboard config, and cross-experiment queries. Use when the user asks about training status, anomalies, checkpoints, experiments, model analysis, or when Guardian MCP server is connected.
---

# Guardian MCP — Agent Skill

Guardian is a sidecar daemon for ML training. It monitors, detects anomalies, makes LLM-guided recovery decisions, and exposes everything through 35 MCP tools. **Training scripts need zero changes.**

---

## Setup & Configuration

### Prerequisites

- Python 3.10+
- `pip install guarftrain` (or clone + `pip install -e .`)
- MCP SDK: `pip install guarftrain[mcp]`
- For LLM-powered agent decisions: `pip install guarftrain[agent]` + set `ANTHROPIC_API_KEY`

### Start Guardian MCP Server

Three modes, pick one based on your scenario:

**A. Co-located with training (best real-time, recommended)**
```bash
guarftrain watch --with-mcp -- python train.py --epochs 20
```
MCP server runs in the same process as guardian watchdog. Shares memory state directly.

**B. Standalone stdio (Claude Code subprocess model)**
```bash
guarftrain serve --transport stdio
```
Reads checkpoint/log files from disk. Suitable when training is already running separately.

**C. One-click: Dashboard + MCP + browser**
```bash
guarftrain start
```
Launches Dashboard (port 8765) + MCP SSE server (port 8766) + opens browser.

### Connect Your Agent Client

**Claude Code / Qoder — stdio mode:**

Add to `.claude/mcp.json` or `.claude/settings.local.json`:
```json
{
  "mcpServers": {
    "guardian": {
      "command": "guarftrain",
      "args": ["serve", "--transport", "stdio"],
      "cwd": "/path/to/your/project"
    }
  }
}
```

**Remote server — SSE mode:**
```bash
# On remote machine:
guarftrain serve --transport sse --port 8766

# SSH tunnel from local:
ssh -L 8766:127.0.0.1:8766 user@server

# Local client config:
{
  "mcpServers": {
    "guardian-remote": {
      "type": "http",
      "url": "http://127.0.0.1:8766/sse"
    }
  }
}
```

### Enable Write Tools (optional)

Write tools (intervention, restart, stop) require explicit opt-in:

1. In `configs/guardian.yaml`:
```yaml
mcp:
  enabled: true
  enable_write_tools: true
```

2. Set auth token:
```bash
export GUARDIAN_MCP_TOKEN=your-secret-token
```

3. Pass token when calling write tools:
```
restart_with_params(action="restart_with_lower_lr", param=0.5, write_token="your-secret-token")
```

### Verify Connection

After connecting, call:
```
get_guardian_mode()
```
Should return mode info + usage guide. If it returns an error, check that guardian process is running and the project path is correct.

---

## First Step — Always Call This

```
get_guardian_mode()
```

Returns: current mode, write tool status, training phase, recommended workflow. This is your orientation call before doing anything else.

## Tool Map

### Read-Only (24 tools, no auth, always available)

| Category | Tools | When to Use |
|----------|-------|-------------|
| **Status** | `get_training_status`, `get_metrics_history`, `get_training_log` | "How's training going?" |
| **Checkpoints** | `list_checkpoints`, `compare_checkpoints` | "Which epoch is best? How do two compare?" |
| **Anomalies** | `get_anomaly_history`, `get_recovery_history` | "Did anything go wrong? What did guardian do?" |
| **Agent** | `get_agent_decision_log`, `get_contract_status`, `list_contract_proposals`, `get_pending_decisions` | "What did the LLM decide? What's pending my review?" |
| **Summary** | `get_summary`, `get_post_training_checklist` | "Give me the final report / what's next" |
| **Experiments** | `list_experiments`, `query_experiment`, `compare_experiments` | "Compare with last week's run" |
| **Model** | `get_model_structure`, `get_gallery_config` | "Show FLOPs breakdown / gallery config" |
| **Dashboard** | `get_dashboard_config`, `recommend_charts`, `list_dashboard_templates` | "What charts should I show? What's the layout?" |
| **Data Import** | `get_import_format`, `inspect_source` | "Help me import WandB/TensorBoard data" |
| **System** | `get_guardian_mode` | First call — orientation |

### Write (11 tools, require write_token + phase gating)

| Tool | Risk | Phase | Notes |
|------|------|-------|-------|
| `trigger_recovery` | High — kills process | During training | Only with explicit user authorization |
| `restart_with_params` | High — kills process | During training | action: `reduce_batch` / `restart_with_lower_lr` / `enable_grad_accum` |
| `stop_training` | High | During training | Irreversible until manually restarted |
| `resolve_decision` | Medium | During training | Review pending LLM decisions from `get_pending_decisions` |
| `approve/reject_contract_proposal` | Low | Any | Review agent's proposed contract expansions |
| `submit_import` | Low | Any | Import external experiment data |
| `run_visualization` | Low | **Post-training only** | Generate model structure HTML |
| `set_gallery_config` | Low | **Post-training only** | Configure image gallery filtering |
| `run_inference` | Low | **Post-training only** | Run classification/detection/segmentation |
| `set_dashboard_config` | Low | Any | Update Dashboard charts/panels; user changes protected by dirty flag |

## Standard Workflows

### 1. "How's training going?" (Status Check)

```
get_training_status()
  → epoch, loss, accuracy, GPU stats, anomaly count

If anomaly_count > 0:
  get_anomaly_history()        → what happened
  get_agent_decision_log()     → what guardian did about it
```

### 2. "Something's wrong" (Diagnosis)

```
get_training_status()          → snapshot
get_anomaly_history()          → all detected anomalies
get_training_log(lines=100)    → raw log tail for context
get_metrics_history(limit=50)  → recent trend
get_recovery_history()         → any auto-recovery attempts?
```

### 3. "Intervene" (Training Control)

```
# Check what's available:
get_guardian_mode()            → write_tools.enabled?
get_pending_decisions()       → any pending LLM decisions to review?

# If user explicitly asks to adjust:
restart_with_params(
  action="restart_with_lower_lr",
  param=0.5,                   # halve the learning rate
  request_id="<unique>"        # idempotency key
)

# Or review a pending decision:
resolve_decision(
  decision_id="abc123",
  override=false               # approve the LLM's suggestion
)
```

### 4. "Training is done" (Post-Training)

```
get_post_training_checklist()  → what can I do now?
get_summary()                  → structured report + AI interpretation

# Generate deliverables:
run_visualization(model_entry="train:build_model")
run_inference(checkpoint_epoch=17, task_type="classification", inputs="./data/test")

# Optional: configure dashboard for future reference
recommend_charts()             → AI suggests what to visualize
set_dashboard_config(charts=..., template="summary")
```

### 5. "Compare with last run" (Cross-Experiment)

```
list_experiments(limit=10)     → find experiment IDs
query_experiment("highest accuracy last week")
compare_experiments(id_a="exp_20260801", id_b="exp_20260808")
```

### 6. "Import external data" (Data Pipeline)

```
get_import_format()            → JSON Schema spec
inspect_source(file_path="./wandb_export.jsonl", lines=20)  → peek at structure

# Agent transforms data to match format, then:
submit_import(
  meta={"name": "wandb_run_42", "source": "wandb"},
  metrics_path="./transformed.jsonl"
)
```

## Key Rules

1. **Never call write tools without user authorization** — especially `trigger_recovery`, `restart_with_params`, `stop_training`
2. **Post-training tools fail during training** — `run_visualization`, `set_gallery_config`, `run_inference` check `training_phase.active`
3. **Write tools need token** — pass `write_token` parameter matching `GUARDIAN_MCP_TOKEN` env var
4. **Idempotency** — pass `request_id` to write tools; 5-minute dedup window prevents double-actions
5. **Pagination** — `get_metrics_history` defaults to 200 records; use `cursor` for more
6. **Dashboard dirty flag** — user edits to Dashboard are protected; `set_dashboard_config` won't overwrite unsaved user changes
