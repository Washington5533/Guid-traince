# Training Guardian Agent · 训练守护智能体

[![CI](https://github.com/user/guarftrain/actions/workflows/ci.yml/badge.svg)](https://github.com/user/guarftrain/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

> **一行命令，训练脚本零行改动，获得完整守护能力。**
>
> *One command. Zero changes to your training script. Full guardian capabilities.*

```bash
python run.py watch --agent -- python train.py --epochs 20
```

---

## What does it do? · 它做什么？

| Phase · 阶段 | Capability · 能力 | How · 方式 |
|-------------|-------------------|------------|
| 训练前 Pre-flight | GPU 显存预估 + batch 推荐 | `python run.py preflight` |
| 训练中 During | GPU+Loss 监控告警 / 崩溃自动恢复 / LLM 决策 | `python run.py watch` |
| 训练后 Post | 摘要+AI 解读 / Checkpoint 分析 / 模型可视化 / 推理 | `python run.py summarize` |
| 跨实验 Cross | 自然语言查询 / 实验对比 / 数据导入 | `python run.py query "best lr?"` |
| 外部接入 External | MCP 28 工具 + Dashboard 面板 | `python run.py start` |

## Quick Start · 快速开始

### Install · 安装

```bash
git clone https://github.com/user/guarftrain.git
cd guarftrain
pip install -r requirements-core.txt    # core (required)
pip install -r requirements-mcp.txt     # MCP access (optional)
pip install -r requirements-dashboard.txt  # web dashboard (optional)
```

### One-liner · 一行命令

```bash
# Pure rule-based guardian (no LLM, zero external dependencies)
python run.py watch -- python train.py --epochs 20

# With LLM agent (intelligent decisions + AI report)
python run.py watch --agent -- python train.py --epochs 20

# With Dashboard + MCP (web panel + external agent access)
python run.py watch --agent --with-mcp --with-dashboard -- python train.py --epochs 20
```

### What does the training script need? · 训练脚本要满足什么？

Four contracts (just good training hygiene):

1. `--resume` / `--ckpt` flags for checkpoint resumption
2. `cp_{epoch}/model.pth` with `epoch/model_state_dict/optimizer_state_dict`
3. Structured logging: `epoch {n} loss {v} val_acc {v} lr {v}`
4. Importable: `train:build_model` / `train:get_dataloaders`

Missing any one? Only the corresponding capability is disabled — training still runs.

四项契约（写好训练脚本的基本功），缺任一项只关对应能力，不阻断训练。

## Architecture · 架构

```
┌─ Guardian Process (sidecar) ────────────────────────────────────┐
│                                                                  │
│  CLI (run.py) ──→ 14 subcommands                                 │
│  ├─ watch ──→ Watchdog: Popen + crash recovery + CLI rewrite    │
│  │             └─ Monitor: log tail + GPU poll + anomaly detect  │
│  │                  └─ AgentAdvisor: LLM decide → intervene       │
│  ├─ serve ──→ MCP Server: 27 tools (18 read + 9 write)          │
│  ├─ start ──→ Dashboard + MCP one-click                        │
│  └─ experiments / query / compare ──→ Cross-experiment analysis  │
│                                                                  │
│  Decision Layers · 决策分层:                                      │
│  ┌─ Contract (hard boundary, human-defined)                     │
│  ├─ Agent (LLM, optional, within action space)                  │
│  ├─ Rules (deterministic, always-on fallback)                   │
│  └─ MCP (external agent access, dual-mode delegation)           │
│                                                                  │
│  Training Process: python train.py (0 changes required)          │
└──────────────────────────────────────────────────────────────────┘
```

## CLI Commands · 命令速查

| Command | Description |
|---------|-------------|
| `watch` | Guard any training command |
| `start` | Dashboard + MCP one-click launch |
| `serve` | Standalone MCP server |
| `contract check` | Validate training script contract |
| `preflight` | GPU memory estimate + batch recommendation |
| `analyze` | Scan existing checkpoints |
| `experiments` | List all historical experiments |
| `query` | Natural language query ("best lr?") |
| `compare` | Compare two experiments |
| `visualize` | Model structure visualization (D3.js HTML) |
| `infer` | Run inference with checkpoint |
| `gallery` | Image filtering + selection |
| `dashboard` | Web control panel (standalone) |
| `project init` | Auto-detect project structure |

## MCP Tools · MCP 工具

**18 read-only** (always available, no auth):

`get_training_status` · `get_metrics_history` · `list_checkpoints` · `compare_checkpoints` · `get_anomaly_history` · `get_recovery_history` · `get_summary` · `get_agent_decision_log` · `get_contract_status` · `list_contract_proposals` · `list_experiments` · `query_experiment` · `compare_experiments` · `get_model_structure` · `get_guardian_mode` · `get_gallery_config` · `get_import_format` · `inspect_source`

**9 write** (token auth + training-phase gating):

`trigger_recovery` · `restart_with_params` · `stop_training` · `approve_contract_proposal` · `reject_contract_proposal` · `run_visualization` · `set_gallery_config` · `run_inference` · `submit_import`

→ Full API reference: [MCP_API_REFERENCE.md](MCP_API_REFERENCE.md)

## Configuration · 配置

Three layers, zero secrets in YAML:

```
DEFAULTS  <  guardian.yaml  <  GUARDIAN_* env vars  <  CLI flags
```

```yaml
# configs/guardian.yaml — only override what you need
watchdog:
  max_retries: 3
monitor:
  poll_interval: 5
mcp:
  enable_write_tools: true
```

```bash
# Env override: GUARDIAN_ + section + __ + key
export GUARDIAN_WATCHDOG__MAX_RETRIES=5
export GUARDIAN_MCP_TOKEN=your-secret   # write tool auth
```

## Project Status · 项目状态

| Metric | Value |
|--------|-------|
| Version | 0.1.0 |
| Modules | 16 (cp_1 ~ cp_16) |
| Production code | ~10,500 lines |
| Tests | 221 (CI on push) |
| MCP tools | 27 (18 read + 9 write) |
| CLI commands | 14 |
| Test coverage | ~13% (core paths: 100%) |
| Python | 3.10+ |

## Docs · 文档索引

| Document | Content |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture & workflow (ZH) |
| [DEPLOYMENT.md](DEPLOYMENT.md) | User manual (ZH) |
| [MCP.md](MCP.md) | MCP integration guide (ZH) |
| [MCP_API_REFERENCE.md](MCP_API_REFERENCE.md) | 27-tool API reference (ZH) |
| [MCP_QUICKSTART.md](MCP_QUICKSTART.md) | 5-minute MCP onboarding (ZH) |
| [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) | Per-module completion report (ZH) |
| [checkpoint/INDEX.md](checkpoint/INDEX.md) | Module index cp_1~cp_16 (ZH) |
| [checkpoint/cp_10.md](checkpoint/cp_10.md) | MCP layer design doc (ZH) |

## License

MIT
