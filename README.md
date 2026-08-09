# Training Guardian Agent · 训练守护智能体

[![CI](https://github.com/Washington5533/guarftrain/actions/workflows/ci.yml/badge.svg)](https://github.com/Washington5533/guarftrain/actions/workflows/ci.yml)
![Version](https://img.shields.io/badge/version-0.2.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

> **一行命令，训练脚本零行改动，获得完整守护能力。**
>
> *One command. Zero changes to your training script. Full guardian capabilities.*

```bash
guarftrain init && guarftrain watch -- python train.py --epochs 20
```

---

## What's New in v0.2.0

| Feature | Description |
|---------|-------------|
| `guarftrain` CLI | `pip install` 后全局可用，替换旧 `python run.py` |
| `guarftrain init` | 自动扫描训练脚本，生成 contract.yaml |
| `guarftrain check` | 环境自检：Python/GPU/依赖/项目结构 |
| Dashboard 远程配置 | 外部 Agent 通过 MCP 控制 Dashboard 图表/面板，用户操作受 dirty flag 保护 |
| Agent 图表推荐 | `chart_selection` 决策点：Agent 分析训练状态，推荐应关注的指标组 |
| MCP 委托模式 | 外部 Claude Code 连接时内置 Agent 进入 provisional 模式，决策可被覆盖 |
| 增量图表更新 | Dashboard 实时推送图表数据，不再全量重建 |
| 依赖瘦身 | 核心安装 ~2MB，torch/anthropic 按需安装 |

---

## What does it do? · 它做什么？

| Phase · 阶段 | Capability · 能力 | How · 方式 |
|-------------|-------------------|------------|
| 训练前 Pre-flight | GPU 显存预估 + batch 推荐 | `guarftrain preflight` |
| 训练中 During | GPU+Loss 监控告警 / 崩溃自动恢复 / LLM 决策 | `guarftrain watch` |
| 训练后 Post | 摘要+AI 解读 / Checkpoint 分析 / 模型可视化 / 推理 | `guarftrain summarize` |
| 跨实验 Cross | 自然语言查询 / 实验对比 / 数据导入 | `guarftrain query "best lr?"` |
| 外部接入 External | MCP 32 工具 + Dashboard 远程配置 + Agent 图表推荐 | `guarftrain start` |

## Quick Start · 快速开始

### Install · 安装

```bash
# 方式 1: pip 安装（推荐，轻量核心 ~2MB，torch 已有不重装）
pip install guarftrain

# 方式 2: 从源码安装
git clone https://github.com/Washington5533/guarftrain.git
cd guarftrain
pip install .

# 按需安装可选组件
pip install guarftrain[agent]       # AI 决策层 (anthropic)
pip install guarftrain[mcp]         # MCP 外部 Agent 接入
pip install guarftrain[dashboard]   # Web 控制面板
pip install guarftrain[full]        # 全部安装
```

### Three steps to guard · 三步守护

```bash
# 1. 初始化项目（自动扫描训练脚本，生成配置）
cd /path/to/your-project
guarftrain init

# 2. 守护训练（纯规则，零外部依赖）
guarftrain watch -- python train.py --epochs 20

# 3. 或启用 AI + Dashboard + MCP
guarftrain watch --agent --with-dashboard --with-mcp -- python train.py --epochs 20
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
│  ├─ serve ──→ MCP Server: 32 tools (22 read + 10 write)          │
│  ├─ start ──→ Dashboard + MCP one-click                        │
│  └─ experiments / query / compare ──→ Cross-experiment analysis  │
│                                                                  │
│  Decision Layers · 决策分层:                                      │
│  ┌─ Contract (hard boundary, human-defined)                     │
│  ├─ Agent (LLM, optional, within action space)                  │
│  ├─ Rules (deterministic, always-on fallback)                   │
│  ├─ MCP (external agent access, dual-mode delegation)           │
│  └─ Dashboard (remote config, dirty-flag user protection)       │
│                                                                  │
│  Training Process: python train.py (0 changes required)          │
└──────────────────────────────────────────────────────────────────┘
```

## CLI Commands · 命令速查

| Command | Description |
|---------|-------------|
| `init` | Auto-detect project + generate contract.yaml |
| `check` | Environment readiness check (deps, GPU, config) |
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
| `project` | Project context management (init/show/scan/fill) |

## MCP Tools · MCP 工具

**24 read-only** (always available, no auth):

`get_training_status` · `get_metrics_history` · `list_checkpoints` · `compare_checkpoints` · `get_anomaly_history` · `get_recovery_history` · `get_summary` · `get_agent_decision_log` · `get_contract_status` · `list_contract_proposals` · `list_experiments` · `query_experiment` · `compare_experiments` · `get_model_structure` · `get_guardian_mode` · `get_gallery_config` · `get_import_format` · `inspect_source` · `get_training_log` · `get_post_training_checklist` · `get_pending_decisions` · `get_dashboard_config` · `recommend_charts` · `list_dashboard_templates`

**11 write** (token auth + training-phase gating):

`trigger_recovery` · `restart_with_params` · `stop_training` · `approve_contract_proposal` · `reject_contract_proposal` · `run_visualization` · `set_gallery_config` · `run_inference` · `submit_import` · `resolve_decision` · `set_dashboard_config`

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
| Version | 0.2.0 |
| Modules | 16 (cp_1 ~ cp_16) |
| Production code | ~10,500 lines |
| Tests | 221 (CI on push) |
| MCP tools | 35 (24 read + 11 write) |
| CLI commands | 16 |
| Test coverage | ~13% (core paths: 100%) |
| Python | 3.10+ |

## Docs · 文档索引

| Document | Content |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | Architecture & workflow (ZH) |
| [DEPLOYMENT.md](DEPLOYMENT.md) | User manual (ZH) |
| [MCP.md](MCP.md) | MCP integration guide (ZH) |
| [MCP_API_REFERENCE.md](MCP_API_REFERENCE.md) | 35-tool API reference (ZH) |
| [MCP_QUICKSTART.md](MCP_QUICKSTART.md) | 5-minute MCP onboarding (ZH) |
| [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md) | Per-module completion report (ZH) |
| [checkpoint/INDEX.md](checkpoint/INDEX.md) | Module index cp_1~cp_16 (ZH) |
| [checkpoint/cp_10.md](checkpoint/cp_10.md) | MCP layer design doc (ZH) |

## License

MIT
