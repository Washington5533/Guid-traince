# Training Guardian Agent · 训练守护智能体

[![PyPI](https://img.shields.io/pypi/v/guarftrain?color=blue)](https://pypi.org/project/guarftrain/)
[![CI](https://github.com/Washington5533/guarftrain/actions/workflows/ci.yml/badge.svg)](https://github.com/Washington5533/guarftrain/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
[![Streamlit Demo](https://img.shields.io/badge/demo-streamlit-red)](https://guarftrain-azvjiidegvdmnnkhczmfq2.streamlit.app/)

> **一行命令，训练脚本零行改动，获得完整守护能力。**
>
> *One command. Zero changes to your training script. Full guardian capabilities.*

```bash
guarftrain init && guarftrain watch -- python train.py --epochs 20
```

---

## What's New in v0.3.0

| Feature | Description |
|---------|-------------|
| 架构分析 (Arch Analysis) | D3 treemap + backbone 可视化，FLOPs/参数量/瓶颈层检测，参考 archify 设计 |
| 远程通信 (Remote Server) | 算力服务器端 FastAPI 服务，PC Dashboard 远程连接，鉴权 token |
| Sub-agent 自主决策 | `--autonomy supervised/auto/full`，自主调整参数/干预训练 |
| DSH Web GUI Plugin | DeepSeek Harness 侧栏面板，实时 metrics/GPU/anomalies/decisions/architecture/history（[插件文档](dsh-plugin/dsh-client-ui-training-guardian/README.zh.md)） |
| CPU 模式兼容 | 无 GPU 时自动降级，训练曲线正常显示，GPU 面板提示不可用 |
| PyTorch >= 1.13 支持 | resource_estimator 回退兼容 PyTorch 1.x |
| MCP 工具扩展 | +1 `analyze_architecture` 工具（共 36 个） |
| Dashboard 架构分析标签 | 独立「架构分析」标签页，treemap/backbone 双视图 |

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
| 训练中 During | GPU+Loss 监控告警 / 崩溃自动恢复 / LLM 决策 / Sub-agent 自主干预 | `guardian watch` |
| 训练后 Post | 摘要+AI 解读 / Checkpoint 分析 / 模型可视化 / 架构分析 | `guarftrain summarize` |
| 跨实验 Cross | 自然语言查询 / 实验对比 / 数据导入 | `guarftrain query "best lr?"` |
| 外部接入 External | MCP 36 工具 + Dashboard 远程配置 + Agent 图表推荐 + 远程通信 | `guarftrain start` |

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

Four contracts (script interface agreements). Each one gates a capability — missing one disables only that feature, training still runs normally.

1. `--resume` / `--ckpt` flags for checkpoint resumption → enables crash recovery + restart-based interventions
2. `cp_{epoch}/model.pth` with `epoch/model_state_dict/optimizer_state_dict` → enables checkpoint analysis + post-training tools
3. Structured logging: `epoch {n} loss {v} val_acc {v} lr {v}` → enables loss anomaly detection + progress monitoring
4. Importable entry: `train:build_model` / `train:get_dataloaders` → enables preflight resource estimation + model visualization + inference

Missing any one? Only the corresponding capability is disabled — training still runs.

四项契约（训练脚本的接口约定），每一项控制一个能力——缺任一项只关闭对应能力，不阻断训练。`guarftrain init` 会自动扫描你的脚本，逐项报告开启/降级状态。

## Architecture · 架构

```
┌─ Guardian Process (sidecar) ────────────────────────────────────┐
│                                                                  │
│  CLI (guarftrain) ──→ 18 subcommands                              │
│  ├─ watch ──→ Watchdog: Popen + crash recovery + CLI rewrite    │
│  │             └─ Monitor: log tail + GPU poll + anomaly detect  │
│  │                  └─ AgentAdvisor: LLM decide → intervene       │
│  │                  └─ Sub-agent: --autonomy (supervised/auto/full) │
│  ├─ remote ──→ FastAPI 远程通信服务（算力服务器端）                │
│  ├─ serve ──→ MCP Server: 36 tools (25 read + 11 write)          │
│  ├─ start ──→ Dashboard + MCP one-click                        │
│  └─ experiments / query / compare ──→ Cross-experiment analysis  │
│                                                                  │
│  Decision Layers · 决策分层:                                      │
│  ┌─ Contract (hard boundary, human-defined)                     │
│  ├─ Agent (LLM, optional, within action space)                  │
│  ├─ Sub-agent (autonomous, --autonomy supervised/auto/full)     │
│  ├─ Rules (deterministic, always-on fallback)                   │
│  ├─ MCP (external agent access, dual-mode delegation)           │
│  └─ Dashboard (remote config, dirty-flag user protection)       │
│                                                                  │
│  Architecture Analysis · 架构分析:                                │
│  └─ ArchAnalyzer: forward hooks → FLOPs → tree → D3 render     │
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
| `remote` | Start remote communication server (compute server side) |
| `contract check` | Validate training script contract |
| `preflight` | GPU memory estimate + batch recommendation |
| `analyze` | Scan existing checkpoints |
| `analyze_architecture` | Analyze model architecture (D3 treemap/backbone) |
| `experiments` | List all historical experiments |
| `query` | Natural language query ("best lr?") |
| `compare` | Compare two experiments |
| `visualize` | Model structure visualization (D3.js HTML) |
| `infer` | Run inference with checkpoint |
| `gallery` | Image filtering + selection |
| `dashboard` | Web control panel (standalone) |
| `project` | Project context management (init/show/scan/fill) |

## MCP Tools · MCP 工具

**25 read-only** (always available, no auth):

`get_training_status` · `get_metrics_history` · `list_checkpoints` · `compare_checkpoints` · `get_anomaly_history` · `get_recovery_history` · `get_summary` · `get_agent_decision_log` · `get_contract_status` · `list_contract_proposals` · `list_experiments` · `query_experiment` · `compare_experiments` · `get_model_structure` · `analyze_architecture` · `get_guardian_mode` · `get_gallery_config` · `get_import_format` · `inspect_source` · `get_training_log` · `get_post_training_checklist` · `get_pending_decisions` · `get_dashboard_config` · `recommend_charts` · `list_dashboard_templates`

**11 write** (token auth + training-phase gating):

`trigger_recovery` · `restart_with_params` · `stop_training` · `approve_contract_proposal` · `reject_contract_proposal` · `run_visualization` · `set_gallery_config` · `run_inference` · `submit_import` · `resolve_decision` · `set_dashboard_config`

→ Full API reference: [docs/MCP_API_REFERENCE.md](docs/MCP_API_REFERENCE.md)

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

## DSH Web GUI Plugin · DSH 插件

配套 DSH Web GUI 插件 `@rrrelink/dsh-client-ui-training-guardian`，在 DSH 侧栏提供六标签页的 Training Guardian 面板（概览/设备/异常/决策/架构/历史），通过 SSE + REST 消费 `guarftrain remote` 服务。

```bash
# 安装插件（profile 目录 ~/.dsh/profiles/web）
dsh plugin add @rrrelink/dsh-client-ui-training-guardian --profile web

# 训练机侧启动数据源
guarftrain remote --port 8765
guarftrain watch -- python train.py --epochs 50
```

- 源码：[dsh-plugin/dsh-client-ui-training-guardian](dsh-plugin/dsh-client-ui-training-guardian)
- 完整使用说明书：[README.zh.md](dsh-plugin/dsh-client-ui-training-guardian/README.zh.md) / [README.md](dsh-plugin/dsh-client-ui-training-guardian/README.md)
- 插件镜像仓库：<https://github.com/Washington5533/Guid-traince>

## Project Status · 项目状态

| Metric | Value |
|--------|-------|
| Version | 0.3.0 |
| Modules | 21 (cp_1 ~ cp_21) |
| Production code | ~12,500 lines |
| Tests | 266 (CI on push) |
| MCP tools | 36 (25 read + 11 write) |
| CLI commands | 18 |
| Test coverage | ~13% (core paths: 100%) |
| Python | 3.10+ |

## Docs · 文档索引

| Document | Content |
|----------|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Architecture & workflow (ZH) |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | User manual (ZH) |
| [docs/MCP.md](docs/MCP.md) | MCP integration guide (ZH) |
| [docs/MCP_API_REFERENCE.md](docs/MCP_API_REFERENCE.md) | 36-tool API reference (ZH) |
| [docs/MCP_QUICKSTART.md](docs/MCP_QUICKSTART.md) | 5-minute MCP onboarding (ZH) |
| [docs/IMPLEMENTATION_REPORT.md](docs/IMPLEMENTATION_REPORT.md) | Per-module completion report (ZH) |
| [dsh-plugin/…/README.zh.md](dsh-plugin/dsh-client-ui-training-guardian/README.zh.md) | DSH plugin user manual (ZH/EN) |

## License

MIT
