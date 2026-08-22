# Training Guardian Agent — 使用说明书

## 1. 项目简介

Training Guardian Agent 是一个 **sidecar-first** 的训练守护系统。以独立进程运行在训练脚本之外，训练脚本**零行改动**即可获得完整守护能力。

**三阶段覆盖：**

| 阶段 | 功能 |
|------|------|
| 训练前 | F8 资源预估（显存/batch/时长）、契约校验 |
| 训练中 | F1 GPU+Loss 监控告警、F6 崩溃自动恢复、Agent 智能决策 |
| 训练后 | F9 日志摘要+AI 解读、F2 Checkpoint 分析、F3 图片筛选展示、F7 推理测试 |
| 跨实验 | F4 NL 查询历史实验、F10 模型管线可视化+组件库改进建议 |

## 2. 快速开始

### 2.1 安装

```bash
# 方式 1: pip 安装（推荐）
pip install guarftrain

# 按需安装可选组件
pip install guarftrain[agent]       # AI 决策层
pip install guarftrain[mcp]         # MCP 外部 Agent 接入
pip install guarftrain[dashboard]   # Web 控制面板
pip install guarftrain[full]        # 全部安装

# 方式 2: 从源码安装
git clone https://github.com/Washington5533/guarftrain.git
cd guarftrain
pip install -r requirements-core.txt       # 核心（必需）
pip install -r requirements-mcp.txt        # MCP 接入（可选）
```

### 2.2 一行命令守护训练

```bash
# 纯规则守护（零外部依赖）
guarftrain watch -- python train.py --epochs 20

# Agent 智能决策（需 API key）
guarftrain watch --agent -- python train.py --epochs 20

# Agent + Dashboard 控制面板
guarftrain watch --with-dashboard --agent -- python train.py --epochs 20

# Agent + MCP 外部接入
guarftrain watch --agent --with-mcp -- python train.py --epochs 20

# 带项目配置（自动读取 .guardian-project.yaml 中的路径）
guarftrain watch --with-dashboard --agent \
  --config ../my-project/configs/guardian.yaml \
  -- python ../my-project/train_clip.py --epochs 20
```

### 2.3 训练脚本需要满足什么？

四项契约（详见 `checkpoint/cp_11.md`）——本质就是写好训练脚本的基本功：

1. `--resume` / `--ckpt`：支持断点续训
2. checkpoint 保存为 `cp_{epoch}/model.pth`，含 `epoch/model_state_dict/optimizer_state_dict`
3. 结构化日志：`epoch {n} loss {v} val_acc {v} lr {v}`
4. 可外部 import：`train_clip:build_model` / `train_clip:get_dataloaders`

缺失任一项只关闭对应能力，不阻断启动。

## 3. 项目上下文（路径自适应）

解决跨项目使用时每次都要手写长路径的问题。

### 3.1 初始化

```bash
# 自动扫描项目结构，生成 .guardian-project.yaml
guarftrain project init /path/to/your/project

# AI 补全缺失项（model entry、task type 等）
guarftrain project fill --agent
```

### 3.2 自动发现

执行命令时按以下优先级解析路径：

```
CLI 显式参数  >  .guardian-project.yaml  >  自动扫描目录  >  默认值
```

`.guardian-project.yaml` 示例（自动生成）：

```yaml
project:
  name: clip-pets
  ckpt_dir: C:/Users/wst/Desktop/anytries/deepfucking/checkpoints
  log_dir: C:/Users/wst/Desktop/anytries/deepfucking/logs
  data_dir: C:/Users/wst/Desktop/anytries/deepfucking/data
model:
  entry: train_clip:build_model
  task_type: classification
```

### 3.3 三种使用方式

```bash
# 方式1：在项目目录内运行（自动发现）
cd /path/to/project && python /path/to/guarftrain/run.py experiments

# 方式2：显式指定项目目录
guarftrain experiments --project-dir /path/to/project

# 方式3：手动覆盖（优先级最高）
guarftrain experiments --log-dir /custom/logs --ckpt-dir /custom/checkpoints
```

## 4. 全部命令

```
guarftrain <command> [options]

训练守护：
  watch         守护任意训练命令
  contract      契约校验 (check / review)
  preflight     训练前资源预检
  analyze       独立扫描已有 checkpoint

查询与分析：
  experiments   列出所有历史实验
  query         自然语言查询（"上次 mAP 最高的 lr 是多少"）
  compare       对比两个实验

模型理解：
  visualize     模型管线可视化（结构图 + FLOPs + 瓶颈 + 改进建议）
  infer         模型推理测试（固定脚本，不生成代码）

展示：
  gallery       图片筛选与展示（agent 提议策略 → 确认 → 执行）

工具：
  project       项目上下文管理（init/show/scan/fill）
  serve         独立启动 MCP server
```

## 5. 训练后功能详解

### 5.1 实验查询（F4）

```bash
# 列出所有实验
guarftrain experiments [--log-dir <path>] [--name <prefix>] [--limit 20]

# NL 查询
guarftrain query "最高准确率的实验，lr是多少" [--agent]

# 对比
guarftrain compare exp_a exp_b [--agent]
```

同名实验自动用时间戳去重。`--name` 可手动设置前缀。

### 5.2 模型结构可视化（F10）

```bash
guarftrain visualize --model train_clip:build_model [--agent]
```

输出交互式 HTML（D3.js 可折叠树）：
- 自动折叠同构层（如 12 个相同的 TransformerBlock → ×12）
- 真实 FLOPs 计算（forward hook + dummy input）
- 瓶颈标注（参数占比 >25%）
- 经典组件库匹配改进建议（含代码）
- 点击展开/收缩，悬停显示详情

启用 `--agent` 后：
- AI 分析瓶颈并匹配组件库（SEBlock、Bottleneck、MultiQueryAttention 等 10+ 组件）
- 无匹配组件时 AI 自行编写新的优化方案

### 5.3 推理测试（F7）

```bash
# 自动选 best checkpoint + 自动检测任务类型
guarftrain infer --ckpt 17 [--task classification] [--inputs <path>]

# 在项目目录内不写路径，自动继承 data_dir
cd /path/to/project && python ../guarftrain/run.py infer --ckpt 17
```

固定推理脚本（不生成代码）：
- `scripts/infer_classification.py`
- `scripts/infer_detection.py`
- `scripts/infer_segmentation.py`

### 5.4 图片筛选（F3）

```bash
guarftrain gallery --ckpt 17 [--data <path>] [--agent]
```

交互流程：
```
agent 提议多套筛选策略（汇报精选 / 难样本 / 边界案例）
  → 终端展示 name + rationale + filters
  → 用户确认: [回车]执行 | [NL修正] | export | cancel
  → 执行推理 + 筛选 → 保存结果 JSON + 可选 Streamlit 展示
```

## 6. 接入外部 Agent（MCP）

```bash
# 独立 MCP server
guarftrain serve --transport stdio

# 或在 watch 时后台启动
guarftrain watch --with-mcp -- python train.py
```

MCP 模式下 guardian agent 进入 provisional 模式，外部 Agent 可接管决策。Claude Code 获得全部 35 个工具的读写权限。

> 完整文档：[docs/MCP.md](MCP.md) · [docs/MCP_API_REFERENCE.md](MCP_API_REFERENCE.md) · [docs/MCP_QUICKSTART.md](MCP_QUICKSTART.md)

### 6.1 MCP 工具列表

**只读工具（18 个）：**

| 工具 | 功能 |
|------|------|
| `get_training_status` | 当前 epoch/step、loss、GPU 状态 |
| `get_metrics_history` | 指标时间序列（分页） |
| `list_checkpoints` | checkpoint 列表 + best/top_k |
| `compare_checkpoints` | 对比两个 checkpoint |
| `get_anomaly_history` | 异常事件 + 应对来源 |
| `get_recovery_history` | 重启记录 |
| `get_summary` | 训练摘要 + AI 解读 |
| `get_agent_decision_log` | agent 决策日志 |
| `get_contract_status` | 契约四项状态 |
| `list_contract_proposals` | agent 提议记录 |
| `list_experiments` | 所有历史实验 |
| `query_experiment` | NL 查询实验 |
| `compare_experiments` | 对比两个实验 |
| `get_model_structure` | 模型结构 JSON（节点+边+FLOPs） |
| `get_guardian_mode` | 当前模式（standalone/mcp_delegated） |
| `get_gallery_config` | 图片筛选策略配置 |
| `get_import_format` | 导入格式规范（JSON Schema） |
| `inspect_source` | 采样外部数据文件 |

**受限写工具（10 个，需 write_token + 阶段保护）：**

| 工具 | 功能 | 训练中 |
|------|------|--------|
| `trigger_recovery` | 手动触发恢复 | ✅ |
| `restart_with_params` | 带参重启 | ✅ |
| `stop_training` | 停止训练 | ✅ |
| `trigger_full_validate` | 完整校验 checkpoint | ✅ |
| `approve_contract_proposal` | 批准契约提议 | ✅ |
| `reject_contract_proposal` | 拒绝契约提议 | ✅ |
| `run_visualization` | 生成模型可视化 HTML | ❌ 仅训练后 |
| `set_gallery_config` | 更新筛选策略 | ❌ 仅训练后 |
| `run_inference` | 触发推理 | ❌ 仅训练后 |
| `submit_import` | 导入外部训练数据 | ✅ |

训练中写工具保护：`set_gallery_config` / `run_visualization` / `run_inference` 仅在训练结束后可用。

### 6.2 双模式架构

```
┌─ Standalone ─────────────────────────────────────┐
│ guardian AgentAdvisor 自主决策                    │
│ 训练中：预设动作集，失败回退规则默认                 │
│ 训练后：创造性策略（需用户确认）                    │
├─ MCP Delegated ──────────────────────────────────┤
│ 外部 Claude Code 决策，guardian agent 让位        │
│ guardian 角色：数据提供者 + 安全执行器              │
│ Claude Code 断开 → 自动恢复 standalone            │
└──────────────────────────────────────────────────┘
```

## 7. 配置参考

### 7.1 guardian.yaml（guardian 自身行为）

```yaml
project:
  name: my-experiment
  ckpt_dir: ./checkpoints
  log_dir: ./logs

watchdog:
  max_retries: 3
  restart_delay: 5
  oom_batch_reduce_ratio: 0.5
  min_batch_size: 8

monitor:
  poll_interval: 5
  sliding_window: 50
  loss_spike_ratio: 0.5
  gpu_temp_threshold: 85

agent:
  enabled: false              # --agent 时自动启用
  provider: anthropic
  decision_timeout: 8

mcp:
  enabled: false
  enable_write_tools: false   # 写工具需显式开启 + write_token

contract:
  path: configs/contract.yaml
```

### 7.2 contract.yaml（训练脚本接口面）

```yaml
script_contract:
  resumable:
    entry: cli
    resume_flag: "--resume"
    ckpt_flag: "--ckpt"
  checkpoint_schema:
    required_keys: [epoch, model_state_dict, optimizer_state_dict]
  metrics_channel:
    type: log_file
    path: ../logs/train.log
    log_pattern: "epoch (\\d+) loss ([\\d.]+) val_acc ([\\d.]+) lr ([\\d.e+-]+)"
  buildable_entry:
    model_fn: "train_clip:build_model"
    dataloader_fn: "train_clip:get_dataloaders"
  cli_mappings:
    optimizer.lr: "--lr"
    dataloader.batch_size: "--batch-size"

metric_registry:
  classification:
    - {name: accuracy, direction: max}
```

### 7.3 环境变量覆盖

```bash
# 全大写 + 双下划线 = 嵌套键
GUARDIAN_WATCHDOG__MAX_RETRIES=5
GUARDIAN_AGENT__DECISION_TIMEOUT=12

# Agent API key
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_AUTH_TOKEN=...       # OAuth / 第三方兼容 API
ANTHROPIC_BASE_URL=...         # 自定义 endpoint

# MCP 写工具口令
GUARDIAN_MCP_TOKEN=your-secret
```

## 8. 实战示例：CLIP Linear Probe 训练

```bash
# 1. 准备训练脚本（满足四项契约）
#    参见 deepfucking/train_clip.py

# 2. 初始化项目
guarftrain project init ../deepfucking

# 3. 守护训练 20 epoch
guarftrain watch --agent \
  --config ../deepfucking/configs/guardian.yaml \
  -- python ../deepfucking/train_clip.py --epochs 20

# 4. 查看训练记录
guarftrain experiments --project-dir ../deepfucking --name clip-pets
guarftrain query "最好的epoch" --project-dir ../deepfucking

# 5. 可视化 CLIP 结构
guarftrain visualize --model clip_adapter:build_model_full

# 6. 推理看效果
guarftrain infer --ckpt 17 --project-dir ../deepfucking

# 7. 图片筛选
guarftrain gallery --ckpt 17 --project-dir ../deepfucking --agent
```

## 9. AI 决策边界

```
训练中（F1/F6）：
  规则判定: "是不是异常" / "能不能恢复"（零延迟，确定性）
  Agent 选择: "怎么应对" / "哪种策略"（预设动作集内选，失败回退默认）

训练后（F3/F7/F10）：
  Agent 主导: 创造性定义策略，执行前需用户确认
  F10 提权: AI 分析瓶颈 → 匹配组件库 → 无匹配则自行编写方案
  
MCP 模式：
  外部 Claude Code 决策，guardian agent 让位
  断开 → 无缝恢复自主决策
  
不变式：
  Agent 的自由度永远是人显式授予的
  任何一层失效都退回上一层的确定性行为
```

## 10. 测试

```bash
# 全量测试
python -m pytest tests/ -q

# 分模块
python -m pytest tests/test_experiment_query.py -q
python -m pytest tests/test_model_viz.py -q
python -m pytest tests/test_gallery.py -q
python -m pytest tests/test_inference.py -q
```
