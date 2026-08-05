# Training Guardian Agent — 部署与功能说明书

## 1. 项目简介

Training Guardian Agent（训练守护 agent）是一个 **sidecar-first** 的训练守护系统。它以独立进程运行在训练脚本之外，开发者**不需要改动训练脚本任何一行代码**，即可获得：

- 训练前：显存预估、安全 batch size 推荐、训练时长预测
- 训练中：GPU 监控、loss 异常检测、LLM 智能决策应对、崩溃自动恢复
- 训练后：结构化摘要、AI 自然语言分析报告
- 全程：契约校验、告警推送（终端/Webhook）、MCP 外部接入

## 2. 环境要求

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| Python | ≥3.10 | |
| PyTorch | ≥2.0 | 训练核心 |
| PyYAML | ≥6.0 | 配置解析 |
| psutil | ≥5.9 | 硬件监控（可选，无 GPU 时自动降级） |
| requests | ≥2.28 | Webhook 告警推送（可选） |
| anthropic | ≥0.40 | Agent 决策层（可选，未安装时自动降级为纯规则） |
| mcp | ≥1.0 | MCP 外部接入（可选，单独安装） |

## 3. 安装

### 3.1 克隆仓库

```bash
git clone <repo-url> guarftrain
cd guarftrain
```

### 3.2 安装依赖

```bash
# 核心依赖（规则引擎 + Agent 决策层）
pip install -r requirements-core.txt

# MCP 接入叠加层（可选，任何时间都可以补装，不需要重启训练）
pip install -r requirements-mcp.txt
```

### 3.3 配置 API Key（使用 Agent 决策层时需要）

```bash
# Anthropic 官方 API
export ANTHROPIC_API_KEY=sk-ant-...

# 或第三方 Anthropic 兼容 API（如 DeepSeek）
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
export ANTHROPIC_AUTH_TOKEN=sk-...
export ANTHROPIC_MODEL=deepseek-v4-pro[1m]
```

### 3.4 验证安装

```bash
python run.py contract check
```

预期输出：

```
训练脚本契约校验（cp_11）
----------------------------------------------------
[OK  ] resumable: --resume / --ckpt
[OK  ] checkpoint_schema: 已声明 [epoch, model_state_dict, optimizer_state_dict]
[OK  ] metrics_channel: log_file @ ../logs/train.log
[OK  ] buildable_entry: train:build_model / train:get_dataloaders
```

## 4. 快速开始

### 4.1 零接触守护（默认路径）

```bash
# 守护任意训练命令，训练脚本 0 行改动
python run.py watch -- python train.py --epochs 20 --batch_size 64
```

### 4.2 带 Agent 决策层

```bash
# Agent 参与异常应对决策 + AI 分析报告
python run.py watch --agent -- python train.py --epochs 20
```

### 4.3 训练前资源预检

```bash
# 显存预估、batch size 推荐、时长预测
python run.py preflight --epochs 20 --total-samples 60000
```

### 4.4 独立断点分析

```bash
# 扫描已有 checkpoint，选出最优模型
python run.py analyze

# 指定判定指标
python run.py analyze --metric val/loss --lower-better
```

## 5. 配置文件

### 5.1 guardian.yaml — guardian 自身工作参数

```yaml
# configs/guardian.yaml
project:
  name: my-experiment        # 实验名
  ckpt_dir: ./checkpoints    # checkpoint 目录
  log_dir: ./logs             # 日志目录

watchdog:
  max_retries: 3              # 连续失败上限
  restart_delay: 10           # 重启前等待（秒）
  oom_batch_reduce_ratio: 0.5 # OOM 时 batch 缩减比例
  min_batch_size: 8           # batch 下限
  no_progress_timeout: 1800   # 挂起告警阈值（秒）
  no_progress_kill_after: null # 挂起重启阈值（null=永不）

monitor:
  poll_interval: 5            # 指标轮询间隔（秒）
  sliding_window: 50          # 滑动窗口大小
  loss_spike_ratio: 0.5       # loss 突增判定比例
  gpu_idle_threshold: 20      # GPU 空转阈值（%）
  gpu_temp_threshold: 85      # GPU 温度告警（°C）

notifier:
  channels: [terminal]        # terminal / webhook / email
  cooldown: 300               # 同类告警静默期（秒）
  webhook_url_env: GUARDIAN_WEBHOOK_URL

agent:
  enabled: false              # --agent 标志覆盖此设置
  provider: anthropic         # anthropic / openai / custom
  model: null                 # 不设则读 ANTHROPIC_MODEL 环境变量
  api_key_env: ANTHROPIC_API_KEY
  decision_timeout: 8         # LLM 决策超时（秒）

mcp:
  enabled: false
  transport: stdio
  enable_write_tools: false   # 写工具默认关闭
```

配置优先级：**命令行参数 > 环境变量 `GUARDIAN_*` > 配置文件 > 内置默认值**

任何未在配置文件中声明的键都使用内置默认值——你只需要写与默认值不同的项。

### 5.2 contract.yaml — 被守护脚本的接口声明

```yaml
# configs/contract.yaml
script_contract:
  # 1. 可续训入口（sidecar 关键：缺失则整个重启路径失效）
  resumable:
    resume_flag: "--resume"
    ckpt_flag: "--ckpt"

  # 2. checkpoint 格式（guardian 靠这些键判断 checkpoint 是否可续训）
  checkpoint_schema:
    required_keys: [epoch, model_state_dict, optimizer_state_dict]

  # 3. 指标通道（缺失则退化为进程级看护）
  metrics_channel:
    type: log_file            # log_file | wandb | metrics_json
    path: ../logs/train.log
    log_pattern: "epoch (\\d+) loss ([\\d.naN]+)"

  # 4. 可 import 入口（preflight / 独立评估需要）
  buildable_entry:
    model_fn: "train:build_model"
    dataloader_fn: "train:get_dataloaders"

  # 重启改写的传参依据
  cli_mappings:
    optimizer.lr: "--lr"
    dataloader.batch_size: "--batch_size"
    dataloader.num_workers: "--num_workers"

  launcher: python
```

## 6. 命令行参考

```
python run.py <command> [options]

命令:
  watch         守护任意训练命令（默认主路径）
  contract      契约校验与审核
  analyze       分析已有 checkpoint
  preflight     训练前资源预检
  serve         启动 MCP server（外部 agent 接入）

示例:
  python run.py watch -- python train.py --epochs 20
  python run.py watch --agent -- python train.py
  python run.py watch --agent --with-mcp -- python train.py
  python run.py watch --max-retries 5 -- python train.py
  python run.py contract check
  python run.py contract review
  python run.py analyze --metric val/loss --lower-better
  python run.py preflight --epochs 50 --total-samples 120000
  python run.py serve --transport stdio
```

## 7. 功能详解

### 7.1 训练守护（watch）

```
python run.py watch -- python train.py
```

Guardian 以子进程方式拉起训练命令，在训练进程外全程看护：

1. **启动前**：校验契约四项，逐项打印开启/降级状态
2. **训练中**：按 `poll_interval` 周期性读取指标通道，检测异常；按 `hardware_poll_interval` 轮询 GPU
3. **异常时**：告警 + （有 agent 时）LLM 选择应对动作
4. **崩溃时**：分类崩溃类型 → 可恢复则从 checkpoint 重启 → 不可恢复则停止并推送诊断
5. **训练后**：自动生成结构化摘要 + AI 解读

### 7.2 Agent 决策层（--agent）

启用后，以下决策点由 LLM 参与：

| 决策点 | 触发条件 | Agent 可选动作 |
|--------|----------|---------------|
| 异常应对 | loss_spike / nan_inf 检测 | ignore / alert_only / restart_with_lower_lr |
| 恢复策略 | OOM / sigkill 崩溃 | reduce_batch / enable_grad_accum / resume_unchanged |
| 最优指标 | best model 判定 | accuracy / f1_macro / mAP50 / mIoU / rmse |
| AI 解读 | 训练结束 | 200-300 字自然语言分析 |

**降级保证：** Agent 调用超时/失败/返回非法动作 → 自动回退规则默认动作。训练不会因为 LLM 不可用而卡住或出错。

### 7.3 资源预检（preflight）

```
python run.py preflight --epochs 20 --total-samples 60000
```

依赖 `buildable_entry` 契约项，在独立进程中 import 模型 + dataloader 后：

1. 统计模型参数量
2. 获取 GPU 信息
3. 用小 batch 实际跑前向+反向，测量显存峰值
4. 线性回归外推各 batch_size 的显存占用
5. 推荐最大安全 batch_size
6. 测量单 step 耗时，预估总训练时长

### 7.4 异常检测规则

| 检测项 | 判定条件 | 响应 |
|--------|----------|------|
| loss_spike | 当前 loss > 窗口均值 × (1 + loss_spike_ratio) | agent 决策或 alert_only |
| loss_stagnation | N 步降幅 < stagnation_threshold | agent 决策或 alert_only |
| nan_inf | loss 为 NaN 或 Inf | 紧急告警（level=error） |
| gpu_idle | GPU 利用率连续 5 次 < idle_threshold | agent 决策或 alert_only |
| gpu_temp | GPU 温度 > temp_threshold | alert_only（硬件安全不交 agent） |

### 7.5 崩溃恢复

| 崩溃类型 | 识别方式 | 恢复策略 |
|----------|----------|----------|
| CUDA OOM | stderr 含 "CUDA out of memory" | 减半 batch_size，从最近 ckpt 重启 |
| 进程被 kill | 退出码 -9 / 137 | 参数不变，从最近 ckpt 续训 |
| 网络波动 | stderr 含 ConnectionError / Timeout | 等待后重试 |
| 代码错误 | stderr 含 TypeError / AttributeError | **0 次重启**，停止并推送诊断 |
| 无法识别 | 退出码非 0 且不匹配任何已知模式 | 保守判为不可恢复，停止重试 |

### 7.6 注册表系统

`select_metric()` 按 4 层优先级推断最优模型判定指标：

1. **config_explicit**: `contract.yaml` 中显式声明 `task_type`
2. **agent_inferred**: Agent 分析指标键名推断任务类型
3. **key_infer**: 从 `mAP`/`mIoU`/`accuracy` 等键名规则推断
4. **fallback**: `val_loss`

Agent 想突破注册表边界 → 只能生成"提议" → 人工审核通过后才写入正式注册表。

## 8. 部署到新项目

只需要改 `configs/contract.yaml`，告诉 guardian 你的训练脚本长什么样：

```yaml
script_contract:
  resumable:
    resume_flag: "--resume"           # 你的脚本用什么 flag
    ckpt_flag: "--ckpt"               # 你的脚本用什么 flag 指定 ckpt 路径
  checkpoint_schema:
    required_keys: [epoch, model_state_dict, optimizer_state_dict]
  metrics_channel:
    type: wandb                        # 如果用 wandb
    path: ./wandb/my-run               # wandb run 目录
    # 或者用 log_file:
    # type: log_file
    # path: ./logs/train.log
    # log_pattern: "epoch (\\d+) loss ([\\d.]+)"
  buildable_entry:
    model_fn: "my_train:build_model"   # 你的 model 构建函数
    dataloader_fn: "my_train:get_loaders"
  cli_mappings:                        # 你的命令行参数映射
    optimizer.lr: "--learning_rate"
    dataloader.batch_size: "--batch"
```

改完验证：

```bash
python run.py contract check
```

然后直接守护：

```bash
python run.py watch --agent -- python my_train.py --epochs 100
```

## 9. 环境变量参考

| 变量 | 用途 | 示例 |
|------|------|------|
| `ANTHROPIC_API_KEY` | Anthropic API 密钥 | `sk-ant-...` |
| `ANTHROPIC_AUTH_TOKEN` | OAuth / 第三方兼容 API token | `sk-...` |
| `ANTHROPIC_BASE_URL` | 自定义 API 端点 | `https://api.deepseek.com/anthropic` |
| `ANTHROPIC_MODEL` | 模型 ID | `deepseek-v4-pro[1m]` |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | 轻量模型回退 | `deepseek-v4-flash` |
| `OPENAI_API_KEY` | OpenAI API 密钥 | `sk-...` |
| `GUARDIAN_WATCHDOG__MAX_RETRIES` | 覆盖 watchdog.max_retries | `5` |
| `GUARDIAN_WEBHOOK_URL` | Webhook 推送地址 | `https://hooks.example.com/...` |
| `GUARDIAN_MCP_TOKEN` | MCP 写工具口令 | `my-secret-token` |

环境变量用双下划线表示层级：`GUARDIAN_WATCHDOG__MAX_RETRIES=5` 覆盖 `watchdog.max_retries`。

## 10. 架构文档

- `ARCHITECTURE.md` — 架构全景图、完整工作流、决策分层
- `IMPLEMENTATION_REPORT.md` — 逐模块对照 checkpoint 设计文档的完成度评估
- `checkpoint/` — 16 份设计文档（cp_1 ~ cp_12 + overview/requirements/configuration/functional_overview）
