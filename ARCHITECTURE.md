# Training Guardian Agent — 架构与工作流

## 一句话

```
guardian watch --agent -- python train.py
```

一行命令，训练进程 **0 行改动**，获得：GPU 监控 → 异常检测 → LLM 决策应对 → 崩溃恢复 → AI 分析报告。

---

## 架构全景

```
┌─────────────────────────────────────────────────────────────────┐
│                    guardian 进程（训练进程之外）                   │
│                                                                 │
│  CLI (run.py)                                                   │
│  ├─ watch ──→ cp_3 watchdog: 包装训练命令，子进程看护              │
│  │              └─ on_tick ──→ cp_2 monitor: 读指标 + GPU 轮询    │
│  │                               └─ 异常 → cp_9 agent: LLM 决策   │
│  │              └─ 崩溃 ──→ classify → cp_9 agent: 恢复策略       │
│  ├─ preflight ──→ cp_1: 显存预估 + batch 推荐                    │
│  ├─ analyze ──→ cp_4: checkpoint 扫描 + best 判定                │
│  ├─ serve ──→ cp_10: MCP 工具层（外部 agent 接入）                │
│  └─ contract ──→ cp_11: 契约校验 + 注册表管理                    │
│                                                                 │
│  训练后: cp_5 summary → cp_9 agent: AI 自然语言解读               │
├─────────────────────────────────────────────────────────────────┤
│                    cp_11 TaskContract（契约层）                    │
│  硬性四项: 可续训 · checkpoint 格式 · 指标通道 · 可 import 入口    │
│  软性注册表: 指标注册表 · 可调路径白名单                             │
├─────────────────────────────────────────────────────────────────┤
│                    cp_9 AgentAdvisor（LLM 决策层）                  │
│  decide() 统一入口 → Anthropic/OpenAI/DeepSeek                    │
│  熔断 · 超时降级 · 动作校验                                        │
├─────────────────────────────────────────────────────────────────┤
│  训练子进程: python train.py（完全不感知 guardian）                 │
│    产出: 日志/wandb → cp_2 读       checkpoints/ → cp_3/cp_4 读   │
└─────────────────────────────────────────────────────────────────┘
```

## 核心工作流：`guardian watch`

### 启动阶段

```
python run.py watch --agent -- python train.py --epochs 20 --batch_size 64
  │
  ├─ 1. cp_11 契约校验
  │     ✓ resumable:       --resume / --ckpt
  │     ✓ checkpoint_schema: [epoch, model_state_dict, optimizer_state_dict]
  │     ✓ metrics_channel:  log_file @ ./logs/train.log
  │     ✓ buildable_entry:  train:build_model / train:get_dataloaders
  │     缺一项 → 关一项能力，不阻断启动（除非 strict_mode）
  │
  ├─ 2. 初始化模块
  │     cp_9 AgentAdvisor    ← ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
  │     cp_2 TrainingMonitor ← log tail + nvidia-smi
  │     cp_4 CheckpointAnalyzer
  │     cp_3 TrainingWatchdog ← 包装训练命令
  │
  └─ 3. 拉起训练子进程
        subprocess.Popen("python train.py --epochs 20 --batch_size 64")
```

### 训练中：看护循环

```
while 训练子进程存活:
  │
  ├─ on_tick() 每个轮询周期:
  │   │
  │   ├─ cp_2 monitor.poll_metrics()
  │   │   ├─ LogFileChannel: tail ./logs/train.log（增量，字节偏移追踪）
  │   │   ├─ WandbChannel:    读 wandb-events.jsonl（增量）
  │   │   ├─ poll_gpu():      nvidia-smi 轮询（利用率/温度/显存/功耗）
  │   │   │
  │   │   └─ 异常检测（纯规则，零延迟）:
  │   │       loss_spike:    当前 loss > 窗口均值 * 1.5 → 触发
  │   │       loss_stagnation: 500 步降幅 < 0.001 → 触发
  │   │       nan_inf:        loss 为 NaN/Inf → 紧急告警
  │   │       gpu_idle:       GPU 利用率连续 5 次 < 20% → 触发
  │   │       gpu_temp:       GPU 温度 > 85°C → 触发
  │   │
  │   │   └─ 异常应对决策:
  │   │       [无 agent] → alert_only（v0 默认）
  │   │       [有 agent] → cp_9 decide("monitor_response")
  │   │         ├─ agent 返回 "ignore" → 不做任何事
  │   │         ├─ agent 返回 "alert_only" → 只告警
  │   │         └─ agent 返回 "restart_with_lower_lr(0.5)"
  │   │            → cp_2 → on_intervention → cp_3.request_intervention()
  │   │              → 下一个看护周期: kill 训练 + 用新参数重启
  │   │
  │   ├─ cp_4 analyzer.poll()
  │   │    扫描 checkpoints/ → 发现新 cp → 校验写完 → 记录
  │   │
  │   └─ cp_3 check_hang()
  │        指标停止前进 + 进程存活 → 只告警（默认）
  │
  ├─ 子进程崩溃:
  │   ├─ classify_crash(exit_code, stderr) [纯规则]
  │   │   OOM / sigkill / network → recoverable
  │   │   TypeError / AttributeError → unrecoverable（0 次重启）
  │   │   无法识别 → unrecoverable（保守，不反复重启）
  │   │
  │   ├─ [有 agent] cp_9 decide("watchdog_recovery")
  │   │   OOM → agent 可选: reduce_batch / enable_grad_accum / resume_unchanged
  │   │   sigkill → agent 可选: resume_unchanged / resume_with_reduced_workers
  │   │
  │   └─ apply_action() → 命令行改写
  │       原始: python train.py --batch_size 64
  │       reduce_batch(0.5): → --batch_size 32（只替换，不追加）
  │       追加 --resume --ckpt checkpoints/cp_10
  │
  └─ retry_count < max_retries:
       等待 restart_delay → 新子进程从 checkpoint 续训
```

### 训练后：摘要生成

```
训练结束 → cp_5 SummaryGenerator.generate()
  │
  ├─ training:    最终 loss、最佳 acc、总 step 数
  ├─ anomaly_events: 每次异常 + agent/rule_default 应对来源
  ├─ restarts:    每次重启的 trigger(crash/intervention/hang)
  │               + 作废 epoch 数 + cmd_before/after diff
  ├─ checkpoints: total / latest / best + metric_source
  ├─ resources:   GPU 均值利用率 / 峰值显存 / GPU·h / 温度
  ├─ lr_schedule: 跨重启拼接的学习率变化点
  │
  └─ AI 解读: cp_9 narrate(summary)
       → DeepSeek/Anthropic API → 200-300 字自然语言分析
         "训练初期出现 5 次 loss 突增，agent 智能选择了
          3 次降学习率重启 + 2 次忽略，整体收敛稳定……"

输出: terminal 表格 + logs/summary_*.json + logs/summary_*.txt
```

---

## 决策分层

```
┌──────────────────────────────────────────┐
│  cp_11 契约边界层（人定死）                 │
│  · 可续训入口 / checkpoint 格式 / 指标通道  │
│  · 指标注册表（5 类任务 + fallback）        │
│  · 可调路径白名单（含幅度上限）              │
│  agent 可提议扩展，需人工审核后才生效         │
├──────────────────────────────────────────┤
│  cp_9 决策层（agent，可选，有动作权）         │
│  · 异常应对: 告警 / 忽略 / 降lr重启          │
│  · 恢复策略: 减batch / 梯度累积 / 原样续训    │
│  · best 指标: accuracy / mAP / mIoU / …    │
│  · AI 解读: 自然语言报告（纯输出）            │
│  超时/失败/越界 → 强制回退规则默认动作         │
├──────────────────────────────────────────┤
│  规则引擎层（必选，可靠，兜底）               │
│  · 滑动窗口异常检测（判定"是不是"）           │
│  · 崩溃分类（判定"能不能恢复"）               │
│  · 进程看护 + 命令行改写（sidecar 唯一干预）    │
├──────────────────────────────────────────┤
│  cp_10 MCP 工具层（外部 agent 接入）          │
│  · Claude Code / OpenClaw 客户端            │
│  · 10 只读工具 + 7 受限写工具                 │
│  · 幂等保证 / 访问日志 / 非阻塞                │
└──────────────────────────────────────────┘
```

---

## 15 个功能模块一览

| 模块 | 文件 | 职责 | 阶段 |
|------|------|------|------|
| **cp_1** | `resource_estimator.py` | 显存测量 → batch 推荐 → 时间预估 | 训练前 |
| **cp_2** | `monitor.py` | 指标 tail + GPU 轮询 + 异常检测 | 训练中 |
| **cp_3** | `watchdog.py` | 子进程看护 + 崩溃恢复 + 命令行改写 | 训练中 |
| **cp_4** | `checkpoint_analyzer.py` | cp 发现/校验/top-k/best 判定 | 训练中/后 |
| **cp_5** | `summary.py` | 结构化摘要 + AI 解读 | 训练后 |
| **cp_6** | `notifier.py` | 终端/webhook 告警 + 静默期 | 全程 |
| **cp_7** | `train.py` | MNIST 参考训练脚本（满足契约四项） | 参考 |
| **cp_8** | `run.py` | CLI 入口: watch/preflight/analyze/serve | 入口 |
| **cp_9** | `agent_advisor.py` | LLM 决策统一入口 + 熔断/降级 | v1 |
| **cp_10** | `mcp_server.py` | MCP 工具暴露（17 工具） | v1 |
| **cp_11** | `task_contract.py` | 契约校验 + 注册表 + 提议审核 | 全程 |
| **cp_12** | `tests/faultbench/` | 故障注入测试（S1/S2/S3 验收） | 测试 |
| - | `config.py` | 配置加载: DEFAULTS < 文件 < 环境变量 < CLI | 基础设施 |
| - | `configs/guardian.yaml` | guardian 自身工作参数 | 配置 |
| - | `configs/contract.yaml` | 训练脚本契约声明 + 注册表 | 配置 |

---

## Agent 决策回路（v1 核心价值）

```
异常发生 → cp_2 检测 ─→ cp_9 LLM ─→ 应对动作
                                    │
                    ┌───────────────┘
                    ▼
              alert_only     restart_with_lower_lr     ignore
              只告警            通知 cp_3              不做任何事
                                │
                    kill 训练子进程
                    命令行改写（新 lr）
                    从最近 checkpoint 重启
                    记录 RestartRecord
                                    │
训练结束 → cp_5 摘要 ← cp_9 AI 解读 ← DeepSeek API
```

### 实测示例

```
loss_spike 检测:
  +157% → agent -> restart_with_lower_lr   (决定干预)
  +117% → agent -> ignore                  (判断为正常波动)
  +96%  → agent -> restart_with_lower_lr   (再次干预)
  +84%  → agent -> ignore
  +76%  → agent -> restart_with_lower_lr

AI 解读:
  "训练初期出现多次 loss 突增，智能体对此采取了差异处理——
   3 次降低学习率重启以稳定训练，2 次判断为正常波动而忽略。
   峰值准确率达 0.75，但最终回落到 0.5，建议排查 loss
   突增根因并考虑使用最佳 checkpoint 部署。"
```

---

## cp_11 注册表推断流程

```
select_metric({"metrics_seen": ["val/accuracy", "train/loss"]})
  │
  ├─ task_type 显式声明? → 直接命中 (source: config_explicit)
  ├─ agent 可用? → cp_9 decide("select_metric")
  │   → agent 分析指标键名 → 分类任务 → accuracy (source: agent_inferred)
  ├─ 规则推断: "mAP" in keys → detection → mAP50
  │           "mIoU" in keys → segmentation → mIoU
  │           "accuracy" in keys → classification → accuracy
  └─ 都不行 → val_loss (source: fallback)

结果记入 analysis_report.json:
  "metric_source": {"name": "accuracy", "direction": "max",
                    "source": "agent_inferred", "task_type": "classification"}
```

---

## 部署形态

| 模式 | 命令 | 适用场景 |
|------|------|----------|
| **纯规则守护** | `guardian watch -- python train.py` | 不依赖 LLM，0 外部依赖 |
| **+ Agent 决策** | `guardian watch --agent -- python train.py` | LLM 参与异常应对 + AI 报告 |
| **+ MCP 接入** | `guardian watch --agent --with-mcp -- python train.py` | 外部 Claude Code 可查看/管理 |
| **资源预检** | `guardian preflight` | 训练前显存预估 |
| **断点分析** | `guardian analyze` | 独立扫描已有 checkpoint |
| **契约校验** | `guardian contract check` | 验证训练脚本满足契约 |
