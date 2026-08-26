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
│  CLI (guarftrain)                                               │
│  ├─ watch ──→ Watchdog: Popen + crash recovery + CLI rewrite    │
│  │             └─ Monitor: log tail + GPU poll + anomaly detect  │
│  │                  └─ AgentAdvisor: LLM decide → intervene       │
│  │                  └─ Sub-agent: --autonomy (supervised/auto/full) │
│  ├─ preflight ──→ ResourceEstimator: 显存预估 + batch 推荐        │
│  ├─ analyze ──→ CheckpointAnalyzer: checkpoint 扫描 + best 判定  │
│  ├─ serve ──→ MCP Server: 36 tools (25 read + 11 write)         │
│  ├─ start ──→ Dashboard + MCP one-click                         │
│  ├─ remote ──→ FastAPI 远程通信服务（算力服务器端）                │
│  ├─ dashboard ──→ Dashboard: HTTP+WS 控制面板（远程配置层）       │
│  ├─ visualize ──→ ModelViz: D3.js 结构可视化                     │
│  ├─ analyze_architecture ──→ ArchAnalyzer: D3 treemap + backbone │
│  └─ contract ──→ TaskContract: 契约校验 + 注册表管理              │
│                                                                 │
│  训练后: SummaryGenerator → Agent AI 解读                         │
├─────────────────────────────────────────────────────────────────┤
│                    TaskContract（契约层）                          │
│  硬性四项: 可续训 · checkpoint 格式 · 指标通道 · 可 import 入口    │
│  软性注册表: 指标注册表 · 可调路径白名单                             │
├─────────────────────────────────────────────────────────────────┤
│                    AgentAdvisor（LLM 决策层）                      │
│  decide() 统一入口 → Anthropic/OpenAI/DeepSeek                   │
│  熔断 · 超时降级 · 动作校验                                        │
├─────────────────────────────────────────────────────────────────┤
│  训练子进程: python train.py（完全不感知 guardian）                 │
│    产出: 日志/wandb → Monitor 读    checkpoints/ → Watchdog 读    │
└──────────────────────────────────────────────────────────────────┘
```

## 核心工作流：`guardian watch`

### 启动阶段

```
guarftrain watch --agent -- python train.py --epochs 20 --batch_size 64
  │
  ├─ 1. TaskContract 契约校验
  │     ✓ resumable:       --resume / --ckpt
  │     ✓ checkpoint_schema: [epoch, model_state_dict, optimizer_state_dict]
  │     ✓ metrics_channel:  log_file @ ./logs/train.log
  │     ✓ buildable_entry:  train:build_model / train:get_dataloaders
  │     缺一项 → 关一项能力，不阻断启动（除非 strict_mode）
  │
  ├─ 2. 初始化模块
  │     AgentAdvisor    ← ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN
  │     TrainingMonitor ← log tail + gpu_monitor
  │     CheckpointAnalyzer
  │     TrainingWatchdog ← 包装训练命令
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
  │   ├─ Monitor.poll_metrics()
  │   │   ├─ LogFileChannel: tail ./logs/train.log（增量，字节偏移追踪）
  │   │   ├─ WandbChannel:    读 wandb-events.jsonl（增量）
  │   │   ├─ poll_gpu():      gpu_monitor 轮询（利用率/温度/显存/功耗）
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
  │   │       [有 agent] → AgentAdvisor.decide("monitor_response")
  │   │         ├─ agent 返回 "ignore" → 不做任何事
  │   │         ├─ agent 返回 "alert_only" → 只告警
  │   │         └─ agent 返回 "restart_with_lower_lr(0.5)"
  │   │            → on_intervention → Watchdog.request_intervention()
  │   │              → 下一个看护周期: kill 训练 + 用新参数重启
  │   │
  │   ├─ CheckpointAnalyzer.poll()
  │   │    扫描 checkpoints/ → 发现新 cp → 校验写完 → 记录
  │   │
  │   └─ Watchdog.check_hang()
  │        指标停止前进 + 进程存活 → 只告警（默认）
  │
  ├─ 子进程崩溃:
  │   ├─ classify_crash(exit_code, stderr) [纯规则]
  │   │   OOM / sigkill / network → recoverable
  │   │   TypeError / AttributeError → unrecoverable（0 次重启）
  │   │   无法识别 → unrecoverable（保守，不反复重启）
  │   │
  │   ├─ [有 agent] AgentAdvisor.decide("watchdog_recovery")
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
训练结束 → SummaryGenerator.generate()
  │
  ├─ training:    最终 loss、最佳 acc、总 step 数
  ├─ anomaly_events: 每次异常 + agent/rule_default 应对来源
  ├─ restarts:    每次重启的 trigger(crash/intervention/hang)
  │               + 作废 epoch 数 + cmd_before/after diff
  ├─ checkpoints: total / latest / best + metric_source
  ├─ resources:   GPU 均值利用率 / 峰值显存 / GPU·h / 温度
  ├─ lr_schedule: 跨重启拼接的学习率变化点
  │
  └─ AI 解读: AgentAdvisor.narrate(summary)
       → DeepSeek/Anthropic API → 200-300 字自然语言分析
         "训练初期出现 5 次 loss 突增，agent 智能选择了
          3 次降学习率重启 + 2 次忽略，整体收敛稳定……"

输出: terminal 表格 + logs/summary_*.json + logs/summary_*.txt
```

---

## 决策分层

```
┌──────────────────────────────────────────┐
│  TaskContract 契约边界层（人定死）           │
│  · 可续训入口 / checkpoint 格式 / 指标通道   │
│  · 指标注册表（5 类任务 + fallback）        │
│  · 可调路径白名单（含幅度上限）              │
│  agent 可提议扩展，需人工审核后才生效         │
├──────────────────────────────────────────┤
│  AgentAdvisor 决策层（agent，可选，有动作权） │
│  · 异常应对: 告警 / 忽略 / 降lr重启          │
│  · 恢复策略: 减batch / 梯度累积 / 原样续训    │
│  · best 指标: accuracy / mAP / mIoU / …    │
│  · 图表推荐: chart_selection               │
│  · AI 解读: 自然语言报告（纯输出）            │
│  · 架构分析: analyze_architecture          │
│  · 子代理: --autonomy 自主决策              │
│  超时/失败/越界 → 强制回退规则默认动作         │
├──────────────────────────────────────────┤
│  Sub-agent 自主决策层（可选）                 │
│  · supervised: 所有决策需用户确认             │
│  · auto: 自动调整参数，重大干预需确认           │
│  · full: 完全自主，无需确认                   │
├──────────────────────────────────────────┤
│  规则引擎层（必选，可靠，兜底）               │
│  · 滑动窗口异常检测（判定"是不是"）           │
│  · 崩溃分类（判定"能不能恢复"）               │
│  · 进程看护 + 命令行改写（sidecar 唯一干预）   │
├──────────────────────────────────────────┤
│  MCP 工具层（外部 agent 接入）                │
│  · Claude Code / OpenClaw 客户端            │
│  · 25 只读工具 + 11 受限写工具（共 36）       │
│  · 委托模式: provisional 可覆盖（见下节）      │
│  · 幂等保证 / 访问日志 / 非阻塞               │
├──────────────────────────────────────────┤
│  Dashboard 远程配置层（v0.2）               │
│  · 初始值: contract.yaml → dashboard 段     │
│  · 覆盖: MCP set_dashboard_config（token）  │
│  · 前端: dirty flag 保护用户手动修改          │
└──────────────────────────────────────────┘
```

---

## MCP 委托模式（双模式增强，v0.2）

```
v0 双模式: 外部 Claude Code 连接时，内置 agent 完全让位。
v0.2 增强: agent 不再让位，而是进入 provisional（临时决策）模式——
           照常决策，但每条决策标记为 agent_provisional，可被外部覆盖。

  on_client_connect
    → set_mode(MCP_DELEGATED) → advisor.set_delegated(True)
        → 模式: autonomous → provisional
          │
          ├─ provisional 模式下每次 decide():
          │   agent 照常决策，来源改写为 agent_provisional
          │   → 决策推入 pending_decisions 队列（TTL 120s，
          │     超时未处理自动转 approved）
          │   → 外部 agent: get_pending_decisions 查看待审决策
          │   → resolve_decision(id, override, action):
          │       override=false → approved（认可内置决策）
          │       override=true  → overridden（用新动作覆盖；
          │                        已执行则经 watchdog 补救）
          │
          └─ on_client_disconnect
              → set_mode(STANDALONE) → set_delegated(False)
                  → 队列中未覆盖的 pending 决策自动转 approved
                  → 恢复 autonomous 自主决策
```

设计意图: 外部 agent 在线时拥有"审核/否决权"，离线时内置 agent
决策照常执行——智能决策永不丢失，外部接管只是可选项。

---

## Dashboard 远程配置（v0.2）

外部 agent（Claude Code）经 MCP 工具远程驱动 Dashboard 图表/面板；
用户手动修改受 dirty flag 保护，远程配置绝不覆盖用户操作。

```
contract.yaml（dashboard 段: template / charts / panels）
  │
  ├─ 启动: cmd_watch 注册进程
  │   POST /api/register（payload 含 dash_config）
  │     → Dashboard server 存入进程状态 _dash_config
  │     → 前端打开详情页 → GET /api/process/<id>/dashboard-config
  │
  ├─ 运行中: MCP 工具（经 HTTP 调 Dashboard API）
  │   set_dashboard_config（写，token 鉴权）
  │     → POST /api/process/<id>/dashboard-config（顶层 key 深度合并）
  │     → WS 推送 {type: "dashboard-config"} → 前端 applyRemoteDashConfig()
  │         图表组/平滑/面板逐项应用，_dashDirty 中的项跳过（绝不覆盖）
  │         用户点"重置"→ 清空 dirty flag，重新接受远程配置
  │   get_dashboard_config（只读，查询当前配置）
  │
  └─ Agent 联动（chart_selection）
      recommend_charts（只读）
        → AgentAdvisor chart_selection 决策点
        → 推荐图表组 + 平滑开关 + 理由
        → 外部 agent 可作为 set_dashboard_config 的输入
      list_dashboard_templates（只读）
        → training / comparison / minimal 三种布局模板
```

---

## 架构图分析（v0.3, ArchAnalyzer）

基于 archify 设计逻辑，剪枝后保留核心能力：

```
parse_model(model_fn)
  ├─ named_modules → 参数量 / 深度
  ├─ forward hooks → 输入输出形状
  ├─ dummy input forward pass
  ├─ FLOPs 计算
  └─ identical block folding (≥4 相同模块 → ×N)
      ↓
compute_stats(graph) → FLOPs/参数量/瓶颈检测
      ↓
build_tree(graph) → D3 可渲染的树结构
      ↓
render → Dashboard / MCP / DSH Plugin 三种视图
```

**瓶颈检测**：单模块参数量占比 >25% 标记为瓶颈层。

**三种视图入口**：
1. Dashboard `架构分析` 标签页 → 独立面板
2. MCP 工具 `analyze_architecture` → 外部 Agent 调用
3. DSH Plugin `ArchTab` → DeepSeek Harness 侧栏面板

---

## 远程通信（v0.3, Remote）

算力服务器端 FastAPI 服务，PC Dashboard 远程连接：

```
算力服务器 (remote server)          PC (Dashboard + MCP)
┌─────────────────────┐            ┌──────────────────┐
│ FastAPI :8765       │←─SSE/TCP──→│ WebSocket client  │
│ /api/decisions/...  │            │ _broadcast()     │
│ _session_store      │            │ _handle_message()│
│ Token auth          │            │ dirty flag 保护  │
└─────────────────────┘            └──────────────────┘
```

- 鉴权 token：GUARDIAN_REMOTE_TOKEN
- 持久化：SQLite session persistence
- 训练中用户操作受 dirty flag 保护（外部 Agent 可覆盖）

---

## Sub-agent 自主决策（v0.3）

`--autonomy` 三级权限：

| 模式 | 行为 |
|------|------|
| `supervised` (默认) | 所有决策需用户确认 |
| `auto` | 自动调整参数（lr/batch），重大干预需确认 |
| `full` | 完全自主，无需确认 |

核心流程：
```
anomaly detected → build prompt → LLM decide → validate contract → execute / queue
```

---

## CPU 模式兼容（v0.3）

- 无 GPU 时 `torch.cuda.is_available()` → False
- CLI 启动时弹出 CPU 警告
- 训练曲线 (loss/accuracy/lr) 正常显示（CPU 曲线，设备无关）
- GPU 面板提示不可用（需 nvidia-smi + CUDA）
- `resource_estimator` 回退兼容 PyTorch 1.x（torch >= 1.13）

---

## 功能模块一览（cp_1 ~ cp_21）

| 模块 | 文件 | 职责 | 阶段 |
|------|------|------|------|
| **cp_1** | `resource_estimator.py` | 显存测量 → batch 推荐 → 时间预估 | 训练前 |
| **cp_2** | `monitor.py` | 指标 tail + GPU 轮询 + 异常检测 | 训练中 |
| **cp_3** | `watchdog.py` | 子进程看护 + 崩溃恢复 + 命令行改写 | 训练中 |
| **cp_4** | `checkpoint_analyzer.py` | cp 发现/校验/top-k/best 判定 | 训练中/后 |
| **cp_5** | `summary.py` | 结构化摘要 + AI 解读 | 训练后 |
| **cp_6** | `notifier.py` | 终端/webhook 告警 + 静默期 | 全程 |
| **cp_7** | `train.py` | MNIST 参考训练脚本（满足契约四项） | 参考 |
| **cp_8** | `cli.py` | CLI 入口: watch/preflight/analyze/serve/dashboard/start | 入口 |
| **cp_9** | `agent_advisor.py` | LLM 决策统一入口 + 熔断/降级 + chart_selection | v1/v0.2 |
| **cp_10** | `mcp_server.py` | MCP 工具暴露（36 工具：25 只读 + 11 写） | v1/v0.2 |
| **cp_11** | `task_contract.py` | 契约校验 + 注册表 + 提议审核 | 全程 |
| **cp_12** | `tests/faultbench/` | 故障注入测试（S1/S2/S3 验收） | 测试 |
| **cp_13** | `gallery.py` | 图片筛选 + 多策略精选 | v1 |
| **cp_14** | `experiment_query.py` | 实验查询 + 自然语言检索 | v1 |
| **cp_15** | `model_viz.py` | 模型结构可视化（D3.js HTML） | v1 |
| **cp_16** | `inference.py` | checkpoint 推理运行器 | v1 |
| **cp_17** | `dashboard/server.py` | Dashboard HTTP+WS 面板 + 远程配置 | v0.2 |
| **cp_18** | `agent_advisor.py` | Agent 图表推荐（chart_selection 决策点） | v0.2 |
| **cp_19** | `mcp_server.py` | MCP Dashboard 工具（config/recommend/templates） | v0.2 |
| **cp_20** | `arch_analyzer.py` | 架构图分析（D3 treemap + backbone + 瓶颈检测） | v0.3 |
| **cp_21** | `dsh-plugin/` | DeepSeek Harness 侧栏面板 + SSE 实时推送 | v0.3 |
| **cp_22** | `remote/server.py` | FastAPI 远程通信服务（算力服务器端） | v0.3 |
| **cp_23** | `gpu_monitor.py` | nvidia-smi GPU 监控（不依赖 torch.cuda） | v0.3 |
| **cp_24** | `sub_agent/` | Sub-agent 自主决策（supervised/auto/full） | v0.3 |
| - | `config.py` | 配置加载: DEFAULTS < 文件 < 环境变量 < CLI | 基础设施 |
| - | `configs/guardian.yaml` | guardian 自身工作参数 | 配置 |
| - | `configs/contract.yaml` | 训练脚本契约声明 + 注册表 + dashboard 段 | 配置 |

---

## Agent 决策回路（v1 核心价值）

```
异常发生 → Monitor 检测 ─→ AgentAdvisor LLM ─→ 应对动作
                                    │
                    ┌───────────────┘
                    ▼
              alert_only     restart_with_lower_lr     ignore
              只告警            通知 Watchdog            不做任何事
                                │
                    kill 训练子进程
                    命令行改写（新 lr）
                    从最近 checkpoint 重启
                    记录 RestartRecord
                                    │
训练结束 → SummaryGenerator ← AgentAdvisor AI 解读 ← DeepSeek API
```

### chart_selection 决策点（v0.2 新增）

```
Dashboard 图表推荐:
  输入: metrics_summary / 可用图表组 / anomaly_count / training_phase
  输出: 推荐图表组（loss/accuracy/lr/gpu/custom）+ 平滑开关 + 理由
  降级: agent 不可用/超时 → 回退默认 {loss, accuracy}（当前配置不变）
```

### analyze_architecture 决策点（v0.3 新增）

```
架构图分析:
  输入: model_entry / project_dir / cached graph
  输出: tree (D3 可渲染) + bottlenecks + FLOPs/参数量
  三种视图: Dashboard / MCP / DSH Plugin
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
  ├─ agent 可用? → AgentAdvisor.decide("select_metric")
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
| **+ Dashboard** | `guardian watch --agent --with-dashboard -- python train.py` | Web 控制面板 + 远程配置 |
| **+ 远程通信** | `guardian remote -- python train.py` | 算力服务器端，PC Dashboard 远程连接 |
| **+ 架构分析** | `guarftrain analyze_architecture --model entry:fn` | 独立架构图分析 |
| **+ Sub-agent** | `guarftrain watch --autonomy auto -- python train.py` | 自动调整参数 |
| **全功能** | `guardian watch --agent --with-dashboard --with-mcp --with-dsh -- python train.py` | 面板 + 外部 agent + 架构分析 |

---

## DSH Web GUI Plugin（v0.3）

DeepSeek Harness 侧栏面板，通过 slot 注入：

```
dsh-web-ui runtime
  ├─ slots.inject('sidebar.training-guardian') → TrainingPanel (React)
  └─ slots.inject('web-ui.plugin.item') → SettingsCard
```

**SSE 实时推送**：metrics / GPU / anomalies / decisions / architecture

**本地化**：zh/en 双语，通过 DSH locale system

**ArchTab 组件**：
- D3.js v7 treemap / backbone 双视图
- params/flops 颜色映射
- 瓶颈层侧栏 + 详情面板
- 深色主题适配（DSH CSS 自定义属性）
