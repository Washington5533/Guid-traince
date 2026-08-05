# Training Guardian Agent — 项目总览

## 项目定位
训练守护 agent，覆盖训练前预检 → 训练中监控 → 训练后分析的完整生命周期。

**架构基线：sidecar-first**（详见 [functional_overview.md](functional_overview.md#架构基线sidecar-first)）。guardian 默认以独立进程形态在训练进程之外运行，包装训练命令（`guardian watch -- python train.py`），指标从训练脚本已有的输出通道读取，干预通过"kill + 从 checkpoint 重启"实现——**开发者的训练脚本 0 行改动**。嵌入式集成（进程内插回调，可 per-step 检测与即时改参）作为可选的精细化升级路径保留。

底层为确定性规则引擎（保证训练不受 AI 故障影响），关键决策点可选接入 LLM 判断（中层 agent 化：规则引擎决定"何时问"，agent 决定"怎么做"，超时/失败强制回退规则默认值）。指标口径与可调参数的边界由任务契约层（cp_11）显式声明，agent 在边界内按具体任务自适应选择，边界本身的扩展需人工审核后才生效。同时通过 MCP server 把全部能力暴露为标准工具，Claude Code / OpenClaw 等外部 agent 客户端可直接接入查看与操作训练。

## 目录结构
```
guarftrain/
├── configs/
│   ├── guardian.yaml          # guardian 自己怎么工作（轮询/阈值/重试/agent/mcp）→ configuration.md
│   └── contract.yaml          # 被守护脚本长什么样（契约四项 + 注册表 + 白名单）→ cp_11.md
├── guardian/                   # 核心包
│   ├── __init__.py
│   ├── resource_estimator.py   # cp_1  纯规则；经 cp_11 buildable_entry 入口 import 后测量
│   ├── monitor.py              # cp_2  外部观测指标(tail wandb/日志) + 规则检测 + agent 应对方式判断（可选）
│   ├── watchdog.py             # cp_3  【sidecar 核心】包装训练命令、进程看护、重启式干预的统一执行路径
│   ├── checkpoint_analyzer.py  # cp_4  轮询发现 ckpt 目录 + 校验 + best/top-k（判定指标来自 cp_11）
│   ├── summary.py              # cp_5  纯规则 + agent 自然语言解读（可选）
│   ├── notifier.py             # cp_6  纯规则，不接入 agent
│   ├── agent_advisor.py        # cp_9  LLM 调用统一入口：超时/降级契约、动作集裁剪
│   ├── task_contract.py        # cp_11 任务契约与自适应层：契约校验、指标/路径选择、agent 提议审核
│   └── mcp_server.py           # cp_10 MCP server：guardian 能力 → 标准工具，读写权限分离
├── train.py                    # cp_7 MNIST 训练脚本（参考被守护对象，不 import guardian）
├── run.py                      # cp_8 CLI 入口（watch / preflight / analyze / serve / contract）
├── checkpoints/                # 运行时生成
│   └── cp_{epoch}/
│       ├── model.pth
│       ├── metrics.json
│       ├── quick_val.json
│       └── full_val.json       # 每 N 个 cp 生成
├── data/                       # MNIST 数据集
├── logs/                       # 训练日志
├── tests/
│   └── faultbench/             # cp_12 故障注入测试装置（假训练脚本 + 场景用例，无需 GPU）
├── requirements-core.txt       # 轻量远程部署：核心训练 + 规则引擎 + agent 决策层（不含 mcp）
└── requirements-mcp.txt        # 可选叠加层：仅 MCP 接入需要，随时可补装，不需要重启训练
```

## 功能模块索引

排期分档见 [functional_overview.md](functional_overview.md#优先级)：**v0 全纯规则**（先把"崩了能救回来"做实），**v1 引入 agent 与 MCP**，v2 为训练之外的产出。

| 编号 | 模块 | 文件 | 阶段 | 档位 | agent 化 | 文档 |
|------|------|------|------|------|----------|------|
| cp_3 | 进程守护与恢复 | watchdog.py | 训练中 | **v0** | 恢复策略判断（v1 叠加） | [cp_3.md](cp_3.md) |
| cp_2 | 训练监控 | monitor.py | 训练中 | **v0** | 应对方式判断（v1 叠加） | [cp_2.md](cp_2.md) |
| cp_6 | 告警推送 | notifier.py | 全程 | **v0** | 无 | [cp_6.md](cp_6.md) |
| cp_4 | 断点分析 | checkpoint_analyzer.py | 训练后 | **v0** | best 指标选择（v1 叠加） | [cp_4.md](cp_4.md) |
| cp_5 | 日志摘要 | summary.py | 训练后 | **v0** | 自然语言解读（v1 叠加） | [cp_5.md](cp_5.md) |
| cp_7 | 参考训练脚本 | train.py | 核心 | **v0** | -（不 import guardian） | [cp_7.md](cp_7.md) |
| cp_8 | CLI 入口 | run.py | 入口 | **v0** | - | [cp_8.md](cp_8.md) |
| cp_11 | 任务契约与自适应层 | task_contract.py | 全程 | v0 契约校验 / v1 注册表与提议 | 是（指标/路径选择 + 受审核提议） | [cp_11.md](cp_11.md) |
| cp_9 | Agent 决策封装 | agent_advisor.py | 全程 | v1 | 是（基础设施） | [cp_9.md](cp_9.md) |
| cp_10 | MCP 工具层 | mcp_server.py | 全程 | v1 | 是（外部接入） | [cp_10.md](cp_10.md) |
| cp_1 | 资源预估 | resource_estimator.py | 训练前 | v1 | 无（batch 推荐受 cp_11 白名单约束） | [cp_1.md](cp_1.md) |
| cp_12 | 故障注入测试装置 | tests/faultbench/ | 测试 | **v0（与 cp_3 同步开发）** | -（不参与生产运行） | [cp_12.md](cp_12.md) |

表格按**实现顺序**排列而非编号顺序：cp_3 排第一，因为 sidecar 架构下它是一切的地基——没有进程看护与重启，其余模块都没有作用对象。cp_12 虽然编号最后，但必须与 cp_3 同步开发：没有故障注入装置，"崩了能救回来"这件事无法被验证，v0 的验收标准就只是一句愿望。

## 协作流程（sidecar 形态）

```
                    guardian 进程（训练进程之外）
  ┌──────────────────────────────────────────────────────────────┐
  │ cp_11 TaskContract：启动时校验脚本契约四项，决定各能力开/关     │
  │        │                                                     │
  │ 训练前  ▼            训练中                      训练后        │
  │ cp_1 资源预估   cp_2 外部观测+规则检测+应对(agent)  cp_4 断点分析 │
  │ (独立进程        │         └──重启式动作──┐        cp_5 日志摘要 │
  │  import 入口)    cp_3 进程看护 ◄──────────┘        cp_6 推送摘要 │
  │                     │  (崩溃恢复 + 主动干预，统一重启路径)      │
  │                 cp_6 推送通知                                 │
  │                                                              │
  │ cp_9  AgentAdvisor：所有 agent 调用的统一出入口（超时/降级）    │
  │ cp_10 MCP Server：供外部 agent（Claude Code/OpenClaw）接入      │
  └────────────────────────┬─────────────────────────────────────┘
                           │ 拉起 / 看护 / kill+重启
                           ▼
                  训练子进程：python train.py（0 行 guardian 代码）
                    └─ 产出：wandb/日志（cp_2 读）、checkpoints/（cp_3/cp_4 读）
```

数据流是单向的：训练子进程只管产出指标和 checkpoint，guardian 只读这些产物 + 通过重启施加影响。两者之间没有进程内耦合。

## 决策点分层原则

| 层级 | 内容 | 由谁做 | 原因 |
|------|------|--------|------|
| 检测/分类 | 是否 NaN、是否 loss 突增、崩溃是否可恢复 | 规则引擎（必须） | 结果影响训练是否继续，必须确定性、零延迟、可离线运行 |
| 应对方式/策略选择 | 发现异常后怎么办、怎么恢复；用什么指标判定"最优模型" | agent（可选，有限动作集/注册表，边界由 cp_11 定义）| 需要综合上下文判断，容错空间大，可以失败降级 |
| 解释/报告 | 为什么会这样、下一步建议 | agent（可选，纯输出）| 不影响训练流程，失败只影响可读性 |
| 边界扩展 | 注册表/白名单没覆盖的任务类型或调整路径 | agent 提议 + 人工审核（cp_11） | 扩大 agent 可选择的空间本身是高杠杆操作，不能由 agent 单方面决定 |

任何 agent 调用超时或出错，直接执行该决策点原本的规则引擎默认动作（详见 [cp_9.md](cp_9.md)）；agent 想让"下次自己能选的范围"变大，只能提议、不能自己拍板（详见 [cp_11.md](cp_11.md)）。

## 配置
两份配置文件职责分离，完整键位参考见 [configuration.md](configuration.md)：

- `configs/guardian.yaml` — guardian 自己怎么工作（轮询间隔、检测阈值、重试次数、agent/MCP 开关），调优时改
- `configs/contract.yaml` — 被守护的训练脚本长什么样（契约四项 + 指标注册表 + 可调路径白名单），接入新项目时改一次

v0 真正必需的配置只有十来行（见 configuration.md"v0 最小配置"），其余全部有默认值。secrets 一律只走环境变量，配置文件里存的是变量名而非值。

## 依赖
依赖拆成两份文件，物理隔离"轻量远程部署"和"MCP 接入叠加层"，详见 [requirements.md](requirements.md)。

- torch, torchvision (核心)
- pyyaml (配置解析)
- psutil, GPUtil (硬件采集, 可选)
- requests (webhook推送, 可选)
- anthropic / openai (agent 决策层调用的 LLM SDK, 可选，未配置时自动禁用 agent 层，属于 `requirements-core.txt`——独立 agent 功能不依赖 MCP)
- mcp (MCP server SDK, 属于 `requirements-mcp.txt`，可选，未安装时不启动工具暴露服务，训练与 agent 决策层不受影响)

## 文档索引

| 文档 | 内容 |
|------|------|
| [functional_overview.md](functional_overview.md) | 功能全景、架构基线（sidecar-first）、AI 分层、部署方式、v0/v1/v2 排期 |
| [configuration.md](configuration.md) | guardian.yaml / contract.yaml 完整键位参考与易错项 |
| [requirements.md](requirements.md) | 依赖分层：core（含 agent）vs mcp（可事后补挂） |
| [cp_1.md](cp_1.md) ~ [cp_12.md](cp_12.md) | 各模块设计与校验标准（见上方模块索引表） |
