# cp_8 · CLI 入口 (run.py)

**文件**: `run.py`
**阶段**: 入口
**核心目标**: 统一命令行入口，支持多种运行模式

> **架构基线：sidecar-first**（见 [functional_overview.md](functional_overview.md#架构基线sidecar-first)）。默认入口是 `watch`——包装**任意**训练命令做进程外守护，被守护的脚本不需要 import guardian。`train` 保留为内置 MNIST 示例的便捷入口（等价于 `watch -- python train.py`），不是主路径。

---

## 命令设计

### 基本用法
```bash
# 【默认路径】守护任意训练命令，被守护脚本 0 行改动
guardian watch -- python train.py --epochs 20
python run.py watch -- python train.py --epochs 20          # 等价写法
python run.py watch --config configs/guardian.yaml -- python my_project/train.py --cfg exp1.yaml

# 守护 + agent 决策层（cp_9），未配置 API key 时自动降级为纯规则
python run.py watch --agent -- python train.py

# 内置 MNIST 示例的便捷入口（等价于 watch -- python train.py）
python run.py train --epochs 50 --batch_size 128

# 校验训练脚本是否满足 cp_11 契约四项，逐项打印开启/降级状态
python run.py contract check
python run.py contract review                                # 审核 agent 提议的注册表新增条目

# 仅运行资源预检，不训练（经 contract.buildable_entry 声明的入口 import）
python run.py preflight

# 分析已有 checkpoint（独立扫描 checkpoints/ 目录）
python run.py analyze
python run.py analyze --metric mAP50                         # 用指定指标重算 best

# 单独启动 MCP server，供 Claude Code / OpenClaw 等外部 agent 接入（cp_10）
python run.py serve --transport stdio

# watch 的同时后台启动 MCP server（同进程共享状态）
python run.py watch --with-mcp -- python train.py
```

`--` 之后的内容原样作为训练命令传给 [cp_3.md](cp_3.md) 的 `TrainingWatchdog.run(train_cmd)`，guardian 不解析也不改写，只在重启时按 `contract.cli_mappings` 追加/替换需要调整的参数。

---

## 子命令

### `watch` — 守护任意训练命令（**默认主路径**）
```python
def cmd_watch(args, train_cmd):
    """
    1. 加载配置 + contract.yaml
    2. TaskContract.validate_script_contract()
       逐项校验四项契约，打印各能力的开启/降级状态
       strict_mode=true 且有缺失 -> 直接退出
    3. 初始化 guardian 各组件（全部在本进程，不进训练进程）
       advisor / notifier / monitor / ckpt_analyzer / summary_gen / watchdog
    4. 可选：后台启动 MCP server（--with-mcp）
    5. watchdog.run(train_cmd)
       - 以子进程方式拉起训练命令
       - 循环：poll 指标通道 -> 规则检测 -> (可选)agent 决策 -> 必要时重启式干预
       - 子进程异常退出 -> 分类 -> 恢复或告警
    6. 训练结束（正常完成或不可恢复）-> analyze_all + summary + 推送
    """
```

### `train` — 内置 MNIST 示例便捷入口
```python
def cmd_train(args):
    """
    等价于 cmd_watch(args, ["python", "train.py", ...透传参数])
    仅为开箱即用的演示/自测方便；真实项目直接用 watch
    """
```

### `contract` — 契约校验与提议审核（cp_11）
```python
def cmd_contract(args):
    """
    check:  validate_script_contract() 逐项输出通过/降级 + 判定依据
            退出码：全通过=0，有降级=0（仅提示），strict_mode 下有缺失=1
    review: list_proposals("pending") 交互式列出待审提议（含 evidence）
            approve_proposal / reject_proposal，写入正式注册表或归档
    """
```

### `preflight` — 仅资源预检
```python
def cmd_preflight(args):
    """
    1. 加载配置 + contract.yaml
    2. 通过 contract.buildable_entry 声明的入口 import model_fn / dataloader_fn
       （契约缺失该项时明确报错退出，说明此命令依赖该契约）
    3. 运行 preflight_check
    4. 输出预检报告
    5. 退出（不训练）
    """
```

### `analyze` — 分析已有 checkpoint
```python
def cmd_analyze(args):
    """
    1. 扫描 checkpoints/ 目录（纯外部，不需要训练进程）
    2. 判定指标：args.metric 显式指定 > TaskContract.select_metric() > val_loss
    3. 对每个 cp_ 运行 quick_validate（如无结果，需 buildable_entry 契约）
    4. 生成 analysis_report.json（含 metric_source）
    5. 终端输出对比表格
    """
```

### `serve` — 单独启动 MCP server
```python
def cmd_serve(args):
    """
    1. 加载配置
    2. 检查 mcp 包是否已安装（requirements-mcp.txt）：
       未安装 -> 打印明确提示("pip install -r requirements-mcp.txt 后重试")，直接退出，不影响任何已在运行的训练进程
    3. 若无正在运行的训练进程，以只读模式启动（跨进程读盘刷新状态）
    4. 若已有训练进程在跑（同一 checkpoints/logs 目录），直接接管其状态快照，无需训练进程重启、无需提前用 --with-mcp 启动
    5. 创建 GuardianMCPServer，注册只读工具（+ 受限写工具，若配置开启）
    6. server.start(transport=args.transport)
    7. 阻塞运行，直到手动终止；终止不影响训练进程
    """
```

`serve` 是"轻量部署 + 按需接入 MCP"这套推荐用法的核心入口：训练可以从一开始就不带 `--with-mcp` 启动（不装 `mcp` 包，更快更轻），之后任何时候想让 Claude Code / OpenClaw 接入，单独跑一次 `run.py serve` 指向同一个 `checkpoints/`/`logs/` 目录即可，不需要动训练进程。

---

## 参数覆盖

`watch` 模式下要区分两类参数：**`--` 之前的是 guardian 自己的参数**，**`--` 之后的原样透传给训练命令**，guardian 不解析（只在重启时按 `cli_mappings` 追加/替换需要调整的项）。

```bash
python run.py watch --agent --with-mcp -- python train.py --epochs 20 --lr 0.001
                   └── guardian 参数 ──┘    └────── 训练命令，原样透传 ──────┘
```

### guardian 自身参数（`--` 之前）

| 命令行参数 | 对应配置项 | 默认值 |
|------------|------------|--------|
| `--config` | - | configs/guardian.yaml |
| `--contract` | contract.path | configs/contract.yaml |
| `--strict-contract` | contract.strict_mode | False（契约缺失时降级而非退出） |
| `--no-monitor` | monitor.enabled | True |
| `--agent` | agent.enabled | False |
| `--with-mcp` | mcp.enabled | False（`mcp` 包未安装时，即使传了该参数也只打警告，训练照常以非 MCP 模式启动，不报错退出） |
| `--transport` | mcp.transport | stdio |
| `--metric` | -（仅 analyze） | None（走 `TaskContract.select_metric()`） |

### 训练命令参数（`--` 之后，仅 `train` 便捷入口会解析）

`train` 子命令为内置 MNIST 示例保留这些参数并透传给 `train.py`；`watch` 模式下这些完全由用户的训练脚本自己定义，guardian 只通过 `contract.cli_mappings` 知道"想调 lr 该加哪个 flag"。

| 参数 | 默认值 | cli_mappings 中的典型映射 |
|------|--------|---------------------------|
| `--epochs` | 20 | - |
| `--batch_size` | 64 | `dataloader.batch_size` |
| `--lr` | 0.001 | `optimizer.lr` |
| `--device` | auto | - |
| `--resume` / `--ckpt` | None | 重启恢复必需（`resumable` 契约项） |

---

## 配置加载

```python
def load_config(path):
    """
    - 使用 pyyaml 解析 YAML
    - 环境变量覆盖: GUARDIAN_XXX
    - 命令行参数覆盖配置文件值
    - 返回合并后的 config dict
    """
```

---

## ✅ 快速校验

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| watch 基本可用 | `python run.py watch -- python train.py --epochs 1` | 训练作为子进程跑完，guardian 捕获退出码 0，输出摘要 |
| watch 守护外部脚本 | `python run.py watch -- python /tmp/other_train.py`（一个不含 guardian 的任意脚本） | 正常守护，证明不依赖内置 train.py |
| `--` 分界解析 | `watch --agent -- python train.py --agent` | guardian 启用 agent；`--agent` 原样透传给训练脚本，不被 guardian 吞掉 |
| train 便捷入口 | `python run.py train --epochs 1` | 等价于 watch 内置示例，训练 1 epoch 完成 |
| contract check | `python run.py contract check` | 逐项打印四项契约状态与判定依据 |
| preflight 命令 | `python run.py preflight` | 输出预检报告后退出 |
| preflight 缺契约 | 删除 `buildable_entry` 声明后运行 | 明确报错说明依赖该契约项，不静默失败 |
| analyze 命令 | 先训练生成 cp，再 `python run.py analyze` | 输出分析表格，含 metric_source |
| 帮助信息 | `python run.py --help` | 显示所有子命令说明，watch 标注为默认路径 |
| serve 命令 | `python run.py serve --transport stdio` | MCP server 启动，无异常 |
| --agent 未配置 key | `python run.py watch --agent -- python train.py --epochs 1`（无 API key） | 正常训练完成，agent 层自动降级，无报错 |
| requirements-core 独立可用 | 只装 `requirements-core.txt`（不装 mcp） | `watch --agent` 正常运行，`--with-mcp` 打警告后以非 MCP 模式继续，不报错退出 |

---

## ✅ 完整校验

| 检查项 | 验证方式 | 通过标准 |
|--------|----------|----------|
| 崩溃自动恢复 | 被守护脚本抛可恢复异常（如注入 OOM） | 自动从最新 ckpt 重启续训，指标不重置 |
| 配置优先级 | 环境变量 + 命令行 + 文件 | 命令行 > 环境变量 > 配置文件 |
| 未知参数 | `--` 之前传入未定义的 guardian 参数 | 友好报错，不崩溃；`--` 之后的未知参数不报错（透传给训练脚本） |
| 配置文件缺失 | 删除 guardian.yaml | 提示文件不存在，给出默认值建议 |
| contract review 闭环 | agent 产生一条 pending 提议后 `contract review` 批准 | 正式注册表新增该条目，proposals 中标记 approved |
| strict 模式退出码 | `--strict-contract` 且契约缺一项 | 启动前退出，退出码非 0，打印缺失清单 |
| --with-mcp 同进程 | `watch --with-mcp -- python train.py`，训练中用 MCP 客户端查询 | 查到的指标与训练进度实时一致 |
| serve 独立进程 | train.py 单独跑 + `run.py serve` 单独跑 | serve 能读到 train.py 写入的最新状态（跨进程） |
| --agent 决策生效 | 配置有效 API key，触发一次异常 | summary 中对应事件 `response.source` 为 agent |
| 训练时未装 mcp | `pip uninstall mcp` 后运行 `python run.py serve` | 打印明确提示（需装 requirements-mcp.txt），直接退出，不影响任何正在运行的训练进程 |
| 事后补挂 MCP | 训练不带 `--with-mcp` 已跑一段时间后，另开终端装 `requirements-mcp.txt` 并 `run.py serve` | 无需重启训练即可通过 MCP 查询到当前训练状态 |
