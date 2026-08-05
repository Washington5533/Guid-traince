# cp_3 · 训练自动恢复 (TrainingWatchdog)

**文件**: `guardian/watchdog.py`
**阶段**: 训练中
**核心目标**: 训练中断后自动诊断原因，从最近 checkpoint 恢复训练；**同时作为 sidecar 模式下所有干预动作的统一执行路径**

> **架构基线：sidecar-first**（见 [functional_overview.md](functional_overview.md#架构基线sidecar-first)）。本模块是 sidecar 形态的核心：`guardian watch -- python train.py` 就是由它包装训练命令、以子进程方式拉起训练并全程看护。它不只处理"崩溃后恢复"，还承担 [cp_2.md](cp_2.md) 里所有重启式动作（`restart_with_lower_lr` / `rollback_to_last_ckpt`）的实际执行——**因为在训练进程之外，重启是唯一可用的干预手段**。

---

## 关键类与方法

### `TrainingWatchdog`

| 方法 | 说明 |
|------|------|
| `__init__(config, notifier, advisor=None)` | 初始化重试计数、恢复策略，绑定可选的 AgentAdvisor |
| `run(train_cmd)` | **sidecar 主入口**：以子进程方式拉起训练命令（`python train.py --resume ...`），监听退出码与 OOM-kill，失败或收到干预请求时按策略重启 |
| `run(train_fn)` | **嵌入模式入口**：在同一进程内循环调用 train_fn，捕获异常后恢复 |
| `apply_intervention(action, context)` | 执行来自 cp_2 的重启式干预：kill 当前训练子进程 → 计算新参数 → 从最近有效 checkpoint 续训。与崩溃恢复共用同一套重启机制 |
| `_classify_crash(exception_or_exitcode)` | 判断中断类型：可恢复 / 不可恢复（纯规则，必须）。sidecar 下依据子进程退出码 + stderr 尾部文本；嵌入模式下依据捕获到的异常对象 |
| `_find_latest_checkpoint(ckpt_dir)` | 扫描 checkpoint 目录，找最新有效 ckpt |
| `_decide_recovery_strategy(exception, context)` | 可恢复中断确认后，决定"怎么恢复"：advisor 可用时问 agent，否则/超时走规则默认策略 |
| `_adjust_params_on_oom(exception)` | OOM 时的规则默认动作：减小 batch_size |
| `_restart_training(ckpt_path, adjusted_params)` | 拼接恢复参数，重新启动训练 |
| `_generate_diagnosis(exception, retry_count)` | 生成中断诊断报告（可选叠加 agent 自然语言解读） |
| `should_retry()` | 判断是否还能重试（未超 max_retries） |

---

## 中断分类

sidecar 模式下 guardian 拿不到训练进程内的异常对象，只能看到**退出码 + stderr 尾部文本**，因此识别方式按模式区分：

### 可恢复（自动续训）
| 异常类型 | sidecar 识别方式（退出码 + stderr） | 嵌入模式识别方式 | 恢复策略 |
|----------|-------------------------------------|------------------|----------|
| OOM (显存溢出) | 非零退出 + stderr 含 `CUDA out of memory` | `RuntimeError` 且消息匹配 OOM | 减小 batch_size * oom_batch_reduce_ratio，从最近 ckpt 恢复 |
| 进程被 kill | 退出码 -9 / 137（SIGKILL，含 cgroup OOM-killer） | 捕获不到（进程已死，由外层看护） | 直接从最近 ckpt 恢复，参数不变 |
| 网络波动 | 非零退出 + stderr 含 `ConnectionError` / `Timeout` | `ConnectionError` / `Timeout` | 等待 restart_delay 后重试 |

### 不可恢复（停止重试，推送告警）
| 异常类型 | sidecar 识别方式 | 嵌入模式识别方式 | 处理 |
|----------|------------------|------------------|------|
| 代码错误 | 非零退出 + stderr 含 `Traceback` 且不匹配上表任何可恢复模式 | `TypeError` / `AttributeError` / `RuntimeError`(非OOM) | 输出 stderr 尾部，推送告警 |
| 数据损坏 | stderr 含 `FileNotFoundError` / `EOFError` | 同名异常 | 推送告警，建议检查数据 |
| 连续失败 | retry_count >= max_retries | 同 | 停止重试，输出诊断报告 |

**分类保守优先**：sidecar 下 stderr 文本匹配不到任何已知可恢复模式时，一律判为不可恢复并停止重试——宁可停下来等人看，也不要对一个真正的代码 bug 反复重启浪费算力。

### 第三类：无退出的故障（挂起）

上面两张表都建立在"进程退出了、有退出码可读"的前提上。但**最贵的故障往往不退出**：死锁、dataloader worker 卡死、NCCL 集合通信 hang、GPU 掉卡后进程僵住。这类故障下进程还活着、GPU 可能还显示占用，但训练已经不再前进——没有退出码，前面整套分类逻辑完全不触发。

sidecar 恰好适合发现这件事（进程外能独立观察），判据是**两条同时成立**：

```
1. 指标通道在 no_progress_timeout 内没有任何新记录（step/epoch 都没往前走）
2. 进程仍然存活（未退出）
```

辅助信号（用于降低误判，不单独作为判据）：GPU 利用率持续接近 0（配合 cp_2 的 `gpu_idle` 采集）。

处理策略比崩溃更保守——**因为"慢"和"挂"从外部看是一样的**：

| 阶段 | 行为 |
|------|------|
| 达到 `no_progress_timeout` | 推送 warning 告警（"疑似无进展，已 N 分钟无新指标"），**不动训练** |
| 达到 `no_progress_kill_after`（默认为 `null`，即不自动处理） | 仅当用户显式配置了该值，才 kill 并从最近 checkpoint 重启，按 `resume_unchanged` 处理 |

默认只告警不动手：一个 epoch 本来就要跑 40 分钟的任务，把 `no_progress_timeout` 设成 10 分钟会导致 guardian 反复误杀正常训练。**这个阈值必须由用户按自己任务的 epoch 时长设定，guardian 不猜默认值**；未配置 `no_progress_kill_after` 时永不因挂起而重启。

```yaml
watchdog:
  no_progress_timeout: 1800        # 秒，多久没有新指标算"疑似无进展"（触发告警）
  no_progress_kill_after: null     # 秒，多久后 kill 重启；null = 永不自动处理，只告警
```

挂起检测依赖 `metrics_channel` 契约项——没有指标通道就无法判断"是否还在前进"，此时该能力自动关闭（与 cp_11 的其余降级项一致）。

---

## 恢复流程

两个触发来源汇入同一条重启路径：**被动**（训练子进程自己挂了）和**主动**（cp_2 检测到异常后决定干预）。

```
[被动] 训练子进程退出，退出码 != 0        [主动] cp_2 判定异常 + 决定重启式干预
  |                                          |
  |- 读取退出码 + stderr 尾部                 |- apply_intervention(action, context)
  |                                          |     action = restart_with_lower_lr / rollback_to_last_ckpt
  |- _classify_crash()  [纯规则]              |
  |    |- 可恢复?                             |- kill 当前训练子进程（SIGTERM，超时后 SIGKILL）
  |    |    |                                 |
  |    |    |- _decide_recovery_strategy()     `----> 汇入同一条重启路径 ------.
  |    |    |    |- advisor 未配置/超时/报错 -> 规则默认策略                    |
  |    |    |    |    OOM -> _adjust_params_on_oom(): batch_size *= 0.5        |
  |    |    |    |    kill / 网络波动 -> 参数不变，直接续训                     |
  |    |    |    `- advisor 已配置 -> agent 在有限动作集里选择（见下）           |
  |    |    |                                                                 |
  |    |    |- _find_latest_checkpoint()  <-------------------------------------'
  |    |    |     扫描 checkpoints/ -> 按 epoch 降序 -> 选第一个有效文件
  |    |    |
  |    |    `- _restart_training(ckpt_path, adjusted_params)
  |    |         等待 restart_delay 秒
  |    |         sidecar: 拼命令行重新拉起子进程（python train.py --resume --ckpt ... --lr ...）
  |    |         嵌入:   重新调用 train_fn(adjusted_params)
  |    |
  |    `- 不可恢复?
  |         |- _generate_diagnosis()（可选叠加 agent 自然语言解读，纯输出不影响流程）
  |         `- notifier.send("训练不可恢复", diagnosis, level="error")
  |
  `- should_retry()
       |- retry_count < max_retries -> 继续
       `- retry_count >= max_retries -> 停止，输出最终诊断
```

---

## 动作执行层（`apply_action`）

前面定义了"agent 能选哪些动作"，这一节定义**这些动作到底怎么落到一条新的命令行上**。这是 sidecar 唯一的干预实现，必须是确定性的、可测的，不能停在"拼接恢复参数"这种描述上。

### 命令行改写规则

```
原始命令（用户在 -- 之后给的，原样保存）：
  python train.py --epochs 100 --batch_size 64 --lr 0.001 --data ./data

动作：reduce_batch(0.5) + 从 cp_10 恢复
  │
  ├─ 1. 查 contract.cli_mappings：dataloader.batch_size -> "--batch_size"
  ├─ 2. 读当前生效值：原命令里 --batch_size 64（原命令没写则取 checkpoint 里记录的值，
  │       两者都没有则该动作不可执行 -> 回退默认动作）
  ├─ 3. 算新值：64 * 0.5 = 32，校验 >= min_value(8) 且降幅 <= max_delta_ratio
  ├─ 4. 替换而非追加：--batch_size 64 -> --batch_size 32
  │       （同一 flag 出现多次时行为依赖用户脚本的 argparse，必须替换，不能追加）
  └─ 5. 追加恢复参数：resume_flag + ckpt_flag（来自 contract.resumable）

新命令：
  python train.py --epochs 100 --batch_size 32 --lr 0.001 --data ./data --resume --ckpt checkpoints/cp_10
```

四条硬性规则：

1. **只替换、不追加**：已存在的 flag 必须原地替换，避免同名参数出现两次导致行为依赖 argparse 实现细节
2. **不认识的参数原样保留**：`--data ./data` 这类 guardian 不理解的参数完整透传，一个字都不改
3. **改写前后 diff 必须落盘**：每次重启在诊断记录里存 `cmd_before` / `cmd_after`，人能一眼看出改了什么
4. **当前值取不到就放弃该动作**：`reduce_batch` 需要知道"现在 batch 是多少"才能算 50%。原命令没显式写、checkpoint 也没记录时，不猜默认值，直接回退到 `resume_unchanged`

### 需要保持的语义等价

`enable_grad_accum(steps)` 这类动作要小心：把 batch 减半 + 梯度累积 2 步，数学上才等价于原 batch。执行层必须成对调整而不是只改一个：

```
enable_grad_accum(2)  →  --batch_size 32 --grad_accum_steps 2   (原 batch=64)
                          两个 flag 必须同时改写，缺一个就不是等价变换
```

若 `cli_mappings` 里缺少 `--grad_accum_steps` 的映射，则整个 `enable_grad_accum` 动作不注册（见上文动作集说明），而不是只改 batch 就当做完成了。

### 明确不支持的场景（v0 已知边界）

这些场景下 guardian **不尝试猜**，直接降级为 `resume_unchanged` 或 `alert_only`，并在日志里说明原因：

| 场景 | 为什么不支持 | 行为 |
|------|--------------|------|
| 参数写在 yaml/config 文件里而非命令行 | 改用户的配置文件属于侵入，且不知道该文件的 schema | 该路径不可调，日志说明"建议将该参数暴露为命令行参数" |
| DDP / torchrun 多进程启动 | `torchrun --nproc_per_node=4 train.py` 下改 batch_size 的语义是 per-GPU 还是全局，取决于用户脚本 | v0 不做参数调整，只做原样重启；契约里可显式声明 `batch_semantics: per_gpu \| global` 后在 v1 支持 |
| batch size 与 LR scaling 联动 | 减半 batch 后是否要同步减半 lr，属于训练策略决策，不是执行层能定的 | 只改被明确选中的路径；如需联动，由 agent 显式选择组合动作 |
| 参数经环境变量传入 | 同 yaml，schema 未知 | 该路径不可调 |

**这张表是设计的一部分，不是遗漏。** 明确说"这些情况我不动手"，比让 guardian 去猜一个可能破坏训练语义的改写要安全得多。

### 重启式干预的算力代价

主动干预和被动恢复共用重启机制，代价也一样：**回滚掉最近 checkpoint 之后已经算过的部分**。

```
save_every=5 epochs，在 epoch 12 检测到 loss 发散并决定降 lr 重启
  -> 最近有效 checkpoint 是 cp_10
  -> epoch 11、12 的计算全部作废，从 epoch 10 用新 lr 重来
  -> 浪费约 2 个 epoch 的算力
```

所以：`save_every` 越小，重启式干预越划算，但 checkpoint 的磁盘和 I/O 开销越大。规则默认动作一律是 `alert_only`（见 cp_2），只有 agent 明确判断"继续训下去的损失大于回滚 N 个 epoch"时才升级为重启式干预；`_generate_diagnosis` 里应记录本次重启浪费的 epoch 数，供事后复盘干预是否值得。

## 恢复策略决策（中层 agent 化）

`_classify_crash` 判定"可恢复/不可恢复"永远是规则引擎的活，这类判断关系到训练要不要继续，不能等一次 LLM 调用，也不能在离线环境里失效。**可恢复**确认之后，"具体怎么恢复"如果配置了 advisor，交给 agent 在有限动作集里选择；未配置或调用失败/超时，直接走规则默认策略（等同于改动前的行为，向下兼容）。动作集里 `reduce_batch(ratio)`、`enable_grad_accum(steps)` 等的合法范围，统一从 cp_11 的 `adjustable_paths` 白名单读取，详见 [cp_11.md](cp_11.md)。

### 有限动作集

这三类动作**本来就是重启式的**，因此 sidecar 与嵌入模式下完全一致，不需要区分——这也是为什么恢复策略这个决策点在 sidecar 化之后一条动作都不用删（对比 cp_2 的异常应对动作集被砍掉了 `skip_batch`）。

| 中断类型 | agent 可选动作 | 规则默认动作 | 命令行映射（sidecar） |
|----------|----------------|--------------|------------------------|
| OOM | `reduce_batch(ratio)` / `enable_grad_accum(steps)` / `reduce_batch_and_grad_accum` | `reduce_batch(0.5)` | `--batch_size` / `--grad_accum_steps` |
| 进程被 kill | `resume_unchanged` / `resume_with_reduced_workers` | `resume_unchanged` | 无 / `--num_workers` |
| 网络波动 | `resume_after_delay(seconds)` / `resume_unchanged` | `resume_after_delay(restart_delay)` | 无（仅 guardian 侧等待） |

`enable_grad_accum` 和 `resume_with_reduced_workers` 要求训练脚本支持对应的命令行参数（`--grad_accum_steps` / `--num_workers`）。脚本不支持时，该动作在 sidecar 模式下不注册进 action_space——agent 看不到就不会选，不存在选了执行不了的情况。

agent 的选择依据：历史恢复记录（哪种策略在过去成功率更高）+ 当前异常上下文（重试次数、显存余量、上次调整幅度）。这部分记录由 `_generate_diagnosis` 一并保存，供下次决策参考，也供 [cp_10.md](cp_10.md) 的 MCP 工具查询。

## Checkpoint 扫描规则

```python
# 扫描 checkpoint 目录，按 epoch 降序排列
# 有效性检查：文件非空、可被 torch.load 读取、包含 epoch 字段
# 优先选择 monitor_metric 最优的，而非最新的
```

---

## ✅ 快速校验

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| 类可实例化 | `TrainingWatchdog(config, notifier)` | 无异常 |
| 正常训练不触发 | 传入必定成功的 train_cmd（如 `python -c "print(1)"`） | 执行 1 次，退出码 0，无重试 |
| 子进程拉起 | `run("python train.py --epochs 1")` | 训练作为子进程正常跑完，guardian 正确捕获退出码 0 |
| OOM 分类（sidecar） | 子进程以非零码退出且 stderr 含 "CUDA out of memory" | classify 返回 "recoverable" |
| 代码错误分类（sidecar） | 子进程 stderr 含 `TypeError` 的 Traceback | classify 返回 "unrecoverable"，不重启 |
| 未知错误保守处理 | stderr 是无法识别的报错文本 | classify 返回 "unrecoverable"，停止重试而非反复重启 |
| 重试计数 | should_retry() 连续调用 | 达到 max_retries 后返回 False |
| 命令行改写：替换 | 原命令含 `--batch_size 64`，执行 `reduce_batch(0.5)` | 新命令为 `--batch_size 32`，该 flag 只出现一次 |
| 命令行改写：透传 | 原命令含 `--data ./data --seed 42` | 新命令原样保留这两个参数，一字未改 |
| 挂起检测触发告警 | 指标通道 `no_progress_timeout` 内无新记录且进程存活 | 推送 warning，**不 kill 训练** |
| 挂起默认不动手 | 同上，`no_progress_kill_after: null`（默认） | 只告警，永不因挂起重启训练 |

---

## ✅ 完整校验

| 检查项 | 验证方式 | 通过标准 |
|--------|----------|----------|
| OOM 自动减 batch | 初始 64，OOM 后恢复 | 恢复时 batch_size=32 |
| batch 下限保护 | batch=8 时 OOM | 减小到 4 后不再减小 |
| Checkpoint 扫描 | 目录含 3 个 ckpt | 选出 epoch 最大的 |
| 损坏 ckpt 跳过 | 一个 ckpt 文件为空 | 自动跳过，选下一个 |
| 连续失败停止 | 训练函数连续抛异常 | max_retries 次后停止 |
| 诊断报告内容 | 不可恢复中断 | 报告含异常类型、堆栈、建议 |
| 恢复后 wandb 接续 | 模拟恢复场景 | 输出日志提示接续 run_id |
| restart_delay 生效 | OOM 恢复 | 恢复前等待指定秒数 |
| advisor 未配置降级 | `TrainingWatchdog(config, notifier, advisor=None)` OOM | 走规则默认策略 `reduce_batch(0.5)`，行为与改动前一致 |
| agent 决策超时降级 | advisor.decide 模拟阻塞超过 decision_timeout | 自动执行规则默认策略，不阻塞恢复流程 |
| agent 返回非法动作 | advisor 返回不在 action_space 内的值 | 拒绝执行，回退规则默认策略，记录警告 |
| agent 选择 grad_accum | OOM 场景，advisor 返回 `enable_grad_accum(4)` | 实际按梯度累积恢复，而非单纯减 batch |
| 历史记录参与决策 | 连续 3 次 OOM，前两次 reduce_batch 均再次 OOM | 第 3 次上下文中包含前两次失败记录，供 agent 参考 |
| classify_crash 不受 advisor 影响 | advisor 配置但调用异常 | 可恢复/不可恢复判定结果不变，仅恢复策略走默认值 |
| 主动干预汇入重启路径 | cp_2 触发 `restart_with_lower_lr(0.5)` | 子进程被 kill，以 `--lr` 减半 + `--resume` 重新拉起，与被动恢复走同一套 `_restart_training` |
| kill 优雅退出 | 主动干预时训练子进程正在写 checkpoint | 先 SIGTERM 等待宽限期，超时才 SIGKILL，不产生半截 checkpoint 文件 |
| 命令行映射缺失被拒 | 训练脚本无 `--grad_accum_steps`，agent 返回 `enable_grad_accum(4)` | 该动作不在 action_space 内（或被 `_validate_action` 拒绝），回退 `reduce_batch(0.5)` |
| 浪费算力记录 | save_every=5，在 epoch 12 主动干预重启 | 诊断记录中标注从 cp_10 恢复、作废约 2 个 epoch |
| SIGKILL/137 识别 | 用 `kill -9` 杀掉训练子进程 | 识别为可恢复，从最近 ckpt 参数不变续训 |
| 训练脚本不可续训 | 脚本无 `--resume` 参数（cp_11 契约缺失） | 自动恢复能力关闭，崩溃时仅告警并给出契约缺失说明，不盲目重启 |
| 当前值取不到则放弃 | 原命令未写 `--batch_size`，checkpoint 也无记录，触发 OOM | 不猜默认值，降级为 `resume_unchanged`，日志说明原因 |
| grad_accum 成对改写 | OOM 场景，agent 返回 `enable_grad_accum(2)`，原 batch=64 | 新命令同时含 `--batch_size 32` 与 `--grad_accum_steps 2`，保持语义等价 |
| 改写 diff 落盘 | 任意一次带参数调整的重启 | 诊断记录含 `cmd_before` / `cmd_after`，可直接比对差异 |
| yaml 参数不动手 | 参数只存在于 config.yaml，未暴露为命令行 | 该路径不可调，不修改用户配置文件，日志给出建议 |
| DDP 场景保守处理 | 用 `torchrun --nproc_per_node=4` 启动并触发 OOM | v0 只做原样重启，不调整 batch_size，日志说明 DDP 下语义未声明 |
| 慢训练不被误判为挂起 | 单 epoch 耗时 > `no_progress_timeout` 的任务 | 仅在配置了 `no_progress_kill_after` 时才可能重启；默认只告警，训练不被打断 |
| 挂起后可选重启 | 显式设 `no_progress_kill_after`，脚本挂起 | 超时后 kill 并按 `resume_unchanged` 续训，记录 `trigger=hang` |
| 挂起检测依赖指标通道 | 契约无 `metrics_channel`，脚本挂起 | 该能力自动关闭，不误判、不重启，明确日志说明 |
