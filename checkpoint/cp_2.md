# cp_2 · 训练监控与异常告警 (TrainingMonitor)

**文件**: `guardian/monitor.py`
**阶段**: 训练中
**核心目标**: 采集训练指标与硬件状态，自适应检测异常，触发告警

> **架构基线：sidecar-first**（见 [functional_overview.md](functional_overview.md#架构基线sidecar-first)）。默认形态下 monitor 跑在**训练进程之外**，指标从训练脚本已有的输出通道读取（tail wandb / tensorboard / 日志文件），硬件指标独立轮询 nvidia-smi。因此：
> - **检测粒度取决于训练脚本的输出频率**，不是真正的 per-step——脚本只按 epoch 打印时就只有 epoch 级检测
> - **所有参数调整类动作经由重启生效**（kill + 从 checkpoint 用新参数续训，执行路径见 [cp_3.md](cp_3.md)），进程内即时干预（`skip_batch`、不重启改 lr）仅在嵌入模式可用
> - **决策不阻塞训练**：monitor 在训练进程外，agent 决策的 8 秒超时期间训练照常运行，这是 sidecar 形态天然带来的性质，不需要额外的异步机制

---

## 关键类与方法

### `TrainingMonitor`

| 方法 | 说明 |
|------|------|
| `__init__(config, notifier, advisor=None)` | 初始化监控规则、滑动窗口，绑定 Notifier 和可选的 AgentAdvisor |
| `on_train_start()` | 重置统计状态 |
| `poll_metrics()` | **sidecar 主路径**：从配置的指标通道（wandb 目录 / tensorboard / 日志文件）读取自上次以来的新增记录，逐条送入检测 |
| `on_step_end(step, metrics)` | **嵌入模式专属**：由训练脚本内的回调直接调用，参数即循环内的 metrics 变量 |
| `on_epoch_end(epoch, metrics)` | 每 epoch 记录验证指标（sidecar 下由 `poll_metrics` 解析出 epoch 级记录时触发） |
| `log_hardware()` | 独立轮询 nvidia-smi 采集 GPU/CPU 指标（按 hardware_log_interval 间隔），与训练进程解耦，两种模式下行为一致 |
| `_check_anomalies(step, metrics)` | 执行全部异常检测规则（纯规则，判定"是不是异常"） |
| `_check_loss_spike(loss)` | Loss 发散检测 |
| `_check_loss_stagnation(loss)` | Loss 停滞检测 |
| `_check_nan_inf(metrics)` | NaN/Inf 检测 |
| `_check_gpu_idle()` | GPU 利用率过低检测 |
| `_check_gpu_temp()` | GPU 温度过高检测 |
| `_decide_response(anomaly_event)` | 异常确认后，决定"怎么应对"：advisor 可用时问 agent，否则/超时走规则默认动作 |
| `get_metrics_history()` | 返回全部指标历史（供 summary 使用） |

### `GuardianCallback` — **嵌入模式专属**

仅当选择嵌入模式（愿意改训练脚本，换取 per-step 检测与进程内即时干预）时使用；sidecar 默认路径下不需要它，训练脚本不 import 任何 guardian 代码。

| 方法 | 说明 |
|------|------|
| `__init__(config)` | 封装 Monitor + Notifier，作为 Lightning Callback 使用 |
| `on_train_batch_end(...)` | 代理到 monitor.on_step_end |
| `on_validation_epoch_end(...)` | 代理到 monitor.on_epoch_end |

---

## 异常检测规则

### Loss 发散
```
当前 loss > 滑动窗口均值 * (1 + loss_spike_ratio)
→ 触发告警: "Loss 突增 +XX%，当前 {loss}，窗口均值 {avg}"
```

### Loss 停滞
```
最近 loss_stagnation_steps 步内 loss 下降 < loss_stagnation_threshold
→ 触发告警: "Loss 停滞 {N} 步，下降仅 {delta}"
```

### NaN/Inf
```
loss 或梯度范数出现 NaN/Inf
→ 触发紧急告警 (level=error)
```

### GPU 空转
```
GPU 利用率持续 < gpu_idle_threshold (连续 5 次采样)
→ 触发告警: "GPU 利用率低 ({util}%)，可能存在数据加载瓶颈"
```

### GPU 温度
```
GPU 温度 > gpu_temp_threshold
→ 触发告警: "GPU 温度过高 ({temp}°C)"
```

---

## 异常应对决策（中层 agent 化）

异常检测本身（上面 5 条规则）永远是确定性判断，不受 agent 影响。异常**确认之后怎么应对**，如果配置了 advisor，交给 agent 在有限动作集里选择；未配置或调用失败/超时，直接走规则默认动作。调用契约与超时/熔断细节见 [cp_9.md](cp_9.md)。有限动作集里涉及具体参数调整幅度（如 `lower_lr` 的 ratio）的上下限，统一从 cp_11 的 `adjustable_paths` 白名单读取，而不是本模块硬编码，详见 [cp_11.md](cp_11.md)。

### 有限动作集（sidecar 默认路径）

| 异常类型 | agent 可选动作 | 规则默认动作（advisor 不可用时） |
|----------|----------------|-----------------------------------|
| loss_spike | `ignore` / `restart_with_lower_lr(ratio)` / `alert_only` | `alert_only` |
| loss_stagnation | `ignore` / `suggest_lr_increase`（纯建议，不自动执行） / `alert_only` | `alert_only` |
| nan_inf | `rollback_to_last_ckpt` / `restart_with_lower_lr(ratio)` / `alert_only` | `alert_only`（level=error，永不静默） |
| gpu_idle | `ignore` / `alert_only` | `alert_only` |
| gpu_temp | `alert_only`（无自动降频动作，硬件安全不交给 agent） | `alert_only` |

**`restart_with_lower_lr` / `rollback_to_last_ckpt` 都是重启式动作**：kill 当前训练进程 → 从最近有效 checkpoint 用调整后的参数续训，统一经由 [cp_3.md](cp_3.md) 的重启机制执行。代价是回滚掉最近 checkpoint 之后已算的部分，`save_every` 间隔越大浪费越多——所以规则默认动作一律是 `alert_only`，只有 agent 明确判断"值得付这个代价"时才升级为重启式干预。

### 嵌入模式额外可用的动作（可选升级）

| 异常类型 | 额外动作 | 说明 |
|----------|----------|------|
| loss_spike | `lower_lr(ratio)` | 进程内直接改 `optimizer.param_groups`，下一步立即生效，无重启开销 |
| loss_spike / nan_inf | `skip_batch` | 跳过当前异常 batch 继续训练，**sidecar 模式下物理上无法实现** |

这两条只在嵌入模式下注册进 action_space；sidecar 模式下 agent 根本看不到它们，因此不存在"选了一个执行不了的动作"的情况。

### 决策流程

```
_check_anomalies() 判定异常 (规则，必须)
  │
  ├─ advisor 未配置 / disabled
  │    └─ 直接执行规则默认动作
  │
  └─ advisor 已配置
       ├─ 组装上下文：异常类型、最近 N 步指标、历史同类异常处理记录
       ├─ advisor.decide(anomaly_event, action_space=有限动作集)
       │    ├─ 在 agent.decision_timeout 内返回 → 执行 agent 选择的动作
       │    └─ 超时 / 报错 / 返回值不在动作集内 → 执行规则默认动作，记录 fallback 原因
       └─ 无论走哪条分支，动作和理由都记入 metrics_history，供 summary 使用
```

`nan_inf` 的告警推送本身不因 advisor 而延迟或静默——advisor 只决定"要不要回滚/降 lr 重启"，`alert_only` 这条底线始终执行。

**sidecar 模式下这段决策不占用训练时间**：monitor 在训练进程之外，`advisor.decide()` 等待的 8 秒里训练照常往前跑。代价是决策生效有延迟（发现异常 → 决策 → 重启，训练在这段时间里继续用旧参数训练了若干步），但不存在"主循环被 LLM 调用卡住"的问题。嵌入模式下则相反：`decide()` 是训练循环内的同步调用，8 秒超时就是实实在在卡住主循环 8 秒——因此嵌入模式建议把 `decision_timeout` 压到 2 秒以内，或只在 epoch 边界触发决策。

---

## 滑动窗口自适应

```python
# 维护最近 sliding_window 个 loss 值
window = deque(maxlen=sliding_window)
window.append(loss)
mean = np.mean(window)
std = np.std(window)
# 告警阈值 = mean + 2*std (自适应，不需要手动设阈值)
```

---

## ✅ 快速校验

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| 类可实例化 | `TrainingMonitor(config, notifier)` | 无异常 |
| 正常 loss 不触发 | 喂入单调递减 loss 序列 | 无告警输出 |
| Loss 发散检测 | 喂入突然跳高的 loss | 触发 loss_spike 告警 |
| NaN 检测 | 喂入 `float('nan')` | 触发 NaN 告警 |
| 指标历史 | `get_metrics_history()` | 返回完整列表 |
| advisor 未配置 | `TrainingMonitor(config, notifier, advisor=None)` 触发异常 | 直接执行规则默认动作，无报错 |
| 日志通道解析 | 准备一个含 `step 10 loss 0.5` 格式的日志文件，调用 `poll_metrics()` | 正确解析出 step 与 loss，送入检测 |
| 增量读取 | 连续两次 `poll_metrics()`，期间日志追加 3 行 | 第二次只处理新增的 3 行，不重复处理旧行 |
| sidecar 动作集不含 skip_batch | sidecar 模式下读取 loss_spike 的 action_space | 不含 `skip_batch` / `lower_lr`，只有重启式动作与 alert_only |

---

## ✅ 完整校验

| 检查项 | 验证方式 | 通过标准 |
|--------|----------|----------|
| 滑动窗口自适应 | 不同量级 loss (0.01 vs 100) | 均能正确检测异常 |
| Loss 停滞检测 | 喂入 N 步不变的 loss | 恰好 N 步后触发 |
| GPU 空转连续计数 | 连续 4 次低 + 1 次正常 | 不触发（需连续 5 次） |
| 静默期防刷屏 | 同一异常连续触发 3 次 | 仅输出 1 次（cooldown 内） |
| 硬件采集间隔 | hardware_log_interval=10 | 按间隔独立轮询 nvidia-smi，与训练进程节奏无关 |
| 多指标同时异常 | loss 和 GPU 同时异常 | 两条告警独立输出 |
| 长时间训练稳定性 | 模拟 10000 步训练 | 内存无泄漏，deque 大小恒定 |
| epoch 级指标记录 | 传入 val/mAP | 正确记录到 history |
| agent 决策超时降级 | advisor.decide 模拟阻塞超过 decision_timeout | 自动走规则默认动作；sidecar 下训练进程在这 8 秒内持续产出新指标不受影响 |
| agent 返回非法动作 | advisor 返回不在 action_space 内的值 | 拒绝执行，回退规则默认动作，记录警告 |
| agent 决策生效（sidecar） | advisor 返回 `restart_with_lower_lr(0.5)` | 训练进程被重启，从最近 ckpt 以半学习率续训，事件记入 history |
| nan_inf 底线不受影响 | advisor 配置但调用异常 | alert_only 仍立即触发，不因 advisor 故障被吞掉 |
| wandb 通道解析 | 指向一个真实 wandb run 目录 | 能读到 loss/lr 时间序列，与 wandb UI 显示一致 |
| 指标通道缺失降级 | contract 未声明 metrics_channel | 退化为进程级看护（存活 + GPU），不报错，明确日志说明 loss 级检测已关闭 |
| 检测粒度受限说明 | 训练脚本仅按 epoch 打印 loss | 只产生 epoch 级检测，不虚报 step 级事件 |
| 嵌入模式动作集扩展 | 以嵌入模式初始化 monitor | action_space 额外包含 `lower_lr` / `skip_batch` |
