# cp_5 · 日志智能摘要 (SummaryGenerator)

**文件**: `guardian/summary.py`  
**阶段**: 训练后  
**核心目标**: 训练结束自动生成结构化摘要，一页纸讲清楚训练结果

---

## 关键类与方法

### `SummaryGenerator`

| 方法 | 说明 |
|------|------|
| `__init__(config, monitor, ckpt_analyzer, watchdog=None, advisor=None)` | 绑定 monitor（获取指标历史）、ckpt_analyzer（获取 checkpoint 信息）、watchdog（获取重启记录）和可选的 AgentAdvisor |
| `generate()` | 主入口：收集所有信息，生成摘要 dict |
| `_collect_basic_info()` | 基础信息：状态、总用时、实际 epoch 数 |
| `_collect_best_model()` | 最优模型信息：哪个 epoch、什么指标（指标口径来自 cp_11 `select_metric`） |
| `_collect_final_metrics()` | 最后一个 epoch 的所有验证指标 |
| `_collect_anomaly_events()` | 从 monitor 获取全部异常事件列表（含每次事件的 agent 应对/恢复决策记录） |
| `_collect_restarts()` | 从 watchdog 获取全部重启记录，`trigger` 三取一：`crash`（进程异常退出后恢复）/ `intervention`（cp_2 主动干预）/ `hang`（无进展超时，仅在显式配置 `no_progress_kill_after` 时可能出现）；含恢复起点与作废的 epoch 数 |
| `_collect_resource_usage()` | GPU 平均利用率、显存峰值、总 GPU 时数 |
| `_collect_lr_schedule()` | 学习率调度历史（sidecar 下从指标通道读取，跨重启拼接） |
| `_generate_ai_narrative(summary)` | 可选：advisor 基于结构化摘要生成自然语言解读（纯输出，失败不影响摘要本身） |
| `print_summary(summary)` | 终端表格输出 |
| `save_summary(summary, output_dir)` | 保存为 JSON + TXT 文件 |

---

## 摘要内容结构

```json
{
  "experiment_id": "run_20260804_153022",
  "status": "completed",
  "duration": "3h 18min",
  "start_time": "2026-08-04T15:30:22",
  "end_time": "2026-08-04T18:48:11",

  "training": {
    "total_epochs": 100,
    "actual_epochs": 100,
    "total_steps": 93800,
    "final_train_loss": 0.023,
    "final_train_accuracy": 0.997
  },

  "best_model": {
    "epoch": 67,
    "checkpoint_path": "checkpoints/cp_66",
    "metric": "val/accuracy",
    "value": 0.9912
  },

  "final_validation": {
    "accuracy": 0.9901,
    "loss": 0.031,
    "per_class_accuracy": {"0": 0.993, "1": 0.988, ...}
  },

  "lr_schedule": [
    {"epoch": 0, "lr": 0.001},
    {"epoch": 30, "lr": 0.0005},
    {"epoch": 60, "lr": 0.0001}
  ],

  "anomaly_events": [
    {
      "step": 4520,
      "epoch": 23,
      "type": "loss_spike",
      "detail": "Loss 突增 +42%",
      "severity": "warning",
      "resolved": true,
      "response": {"source": "agent", "action": "restart_with_lower_lr(0.5)", "restart": true, "wasted_epochs": 2}
    },
    {
      "step": 8900,
      "epoch": 45,
      "type": "gpu_temp",
      "detail": "GPU 温度 87°C",
      "severity": "warning",
      "resolved": true,
      "response": {"source": "rule_default", "action": "alert_only"}
    }
  ],

  "resources": {
    "gpu_name": "NVIDIA RTX 4090",
    "gpu_util_avg": 94.2,
    "gpu_mem_peak_gb": 11.2,
    "gpu_hours": 3.3
  },

  "checkpoints": {
    "total_saved": 10,
    "retained": 5,
    "best": "cp_66"
  },

  "restarts": [
    {
      "epoch": 23,
      "trigger": "intervention",
      "reason": "loss_spike -> restart_with_lower_lr(0.5)",
      "resumed_from": "checkpoints/cp_22",
      "wasted_epochs": 1
    },
    {
      "epoch": 51,
      "trigger": "crash",
      "reason": "CUDA out of memory -> reduce_batch(0.5)",
      "resumed_from": "checkpoints/cp_50",
      "wasted_epochs": 1
    }
  ],

  "ai_narrative": "本次训练在 epoch 23 出现短暂 loss 突增，agent 判断为学习率偏高触发的正常震荡，自动降低学习率后收敛；GPU 温度在 epoch 45 短暂偏高但未影响训练。整体收敛稳定，建议下次训练可直接采用当前学习率调度。"
}
```

`response.source` 标注这条事件的应对/恢复动作来自 `agent`（advisor 决策生效）还是 `rule_default`（advisor 未配置或降级），方便复盘 agent 决策的实际效果。`ai_narrative` 字段仅在配置了 advisor 时生成，失败或未配置时该字段省略，不影响摘要其余部分。advisor 调用契约见 [cp_9.md](cp_9.md)。

## 终端输出示例

```
╔══════════════════════════════════════════════╗
║            训练摘要 #run_20260804             ║
╠══════════════════════════════════════════════╣
║ 状态: ✅ 完成                                ║
║ 用时: 3h 18min | 100/100 epochs              ║
║ 最优模型: epoch=67, accuracy=0.9912          ║
║ 最终指标: acc=0.990, loss=0.031              ║
║ 学习率调度: 3 次衰减 (epoch 0/30/60)         ║
║ 异常事件:                                    ║
║   ⚠️ epoch=23 loss 突增(+42%)，降lr重启后恢复 ║
║   ⚠️ epoch=45 GPU温度达87°C                  ║
║ 重启: 2 次 (1 干预 + 1 崩溃)，作废 2 epoch   ║
║ 资源: GPU均值94% | 显存峰值11.2GB            ║
║ Checkpoint: 保存10个, 保留5个, 最优cp_66     ║
╚══════════════════════════════════════════════╝
```

`restarts` 是 sidecar 形态下特有的一段：所有干预都经由重启生效（见 [cp_3.md](cp_3.md)），因此"这次训练被重启了几次、每次作废多少算力"是复盘时最该看的信息之一——它直接回答"agent 的干预到底值不值"。`wasted_epochs` 合计越接近总 epoch 数，说明 `save_every` 设得太稀疏或干预过于激进。

---

## ✅ 快速校验

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| 类可实例化 | `SummaryGenerator(config, monitor, ckpt_analyzer)` | 无异常 |
| 空监控数据 | monitor 无历史记录时 generate() | 返回空摘要，不崩溃 |
| 终端输出 | `print_summary(summary)` | 表格格式正确，无乱码 |
| 文件保存 | `save_summary(summary, "./logs")` | logs/ 下生成 json + txt 两个文件 |
| JSON 可解析 | 保存后 load JSON | 结构完整，字段齐全 |
| advisor 未配置 | `SummaryGenerator(..., advisor=None)` | 正常生成摘要，无 `ai_narrative` 字段 |
| watchdog 未绑定 | `SummaryGenerator(..., watchdog=None)` | 正常生成摘要，`restarts` 为空数组，不崩溃 |

---

## ✅ 完整校验

| 检查项 | 验证方式 | 通过标准 |
|--------|----------|----------|
| 指标准确性 | 对比 monitor 原始数据 | 所有数值与原始记录一致 |
| 异常事件完整 | 训练中触发 3 次告警 | summary 中包含 3 条事件 |
| 最优模型正确 | 对比 ckpt_analyzer.best | epoch 和 metric 值一致 |
| 时间计算 | 记录 start/end 时间 | duration 与实际差值匹配 |
| 资源统计 | 训练后对比 GPUtil 记录 | GPU 均值/峰值在 5% 误差内 |
| 中断训练摘要 | 训练在第 50 epoch 中断 | status="interrupted"，actual_epochs=50 |
| 多格式输出 | formats=["json","txt"] | 两种格式均生成且内容一致 |
| 历史查询 | 保存 5 次实验摘要 | 可按 metric 排序查询 |
| response 来源标注 | 一条事件由 agent 处理，一条走默认 | 两条 `response.source` 分别为 `agent` / `rule_default` |
| agent 解读失败降级 | advisor.narrate 抛异常 | `ai_narrative` 字段省略，摘要其余部分不受影响，仍能保存 |
| 解读内容一致性 | 对比 ai_narrative 与结构化字段 | 叙述中的数值与 summary 字段一致，无编造 |
| 重启记录区分来源 | 一次主动干预 + 一次 OOM 崩溃恢复 | `restarts` 两条，`trigger` 分别为 `intervention` / `crash` |
| 跨重启指标拼接 | 训练被重启 2 次后正常完成 | `lr_schedule` / `total_steps` 跨重启连续，不因重启断段或重复计数 |
| 作废算力统计 | save_every=5，在 epoch 12 干预重启 | 对应 `restarts[].wasted_epochs` 为 2，与 cp_3 诊断记录一致 |
| 指标口径标注 | 绑定 task_contract | `best_model.metric` 与 cp_4 的 `metric_source` 一致 |
