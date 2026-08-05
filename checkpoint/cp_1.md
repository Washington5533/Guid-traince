# cp_1 · 资源预估 (ResourceEstimator)

**文件**: `guardian/resource_estimator.py`
**阶段**: 训练前
**核心目标**: 训练启动前测量显存占用，推荐安全 batch size，预估训练时长

> **架构基线：sidecar-first**（见 [functional_overview.md](functional_overview.md#架构基线sidecar-first)）。本模块以**独立进程**运行（`run.py preflight`，或 `guardian watch` 在拉起训练前先跑一次），通过 [cp_11.md](cp_11.md) 的 `buildable_entry` 契约项 import `build_model` / `get_dataloaders` 后测量。该契约项缺失时本模块不可用，明确报错说明依赖，不静默跳过。
>
> **sidecar 形态顺带解决了一个隐患**：`_measure_memory()` 会用小 batch 实际跑前向+反向，这会污染 BatchNorm running stats、optimizer 动量、RNG 状态。嵌入模式下必须做 snapshot/restore 才能保证正式训练的可复现性；**独立进程下这些副作用随进程退出一起消失**，训练进程从干净状态启动，不需要任何快照恢复逻辑。

---

## 关键类与方法

### `ResourceEstimator`

| 方法 | 说明 |
|------|------|
| `__init__(config)` | 从 config.preflight 读取 test_batch_sizes, memory_margin |
| `preflight_check(model, dataloader_fn, device, total_samples, epochs, target_batch_size)` | 主入口，执行完整预检，返回 report dict |
| `_get_gpu_info(device)` | 获取 GPU 型号、总显存、可用显存 |
| `_measure_memory(model, dataloader_fn, device)` | 用小 batch 实际跑前向+反向，记录显存峰值 |
| `_predict_memory(measurements, total_mem)` | 两点线性回归，外推各 batch size 显存占用 |
| `_recommend_batch_size(predictions, total_mem)` | 根据 memory_margin 推荐最大安全 batch size；若绑定 cp_11 的 `TaskContract`，推荐值同时不超过 `adjustable_paths` 中 `dataloader.batch_size` 的上限约束 |
| `_estimate_time(model, dataloader_fn, device, total_samples, batch_size, epochs)` | 测量单 step 耗时，预估总训练时间 |
| `print_report(report)` | 终端表格输出预检报告 |

---

## 预检流程

```
preflight_check()
  │
  ├─ 1. 统计模型参数量 (总参数 / 可训练参数 / MB)
  │
  ├─ 2. 获取 GPU 信息 (型号 / 总显存 / 可用显存)
  │     └─ CPU 模式: 跳过显存预估，仅统计参数和时间
  │
  ├─ 3. 显存测量
  │     ├─ batch_size=1: 前向+反向 → 记录峰值
  │     ├─ batch_size=2: 前向+反向 → 记录峰值
  │     └─ batch_size=4: 前向+反向 → 记录峰值 (如配置)
  │
  ├─ 4. 线性回归
  │     └─ y = a*x + b (x=batch_size, y=显存)
  │         a = 可变开销 (每样本)
  │         b = 固定开销 (模型参数+框架)
  │
  ├─ 5. 推荐 batch_size
  │     └─ 最大 batch_size 使得 y < total_mem * (1 - margin)
  │
  └─ 6. 时间预估
        ├─ 测量 3 个 step 平均耗时
        ├─ steps_per_epoch = total_samples / batch_size
        └─ total_time = steps_per_epoch * epochs * avg_step_time
```

## 输出示例

```
╔══════════════════════════════════════════════╗
║            训练资源预估报告                    ║
╠══════════════════════════════════════════════╣
║ GPU: NVIDIA RTX 4090 (24GB)                  ║
║ 模型参数量: 0.6M | 2.4MB                     ║
║ 显存占用预估:                                ║
║   batch_size=32  → 1.2GB  ✅                  ║
║   batch_size=64  → 2.1GB  ✅                  ║
║   batch_size=128 → 3.9GB  ✅                  ║
║   batch_size=256 → 7.5GB  ✅                  ║
║ 推荐 batch_size: 256                         ║
║ 预估总用时: 12min (20 epochs, 60000 samples)  ║
╚══════════════════════════════════════════════╝
```

---

## ✅ 快速校验

> 目标：5 分钟内确认模块基本可用

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| 类可实例化 | `ResourceEstimator(config)` 不报错 | 无异常 |
| GPU 信息获取 | `_get_gpu_info(device)` 返回 dict | 包含 `available`, `name`, `total_mem_gb` |
| CPU 模式兼容 | device="cpu" 时跳过显存测量 | report 中无 memory_predictions |
| 参数量统计 | 对 MNIST CNN 调用 | param_total > 0，约 600K 参数 |
| 终端输出 | `print_report(report)` | 表格格式输出无报错 |

---

## ✅ 完整校验

> 目标：确认模块在生产场景下的准确性

| 检查项 | 验证方式 | 通过标准 |
|--------|----------|----------|
| 显存测量精度 | 对比 nvidia-smi 实际值 | 误差 < 15% |
| 线性回归合理性 | 至少 2 个采样点，R² > 0.95 | 拟合曲线单调递增 |
| batch size 推荐安全 | 推荐值实际运行不 OOM | 10 次运行 0 OOM |
| 时间预估偏差 | 实际训练完成后对比 | 偏差 < 30% |
| 大模型兼容 | 用更大模型 (如 ResNet18) 测试 | 正常运行不崩溃 |
| 多 GPU 场景 | CUDA_VISIBLE_DEVICES 设置多卡 | 正确报告第一张卡信息 |
| 无 GPU 降级 | 纯 CPU 环境运行 | 自动降级，仅输出参数+时间 |
| 配置边界值 | memory_margin=0 / 0.5 | 不崩溃，给出合理推荐 |
| 测量无副作用 | preflight 跑完后立刻正常训练，与从未跑 preflight 的一次训练对比 | 相同随机种子下 loss 曲线完全一致——独立进程下探测性前反向不污染训练状态 |
| 契约缺失明确报错 | contract 未声明 `buildable_entry` | 明确提示该命令依赖此契约项并退出，不静默跳过预检 |
| 推荐值受白名单约束 | `adjustable_paths` 中 batch_size 上限设为 64，物理可跑 256 | 推荐值不超过 64，并说明受白名单限制 |
