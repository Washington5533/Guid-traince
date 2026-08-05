# cp_4 · 断点分析与校验 (CheckpointAnalyzer)

**文件**: `guardian/checkpoint_analyzer.py`
**阶段**: 训练后 / 训练中
**核心目标**: 管理 checkpoint 目录，对每个 cp_{epoch} 执行快速校验和完整校验，追踪最优模型

> **架构基线：sidecar-first**（见 [functional_overview.md](functional_overview.md#架构基线sidecar-first)）。sidecar 下本模块**不负责保存 checkpoint**——保存由训练脚本自己做（[cp_11.md](cp_11.md) 的 `checkpoint_schema` 契约项规定必需字段）。guardian 只做进程外的三件事：轮询发现新出现的 `cp_*/` 目录、对其做校验、维护 `best.pth` 与 top-k。因此：
> - `save_checkpoint()` 仅在**嵌入模式**下使用；sidecar 下改为 `watch_ckpt_dir()` 轮询发现
> - `quick_validate` / `full_validate` 需要 `buildable_entry` 契约项（要重建模型结构才能加载权重跑评估），缺失时只读 `metrics.json` 里训练脚本自己记的指标，不做独立评估
> - **必须处理"半截文件"**：训练进程可能正在写 checkpoint，guardian 在进程外无法知道写没写完，只能靠原子性判据（见下方"并发安全"）

---

## Checkpoint 目录结构

```
checkpoints/
├── cp_2/
│   ├── model.pth           # 模型权重
│   ├── optimizer.pth       # 优化器状态
│   ├── metrics.json        # 训练指标快照
│   ├── quick_val.json      # 快速校验结果
│   └── full_val.json       # 完整校验结果 (每 N 个 cp 生成)
├── cp_4/
│   ├── model.pth
│   ├── metrics.json
│   ├── quick_val.json
│   └── full_val.json
├── cp_6/
│   ├── ...
├── best.pth                # 自动维护的最优模型软链接
└── analysis_report.json    # 全量 checkpoint 分析报告
```

---

## 关键类与方法

### `CheckpointAnalyzer`

| 方法 | 说明 |
|------|------|
| `__init__(config, model, val_loader, device, notifier, task_contract=None)` | 初始化；`model`/`val_loader` 在 sidecar 下经 `buildable_entry` 契约 import 得到（缺失则以"只读 metrics.json"降级模式运行）。可选绑定 cp_11 的 `TaskContract`，用于 best-model 判定指标选择 |
| `watch_ckpt_dir()` | **sidecar 主路径**：按间隔轮询 `checkpoints/`，发现新的、且已写完（见"并发安全"）的 `cp_*/` 目录时触发校验与 best/top-k 维护 |
| `save_checkpoint(epoch, model, optimizer, metrics, ckpt_dir)` | **嵌入模式专属**：保存一个 cp_{epoch} 目录。sidecar 下保存由训练脚本自己完成，本方法不参与 |
| `quick_validate(epoch, ckpt_dir)` | **快速校验**：用 sample_ratio 的子集评估，写入 quick_val.json |
| `full_validate(epoch, ckpt_dir)` | **完整校验**：全量验证集评估，写入 full_val.json |
| `should_full_validate(epoch)` | 判断当前 cp 是否需要完整校验（every_n_checkpoints） |
| `update_best(ckpt_dir, metric_value)` | 对比历史，更新 best.pth 软链接；`metric_value` 采用的口径由 `task_contract.select_metric()` 给出（未绑定 task_contract 时回退配置里手写的 `monitor_metric`，向下兼容） |
| `cleanup_top_k()` | 保留 top_k 个 checkpoint，删除其余 |
| `analyze_all()` | 扫描所有 cp_ 目录，生成 analysis_report.json |
| `compare_checkpoints(cp_a, cp_b)` | 对比两个 checkpoint的指标差异 |
| `get_best_checkpoint()` | 返回最优 checkpoint 路径 |

---

## 快速校验 vs 完整校验

### 快速校验 (每次 cp 保存后执行)
```python
def quick_validate(epoch, ckpt_dir):
    """
    - 加载 ckpt 中的模型权重
    - 从 val_loader 中随机抽取 sample_ratio (5%) 的样本
    - 跑推理，计算 accuracy / loss
    - 结果写入 quick_val.json:
      {
        "epoch": 4,
        "sample_count": 500,
        "accuracy": 0.962,
        "loss": 0.123,
        "time_seconds": 2.3
      }
    """
```

### 完整校验 (每 N 个 cp 执行一次)
```python
def full_validate(epoch, ckpt_dir):
    """
    - 加载 ckpt 中的模型权重
    - 用全量 val_loader 评估
    - 计算 accuracy / loss / precision / recall / F1 (per-class)
    - 生成混淆矩阵
    - 结果写入 full_val.json:
      {
        "epoch": 10,
        "sample_count": 10000,
        "accuracy": 0.987,
        "loss": 0.042,
        "per_class": {"0": 0.99, "1": 0.98, ...},
        "confusion_matrix": [[...]],
        "time_seconds": 45.2
      }
    """
```

### 调度规则

sidecar 下"save"这一步由训练脚本完成，guardian 从"发现新目录"开始接手：

```
训练脚本写出 cp_2/  ──轮询发现且判定写完──→  quick_validate → update_best
训练脚本写出 cp_4/  ──轮询发现且判定写完──→  quick_validate → update_best
...
训练脚本写出 cp_10/ ──轮询发现且判定写完──→  quick_validate → full_validate → update_best → cleanup_top_k
```

嵌入模式下则是同步的 `save → quick_validate → update_best`（每 save_every_n_epochs，full_validate 每 N 个 cp 一次）。

---

## 并发安全（sidecar 特有）

guardian 在进程外轮询目录，可能撞上训练脚本**正在写** checkpoint 的瞬间。判定"写完了"用两条判据，任一不满足就跳过本轮、下次轮询再看：

1. **文件稳定性**：目录内所有文件的 size 与 mtime 在连续两次轮询间隔内不再变化
2. **可加载性**：`torch.load` 能成功读出且含 `checkpoint_schema.required_keys` 里的全部必需键

推荐训练脚本用**原子写**（先写 `cp_10.tmp/` 再 `os.rename` 为 `cp_10/`），这样 guardian 只要看到目标目录存在就一定是完整的——`rename` 在同一文件系统内是原子操作。这条作为 `checkpoint_schema` 的推荐实践写入契约文档，不强制（不用原子写时靠上面两条判据兜底，只是多等一个轮询周期）。

**guardian 从不删除训练脚本正在使用的 checkpoint**：`cleanup_top_k()` 跳过最近 `keep_recent`（默认 2）个 epoch 的目录，避免删掉训练脚本下一次 resume 可能要用的文件。

---

## 分析报告 (analysis_report.json)

```json
{
  "total_checkpoints": 10,
  "best_checkpoint": {
    "epoch": 16,
    "path": "checkpoints/cp_16",
    "metric_value": 0.9912
  },
  "metric_source": {
    "name": "accuracy",
    "direction": "max",
    "source": "agent_inferred"
  },
  "checkpoints": [
    {
      "epoch": 2,
      "quick_val": {"accuracy": 0.945, "loss": 0.189},
      "full_val": null,
      "is_top_k": false,
      "deleted": true
    },
    {
      "epoch": 10,
      "quick_val": {"accuracy": 0.987, "loss": 0.042},
      "full_val": {"accuracy": 0.986, "loss": 0.044, "per_class": {...}},
      "is_top_k": true,
      "deleted": false
    }
  ],
  "metric_trend": [0.945, 0.962, 0.975, 0.981, 0.987, ...],
  "best_epoch": 16,
  "improvement_rate": "最近 5 个 cp 平均提升 0.2%"
}
```

`metric_source.source` 取值 `config_explicit`（config 里显式声明了 task_type/指标）/ `agent_inferred`（cp_11 的 `select_metric()` 按任务线索推断）/ `fallback`（未绑定 task_contract 或推断失败，退回 `val_loss`）——三种情况下 `best_checkpoint` 的判定逻辑完全一致，只是指标口径的来源不同，方便复盘"最优模型是按什么标准选出来的"。

---

## ✅ 快速校验

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| 保存 checkpoint | `save_checkpoint(epoch=2, ...)` | cp_2/ 目录生成，含 3 个文件 |
| 快速校验运行 | `quick_validate(2, "cp_2")` | quick_val.json 生成，含 accuracy |
| 校验调度 | `should_full_validate(5)` | False; `should_full_validate(10)` → True |
| best 更新 | 连续 save 3 个不同指标的 cp | best.pth 指向指标最优的 |
| top_k 清理 | 保存 8 个 cp (top_k=5) | 仅保留 5 个，旧目录删除 |
| task_contract 未绑定 | `CheckpointAnalyzer(..., task_contract=None)` | 沿用配置里的 `monitor_metric`，行为与改动前一致 |

---

## ✅ 完整校验

| 检查项 | 验证方式 | 通过标准 |
|--------|----------|----------|
| 完整校验精度 | full_validate 与手动 eval 对比 | 指标完全一致 |
| per-class 指标 | 10 类 MNIST | 每类都有 precision/recall |
| 混淆矩阵 | full_validate 输出 | 10x10 矩阵，行列和正确 |
| 损坏 ckpt 容错 | 手动截断 model.pth | 校验失败时写入 error 信息，不崩溃 |
| 跨 epoch 趋势 | analyze_all() 运行 | metric_trend 单调/符合预期 |
| 磁盘空间 | 大量 cp 后 cleanup | 旧目录确实被删除 |
| 并发安全：撞上半截文件 | guardian 轮询时训练脚本正在写 cp_10/ | 判定未写完，跳过本轮，下次轮询正常校验，不报错也不误判为损坏 |
| 原子写路径 | 训练脚本用 `cp_10.tmp/` → rename | guardian 只在 rename 完成后看到目录，首次轮询即可校验 |
| 轮询发现新 cp | sidecar 模式，训练脚本写出 3 个 cp | 全部被发现并校验，与嵌入模式下 save_checkpoint 触发的结果一致 |
| 不删最近 checkpoint | top_k=2，keep_recent=2，已有 6 个 cp | 最近 2 个 epoch 的目录一定保留，不影响训练脚本下次 resume |
| best.pth 软链接 | 跨文件系统（如 NAS） | 不支持软链接时回退为复制 |
| 快速校验一致性 | 同一 ckpt 多次 quick_val | 因随机采样结果略有差异，但在 2% 内 |
| compare_checkpoints | cp_4 vs cp_10 | 输出指标差值、参数差异 |
| select_metric 集成 | 绑定 task_contract，任务为分类 | `metric_source.name` 为 accuracy/f1_macro 而非硬编码值，且与 cp_11 记录一致 |
| 无 buildable_entry 降级 | contract 未声明可 import 入口 | 跳过独立评估，只读 `metrics.json` 里训练脚本自记的指标做 best/趋势，不崩溃且明确标注降级 |
