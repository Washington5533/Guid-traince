# cp_7 · MNIST 训练脚本 (train.py)

**文件**: `train.py`
**阶段**: 核心
**核心目标**: 经典手写数字识别训练脚本，作为 Guardian 的参考被守护对象——**脚本本身不 import 任何 guardian 代码**，只满足 [cp_11.md](cp_11.md) 的脚本契约

> **架构基线：sidecar-first**（见 [functional_overview.md](functional_overview.md#架构基线sidecar-first)）。本文档的主示例是"一个干净的训练脚本 + guardian 在外部守护"，这也是 guardian 对任何真实项目的默认接入方式。嵌入式集成（在循环内插回调）保留在文末附录，作为可选的精细化升级路径。

---

## 模型结构

### `SimpleCNN`

```
输入: (1, 28, 28) 灰度图
  │
  ├─ Conv2d(1, 32, 3) + ReLU + MaxPool(2)    → (32, 13, 13)
  ├─ Conv2d(32, 64, 3) + ReLU + MaxPool(2)   → (64, 5, 5)
  ├─ Flatten                                   → (1600)
  ├─ Linear(1600, 128) + ReLU + Dropout(0.5)  → (128)
  └─ Linear(128, 10)                           → (10)

参数量: ~600K
```

---

## 脚本结构

### 数据加载
```python
def get_dataloaders(config):
    """
    - torchvision.datasets.MNIST
    - 自动下载到 ./data/
    - 训练集: 60000 张, 验证集: 10000 张
    - 预处理: ToTensor + Normalize(0.1307, 0.3081)
    - 返回: (train_loader, val_loader, total_train_samples)
    """
```

### 训练循环
```python
def train_epoch(model, loader, optimizer, device):
    """单 epoch 训练，返回平均 loss 和 accuracy"""

def validate(model, loader, device):
    """全量验证，返回 accuracy, loss, per_class_acc"""
```

### 主流程（sidecar 默认路径：脚本里没有 guardian）

```python
def main(args):
    # 训练脚本只做训练该做的事，不 import guardian，不插回调
    config = load_config(args.config)
    model = build_model()                        # 契约要求：可被外部 import
    train_loader, val_loader, n = get_dataloaders(config)   # 契约要求：可被外部 import
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 契约要求 1：可续训
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(resolve_ckpt(args.ckpt))          # 含 epoch / model / optimizer / rng_state
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, args.device)
        val_acc, val_loss = validate(model, val_loader, args.device)

        # 契约要求 2：指标可观测（wandb / tensorboard / 结构化日志三选一即可）
        wandb.log({"epoch": epoch, "train/loss": train_loss, "val/accuracy": val_acc})
        print(f"epoch {epoch} train_loss {train_loss:.4f} val_acc {val_acc:.4f}", flush=True)

        # 契约要求 3：checkpoint 含必需字段
        if epoch % args.save_every == 0:
            save_ckpt(f"checkpoints/cp_{epoch}", epoch, model, optimizer)
```

对应的命令行参数（供 [cp_3.md](cp_3.md) 重启时拼接）：`--resume` / `--ckpt` / `--lr` / `--batch_size` / `--epochs` / `--save_every`。

**guardian 在外部守护，不进这个文件：**

```bash
# 全功能守护：进程看护 + 指标监控 + 崩溃恢复 + 摘要，train.py 0 行改动
guardian watch --config configs/guardian.yaml -- \
    python train.py --epochs 20 --resume --ckpt latest

# 训练前预检（独立命令，通过契约声明的 build_model/get_dataloaders 入口 import）
python run.py preflight

# 训练后分析（独立扫描 checkpoints/ 目录）
python run.py analyze
```

---

## 集成点（sidecar 默认路径）

guardian 各模块都在训练进程之外工作，"集成位置"是**外部触发点**而非脚本内的插入点：

| 模块 | 运行位置 | 数据来源 / 触发时机 |
|------|----------|---------------------|
| ResourceEstimator | `run.py preflight` 独立进程 | 通过契约声明的 `build_model` / `get_dataloaders` 入口 import 后测量；训练启动前 |
| TrainingMonitor | guardian 进程 | tail wandb 目录 / 日志文件 + 独立轮询 nvidia-smi；按 poll 间隔 |
| TrainingWatchdog | guardian 进程（**主路径**） | 包装训练命令、监听子进程退出码；崩溃时或收到 cp_2 干预请求时 |
| CheckpointAnalyzer | guardian 进程 / `run.py analyze` | 独立扫描 `checkpoints/` 目录；发现新 cp 目录时或训练后 |
| SummaryGenerator | guardian 进程 | 汇总上述各模块落盘的结果；训练结束后 |
| Notifier | guardian 进程 | 被 Monitor / Watchdog / Analyzer 调用 |
| AgentAdvisor | guardian 进程 | 注入给 Monitor / Watchdog / SummaryGenerator / TaskContract，未配置时零成本降级 |
| TaskContract | guardian 进程 | 启动时校验脚本契约四项；`select_metric` / `select_adjust_path` 决策点 |
| GuardianMCPServer | 独立进程（`run.py serve`）或 guardian 进程内后台线程 | 全程，供外部 agent 客户端查询/操作 |

**训练脚本对以上全部无感知**——它只是被 `guardian watch` 拉起的一个普通子进程。

---

## 附录：嵌入式集成（可选精细化升级）

仅当需要 per-step 检测或进程内即时干预（`skip_batch`、不重启改 lr）时才选择这条路径。**代价是训练脚本必须改**：非 Lightning 项目还需要把循环改造成能在正确位置插入回调。

```python
def main(args):
    config = load_config("configs/guardian.yaml")

    advisor = AgentAdvisor(config["agent"]) if config["agent"].get("enabled") else None
    notifier = Notifier(config["notifier"])
    monitor = TrainingMonitor(config["monitor"], notifier, advisor=advisor)   # 嵌入模式：action_space 含 skip_batch / lower_lr
    ckpt_analyzer = CheckpointAnalyzer(..., task_contract=contract)
    summary_gen = SummaryGenerator(..., advisor=advisor)

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            loss = train_step(model, batch, optimizer)
            monitor.on_step_end(step, {"loss": loss})    # per-step 检测；可进程内改 optimizer

        val_acc, val_loss = validate(...)
        monitor.on_epoch_end(epoch, {"val/accuracy": val_acc})

        if epoch % save_every == 0:
            ckpt_analyzer.save_checkpoint(...)
            ckpt_analyzer.quick_validate(...)
            if ckpt_analyzer.should_full_validate(epoch):
                ckpt_analyzer.full_validate(...)
            ckpt_analyzer.update_best(...)
            ckpt_analyzer.cleanup_top_k()

    summary = summary_gen.generate()
    summary_gen.print_summary(summary)
```

Lightning 项目可以用 `GuardianCallback`（见 [cp_2.md](cp_2.md)）零改动接入这条路径：`Trainer(callbacks=[GuardianCallback(config)])`。

**嵌入模式注意事项**：`advisor.decide()` 是训练循环内的同步调用，`decision_timeout` 默认 8 秒意味着最多卡住主循环 8 秒——嵌入模式下建议压到 2 秒以内，或只在 epoch 边界触发决策。sidecar 模式不存在这个问题（monitor 在训练进程外）。

---

## ✅ 快速校验

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| 脚本独立可运行 | `python train.py --epochs 1`（不经 guardian） | 不报错，训练 1 epoch 完成 |
| **脚本无 guardian 依赖** | `grep -r "guardian" train.py` | 无任何匹配——sidecar 路径下训练脚本不 import guardian |
| MNIST 自动下载 | 删除 data/ 目录后运行 | 自动下载并解压 |
| checkpoint 生成 | epochs=2, save_every=2 | checkpoints/cp_2/ 目录存在，含契约要求的必需字段 |
| 契约四项齐备 | `run.py contract check`（cp_11） | resumable / checkpoint_schema / metrics_channel / buildable_entry 全部通过 |
| 被守护运行 | `guardian watch -- python train.py --epochs 1` | 训练正常完成，guardian 侧记录到指标与退出码 0 |
| 终端输出 | 运行脚本 | 训练进度打印含 epoch/loss/acc，可被日志通道解析 |

---

## ✅ 完整校验

| 检查项 | 验证方式 | 通过标准 |
|--------|----------|----------|
| 训练收敛 | 20 epochs | val_accuracy > 0.98 |
| full_val.json | epoch=10 (every 5 cp) | 含 per_class 和 confusion_matrix |
| best.pth 维护 | 训练 20 epochs | 指向 accuracy 最高的 cp |
| top_k 清理 | save_top_k=5, 保存 10 个 | 磁盘仅保留 5 个 cp |
| 断点续训 | `--resume --ckpt checkpoints/cp_10` | 从 epoch 11 接续，指标不重置 |
| 资源预检 | 对比实际显存 | 误差 < 15% (GPU) |
| CPU 模式 | `device: cpu` | 正常训练，跳过显存预估 |
| 分析报告 | 训练结束 | analysis_report.json 生成且内容完整 |
| 重启后 wandb 接续 | guardian 主动干预重启一次 | 新进程沿用同一 run_id，指标曲线连续不断段 |
| 嵌入模式仍可用 | 按附录改造后运行 | per-step 检测生效，`skip_batch` 出现在 action_space 中 |

> **测试用例局限（v0 已知）**：SimpleCNN 约 60 万参数、CPU 数分钟跑完，**不会 OOM、不会把 GPU 烤到告警阈值、不会跑够时长触发多轮自动恢复**。而 OOM 恢复、温度告警、重启式干预恰恰是本项目的核心价值。cp_2/cp_3 里这类校验项需要**故障注入**（人为抛 OOM、mock nvidia-smi 返回高温、`kill -9` 训练子进程）或换一个真正吃显存的模型来验证，不能靠跑通 MNIST 就认为通过。
