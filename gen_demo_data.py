"""生成 1000 epoch 演示数据，覆盖全部故障场景。只运行一次。"""

import json
import math
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "demo_logs" / "mnist-guardian"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(42)

# ===== 1. 1000 epoch 指标 =====
metrics = []
current_lr = 0.001
current_batch = 64

# 故障事件记录
anomalies = []
restarts = []
checkpoint_metrics = []

# 模拟 epoch 0-999
for ep in range(1000):
    # ---- 基础 loss 曲线：指数衰减 + 噪声 ----
    base_loss = 2.3 * math.exp(-ep / 80) + 0.05 * (1 - math.exp(-ep / 400))
    noise = random.gauss(0, 0.02 * math.exp(-ep / 200) + 0.005)
    loss = max(0.001, base_loss + noise)

    # val_acc 曲线：sigmoid 上升
    val_acc = 1.0 / (1.0 + math.exp(-(ep - 60) / 50)) * 0.92 + 0.08
    val_acc += random.gauss(0, 0.01)
    val_acc = min(0.99, max(0.05, val_acc))

    # ---- 故障注入 ----

    # epoch 48: loss_spike — LR 太高导致震荡
    if ep == 48:
        loss = 1.85  # 窗口均值 ~0.65，突增 ~185%
        anomalies.append({
            "epoch": ep, "type": "loss_spike",
            "detail": f"Loss 突增 +185%，当前 {loss:.4f}，窗口均值 0.6500",
            "severity": "warning",
            "response": {"action": "alert_only", "source": "rule_default", "restart": False},
        })

    # epoch 130-140: GPU idle — 数据加载瓶颈
    gpu_util = 88 + random.gauss(0, 5)
    if 130 <= ep <= 140:
        gpu_util = 12 + random.gauss(0, 3)
        if ep == 135:
            anomalies.append({
                "epoch": ep, "type": "gpu_idle",
                "detail": f"GPU 0 利用率持续偏低（14%，连续 6 次），可能存在数据加载瓶颈",
                "severity": "warning",
                "response": {"action": "alert_only", "source": "rule_default", "restart": False},
            })

    # epoch 198: OOM crash → batch 减半
    if ep == 198:
        restarts.append({
            "epoch": ep, "trigger": "oom",
            "cmd_before": f"--batch_size {current_batch}",
            "cmd_after": f"--batch_size {current_batch // 2}",
            "applied": {"dataloader.batch_size": {"from": current_batch, "to": current_batch // 2}},
            "resumed_from": f"cp_{ep - 1}",
            "wasted_epochs": 1,
            "success": True,
        })
        current_batch //= 2
        anomalies.append({
            "epoch": ep, "type": "oom_recovery",
            "detail": f"OOM 崩溃，batch_size 从 {current_batch * 2} 降至 {current_batch}，从 cp_{ep - 1} 恢复",
            "severity": "error",
            "response": {"action": "restart_with_lower_batch", "source": "rule_default", "restart": True},
        })

    # epoch 245: GPU 温度过高
    gpu_temp = 62 + random.gauss(0, 8)
    if ep == 245:
        gpu_temp = 89
        anomalies.append({
            "epoch": ep, "type": "gpu_temp",
            "detail": f"GPU 0 温度过高（89°C > 85°C）",
            "severity": "warning",
            "response": {"action": "alert_only", "source": "rule_default", "restart": False},
        })

    # epoch 298: Agent 决定降 LR 重启
    if ep == 298:
        current_lr *= 0.5
        restarts.append({
            "epoch": ep, "trigger": "agent_decision",
            "cmd_before": f"--lr {current_lr * 2}",
            "cmd_after": f"--lr {current_lr}",
            "applied": {"optimizer.lr": {"from": current_lr * 2, "to": current_lr}},
            "resumed_from": f"cp_{ep - 1}",
            "wasted_epochs": 1,
            "success": True,
        })
        anomalies.append({
            "epoch": ep, "type": "agent_restart_lr",
            "detail": f"Agent 检测到 loss 下降趋缓，决策：restart_with_lower_lr (ratio=0.5)，lr {current_lr * 2:.4f} → {current_lr:.4f}",
            "severity": "info",
            "response": {"action": "restart_with_lower_lr", "source": "agent", "restart": True, "param": {"ratio": 0.5}},
        })

    # epoch 350: loss_stagnation — 500 步几乎不动
    if ep == 350:
        loss = loss  # 保持 base
        anomalies.append({
            "epoch": ep, "type": "loss_stagnation",
            "detail": f"Loss 停滞 500 步，下降仅 0.000001",
            "severity": "warning",
            "response": {"action": "suggest_lr_increase", "source": "agent", "restart": False},
        })

    # epoch 445: kill -9 crash → 从 ckpt 续训
    if ep == 445:
        restarts.append({
            "epoch": ep, "trigger": "crash",
            "cmd_before": f"--batch_size {current_batch} --lr {current_lr}",
            "cmd_after": f"--batch_size {current_batch} --lr {current_lr} --resume cp_{ep - 1}",
            "applied": {},
            "resumed_from": f"cp_{ep - 1}",
            "wasted_epochs": 1,
            "success": True,
        })
        anomalies.append({
            "epoch": ep, "type": "crash_recovery",
            "detail": f"训练进程被 kill -9，自动从 cp_{ep - 1} 续训，参数不变",
            "severity": "error",
            "response": {"action": "resume", "source": "rule_default", "restart": True},
        })

    # epoch 520: 又一次 loss_spike
    if ep == 520:
        loss = 0.89
        anomalies.append({
            "epoch": ep, "type": "loss_spike",
            "detail": f"Loss 突增 +320%，当前 {loss:.4f}，窗口均值 0.2100",
            "severity": "warning",
            "response": {"action": "alert_only", "source": "rule_default", "restart": False},
        })

    # epoch 600: Agent 再一次降 LR
    if ep == 598:
        current_lr *= 0.2
        restarts.append({
            "epoch": ep, "trigger": "agent_decision",
            "cmd_before": f"--lr {current_lr * 5}",
            "cmd_after": f"--lr {current_lr}",
            "applied": {"optimizer.lr": {"from": current_lr * 5, "to": current_lr}},
            "resumed_from": f"cp_{ep - 1}",
            "wasted_epochs": 1,
            "success": True,
        })
        anomalies.append({
            "epoch": ep, "type": "agent_restart_lr",
            "detail": f"Agent 二次决策：restart_with_lower_lr (ratio=0.2)，lr {current_lr * 5:.6f} → {current_lr:.6f}",
            "severity": "info",
            "response": {"action": "restart_with_lower_lr", "source": "agent", "restart": True, "param": {"ratio": 0.2}},
        })

    # epoch 680: GPU idle 再现
    if 680 <= ep <= 688:
        gpu_util = 10 + random.gauss(0, 3)
        if ep == 684:
            anomalies.append({
                "epoch": ep, "type": "gpu_idle",
                "detail": f"GPU 0 利用率持续偏低（13%，连续 5 次），建议检查 DataLoader num_workers",
                "severity": "warning",
                "response": {"action": "ignore", "source": "agent", "restart": False},
            })

    # epoch 748: NaN loss → 回滚到 cp_747
    if ep == 748:
        loss = float("nan")
        val_acc = float("nan")
        restarts.append({
            "epoch": ep, "trigger": "nan_inf",
            "cmd_before": f"--batch_size {current_batch} --lr {current_lr}",
            "cmd_after": f"--batch_size {current_batch} --lr {current_lr * 0.5} --resume cp_{ep - 1}",
            "applied": {"optimizer.lr": {"from": current_lr, "to": current_lr * 0.5}},
            "resumed_from": f"cp_{ep - 1}",
            "wasted_epochs": 1,
            "success": True,
        })
        current_lr *= 0.5
        anomalies.append({
            "epoch": ep, "type": "nan_inf",
            "detail": f"loss 为 nan，回滚到 cp_{ep - 1} 并降 LR 到 {current_lr:.6f}",
            "severity": "error",
            "response": {"action": "rollback_to_last_ckpt", "source": "rule_default", "restart": True},
        })

    # epoch 795: code error → 0 retries
    if ep == 795:
        anomalies.append({
            "epoch": ep, "type": "code_error",
            "detail": "TypeError: unsupported operand type(s) for +: 'int' and 'str'，判定为不可恢复错误，停止重试",
            "severity": "error",
            "response": {"action": "stop", "source": "rule_default", "restart": False},
        })

    # 确保 NaN 不进入数据流
    if math.isnan(loss):
        loss = None
        val_acc = None

    gpu_temp = max(45, min(95, gpu_temp))
    gpu_util = max(0, min(100, gpu_util))

    rec = {
        "raw": f"epoch {ep} loss {loss:.4f} val_acc {val_acc:.4f} lr {current_lr}" if loss is not None else f"epoch {ep} loss nan val_acc nan lr {current_lr}",
        "step": ep,
        "loss": round(loss, 4) if loss is not None else None,
        "val_acc": round(val_acc, 4) if val_acc is not None else None,
        "lr": current_lr,
    }
    metrics.append(rec)

    # checkpoint 每 50 epoch 存入
    if ep % 50 == 0 and loss is not None:
        checkpoint_metrics.append({
            "epoch": ep,
            "path": f"checkpoints/cp_{ep}",
            "metrics": {
                "epoch": ep,
                "train/loss": round(loss, 4),
                "val/accuracy": round(val_acc, 4),
                "lr": current_lr,
                "batch_size": current_batch,
            },
        })

# ===== 2. 写入 metrics.jsonl =====
with (OUT / "metrics.jsonl").open("w", encoding="utf-8") as f:
    for m in metrics:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")
print(f"metrics.jsonl: {len(metrics)} 条")

# ===== 3. 写入 meta.json =====
meta = {
    "process_id": "mnist-guardian",
    "name": "MNIST Guardian Demo · 1000 Epochs",
    "command": "guarftrain watch --agent --with-dashboard --with-mcp -- python train.py --epochs 1000",
    "project_dir": "",
    "status": "completed",
    "registered_at": 1750000000.0,
    "finished_at": 1750086400.0,
    "config": {"agent_enabled": True, "mcp_enabled": True, "dashboard_enabled": True},
    "model_entry": "train:build_model",
    "log_file": "logs/train.log",
}
with (OUT / "meta.json").open("w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("meta.json")

# ===== 4. 写入 summary =====
summary = {
    "experiment_id": "mnist-guardian",
    "status": "completed",
    "duration_seconds": 86400,
    "duration": "24h 0m 0s",
    "exit_code": 0,
    "training": {
        "total_epochs": 1000,
        "final_loss": metrics[-1]["loss"],
        "final_val_acc": metrics[-1]["val_acc"],
        "best_val_acc": max((m["val_acc"] for m in metrics if m["val_acc"] is not None), default=0),
        "records": len(metrics),
    },
    "anomaly_events": anomalies,
    "restarts": restarts,
    "checkpoints": {
        "total": len(checkpoint_metrics),
        "latest": f"cp_{checkpoint_metrics[-1]['epoch']}",
        "best": f"cp_{max(checkpoint_metrics, key=lambda x: x['metrics']['val/accuracy'])['epoch']}",
        "metric": "accuracy",
        "metric_source": {"name": "accuracy", "direction": "max", "source": "contract", "task_type": "classification"},
        "checkpoints": checkpoint_metrics,
    },
    "resources": {
        "gpu_util_avg": 78.5,
        "gpu_hours": 24.0,
        "gpu_mem_peak_mb": 8192,
        "gpu_mem_peak_gb": 8.0,
        "gpu_temp_avg": 68.5,
        "gpu_temp_max": 89.0,
        "gpu_count": 1,
        "gpu_name": "NVIDIA RTX 4090",
    },
    "lr_schedule": [
        {"epoch": 0, "lr": 0.001},
        {"epoch": 300, "lr": 0.0005},
        {"epoch": 600, "lr": 0.0001},
        {"epoch": 750, "lr": 0.00005},
    ],
    "ai_narrative": (
        '本次训练任务历时 24 小时完成 1000 epoch，经历了丰富的故障场景。\n\n'
        '训练初期（epoch 0-50），模型快速学习，loss 从 2.3 急剧下降至 0.65。epoch 48 时出现一次 loss_spike（突增 185%），'
        '规则判定为告警级别，未触发重启。epoch 130-140 期间 GPU 利用率下降至 14%，系统检测到 gpu_idle 并发出数据加载瓶颈告警。\n\n'
        '训练中期（epoch 198），发生 OOM 崩溃。Guardian 自动将 batch_size 从 64 降至 32，从 cp_197 恢复，损失 1 个 epoch 算力。'
        'epoch 245 GPU 温度飙升至 89C，触发温度告警（硬件安全不交给 Agent 决策）。\n\n'
        'epoch 298，Agent 分析 loss 曲线后主动决策 restart_with_lower_lr（ratio=0.5），lr 从 0.001 降至 0.0005。'
        'epoch 350 检测到 loss_stagnation（500 步仅下降 0.000001），Agent 建议 suggest_lr_increase 但未自动执行。\n\n'
        'epoch 445，训练进程意外被 kill -9。Watchdog 检测到崩溃后自动从 cp_444 续训，参数保持不变（kill 不是 OOM）。'
        'epoch 520 再次出现 loss_spike（+320%），系统告警。\n\n'
        'epoch 598，Agent 二次决策重启降 LR（ratio=0.2），lr 降至 0.0001。epoch 680 GPU 再次 idle，Agent 判定为 ignore。\n\n'
        'epoch 748 遭遇最严重故障 - loss 变为 NaN。系统立即回滚到 cp_747 并将 LR 减半至 0.00005。\n\n'
        'epoch 795，发生 TypeError 代码错误。Guardian 判定为不可恢复（code_error），停止重试，符合 S3 验收标准。'
        '虽然训练提前终止，但已产出 16 个 checkpoint，最佳 val_acc 在 epoch 750 达到。\n\n'
        '总结：本次训练充分展示了 Guardian 的故障处理能力：OOM 恢复（S1）、崩溃续训（S2）、代码错误不反复重启（S3）、'
        'batch 下限保护（S5）、Agent 智能决策、NaN 回滚、GPU 监控告警等全部核心场景。'
    ),
}
with (OUT.parent / "summary_demo.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"summary_demo.json: {len(anomalies)} 条异常, {len(restarts)} 次重启, {len(checkpoint_metrics)} 个 checkpoint")

# ===== 5. 写入 anomalies.jsonl (独立文件供前端展示) =====
with (OUT / "anomalies.jsonl").open("w", encoding="utf-8") as f:
    for a in anomalies:
        f.write(json.dumps(a, ensure_ascii=False) + "\n")
print(f"anomalies.jsonl: {len(anomalies)} 条")

print("\n✅ 演示数据生成完毕！")
