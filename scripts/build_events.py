#!/usr/bin/env python3
"""从已有的演示训练数据组装 guardian 事件流（SSE 回放用）。

输入:
  demo_logs/mnist-guardian/metrics.jsonl   1000 epoch 指标
  demo_logs/mnist-guardian/anomalies.jsonl 12 条异常
  demo_logs/mnist-guardian/meta.json       会话元信息
  demo_logs/summary_demo.json              资源/重启/总结
  logs/viz/model_viz_mnist-guardian.html   架构分析树 (treeData)

输出:
  logs/remote/mnist-demo/events.jsonl      按时间排序的事件流
  logs/remote/mnist-demo/arch.json         /api/arch/analyze 响应
"""
import datetime as _dt
import json
import math
import os
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo_logs" / "mnist-guardian"
SUMMARY = ROOT / "demo_logs" / "summary_demo.json"
VIZ = ROOT / "logs" / "viz" / "model_viz_mnist-guardian.html"
OUT = ROOT / "logs" / "remote" / "mnist-demo"

random.seed(7)

BASE_TS = 1750000000.0  # 与 demo meta.registered_at 一致


def iso(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat()


# ---- 1. 读入源数据 -------------------------------------------------------
metrics = []
with (DEMO / "metrics.jsonl").open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            metrics.append(json.loads(line))
print(f"metrics: {len(metrics)}")

anomalies = []
with (DEMO / "anomalies.jsonl").open(encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            anomalies.append(json.loads(line))
print(f"anomalies: {len(anomalies)}")

meta = json.loads((DEMO / "meta.json").read_text(encoding="utf-8"))
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
resources = summary.get("resources", {})

# ---- 2. 架构分析树（从 model_viz HTML 提取）-------------------------------
arch_result = None
if VIZ.exists():
    html = VIZ.read_text(encoding="utf-8")
    def _extract(name: str):
        m = re.search(rf"const {name} = (.*?);\n", html, re.S)
        return json.loads(m.group(1)) if m else None
    tree = _extract("treeData")
    bottlenecks = _extract("bottlenecks") or []
    improvements = _extract("improvements") or []
    if tree:
        total_params = sum(c.get("params", 0) for c in tree.get("children", []))
        arch_result = {
            "ok": True,
            "model_name": "VisionTransformer (CLIP)",
            "total_params": total_params,
            "total_flops": 114816,
            "total_flops_m": round(114816 / 1e6, 1),
            "module_count": len(tree.get("children", [])),
            "layer_count": len(bottlenecks),
            "bottleneck_count": len(bottlenecks),
            "bottlenecks": bottlenecks,
            "layer_stats": [],
            "improvements": improvements,
            "tree": tree,
            "elapsed_ms": 42.0,
            "narration": (
                "骨干为 VisionTransformer（CLIP 风格），encoder 占 99.98% 参数量，"
                "其中 transformer.resblocks 12 层重复块占 96.8%。瓶颈集中在 encoder.transformer，"
                "建议优先优化注意力层；head 仅 0.0%，无优化空间。"
            ),
        }
        print(f"arch tree extracted: {total_params} params, {len(bottlenecks)} bottlenecks")

# ---- 3. 组装事件流 --------------------------------------------------------
events = []
events.append({
    "timestamp": BASE_TS,
    "type": "training_start",
    "data": {"command": meta.get("command", ""), "session_id": "mnist-demo"},
})

# 决策事件定义（来自 summary.restarts 中的 agent 决策 + 异常）
DECISIONS = {
    298: {
        "id": "d-298", "tool": "restart_with_lower_lr",
        "action": {"lr": "0.0010 → 0.0005", "ratio": 0.5, "resume_from": "cp_297"},
        "agent_id": "agent-1", "phase": "auto",
        "detail": "Agent 检测 loss 下降趋缓，决策降低学习率并重启",
    },
    350: {
        "id": "d-350", "tool": "suggest_lr_increase",
        "action": {"suggestion": "loss 停滞 500 步，建议提升 LR", "auto": False},
        "agent_id": "agent-1", "phase": "supervised",
        "detail": "loss_stagnation：Agent 给出建议，等待人工批准",
    },
    598: {
        "id": "d-598", "tool": "restart_with_lower_lr",
        "action": {"lr": "0.0005 → 0.0001", "ratio": 0.2, "resume_from": "cp_597"},
        "agent_id": "agent-1", "phase": "auto",
        "detail": "Agent 二次决策：lr 再降 80% 并重启",
    },
    684: {
        "id": "d-684", "tool": "ignore",
        "action": {"decision": "GPU idle 判定可忽略（数据加载瓶颈，非训练问题）"},
        "agent_id": "agent-1", "phase": "auto",
        "detail": "gpu_idle：Agent 判定忽略",
    },
}

# 异常按 epoch 索引
anom_by_epoch = {a["epoch"]: a for a in anomalies}

gpu_name = resources.get("gpu_name", "NVIDIA RTX 4090")
gpu_mem_total = 24576  # MB
gpu_base_mem = int(resources.get("gpu_mem_peak_mb", 8192))
gpu_util_avg = resources.get("gpu_util_avg", 78.5)
gpu_temp_avg = resources.get("gpu_temp_avg", 68.5)

first_ts = None
last_ts = None
for ep, m in enumerate(metrics):
    ts = BASE_TS + 60 + ep * 12.0  # 每 epoch 12s
    if first_ts is None:
        first_ts = ts
    last_ts = ts

    # GPU 状态：每 50 epoch 一条
    if ep % 50 == 0:
        util = gpu_util_avg + random.gauss(0, 8)
        temp = gpu_temp_avg + random.gauss(0, 6)
        if 130 <= ep <= 140:
            util = 12 + random.gauss(0, 3)
        if 680 <= ep <= 688:
            util = 10 + random.gauss(0, 3)
        if ep == 245:
            temp = 89.0
        util = max(0.0, min(100.0, util))
        temp = max(45.0, min(95.0, temp))
        mem_used = max(1024, min(gpu_mem_total - 2048, gpu_base_mem + random.gauss(0, 200)))
        events.append({
            "timestamp": ts,
            "type": "gpu_status",
            "data": {
                "devices": [{
                    "name": gpu_name,
                    "index": 0,
                    "utilization": round(util, 1),
                    "temperature": round(temp, 1),
                    "memory_used": int(mem_used),
                    "memory_total": gpu_mem_total,
                    "power": round(util / 100 * 350 + random.gauss(0, 10), 1),
                }],
            },
        })

    events.append({"timestamp": ts, "type": "metrics", "data": m})

    if ep in anom_by_epoch:
        a = dict(anom_by_epoch[ep])
        a["timestamp"] = iso(ts)
        a["session_id"] = "mnist-demo"
        events.append({"timestamp": ts, "type": "anomaly", "data": a})

    if ep in DECISIONS:
        d = dict(DECISIONS[ep])
        d["timestamp"] = iso(ts)
        d["session_id"] = "mnist-demo"
        events.append({"timestamp": ts, "type": "decision", "data": d})

# 架构分析（训练结束后推送一次 narration）
if arch_result:
    events.append({
        "timestamp": (last_ts or BASE_TS) + 5,
        "type": "arch_analysis",
        "data": {
            "narration": arch_result["narration"],
            "model_name": arch_result["model_name"],
            "total_params": arch_result["total_params"],
            "bottleneck_count": arch_result["bottleneck_count"],
        },
    })

events.append({
    "timestamp": (last_ts or BASE_TS) + 10,
    "type": "training_end",
    "data": {"status": "completed", "exit_code": 0},
})

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "events.jsonl").open("w", encoding="utf-8") as f:
    for e in events:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
print(f"events.jsonl: {len(events)} 条 ({OUT})")

if arch_result:
    with (OUT / "arch.json").open("w", encoding="utf-8") as f:
        json.dump(arch_result, f, ensure_ascii=False, indent=2)
    print("arch.json written")

# 统计
from collections import Counter
print(Counter(e["type"] for e in events))
