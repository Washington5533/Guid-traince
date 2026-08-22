"""cp_5 · 日志摘要：训练结束生成一页结构化报告。

sidecar 形态下多一段 `restarts`：所有干预都经由重启生效，因此"被重启了
几次、每次作废多少算力"是复盘时最该看的信息——它直接回答"干预值不值"。
详见 checkpoint/cp_5.md

v0 不含 ai_narrative（那需要 cp_9 的 advisor）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from guardian.logging_config import get_logger

logger = get_logger(__name__)


class SummaryGenerator:
    def __init__(
        self,
        config: dict | None = None,
        monitor: Any = None,
        ckpt_analyzer: Any = None,
        watchdog: Any = None,
        advisor: Any = None,
    ):
        self.cfg = config or {}
        self.monitor = monitor
        self.ckpt_analyzer = ckpt_analyzer
        self.watchdog = watchdog
        self.advisor = advisor          # v1；v0 恒为 None
        self.start_time = time.time()

    def generate(self, run_result: dict | None = None) -> dict[str, Any]:
        result = run_result or {}
        end = time.time()
        summary: dict[str, Any] = {
            "experiment_id": self.cfg.get("name", "guardian-run"),
            "status": result.get("status", "unknown"),
            "duration_seconds": round(end - self.start_time, 1),
            "duration": _fmt_duration(end - self.start_time),
            "exit_code": result.get("exit_code"),
        }
        if result.get("crash"):
            summary["failure"] = {"kind": result["crash"], "detail": result.get("detail", "")}

        summary["training"] = self._collect_training()
        summary["anomaly_events"] = self._collect_anomalies()
        summary["restarts"] = self._collect_restarts()
        summary["checkpoints"] = self._collect_checkpoints()
        summary["resources"] = self._collect_resource_usage()
        summary["lr_schedule"] = self._collect_lr_schedule()
        summary["agent_decisions"] = self._collect_agent_decisions()

        narrative = self._generate_ai_narrative(summary)
        if narrative:
            summary["ai_narrative"] = narrative
        return summary

    def _collect_training(self) -> dict[str, Any]:
        if self.monitor is None:
            return {}
        hist = self.monitor.get_metrics_history()
        if not hist:
            return {"records": 0, "note": "无指标记录（指标通道未配置或训练未产出）"}
        steps = [r["step"] for r in hist if r.get("step") is not None]
        losses = [r["loss"] for r in hist if isinstance(r.get("loss"), (int, float))]
        accs = [r["val_acc"] for r in hist if isinstance(r.get("val_acc"), (int, float))]
        out: dict[str, Any] = {"records": len(hist)}
        if steps:
            out["last_step"] = max(steps)
        if losses:
            out["final_loss"] = losses[-1]
            out["min_loss"] = min(losses)
        if accs:
            out["final_val_acc"] = accs[-1]
            out["best_val_acc"] = max(accs)
        return out

    def _collect_anomalies(self) -> list[dict[str, Any]]:
        if self.monitor is None:
            return []
        return self.monitor.get_anomaly_history()

    def _collect_restarts(self) -> list[dict[str, Any]]:
        if self.watchdog is None:
            return []
        return [r.to_dict() for r in self.watchdog.restarts]

    def _collect_checkpoints(self) -> dict[str, Any]:
        if self.ckpt_analyzer is None:
            return {}
        return self.ckpt_analyzer.report()

    def _collect_resource_usage(self) -> dict[str, Any]:
        """GPU 平均利用率、显存峰值、GPU 时数。"""
        gpu_hist = getattr(self.monitor, "get_gpu_history", None)
        if gpu_hist is None:
            return {}
        records = gpu_hist()
        if not records:
            return {}
        valid = [r for r in records if "error" not in r]
        if not valid:
            return {"note": "GPU 采样全部失败"}

        # GPU 名称取第一次成功采样的
        gpu_name = "GPU"
        utils = []
        mem_used = []
        temps = []
        for r in valid:
            utils.append(r.get("util_pct"))
            mem_used.append(r.get("mem_used_mb"))
            temps.append(r.get("temperature_c"))

        out: dict[str, Any] = {}
        clean_utils = [u for u in utils if u is not None]
        clean_mem = [m for m in mem_used if m is not None]
        clean_temps = [t for t in temps if t is not None]

        if clean_utils:
            out["gpu_util_avg"] = round(sum(clean_utils) / len(clean_utils), 1)
            # GPU 时数：平均利用率 * 训练时长 / 100（利用率是百分比）
            if self.start_time:
                hours = (time.time() - self.start_time) / 3600
                out["gpu_hours"] = round(sum(clean_utils) / len(clean_utils) / 100 * hours, 2)

        if clean_mem:
            out["gpu_mem_peak_mb"] = round(max(clean_mem), 1)
            # 显存峰值 GB
            out["gpu_mem_peak_gb"] = round(max(clean_mem) / 1024, 2)

        if clean_temps:
            out["gpu_temp_avg"] = round(sum(clean_temps) / len(clean_temps), 1)
            out["gpu_temp_max"] = round(max(clean_temps), 1)

        # 多卡统计
        indices = {r.get("index") for r in valid if r.get("index") is not None}
        out["gpu_count"] = len(indices) if indices else 1

        if gpu_name:
            out["gpu_name"] = gpu_name

        return out

    def _collect_lr_schedule(self) -> list[dict[str, Any]]:
        """学习率调度历史：从指标通道提取 lr 变化点，跨重启拼接。"""
        if self.monitor is None:
            return []
        hist = self.monitor.get_metrics_history()
        schedule: list[dict[str, Any]] = []
        last_lr = None
        for r in hist:
            lr = r.get("lr")
            if lr is not None and lr != last_lr:
                schedule.append({
                    "step": r.get("step"),
                    "lr": lr,
                })
                last_lr = lr
        return schedule

    def _collect_agent_decisions(self) -> list[dict[str, Any]]:
        """收集 agent 决策日志（内存 + 持久化文件）。"""
        if self.advisor is None:
            return []
        # 优先内存
        mem = self.advisor.decision_log
        if mem:
            return mem
        # 回退：从持久化文件读取
        log_path = getattr(self.advisor, "_log_path", None)
        if log_path:
            return self.advisor.load_log(log_path)
        return []

    def _generate_ai_narrative(self, summary: dict) -> str | None:
        """v1：advisor 基于结构化摘要生成自然语言解读。失败/未配置则省略。"""
        if self.advisor is None:
            return None
        try:
            return self.advisor.narrate(summary)
        except Exception:
            logger.warning("生成 AI 自然语言解读失败，摘要中省略该字段", exc_info=True)
            return None      # 纯输出，失败只影响可读性

    # --- 输出 -------------------------------------------------------

    def render(self, summary: dict) -> str:
        status_icon = {"completed": "完成", "failed": "失败", "stopped": "已停止"}.get(
            summary.get("status", ""), summary.get("status", "?")
        )
        lines = [
            "=" * 56,
            f"  训练摘要 · {summary.get('experiment_id')}",
            "=" * 56,
            f"  状态: {status_icon}    用时: {summary.get('duration')}",
        ]
        tr = summary.get("training") or {}
        if tr.get("last_step") is not None:
            lines.append(f"  进度: 最后 step/epoch = {tr['last_step']}")
        if tr.get("final_loss") is not None:
            lines.append(f"  Loss: 最终 {tr['final_loss']:.4f} | 最低 {tr['min_loss']:.4f}")
        if tr.get("best_val_acc") is not None:
            lines.append(f"  验证: 最终 {tr.get('final_val_acc'):.4f} | 最佳 {tr['best_val_acc']:.4f}")

        events = summary.get("anomaly_events") or []
        if events:
            lines.append(f"  异常事件: {len(events)} 条")
            for ev in events[:5]:
                lines.append(f"    - [{ev.get('severity')}] {ev.get('type')}: {ev.get('detail')}")
            if len(events) > 5:
                lines.append(f"    ... 另有 {len(events) - 5} 条")
        else:
            lines.append("  异常事件: 无")

        restarts = summary.get("restarts") or []
        if restarts:
            wasted = sum(r.get("wasted_epochs") or 0 for r in restarts)
            by_trigger: dict[str, int] = {}
            for r in restarts:
                by_trigger[r.get("trigger", "?")] = by_trigger.get(r.get("trigger", "?"), 0) + 1
            detail = " + ".join(f"{v} {k}" for k, v in sorted(by_trigger.items()))
            lines.append(f"  重启: {len(restarts)} 次（{detail}），作废约 {wasted} epoch")
            for r in restarts[:5]:
                applied = r.get("applied") or {}
                changes = ", ".join(
                    f"{k.split('.')[-1]} {v.get('from')}->{v.get('to')}" for k, v in applied.items()
                ) or "参数不变"
                lines.append(f"    - {r.get('trigger')}: {changes}"
                             + (f"（{r['skipped']}）" if r.get("skipped") else ""))
        else:
            lines.append("  重启: 无")

        ck = summary.get("checkpoints") or {}
        if ck.get("total"):
            lines.append(f"  Checkpoint: 共 {ck['total']} 个，最新 {ck.get('latest')}")

        # GPU 资源统计
        res = summary.get("resources") or {}
        if res and res.get("gpu_util_avg") is not None:
            parts = [f"GPU 均值 {res['gpu_util_avg']}%"]
            if res.get("gpu_mem_peak_gb"):
                parts.append(f"显存峰值 {res['gpu_mem_peak_gb']}GB")
            if res.get("gpu_hours"):
                parts.append(f"{res['gpu_hours']} GPU·h")
            lines.append(f"  资源: {', '.join(parts)}")

        # 学习率调度
        lr_hist = summary.get("lr_schedule") or []
        if len(lr_hist) > 1:
            changes = ", ".join(
                f"step {p['step']}→{p['lr']:g}" for p in lr_hist[:5]
            )
            if len(lr_hist) > 5:
                changes += f" ... 共 {len(lr_hist)} 次变化"
            lines.append(f"  LR 调度: {changes}")

        if summary.get("failure"):
            f = summary["failure"]
            lines.append(f"  失败原因: {f.get('kind')} — {f.get('detail')}")
        if summary.get("ai_narrative"):
            lines.append(f"  AI 解读: {summary['ai_narrative']}")
        lines.append("=" * 56)
        return "\n".join(lines)

    def print_summary(self, summary: dict) -> None:
        logger.info("%s", self.render(summary))

    def save_summary(self, summary: dict, output_dir: str | Path) -> tuple[Path, Path]:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        jpath = out / f"summary_{stamp}.json"
        tpath = out / f"summary_{stamp}.txt"
        jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        tpath.write_text(self.render(summary), encoding="utf-8")
        return jpath, tpath


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}min"
    if m:
        return f"{m}min {s}s"
    return f"{s}s"
