"""cp_14 · 跨实验查询 (ExperimentQuery)。

训练后只读推理：扫描历史 summary JSON → NL 查询翻译 → 结构化执行 →
格式化回答。不修改任何训练数据，纯文本输出。

AI 边界：
- 能做: NL→结构化查询翻译、跨实验差异分析、基于历史的参数推荐
- 不能做: 执行查询（规则引擎执行）、修改历史记录、建议自动生效
- 降级: AI 不可用 → 固定模板查询（最近 N 次/按指标排序/按状态筛选）

详见 checkpoint/cp_14.md
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from guardian.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 查询 DSL（AI 翻译目标格式，规则引擎执行）
# ---------------------------------------------------------------------------

QUERY_SYSTEM_PROMPT = (
    "你是一个训练实验数据分析助手。将用户的自然语言问题翻译为结构化查询。\n\n"
    "输出必须为 JSON 格式：\n"
    '{"filters": [{"field": "字段名", "op": "eq|gt|lt|gte|lte|contains", "value": ...}], '
    '"sort_by": "排序字段", "sort_order": "desc|asc", "limit": N, '
    '"fields": ["需要的字段1", "字段2"], '
    '"interpretation": "你对用户问题的理解（一句话中文）"}\n\n'
    "可用字段: experiment_id, status, duration_seconds, exit_code, "
    "final_loss, min_loss, final_val_acc, best_val_acc, "
    "gpu_util_avg, gpu_mem_peak_gb, gpu_hours, "
    "anomaly_count, restart_count, "
    "best_checkpoint_epoch, best_metric_name, best_metric_value, "
    "lr_values (list), batch_size, timestamp\n\n"
    "规则：\n"
    '1. "上次"/"最近一次" → sort_by="timestamp", sort_order="desc", limit=1\n'
    '2. "最高"/"最好" → sort_by=对应指标字段, sort_order="desc", limit=1\n'
    '3. "最低"/"最差" → sort_by=对应指标字段, sort_order="asc", limit=1\n'
    '4. "所有"/"列出" → limit=20 或不设 limit\n'
    '5. 如果用户的问题无法翻译，返回 {"error": "无法理解，请换个问法"}\n'
    "只返回 JSON，不要任何解释文字。"
)


# ---------------------------------------------------------------------------
# ExperimentQuery
# ---------------------------------------------------------------------------

class ExperimentQuery:
    """跨实验 NL 查询引擎。扫描 logs/summary_*.json 构建内存索引。"""

    def __init__(
        self,
        config: dict | None = None,
        advisor: Any = None,
    ):
        self.cfg = config or {}
        self.log_dir = Path(self.cfg.get("log_dir", "./logs"))
        self.max_compare = int(self.cfg.get("max_compare_experiments", 5))
        self.advisor = advisor  # v1；v0 恒为 None
        self._index: list[dict[str, Any]] = []
        self._last_scan = 0.0
        self._scan_ttl = 5.0  # 5 秒内不重复扫描
        # 允许用户手动设置实验名称前缀
        self._custom_name: str | None = self.cfg.get("name")

    def set_name(self, name: str) -> None:
        """手动设置实验名称（覆盖 summary 中的 experiment_id）。"""
        self._custom_name = name
        self._index = []  # 强制重新扫描

    # ------------------------------------------------------------------
    # 索引管理
    # ------------------------------------------------------------------

    def _scan(self, force: bool = False) -> list[dict[str, Any]]:
        """扫描 log_dir 下的 summary_*.json，构建轻量实验索引。"""
        now = time.monotonic()
        if not force and self._index and (now - self._last_scan) < self._scan_ttl:
            return self._index

        if not self.log_dir.exists():
            self._index = []
            return self._index

        index: list[dict[str, Any]] = []
        for fpath in sorted(self.log_dir.glob("summary_*.json"), reverse=True):
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                # 损坏/不可读的 summary 文件属正常情况，静默跳过
                continue

            training = data.get("training") or {}
            ckpt = data.get("checkpoints") or {}
            resources = data.get("resources") or {}
            anomalies = data.get("anomaly_events") or []
            restarts = data.get("restarts") or []
            lr_schedule = data.get("lr_schedule") or []

            # 提取 lr 值列表
            lr_values = [p.get("lr") for p in lr_schedule if p.get("lr") is not None]

            # 提取 batch_size（从 checkpoint 的 metric_source 附近推断）
            batch_size = None
            for c in ckpt.get("checkpoints", []):
                bs = c.get("metrics", {}).get("batch_size")
                if bs is not None:
                    batch_size = bs
                    break

            # 名称优先级: 手动设置 > summary 自带 > 文件名推断
            raw_id = self._custom_name or data.get("experiment_id", fpath.stem)
            entry = {
                "experiment_id": raw_id,
                "status": data.get("status", "unknown"),
                "duration_seconds": data.get("duration_seconds"),
                "duration": data.get("duration"),
                "exit_code": data.get("exit_code"),
                "final_loss": training.get("final_loss"),
                "min_loss": training.get("min_loss"),
                "final_val_acc": training.get("final_val_acc"),
                "best_val_acc": training.get("best_val_acc"),
                "gpu_util_avg": resources.get("gpu_util_avg"),
                "gpu_mem_peak_gb": resources.get("gpu_mem_peak_gb"),
                "gpu_hours": resources.get("gpu_hours"),
                "anomaly_count": len(anomalies),
                "restart_count": len(restarts),
                "best_checkpoint_epoch": ckpt.get("best", {}).get("epoch") if ckpt.get("best") else None,
                "best_metric_name": ckpt.get("metric"),
                "best_metric_value": (
                    ckpt.get("best", {}).get("metrics", {}).get(ckpt.get("metric", ""))
                    if ckpt.get("best") and ckpt.get("metric") else None
                ),
                "lr_values": lr_values,
                "batch_size": batch_size,
                "timestamp": _extract_timestamp(data, fpath),
                "_source_file": str(fpath),
            }

            # 把其他所有 training 里的键也暴露出来（如 mAP, mIoU 等自定义指标）
            for k, v in training.items():
                if k not in entry and isinstance(v, (int, float, str)):
                    entry[k] = v

            index.append(entry)

        # ---- 去重：同名实验用时间戳 + AI 命名区分 ----
        seen: dict[str, int] = {}
        for e in index:
            eid = e["experiment_id"]
            seen[eid] = seen.get(eid, 0) + 1

        duplicates = [k for k, v in seen.items() if v > 1]

        if duplicates and self.advisor is not None and self.advisor.is_enabled("experiment_query"):
            # 尝试 AI 批量命名
            try:
                dup_data = [
                    {"experiment_id": e["experiment_id"], "timestamp": e.get("timestamp"),
                     "status": e.get("status"), "best_metric_name": e.get("best_metric_name"),
                     "best_metric_value": e.get("best_metric_value"),
                     "duration": e.get("duration")}
                    for e in index if e["experiment_id"] in duplicates
                ]
                names = self.advisor.suggest("experiment_names", {"duplicates": dup_data})
                if names and isinstance(names, dict):
                    name_map = names.get("names", names)
                    for e in index:
                        old = e["experiment_id"]
                        if old in name_map and isinstance(name_map[old], str):
                            e["experiment_id"] = name_map[old]
                            e["_renamed_by"] = "agent"
            except Exception:
                logger.warning("实验重名 AI 批量命名失败，回退为时间戳后缀区分", exc_info=True)

        # 仍未区分的重复项：追加时间戳后缀
        for e in index:
            if e["experiment_id"] in duplicates and not e.get("_renamed_by"):
                ts = e.get("timestamp", "")
                # 取简短时间戳：2026-08-06 14:47 → 0806_1447
                short_ts = _short_ts(ts)
                e["experiment_id"] = f"{e['experiment_id']}_{short_ts}"
                e["_renamed_by"] = "timestamp"

        self._index = index
        self._last_scan = now
        return index

    # ------------------------------------------------------------------
    # 公开 API
    # ------------------------------------------------------------------

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        """列出所有实验摘要（精简字段，适合终端/MCP 展示）。"""
        index = self._scan()
        out = []
        for e in index[:limit]:
            out.append({
                "experiment_id": e["experiment_id"],
                "status": e["status"],
                "duration": e.get("duration"),
                "best_metric_name": e.get("best_metric_name"),
                "best_metric_value": e.get("best_metric_value"),
                "final_loss": e.get("final_loss"),
                "restart_count": e.get("restart_count"),
                "timestamp": e.get("timestamp"),
            })
        return out

    def get_experiment(self, exp_id: str) -> dict[str, Any] | None:
        """按 experiment_id 获取完整实验数据。"""
        index = self._scan()
        for e in index:
            if e["experiment_id"] == exp_id:
                # 加载完整 JSON
                src = e.get("_source_file")
                if src and Path(src).exists():
                    try:
                        return json.loads(Path(src).read_text(encoding="utf-8"))
                    except (ValueError, OSError):
                        # 完整 JSON 加载失败时回退返回索引条目，静默跳过
                        pass
                return e
        return None

    def query(self, question: str) -> dict[str, Any]:
        """NL 查询入口。

        返回: {
            "question": str,
            "interpretation": str | None,   # AI 对问题的理解
            "source": "agent" | "template", # 查询翻译来源
            "results": [...],
            "answer": str,                  # 自然语言回答
        }
        """
        index = self._scan()
        if not index:
            return {
                "question": question,
                "interpretation": None,
                "source": "template",
                "results": [],
                "answer": "暂无实验记录。",
            }

        # 尝试 AI 翻译
        structured = None
        interpretation = None
        source = "template"

        if self.advisor is not None and self.advisor.is_enabled("experiment_query"):
            try:
                raw = self.advisor.suggest(
                    "experiment_query",
                    {"question": question, "available_fields": list(index[0].keys()) if index else []},
                )
                if raw and isinstance(raw, dict) and "error" not in raw:
                    structured = raw
                    interpretation = raw.get("interpretation")
                    source = "agent"
            except Exception:
                logger.warning("AI 查询翻译失败，回退为模板查询", exc_info=True)

        # 回退：模板匹配
        if structured is None:
            structured = _template_query(question)
            interpretation = structured.get("interpretation")
            source = "template"

        # 执行查询
        results = _execute_query(index, structured)
        answer = _format_answer(question, results, interpretation, len(index))

        return {
            "question": question,
            "interpretation": interpretation,
            "source": source,
            "results": results,
            "answer": answer,
        }

    def compare(self, id_a: str, id_b: str) -> dict[str, Any]:
        """对比两个实验。

        返回: {
            "experiment_a": {...},
            "experiment_b": {...},
            "diffs": [{"field": ..., "a": ..., "b": ..., "delta": ...}],
            "analysis": str | None,   # AI 分析（可选）
        }
        """
        exp_a = self.get_experiment(id_a)
        exp_b = self.get_experiment(id_b)

        if not exp_a:
            return {"error": f"实验 {id_a!r} 不存在"}
        if not exp_b:
            return {"error": f"实验 {id_b!r} 不存在"}

        # 数值字段对比
        _numeric_fields = [
            "duration_seconds", "final_loss", "min_loss",
            "final_val_acc", "best_val_acc",
            "gpu_util_avg", "gpu_mem_peak_gb", "gpu_hours",
            "anomaly_count", "restart_count",
        ]

        diffs = []
        for field in _numeric_fields:
            va = _get_nested(exp_a, field)
            vb = _get_nested(exp_b, field)
            if va is not None and vb is not None and isinstance(va, (int, float)) and isinstance(vb, (int, float)):
                diffs.append({
                    "field": field,
                    "a": va,
                    "b": vb,
                    "delta": round(vb - va, 6),
                })

        # 参数对比
        lr_a = _extract_lr(exp_a)
        lr_b = _extract_lr(exp_b)
        bs_a = _extract_bs(exp_a)
        bs_b = _extract_bs(exp_b)

        param_diffs = {}
        if lr_a != lr_b:
            param_diffs["learning_rate"] = {"a": lr_a, "b": lr_b}
        if bs_a != bs_b:
            param_diffs["batch_size"] = {"a": bs_a, "b": bs_b}

        # AI 分析
        analysis = None
        if self.advisor is not None and self.advisor.is_enabled("summary_narrative"):
            try:
                analysis = self.advisor.narrate({
                    "type": "experiment_comparison",
                    "experiment_a": {"id": id_a, "status": exp_a.get("status"), "training": exp_a.get("training")},
                    "experiment_b": {"id": id_b, "status": exp_b.get("status"), "training": exp_b.get("training")},
                    "diffs": diffs,
                    "param_diffs": param_diffs,
                })
            except Exception:
                logger.warning("AI 实验对比分析失败，analysis 字段将为空", exc_info=True)

        return {
            "experiment_a": {"id": id_a, "status": exp_a.get("status"), "duration": exp_a.get("duration")},
            "experiment_b": {"id": id_b, "status": exp_b.get("status"), "duration": exp_b.get("duration")},
            "diffs": diffs,
            "param_diffs": param_diffs,
            "analysis": analysis,
        }

    def suggest_params(self, task_type: str = "classification") -> dict[str, Any]:
        """基于历史最佳实验推荐参数。

        返回: {"recommended_lr": ..., "recommended_batch_size": ..., "based_on": ..., "reasoning": ...}
        """
        index = self._scan()
        completed = [e for e in index if e.get("status") == "completed"]

        if not completed:
            return {"error": "没有已完成的实验可供参考"}

        # 找 best_metric_value 最高的
        best = None
        for e in completed:
            if e.get("best_metric_value") is not None:
                if best is None or e["best_metric_value"] > best.get("best_metric_value", 0):
                    best = e

        if not best:
            return {"error": "没有可用指标数据"}

        rec = {
            "based_on": best["experiment_id"],
            "best_metric_name": best.get("best_metric_name"),
            "best_metric_value": best.get("best_metric_value"),
        }

        if best.get("lr_values"):
            rec["recommended_lr"] = best["lr_values"][0]
        if best.get("batch_size"):
            rec["recommended_batch_size"] = best["batch_size"]

        # AI 推理
        if self.advisor is not None and self.advisor.is_enabled("summary_narrative"):
            try:
                reasoning = self.advisor.narrate({
                    "type": "param_recommendation",
                    "task_type": task_type,
                    "best_experiment": best,
                    "all_experiments_count": len(completed),
                })
                if reasoning:
                    rec["reasoning"] = reasoning
            except Exception:
                logger.warning("AI 参数推荐推理失败，仅返回规则推荐结果", exc_info=True)

        return rec


# ---------------------------------------------------------------------------
# 查询执行（纯规则，不依赖 AI）
# ---------------------------------------------------------------------------

def _execute_query(index: list[dict], query: dict) -> list[dict]:
    """在内存索引上执行结构化查询。"""
    results = list(index)

    # 应用过滤
    for f in query.get("filters", []):
        field = f.get("field", "")
        op = f.get("op", "eq")
        value = f.get("value")
        results = [r for r in results if _match(r.get(field), op, value)]

    # 排序
    sort_by = query.get("sort_by", "timestamp")
    reverse = query.get("sort_order", "desc") == "desc"
    if sort_by:
        results.sort(
            key=lambda r: (_sort_key(r.get(sort_by))),
            reverse=reverse,
        )

    # 限制
    limit = query.get("limit")
    if limit and isinstance(limit, int) and limit > 0:
        results = results[:limit]

    # 字段筛选
    fields = query.get("fields")
    if fields and isinstance(fields, list):
        results = [{k: r.get(k) for k in fields if k in r} for r in results]
    else:
        # 默认精简字段
        _default_fields = [
            "experiment_id", "status", "duration", "best_metric_name",
            "best_metric_value", "final_loss", "timestamp",
        ]
        results = [{k: r.get(k) for k in _default_fields if k in r} for r in results]

    return results


def _match(val: Any, op: str, target: Any) -> bool:
    """单个过滤条件匹配。"""
    if val is None:
        return False
    if op == "eq":
        return val == target
    if op == "gt":
        return isinstance(val, (int, float)) and isinstance(target, (int, float)) and val > target
    if op == "lt":
        return isinstance(val, (int, float)) and isinstance(target, (int, float)) and val < target
    if op == "gte":
        return isinstance(val, (int, float)) and isinstance(target, (int, float)) and val >= target
    if op == "lte":
        return isinstance(val, (int, float)) and isinstance(target, (int, float)) and val <= target
    if op == "contains":
        return isinstance(val, str) and isinstance(target, str) and target.lower() in val.lower()
    return False


def _sort_key(val: Any):
    """排序键：None 排最后。"""
    if val is None:
        return (1, 0)
    if isinstance(val, bool):
        return (0, int(val))
    return (0, val)


def _template_query(question: str) -> dict:
    """规则模板：匹配常见 NL 模式，不依赖 AI。"""
    q = question.lower().strip()

    # "上次/last/最近"
    if any(w in q for w in ("上次", "最近", "last", "latest")):
        return {
            "sort_by": "timestamp",
            "sort_order": "desc",
            "limit": 1,
            "interpretation": "查找最近一次实验",
        }

    # "所有" / "列出" / "list"
    if any(w in q for w in ("所有", "列出", "全部", "list", "all")):
        return {
            "sort_by": "timestamp",
            "sort_order": "desc",
            "limit": 20,
            "interpretation": "列出最近实验",
        }

    # "最高" / "最好" / "best" / "highest"
    if any(w in q for w in ("最高", "最好", "最优", "best", "highest", "max")):
        # 尝试识别具体指标
        for metric, field in _METRIC_ALIASES:
            if metric in q:
                return {
                    "sort_by": field,
                    "sort_order": "desc",
                    "limit": 1,
                    "interpretation": f"查找{metric}最高的实验",
                }
        return {
            "sort_by": "best_metric_value",
            "sort_order": "desc",
            "limit": 1,
            "interpretation": "查找指标最优的实验",
        }

    # 默认：返回最近 5 次
    return {
        "sort_by": "timestamp",
        "sort_order": "desc",
        "limit": 5,
        "interpretation": "查找最近的实验（默认）",
    }


_METRIC_ALIASES = [
    ("准确", "best_val_acc"),
    ("accuracy", "best_val_acc"),
    ("mAP", "best_metric_value"),
    ("loss", "min_loss"),
    ("损失", "min_loss"),
]


def _format_answer(
    question: str,
    results: list[dict],
    interpretation: str | None,
    total: int,
) -> str:
    """把查询结果格式化为自然语言回答。"""
    if not results:
        return f"未找到匹配的实验记录。（共 {total} 次实验）"

    if len(results) == 1:
        r = results[0]
        parts = [f"实验 {r.get('experiment_id', '?')}"]
        if r.get("status"):
            parts.append(f"状态: {r['status']}")
        if r.get("best_metric_name") and r.get("best_metric_value") is not None:
            parts.append(f"{r['best_metric_name']}={r['best_metric_value']}")
        if r.get("final_loss") is not None:
            parts.append(f"final_loss={r['final_loss']:.4f}")
        if r.get("duration"):
            parts.append(f"用时: {r['duration']}")
        return " · ".join(parts)

    return f"找到 {len(results)} 条匹配记录（共 {total} 次实验）。"


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _extract_timestamp(data: dict, fpath: Path) -> str | None:
    """从 summary JSON 提取时间戳。"""
    ts = data.get("timestamp")
    if ts:
        return str(ts)
    # 从文件名推断: summary_20260805_233601.json
    stem = fpath.stem
    if stem.startswith("summary_"):
        date_part = stem[len("summary_"):]
        if len(date_part) >= 14:
            t = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]} " \
                f"{date_part[9:11]}:{date_part[11:13]}:{date_part[13:15]}"
            return t
    return None


def _get_nested(d: dict, key: str, default=None) -> Any:
    """从 dict 取键，兼容嵌套路径和 flat key。"""
    # 先查 training 子 dict
    training = d.get("training") or {}
    if key in training:
        return training[key]
    # 再查 resources
    resources = d.get("resources") or {}
    if key in resources:
        return resources[key]
    # fallback: 顶层或 index 条目
    val = d.get(key)
    if val is not None:
        return val
    # 从 index 风格的条目查
    anomalies = d.get("anomaly_events") or []
    restarts = d.get("restarts") or []
    if key == "anomaly_count":
        return len(anomalies)
    if key == "restart_count":
        return len(restarts)
    return default


def _short_ts(ts: str) -> str:
    """2026-08-06 14:47:01 → 0806_1447"""
    if not ts:
        return "unknown"
    clean = ts.replace("-", "").replace(":", "").replace(" ", "_")
    if len(clean) >= 13:
        return clean[4:8] + "_" + clean[9:13]
    return clean[:12] if len(clean) > 8 else clean


def _extract_lr(exp: dict) -> Any:
    """从实验数据中提取 lr。"""
    lr_schedule = exp.get("lr_schedule") or []
    if lr_schedule:
        return lr_schedule[0].get("lr")
    training = exp.get("training") or {}
    return training.get("lr")


def _extract_bs(exp: dict) -> Any:
    """从实验数据中提取 batch_size。"""
    training = exp.get("training") or {}
    bs = training.get("batch_size")
    if bs is not None:
        return bs
    ckpt = exp.get("checkpoints") or {}
    for c in ckpt.get("checkpoints", []):
        bs = c.get("metrics", {}).get("batch_size")
        if bs is not None:
            return bs
    return None
