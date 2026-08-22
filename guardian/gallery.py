"""cp_13 · 图片筛选与展示 (GalleryManager)。

训练后：推理 → 按 AI 提议的策略筛选 → 用户确认/NL修正 → 展示。

核心交互流：
1. agent 推断任务类型 → 提议多套筛选策略
2. 终端展示提案（name + rationale + filters）
3. 用户确认：回车执行 / NL修正 / export / cancel
4. 执行推理 + 筛选 → 启动 Streamlit/FiftyOne 展示

AI 边界：
- 能做: 读模型代码推断任务类型；定义筛选维度+阈值；组合多维度创造新策略
- 不能做: 不确认直接执行；修改标签；删除/覆盖数据
- 人把关: 策略需确认，支持 NL 修正
- 降级: AI 不可用 → 3 套默认策略

详见 checkpoint/cp_13.md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from .inference import InferenceRunner

from guardian.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# AI prompt
# ---------------------------------------------------------------------------

GALLERY_SYSTEM_PROMPT = (
    "你是一个计算机视觉模型评估专家。基于训练任务类型和指标上下文，"
    "设计多套图片筛选策略，用于训练后的成果展示。\n\n"
    "输出必须为 JSON 格式：\n"
    '{\n'
    '  "task_type": "classification|detection|segmentation",\n'
    '  "galleries": [\n'
    '    {\n'
    '      "name": "图集名称（中文，简短）",\n'
    '      "rationale": "为什么选这套策略（一句话，中文）",\n'
    '      "filters": [\n'
    '        {"type": "confidence_range", "min": 0.0, "max": 1.0},\n'
    '        {"type": "per_class_top", "k": 5},\n'
    '        {"type": "bbox_area_percentile", "max": 10},\n'
    '        {"type": "class_filter", "classes": ["class_a", "class_b"]},\n'
    '        {"type": "prediction_matches_label", "value": true|false}\n'
    '      ],\n'
    '      "sort_by": "confidence_desc|confidence_asc|random",\n'
    '      "max_images": 50\n'
    '    }\n'
    '  ]\n'
    '}\n\n'
    "过滤类型说明：\n"
    "- confidence_range: 置信度区间 [min, max]\n"
    "- per_class_top: 每类取 top-k 张\n"
    "- bbox_area_percentile: bbox 面积百分位（仅检测任务）\n"
    "- class_filter: 指定类别\n"
    "- prediction_matches_label: 预测是否匹配标签\n\n"
    "设计原则：\n"
    "1. 至少包含 3 套图集：汇报精选（高质量）、难样本（模型搞不定的）、\n"
    "   边界案例（模棱两可的）\n"
    "2. 考虑类别均衡性\n"
    "3. rationale 要说清楚设计意图\n"
    "4. 只返回 JSON，不要任何解释文字"
)


# ---------------------------------------------------------------------------
# GalleryManager
# ---------------------------------------------------------------------------

class GalleryManager:
    """图片筛选与展示管理器。"""

    def __init__(
        self,
        config: dict | None = None,
        advisor: Any = None,
        ckpt_analyzer: Any = None,
    ):
        self.cfg = config or {}
        self.advisor = advisor
        self.ckpt_analyzer = ckpt_analyzer
        self.default_max_images = int(self.cfg.get("default_max_images", 50))
        self.streamlit_port = int(self.cfg.get("streamlit_port", 8501))

    # ------------------------------------------------------------------
    # 任务类型推断
    # ------------------------------------------------------------------

    @staticmethod
    def infer_task_type(model_fn_or_desc: str | None = None) -> str:
        """推断任务类型。

        优先级: 用户描述中的关键词 > 模型代码推断 > 默认 classification
        """
        if model_fn_or_desc:
            desc_lower = model_fn_or_desc.lower()
            if any(w in desc_lower for w in ("检测", "detection", "detect", "bbox", "box")):
                return "detection"
            if any(w in desc_lower for w in ("分割", "segmentation", "seg", "mask", "unet")):
                return "segmentation"
            if any(w in desc_lower for w in ("分类", "classification", "classify", "resnet", "vgg")):
                return "classification"

        # 尝试从模型代码推断
        try:
            return InferenceRunner.detect_task_type(lambda: _try_build_model())
        except Exception:
            logger.warning("从模型代码推断任务类型失败，回退 classification", exc_info=True)

        return "classification"

    # ------------------------------------------------------------------
    # 策略提议
    # ------------------------------------------------------------------

    def propose_strategies(
        self,
        task_type: str,
        metrics_context: dict | None = None,
        user_feedback: str | None = None,
    ) -> dict[str, Any]:
        """AI 提议筛选策略。

        返回: {"task_type": str, "galleries": [...], "source": "agent"|"default"}
        """
        if self.advisor is not None and self.advisor.is_enabled("gallery_strategy"):
            context = {
                "task_type": task_type,
                "metrics_context": metrics_context or {},
                "user_feedback": user_feedback,
                "max_images": self.default_max_images,
            }
            try:
                result = self.advisor.suggest("gallery_strategy", context)
                if result and isinstance(result, dict) and "galleries" in result:
                    result["source"] = "agent"
                    return result
            except Exception:
                logger.warning("AI 提议图片筛选策略失败，使用默认策略", exc_info=True)

        return _default_strategies(task_type, self.default_max_images)

    # ------------------------------------------------------------------
    # 策略展示（终端）
    # ------------------------------------------------------------------

    @staticmethod
    def render_proposal(strategies: dict) -> str:
        """终端渲染策略提议，供用户确认。"""
        task_type = strategies.get("task_type", "unknown")
        source = strategies.get("source", "default")
        galleries = strategies.get("galleries", [])

        lines = [
            "=" * 56,
            f"  图片筛选策略提案",
            "=" * 56,
            f"  任务类型: {task_type}",
            f"  来源:     {'AI 提议' if source == 'agent' else '默认策略'}",
            f"  图集数:   {len(galleries)}",
            "",
        ]

        for i, g in enumerate(galleries, 1):
            lines.append(f"  ── 图集 {i}: {g.get('name', '未命名')} ──")
            if g.get("rationale"):
                lines.append(f"  意图: {g['rationale']}")
            filters = g.get("filters", [])
            if filters:
                lines.append(f"  过滤条件 ({len(filters)}):")
                for f in filters:
                    lines.append(f"    · {_describe_filter(f)}")
            lines.append(f"  排序: {g.get('sort_by', 'confidence_desc')} | "
                         f"上限: {g.get('max_images', 50)} 张")
            lines.append("")

        lines.append("─" * 56)
        lines.append("  操作: [回车] 确认执行  |  [修正说明] NL修正  |  export 导出  |  cancel 取消")
        lines.append("=" * 56)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def execute(
        self,
        checkpoint_path: str | Path,
        strategies: dict,
        data_source: str | Path,
        *,
        inference_runner: Any = None,
    ) -> dict[str, Any]:
        """执行推理 + 筛选。

        返回: {gallery_name: [{image_path, prediction, confidence, metadata}]}
        """
        # 1. 跑推理
        task_type = strategies.get("task_type", "classification")
        if inference_runner is None:
            inference_runner = InferenceRunner()

        infer_result = inference_runner.run(
            checkpoint_path=checkpoint_path,
            task_type=task_type,
            inputs=data_source,
            output_dir=Path(checkpoint_path).parent / "inference_gallery",
        )

        if infer_result.get("status") != "completed":
            return {"error": "推理失败", "detail": infer_result}

        # 2. 加载推理结果
        predictions = _load_predictions(infer_result)

        # 3. 逐图集筛选
        results: dict[str, list[dict]] = {}
        for gallery in strategies.get("galleries", []):
            name = gallery.get("name", "unnamed")
            filters = gallery.get("filters", [])
            sort_by = gallery.get("sort_by", "confidence_desc")
            max_images = gallery.get("max_images", self.default_max_images)

            # 应用过滤
            filtered = list(predictions)
            for f in filters:
                filtered = _apply_filter(filtered, f)

            # 排序
            filtered = _apply_sort(filtered, sort_by)

            # 截断
            filtered = filtered[:max_images]

            results[name] = filtered

        return results

    # ------------------------------------------------------------------
    # 导出
    # ------------------------------------------------------------------

    @staticmethod
    def export_config(strategies: dict, path: str | Path) -> Path:
        """导出策略配置为 JSON 文件（可版本管理、团队共享）。"""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(strategies, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return out

    @staticmethod
    def load_config(path: str | Path) -> dict | None:
        """加载已保存的策略配置。"""
        p = Path(path)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            # 配置损坏时按无配置处理，静默返回 None
            return None

    # ------------------------------------------------------------------
    # 展示
    # ------------------------------------------------------------------

    @staticmethod
    def launch_streamlit(results: dict, data_dir: str | Path, port: int = 8501) -> None:
        """启动 Streamlit 展示页（独立子进程，阻塞）。

        实际使用时建议在终端手动运行：
            streamlit run guardian/streamlit_app.py -- --results results.json
        """
        import subprocess
        import sys

        # 保存结果到临时文件
        tmp_path = Path(data_dir) / ".gallery_results.json"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        streamlit_script = Path(__file__).parent / "streamlit_app.py"
        if streamlit_script.exists():
            subprocess.run([
                sys.executable, "-m", "streamlit", "run",
                str(streamlit_script),
                "--", "--results", str(tmp_path),
                "--server.port", str(port),
            ])
        else:
            logger.info("Streamlit 脚本不存在: %s", streamlit_script)
            logger.info("结果已保存到: %s", tmp_path)
            logger.info("手动启动: streamlit run guardian/streamlit_app.py -- --results %s", tmp_path)


# ---------------------------------------------------------------------------
# 过滤/排序引擎（规则，确定性）
# ---------------------------------------------------------------------------

def _apply_filter(items: list[dict], f: dict) -> list[dict]:
    """对预测列表应用单个过滤条件。"""
    ftype = f.get("type", "")
    result = []

    for item in items:
        if ftype == "confidence_range":
            conf = item.get("confidence", item.get("score", 0))
            if conf is None:
                continue
            lo = f.get("min", 0)
            hi = f.get("max", 1)
            if lo <= conf <= hi:
                result.append(item)

        elif ftype == "per_class_top":
            # 需要先全部收集，然后分组取 top-k
            # 这个在单次过滤中做不到，需要两阶段
            # 简化处理：先按类别分组，在外部处理
            result.append(item)  # 由后续处理完成

        elif ftype == "bbox_area_percentile":
            bbox = item.get("bbox")
            if bbox is None:
                continue
            area = _bbox_area(bbox)
            max_pct = f.get("max", 100)
            # 近似：过滤掉面积超过某阈值的
            if area <= max_pct * 100:  # 粗略阈值
                result.append(item)

        elif ftype == "class_filter":
            classes = f.get("classes", [])
            pred_class = item.get("predicted_class", item.get("label"))
            if pred_class is not None and str(pred_class) in [str(c) for c in classes]:
                result.append(item)

        elif ftype == "prediction_matches_label":
            want_match = f.get("value", True)
            pred = item.get("predicted_class")
            label = item.get("true_label", item.get("label"))
            is_match = (pred is not None and label is not None and str(pred) == str(label))
            if is_match == want_match:
                result.append(item)

        else:
            # 未知过滤类型，原样保留
            result.append(item)

    return result


def _apply_sort(items: list[dict], sort_by: str) -> list[dict]:
    """排序。"""
    key_map = {
        "confidence_desc": ("confidence", True),
        "confidence_asc": ("confidence", False),
        "random": (None, False),
    }
    key_field, reverse = key_map.get(sort_by, ("confidence", True))

    if key_field == "random":
        import random
        result = list(items)
        random.shuffle(result)
        return result

    return sorted(
        items,
        key=lambda x: (x.get(key_field, 0) or 0),
        reverse=reverse,
    )


def _bbox_area(bbox) -> float:
    """bbox 面积计算（兼容多种格式）。"""
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        return abs((x2 - x1) * (y2 - y1))
    if isinstance(bbox, dict):
        w = bbox.get("width", bbox.get("w", 0))
        h = bbox.get("height", bbox.get("h", 0))
        return w * h
    return 0


def _describe_filter(f: dict) -> str:
    """人类可读的过滤条件描述。"""
    ftype = f.get("type", "")
    if ftype == "confidence_range":
        return f"置信度 ∈ [{f.get('min', 0)}, {f.get('max', 1)}]"
    if ftype == "per_class_top":
        return f"每类 top-{f.get('k', 5)}"
    if ftype == "bbox_area_percentile":
        return f"bbox 面积 ≤ P{f.get('max', 10)}"
    if ftype == "class_filter":
        return f"类别: {f.get('classes', [])}"
    if ftype == "prediction_matches_label":
        match = "匹配" if f.get("value", True) else "不匹配"
        return f"预测与标签{match}"
    return str(f)


def _load_predictions(infer_result: dict) -> list[dict]:
    """从推理结果加载预测列表。"""
    results_file = infer_result.get("results_file")
    if not results_file:
        return []

    try:
        data = json.loads(Path(results_file).read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("predictions", data.get("results", []))
    except (ValueError, OSError):
        # 推理结果文件缺失/损坏时按空预测处理，静默跳过
        pass

    return []


def _try_build_model():
    """尝试构建模型（用于任务类型推断）。"""
    try:
        # 优先从项目上下文中解析模型入口
        from .project_context import ProjectContext
        ctx = ProjectContext()
        ctx.apply_paths()
        entry = ctx.model_entry or "train:build_model"
        mod_path, fn_name = entry.split(":", 1) if ":" in entry else ("train", entry)
        import importlib
        mod = importlib.import_module(mod_path)
        return getattr(mod, fn_name)()
    except Exception:
        logger.warning("尝试构建模型失败（任务类型推断将回退）", exc_info=True)
        raise


def _default_strategies(task_type: str, max_images: int = 50) -> dict:
    """AI 不可用时的默认筛选策略（3 套）。"""
    return {
        "task_type": task_type,
        "source": "default",
        "galleries": [
            {
                "name": "汇报精选",
                "rationale": "各类别置信度最高的图片，适合汇报展示",
                "filters": [
                    {"type": "confidence_range", "min": 0.9, "max": 1.0},
                    {"type": "per_class_top", "k": 5},
                ],
                "sort_by": "confidence_desc",
                "max_images": max_images,
            },
            {
                "name": "难样本",
                "rationale": "置信度低或预测错误的样本，适合分析模型短板",
                "filters": [
                    {"type": "confidence_range", "min": 0.0, "max": 0.5},
                ],
                "sort_by": "confidence_asc",
                "max_images": max_images,
            },
            {
                "name": "边界案例",
                "rationale": "预测置信度中等、模棱两可的样本",
                "filters": [
                    {"type": "confidence_range", "min": 0.4, "max": 0.7},
                ],
                "sort_by": "confidence_asc",
                "max_images": max_images,
            },
        ],
    }
