"""推理执行器 (InferenceRunner)。

共享模块：被 F3（gallery）和 F7（infer）复用。
不生成代码——只选择固定脚本 + subprocess 执行。

AI 边界：
- 能做: 推荐 checkpoint、推荐测试输入（纯文本输出）
- 不能做: 生成代码、自动执行推理
- 降级: AI 不可用 → best ckpt + 随机采样

详见 checkpoint/cp_16.md
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


# 固定推理脚本映射
_TASK_SCRIPTS = {
    "classification": "infer_classification.py",
    "detection": "infer_detection.py",
    "segmentation": "infer_segmentation.py",
}


class InferenceRunner:
    """固定脚本推理执行器。"""

    def __init__(self, config: dict | None = None):
        self.cfg = config or {}
        self.scripts_dir = Path(self.cfg.get("scripts_dir", "./scripts"))
        self.default_batch_size = int(self.cfg.get("default_batch_size", 32))
        self.device = self.cfg.get("device", "cuda")

    # ------------------------------------------------------------------
    # 任务检测
    # ------------------------------------------------------------------

    @staticmethod
    def detect_task_type(model_fn) -> str:
        """规则推断任务类型（基于模型 forward 返回值）。

        不依赖 AI——读取模型代码或用户指定。
        回退到 "classification"。
        """
        try:
            model = model_fn()
            # 尝试用 dummy input 推断
            import torch
            model.eval()
            dummy = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                output = model(dummy)

            if isinstance(output, dict):
                keys = set(str(k).lower() for k in output.keys())
                if "boxes" in keys or "bbox" in keys:
                    return "detection"
                if "masks" in keys or "seg" in keys or "out" in keys:
                    return "segmentation"
                if "logits" in keys:
                    return "classification"

            if isinstance(output, (list, tuple)):
                if len(output) >= 2:
                    return "detection"  # 常见: (boxes, scores) 或类似
                return "classification"

            if isinstance(output, torch.Tensor):
                if output.dim() == 2:
                    return "classification"  # [B, num_classes]
                if output.dim() >= 3:
                    # [B, C, H, W] → 可能是分割
                    if output.shape[1] > 10:
                        return "segmentation"
                    return "classification"

        except Exception:
            pass

        return "classification"  # 默认

    # ------------------------------------------------------------------
    # 推荐
    # ------------------------------------------------------------------

    @staticmethod
    def recommend_checkpoint(
        ckpt_analyzer: Any,
        advisor: Any = None,
    ) -> int | None:
        """推荐用于推理的 checkpoint epoch。

        - 规则默认: best checkpoint
        - AI 可用时: 综合 latest + best + 稳定性推荐
        """
        if ckpt_analyzer is None:
            return None

        report = ckpt_analyzer.report() if hasattr(ckpt_analyzer, "report") else {}
        best = report.get("best") or {}
        best_epoch = best.get("epoch")

        # AI 推荐
        if advisor is not None and advisor.is_enabled("summary_narrative"):
            try:
                context = {
                    "checkpoints": [
                        {"epoch": c.get("epoch"), "metrics": c.get("metrics", {})}
                        for c in report.get("checkpoints", [])[:10]
                    ],
                    "best_epoch": best_epoch,
                    "total": report.get("total", 0),
                }
                suggestion = advisor.suggest("inference_checkpoint", context)
                if suggestion and isinstance(suggestion, dict):
                    recommended = suggestion.get("recommended_epoch")
                    if isinstance(recommended, int) and recommended > 0:
                        return recommended
            except Exception:
                pass

        return best_epoch

    @staticmethod
    def recommend_inputs(
        data_dir: str | Path,
        task_type: str,
        advisor: Any = None,
        n_samples: int = 50,
    ) -> list[str]:
        """推荐测试输入文件列表。

        - 规则默认: 随机采样
        - AI 可用时: 根据任务类型推荐（如检测任务建议含小目标场景）
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            return []

        # 收集所有图片文件
        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        all_images = sorted(
            [str(p) for p in data_path.rglob("*") if p.suffix.lower() in image_exts]
        )

        if not all_images:
            return []

        # 简单随机采样
        import random
        if len(all_images) <= n_samples:
            return all_images
        return random.sample(all_images, n_samples)

    # ------------------------------------------------------------------
    # 执行
    # ------------------------------------------------------------------

    def run(
        self,
        checkpoint_path: str | Path,
        task_type: str,
        inputs: str | Path,
        output_dir: str | Path,
        *,
        batch_size: int | None = None,
        model_fn_ref: str | None = None,
    ) -> dict[str, Any]:
        """使用固定推理脚本执行推理（独立子进程）。

        返回: {
            "status": "completed" | "failed",
            "task_type": str,
            "checkpoint": str,
            "output_dir": str,
            "num_inputs": int,
            "results_file": str | None,
            "stdout": str,
            "stderr": str,
            "exit_code": int,
        }
        """
        script_name = _TASK_SCRIPTS.get(task_type)
        if script_name is None:
            return {
                "status": "failed",
                "error": f"不支持的任务类型: {task_type!r}。可选: {list(_TASK_SCRIPTS)}",
            }

        script_path = self.scripts_dir / script_name
        if not script_path.exists():
            return {
                "status": "failed",
                "error": f"推理脚本不存在: {script_path}",
            }

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        bs = batch_size or self.default_batch_size

        cmd = [
            sys.executable, str(script_path),
            "--checkpoint", str(checkpoint_path),
            "--inputs", str(inputs),
            "--output", str(out),
            "--batch-size", str(bs),
            "--device", self.device,
        ]
        if model_fn_ref:
            cmd.extend(["--model", model_fn_ref])

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 小时上限
                cwd=str(Path(script_path).parent.parent),  # 项目根目录
            )
        except subprocess.TimeoutError:
            return {
                "status": "failed",
                "error": "推理超时（>1h）",
                "task_type": task_type,
                "checkpoint": str(checkpoint_path),
                "output_dir": str(out),
            }

        # 查找结果文件
        results_file = None
        for pattern in ["results.json", "predictions.json", "inference_results.json"]:
            candidate = out / pattern
            if candidate.exists():
                results_file = str(candidate)
                break

        result = {
            "status": "completed" if proc.returncode == 0 else "failed",
            "task_type": task_type,
            "checkpoint": str(checkpoint_path),
            "output_dir": str(out),
            "num_inputs": _count_inputs(inputs),
            "results_file": results_file,
            "stdout": proc.stdout[-5000:] if proc.stdout else "",
            "stderr": proc.stderr[-2000:] if proc.stderr else "",
            "exit_code": proc.returncode,
        }

        # 尝试加载结果摘要
        if results_file:
            try:
                data = json.loads(Path(results_file).read_text(encoding="utf-8"))
                if isinstance(data, list):
                    result["num_results"] = len(data)
                elif isinstance(data, dict):
                    result["num_results"] = len(data.get("predictions", data.get("results", [])))
            except (ValueError, OSError):
                pass

        return result


def _count_inputs(inputs: str | Path) -> int:
    """统计输入目录下的图片数量。"""
    p = Path(inputs)
    if p.is_file():
        return 1
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        return sum(1 for f in p.rglob("*") if f.suffix.lower() in exts)
    return 0
