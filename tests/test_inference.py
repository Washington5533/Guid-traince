"""cp_16 · InferenceRunner 单元测试。

覆盖 checkpoint/cp_16.md 的快速校验表。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from guardian.inference import InferenceRunner, _TASK_SCRIPTS


# ---------------------------------------------------------------------------
# 快速校验
# ---------------------------------------------------------------------------

class TestInferenceRunner:
    """cp_16 单元测试。"""

    def test_detect_task_type_classification(self):
        mock_model = MagicMock()
        import torch
        mock_model.return_value = torch.randn(1, 10)  # [B, num_classes]
        result = InferenceRunner.detect_task_type(mock_model)
        assert result == "classification"

    def test_detect_task_type_fallback(self):
        # 模型构建失败 → 回退 classification
        result = InferenceRunner.detect_task_type(lambda: (_ for _ in ()).throw(RuntimeError()))
        assert result == "classification"

    def test_recommend_checkpoint_no_analyzer(self):
        assert InferenceRunner.recommend_checkpoint(None) is None

    def test_recommend_checkpoint_with_analyzer(self):
        mock_analyzer = MagicMock()
        mock_analyzer.report.return_value = {
            "total": 5,
            "best": {"epoch": 20, "metrics": {"val/accuracy": 0.95}},
            "checkpoints": [{"epoch": i * 5, "metrics": {}} for i in range(5)],
        }
        epoch = InferenceRunner.recommend_checkpoint(mock_analyzer)
        assert epoch == 20

    def test_recommend_inputs_empty(self, tmp_path):
        result = InferenceRunner.recommend_inputs(tmp_path, "classification")
        assert result == []

    def test_recommend_inputs_with_images(self, tmp_path):
        # 创建假图片
        (tmp_path / "img1.jpg").write_text("fake")
        (tmp_path / "img2.png").write_text("fake")
        (tmp_path / "not_image.txt").write_text("fake")
        result = InferenceRunner.recommend_inputs(tmp_path, "classification")
        assert len(result) == 2

    def test_run_invalid_task_type(self, tmp_path):
        ir = InferenceRunner({"scripts_dir": str(tmp_path)})
        result = ir.run("fake.pt", "invalid_task", "./data", tmp_path / "out")
        assert result["status"] == "failed"
        assert "不支持" in result.get("error", "")

    def test_run_script_not_found(self, tmp_path):
        ir = InferenceRunner({"scripts_dir": str(tmp_path)})
        result = ir.run("fake.pt", "classification", "./data", tmp_path / "out")
        assert result["status"] == "failed"
        assert "不存在" in result.get("error", "")

    def test_run_script_exists(self, tmp_path):
        # 创建最小推理脚本
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        script = script_dir / "infer_classification.py"
        script.write_text(
            "import sys, json, argparse\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--checkpoint')\n"
            "p.add_argument('--inputs')\n"
            "p.add_argument('--output')\n"
            "p.add_argument('--batch-size', type=int)\n"
            "p.add_argument('--device')\n"
            "p.add_argument('--model', default=None)\n"
            "args = p.parse_args()\n"
            "from pathlib import Path\n"
            "Path(args.output).mkdir(parents=True, exist_ok=True)\n"
            "(Path(args.output) / 'results.json').write_text('[]')\n"
            "print('done')\n"
        )

        ir = InferenceRunner({"scripts_dir": str(script_dir)})
        out = tmp_path / "inference_out"
        result = ir.run("checkpoints/cp_0/model.pth", "classification", str(tmp_path), out)
        # 脚本成功运行即 passed（即使 checkpoint 不存在）
        assert result["exit_code"] == 0 or result["status"] == "completed"

    def test_task_script_mapping(self):
        assert _TASK_SCRIPTS["classification"] == "infer_classification.py"
        assert _TASK_SCRIPTS["detection"] == "infer_detection.py"
        assert _TASK_SCRIPTS["segmentation"] == "infer_segmentation.py"
