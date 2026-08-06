"""cp_13 · GalleryManager 单元测试。

覆盖 checkpoint/cp_13.md 的快速校验表。
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from guardian.gallery import (
    GalleryManager,
    _apply_filter,
    _apply_sort,
    _default_strategies,
    _describe_filter,
)


# ---------------------------------------------------------------------------
# 快速校验
# ---------------------------------------------------------------------------

class TestGalleryManager:
    """cp_13 单元测试。"""

    def test_infer_task_type_from_desc(self):
        assert GalleryManager.infer_task_type("检测任务") == "detection"
        assert GalleryManager.infer_task_type("segmentation model") == "segmentation"
        assert GalleryManager.infer_task_type("图像分类") == "classification"
        assert GalleryManager.infer_task_type(None) == "classification"

    def test_default_strategies(self):
        result = _default_strategies("classification")
        assert result["source"] == "default"
        assert len(result["galleries"]) == 3
        names = [g["name"] for g in result["galleries"]]
        assert "汇报精选" in names
        assert "难样本" in names
        assert "边界案例" in names

    def test_propose_strategies_without_advisor(self):
        gm = GalleryManager()
        result = gm.propose_strategies("classification")
        assert result["source"] == "default"
        assert len(result["galleries"]) == 3

    def test_propose_strategies_with_advisor(self):
        mock_advisor = MagicMock()
        mock_advisor.is_enabled.return_value = True
        mock_advisor.suggest.return_value = {
            "task_type": "detection",
            "galleries": [
                {
                    "name": "自定义图集",
                    "rationale": "AI 创建",
                    "filters": [{"type": "confidence_range", "min": 0.8, "max": 1.0}],
                    "sort_by": "confidence_desc",
                    "max_images": 30,
                }
            ],
        }
        gm = GalleryManager(advisor=mock_advisor)
        result = gm.propose_strategies("detection")
        assert result["source"] == "agent"
        assert len(result["galleries"]) == 1
        assert result["galleries"][0]["name"] == "自定义图集"

    def test_propose_strategies_advisor_fails(self):
        mock_advisor = MagicMock()
        mock_advisor.is_enabled.return_value = True
        mock_advisor.suggest.side_effect = RuntimeError("API down")
        gm = GalleryManager(advisor=mock_advisor)
        result = gm.propose_strategies("classification")
        assert result["source"] == "default"

    def test_render_proposal(self):
        strategies = _default_strategies("detection")
        text = GalleryManager.render_proposal(strategies)
        assert "图片筛选策略提案" in text
        assert "汇报精选" in text
        assert "detection" in text

    def test_export_and_load_config(self, tmp_path):
        strategies = _default_strategies("classification")
        config_path = tmp_path / "gallery_config.json"
        exported = GalleryManager.export_config(strategies, config_path)
        assert exported == config_path
        assert config_path.exists()

        loaded = GalleryManager.load_config(config_path)
        assert loaded is not None
        assert loaded["source"] == "default"
        assert len(loaded["galleries"]) == 3

    def test_load_config_missing(self, tmp_path):
        assert GalleryManager.load_config(tmp_path / "nonexistent.json") is None

    def test_load_config_invalid(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        assert GalleryManager.load_config(p) is None

    def test_execute_no_inference(self):
        """验证 execute 在无推理结果时返回错误。"""
        gm = GalleryManager()
        mock_ir = MagicMock()
        mock_ir.run.return_value = {"status": "failed", "error": "no model"}
        result = gm.execute("fake.pt", _default_strategies("classification"), "./data", inference_runner=mock_ir)
        assert "error" in result


# ---------------------------------------------------------------------------
# 过滤/排序引擎
# ---------------------------------------------------------------------------

class TestFilterEngine:

    def test_confidence_range(self):
        items = [
            {"confidence": 0.95},
            {"confidence": 0.30},
            {"confidence": 0.50},
        ]
        result = _apply_filter(items, {"type": "confidence_range", "min": 0.5, "max": 1.0})
        assert len(result) == 2

    def test_class_filter(self):
        items = [
            {"predicted_class": "cat", "confidence": 0.9},
            {"predicted_class": "dog", "confidence": 0.8},
            {"predicted_class": "cat", "confidence": 0.7},
        ]
        result = _apply_filter(items, {"type": "class_filter", "classes": ["cat"]})
        assert len(result) == 2

    def test_prediction_matches_label(self):
        items = [
            {"predicted_class": "cat", "true_label": "cat", "confidence": 0.9},
            {"predicted_class": "dog", "true_label": "cat", "confidence": 0.8},
        ]
        result = _apply_filter(items, {"type": "prediction_matches_label", "value": True})
        assert len(result) == 1

    def test_prediction_mismatch(self):
        items = [
            {"predicted_class": "cat", "true_label": "cat", "confidence": 0.9},
            {"predicted_class": "dog", "true_label": "cat", "confidence": 0.8},
        ]
        result = _apply_filter(items, {"type": "prediction_matches_label", "value": False})
        assert len(result) == 1

    def test_unknown_filter_passthrough(self):
        items = [{"confidence": 0.9}]
        result = _apply_filter(items, {"type": "unknown_type"})
        assert len(result) == 1

    def test_sort_confidence_desc(self):
        items = [
            {"confidence": 0.3},
            {"confidence": 0.9},
            {"confidence": 0.5},
        ]
        result = _apply_sort(items, "confidence_desc")
        assert result[0]["confidence"] == 0.9
        assert result[-1]["confidence"] == 0.3

    def test_sort_confidence_asc(self):
        items = [{"confidence": 0.9}, {"confidence": 0.3}]
        result = _apply_sort(items, "confidence_asc")
        assert result[0]["confidence"] == 0.3

    def test_sort_random(self):
        items = [{"confidence": i / 10} for i in range(10)]
        result = _apply_sort(items, "random")
        assert len(result) == 10


# ---------------------------------------------------------------------------
# 描述
# ---------------------------------------------------------------------------

def test_describe_filter():
    assert "置信度" in _describe_filter({"type": "confidence_range", "min": 0.5, "max": 1.0})
    assert "top-5" in _describe_filter({"type": "per_class_top", "k": 5})
    assert "cat" in _describe_filter({"type": "class_filter", "classes": ["cat", "dog"]})
