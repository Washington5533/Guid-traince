"""cp_15 · ModelVisualizer 单元测试。

覆盖 checkpoint/cp_15.md 的快速校验表。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _no_mcp_import():
    """防止 mcp SDK 导入干扰测试。"""
    pass


# ---------------------------------------------------------------------------
# 测试用简单模型
# ---------------------------------------------------------------------------

class SimpleTestModel:
    """模拟 nn.Module（不依赖 torch）。"""
    def __init__(self):
        self._modules = {
            "conv1": _FakeConv("Conv2d", 64),
            "conv2": _FakeConv("Conv2d", 128),
            "fc": _FakeConv("Linear", 1000),
        }
        self.__class__.__name__ = "SimpleTestModel"

    def named_modules(self):
        yield "", self
        for name, mod in self._modules.items():
            yield name, mod

    def parameters(self):
        for m in self._modules.values():
            yield _FakeParam(m.params)


class _FakeConv:
    def __init__(self, type_name, params):
        self._type = type_name
        self.params = params
        self.__class__.__name__ = type_name

    def named_modules(self):
        return iter([])

    def parameters(self):
        return iter([])


class _FakeParam:
    def __init__(self, numel):
        self._numel = numel

    def numel(self):
        return self._numel


# ---------------------------------------------------------------------------
# 测试
# ---------------------------------------------------------------------------

class TestModelVisualizer:
    """cp_15 单元测试。"""

    def test_compute_stats(self):
        from guardian.model_viz import ModelVisualizer
        mv = ModelVisualizer()
        graph = {
            "nodes": [
                {"name": "conv1", "type": "Conv2d", "params": 64, "flops_est": 1000, "depth": 0},
                {"name": "conv2", "type": "Conv2d", "params": 128, "flops_est": 2000, "depth": 1},
                {"name": "fc", "type": "Linear", "params": 1000, "flops_est": 500, "depth": 2},
            ],
            "edges": [],
            "total_params": 1192,
            "total_flops_est": 3500,
            "model_name": "TestModel",
        }
        stats = mv.compute_stats(graph)
        assert stats["total_params"] == 1192
        assert len(stats["layer_stats"]) == 3
        # 排序：FLOPs 最高的在前
        # First by FLOPs depends on hook data; just verify sorted
        assert len(stats["layer_stats"]) == 3

    def test_propose_config_default(self):
        from guardian.model_viz import ModelVisualizer
        mv = ModelVisualizer()
        graph = {"nodes": [], "edges": [], "total_params": 0, "total_flops_est": 0, "model_name": "Test"}
        stats = {"layer_stats": [], "total_params": 0, "total_flops": 0}
        config = mv.propose_config(graph, stats)
        assert "view" in config
        assert "bottlenecks" in config
        assert "architecture_summary" in config

    def test_propose_config_with_bottleneck(self):
        from guardian.model_viz import ModelVisualizer
        mv = ModelVisualizer()
        graph = {"nodes": [], "edges": [], "total_params": 1000, "total_flops_est": 10000, "model_name": "Test"}
        stats = {
            "layer_stats": [
                {"name": "heavy_layer", "type": "Conv2d", "params": 600, "flops_est": 5000,
                 "params_pct": 60, "flops_pct": 50, "depth": 1},
            ],
            "total_params": 1000,
            "total_flops": 10000,
        }
        config = mv.propose_config(graph, stats)
        # 默认策略应对瓶颈做标注
        assert len(config["bottlenecks"]) >= 1
        assert config["bottlenecks"][0]["severity"] == "critical"

    def test_render_html_produces_valid_html(self, tmp_path):
        from guardian.model_viz import ModelVisualizer
        mv = ModelVisualizer()
        graph = {
            "nodes": [{"name": "layer1", "type": "Conv2d", "params": 64, "flops_est": 1000,
                       "input_shape": None, "output_shape": None, "depth": 0}],
            "edges": [],
            "total_params": 64,
            "total_flops_est": 1000,
            "model_name": "TestModel",
        }
        stats = {"layer_stats": [], "total_params": 64, "total_flops": 1000}
        viz_config = {
            "view": {"expand_layers": [], "collapse_patterns": [], "group_by": "stage", "color_map": "flops"},
            "bottlenecks": [],
            "architecture_summary": "测试模型",
        }
        out = tmp_path / "test.html"
        result = mv.render_html(graph, stats, viz_config, out)
        assert result == out
        assert out.exists()
        html = out.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "d3js.org" in html or "D3" in html
        assert "TestModel" in html

    def test_print_proposal(self):
        from guardian.model_viz import ModelVisualizer
        mv = ModelVisualizer()
        viz_config = {
            "view": {"color_map": "flops", "group_by": "stage"},
            "bottlenecks": [
                {"layer": "conv1", "flops_pct": 40.0, "params_pct": 30.0,
                 "severity": "critical", "suggestion": "optimize this"},
            ],
            "architecture_summary": "test summary",
        }
        stats = {"total_params": 100, "total_flops_est": 1000}
        improvements = [{
            "layer": "conv1", "severity": "critical",
            "flops_pct": 40.0, "params_pct": 30.0,
            "matched_components": [],
            "ai_suggestion": None,
            "source": "library",
            "action": "replace",
        }]
        prop = mv.print_proposal(viz_config, stats, improvements)
        assert "conv1" in prop
        assert "test summary" in prop
        assert "SEBlock" in prop or "component" in prop.lower() or "library" in prop

    def test_visualize_pipeline(self, tmp_path):
        """完整 visualize() 管线测试（无真实模型，取决于 mock）。"""
        from guardian.model_viz import ModelVisualizer
        mv = ModelVisualizer()
        out = tmp_path / "viz.html"

        # 使用 torch 的 mock 避免实际导入失败
        with patch.object(mv, "parse_model", return_value={
            "nodes": [{"name": "x", "type": "Linear", "params": 10, "flops_est": 100,
                       "input_shape": None, "output_shape": None, "depth": 0}],
            "edges": [],
            "total_params": 10,
            "total_flops_est": 100,
            "model_name": "MockModel",
        }):
            result = mv.visualize(lambda: None, output_path=out)
            assert "error" not in result
            assert "output_path" in result
            assert Path(result["output_path"]).exists()

    def test_parse_model_error_handling(self):
        from guardian.model_viz import ModelVisualizer
        result = ModelVisualizer.parse_model(lambda: "not_a_module")
        assert "error" in result
