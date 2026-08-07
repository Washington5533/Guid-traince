"""project_context.py 测试：项目上下文探测、配置文件解析。"""

import tempfile
from pathlib import Path

import pytest

from guardian.project_context import ProjectContext, PROJECT_TEMPLATE, resolve_paths


# ---------------------------------------------------------------------------
# PROJECT_TEMPLATE
# ---------------------------------------------------------------------------

class TestTemplate:
    def test_has_required_sections(self):
        assert "project" in PROJECT_TEMPLATE
        assert "model" in PROJECT_TEMPLATE
        assert "paths" in PROJECT_TEMPLATE

    def test_is_valid_yaml(self):
        import yaml
        raw = yaml.dump(PROJECT_TEMPLATE)
        parsed = yaml.safe_load(raw)
        assert isinstance(parsed, dict)


# ---------------------------------------------------------------------------
# ProjectContext 基本属性
# ---------------------------------------------------------------------------

class TestProjectContextBasics:
    def test_empty_context(self):
        ctx = ProjectContext(start_dir=tempfile.mkdtemp())
        # 新目录无配置文件 → 默认值
        assert isinstance(ctx.name, str)

    def test_ckpt_dir_is_string(self):
        ctx = ProjectContext(start_dir=tempfile.mkdtemp())
        assert isinstance(ctx.ckpt_dir, str)
        assert len(ctx.ckpt_dir) > 0

    def test_default_task_type(self):
        ctx = ProjectContext(start_dir=tempfile.mkdtemp())
        assert ctx.task_type == "classification"


# ---------------------------------------------------------------------------
# 配置文件读写
# ---------------------------------------------------------------------------

class TestConfigFile:
    def test_save_and_reload(self, tmp_path):
        """保存 data 后，新实例能读到相同值。"""
        ctx = ProjectContext(start_dir=str(tmp_path))
        ctx.data["project"]["name"] = "test-project"
        ctx.data["model"]["entry"] = "train:build_model"
        ctx.data["model"]["task_type"] = "classification"

        config_path = ctx.save(tmp_path / ".guardian-project.yaml")
        assert config_path.exists()

        ctx2 = ProjectContext(start_dir=str(tmp_path))
        assert ctx2.name == "test-project"
        assert ctx2.model_entry == "train:build_model"

    def test_partial_config_fills_defaults(self, tmp_path):
        """只写了部分字段的配置文件，其他字段保持默认。"""
        import yaml
        config_path = tmp_path / ".guardian-project.yaml"
        config_path.write_text(yaml.dump({
            "project": {"name": "partial"},
        }), encoding="utf-8")

        ctx = ProjectContext(start_dir=str(tmp_path))
        assert ctx.name == "partial"
        assert ctx.model_entry is None  # 未配置


# ---------------------------------------------------------------------------
# 路径探测辅助方法
# ---------------------------------------------------------------------------

class TestPathHelpers:
    def test_apply_paths_no_extra(self, tmp_path):
        ctx = ProjectContext(start_dir=str(tmp_path))
        ctx.apply_paths()  # 不抛异常

    def test_apply_paths_with_dirs(self, tmp_path):
        extra_dir = tmp_path / "extras"
        extra_dir.mkdir()
        ctx = ProjectContext(start_dir=str(tmp_path))
        ctx.data["paths"]["extra_sys_paths"] = [str(extra_dir)]
        ctx.apply_paths()  # 不抛异常
        import sys
        assert str(extra_dir) in sys.path

    def test_status_returns_list(self, tmp_path):
        ctx = ProjectContext(start_dir=str(tmp_path))
        ctx.data["project"]["name"] = "status-test"
        result = ctx.status()
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# resolve_paths 辅助函数
# ---------------------------------------------------------------------------

class TestResolvePaths:
    def test_returns_project_context(self, tmp_path):
        ctx = ProjectContext(start_dir=str(tmp_path))
        result = resolve_paths(None, ctx)
        assert isinstance(result, ProjectContext)

    def test_cli_overrides_take_priority(self, tmp_path):
        ctx = ProjectContext(start_dir=str(tmp_path))
        cli_log = str(tmp_path / "cli_logs")
        auto_log = str(tmp_path / "auto_logs")
        ctx.data["project"]["log_dir"] = auto_log

        class Args:
            log_dir = cli_log
            ckpt_dir = None
            project_dir = None
            data = None

        resolve_paths(Args(), ctx)
        assert ctx.log_dir == cli_log  # CLI 覆盖
