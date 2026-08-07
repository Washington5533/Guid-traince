"""config.py 测试：分层加载、密钥检测、环境变量覆盖、类型转换。"""

import os
import tempfile
from pathlib import Path

import pytest

from guardian.config import (
    DEFAULTS,
    ConfigError,
    _check_secrets,
    _coerce,
    _deep_merge,
    _env_overrides,
    _unknown_keys,
    load_config,
    resolve_secret,
)


# ---------------------------------------------------------------------------
# _deep_merge
# ---------------------------------------------------------------------------

class TestDeepMerge:
    def test_override_scalar(self):
        base = {"a": 1, "b": 2}
        assert _deep_merge(base, {"a": 99}) == {"a": 99, "b": 2}

    def test_nested_merge(self):
        base = {"x": {"y": 1, "z": 2}}
        override = {"x": {"y": 99}}
        result = _deep_merge(base, override)
        assert result["x"]["y"] == 99
        assert result["x"]["z"] == 2  # 未覆盖的保留

    def test_does_not_mutate_base(self):
        base = {"a": 1}
        _deep_merge(base, {"a": 2})
        assert base["a"] == 1  # 原值不变

    def test_new_key_added(self):
        base = {"a": 1}
        result = _deep_merge(base, {"b": 2})
        assert result == {"a": 1, "b": 2}


# ---------------------------------------------------------------------------
# _check_secrets
# ---------------------------------------------------------------------------

class TestCheckSecrets:
    def test_rejects_plaintext_api_key(self):
        with pytest.raises(ConfigError, match="不接受明文 secret"):
            _check_secrets({"agent": {"api_key": "sk-123"}})

    def test_rejects_write_token_in_mcp(self):
        with pytest.raises(ConfigError, match="不接受明文 secret"):
            _check_secrets({"mcp": {"write_token": "my-token"}})

    def test_accepts_api_key_env(self):
        # api_key_env 是合法的——存的是环境变量名而非 secret 本身
        _check_secrets({"agent": {"api_key_env": "MY_KEY"}})

    def test_accepts_normal_values(self):
        _check_secrets({"watchdog": {"max_retries": 5}})


# ---------------------------------------------------------------------------
# _coerce
# ---------------------------------------------------------------------------

class TestCoerce:
    def test_null_variants(self):
        for v in ("null", "none", "NULL", ""):
            assert _coerce(v) is None

    def test_booleans(self):
        assert _coerce("true") is True
        assert _coerce("TRUE") is True
        assert _coerce("false") is False

    def test_integer(self):
        assert _coerce("42") == 42
        assert _coerce("-1") == -1

    def test_float(self):
        assert _coerce("3.14") == 3.14

    def test_comma_list(self):
        assert _coerce("terminal, webhook") == ["terminal", "webhook"]

    def test_string_passthrough(self):
        assert _coerce("some-string") == "some-string"


# ---------------------------------------------------------------------------
# _unknown_keys
# ---------------------------------------------------------------------------

class TestUnknownKeys:
    def test_detects_typo(self):
        unknown = _unknown_keys({"watchdog": {"max_retrise": 3}}, DEFAULTS)
        assert any("max_retrise" in u for u in unknown)

    def test_no_false_positive(self):
        unknown = _unknown_keys({"watchdog": {"max_retries": 3}}, DEFAULTS)
        assert len(unknown) == 0

    def test_nested_unknown(self):
        unknown = _unknown_keys({"monitor": {"poll_interval": 5, "bad_key": 1}}, DEFAULTS)
        assert any("bad_key" in u for u in unknown)


# ---------------------------------------------------------------------------
# _env_overrides
# ---------------------------------------------------------------------------

class TestEnvOverrides:
    def test_override_watchdog(self, monkeypatch):
        monkeypatch.setenv("GUARDIAN_WATCHDOG__MAX_RETRIES", "7")
        overrides = _env_overrides()
        assert overrides["watchdog"]["max_retries"] == 7

    def test_override_float(self, monkeypatch):
        monkeypatch.setenv("GUARDIAN_MONITOR__LOSS_SPIKE_RATIO", "0.8")
        overrides = _env_overrides()
        assert overrides["monitor"]["loss_spike_ratio"] == 0.8

    def test_ignores_non_guardian_vars(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        overrides = _env_overrides()
        assert "path" not in overrides  # noqa — key wouldn't be added

    def test_ignores_unknown_section(self, monkeypatch):
        # GUARDIAN_FOO__BAR 的 foo 不在 DEFAULTS 顶层 → 忽略
        monkeypatch.setenv("GUARDIAN_FOO__BAR", "1")
        overrides = _env_overrides()
        assert "foo" not in overrides


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_defaults_only(self):
        cfg = load_config(None)
        assert cfg["watchdog"]["max_retries"] == DEFAULTS["watchdog"]["max_retries"]
        assert "_warnings" in cfg

    def test_missing_file_warns(self):
        cfg = load_config("/nonexistent/path/config.yaml")
        assert any("不存在" in w for w in cfg["_warnings"])

    def test_file_override(self, tmp_path):
        import yaml
        p = tmp_path / "guardian.yaml"
        p.write_text(yaml.dump({"watchdog": {"max_retries": 99}}), encoding="utf-8")
        cfg = load_config(str(p))
        assert cfg["watchdog"]["max_retries"] == 99

    def test_cli_override_highest_priority(self, tmp_path):
        import yaml
        p = tmp_path / "guardian.yaml"
        p.write_text(yaml.dump({"watchdog": {"max_retries": 10}}), encoding="utf-8")
        cfg = load_config(str(p), cli_overrides={"watchdog": {"max_retries": 5}})
        assert cfg["watchdog"]["max_retries"] == 5  # CLI 覆盖文件

    def test_strict_mode_rejects_unknown(self, tmp_path):
        import yaml
        p = tmp_path / "guardian.yaml"
        p.write_text(yaml.dump({"watchdog": {"typo_key": 1}}), encoding="utf-8")
        with pytest.raises(ConfigError, match="未识别"):
            load_config(str(p), strict_unknown=True)

    def test_strict_mode_off_just_warns(self, tmp_path):
        import yaml
        p = tmp_path / "guardian.yaml"
        p.write_text(yaml.dump({"watchdog": {"typo_key": 1}}), encoding="utf-8")
        cfg = load_config(str(p), strict_unknown=False)
        assert any("未识别" in w for w in cfg["_warnings"])

    def test_rejects_plaintext_secret_in_file(self, tmp_path):
        import yaml
        p = tmp_path / "guardian.yaml"
        p.write_text(yaml.dump({"mcp": {"write_token": "hardcoded-secret"}}), encoding="utf-8")
        with pytest.raises(ConfigError, match="不接受明文 secret"):
            load_config(str(p))
