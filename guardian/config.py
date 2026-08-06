"""配置加载：内置默认值 < 配置文件 < 环境变量 < 命令行。

对应 checkpoint/configuration.md。secrets 一律只走环境变量，
配置文件里存的是变量名（如 api_key_env）而非值本身。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml 是核心依赖
    yaml = None

ENV_PREFIX = "GUARDIAN_"

# 禁止写在 yaml 里的键名：这些是 secret 本身，只接受 *_env 形式
FORBIDDEN_SECRET_KEYS = {"api_key", "write_token", "webhook_url", "smtp_password"}

DEFAULTS: dict[str, Any] = {
    "project": {
        "name": "guardian-run",
        "ckpt_dir": "./checkpoints",
        "log_dir": "./logs",
        "device": "auto",
    },
    "watchdog": {
        "max_retries": 3,
        "restart_delay": 10,
        "oom_batch_reduce_ratio": 0.5,
        "min_batch_size": 8,
        "sigterm_grace": 30,
        "no_progress_timeout": 1800,
        "no_progress_kill_after": None,
        "keep_training_on_exit": True,
    },
    "monitor": {
        "enabled": True,
        "poll_interval": 10,
        "hardware_poll_interval": 30,
        "sliding_window": 50,
        "loss_spike_ratio": 0.5,
        "loss_stagnation_steps": 500,
        "loss_stagnation_threshold": 0.001,
        "gpu_idle_threshold": 20,
        "gpu_temp_threshold": 85,
    },
    "notifier": {
        "channels": ["terminal"],
        "cooldown": 300,
        "webhook_url_env": "GUARDIAN_WEBHOOK_URL",
        "webhook_timeout": 10,
    },
    "checkpoint": {
        "poll_interval": 30,
        "save_top_k": 5,
        "keep_recent": 2,
        "quick_val_sample_ratio": 0.05,
        "full_val_every_n": 5,
        "stability_checks": 2,
    },
    "preflight": {
        "enabled": True,
        "test_batch_sizes": [1, 2, 4],
        "memory_margin": 0.2,
    },
    "agent": {
        "enabled": False,
        "provider": "anthropic",
        "model": None,
        "api_key_env": "ANTHROPIC_API_KEY",
        "decision_timeout": 8,
        "consecutive_failure_threshold": 5,
        "circuit_breaker_cooldown": 600,
        "decision_points": {
            "monitor_response": True,
            "watchdog_recovery": True,
            "summary_narrative": True,
            "select_metric": True,
            "select_adjust_path": True,
        },
    },
    "mcp": {
        "enabled": False,
        "transport": "stdio",
        "tcp_port": 8765,
        "enable_write_tools": False,
        "write_token_env": "GUARDIAN_MCP_TOKEN",
        "state_refresh_interval": 5,
        "dedup_window": 300,
        "default_result_limit": 200,
    },
    "contract": {
        "path": "configs/contract.yaml",
        "strict_mode": False,
        "agent_can_propose": True,
        "proposal_log": "logs/contract_proposals.json",
    },
    "gallery": {
        "default_max_images": 50,
        "streamlit_port": 8501,
        "supported_tasks": ["classification", "detection", "segmentation"],
    },
    "visualization": {
        "color_map_default": "flops",
        "bottleneck_threshold_pct": 25,
        "output_format": "html",
    },
    "experiment_query": {
        "log_dir": "./logs",
        "max_compare_experiments": 5,
    },
    "inference": {
        "scripts_dir": "./scripts",
        "default_batch_size": 32,
        "device": "cuda",
    },
}


class ConfigError(Exception):
    """配置非法（如把 secret 明文写进 yaml）。"""


def _deep_merge(base: dict, override: dict) -> dict:
    """递归合并，override 覆盖 base。不修改入参。"""
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _check_secrets(cfg: dict, path: str = "") -> None:
    """拒绝把 secret 明文写进配置文件。"""
    for key, val in cfg.items():
        full = f"{path}.{key}" if path else key
        if key in FORBIDDEN_SECRET_KEYS:
            raise ConfigError(
                f"配置项 {full!r} 不接受明文 secret。请改用 {key}_env "
                f"指定环境变量名，例如 {key}_env: MY_ENV_VAR"
            )
        if isinstance(val, dict):
            _check_secrets(val, full)


def _unknown_keys(cfg: dict, ref: dict, path: str = "") -> list[str]:
    """列出不在 DEFAULTS 里的键，用于提示拼写错误。

    只对有参考结构的层级递归；自由形态的子树（如 contract 的映射表）跳过。
    """
    unknown: list[str] = []
    for key, val in cfg.items():
        full = f"{path}.{key}" if path else key
        if key not in ref:
            unknown.append(full)
            continue
        if isinstance(val, dict) and isinstance(ref[key], dict) and ref[key]:
            unknown.extend(_unknown_keys(val, ref[key], full))
    return unknown


def _coerce(raw: str) -> Any:
    """把环境变量字符串转成合适的类型。"""
    low = raw.strip().lower()
    if low in ("null", "none", ""):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if "," in raw:
        return [item.strip() for item in raw.split(",")]
    return raw


def _env_overrides() -> dict:
    """GUARDIAN_WATCHDOG__MAX_RETRIES=5 -> {"watchdog": {"max_retries": 5}}

    双下划线表示层级。secrets 用的 *_ENV 变量本身不参与覆盖。
    """
    out: dict = {}
    for env_key, raw in os.environ.items():
        if not env_key.startswith(ENV_PREFIX):
            continue
        body = env_key[len(ENV_PREFIX):]
        if not body:
            continue
        parts = [p.lower() for p in body.split("__") if p]
        if not parts:
            continue
        # 顶层段必须是已知配置节，否则可能是 GUARDIAN_WEBHOOK_URL 这类 secret
        if parts[0] not in DEFAULTS:
            continue
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce(raw)
    return out


def load_config(
    path: str | Path | None = None,
    cli_overrides: dict | None = None,
    *,
    strict_unknown: bool = False,
) -> dict:
    """加载配置：DEFAULTS < 文件 < 环境变量 < 命令行。

    文件缺失不报错——用默认值继续，并在 config["_warnings"] 里记一条，
    对应 configuration.md 的"全默认值可用"校验项。
    """
    warnings: list[str] = []
    file_cfg: dict = {}

    if path is not None:
        p = Path(path)
        if p.exists():
            if yaml is None:
                raise ConfigError("需要 pyyaml 才能解析配置文件：pip install pyyaml")
            loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ConfigError(f"{p} 顶层必须是映射（key: value），实际为 {type(loaded).__name__}")
            file_cfg = loaded
            _check_secrets(file_cfg)
        else:
            warnings.append(f"配置文件不存在：{p}，使用内置默认值（见 checkpoint/configuration.md）")

    cfg = _deep_merge(DEFAULTS, file_cfg)
    cfg = _deep_merge(cfg, _env_overrides())
    if cli_overrides:
        cfg = _deep_merge(cfg, {k: v for k, v in cli_overrides.items() if v is not None})

    unknown = _unknown_keys(file_cfg, DEFAULTS)
    if unknown:
        msg = "配置中有未识别的键（可能拼写错误）：" + ", ".join(sorted(unknown))
        if strict_unknown:
            raise ConfigError(msg)
        warnings.append(msg)

    cfg["_warnings"] = warnings
    return cfg


def resolve_secret(cfg: dict, env_key_name: str) -> str | None:
    """按 *_env 指向的环境变量取 secret 值，取不到返回 None。"""
    env_var = cfg.get(env_key_name)
    if not env_var:
        return None
    return os.environ.get(str(env_var)) or None
