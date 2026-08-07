"""Guardian 统一日志配置。

使用方式:
    from guardian.logging_config import get_logger
    logger = get_logger(__name__)

配置（guardian.yaml 中）:
    logging:
      level: INFO           # DEBUG | INFO | WARNING | ERROR
      file: logs/guardian.log  # 文件输出（可选）
      file_level: DEBUG     # 文件日志级别（默认 DEBUG）
      format: "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
      datefmt: "%m-%d %H:%M:%S"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

_loggers: dict[str, logging.Logger] = {}
_initialized: bool = False
_root_handler: logging.Handler | None = None
_file_handler: logging.Handler | None = None

# 默认格式
DEFAULT_FORMAT = "%(asctime)s [%(levelname)-5s] %(name)s: %(message)s"
DEFAULT_DATEFMT = "%m-%d %H:%M:%S"


def configure(cfg: dict[str, Any] | None = None) -> None:
    """全局初始化（run.py 启动时调用一次）。"""
    global _initialized, _root_handler, _file_handler
    if _initialized:
        return

    log_cfg = (cfg or {}).get("logging") or {}
    level_name = log_cfg.get("level", "INFO").upper()
    file_path = log_cfg.get("file")
    file_level_name = log_cfg.get("file_level", "DEBUG").upper()
    fmt = log_cfg.get("format", DEFAULT_FORMAT)
    datefmt = log_cfg.get("datefmt", DEFAULT_DATEFMT)

    root = logging.getLogger("guardian")
    root.setLevel(logging.DEBUG)  # root 设最低，handler 各控各的

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # 控制台 handler
    _root_handler = logging.StreamHandler(sys.stderr)
    _root_handler.setLevel(getattr(logging, level_name, logging.INFO))
    _root_handler.setFormatter(formatter)
    root.addHandler(_root_handler)

    # 文件 handler（可选）
    if file_path:
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        _file_handler = logging.FileHandler(str(p), encoding="utf-8")
        _file_handler.setLevel(getattr(logging, file_level_name, logging.DEBUG))
        _file_handler.setFormatter(formatter)
        root.addHandler(_file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """获取 guardian 命名空间下的 logger。

    name 通常是 __name__（如 guardian.mcp_server），自动去掉 guardian. 前缀后拼接。
    """
    if not _initialized:
        configure()  # 自动初始化（使用默认配置）

    # 规范化名称：guardian.xxx
    if name.startswith("guardian."):
        pass
    elif name.startswith("guardian"):
        name = "guardian." + name[len("guardian"):].lstrip(".")
    else:
        # 外部调用（如 run.py）→ 直接挂在 guardian 下
        name = f"guardian.{name.split('.')[-1]}" if "." in name else f"guardian.{name}"

    if name not in _loggers:
        _loggers[name] = logging.getLogger(name)
    return _loggers[name]
