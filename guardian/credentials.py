"""凭据加载：JSON 文件 > 环境变量。

查找 .guardian-credentials.json 的顺序：
1. --config 指定的目录
2. 当前目录及父目录
3. ~/.guardian-credentials.json (全局)
4. 回退环境变量

文件格式:
{
  "base_url": "https://api.deepseek.com/anthropic",
  "auth_token": "sk-xxx",
  "api_key": "sk-xxx",
  "model": "deepseek-v4-pro[1m]",
  "provider": "anthropic"
}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


CREDENTIALS_FILENAME = ".guardian-credentials.json"


def find_credentials_file(start_dir: str | Path | None = None) -> Path | None:
    """向上搜索 .guardian-credentials.json。"""
    start = Path(start_dir) if start_dir else Path.cwd()
    for parent in [start] + list(start.parents)[:5]:
        candidate = parent / CREDENTIALS_FILENAME
        if candidate.exists():
            return candidate
    # 全局
    home = Path.home() / CREDENTIALS_FILENAME
    if home.exists():
        return home
    return None


def load_credentials(start_dir: str | Path | None = None) -> dict[str, str]:
    """加载凭据，JSON 文件优先于环境变量。

    返回的 dict 可直接用于更新 os.environ 或传给 AgentAdvisor 的配置。
    """
    result: dict[str, str] = {}

    # 1. JSON 文件
    path = find_credentials_file(start_dir)
    if path:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # 直接复制到环境变量
                env_map = {
                    "base_url": "ANTHROPIC_BASE_URL",
                    "auth_token": "ANTHROPIC_AUTH_TOKEN",
                    "api_key": "ANTHROPIC_API_KEY",
                    "model": "ANTHROPIC_MODEL",
                    "provider": "ANTHROPIC_PROVIDER",
                }
                for json_key, env_key in env_map.items():
                    if json_key in data and data[json_key]:
                        result[env_key] = str(data[json_key])
        except (json.JSONDecodeError, OSError):
            pass

    # 2. 环境变量回退（JSON 文件不覆盖已有的环境变量）
    for env_key in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                    "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
                    "OPENAI_API_KEY"):
        if env_key not in result and os.environ.get(env_key):
            result[env_key] = os.environ[env_key]

    return result


def apply_credentials(cred: dict[str, str]) -> None:
    """将凭据写入 os.environ，使现有代码透明生效。"""
    for key, value in cred.items():
        os.environ[key] = value
