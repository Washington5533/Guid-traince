"""Guardian Dashboard — 独立 HTTP + WebSocket 控制面板。

用法:
    # 独立进程
    python run.py dashboard --port 8765

    # 训练 + 面板同进程
    python run.py watch --with-dashboard -- python train.py
"""

from .server import DashboardServer

__all__ = ["DashboardServer"]
