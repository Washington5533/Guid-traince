"""远程通信客户端 SDK。

跑在 PC 端，连接算力服务器：
- SSE 订阅实时事件流
- REST API 查询历史数据和审批动作
- 断线自动重连 + 补拉历史数据
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Optional

__all__ = ["GuardianClient", "ConnectionStatus"]

logger = logging.getLogger(__name__)


class ConnectionStatus:
    """连接状态枚举。"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


# 事件类型常量
EVENT_METRICS = "metrics"
EVENT_ANOMALY = "anomaly"
EVENT_DECISION = "decision"
EVENT_GPU_STATUS = "gpu_status"
EVENT_TRAINING_START = "training_start"
EVENT_TRAINING_END = "training_end"
EVENT_CRASH = "crash"
EVENT_LOG_LINE = "log_line"
EVENT_HEARTBEAT = "heartbeat"


class GuardianClient:
    """Guardian 远程客户端 SDK。

    使用示例：
        client = GuardianClient("http://192.168.1.100:8765")

        @client.on(EVENT_METRICS)
        def on_metrics(data):
            print(f"Loss: {data.get('loss')}")

        @client.on(EVENT_ANOMALY)
        def on_anomaly(data):
            print(f"Anomaly: {data.get('description')}")

        client.connect()
        ...
        client.disconnect()
    """

    def __init__(self, server_url: str, auth_token: str | None = None,
                 auto_reconnect: bool = True, reconnect_interval: float = 3.0):
        """
        Args:
            server_url: 算力服务器地址，如 "http://192.168.1.100:8765"
            auth_token: 可选鉴权 token
            auto_reconnect: 断线自动重连
            reconnect_interval: 重连间隔（秒）
        """
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval

        # 事件回调
        self._handlers: dict[str, list[Callable]] = {}
        self._status_handlers: list[Callable] = []

        # 连接状态
        self._status = ConnectionStatus.DISCONNECTED
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._running = False

        # SSE 连接
        self._sse_response = None
        self._sse_line_buffer = ""

    # ── 事件订阅 ──────────────────────────────────────────────────────

    def on(self, event_type: str, callback: Callable) -> None:
        """订阅事件。

        Args:
            event_type: 事件类型（EVENT_METRICS / EVENT_ANOMALY 等）
            callback:  回调函数 (data: dict) -> None
        """
        self._handlers.setdefault(event_type, []).append(callback)

    def off(self, event_type: str, callback: Callable) -> None:
        """取消事件订阅。"""
        if event_type in self._handlers:
            self._handlers[event_type] = [c for c in self._handlers[event_type] if c is not callback]

    def on_status_change(self, callback: Callable) -> None:
        """订阅连接状态变化。callback(status: str) -> None"""
        self._status_handlers.append(callback)

    # ── 连接管理 ──────────────────────────────────────────────────────

    def connect(self) -> bool:
        """连接到算力服务器（阻塞直到连接成功或失败）。"""
        self._set_status(ConnectionStatus.CONNECTING)
        try:
            # 测试连接
            info = self._http_get("/api/health", timeout=5)
            if info.get("status") != "ok":
                self._set_status(ConnectionStatus.ERROR)
                return False
        except Exception as exc:
            logger.error("连接服务器失败: %s", exc)
            self._set_status(ConnectionStatus.ERROR)
            return False

        # 启动事件循环
        self._running = True
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._task = self._loop.create_task(self._event_loop())
        self._set_status(ConnectionStatus.CONNECTED)
        return True

    def disconnect(self) -> None:
        """断开连接。"""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._set_status(ConnectionStatus.DISCONNECTED)

    def run_forever(self) -> None:
        """阻塞运行事件循环（在独立线程中）。"""
        if not self._loop:
            self.connect()
        if self._loop:
            try:
                self._loop.run_forever()
            except asyncio.CancelledError:
                pass

    # ── REST API ─────────────────────────────────────────────────────

    def get_training_status(self, session_id: str) -> dict:
        """获取训练状态。"""
        return self._http_get(f"/api/sessions/{session_id}/status")

    def get_metrics(self, session_id: str, limit: int = 200) -> dict:
        """获取指标历史。"""
        return self._http_get(f"/api/sessions/{session_id}/metrics?limit={limit}")

    def get_anomalies(self, session_id: str, limit: int = 50) -> dict:
        """获取异常历史。"""
        return self._http_get(f"/api/sessions/{session_id}/anomalies?limit={limit}")

    def get_decisions(self, session_id: str, limit: int = 50) -> dict:
        """获取决策日志。"""
        return self._http_get(f"/api/sessions/{session_id}/decisions?limit={limit}")

    def get_gpu_status(self) -> dict:
        """获取 GPU 状态。"""
        return self._http_get("/api/device/gpu")

    def get_device_info(self) -> dict:
        """获取设备信息。"""
        return self._http_get("/api/device/info")

    def get_pending_actions(self, session_id: str) -> dict:
        """获取待审批动作列表。"""
        return self._http_get(f"/api/sessions/{session_id}/pending")

    def approve(self, session_id: str, action_id: str) -> dict:
        """审批通过动作。"""
        return self._http_post(
            f"/api/sessions/{session_id}/approve",
            {"action_id": action_id},
        )

    def reject(self, session_id: str, action_id: str, reason: str = "") -> dict:
        """驳回动作。"""
        return self._http_post(
            f"/api/sessions/{session_id}/reject",
            {"action_id": action_id, "reason": reason},
        )

    def restart_training(self, session_id: str, action: str = "resume_unchanged",
                         params: dict | None = None) -> dict:
        """手动触发重启。"""
        return self._http_post(
            f"/api/sessions/{session_id}/restart",
            {"action": action, "params": params or {}},
        )

    def load_history(self, session_id: str, since: float | None = None) -> dict:
        """加载持久化历史事件（断线补传）。"""
        url = f"/api/events/{session_id}"
        if since:
            url += f"?since={since}"
        return self._http_get(url)

    def list_sessions(self) -> list[dict]:
        """列出所有训练会话。"""
        return self._http_get("/api/sessions")

    # ── 内部 ─────────────────────────────────────────────────────────

    async def _event_loop(self) -> None:
        """SSE 事件循环。"""
        while self._running:
            try:
                await self._connect_sse()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("SSE 连接异常: %s", exc)
                if self.auto_reconnect and self._running:
                    self._set_status(ConnectionStatus.RECONNECTING)
                    await asyncio.sleep(self.reconnect_interval)
                else:
                    break
        self._set_status(ConnectionStatus.DISCONNECTED)

    async def _connect_sse(self) -> None:
        """连接 SSE 流。"""
        url = f"{self.server_url}/sse"
        headers = {}
        if self.auth_token:
            headers["X-Auth-Token"] = self.auth_token

        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=None)) as resp:
                resp.raise_for_status()
                self._set_status(ConnectionStatus.CONNECTED)
                async for line in resp.content:
                    if not self._running:
                        break
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    self._process_sse_line(line)

    def _process_sse_line(self, line: str) -> None:
        """处理一条 SSE 行。"""
        if line.startswith("event:"):
            self._sse_event_type = line[6:].strip()
        elif line.startswith("data:"):
            data_str = line[5:].strip()
            try:
                data = json.loads(data_str)
                self._dispatch_event(self._sse_event_type or "message", data)
            except json.JSONDecodeError:
                pass

    def _dispatch_event(self, event_type: str, data: dict) -> None:
        """分发事件到注册的回调。"""
        for callback in self._handlers.get(event_type, []):
            try:
                callback(data)
            except Exception:
                logger.debug("事件回调异常: %s", event_type, exc_info=True)

        # 万能回调
        for callback in self._handlers.get("*", []):
            try:
                callback(event_type, data)
            except Exception:
                pass

    def _set_status(self, status: str) -> None:
        self._status = status
        for callback in self._status_handlers:
            try:
                callback(status)
            except Exception:
                pass

    # ── HTTP 辅助 ────────────────────────────────────────────────────

    def _http_get(self, path: str, timeout: float = 10.0) -> dict:
        """发送 HTTP GET 请求。"""
        import urllib.request
        url = f"{self.server_url}{path}"
        req = urllib.request.Request(url, method="GET")
        if self.auth_token:
            req.add_header("X-Auth-Token", self.auth_token)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post(self, path: str, body: dict, timeout: float = 10.0) -> dict:
        """发送 HTTP POST 请求。"""
        import urllib.request
        url = f"{self.server_url}{path}"
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.auth_token:
            req.add_header("X-Auth-Token", self.auth_token)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_connected(self) -> bool:
        return self._status == ConnectionStatus.CONNECTED

    def __repr__(self) -> str:
        return f"GuardianClient(url={self.server_url}, status={self._status})"
