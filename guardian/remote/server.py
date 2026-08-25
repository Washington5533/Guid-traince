"""远程通信服务端。

跑在算力服务器上，提供：
- SSE 长连接：实时推送 metrics / anomalies / decisions / gpu_status 事件
- REST API：PC 端轮询 + 审批动作 + 查询历史数据

通信模式：
    push   = 服务器 SSE 推送，PC 在线时零延迟
    pull   = PC 轮询 REST API
    hybrid = SSE 推送 + 断线后 PC 补拉持久化数据（默认）
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

__all__ = ["RemoteServer", "RemoteHandler"]

logger = logging.getLogger(__name__)

# 尝试导入 FastAPI/uvicorn（可选依赖）
_FASTAPI_OK = False
try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse, Response
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    _FASTAPI_OK = True
except ImportError:
    logger.warning("FastAPI/uvicorn 未安装，远程功能不可用。pip install fastapi uvicorn")


# ── 事件类型 ─────────────────────────────────────────────────────────

EVENT_TYPES = {
    "metrics":        "训练指标更新",
    "anomaly":        "异常事件",
    "decision":       "Sub-agent 决策",
    "gpu_status":     "GPU 设备状态",
    "training_start": "训练开始",
    "training_end":   "训练结束",
    "crash":          "训练崩溃",
    "log_line":       "日志行",
    "heartbeat":      "心跳",
    "arch_analysis":  "架构图 AI 解读",
}


# ── 处理器接口 ───────────────────────────────────────────────────────

class RemoteHandler:
    """RemoteServer 的回调接口。

    由 watch 主循环实现，RemoteServer 通过此接口与 guardian 核心交互。
    """

    def get_training_status(self, session_id: str) -> dict:
        """返回训练状态。"""
        raise NotImplementedError

    def get_metrics_history(self, session_id: str, limit: int = 200, offset: int = 0) -> dict:
        """返回指标历史。"""
        raise NotImplementedError

    def get_gpu_status(self) -> dict:
        """返回 GPU 状态。"""
        raise NotImplementedError

    def approve_action(self, session_id: str, action_id: str) -> dict:
        """PC 端审批通过动作。"""
        raise NotImplementedError

    def reject_action(self, session_id: str, action_id: str, reason: str = "") -> dict:
        """PC 端驳回动作。"""
        raise NotImplementedError

    def get_decision_log(self, session_id: str, limit: int = 50) -> list[dict]:
        """返回决策日志。"""
        raise NotImplementedError

    def get_anomaly_history(self, session_id: str, limit: int = 50) -> list[dict]:
        """返回异常事件历史。"""
        raise NotImplementedError

    def get_training_log(self, session_id: str, lines: int = 100, grep: str = "") -> list[str]:
        """返回训练日志尾部。"""
        raise NotImplementedError

    def get_device_info(self) -> dict:
        """返回设备信息（CPU/内存/GPU 概要）。"""
        raise NotImplementedError


# ── Server ───────────────────────────────────────────────────────────

class RemoteServer:
    """远程通信服务端。

    提供 SSE 实时推送 + REST API 双通道。

    使用：
        handler = MyRemoteHandler()  # 实现 RemoteHandler 接口
        server = RemoteServer(handler, port=8765)
        server.start()  # 后台线程启动
        ...
        server.stop()
    """

    def __init__(self, handler: RemoteHandler, port: int = 8765,
                 host: str = "0.0.0.0", auth_token: str | None = None,
                 persist_dir: str | Path | None = None,
                 agent_advisor=None):
        self.handler = handler
        self.port = port
        self.host = host
        self.auth_token = auth_token
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self.agent_advisor = agent_advisor

        # SSE 订阅者
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._global_subs: list[asyncio.Queue] = []
        self._sub_lock = threading.Lock()

        # 训练会话注册
        self._sessions: dict[str, dict] = {}
        self._session_lock = threading.Lock()

        # FastAPI app
        self.app = None
        self._server: uvicorn.Server | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._running = False

        if _FASTAPI_OK:
            self.app = self._build_app()

    # ── 生命周期 ──────────────────────────────────────────────────────

    def start(self) -> None:
        """启动 server（后台线程）。"""
        if not _FASTAPI_OK:
            logger.warning("FastAPI 未安装，RemoteServer 无法启动")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("RemoteServer 启动: http://%s:%d", self.host, self.port)

    def stop(self) -> None:
        """停止 server。"""
        self._running = False
        if self._server:
            self._server.should_exit = True
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("RemoteServer 已停止")

    def _run(self) -> None:
        """在后台线程中运行 uvicorn。"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._server.serve())

    # ── 事件推送 ──────────────────────────────────────────────────────

    def register_session(self, session_id: str, meta: dict) -> None:
        """注册一个训练会话。"""
        with self._session_lock:
            self._sessions[session_id] = {
                "meta": meta,
                "registered_at": time.time(),
                "subscriber_count": 0,
            }

    def push_event(self, session_id: str, event_type: str, data: dict) -> None:
        """SSE 推送事件到所有订阅者。"""
        if not self._running:
            return
        payload = json.dumps({
            "type": event_type,
            "session_id": session_id,
            "timestamp": time.time(),
            "data": data,
        }, ensure_ascii=False, default=str)

        # 持久化
        self._persist_event(session_id, event_type, data)

        # 异步推送到所有订阅者
        try:
            loop = self._loop
            if loop and loop.is_running():
                coro = self._broadcast(session_id, event_type, payload)
                asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:
            pass

    async def _broadcast(self, session_id: str, event_type: str, payload: str) -> None:
        """异步广播到所有 SSE 订阅者。"""
        # 会话级订阅者
        queues = []
        with self._sub_lock:
            if session_id in self._subscribers:
                queues.extend(self._subscribers[session_id])
            # 全局订阅者
            queues.extend(self._global_subs)

        for q in queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def subscribe(self, session_id: str | None = None) -> asyncio.Queue:
        """创建一个 SSE 订阅队列。session_id=None 表示全局订阅。"""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        with self._sub_lock:
            if session_id:
                self._subscribers.setdefault(session_id, []).append(q)
            else:
                self._global_subs.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue, session_id: str | None = None) -> None:
        """取消订阅。"""
        with self._sub_lock:
            if session_id and session_id in self._subscribers:
                self._subscribers[session_id] = [
                    sq for sq in self._subscribers[session_id] if sq is not q
                ]
            if q in self._global_subs:
                self._global_subs.remove(q)

    # ── 持久化 ────────────────────────────────────────────────────────

    def _persist_event(self, session_id: str, event_type: str, data: dict) -> None:
        """将事件追加到 JSONL 持久化文件。"""
        if not self.persist_dir:
            return
        try:
            d = self.persist_dir / session_id
            d.mkdir(parents=True, exist_ok=True)
            path = d / "events.jsonl"
            entry = {
                "timestamp": time.time(),
                "type": event_type,
                "data": data,
            }
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.debug("事件持久化失败", exc_info=True)

    def _replay_recent(self, session_id: str | None = None, limit: int = 50) -> list[str]:
        """生成 SSE 断线补拉的回放消息（hybrid 模式）。

        session_id 为 None 时回放所有会话的最近事件（全局端点）；
        否则只回放该会话。返回已序列化的 SSE data 负载（按时间升序）。
        """
        if not self.persist_dir:
            return []
        entries: list[dict] = []
        try:
            if session_id:
                entries = list(self.load_persisted_events(session_id))
            else:
                for sub in self.persist_dir.iterdir():
                    if not sub.is_dir():
                        continue
                    for e in self.load_persisted_events(sub.name):
                        e["_session_id"] = sub.name
                        entries.append(e)
            entries.sort(key=lambda e: float(e.get("timestamp", 0)))
            entries = entries[-limit:]
        except OSError:
            return []
        out: list[str] = []
        for e in entries:
            payload = {
                "type": e.get("type"),
                "session_id": e.get("_session_id") or session_id,
                "timestamp": e.get("timestamp"),
                "data": e.get("data"),
            }
            out.append(json.dumps(payload, ensure_ascii=False, default=str))
        return out

    def load_persisted_events(self, session_id: str, since: float | None = None) -> list[dict]:
        """加载持久化的事件（PC 端断线补传用）。"""
        if not self.persist_dir:
            return []
        path = self.persist_dir / session_id / "events.jsonl"
        if not path.exists():
            return []
        events = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if since and entry.get("timestamp", 0) < since:
                            continue
                        events.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            logger.debug("加载持久化事件失败", exc_info=True)
        return events

    # ── FastAPI App ───────────────────────────────────────────────────

    def _build_app(self) -> Any:
        """构建 FastAPI 应用。"""
        app = FastAPI(title="Guardian Remote Server", version="0.1.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # ── Auth helpers ──────────────────────────────────────────────

        def _check_auth_header(request: Request) -> bool:
            if not self.auth_token:
                return True
            token = request.headers.get("X-Auth-Token", "")
            return token == self.auth_token

        def _check_auth_query(request: Request) -> bool:
            """Check auth from query param — EventSource cannot set custom headers."""
            if not self.auth_token:
                return True
            token = request.query_params.get("token", "")
            return token == self.auth_token

        # ── SSE 端点 ──────────────────────────────────────────────────
        @app.get("/sse")
        async def sse_global(request: Request):
            """全局 SSE 端点（不指定 session）。"""
            if not _check_auth_query(request):
                return Response(content="unauthorized", status_code=401,
                                media_type="text/plain")
            q = self.subscribe()
            async def event_stream():
                try:
                    # 发送初始心跳
                    yield f"event: heartbeat\ndata: {json.dumps({'ts': time.time()})}\n\n"
                    # 回放最近持久化事件（断线补拉，hybrid 模式）
                    for msg in self._replay_recent(session_id=None, limit=50):
                        yield f"event: message\ndata: {msg}\n\n"
                    while True:
                        msg = await q.get()
                        yield f"event: message\ndata: {msg}\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    self.unsubscribe(q)

            from fastapi.responses import StreamingResponse
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        @app.get("/sse/{session_id}")
        async def sse_session(session_id: str, request: Request):
            """会话级 SSE 端点。"""
            if not _check_auth_query(request):
                return Response(content="unauthorized", status_code=401,
                                media_type="text/plain")
            q = self.subscribe(session_id)
            async def event_stream():
                try:
                    yield f"event: heartbeat\ndata: {json.dumps({'session_id': session_id, 'ts': time.time()})}\n\n"
                    # 回放本会话最近持久化事件（断线补拉，hybrid 模式）
                    for msg in self._replay_recent(session_id=session_id, limit=50):
                        yield f"event: message\ndata: {msg}\n\n"
                    while True:
                        msg = await q.get()
                        yield f"event: message\ndata: {msg}\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    self.unsubscribe(q, session_id)

            from fastapi.responses import StreamingResponse
            return StreamingResponse(event_stream(), media_type="text/event-stream")

        # ── REST API ─────────────────────────────────────────────────

        def _check_auth_header(request: Any) -> bool:
            if not self.auth_token:
                return True
            token = request.headers.get("X-Auth-Token", "")
            return token == self.auth_token

        @app.get("/api/health")
        async def health():
            return {"status": "ok", "gpu_count": self.handler.get_gpu_status().get("gpu_count", 0)}

        @app.get("/api/sessions")
        async def list_sessions():
            with self._session_lock:
                return list(self._sessions.values())

        @app.get("/api/sessions/{session_id}/status")
        async def get_status(session_id: str):
            status = self.handler.get_training_status(session_id)
            return JSONResponse(status)

        @app.get("/api/sessions/{session_id}/metrics")
        async def get_metrics(session_id: str, limit: int = 200, offset: int = 0):
            result = self.handler.get_metrics_history(session_id, limit, offset)
            return JSONResponse(result)

        @app.get("/api/sessions/{session_id}/anomalies")
        async def get_anomalies(session_id: str, limit: int = 50):
            history = self.handler.get_anomaly_history(session_id, limit)
            return JSONResponse({"anomalies": history})

        @app.get("/api/sessions/{session_id}/decisions")
        async def get_decisions(session_id: str, limit: int = 50):
            log = self.handler.get_decision_log(session_id, limit)
            return JSONResponse({"decisions": log})

        @app.get("/api/sessions/{session_id}/log")
        async def get_log(session_id: str, lines: int = 100, grep: str = ""):
            log_lines = self.handler.get_training_log(session_id, lines, grep)
            return JSONResponse({"lines": log_lines})

        @app.get("/api/sessions/{session_id}/pending")
        async def get_pending(session_id: str):
            """获取待审批动作列表（Sub-agent 发出但未处理的）。"""
            # 由 handler 转发到 sub_agent
            pending = self.handler.get_pending_actions(session_id)
            return JSONResponse({"pending": pending})

        @app.post("/api/sessions/{session_id}/approve")
        async def approve_action(session_id: str, request: Request):
            if not _check_auth_header(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            body = await request.json()
            action_id = body.get("action_id", "")
            if not action_id:
                return JSONResponse({"error": "missing action_id"}, status_code=400)
            result = self.handler.approve_action(session_id, action_id)
            return JSONResponse(result)

        @app.post("/api/sessions/{session_id}/reject")
        async def reject_action(session_id: str, request: Request):
            if not _check_auth_header(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            body = await request.json()
            action_id = body.get("action_id", "")
            reason = body.get("reason", "")
            result = self.handler.reject_action(session_id, action_id, reason)
            return JSONResponse(result)

        @app.get("/api/device/gpu")
        async def get_gpu():
            return JSONResponse(self.handler.get_gpu_status())

        @app.get("/api/device/info")
        async def get_device_info():
            return JSONResponse(self.handler.get_device_info())

        @app.get("/api/events/{session_id}")
        async def get_events(session_id: str, since: float | None = None):
            """加载持久化事件（PC 断线补传）。"""
            events = self.load_persisted_events(session_id, since)
            return JSONResponse({"events": events})

        @app.post("/api/sessions/{session_id}/restart")
        async def restart_training(session_id: str, request: Request):
            """手动触发重启。"""
            if not _check_auth_header(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            body = await request.json()
            action = body.get("action", "resume_unchanged")
            params = body.get("params", {})
            # 转发到 handler
            result = self.handler.trigger_recovery(session_id, action, params)
            return JSONResponse(result)

        # ── Architecture analysis ─────────────────────────────────────

        @app.post("/api/arch/analyze")
        async def arch_analyze(request: Request):
            """分析模型架构，返回 D3 可渲染的 JSON 数据。

            Request body:
                { "model_entry": "module:function", "project_dir": "/path",
                  "session_id": "...", "agent": true }

            也可以从已注册会话取 model_entry（通过 session_id 参数）。
            若 agent=true 且 agent_advisor 可用，会在后台线程调用 LLM 生成解读，
            然后通过 SSE arch_analysis 事件推送。
            """
            if not _check_auth_header(request):
                return JSONResponse({"error": "unauthorized"}, status_code=401)
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "invalid JSON body"}, 400)

            model_entry = body.get("model_entry", "")
            project_dir = body.get("project_dir", "")
            session_id = body.get("session_id", "")
            want_agent = body.get("agent", False)

            # Fall back to registered session meta if model_entry not provided.
            if not model_entry and session_id:
                with self._session_lock:
                    sess = self._sessions.get(session_id, {})
                    meta = sess.get("meta", {})
                    model_entry = model_entry or meta.get("model_entry", "")
                    project_dir = project_dir or meta.get("project_dir", "")

            if not model_entry:
                return JSONResponse(
                    {"error": "缺少 model_entry（格式: module:function）"}, 400
                )

            try:
                from ..arch_analyzer import ArchAnalyzer
                import sys
                import importlib

                analyzer = ArchAnalyzer()

                if project_dir and project_dir not in sys.path:
                    sys.path.insert(0, project_dir)

                mod_parts = model_entry.split(":", 1)
                if len(mod_parts) != 2:
                    return JSONResponse(
                        {"error": f"invalid model_entry: {model_entry}（需要 module:function 格式）"},
                        400,
                    )
                mod = importlib.import_module(mod_parts[0])
                model_fn = getattr(mod, mod_parts[1])
                result = analyzer.analyze(model_fn)

                if "error" in result:
                    return JSONResponse(result, 400)

                # Agent narration: 后台线程调用 LLM 解读，通过 SSE 推送
                if want_agent and self.agent_advisor and self.agent_advisor.is_enabled("summary_narrative"):
                    _sid = session_id or "default"
                    threading.Thread(
                        target=self._narrate_arch_result,
                        args=(result, _sid),
                        daemon=True,
                        name="arch-narrate",
                    ).start()
                    result["agent_pending"] = True

                return JSONResponse(result)
            except Exception as e:
                logger.warning("架构分析失败: %s", e, exc_info=True)
                return JSONResponse({"error": str(e)}, 500)

        return app

    def _narrate_arch_result(self, arch_data: dict, session_id: str) -> None:
        """后台线程：调用 AgentAdvisor.narrate() 生成解读，通过 SSE 推送。"""
        try:
            advisor = self.agent_advisor
            if not advisor:
                return
            prompt_template = (
                "以下是一个深度学习模型的架构分析结果。请用中文生成一段 200-400 字的解读，"
                "包含：模型规模概述、参数分布特点、计算瓶颈、优化建议。\n\n"
                "{summary}"
            )
            narration = advisor.narrate(arch_data, prompt_template)
            if narration:
                self.push_event(session_id, "arch_analysis", {
                    "narration": narration,
                    "model_name": arch_data.get("model_name", "Model"),
                    "total_params": arch_data.get("total_params", 0),
                    "bottleneck_count": arch_data.get("bottleneck_count", 0),
                })
            else:
                self.push_event(session_id, "arch_analysis", {
                    "narration": None,
                    "error": "AI 解读生成失败，请检查 API Key 配置",
                })
        except Exception as e:
            logger.warning("架构 AI 解读失败: %s", e, exc_info=True)
            self.push_event(session_id, "arch_analysis", {
                "narration": None,
                "error": str(e),
            })

    def __repr__(self) -> str:
        return f"RemoteServer(host={self.host}:{self.port}, running={self._running})"
