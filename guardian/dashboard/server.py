"""Dashboard Server — 独立 HTTP + WebSocket 控制面板。

启动方式：
    python run.py dashboard              # 独立进程
    python run.py watch --with-dashboard  # 训练 + 面板同进程

API 端点：
    GET  /api/processes                  - 进程列表
    GET  /api/process/<id>               - 进程详情
    GET  /api/process/<id>/metrics       - 指标历史（分页）
    GET  /api/process/<id>/log           - 日志 tail（分页+实时）
    GET  /api/process/<id>/model         - 模型结构
    GET  /api/process/<id>/gallery       - 图库数据
    GET  /api/process/<id>/anomalies     - 异常事件
    POST /api/process/<id>/ai/analyze    - AI 分析
    POST /api/process/<id>/ai/chat       - AI 交互追问
    POST /api/compare                    - 多进程对比
    WS   /ws/process/<id>                - 实时流

解耦保证：dashboard 运行在守护线程，崩溃不影响训练，训练不阻塞页面。
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

# FastAPI 可选导入
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
    _FASTAPI_OK = True
except ImportError:
    _FASTAPI_OK = False
    FastAPI = object
    WebSocket = object


class DashboardServer:
    """控制面板 HTTP + WebSocket 服务器。"""

    def __init__(
        self,
        config: dict | None = None,
        *,
        port: int = 8765,
        host: str = "127.0.0.1",
        static_dir: str | Path | None = None,
    ):
        self.cfg = config or {}
        self.port = int(self.cfg.get("dashboard", {}).get("port", port))
        self.host = self.cfg.get("dashboard", {}).get("host", host)
        self.static_dir = Path(static_dir) if static_dir else Path(__file__).parent / "static"

        # 状态注册表：{process_id: ProcessState}
        self._processes: dict[str, dict] = {}
        # WebSocket 连接：{process_id: [ws, ...]}
        self._subscribers: dict[str, list[WebSocket]] = {}
        # 全局订阅
        self._global_subs: list[WebSocket] = []
        self._lock = threading.Lock()

        self.app = self._build_app() if _FASTAPI_OK else None

    # ------------------------------------------------------------------
    # 进程注册
    # ------------------------------------------------------------------

    def register_process(self, process_id: str, state: dict) -> None:
        """注册或更新一个被守护的训练进程。"""
        state.setdefault("process_id", process_id)
        state.setdefault("registered_at", time.time())
        with self._lock:
            self._processes[process_id] = state
        self._safe_broadcast_global({"type": "process_update", "process_id": process_id, "status": state.get("status")})

    def _safe_broadcast_process(self, process_id: str, msg: dict) -> None:
        if hasattr(self, "_loop") and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._broadcast_process(process_id, msg), self._loop)

    def _safe_broadcast_global(self, msg: dict) -> None:
        if hasattr(self, "_loop") and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._broadcast_global(msg), self._loop)

    def update_process(self, process_id: str, patch: dict) -> None:
        with self._lock:
            if process_id in self._processes:
                self._processes[process_id].update(patch)
            else:
                return
        self._safe_broadcast_process(process_id, {"type": "state_patch", "patch": patch})

    def push_metrics(self, process_id: str, metrics: dict) -> None:
        self._safe_broadcast_process(process_id, {"type": "metrics", "data": metrics})

    def push_log_line(self, process_id: str, line: str) -> None:
        self._safe_broadcast_process(process_id, {"type": "log_line", "line": line})

    def bind_guardian(self, process_id: str, *, monitor=None, watchdog=None, advisor=None,
                      analyzer=None, summary=None, contract=None):
        """绑定 guardian 模块实例（共享模式）。"""
        with self._lock:
            if process_id in self._processes:
                self._processes[process_id]["_monitor"] = monitor
                self._processes[process_id]["_watchdog"] = watchdog
                self._processes[process_id]["_advisor"] = advisor
                self._processes[process_id]["_analyzer"] = analyzer
                self._processes[process_id]["_summary"] = summary
                self._processes[process_id]["_contract"] = contract

    # ------------------------------------------------------------------
    # FastAPI 构建
    # ------------------------------------------------------------------

    def _build_app(self):
        app = FastAPI(title="Guardian Dashboard", docs_url=None, redoc_url=None)

        # ---- 进程注册（HTTP，供 watch --with-dashboard 调用） ----
        @app.post("/api/register")
        async def register(payload: dict):
            pid = payload.get("process_id", f"proc_{len(self._processes)}")
            with self._lock:
                self._processes[pid] = {
                    "process_id": pid,
                    "name": payload.get("name", pid),
                    "status": payload.get("status", "starting"),
                    "command": payload.get("command", ""),
                    "model_entry": payload.get("model_entry", ""),
                    "project_dir": payload.get("project_dir", ""),
                    "log_file": payload.get("log_file", ""),
                    "registered_at": time.time(),
                    "_metrics_history": [],
                    "_gpu_history": [],
                    "_log_lines": [],
                }
            return JSONResponse({"ok": True, "process_id": pid})

        @app.post("/api/process/{process_id}/push")
        async def push_update(process_id: str, payload: dict):
            with self._lock:
                if process_id not in self._processes:
                    return JSONResponse({"error": "unknown process"}, 404)
                s = self._processes[process_id]
                patch = payload.get("patch", {})
                s.update(patch)
                # 处理 log_file（可能是注册后补传的）
                if "log_file" in patch:
                    s["log_file"] = patch["log_file"]
                # 累积 metrics 历史
                mdata = payload.get("data")
                if mdata:
                    hist = s.setdefault("_metrics_history", [])
                    hist.append(mdata)
                    if len(hist) > 2000:
                        s["_metrics_history"] = hist[-2000:]
                    s["latest_metrics"] = mdata
                # 累积 GPU 历史
                gpu = payload.get("gpu")
                if gpu:
                    gh = s.setdefault("_gpu_history", [])
                    gh.append(gpu)
                    if len(gh) > 500:
                        s["_gpu_history"] = gh[-500:]
            mtype = payload.get("type")
            if mtype == "metrics" and payload.get("data"):
                await self._broadcast_process(process_id, {"type": "metrics", "data": payload["data"]})
            elif mtype == "log_line" and payload.get("line"):
                line = payload["line"]
                logs = s.setdefault("_log_lines", [])
                logs.append(line)
                if len(logs) > 2000:
                    s["_log_lines"] = logs[-2000:]
                await self._broadcast_process(process_id, {"type": "log_line", "line": line})
            return JSONResponse({"ok": True})

        # ---- 静态文件 ----
        @app.get("/", response_class=HTMLResponse)
        async def index():
            html = (self.static_dir / "index.html").read_text(encoding="utf-8")
            return HTMLResponse(content=html)

        # ---- API: 进程列表 ----
        @app.get("/api/processes")
        async def list_processes():
            with self._lock:
                procs = []
                for pid, s in self._processes.items():
                    procs.append({
                        "process_id": s.get("process_id", pid),
                        "name": s.get("name", pid),
                        "status": s.get("status", "unknown"),
                        "epoch": s.get("epoch"),
                        "max_epoch": s.get("max_epoch"),
                        "latest_metrics": s.get("latest_metrics", {}),
                        "latest_gpu": s.get("latest_gpu", {}),
                        "created_at": s.get("registered_at", 0),
                        "anomaly_count": s.get("anomaly_count", 0),
                        "restart_count": s.get("restart_count", 0),
                        "command": s.get("command", ""),
                    })
            return JSONResponse(content={"processes": procs})

        # ---- API: 进程详情 ----
        def _get_state(pid):
            with self._lock:
                return dict(self._processes.get(pid, {}))

        @app.get("/api/process/{process_id}")
        async def process_detail(process_id: str):
            s = _get_state(process_id)
            if not s:
                return JSONResponse({"error": "not found"}, 404)

            hist = s.get("_metrics_history", [])
            gpu_hist = s.get("_gpu_history", [])
            latest_metrics = hist[-1] if hist else {}
            latest_gpu = gpu_hist[-1] if gpu_hist else {}
            discovered = _discover_metrics(hist)

            return JSONResponse(content={
                "process_id": s.get("process_id", process_id),
                "name": s.get("name", process_id),
                "status": s.get("status", "unknown"),
                "command": s.get("command", ""),
                "epoch": s.get("epoch") or latest_metrics.get("epoch") or latest_metrics.get("step"),
                "max_epoch": s.get("max_epoch"),
                "latest_metrics": latest_metrics,
                "latest_gpu": latest_gpu,
                "anomaly_count": s.get("anomaly_count", 0),
                "restart_count": s.get("restart_count", 0),
                "discovered_metrics": discovered,
                "metrics_count": len(hist),
                "anomalies": s.get("_anomalies", [])[-20:],
                "restarts": s.get("_restarts", []),
            })

        # ---- API: 指标历史（分页+倒查） ----
        @app.get("/api/process/{process_id}/metrics")
        async def metrics_history(
            process_id: str,
            limit: int = Query(200, ge=10, le=2000),
            cursor: int = Query(0, ge=0),
        ):
            s = _get_state(process_id)
            if not s:
                return JSONResponse({"error": "not found"}, 404)
            hist = s.get("_metrics_history", [])
            total = len(hist)
            start = max(0, total - limit - cursor)
            end = total - cursor if cursor else total
            return JSONResponse(content={
                "metrics": hist[max(0, start):end],
                "total": total, "cursor": cursor, "limit": limit,
            })

        # ---- API: 日志 ----
        @app.get("/api/process/{process_id}/log")
        async def tail_log(
            process_id: str,
            lines: int = Query(100, ge=10, le=1000),
            offset: int = Query(0, ge=0),
        ):
            s = _get_state(process_id)
            if not s:
                return JSONResponse({"error": "not found"}, 404)
            # 优先读文件，回退到存储的日志行
            log_path = s.get("log_file")
            all_lines = []
            if log_path and Path(log_path).exists():
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                all_lines = text.splitlines()
            else:
                all_lines = s.get("_log_lines", [])
            total = len(all_lines)
            result = all_lines[-offset-lines:total-offset] if offset else all_lines[-lines:]
            return JSONResponse({"lines": result, "total": total, "offset": offset})

        # ---- API: 模型结构 ----
        @app.get("/api/process/{process_id}/model")
        async def model_structure(process_id: str):
            s = _get_state(process_id)
            if not s:
                return JSONResponse({"error": "not found"}, 404)
            result = s.get("_model_graph")
            if result:
                return JSONResponse(content=result)
            model_fn_ref = s.get("model_entry")
            if not model_fn_ref:
                return JSONResponse({"error": "no model_entry configured"}, 400)
            try:
                from ..model_viz import ModelVisualizer
                # 确保项目目录在 sys.path 中
                proj_dir = s.get("project_dir", "")
                if proj_dir and proj_dir not in sys.path:
                    sys.path.insert(0, proj_dir)
                mod_parts = model_fn_ref.split(":", 1)
                if len(mod_parts) != 2:
                    return JSONResponse({"error": f"invalid model_entry: {model_fn_ref}"}, 400)
                import importlib
                mod = importlib.import_module(mod_parts[0])
                model_fn = getattr(mod, mod_parts[1])
                mv = ModelVisualizer()
                graph = mv.parse_model(model_fn)
                stats = mv.compute_stats(graph)
                result = {**graph, "layer_stats": stats.get("layer_stats", [])}
                with self._lock:
                    self._processes[process_id]["_model_graph"] = result
                return JSONResponse(content=result)
            except Exception as e:
                return JSONResponse({"error": str(e)}, 500)

        # ---- API: 异常事件 ----
        @app.get("/api/process/{process_id}/anomalies")
        async def anomalies(process_id: str):
            s = _get_state(process_id)
            if not s: return JSONResponse({"error": "not found"}, 404)
            return JSONResponse(content={"anomalies": s.get("_anomalies", [])})

        # ---- API: AI 分析 ----
        @app.post("/api/process/{process_id}/ai/analyze")
        async def ai_analyze(process_id: str):
            s = _get_state(process_id)
            if not s: return JSONResponse({"error": "not found"}, 404)
            hist = s.get("_metrics_history", [])
            summary = _summarize_metrics(hist)
            # 尝试自建 advisor
            try:
                from ..credentials import load_credentials, apply_credentials
                apply_credentials(load_credentials())
                from ..agent_advisor import AgentAdvisor
                advisor = AgentAdvisor({"enabled": True, "provider": "anthropic",
                                        "decision_timeout": 15})
                if advisor.is_enabled():
                    ctx = {"status": s.get("status"), "latest_metrics": hist[-1] if hist else {},
                           "metrics_summary": summary, "anomaly_count": s.get("anomaly_count", 0)}
                    text = advisor.narrate({"type": "dashboard_analysis", **ctx})
                    if text:
                        return JSONResponse({"analysis": text, "source": "agent"})
            except Exception:
                pass
            return JSONResponse({
                "analysis": f"训练状态: {s.get('status')}, 最新 loss: {summary.get('loss_last', '?')}, 异常数: {s.get('anomaly_count', 0)}",
                "source": "summary", "context": {"status": s.get("status"), "metrics_summary": summary}
            })

        @app.post("/api/process/{process_id}/ai/chat")
        async def ai_chat(process_id: str, question: str = ""):
            if not question:
                return JSONResponse({"answer": "请输入问题"})
            s = _get_state(process_id)
            if not s: return JSONResponse({"error": "not found"}, 404)
            hist = s.get("_metrics_history", [])
            try:
                from ..credentials import load_credentials, apply_credentials
                apply_credentials(load_credentials())
                from ..agent_advisor import AgentAdvisor
                advisor = AgentAdvisor({"enabled": True, "provider": "anthropic", "decision_timeout": 15})
                if advisor.is_enabled():
                    ctx = {"status": s.get("status"), "question": question,
                           "latest_metrics": hist[-1] if hist else {},
                           "metrics_summary": _summarize_metrics(hist)}
                    ans = advisor.narrate({"type": "chat", "question": question, "context": ctx})
                    if ans:
                        return JSONResponse({"answer": ans})
            except Exception:
                pass
            return JSONResponse({"answer": "AI 调用失败，请检查凭据配置"})

        # ---- API: 图库（最小可用） ----
        @app.get("/api/process/{process_id}/gallery")
        async def gallery_list(process_id: str):
            s = _get_state(process_id)
            gallery_data = s.get("_gallery_results") if s else None
            if not gallery_data:
                return JSONResponse({"galleries": {}, "note": "尚未生成图库，请在 CLI 中运行 gallery 或调用 API"})
            return JSONResponse(content=gallery_data)

        # ---- API: 多进程对比 ----
        @app.post("/api/compare")
        async def compare_processes(payload: dict):
            ids = payload.get("process_ids", [])
            results = []
            for pid in ids[:5]:
                s = _get_state(pid)
                if not s: continue
                monitor = s.get("_monitor")
                hist = monitor.get_metrics_history() if monitor else []
                results.append({
                    "process_id": pid, "name": s.get("name", pid),
                    "status": s.get("status"),
                    "latest_metrics": hist[-1] if hist else {},
                    "anomaly_count": len(monitor.get_anomaly_history() if monitor else []),
                    "restart_count": sum(1 for _ in (s.get("_watchdog").restarts if s.get("_watchdog") else [])),
                    "metrics_summary": _summarize_metrics(hist),
                })
            return JSONResponse({"comparisons": results})

        # ---- WebSocket: 单进程实时流 ----
        @app.websocket("/ws/process/{process_id}")
        async def ws_process(ws: WebSocket, process_id: str):
            await ws.accept()
            with self._lock:
                self._subscribers.setdefault(process_id, []).append(ws)
            try:
                while True:
                    await ws.receive_text()
            except (WebSocketDisconnect, ConnectionResetError, OSError):
                pass
            finally:
                with self._lock:
                    if process_id in self._subscribers:
                        self._subscribers[process_id] = [w for w in self._subscribers[process_id] if w != ws]

        # ---- WebSocket: 全局事件流 ----
        @app.websocket("/ws")
        async def ws_global(ws: WebSocket):
            await ws.accept()
            self._global_subs.append(ws)
            try:
                while True:
                    await ws.receive_text()
            except (WebSocketDisconnect, ConnectionResetError, OSError):
                pass
            finally:
                self._global_subs.remove(ws)

        return app

    # ------------------------------------------------------------------
    # 广播
    # ------------------------------------------------------------------

    async def _broadcast_process(self, process_id, msg):
        for ws in self._subscribers.get(process_id, []):
            try:
                await ws.send_json(msg)
            except Exception:
                pass

    async def _broadcast_global(self, msg):
        for ws in self._global_subs:
            try:
                await ws.send_json(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 启动
    # ------------------------------------------------------------------

    def start(self, blocking: bool = True):
        if not _FASTAPI_OK:
            print("[dashboard] FastAPI/uvicorn 未安装。pip install fastapi uvicorn")
            return None
        if blocking:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
            server = uvicorn.Server(config)
            self._loop.run_until_complete(server.serve())
        else:
            return self._start_in_thread()

    def _start_in_thread(self):
        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="warning")
            server = uvicorn.Server(config)
            self._loop.run_until_complete(server.serve())
        t = threading.Thread(target=_run, daemon=True, name="dashboard")
        t.start()
        return t


# ---------------------------------------------------------------------------
# 指标自动发现
# ---------------------------------------------------------------------------

def _discover_metrics(hist: list) -> dict:
    """扫描指标历史，按类别分组。"""
    if not hist:
        return {"losses": [], "accuracies": [], "learning_rates": [], "gpu": [], "custom": []}
    keys = list(hist[-1].keys())
    result = {"losses": [], "accuracies": [], "learning_rates": [], "gpu": [], "custom": []}
    for k in keys:
        if k in ("step", "epoch", "timestamp"):
            continue
        kl = k.lower()
        if "loss" in kl:
            result["losses"].append(k)
        elif any(w in kl for w in ("acc", "map", "miou", "f1", "precision", "recall")):
            result["accuracies"].append(k)
        elif "lr" in kl:
            result["learning_rates"].append(k)
        elif kl.startswith("gpu"):
            result["gpu"].append(k)
        elif _is_numeric_field(hist, k):
            result["custom"].append(k)
    return result


def _is_numeric_field(hist, key) -> bool:
    for r in hist[-20:]:
        if key in r and r[key] is not None and isinstance(r[key], (int, float)):
            return True
    return False


def _summarize_metrics(hist: list) -> dict:
    """从完整指标历史提取关键摘要。"""
    if not hist:
        return {}
    first = hist[0]
    last = hist[-1]
    summary = {"total_records": len(hist), "first_step": first.get("step"), "last_step": last.get("step")}
    for key in ("loss", "val_loss", "val_acc", "accuracy", "lr"):
        vals = [r[key] for r in hist if key in r and isinstance(r.get(key), (int, float))]
        if vals:
            summary[f"{key}_min"] = min(vals)
            summary[f"{key}_max"] = max(vals)
            summary[f"{key}_last"] = vals[-1]
    return summary
