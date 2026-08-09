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
import uuid
from pathlib import Path

from guardian.logging_config import get_logger

logger = get_logger(__name__)
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
        # 历史进程（从磁盘加载）：{process_id: {meta, metrics_path}}
        self._history: dict[str, dict] = {}
        # WebSocket 连接：{process_id: [ws, ...]}
        self._subscribers: dict[str, list[WebSocket]] = {}
        # 全局订阅
        self._global_subs: list[WebSocket] = []
        self._lock = threading.Lock()
        # 持久化根目录（从配置读取，默认 ./logs）
        self._persist_root = Path(self.cfg.get("project", {}).get("log_dir", "./logs"))
        self._load_history()

        self._start_time = time.monotonic()
        self.app = self._build_app() if _FASTAPI_OK else None

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    def _persist_meta(self, process_id: str, state: dict) -> None:
        """写入/更新进程元信息到 logs/{pid}/meta.json。"""
        try:
            d = self._persist_root / process_id
            d.mkdir(parents=True, exist_ok=True)
            meta = {
                "process_id": process_id,
                "name": state.get("name", process_id),
                "command": state.get("command", ""),
                "project_dir": state.get("project_dir", ""),
                "status": state.get("status", "unknown"),
                "registered_at": state.get("registered_at", 0),
                "finished_at": state.get("finished_at"),
                "config": state.get("config", {}),
                "model_entry": state.get("model_entry", ""),
                "log_file": state.get("log_file", ""),
            }
            (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.warning("写入 meta.json 失败: %s", d, exc_info=True)

    def _persist_metrics_line(self, process_id: str, data: dict) -> None:
        """追加一条指标到 logs/{pid}/metrics.jsonl。"""
        try:
            d = self._persist_root / process_id
            d.mkdir(parents=True, exist_ok=True)
            with (d / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
        except Exception:
            logger.warning("写入 metrics.jsonl 失败: %s", d, exc_info=True)

    def _load_history(self) -> None:
        """启动时扫描 logs/ 目录，加载含 meta.json 的子目录作为历史进程。"""
        root = self._persist_root
        if not root.is_dir():
            return
        for d in root.iterdir():
            if not d.is_dir():
                continue
            meta_file = d / "meta.json"
            metrics_file = d / "metrics.jsonl"
            if meta_file.is_file():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    pid = meta.get("process_id", d.name)
                    self._history[pid] = {
                        "meta": meta,
                        "metrics_path": str(metrics_file) if metrics_file.is_file() else None,
                        "dir": str(d),
                    }
                except Exception:
                    logger.warning("加载历史进程元信息失败: %s", d, exc_info=True)

    def _read_history_metrics(self, pid: str) -> list:
        """从 JSONL 文件读取历史指标数据。"""
        info = self._history.get(pid)
        if not info or not info.get("metrics_path"):
            return []
        try:
            lines = []
            with open(info["metrics_path"], "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        lines.append(json.loads(line))
            return lines
        except Exception:
            logger.warning("Failed to read history metrics: %s", info.get("metrics_path"), exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Summary lookup
    # ------------------------------------------------------------------

    def _find_summary_for_pid(self, process_id: str) -> Path | None:
        """Find the summary JSON file matching a process_id."""
        root = self._persist_root
        if not root.is_dir():
            return None
        for f in sorted(root.glob("summary_*.json"), reverse=True):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if data.get("experiment_id") == process_id:
                    return f
            except Exception:
                continue
        summaries = sorted(root.glob("summary_*.json"), reverse=True)
        if summaries:
            return summaries[0]
        return None

    # ------------------------------------------------------------------
    # External import
    # ------------------------------------------------------------------

    def import_process(self, meta: dict, metrics: list[dict]) -> dict:
        """导入外部训练数据。

        Args:
            meta: 元信息，必须含 name 字段
            metrics: 指标列表，每条为 dict，至少含一个数值字段

        Returns:
            {"process_id": "import_xxx", "records": N, "status": "imported"}
            或 {"error": "...", "detail": "..."}
        """
        # 校验 meta
        if not isinstance(meta, dict) or not meta.get("name"):
            return {"error": "meta 必须含 name 字段", "detail": "meta 必须含 name（字符串）"}

        # 校验 metrics
        if not isinstance(metrics, list) or len(metrics) == 0:
            return {"error": "metrics 不能为空", "detail": "metrics 必须为非空列表"}
        if len(metrics) > 100000:
            return {"error": "指标数量超限", "detail": f"单次上限 100000 条，当前 {len(metrics)} 条"}
        for i, m in enumerate(metrics):
            if not isinstance(m, dict):
                return {"error": f"metrics[{i}] 格式错误", "detail": "每条必须为 dict"}
            if not any(isinstance(v, (int, float)) for v in m.values()):
                return {"error": f"metrics[{i}] 无数值", "detail": "每条至少含一个数值字段"}

        # 生成唯一 process_id
        process_id = f"import_{uuid.uuid4().hex[:8]}"
        state = {
            "process_id": process_id,
            "name": meta["name"],
            "command": meta.get("command", ""),
            "project_dir": meta.get("project_dir", ""),
            "status": "imported",
            "registered_at": time.time(),
            "finished_at": time.time(),
            "config": {"source": meta.get("source", "external")},
            "model_entry": "",
            "log_file": "",
        }

        # 持久化
        self._persist_meta(process_id, state)
        for m in metrics:
            self._persist_metrics_line(process_id, m)

        # 刷新历史
        self._load_history()

        return {"process_id": process_id, "records": len(metrics), "status": "imported"}

    # ------------------------------------------------------------------
    # 进程注册
    # ------------------------------------------------------------------

    def register_process(self, process_id: str, state: dict) -> None:
        """注册或更新一个被守护的训练进程。"""
        state.setdefault("process_id", process_id)
        state.setdefault("registered_at", time.time())
        with self._lock:
            self._processes[process_id] = state
        self._persist_meta(process_id, state)
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
            self._persist_meta(pid, self._processes[pid])
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
                # 累积 metrics 历史 + 持久化
                mdata = payload.get("data")
                if mdata:
                    hist = s.setdefault("_metrics_history", [])
                    hist.append(mdata)
                    if len(hist) > 2000:
                        s["_metrics_history"] = hist[-2000:]
                    s["latest_metrics"] = mdata
                    self._persist_metrics_line(process_id, mdata)
                # 累积 GPU 历史
                gpu = payload.get("gpu")
                if gpu:
                    gh = s.setdefault("_gpu_history", [])
                    gh.append(gpu)
                    if len(gh) > 500:
                        s["_gpu_history"] = gh[-500:]
                # 状态变更时更新 meta
                if "status" in patch:
                    self._persist_meta(process_id, s)
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

        # ---- 健康检查 ----
        @app.get("/health")
        async def health():
            import time as _time
            live = len(self._processes)
            hist = len(self._history)
            return JSONResponse({
                "status": "ok",
                "version": "0.2.0",
                "live_processes": live,
                "history_processes": hist,
                "uptime_seconds": round(_time.monotonic() - getattr(self, "_start_time", _time.monotonic()), 0),
            })

        # ---- 静态文件 ----
        @app.get("/", response_class=HTMLResponse)
        async def index():
            html = (self.static_dir / "index.html").read_text(encoding="utf-8")
            return HTMLResponse(content=html)

        # ---- API: 进程列表（含历史） ----
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
                        "source": "live",
                    })
            # 追加历史进程（去重：已在 live 中的不重复）
            live_ids = {p["process_id"] for p in procs}
            for pid, info in self._history.items():
                if pid in live_ids:
                    continue
                meta = info.get("meta", {})
                procs.append({
                    "process_id": pid,
                    "name": meta.get("name", pid),
                    "status": meta.get("status", "unknown"),
                    "command": meta.get("command", ""),
                    "created_at": meta.get("registered_at", 0),
                    "finished_at": meta.get("finished_at"),
                    "source": "history",
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
                for ep in s.get("extra_paths", []):
                    if ep not in sys.path:
                        sys.path.insert(0, ep)
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

        # ---- API: 生成模型架构图 HTML ----
        @app.post("/api/process/{process_id}/model/viz")
        async def model_viz_generate(process_id: str):
            s = _get_state(process_id)
            if not s:
                return JSONResponse({"error": "not found"}, 404)
            graph = s.get("_model_graph")
            if not graph:
                return JSONResponse({"error": "请先加载模型结构（点击模型结构 Tab）"}, 400)
            try:
                from ..model_viz import ModelVisualizer, _default_viz_config
                mv = ModelVisualizer()
                stats = mv.compute_stats(graph)
                viz_config = _default_viz_config(graph, stats)
                viz_dir = self._persist_root / "viz"
                viz_dir.mkdir(parents=True, exist_ok=True)
                safe_name = process_id.replace("/", "_").replace("\\", "_")
                out_path = viz_dir / f"model_viz_{safe_name}.html"
                mv.render_html(graph, stats, viz_config, out_path)
                url = f"/api/viz/{out_path.name}"
                return JSONResponse({"url": url, "path": str(out_path)})
            except ModuleNotFoundError as e:
                return JSONResponse({
                    "error": f"缺少依赖: {e}",
                    "detail": "模型代码需要原始训练环境的全部依赖。guarftrain 作为轻量监控工具，"
                              "不预装所有深度学习框架。请手动安装缺失的包，或使用 CLI 生成: "
                              "python run.py visualize --model train:build_model"
                }, status_code=503)
            except Exception as e:
                return JSONResponse({"error": str(e)}, 500)

        # ---- 模型架构图 HTML 文件服务 ----
        @app.get("/api/viz/{filename}")
        async def serve_viz_html(filename: str):
            viz_path = self._persist_root / "viz" / filename
            if not viz_path.exists() or not viz_path.is_file():
                return JSONResponse({"error": "file not found"}, 404)
            html = viz_path.read_text(encoding="utf-8")
            return HTMLResponse(content=html)

        # ---- API: 模型架构图内嵌 HTML ----
        @app.get("/api/process/{process_id}/model/viz-html")
        async def model_viz_inline_html(process_id: str):
            """返回生成的架构图 HTML 内容（供 iframe srcdoc 嵌入）。"""
            s = _get_state(process_id)
            if not s:
                return JSONResponse({"error": "not found"}, 404)
            graph = s.get("_model_graph")
            if not graph:
                return JSONResponse({"error": "请先加载模型结构"}, 400)
            try:
                from ..model_viz import ModelVisualizer, _default_viz_config
                mv = ModelVisualizer()
                stats = mv.compute_stats(graph)
                viz_config = _default_viz_config(graph, stats)
                viz_dir = self._persist_root / "viz"
                viz_dir.mkdir(parents=True, exist_ok=True)
                safe_name = process_id.replace("/", "_").replace("\\", "_")
                out_path = viz_dir / f"model_viz_{safe_name}.html"
                mv.render_html(graph, stats, viz_config, out_path)
                html_content = out_path.read_text(encoding="utf-8")
                return JSONResponse({"html": html_content})
            except Exception as e:
                return JSONResponse({"error": str(e)}, 500)

        # ---- API: 历史进程列表 ----
        @app.get("/api/history")
        async def list_history():
            items = []
            for pid, info in self._history.items():
                meta = info.get("meta", {})
                # 读取最后一条指标作为摘要
                last_metrics = {}
                hist = self._read_history_metrics(pid)
                if hist:
                    last_metrics = hist[-1]
                items.append({
                    "process_id": pid,
                    "name": meta.get("name", pid),
                    "command": meta.get("command", ""),
                    "status": meta.get("status", "unknown"),
                    "registered_at": meta.get("registered_at", 0),
                    "finished_at": meta.get("finished_at"),
                    "metrics_count": len(hist),
                    "latest_metrics": last_metrics,
                })
            return JSONResponse({"history": items})

        # ---- API: 历史进程详情 ----
        @app.get("/api/history/{process_id}")
        async def history_detail(process_id: str):
            info = self._history.get(process_id)
            if not info:
                return JSONResponse({"error": "not found"}, 404)
            meta = info.get("meta", {})
            hist = self._read_history_metrics(process_id)

            # 尝试加载完整 summary JSON（v2 格式，含异常/重启/资源/checkpoint）
            summary_path = self._find_summary_for_pid(process_id)
            summary_data = {}
            if summary_path:
                try:
                    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            anomalies = summary_data.get("anomaly_events") or []
            restarts = summary_data.get("restarts") or []
            resources = summary_data.get("resources") or {}
            checkpoints = summary_data.get("checkpoints") or {}
            lr_schedule = summary_data.get("lr_schedule") or []
            # GPU 数据从 summary 的 resources 或指标历史推断
            latest_gpu = {}
            if resources:
                latest_gpu = {
                    "util_pct": resources.get("gpu_util_avg"),
                    "mem_used_mb": resources.get("gpu_mem_peak_mb"),
                    "mem_total_mb": None,  # summary 不存显存总量
                }

            discovered = _discover_metrics(hist)

            return JSONResponse({
                "process_id": process_id,
                "name": meta.get("name", process_id),
                "status": meta.get("status", "unknown"),
                "command": meta.get("command", ""),
                "registered_at": meta.get("registered_at", 0),
                "finished_at": meta.get("finished_at"),
                "latest_metrics": hist[-1] if hist else {},
                "latest_gpu": latest_gpu,
                "discovered_metrics": discovered,
                "metrics_count": len(hist),
                "anomaly_count": len(anomalies),
                "restart_count": len(restarts),
                "anomalies": anomalies[-20:],     # 最近 20 条
                "restarts": restarts,
                "resources": resources,
                "checkpoints": checkpoints,
                "lr_schedule": lr_schedule,
                "source": "history",
            })

        # ---- API: 历史指标数据 ----
        @app.get("/api/history/{process_id}/metrics")
        async def history_metrics(
            process_id: str,
            limit: int = Query(500, ge=10, le=5000),
            cursor: int = Query(0, ge=0),
        ):
            info = self._history.get(process_id)
            if not info:
                return JSONResponse({"error": "not found"}, 404)
            hist = self._read_history_metrics(process_id)
            total = len(hist)
            start = max(0, total - limit - cursor)
            end = total - cursor if cursor else total
            return JSONResponse({
                "metrics": hist[max(0, start):end],
                "total": total, "cursor": cursor, "limit": limit,
            })

        # ---- API: 历史进程日志 ----
        @app.get("/api/history/{process_id}/log")
        async def history_log(
            process_id: str,
            lines: int = Query(100, ge=10, le=1000),
            offset: int = Query(0, ge=0),
            grep: str | None = None,
        ):
            info = self._history.get(process_id)
            if not info:
                return JSONResponse({"error": "not found"}, 404)
            meta = info.get("meta", {})
            # 尝试找到训练日志文件
            log_path = meta.get("log_file")
            if not log_path or not Path(log_path).is_file():
                # 回退：扫描常见位置
                for candidate in [
                    self._persist_root / "train.log",
                    self._persist_root.parent / "logs" / "train.log",
                    self._persist_root.parent / "train.log",
                ]:
                    p = Path(candidate).resolve()
                    if p.is_file():
                        log_path = str(p)
                        break
            if not log_path or not Path(log_path).is_file():
                return JSONResponse({"lines": [], "total": 0, "error": "未找到训练日志文件"}, status_code=404)
            try:
                text = Path(log_path).read_text(encoding="utf-8", errors="replace")
                all_lines = text.splitlines()
                if grep:
                    all_lines = [l for l in all_lines if grep.lower() in l.lower()]
                total = len(all_lines)
                start = max(0, total - lines - offset)
                end = total - offset if offset else total
                return JSONResponse({
                    "lines": all_lines[start:end],
                    "total": total,
                    "offset": offset,
                    "log_file": log_path,
                })
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

        # ---- API: 历史进程 AI 分析 ----
        @app.post("/api/history/{process_id}/ai/analyze")
        async def history_ai_analyze(process_id: str):
            info = self._history.get(process_id)
            if not info:
                return JSONResponse({"error": "not found"}, 404)
            meta = info.get("meta", {})
            hist = self._read_history_metrics(process_id)
            summary = _summarize_metrics(hist)
            try:
                from ..credentials import load_credentials, apply_credentials
                apply_credentials(load_credentials())
                from ..agent_advisor import AgentAdvisor
                import os as _os
                _provider = _os.environ.get("GUARDIAN_AI_PROVIDER", "anthropic")
                _model = _os.environ.get("GUARDIAN_AI_MODEL") or None
                _cfg = {"enabled": True, "provider": _provider, "decision_timeout": 15}
                if _model:
                    _cfg["model"] = _model
                advisor = AgentAdvisor(_cfg)
                if advisor.is_enabled():
                    ctx = {"status": meta.get("status"), "latest_metrics": hist[-1] if hist else {},
                           "metrics_summary": summary, "anomaly_count": 0,
                           "process_name": meta.get("name", process_id)}
                    text = advisor.narrate({"type": "dashboard_analysis", **ctx})
                    if text:
                        return JSONResponse({"analysis": text, "source": "agent"})
            except Exception:
                logger.warning("历史进程 AI 分析失败: %s", process_id, exc_info=True)
            return JSONResponse({
                "analysis": f"历史实验 {meta.get('name', process_id)}: 共 {len(hist)} 条指标, 最终 loss: {summary.get('loss_last', '?')}",
                "source": "summary", "context": {"metrics_summary": summary}
            })

        # ---- API: 历史进程 AI 对话 ----
        @app.post("/api/history/{process_id}/ai/chat")
        async def history_ai_chat(process_id: str, question: str = ""):
            if not question:
                return JSONResponse({"answer": "请输入问题"}, status_code=400)
            info = self._history.get(process_id)
            if not info:
                return JSONResponse({"error": "not found"}, 404)
            meta = info.get("meta", {})
            hist = self._read_history_metrics(process_id)
            try:
                from ..credentials import load_credentials, apply_credentials
                apply_credentials(load_credentials())
                from ..agent_advisor import AgentAdvisor
                import os as _os
                _provider = _os.environ.get("GUARDIAN_AI_PROVIDER", "anthropic")
                _model = _os.environ.get("GUARDIAN_AI_MODEL") or None
                _cfg = {"enabled": True, "provider": _provider, "decision_timeout": 15}
                if _model:
                    _cfg["model"] = _model
                advisor = AgentAdvisor(_cfg)
                if advisor.is_enabled():
                    ctx = {"status": meta.get("status"), "question": question,
                           "latest_metrics": hist[-1] if hist else {},
                           "metrics_summary": _summarize_metrics(hist),
                           "process_name": meta.get("name", process_id)}
                    ans = advisor.narrate({"type": "chat", "question": question, "context": ctx})
                    if ans:
                        return JSONResponse({"answer": ans})
            except Exception:
                logger.warning("History AI chat failed: %s", process_id, exc_info=True)
            return JSONResponse({"answer": "AI 调用失败，请检查凭据配置"})

        # ---- API: 历史进程模型结构 ----
        @app.get("/api/history/{process_id}/model")
        async def history_model_structure(process_id: str):
            info = self._history.get(process_id)
            if not info:
                return JSONResponse({"error": "not found"}, 404)
            meta = info.get("meta", {})
            model_entry = meta.get("model_entry", "")
            if not model_entry:
                return JSONResponse({"error": "该历史进程未配置 model_entry"}, 400)
            try:
                from ..model_viz import ModelVisualizer
                proj_dir = meta.get("project_dir", "")
                if proj_dir and proj_dir not in sys.path:
                    sys.path.insert(0, proj_dir)
                mod_parts = model_entry.split(":", 1)
                if len(mod_parts) != 2:
                    return JSONResponse({"error": f"invalid model_entry: {model_entry}"}, 400)
                import importlib
                mod = importlib.import_module(mod_parts[0])
                model_fn = getattr(mod, mod_parts[1])
                mv = ModelVisualizer()
                graph = mv.parse_model(model_fn)
                stats = mv.compute_stats(graph)
                return JSONResponse({**graph, "layer_stats": stats.get("layer_stats", [])})
            except Exception as e:
                return JSONResponse({"error": str(e)}, 500)

        # ---- API: 历史进程模型架构图 HTML ----
        @app.get("/api/history/{process_id}/model/viz-html")
        async def history_model_viz_html(process_id: str):
            info = self._history.get(process_id)
            if not info:
                return JSONResponse({"error": "not found"}, 404)
            meta = info.get("meta", {})
            model_entry = meta.get("model_entry", "")
            if not model_entry:
                return JSONResponse({"error": "No model_entry configured"}, 400)
            try:
                from ..model_viz import ModelVisualizer, _default_viz_config
                proj_dir = meta.get("project_dir", "")
                if proj_dir and proj_dir not in sys.path:
                    sys.path.insert(0, proj_dir)
                mod_parts = model_entry.split(":", 1)
                if len(mod_parts) != 2:
                    return JSONResponse({"error": f"invalid model_entry: {model_entry}"}, 400)
                import importlib
                mod = importlib.import_module(mod_parts[0])
                model_fn = getattr(mod, mod_parts[1])
                mv = ModelVisualizer()
                graph = mv.parse_model(model_fn)
                stats = mv.compute_stats(graph)
                viz_config = _default_viz_config(graph, stats)
                viz_dir = self._persist_root / "viz"
                viz_dir.mkdir(parents=True, exist_ok=True)
                safe_name = process_id.replace("/", "_").replace("\\", "_")
                out_path = viz_dir / f"model_viz_hist_{safe_name}.html"
                mv.render_html(graph, stats, viz_config, out_path)
                return JSONResponse({"html": out_path.read_text(encoding="utf-8")})
            except Exception as e:
                return JSONResponse({"error": str(e)}, 500)

        # ---- API: 历史进程图库 ----
        @app.get("/api/history/{process_id}/gallery")
        async def history_gallery(process_id: str):
            info = self._history.get(process_id)
            if not info:
                return JSONResponse({"error": "not found"}, 404)
            # 尝试从日志目录加载图库结果
            pid_dir = self._persist_root / process_id
            gallery_file = pid_dir / "gallery_results.json"
            if gallery_file.is_file():
                try:
                    data = json.loads(gallery_file.read_text(encoding="utf-8"))
                    return JSONResponse(content=data)
                except Exception:
                    pass
            return JSONResponse({"galleries": {}, "note": "尚未生成图库，请在 CLI 中运行 gallery 命令"})

        # ---- API: MCP 自定义指标推送 ----
        @app.post("/api/process/{process_id}/metrics/custom")
        async def push_custom_metrics(process_id: str, payload: dict):
            """接收任意 key-value 指标，支持 MCP 工具推送自定义指标。"""
            s = _get_state(process_id)
            if not s:
                return JSONResponse({"error": "not found"}, 404)
            data = payload.get("data", {})
            group = payload.get("group", "custom")
            # 将 group 信息写入数据中以便前端分组
            data["_group"] = group
            data["_ts"] = time.time()
            with self._lock:
                if process_id in self._processes:
                    hist = self._processes[process_id].setdefault("_metrics_history", [])
                    # 合并到最新一条或新建
                    if hist and not data.get("step"):
                        last = hist[-1]
                        merged = {**last, **data}
                        hist[-1] = merged
                    else:
                        hist.append(data)
                    self._processes[process_id]["latest_metrics"] = hist[-1]
            self._persist_metrics_line(process_id, data)
            await self._broadcast_process(process_id, {"type": "metrics", "data": data})
            return JSONResponse({"ok": True, "group": group})

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
                import os as _os
                _provider = _os.environ.get("GUARDIAN_AI_PROVIDER", "anthropic")
                _model = _os.environ.get("GUARDIAN_AI_MODEL") or None
                _cfg = {"enabled": True, "provider": _provider, "decision_timeout": 15}
                if _model:
                    _cfg["model"] = _model
                advisor = AgentAdvisor(_cfg)
                if advisor.is_enabled():
                    ctx = {"status": s.get("status"), "latest_metrics": hist[-1] if hist else {},
                           "metrics_summary": summary, "anomaly_count": s.get("anomaly_count", 0)}
                    text = advisor.narrate({"type": "dashboard_analysis", **ctx})
                    if text:
                        return JSONResponse({"analysis": text, "source": "agent"})
            except Exception:
                logger.warning("AI 分析调用失败", exc_info=True)
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
                import os as _os
                _provider = _os.environ.get("GUARDIAN_AI_PROVIDER", "anthropic")
                _model = _os.environ.get("GUARDIAN_AI_MODEL") or None
                _cfg = {"enabled": True, "provider": _provider, "decision_timeout": 15}
                if _model:
                    _cfg["model"] = _model
                advisor = AgentAdvisor(_cfg)
                if advisor.is_enabled():
                    ctx = {"status": s.get("status"), "question": question,
                           "latest_metrics": hist[-1] if hist else {},
                           "metrics_summary": _summarize_metrics(hist)}
                    ans = advisor.narrate({"type": "chat", "question": question, "context": ctx})
                    if ans:
                        return JSONResponse({"answer": ans})
            except Exception:
                logger.warning("AI 对话调用失败", exc_info=True)
            return JSONResponse({"answer": "AI 调用失败，请检查凭据配置"})

        # ---- API: 图库（最小可用） ----
        @app.get("/api/process/{process_id}/gallery")
        async def gallery_list(process_id: str):
            s = _get_state(process_id)
            gallery_data = s.get("_gallery_results") if s else None
            if not gallery_data:
                return JSONResponse({"galleries": {}, "note": "尚未生成图库，请在 CLI 中运行 gallery 或点击下方按钮生成"})
            return JSONResponse(content=gallery_data)

        @app.post("/api/process/{process_id}/gallery/generate")
        async def gallery_generate(process_id: str, payload: dict | None = None):
            """训练完成后在网页端一键生成图库。"""
            s = _get_state(process_id)
            if not s:
                return JSONResponse({"error": "not found"}, 404)

            # 找最佳 checkpoint epoch
            ckpt_epoch = (payload or {}).get("ckpt_epoch")
            if not ckpt_epoch:
                # 从进程状态推断
                ckpt_epoch = s.get("epoch")
            if not ckpt_epoch:
                return JSONResponse({"error": "无法确定 checkpoint epoch，请先完成至少 1 个 epoch 的训练"}, 400)

            # 确定数据源
            data_source = (payload or {}).get("data_source")
            if not data_source:
                data_source = s.get("project_dir", "")
                if data_source:
                    data_source = str(Path(data_source) / "data")
                else:
                    data_source = "./data"

            try:
                from ..gallery import GalleryManager
                from ..inference import InferenceRunner

                gm = GalleryManager(advisor=getattr(s.get("_advisor"), None, None))
                ir = InferenceRunner()

                # 确定 checkpoint 路径
                ckpt_dir = self.cfg.get("project", {}).get("ckpt_dir", "./checkpoints")
                ckpt_path = Path(ckpt_dir) / f"cp_{ckpt_epoch}" / "model.pth"
                if not ckpt_path.exists():
                    # 尝试从进程信息获取
                    extra = s.get("extra_paths", [])
                    for ep in extra:
                        ckpt_path = Path(ep) / f"cp_{ckpt_epoch}" / "model.pth"
                        if ckpt_path.exists():
                            break
                    else:
                        return JSONResponse(
                            {"error": f"checkpoint 不存在: {ckpt_path}"}, 400)

                # 判断任务类型并提议策略
                task_type = gm.infer_task_type()
                strategies = gm.propose_strategies(task_type)

                # 执行推理 + 筛选
                results = gm.execute(str(ckpt_path), strategies, data_source, inference_runner=ir)

                if "error" in results:
                    return JSONResponse(results, 500)

                # 缓存到进程状态
                with self._lock:
                    if process_id in self._processes:
                        self._processes[process_id]["_gallery_results"] = results

                # 保存到磁盘
                out_dir = self._persist_root / process_id
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "gallery_results.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

                return JSONResponse({
                    "status": "completed",
                    "galleries": {name: len(imgs) for name, imgs in results.items()},
                })
            except ModuleNotFoundError as e:
                return JSONResponse({
                    "error": f"缺少依赖: {e}",
                    "detail": "图库生成需要 torch/torchvision。请在训练环境中安装。"
                }, status_code=503)
            except Exception as e:
                logger.error("Gallery generation failed: %s", e, exc_info=True)
                return JSONResponse({"error": str(e)}, 500)

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
