"""日志持久化 + 断线补传。

持久化目录结构：
    logs/
    ├── {session_id}/
    │   ├── meta.json           ← 训练元信息
    │   ├── metrics.jsonl       ← 指标流
    │   ├── events.jsonl        ← 所有事件（SSE 推送 + 持久化）
    │   ├── gpu_snapshots.jsonl ← GPU 快照
    │   ├── decisions.jsonl     ← Sub-agent 决策日志
    │   └── summary.json        ← 训练结束摘要
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

__all__ = ["PersistenceManager"]

logger = logging.getLogger(__name__)


class PersistenceManager:
    """日志持久化管理器。

    负责：
    1. 写入训练元信息、指标、事件到 JSONL 文件
    2. PC 端断线后从文件补拉历史数据
    3. 训练结束生成 summary.json
    """

    def __init__(self, root_dir: str | Path = "./logs"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def session_dir(self, session_id: str) -> Path:
        """返回会话目录（自动创建）。"""
        d = self.root / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ── 写入 ─────────────────────────────────────────────────────────

    def write_meta(self, session_id: str, meta: dict) -> None:
        """写入训练元信息。"""
        path = self.session_dir(session_id) / "meta.json"
        payload = {
            "session_id": session_id,
            "name": meta.get("name", session_id),
            "command": meta.get("command", ""),
            "status": meta.get("status", "running"),
            "registered_at": meta.get("registered_at", time.time()),
            "model_entry": meta.get("model_entry", ""),
            "log_file": meta.get("log_file", ""),
            "project_dir": meta.get("project_dir", ""),
            "total_epochs": meta.get("total_epochs", 0),
            "device_info": meta.get("device_info", {}),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def append_metrics(self, session_id: str, metrics: dict) -> None:
        """追加一条指标到 JSONL。"""
        path = self.session_dir(session_id) / "metrics.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.debug("写入 metrics.jsonl 失败", exc_info=True)

    def append_event(self, session_id: str, event_type: str, data: dict) -> None:
        """追加一个事件到 events.jsonl。"""
        path = self.session_dir(session_id) / "events.jsonl"
        entry = {"timestamp": time.time(), "type": event_type, "data": data}
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.debug("写入 events.jsonl 失败", exc_info=True)

    def append_decision(self, session_id: str, decision: dict) -> None:
        """追加一条决策日志到 decisions.jsonl。"""
        path = self.session_dir(session_id) / "decisions.jsonl"
        entry = {"timestamp": time.time(), **decision}
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.debug("写入 decisions.jsonl 失败", exc_info=True)

    def append_gpu_snapshot(self, session_id: str, snapshot: dict) -> None:
        """追加一条 GPU 快照。"""
        path = self.session_dir(session_id) / "gpu_snapshots.jsonl"
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(snapshot, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logger.debug("写入 gpu_snapshots.jsonl 失败", exc_info=True)

    def write_summary(self, session_id: str, summary: dict) -> None:
        """写入训练结束摘要。"""
        path = self.session_dir(session_id) / "summary.json"
        try:
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            logger.debug("写入 summary.json 失败", exc_info=True)

    # ── 读取 ─────────────────────────────────────────────────────────

    def read_jsonl(self, session_id: str, filename: str,
                   since: float | None = None) -> list[dict]:
        """读取 JSONL 文件，可选按时间过滤。"""
        path = self.session_dir(session_id) / filename
        if not path.exists():
            return []
        records = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if since and rec.get("timestamp", 0) < since:
                            continue
                        records.append(rec)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            logger.debug("读取 %s 失败", filename, exc_info=True)
        return records

    def read_meta(self, session_id: str) -> dict | None:
        """读取训练元信息。"""
        path = self.session_dir(session_id) / "meta.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def read_metrics(self, session_id: str, since: float | None = None) -> list[dict]:
        """读取指标历史。"""
        return self.read_jsonl(session_id, "metrics.jsonl", since)

    def read_events(self, session_id: str, since: float | None = None) -> list[dict]:
        """读取事件历史。"""
        return self.read_jsonl(session_id, "events.jsonl", since)

    def read_decisions(self, session_id: str, since: float | None = None) -> list[dict]:
        """读取决策日志。"""
        return self.read_jsonl(session_id, "decisions.jsonl", since)

    def read_gpu_snapshots(self, session_id: str, since: float | None = None) -> list[dict]:
        """读取 GPU 快照。"""
        return self.read_jsonl(session_id, "gpu_snapshots.jsonl", since)

    def read_summary(self, session_id: str) -> dict | None:
        """读取训练摘要。"""
        path = self.session_dir(session_id) / "summary.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    # ── 会话管理 ─────────────────────────────────────────────────────

    def list_sessions(self) -> list[dict]:
        """列出所有持久化的训练会话。"""
        sessions = []
        if not self.root.is_dir():
            return sessions
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    sessions.append(meta)
                except Exception:
                    continue
        return sessions

    def cleanup_old(self, keep_days: int = 30) -> int:
        """清理超过 N 天的旧会话。返回删除的会话数。"""
        cutoff = time.time() - keep_days * 86400
        deleted = 0
        if not self.root.is_dir():
            return 0
        for d in list(self.root.iterdir()):
            if not d.is_dir():
                continue
            try:
                if d.stat().st_mtime < cutoff:
                    import shutil
                    shutil.rmtree(d)
                    deleted += 1
            except Exception:
                continue
        return deleted
