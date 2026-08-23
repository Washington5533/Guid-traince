"""GPU 设备状态监控。

独立于训练监控的 GPU 轮询模块。采集所有 GPU 的：
- 利用率
- 温度
- 显存使用
- 功耗

支持快照持久化和摘要统计。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

__all__ = ["GpuSnapshot", "GpuSummary", "GpuMonitor"]

logger = logging.getLogger(__name__)


@dataclass
class GpuSnapshot:
    """单次 GPU 快照。"""
    timestamp: float
    gpu_id: int
    name: str = ""
    utilization: int = 0          # GPU 利用率 (%)
    memory_used_mb: int = 0       # 已用显存 (MB)
    memory_total_mb: int = 0      # 总显存 (MB)
    temperature_c: int = 0        # 温度 (°C)
    power_draw_w: float = 0.0     # 功耗 (W)
    power_limit_w: float = 0.0    # 功耗上限 (W)

    @property
    def memory_percent(self) -> float:
        if self.memory_total_mb <= 0:
            return 0.0
        return round(self.memory_used_mb / self.memory_total_mb * 100, 1)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GpuSummary:
    """GPU 状态摘要（一段时间内的统计）。"""
    gpu_id: int
    name: str = ""
    utilization_avg: float = 0.0
    utilization_max: float = 0.0
    utilization_min: float = 0.0
    temperature_avg: float = 0.0
    temperature_max: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    memory_percent: float = 0.0
    power_draw_avg_w: float = 0.0
    snapshot_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class GpuMonitor:
    """GPU 设备状态采集器。

    使用 nvidia-smi 采集 GPU 状态，支持：
    - 定时轮询
    - 快照持久化到 JSONL 文件
    - 摘要统计

    使用示例：
        monitor = GpuMonitor(poll_interval=5)
        monitor.start()  # 后台线程
        snapshots = monitor.poll()  # 手动触发
        summary = monitor.get_summary(gpu_id=0, window_minutes=30)
        monitor.stop()
    """

    def __init__(self, poll_interval: float = 5.0, persist_dir: str | Path | None = None):
        self.poll_interval = poll_interval
        self.persist_dir = Path(persist_dir) if persist_dir else None

        self._history: dict[int, list[GpuSnapshot]] = {}  # {gpu_id: [snapshots]}
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

        self._total_snapshots = 0

    # ── 采集 ─────────────────────────────────────────────────────────

    def poll(self) -> list[GpuSnapshot]:
        """执行一次 GPU 轮询，返回所有 GPU 快照列表。"""
        try:
            output = self._run_nvidia_smi()
            return self._parse_smi_output(output)
        except FileNotFoundError:
            logger.warning("nvidia-smi 未找到，GPU 监控不可用")
            return []
        except Exception as exc:
            logger.warning("GPU 轮询失败: %s", exc)
            return []

    def start(self) -> None:
        """启动后台轮询线程。"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("GpuMonitor 后台轮询已启动 (interval=%.1fs)", self.poll_interval)

    def stop(self) -> None:
        """停止后台轮询。"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("GpuMonitor 已停止")

    def _poll_loop(self) -> None:
        """后台轮询循环。"""
        while self._running:
            try:
                snapshots = self.poll()
                self._persist_snapshots(snapshots)
            except Exception:
                pass
            time.sleep(self.poll_interval)

    # ── 查询 ─────────────────────────────────────────────────────────

    def get_snapshots(self, gpu_id: int, since: float | None = None) -> list[GpuSnapshot]:
        """获取指定 GPU 的快照历史。"""
        with self._lock:
            snaps = self._history.get(gpu_id, [])
            if since:
                snaps = [s for s in snaps if s.timestamp >= since]
            return list(snaps)

    def get_all_snapshots(self, since: float | None = None) -> dict[int, list[GpuSnapshot]]:
        """获取所有 GPU 的快照历史。"""
        with self._lock:
            if since:
                return {gid: [s for s in snaps if s.timestamp >= since]
                        for gid, snaps in self._history.items()}
            return {gid: list(snaps) for gid, snaps in self._history.items()}

    def get_summary(self, gpu_id: int, window_minutes: float = 60.0) -> GpuSummary:
        """获取指定 GPU 的最近 N 分钟状态摘要。"""
        since = time.time() - window_minutes * 60
        snaps = self.get_snapshots(gpu_id, since)
        if not snaps:
            return GpuSummary(gpu_id=gpu_id, snapshot_count=0)

        utils = [s.utilization for s in snaps]
        temps = [s.temperature_c for s in snaps]
        powers = [s.power_draw_w for s in snaps]

        return GpuSummary(
            gpu_id=gpu_id,
            name=snaps[0].name,
            utilization_avg=round(sum(utils) / len(utils), 1),
            utilization_max=max(utils),
            utilization_min=min(utils),
            temperature_avg=round(sum(temps) / len(temps), 1),
            temperature_max=max(temps),
            memory_used_gb=round(snaps[-1].memory_used_mb / 1024, 2),
            memory_total_gb=round(snaps[-1].memory_total_mb / 1024, 2),
            memory_percent=snaps[-1].memory_percent,
            power_draw_avg_w=round(sum(powers) / len(powers), 1),
            snapshot_count=len(snaps),
        )

    def get_all_summaries(self, window_minutes: float = 60.0) -> list[GpuSummary]:
        """获取所有 GPU 的状态摘要。"""
        with self._lock:
            gpu_ids = list(self._history.keys())
        return [self.get_summary(gpu_id, window_minutes) for gpu_id in gpu_ids]

    @property
    def gpu_count(self) -> int:
        with self._lock:
            return len(self._history)

    @property
    def total_snapshots(self) -> int:
        return self._total_snapshots

    # ── 持久化 ────────────────────────────────────────────────────────

    def _persist_snapshots(self, snapshots: list[GpuSnapshot]) -> None:
        """将快照追加到 JSONL 文件。"""
        if not self.persist_dir or not snapshots:
            return
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            path = self.persist_dir / "gpu_snapshots.jsonl"
            with open(path, "a", encoding="utf-8") as f:
                for s in snapshots:
                    f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
            self._total_snapshots += len(snapshots)
        except Exception:
            logger.debug("GPU 快照持久化失败", exc_info=True)

    def load_from_disk(self, path: str | Path | None = None) -> int:
        """从 JSONL 文件加载历史快照。返回加载的记录数。"""
        p = Path(path) if path else (self.persist_dir / "gpu_snapshots.jsonl" if self.persist_dir else None)
        if not p or not p.exists():
            return 0

        count = 0
        with self._lock:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        snap = GpuSnapshot(**data)
                        self._history.setdefault(snap.gpu_id, []).append(snap)
                        count += 1
                    except Exception:
                        continue
        logger.info("从磁盘加载 %d 条 GPU 快照", count)
        return count

    # ── nvidia-smi 解析 ──────────────────────────────────────────────

    @staticmethod
    def _run_nvidia_smi() -> str:
        """执行 nvidia-smi 并返回 CSV 格式输出。"""
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10, check=True)
        return result.stdout

    @staticmethod
    def _parse_smi_output(output: str) -> list[GpuSnapshot]:
        """解析 nvidia-smi CSV 输出。"""
        snapshots = []
        now = time.time()
        for line in output.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 8:
                continue
            try:
                snap = GpuSnapshot(
                    timestamp=now,
                    gpu_id=int(parts[0]),
                    name=parts[1],
                    utilization=int(parts[2]) if parts[2] != "N/A" else 0,
                    memory_used_mb=int(parts[3]) if parts[3] != "N/A" else 0,
                    memory_total_mb=int(parts[4]) if parts[4] != "N/A" else 0,
                    temperature_c=int(parts[5]) if parts[5] != "N/A" else 0,
                    power_draw_w=float(parts[6]) if parts[6] != "N/A" else 0.0,
                    power_limit_w=float(parts[7]) if parts[7] != "N/A" else 0.0,
                )
                snapshots.append(snap)
            except (ValueError, IndexError):
                continue
        return snapshots

    def _store_snapshots(self, snapshots: list[GpuSnapshot]) -> None:
        """存储快照到内存（线程安全）。"""
        with self._lock:
            for snap in snapshots:
                self._history.setdefault(snap.gpu_id, []).append(snap)
                # 限制每个 GPU 的内存快照数
                if len(self._history[snap.gpu_id]) > 10000:
                    self._history[snap.gpu_id] = self._history[snap.gpu_id][-5000:]
            self._total_snapshots += len(snapshots)
