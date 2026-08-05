"""cp_4 · Checkpoint 发现与分析。

sidecar 下本模块**不负责保存 checkpoint**——保存由训练脚本自己做（契约
规定必需字段）。guardian 只做进程外的三件事：轮询发现新目录、校验、
维护 best/top-k。详见 checkpoint/cp_4.md

关键约束：训练进程可能正在写 checkpoint，guardian 在进程外无法知道写完了
没有，只能靠稳定性判据 + 可加载性判断。且绝不删训练脚本可能正在用的
最近几个 checkpoint。
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_CP_RE = re.compile(r"^cp_(\d+)$")


@dataclass
class CheckpointInfo:
    epoch: int
    path: Path
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"epoch": self.epoch, "path": str(self.path), "metrics": self.metrics}


class CheckpointAnalyzer:
    def __init__(
        self,
        config: dict | None = None,
        ckpt_dir: str | Path = "./checkpoints",
        contract: Any = None,
        notifier: Any = None,
    ):
        self.cfg = config or {}
        self.ckpt_dir = Path(ckpt_dir)
        self.contract = contract
        self.notifier = notifier

        self.save_top_k = int(self.cfg.get("save_top_k", 5))
        self.keep_recent = int(self.cfg.get("keep_recent", 2))
        self.stability_checks = int(self.cfg.get("stability_checks", 2))

        self.required_keys = contract.checkpoint_required_keys() if contract else []
        self.known: dict[int, CheckpointInfo] = {}
        self._fingerprints: dict[int, tuple[tuple[str, int, float], ...]] = {}
        self._stable_counts: dict[int, int] = {}

    # --- 发现 -------------------------------------------------------

    def _candidates(self) -> list[tuple[int, Path]]:
        if not self.ckpt_dir.exists():
            return []
        out = []
        for child in self.ckpt_dir.iterdir():
            if not child.is_dir():
                continue
            m = _CP_RE.match(child.name)
            if m:
                out.append((int(m.group(1)), child))
        out.sort(key=lambda p: p[0])
        return out

    @staticmethod
    def _fingerprint(path: Path) -> tuple[tuple[str, int, float], ...]:
        items = []
        for f in sorted(path.iterdir()):
            if f.is_file():
                st = f.stat()
                items.append((f.name, st.st_size, st.st_mtime))
        return tuple(items)

    def _is_written(self, epoch: int, path: Path) -> bool:
        """判定"写完了"：连续 N 次轮询 size/mtime 不变 + 含必需键。

        原子写（tmp + rename）的目录首次就稳定，不用多等。
        """
        try:
            fp = self._fingerprint(path)
        except OSError:
            return False
        if not fp:
            return False
        if self._fingerprints.get(epoch) == fp:
            self._stable_counts[epoch] = self._stable_counts.get(epoch, 1) + 1
        else:
            self._fingerprints[epoch] = fp
            self._stable_counts[epoch] = 1
        if self._stable_counts[epoch] < self.stability_checks:
            return False
        return self._has_required_keys(path)

    def _has_required_keys(self, path: Path) -> bool:
        if not self.required_keys:
            return True
        weights = [p for p in path.glob("*.pth") if p.stat().st_size > 0]
        if not weights:
            return False
        try:
            import torch
        except ImportError:  # pragma: no cover
            return True
        for cand in weights:
            try:
                obj = torch.load(cand, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if isinstance(obj, dict) and all(k in obj for k in self.required_keys):
                return True
        return False

    def poll(self) -> list[CheckpointInfo]:
        """扫描目录，返回本轮新发现且已写完的 checkpoint。"""
        found: list[CheckpointInfo] = []
        for epoch, path in self._candidates():
            if epoch in self.known:
                continue
            if not self._is_written(epoch, path):
                continue
            info = CheckpointInfo(epoch, path, _read_metrics(path))
            self.known[epoch] = info
            found.append(info)
        return found

    # --- best / 清理 ------------------------------------------------

    def best(self, metric: str = "val/accuracy", higher_better: bool = True) -> CheckpointInfo | None:
        scored = [
            (info.metrics.get(metric), info)
            for info in self.known.values()
            if isinstance(info.metrics.get(metric), (int, float))
        ]
        if not scored:
            return None
        return (max if higher_better else min)(scored, key=lambda p: p[0])[1]

    def cleanup(self, metric: str = "val/accuracy", higher_better: bool = True) -> list[int]:
        """保留 top-k，但**无条件保留最近 keep_recent 个** epoch。

        guardian 在进程外不知道训练脚本下次 resume 会读哪个文件，
        删掉最近的 checkpoint 可能让恢复失败。
        """
        if self.save_top_k <= 0 or not self.known:
            return []
        epochs = sorted(self.known)
        protected = set(epochs[-self.keep_recent:]) if self.keep_recent > 0 else set()

        scored = [
            (info.metrics.get(metric), ep)
            for ep, info in self.known.items()
            if isinstance(info.metrics.get(metric), (int, float))
        ]
        if scored:
            scored.sort(key=lambda p: p[0], reverse=higher_better)
            keep = {ep for _, ep in scored[: self.save_top_k]}
        else:
            keep = set(epochs[-self.save_top_k:])
        keep |= protected

        removed: list[int] = []
        for ep in epochs:
            if ep in keep:
                continue
            try:
                shutil.rmtree(self.known[ep].path)
            except OSError:
                continue
            removed.append(ep)
            self.known.pop(ep, None)
        return removed

    # --- 报告 -------------------------------------------------------

    def report(self, metric: str = "", higher_better: bool | None = None) -> dict[str, Any]:
        """生成 checkpoint 分析报告。

        未指定 metric 时，尝试从 contract.select_metric() 获取最优判定指标；
        仍未取得则回退 val/accuracy。
        """
        metric_source: dict[str, Any] = {"name": metric or "val/accuracy",
                                          "direction": "max", "source": "hardcoded"}

        if (not metric or higher_better is None) and self.contract is not None:
            try:
                # 从已知 checkpoint 的 metrics 键名中收集线索
                seen_keys: list[str] = []
                for info in self.known.values():
                    seen_keys.extend(info.metrics.keys())
                selected = self.contract.select_metric({
                    "metrics_seen": list(set(seen_keys)),
                })
                if not metric:
                    metric = selected.get("metric", "val/accuracy")
                if higher_better is None:
                    higher_better = selected.get("direction", "max") == "max"
                metric_source = {
                    "name": metric,
                    "direction": "max" if higher_better else "min",
                    "source": selected.get("source", "fallback"),
                }
                if selected.get("task_type"):
                    metric_source["task_type"] = selected["task_type"]
            except Exception:
                if not metric:
                    metric = "val/accuracy"
                if higher_better is None:
                    higher_better = True

        if not metric:
            metric = "val/accuracy"
        if higher_better is None:
            higher_better = True

        best = self.best(metric, higher_better)
        epochs = sorted(self.known)
        return {
            "total": len(self.known),
            "latest": f"cp_{epochs[-1]}" if epochs else None,
            "best": best.to_dict() if best else None,
            "metric": metric,
            "metric_source": metric_source,
            "checkpoints": [self.known[e].to_dict() for e in epochs],
        }


def _read_metrics(path: Path) -> dict[str, Any]:
    mpath = path / "metrics.json"
    if not mpath.exists():
        return {}
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (ValueError, OSError):
        return {}
