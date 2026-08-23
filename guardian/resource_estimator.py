"""cp_1 · 资源预估 (ResourceEstimator)。

训练启动前测量显存占用，推荐安全 batch size，预估训练时长。
以独立进程运行——探测性前向+反向的副作用（BatchNorm running stats、
optimizer 动量、RNG 状态）随进程退出一起消失，训练进程从干净状态启动。

依赖 cp_11 的 buildable_entry 契约项 import model_fn / dataloader_fn；
该契约项缺失时明确报错，不静默跳过。详见 checkpoint/cp_1.md
"""

from __future__ import annotations

import math
import time
from typing import Any

from guardian.logging_config import get_logger

logger = get_logger(__name__)


class ResourceEstimator:
    """训练前资源预估。"""

    def __init__(self, config: dict | None = None):
        self.cfg = config or {}
        self.test_batch_sizes: list[int] = list(
            self.cfg.get("test_batch_sizes", [1, 2, 4])
        )
        self.memory_margin = float(self.cfg.get("memory_margin", 0.2))

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def preflight_check(
        self,
        model_fn,
        dataloader_fn,
        device: str = "cuda",
        total_samples: int | None = None,
        epochs: int | None = None,
        target_batch_size: int | None = None,
        batch_upper_bound: int | None = None,
    ) -> dict[str, Any]:
        """执行完整预检，返回 report dict。

        batch_upper_bound: cp_11 adjustable_paths 里 batch_size 的上限约束。
            推荐值不会超过这个数，并会在报告里说明受白名单限制。
        """
        report: dict[str, Any] = {}

        # 1. 参数量统计
        report["params"] = self._count_params(model_fn, device)

        # 2. GPU 信息
        gpu = self._get_gpu_info(device)
        report["gpu"] = gpu

        # 3. 显存测量 + 线性回归 + 推荐
        if gpu["available"] and device != "cpu":
            measurements = self._measure_memory(model_fn, dataloader_fn, device)
            report["measurements"] = measurements
            predictions = self._predict_memory(measurements)
            report["memory_predictions"] = predictions
            report["recommended_batch_size"] = self._recommend_batch_size(
                predictions, gpu["total_mem_gb"], upper_bound=batch_upper_bound,
            )
        else:
            report["measurements"] = None
            report["memory_predictions"] = None
            report["recommended_batch_size"] = None

        # 4. 时间预估
        if total_samples is not None and epochs is not None:
            bs = target_batch_size or report.get("recommended_batch_size") or self.test_batch_sizes[0]
            report["time_estimate"] = self._estimate_time(
                model_fn, dataloader_fn, device, total_samples, int(bs), epochs,
            )
        else:
            report["time_estimate"] = None

        return report

    # ------------------------------------------------------------------
    # 参数量
    # ------------------------------------------------------------------

    @staticmethod
    def _count_params(model_fn, device: str) -> dict[str, Any]:
        try:
            model = model_fn()
        except Exception as exc:
            return {"error": f"model_fn() 调用失败: {exc}"}

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # 估算显存：fp32 下每参数 4 bytes
        size_mb = round(total * 4 / (1024 * 1024), 2)
        return {
            "total": total,
            "trainable": trainable,
            "size_mb": size_mb,
            "friendly": _friendly_params(total),
        }

    # ------------------------------------------------------------------
    # GPU 信息
    # ------------------------------------------------------------------

    @staticmethod
    def _get_gpu_info(device: str) -> dict[str, Any]:
        if device == "cpu":
            return {"available": False, "name": "CPU", "total_mem_gb": None,
                    "free_mem_gb": None, "note": "CPU 模式，跳过显存预估"}

        try:
            import torch
        except ImportError:
            return {"available": False, "name": "未知", "total_mem_gb": None,
                    "free_mem_gb": None, "error": "torch 未安装"}

        if not torch.cuda.is_available():
            return {"available": False, "name": "CPU (无 CUDA)", "total_mem_gb": None,
                    "free_mem_gb": None}

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(
                torch.cuda.current_device()
            )
        except Exception:
            # torch.cuda.mem_get_info 需要 PyTorch >= 2.0；1.x 版本回退到 nvidia-smi / GPUtil
            logger.warning(
                "torch.cuda.mem_get_info() 不可用（需要 PyTorch >= 2.0，"
                "检测到较低版本时自动回退），改用 nvidia-smi / GPUtil 读取显存",
                exc_info=True,
            )
            # 回退：用 nvidia-smi 或 GPUtil
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    g = gpus[0]
                    return {
                        "available": True,
                        "name": g.name,
                        "total_mem_gb": round(g.memoryTotal / 1024, 1),
                        "free_mem_gb": round(g.memoryFree / 1024, 1),
                    }
            except ImportError:
                # GPUtil 为可选依赖，缺失时回退返回空显存信息
                pass
            return {"available": True, "name": "CUDA GPU", "total_mem_gb": None,
                    "free_mem_gb": None, "note": "无法读取显存信息"}

        return {
            "available": True,
            "name": torch.cuda.get_device_name(torch.cuda.current_device()),
            "total_mem_gb": round(total_bytes / (1024 ** 3), 2),
            "free_mem_gb": round(free_bytes / (1024 ** 3), 2),
        }

    # ------------------------------------------------------------------
    # 显存测量
    # ------------------------------------------------------------------

    @staticmethod
    def _measure_memory(
        model_fn, dataloader_fn, device: str,
    ) -> list[dict[str, Any]]:
        """用小 batch 实际跑前向+反向，记录每个 batch_size 的显存峰值。"""
        try:
            import torch
        except ImportError:
            return []

        results: list[dict[str, Any]] = []
        # 从小到大的顺序测量，避免大 batch 先 OOM 后污染后续测量
        test_sizes = sorted({1, 2, 4})
        for bs in test_sizes:
            try:
                model = model_fn().to(device)
                _, loader, _ = _safe_dataloader(dataloader_fn, bs)
                batch = _first_batch(loader, device)
                if batch is None:
                    results.append({"batch_size": bs, "error": "无法获取 batch 数据"})
                    continue

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                if hasattr(torch.cuda, "reset_accumulated_memory_stats"):
                    torch.cuda.reset_accumulated_memory_stats(device)

                optimizer = torch.optim.Adam(model.parameters())
                model.train()
                out = model(batch)
                if isinstance(out, tuple):
                    out = out[0]
                loss = out.sum()
                loss.backward()
                optimizer.step()

                peak_bytes = torch.cuda.max_memory_allocated(device)
                peak_gb = round(peak_bytes / (1024 ** 3), 3)
                results.append({"batch_size": bs, "peak_memory_gb": peak_gb})

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    results.append({"batch_size": bs, "error": "OOM", "detail": str(exc)[:200]})
                else:
                    results.append({"batch_size": bs, "error": str(exc)[:200]})
            except Exception as exc:
                # 错误已记录进 results 条目（调用方可见），不再单独记日志
                results.append({"batch_size": bs, "error": str(exc)[:200]})
            finally:
                # 每次测量后清理，避免上一次的显存残留影响下一次
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return results

    # ------------------------------------------------------------------
    # 线性回归
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_memory(
        measurements: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """两点线性回归：y = a*x + b。x=batch_size, y=peak_memory_gb。

        a = 每样本可变开销, b = 模型+框架固定开销。
        至少需要 2 个有效的测量点。
        """
        points = [
            (m["batch_size"], m["peak_memory_gb"])
            for m in measurements
            if "peak_memory_gb" in m
        ]
        if len(points) < 2:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        n = len(xs)
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n

        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        if den == 0:
            return None

        a = num / den       # 每样本可变开销 (GB)
        b = y_mean - a * x_mean  # 固定开销 (GB)

        # R²
        ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
        ss_tot = sum((y - y_mean) ** 2 for y in ys)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return {"a_gb_per_sample": round(a, 6), "b_fixed_gb": round(b, 4),
                "r_squared": round(r2, 4), "num_points": n}

    # ------------------------------------------------------------------
    # 推荐
    # ------------------------------------------------------------------

    @staticmethod
    def _recommend_batch_size(
        predictions: dict | None,
        total_mem_gb: float | None,
        margin: float = 0.2,
        upper_bound: int | None = None,
    ) -> dict[str, Any]:
        """根据 memory_margin 推荐最大安全 batch size。"""
        if predictions is None or total_mem_gb is None:
            return {"batch_size": None, "reason": "显存信息或测量数据不足"}

        a = predictions["a_gb_per_sample"]
        b = predictions["b_fixed_gb"]
        available = total_mem_gb * (1 - margin)
        if a <= 0:
            return {"batch_size": None, "reason": "线性回归斜率为非正，无法外推"}

        max_bs = int((available - b) / a)
        if max_bs < 1:
            return {"batch_size": 1, "reason": f"即使 batch_size=1 也可能 OOM（固定开销 {b:.2f}GB）",
                    "constrained_by_whitelist": False}

        constrained = False
        if upper_bound is not None and max_bs > upper_bound:
            constrained = True
            max_bs = upper_bound

        return {
            "batch_size": max_bs,
            "reason": (
                f"在 {total_mem_gb:.1f}GB 显存、{margin:.0%} 安全余量下，"
                f"最大安全 batch_size = {max_bs}"
                + (f"（受 cp_11 白名单上限 {upper_bound} 约束）" if constrained else "")
            ),
            "constrained_by_whitelist": constrained,
        }

    # ------------------------------------------------------------------
    # 时间预估
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_time(
        model_fn,
        dataloader_fn,
        device: str,
        total_samples: int,
        batch_size: int,
        epochs: int,
        warmup_steps: int = 1,
        measure_steps: int = 3,
    ) -> dict[str, Any]:
        """测量单 step 耗时，外推总训练时间。"""
        try:
            import torch
        except ImportError:
            return {"error": "torch 未安装"}

        try:
            model = model_fn().to(device)
            _, loader, _ = _safe_dataloader(dataloader_fn, batch_size)
        except Exception as exc:
            return {"error": f"构建模型/数据加载器失败: {exc}"}

        optimizer = torch.optim.Adam(model.parameters())
        model.train()

        # warmup
        for _ in range(warmup_steps):
            batch = _first_batch(loader, device)
            if batch is None:
                return {"error": "无法获取 batch 数据"}
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        # 测量
        times: list[float] = []
        for _ in range(measure_steps):
            batch = _first_batch(loader, device)
            if batch is None:
                break
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = model(batch)
            if isinstance(out, tuple):
                out = out[0]
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)

        if not times:
            return {"error": "测量失败"}

        avg_step_s = sum(times) / len(times)
        steps_per_epoch = math.ceil(total_samples / max(batch_size, 1))
        total_s = avg_step_s * steps_per_epoch * epochs

        return {
            "avg_step_ms": round(avg_step_s * 1000, 1),
            "steps_per_epoch": steps_per_epoch,
            "total_epochs": epochs,
            "estimated_total": _fmt_duration(total_s),
            "estimated_total_seconds": round(total_s, 1),
        }

    # ------------------------------------------------------------------
    # 终端输出
    # ------------------------------------------------------------------

    @staticmethod
    def print_report(report: dict) -> None:
        """终端表格输出预检报告。"""
        lines = [
            "=" * 56,
            "  训练资源预估报告",
            "=" * 56,
        ]

        # 参数
        params = report.get("params") or {}
        if params.get("error"):
            lines.append(f"  模型: 错误 — {params['error']}")
        else:
            lines.append(f"  模型参数量: {params.get('friendly', '?')}  |  {params.get('size_mb', '?')} MB")

        # GPU
        gpu = report.get("gpu") or {}
        lines.append(f"  GPU: {gpu.get('name', '?')}"
                     + (f" ({gpu['total_mem_gb']}GB)" if gpu.get("total_mem_gb") else ""))

        # 显存预估
        preds = report.get("memory_predictions")
        if preds:
            lines.append("  显存占用预估 (batch_size -> 峰值显存):")
            # 展示从推荐值的 1/4 到 2 倍的几个采样点
            rec = report.get("recommended_batch_size") or {}
            rec_bs = rec.get("batch_size")
        else:
            lines.append("  显存预估: 无（CPU 模式或测量数据不足）")

        # 推荐
        rec = report.get("recommended_batch_size") or {}
        if rec.get("batch_size"):
            lines.append(f"  推荐 batch_size: {rec['batch_size']}"
                         + (" (受 cp_11 白名单约束)" if rec.get("constrained_by_whitelist") else ""))
        elif rec.get("reason"):
            lines.append(f"  推荐: {rec['reason']}")

        # 时间
        tm = report.get("time_estimate") or {}
        if tm.get("estimated_total"):
            lines.append(f"  预估总用时: {tm['estimated_total']}"
                         f" ({tm.get('total_epochs')} epochs, {tm.get('steps_per_epoch')} steps/epoch)")
        elif tm.get("error"):
            lines.append(f"  时间预估: 错误 — {tm['error']}")

        lines.append("=" * 56)
        logger.info("%s", "\n".join(lines))


# -----------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------

def _safe_dataloader(dataloader_fn, batch_size: int):
    """调用 dataloader_fn，自动注入 batch_size。返回 (train_loader, val_loader, n_samples)。"""
    try:
        result = dataloader_fn({"batch_size": batch_size, "num_workers": 0})
    except TypeError:
        result = dataloader_fn(batch_size=batch_size, num_workers=0)
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        loader0, loader1 = result[0], result[1]
        n = result[2] if len(result) >= 3 else 0
        return loader0, loader1, n
    return result, result, 0


def _first_batch(loader, device: str):
    """从 DataLoader 取一个 batch 并移到 device。"""
    try:
        import torch
    except ImportError:
        return None
    try:
        batch = next(iter(loader))
    except StopIteration:
        return None
    if isinstance(batch, (list, tuple)):
        return batch[0].to(device)
    return batch.to(device)


def _friendly_params(total: int) -> str:
    if total >= 1e9:
        return f"{total / 1e9:.2f}B"
    if total >= 1e6:
        return f"{total / 1e6:.2f}M"
    if total >= 1e3:
        return f"{total / 1e3:.1f}K"
    return str(total)


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}min"
    if m:
        return f"{m}min {s}s"
    return f"{s}s"
