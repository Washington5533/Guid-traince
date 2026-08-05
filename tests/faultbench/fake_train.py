"""cp_12 · 可编程失败的假训练脚本。

满足 cp_11 契约四项，但行为完全可控：可以在指定时机以指定方式失败。
故障来自这个脚本和环境，不来自 guardian 内部的 mock——guardian 走的是
完整的真实代码路径。详见 checkpoint/cp_12.md

它自己不 import guardian（和真实训练脚本一样）。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

OOM_TEXT = (
    "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB "
    "(GPU 0; 23.70 GiB total capacity; 21.10 GiB already allocated)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="fake trainer for faultbench")
    # --- 与真实训练脚本一致的参数（契约要求） ---
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--ckpt_dir", default="./checkpoints")
    p.add_argument("--log_file", default="./logs/train.log")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--epoch_seconds", type=float, default=0.05)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ckpt", default=None)
    # --- 故障注入开关 ---
    p.add_argument("--fail-at", type=int, default=None, help="在该 epoch 失败")
    p.add_argument("--fail-mode", default=None,
                   choices=["oom", "type_error", "unknown_error", "oom_if_batch_gt"])
    p.add_argument("--fail-threshold", type=int, default=None,
                   help="oom_if_batch_gt: batch 超过该值才 OOM")
    p.add_argument("--self-kill-at", type=int, default=None, help="在该 epoch 自杀(SIGKILL)")
    p.add_argument("--hang-at", type=int, default=None, help="在该 epoch 挂起不退出")
    p.add_argument("--nan-at", type=int, default=None, help="在该 epoch 输出 NaN loss")
    p.add_argument("--loss-spike-at", type=int, default=None)
    p.add_argument("--loss-spike-ratio", type=float, default=0.5)
    return p.parse_args()


def save_ckpt(ckpt_dir: Path, epoch: int, loss: float, args) -> None:
    """写出符合 cp_11 checkpoint_schema 的 checkpoint（原子写：tmp + rename）。"""
    final = ckpt_dir / f"cp_{epoch}"
    tmp = ckpt_dir / f"cp_{epoch}.tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": {"w": [0.1 * epoch]},
        "optimizer_state_dict": {"lr": args.lr},
        "rng_state": [epoch, 42],
    }
    try:
        import torch
        torch.save(payload, tmp / "model.pth")
    except ImportError:
        (tmp / "model.pth").write_text(json.dumps(payload), encoding="utf-8")

    (tmp / "metrics.json").write_text(
        json.dumps({"epoch": epoch, "train/loss": loss, "val/accuracy": min(0.99, 0.5 + 0.05 * epoch),
                    "lr": args.lr, "batch_size": args.batch_size}, ensure_ascii=False),
        encoding="utf-8",
    )
    if final.exists():
        for f in final.iterdir():
            f.unlink()
        final.rmdir()
    tmp.rename(final)


def load_start_epoch(args) -> int:
    """契约要求 1：可续训。从 --ckpt 指定的目录读回 epoch。"""
    if not args.resume or not args.ckpt:
        return 0
    mpath = Path(args.ckpt) / "metrics.json"
    if not mpath.exists():
        return 0
    try:
        data = json.loads(mpath.read_text(encoding="utf-8"))
        return int(data.get("epoch", -1)) + 1
    except (ValueError, OSError):
        return 0


def main() -> int:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start = load_start_epoch(args)
    if start:
        print(f"resumed from {args.ckpt}, starting at epoch {start}", flush=True)

    with log_path.open("a", encoding="utf-8") as log:
        for epoch in range(start, args.epochs):
            time.sleep(args.epoch_seconds)

            loss = max(0.01, 1.0 / (epoch + 1))
            if args.loss_spike_at is not None and epoch == args.loss_spike_at:
                loss *= (1.0 + args.loss_spike_ratio) * 3
            if args.nan_at is not None and epoch == args.nan_at:
                loss = float("nan")

            # 契约要求 2：指标可观测（日志通道）
            line = (f"epoch {epoch} loss {loss:.6f} "
                    f"val_acc {min(0.99, 0.5 + 0.05 * epoch):.4f} lr {args.lr:g}")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            os.fsync(log.fileno())

            # --- 故障注入 ---
            if args.hang_at is not None and epoch == args.hang_at:
                print("hanging forever (simulated deadlock)", flush=True)
                while True:
                    time.sleep(3600)

            if args.self_kill_at is not None and epoch == args.self_kill_at:
                # 真实的外部 kill 只发生一次；用 marker 保证续训后不再自杀，
                # 否则会变成"每次重启都在同一 epoch 被杀"的无限循环（不是真实故障形态）
                marker = ckpt_dir / ".self_killed"
                if not marker.exists():
                    marker.write_text(str(epoch), encoding="utf-8")
                    print("self-killing with SIGKILL", flush=True)
                    log.flush()
                    log.close()
                    if hasattr(signal, "SIGKILL"):
                        os.kill(os.getpid(), signal.SIGKILL)
                    sys.exit(137)  # Windows 无 SIGKILL，用 shell 风格退出码等价表示

            if args.fail_mode == "oom_if_batch_gt":
                # --fail-at 未指定时每个 epoch 都检查（epoch 0 即失败，此时还没有 ckpt）；
                # 指定时只在该 epoch 检查，便于构造"已有 ckpt 可回滚"的场景
                due = args.fail_at is None or epoch == args.fail_at
                if due and args.fail_threshold is not None and args.batch_size > args.fail_threshold:
                    print(OOM_TEXT, file=sys.stderr, flush=True)
                    return 1
            elif args.fail_mode and args.fail_at is not None and epoch == args.fail_at:
                if args.fail_mode == "oom":
                    print(OOM_TEXT, file=sys.stderr, flush=True)
                elif args.fail_mode == "type_error":
                    print("Traceback (most recent call last):\n"
                          '  File "train.py", line 42, in train\n'
                          "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
                          file=sys.stderr, flush=True)
                elif args.fail_mode == "unknown_error":
                    print("Fatal: widget subsystem returned code 0x8badf00d",
                          file=sys.stderr, flush=True)
                return 1

            if epoch % args.save_every == 0:
                save_ckpt(ckpt_dir, epoch, loss, args)

    print("training completed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
