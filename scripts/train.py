"""cp_7 · MNIST 参考训练脚本 (train.py)

被 guardian 守护的示例训练脚本，满足 checkpoint/cp_11.md 的契约四项，
自身完全不 import 任何 guardian 代码——sidecar 默认路径下训练脚本
对 guardian 的存在无感知，被当作普通子进程拉起。

用法（不经 guardian，独立可跑）：
    python train.py --epochs 5

被 guardian 守护：
    guarftrain watch -- python train.py --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SimpleCNN(nn.Module):
    """~600K 参数，MNIST 手写数字分类。"""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def build_model() -> SimpleCNN:
    """契约要求：buildable_entry.model_fn，供 cp_1 独立进程 import 后测量。"""
    return SimpleCNN()


def get_dataloaders(config: dict | None = None) -> tuple[DataLoader, DataLoader, int]:
    """契约要求：buildable_entry.dataloader_fn，供 cp_1 独立进程 import 后测量。

    自动下载 MNIST 到 ./data/。返回 (train_loader, val_loader, 训练集样本数)。
    """
    from torchvision import datasets, transforms

    cfg = config or {}
    data_dir = cfg.get("data_dir", "./data")
    batch_size = int(cfg.get("batch_size", 64))
    num_workers = int(cfg.get("num_workers", 0))

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
    val_set = datasets.MNIST(data_dir, train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, len(train_set)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, device: str) -> tuple[float, float]:
    """单 epoch 训练，返回 (平均 loss, accuracy)。"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def validate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    """全量验证，返回 (accuracy, loss)。"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)
    return correct / max(total, 1), total_loss / max(total, 1)


def save_ckpt(ckpt_dir: Path, epoch: int, model: nn.Module, optimizer,
              metrics: dict) -> None:
    """契约要求：checkpoint_schema 必需键 epoch/model_state_dict/optimizer_state_dict。

    原子写（tmp 目录 + rename），避免 guardian 在写入过程中轮询到半截文件。
    """
    final = ckpt_dir / f"cp_{epoch}"
    tmp = ckpt_dir / f"cp_{epoch}.tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": torch.get_rng_state(),
    }
    torch.save(payload, tmp / "model.pth")
    (tmp / "metrics.json").write_text(
        json.dumps({"epoch": epoch, **metrics}, ensure_ascii=False), encoding="utf-8",
    )

    if final.exists():
        for f in final.iterdir():
            f.unlink()
        final.rmdir()
    tmp.rename(final)


def resolve_ckpt(ckpt_arg: str) -> Path:
    """`--ckpt` 指向的 cp_N 目录（可能是目录本身，也可能是目录内的 model.pth）。"""
    p = Path(ckpt_arg)
    if p.is_dir():
        return p / "model.pth"
    return p


def load_start_epoch(model: nn.Module, optimizer, args) -> int:
    """契约要求 1：可续训。从 --ckpt 指定的 checkpoint 恢复模型/优化器状态。"""
    if not args.resume or not args.ckpt:
        return 0
    ckpt_path = resolve_ckpt(args.ckpt)
    if not ckpt_path.exists():
        print(f"[警告] --ckpt {args.ckpt} 不存在，从头开始训练", flush=True)
        return 0
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if "rng_state" in ckpt:
        torch.set_rng_state(ckpt["rng_state"].to(torch.uint8).cpu())
    print(f"resumed from {args.ckpt}, starting at epoch {ckpt['epoch'] + 1}", flush=True)
    return ckpt["epoch"] + 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MNIST 参考训练脚本（guardian 守护对象）")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--ckpt_dir", default="./checkpoints")
    p.add_argument("--log_file", default="./logs/train.log")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--device", default="cpu")
    # 契约要求 1：可续训
    p.add_argument("--resume", action="store_true", help="从 --ckpt 指定的断点续训")
    p.add_argument("--ckpt", default=None, help="断点目录，如 checkpoints/cp_5")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader, val_loader, _ = get_dataloaders({
        "data_dir": args.data_dir, "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    })

    start_epoch = load_start_epoch(model, optimizer, args)

    with log_path.open("a", encoding="utf-8") as log:
        for epoch in range(start_epoch, args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            val_acc, val_loss = validate(model, val_loader, device)

            # 契约要求 2：指标可观测（结构化日志行，可被 cp_2 的 log_pattern 解析）
            line = (f"epoch {epoch} loss {train_loss:.6f} val_acc {val_acc:.4f} "
                    f"lr {args.lr:g}")
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()
            os.fsync(log.fileno())

            # 契约要求 3：checkpoint 含必需字段
            if epoch % args.save_every == 0:
                save_ckpt(ckpt_dir, epoch, model, optimizer, {
                    "train/loss": train_loss, "train/accuracy": train_acc,
                    "val/loss": val_loss, "val/accuracy": val_acc,
                    "lr": args.lr, "batch_size": args.batch_size,
                })

    print("training completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
