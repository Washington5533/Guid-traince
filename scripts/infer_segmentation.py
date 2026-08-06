#!/usr/bin/env python
"""固定分割推理脚本。

用法:
    python scripts/infer_segmentation.py \\
        --checkpoint checkpoints/cp_20/model.pth \\
        --inputs data/test/ \\
        --output logs/inference/

生成 logs/inference/results.json: [
    {"image_path": "...", "mask_path": "mask_000.png", "per_class_iou": {...}}
]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class ImageFolderDataset(Dataset):
    """从目录加载图片。"""

    def __init__(self, root: str | Path, transform=None, image_size: int = 512):
        self.root = Path(root)
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.samples: list[str] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() in exts and p.is_file():
                self.samples.append(str(p))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            orig_size = img.size  # (W, H)
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 512, 512)
            orig_size = (512, 512)
        return tensor, path, orig_size


def load_model(checkpoint_path: str, model_fn=None):
    """加载模型。"""
    if model_fn is None:
        try:
            from train import build_model
            model_fn = build_model
        except Exception:
            print("错误: 需要 --model 参数指定模型入口", file=sys.stderr)
            sys.exit(1)

    model = model_fn()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(state, dict):
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="固定分割推理脚本")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", default="./logs/inference")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", help="model entry (e.g. train:build_model)")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=21,
                        help="类别数（含背景）")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    # 加载模型
    model_fn = None
    if args.model:
        mod_path, fn_name = args.model.split(":", 1)
        import importlib
        mod = importlib.import_module(mod_path)
        model_fn = getattr(mod, fn_name)

    model = load_model(args.checkpoint, model_fn).to(device)

    # 数据
    dataset = ImageFolderDataset(args.inputs, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    masks_dir = out / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with torch.no_grad():
        for images, paths, orig_sizes in loader:
            images = images.to(device)
            outputs = model(images)

            # 适配多种分割模型输出格式
            if isinstance(outputs, dict):
                logits = outputs.get("out", outputs.get("mask", outputs.get("logits")))
            elif isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)  # [B, C, H, W]
            preds = torch.argmax(probs, dim=1)     # [B, H, W]

            for i, path in enumerate(paths):
                mask_name = f"mask_{Path(path).stem}.png"
                mask_path = masks_dir / mask_name

                # 保存 mask 为 PNG
                mask_np = preds[i].cpu().numpy().astype(np.uint8)
                mask_img = Image.fromarray(mask_np)
                mask_img.save(str(mask_path))

                # 计算每个类别的置信度
                per_class_conf = {}
                for c in range(probs.shape[1]):
                    class_mask = (preds[i] == c)
                    if class_mask.sum() > 0:
                        per_class_conf[c] = round(probs[i, c][class_mask].mean().item(), 4)

                results.append({
                    "image_path": path,
                    "mask_path": str(mask_path),
                    "orig_size": list(orig_sizes[i]) if isinstance(orig_sizes, tuple) else list(orig_sizes[i]) if hasattr(orig_sizes, '__getitem__') else list(orig_sizes),
                    "per_class_confidence": per_class_conf,
                    "dominant_class": int(preds[i].mode().values.item()) if hasattr(preds[i], 'mode') else int(torch.mode(preds[i].flatten()).values.item()),
                })

    # 保存
    out_file = out / "results.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"推理完成: {len(results)} 张图片")
    print(f"  Mask 目录: {masks_dir}")
    print(f"  结果: {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
