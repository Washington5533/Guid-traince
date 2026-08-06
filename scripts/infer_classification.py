#!/usr/bin/env python
"""固定分类推理脚本。

用法:
    python scripts/infer_classification.py \\
        --checkpoint checkpoints/cp_20/model.pth \\
        --inputs data/test/ \\
        --output logs/inference/

生成 logs/inference/results.json: [
    {"image_path": "...", "true_label": "...", "predicted_class": "...",
     "confidence": 0.95, "top5": [("class_a", 0.95), ...]}
]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class ImageFolderDataset(Dataset):
    """从目录加载图片，可选标签（从子目录名推断）。"""

    def __init__(self, root: str | Path, transform=None, image_size: int = 224):
        self.root = Path(root)
        self.transform = transform or transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.samples: list[tuple[str, str | None]] = []
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() in exts and p.is_file():
                label = p.parent.name if p.parent != self.root else None
                self.samples.append((str(p), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        return tensor, path, label


def load_model(checkpoint_path: str, model_fn=None):
    """加载模型。"""
    if model_fn is None:
        from train import build_model
        model_fn = build_model

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
    parser = argparse.ArgumentParser(description="固定分类推理脚本")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", default="./logs/inference")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", help="model entry (e.g. train:build_model)")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--class-names", help="JSON file mapping class index to name")
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

    # 加载类别名
    class_names = None
    if args.class_names:
        class_names = json.loads(Path(args.class_names).read_text(encoding="utf-8"))

    # 数据
    dataset = ImageFolderDataset(args.inputs, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    results = []
    with torch.no_grad():
        for images, paths, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            top5_vals, top5_idx = torch.topk(probs, min(5, probs.shape[1]), dim=1)

            for i, path in enumerate(paths):
                top5 = []
                for j in range(top5_idx.shape[1]):
                    cls_idx = top5_idx[i, j].item()
                    cls_name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
                    top5.append((cls_name, round(top5_vals[i, j].item(), 4)))

                results.append({
                    "image_path": path,
                    "true_label": labels[i],
                    "predicted_class": top5[0][0] if top5 else None,
                    "confidence": top5[0][1] if top5 else 0.0,
                    "top5": top5,
                })

    # 保存
    out_file = out / "results.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # 摘要
    if results:
        confs = [r["confidence"] for r in results]
        print(f"推理完成: {len(results)} 张图片")
        print(f"  平均置信度: {sum(confs) / len(confs):.4f}")
        print(f"  Top-1 准确率: 无法计算（缺少标签）" if not any(r.get("true_label") for r in results) else "  OK")
        print(f"  结果: {out_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
