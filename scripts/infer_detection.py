#!/usr/bin/env python
"""固定检测推理脚本。

用法:
    python scripts/infer_detection.py \\
        --checkpoint checkpoints/cp_20/model.pth \\
        --inputs data/test/ \\
        --output logs/inference/

生成 logs/inference/results.json: [
    {"image_path": "...", "predictions": [
        {"bbox": [x1,y1,x2,y2], "class": "...", "confidence": 0.85}
    ]}
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
    """从目录加载图片（检测任务不强制需要标签）。"""

    def __init__(self, root: str | Path, transform=None, image_size: int = 640):
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
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 640, 640)
        return tensor, path


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
    parser = argparse.ArgumentParser(description="固定检测推理脚本")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", default="./logs/inference")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", help="model entry (e.g. train:build_model)")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--class-names", help="JSON file mapping class index to name")
    parser.add_argument("--conf-threshold", type=float, default=0.25,
                        help="置信度阈值（低于此值的结果被过滤）")
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

    # 类别名
    class_names = None
    if args.class_names:
        class_names = json.loads(Path(args.class_names).read_text(encoding="utf-8"))

    # 数据
    dataset = ImageFolderDataset(args.inputs, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    results = []
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            outputs = model(images)

            # 适配多种检测模型输出格式
            predictions = _parse_detection_output(outputs, class_names, args.conf_threshold)

            for i, path in enumerate(paths):
                img_preds = [p for p in predictions if p.get("image_index") == i] if isinstance(predictions, list) else []
                # 如果 _parse_detection_output 返回了 per-image 的结果
                if not img_preds:
                    img_preds = _extract_per_image(predictions, i)

                results.append({
                    "image_path": path,
                    "predictions": img_preds,
                })

    # 保存
    out_file = out / "results.json"
    out_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    total_preds = sum(len(r["predictions"]) for r in results)
    print(f"推理完成: {len(results)} 张图片, {total_preds} 个检测框")
    print(f"  结果: {out_file}")

    return 0


def _parse_detection_output(outputs, class_names, conf_threshold):
    """尝试解析多种检测模型输出格式。"""
    results = []

    if isinstance(outputs, dict):
        boxes = outputs.get("boxes", outputs.get("bbox", outputs.get("pred_boxes")))
        scores = outputs.get("scores", outputs.get("confidence", outputs.get("pred_scores")))
        labels = outputs.get("labels", outputs.get("classes", outputs.get("pred_classes")))

        if boxes is not None and scores is not None:
            for i in range(len(boxes)):
                if scores[i].item() >= conf_threshold:
                    bbox = boxes[i].tolist()
                    cls_idx = int(labels[i].item()) if labels is not None else 0
                    cls_name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
                    results.append({
                        "bbox": bbox[:4],
                        "class": cls_name,
                        "confidence": round(scores[i].item(), 4),
                    })
    elif isinstance(outputs, (list, tuple)):
        # 假设是 (boxes, scores) 或类似
        for item in outputs:
            if isinstance(item, torch.Tensor) and item.dim() >= 2:
                if item.shape[1] >= 6:
                    # YOLO 格式: [batch_idx, class, conf, x1, y1, x2, y2]
                    for row in item:
                        vals = row.tolist()
                        if vals[2] >= conf_threshold:
                            cls_idx = int(vals[1])
                            cls_name = class_names[cls_idx] if class_names and cls_idx < len(class_names) else str(cls_idx)
                            results.append({
                                "bbox": vals[3:7],
                                "class": cls_name,
                                "confidence": round(vals[2], 4),
                            })

    return results


def _extract_per_image(predictions, img_idx):
    """从批量预测中提取单张图片的结果。"""
    # 尝试按 image_index 筛选
    per_img = [p for p in predictions if p.get("image_index") == img_idx]
    if per_img:
        return per_img
    # 如果只有一个 image，直接返回所有
    if img_idx == 0:
        return predictions
    return []


if __name__ == "__main__":
    sys.exit(main())
