#!/usr/bin/env python
"""CLIP 模型适配器 — 供 guarftrain 工具调用。

用法:
    # 可视化 CLIP 结构
    guarftrain visualize --model scripts/clip_adapter:build_model_for_viz

    # 推理（需指定 checkpoint）
    guarftrain infer --ckpt 0 --inputs /path/to/clip-project/data/oxford-iiit-pet/images/ --task classification
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 外部 CLIP 项目位置：默认同级目录 ../clip-project，
# 可用环境变量 GUARDIAN_CLIP_PROJECT_DIR 指定自己的项目路径。
_CLIP_PROJECT = Path(
    os.environ.get(
        "GUARDIAN_CLIP_PROJECT_DIR",
        str(Path(__file__).resolve().parent.parent.parent / "clip-project"),
    )
)
_CLIP_SOURCE = _CLIP_PROJECT / "clip_sourse" / "CLIP"

if str(_CLIP_SOURCE) not in sys.path:
    sys.path.insert(0, str(_CLIP_SOURCE))
if str(_CLIP_PROJECT) not in sys.path:
    sys.path.insert(0, str(_CLIP_PROJECT))


def build_model_for_viz():
    """构建 CLIP 视觉编码器（用于模型结构可视化）。

    guarftrain 调用: visualize --model scripts/clip_adapter:build_model_for_viz
    """
    import torch
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    # 只返回视觉编码器部分（核心计算图）
    return model.visual


def build_model_full():
    """构建完整 CLIP 模型（含文本编码器）。

    guarftrain 调用: visualize --model scripts/clip_adapter:build_model_full
    """
    import torch
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    return model


def build_model_with_head():
    """构建 CLIP + 线性分类头（step2_linear_probe.py 的训练产物）。

    需要 best_head.pt 在 CLIP 项目目录下。
    guarftrain 调用: visualize --model scripts/clip_adapter:build_model_with_head
    """
    import torch
    import torch.nn as nn
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)

    class CLIPClassifier(nn.Module):
        def __init__(self, clip_model, num_classes):
            super().__init__()
            self.clip = clip_model
            self.head = nn.Linear(512, num_classes)
            for param in self.clip.parameters():
                param.requires_grad = False

        def forward(self, images):
            with torch.no_grad():
                features = self.clip.encode_image(images).float()
            return self.head(features)

    # Oxford-IIIT Pets 有 37 类
    model = CLIPClassifier(clip_model, num_classes=37)

    # 加载训练好的分类头
    head_path = _CLIP_PROJECT / "best_head.pt"
    if head_path.exists():
        state = torch.load(str(head_path), map_location=device)
        # best_head.pt 存储的是 head.state_dict()（纯 Linear 权重）
        if "weight" in state:
            model.head.load_state_dict(state)
        else:
            model.load_state_dict(state, strict=False)
        print(f"[clip_adapter] 已加载分类头: {head_path}", flush=True)

    return model


# --------------- 为 guarftrain infer 提供入口 ---------------

def get_dataloaders(batch_size=32):
    """返回 (train_loader, test_loader, class_names)。

    供 guarftrain 的 preflight / infer 使用。
    """
    import torch
    from torchvision.datasets import OxfordIIITPet
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def _to_rgb(img):
        return img.convert("RGB")

    transform = Compose([
        Resize(224), CenterCrop(224), _to_rgb, ToTensor(),
        Normalize(CLIP_MEAN, CLIP_STD),
    ])

    data_root = str(_CLIP_PROJECT / "data")

    test_data = OxfordIIITPet(
        root=data_root, split="test", download=False,
        transform=transform,
    )

    loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size,
        shuffle=False, num_workers=0,
    )

    return loader, test_data.classes
