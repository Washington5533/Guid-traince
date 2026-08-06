"""经典深度学习组件库。

供 ModelVisualizer 的 AI 分析使用：发现模型瓶颈时，从库中匹配可替换的
高效组件。若无匹配则 AI 自行编写新组件代码。

组件格式:
    {
        "name": "组件名",
        "description": "一句话描述",
        "tags": ["分类标签"],
        "replaces": ["可替换的目标层类型"],
        "flops_saving": "高/中/低",
        "params_saving": "高/中/低",
        "when_to_use": "适用场景",
        "code": "可执行的 PyTorch 代码片段",
    }
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# 经典组件库
# ---------------------------------------------------------------------------

COMPONENT_LIBRARY: list[dict[str, Any]] = [

    # ---- 卷积优化 ----

    {
        "name": "DepthwiseSeparableConv",
        "description": "深度可分离卷积：将标准卷积拆分为 depthwise + pointwise，大幅降低 FLOPs 和参数量",
        "tags": ["卷积优化", "轻量化", "FLOPs降低"],
        "replaces": ["Conv2d", "Conv3d"],
        "flops_saving": "高（约 8-9 倍降低）",
        "params_saving": "高（约 8-9 倍降低）",
        "when_to_use": "3×3 卷积占计算量 30% 以上，且通道数 ≥ 64 时效果显著。不适合 1×1 卷积。",
        "code": (
            "nn.Sequential(\n"
            "    # Depthwise: 每个通道独立卷积\n"
            "    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),\n"
            "    nn.BatchNorm2d(in_channels),\n"
            "    nn.ReLU(inplace=True),\n"
            "    # Pointwise: 1×1 混合通道\n"
            "    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),\n"
            "    nn.BatchNorm2d(out_channels),\n"
            "    nn.ReLU(inplace=True),\n"
            ")"
        ),
        "shape_note": "in_channels, out_channels 需从原始层推断",
    },

    {
        "name": "BottleneckResBlock",
        "description": "ResNet 瓶颈块：1×1降维 → 3×3 → 1×1升维 + skip connection，用于替换大卷积",
        "tags": ["残差", "瓶颈", "深层网络"],
        "replaces": ["Conv2d", "Sequential"],
        "flops_saving": "中（取决于压缩比）",
        "params_saving": "中（取决于压缩比）",
        "when_to_use": "连续两个 3×3 卷积、通道数 ≥ 256、网络深度 > 10 层",
        "code": (
            "class BottleneckResBlock(nn.Module):\n"
            "    def __init__(self, in_channels, out_channels, stride=1, expansion=4):\n"
            "        super().__init__()\n"
            "        mid_channels = out_channels // expansion\n"
            "        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)\n"
            "        self.bn1 = nn.BatchNorm2d(mid_channels)\n"
            "        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False)\n"
            "        self.bn2 = nn.BatchNorm2d(mid_channels)\n"
            "        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)\n"
            "        self.bn3 = nn.BatchNorm2d(out_channels)\n"
            "        self.relu = nn.ReLU(inplace=True)\n"
            "        self.downsample = (\n"
            "            nn.Sequential(\n"
            "                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),\n"
            "                nn.BatchNorm2d(out_channels),\n"
            "            ) if in_channels != out_channels or stride != 1 else nn.Identity()\n"
            "        )\n"
            "    def forward(self, x):\n"
            "        identity = self.downsample(x)\n"
            "        out = self.relu(self.bn1(self.conv1(x)))\n"
            "        out = self.relu(self.bn2(self.conv2(out)))\n"
            "        out = self.bn3(self.conv3(out))\n"
            "        out += identity\n"
            "        return self.relu(out)"
        ),
    },

    # ---- 注意力机制 ----

    {
        "name": "SEBlock",
        "description": "Squeeze-and-Excitation：通道注意力，通过学习每个通道的重要性权重来提升特征表达",
        "tags": ["注意力", "通道", "即插即用"],
        "replaces": ["Conv2d", "Linear", "Sequential"],
        "flops_saving": "几乎无额外开销（<0.1%）",
        "params_saving": "几乎无额外开销",
        "when_to_use": "任意卷积/全连接层之后，通道数 ≥ 32。几乎零成本提升。",
        "code": (
            "class SEBlock(nn.Module):\n"
            "    \"\"\"Squeeze-and-Excitation block — 通道注意力\"\"\"\n"
            "    def __init__(self, channels, reduction=16):\n"
            "        super().__init__()\n"
            "        self.squeeze = nn.AdaptiveAvgPool2d(1)\n"
            "        self.excite = nn.Sequential(\n"
            "            nn.Linear(channels, channels // reduction, bias=False),\n"
            "            nn.ReLU(inplace=True),\n"
            "            nn.Linear(channels // reduction, channels, bias=False),\n"
            "            nn.Sigmoid(),\n"
            "        )\n"
            "    def forward(self, x):\n"
            "        b, c, _, _ = x.shape\n"
            "        w = self.squeeze(x).view(b, c)\n"
            "        w = self.excite(w).view(b, c, 1, 1)\n"
            "        return x * w\n\n"
            "# 使用方式：在任意卷积后插入\n"
            "# self.se = SEBlock(out_channels, reduction=16)\n"
            "# x = self.se(self.conv(x))"
        ),
    },

    {
        "name": "CBAM",
        "description": "Convolutional Block Attention Module：通道 + 空间双注意力",
        "tags": ["注意力", "通道", "空间", "即插即用"],
        "replaces": ["Conv2d", "Sequential"],
        "flops_saving": "低（约 1-3% 额外开销）",
        "params_saving": "低",
        "when_to_use": "需要同时关注'哪些通道重要'和'空间上哪重要'的场景。检测和分割任务优先推荐。",
        "code": (
            "class CBAM(nn.Module):\n"
            "    \"\"\"通道注意力 + 空间注意力\"\"\"\n"
            "    def __init__(self, channels, reduction=16, kernel_size=7):\n"
            "        super().__init__()\n"
            "        # 通道注意力\n"
            "        self.channel_attn = nn.Sequential(\n"
            "            nn.AdaptiveAvgPool2d(1),\n"
            "            nn.Conv2d(channels, channels // reduction, 1),\n"
            "            nn.ReLU(inplace=True),\n"
            "            nn.Conv2d(channels // reduction, channels, 1),\n"
            "            nn.Sigmoid(),\n"
            "        )\n"
            "        # 空间注意力\n"
            "        self.spatial_attn = nn.Sequential(\n"
            "            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2),\n"
            "            nn.Sigmoid(),\n"
            "        )\n"
            "    def forward(self, x):\n"
            "        # 通道注意力\n"
            "        ca = self.channel_attn(x)\n"
            "        x = x * ca\n"
            "        # 空间注意力\n"
            "        avg_out = x.mean(dim=1, keepdim=True)\n"
            "        max_out, _ = x.max(dim=1, keepdim=True)\n"
            "        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))\n"
            "        return x * sa"
        ),
    },

    # ---- 激活函数替换 ----

    {
        "name": "GELUtoSiLU",
        "description": "QuickGELU → SiLU 替换：SiLU（Swish）在训练时有更好的梯度流",
        "tags": ["激活函数", "微调", "训练优化"],
        "replaces": ["QuickGELU", "GELU"],
        "flops_saving": "无",
        "params_saving": "无",
        "when_to_use": "CLIP/Transformer 中的 QuickGELU 换成 SiLU 通常不影响精度，但训练更稳定",
        "code": (
            "# 替换: from clip.model import QuickGELU\n"
            "# 改为:\n"
            "# self.activation = nn.SiLU()  # SiLU = Swish, 等价于 x * sigmoid(x)\n"
            "# QuickGELU 在 fp16 下有问题，SiLU 数值更稳定"
        ),
    },

    # ---- Transformer 优化 ----

    {
        "name": "MultiQueryAttention",
        "description": "Multi-Query Attention：多个 query head 共享一组 key/value head，减少 KV cache",
        "tags": ["Transformer", "注意力优化", "推理加速"],
        "replaces": ["MultiheadAttention"],
        "flops_saving": "低（理论上的 FLOPs 减少，但在推理时 KV cache 大幅缩小）",
        "params_saving": "中（key/value 参数量减半以上）",
        "when_to_use": "Transformer 层数 ≥ 6、需要推理加速时。生成式任务优先推荐。",
        "code": (
            "class MultiQueryAttention(nn.Module):\n"
            "    \"\"\"Multi-Query Attention: Q 多头，K/V 单头\"\"\"\n"
            "    def __init__(self, dim, num_heads, dropout=0.0):\n"
            "        super().__init__()\n"
            "        self.num_heads = num_heads\n"
            "        self.head_dim = dim // num_heads\n"
            "        self.q_proj = nn.Linear(dim, dim)\n"
            "        # K, V 只投影到一个头\n"
            "        self.k_proj = nn.Linear(dim, self.head_dim)\n"
            "        self.v_proj = nn.Linear(dim, self.head_dim)\n"
            "        self.out_proj = nn.Linear(dim, dim)\n"
            "        self.dropout = nn.Dropout(dropout)\n"
            "    def forward(self, x):\n"
            "        B, N, C = x.shape\n"
            "        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)\n"
            "        k = self.k_proj(x).unsqueeze(1)  # [B, 1, N, head_dim]\n"
            "        v = self.v_proj(x).unsqueeze(1)\n"
            "        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)\n"
            "        attn = self.dropout(attn.softmax(dim=-1))\n"
            "        x = (attn @ v).transpose(1, 2).reshape(B, N, C)\n"
            "        return self.out_proj(x)"
        ),
    },

    # ---- 归一化 ----

    {
        "name": "LayerNorm2BN",
        "description": "LayerNorm → BatchNorm 替换：在 CNN 中 BatchNorm 通常比 LayerNorm 效果更好",
        "tags": ["归一化", "CNN优化"],
        "replaces": ["LayerNorm"],
        "flops_saving": "无",
        "params_saving": "无",
        "when_to_use": "ViT 等 Transformer 用于图像时，部分 LayerNorm 可换为 BatchNorm。注意：需要 batch_size > 1",
        "code": (
            "# 替换 LayerNorm 为 BatchNorm2d:\n"
            "# Before: self.norm = nn.LayerNorm(dim)\n"
            "# After:  self.norm = nn.BatchNorm2d(dim)\n"
            "# 注意：需要确保输入是 4D tensor [B, C, H, W] 而非 [B, N, C]"
        ),
    },

    # ---- 轻量化 ----

    {
        "name": "InvertedResidualBlock",
        "description": "MobileNetV2 倒残差块：先升维 → depthwise → 降维 + 残差连接",
        "tags": ["轻量化", "MobileNet", "FLOPs降低"],
        "replaces": ["Conv2d", "Sequential", "ResidualAttentionBlock"],
        "flops_saving": "高",
        "params_saving": "高",
        "when_to_use": "目标是在移动端/边缘设备部署。不适合需要大感受野的任务。",
        "code": (
            "class InvertedResidual(nn.Module):\n"
            "    \"\"\"MobileNetV2 倒残差块\"\"\"\n"
            "    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):\n"
            "        super().__init__()\n"
            "        hidden = in_channels * expand_ratio\n"
            "        self.use_residual = (stride == 1 and in_channels == out_channels)\n"
            "        layers = []\n"
            "        if expand_ratio != 1:\n"
            "            layers.extend([\n"
            "                nn.Conv2d(in_channels, hidden, 1, bias=False),\n"
            "                nn.BatchNorm2d(hidden),\n"
            "                nn.ReLU6(inplace=True),\n"
            "            ])\n"
            "        layers.extend([\n"
            "            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),\n"
            "            nn.BatchNorm2d(hidden),\n"
            "            nn.ReLU6(inplace=True),\n"
            "            nn.Conv2d(hidden, out_channels, 1, bias=False),\n"
            "            nn.BatchNorm2d(out_channels),\n"
            "        ])\n"
            "        self.conv = nn.Sequential(*layers)\n"
            "    def forward(self, x):\n"
            "        return x + self.conv(x) if self.use_residual else self.conv(x)"
        ),
    },

    # ---- 新型架构 ----

    {
        "name": "PatchMerging",
        "description": "Swin Transformer 的 Patch Merging：下采样同时增通道，替代 stride-2 卷积",
        "tags": ["Transformer", "下采样", "层次化"],
        "replaces": ["Conv2d(kernel_size>1,stride=2)"],
        "flops_saving": "中",
        "params_saving": "中",
        "when_to_use": "ViT 类模型需要层次化特征金字塔时（检测/分割任务）",
        "code": (
            "class PatchMerging(nn.Module):\n"
            "    \"\"\"Swin Transformer Patch Merging — 2×下采样\"\"\"\n"
            "    def __init__(self, dim):\n"
            "        super().__init__()\n"
            "        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)\n"
            "        self.norm = nn.LayerNorm(4 * dim)\n"
            "    def forward(self, x, H, W):\n"
            "        B, N, C = x.shape\n"
            "        x = x.view(B, H, W, C)\n"
            "        # 隔点采样 + 拼接 → [B, H/2, W/2, 4C]\n"
            "        x = torch.cat([\n"
            "            x[:, 0::2, 0::2, :], x[:, 1::2, 0::2, :],\n"
            "            x[:, 0::2, 1::2, :], x[:, 1::2, 1::2, :]\n"
            "        ], dim=-1)\n"
            "        x = x.view(B, -1, 4 * C)\n"
            "        x = self.norm(x)\n"
            "        return self.reduction(x), x.view(B, H//2, W//2, 2*C).shape[1]"
        ),
    },

    # ---- 训练技巧 ----

    {
        "name": "StochasticDepth",
        "description": "随机深度（DropPath）：训练时随机跳过整个残差块，正则化深层网络",
        "tags": ["正则化", "训练技巧", "深层网络"],
        "replaces": ["ResidualAttentionBlock", "Bottleneck"],
        "flops_saving": "不适用（训练时反而增加，仅推理时不变）",
        "params_saving": "无",
        "when_to_use": "Transformer ≥ 12 层或 ResNet ≥ 50 层时推荐。减少过拟合，提升泛化。",
        "code": (
            "class DropPath(nn.Module):\n"
            "    \"\"\"Stochastic Depth (DropPath)\"\"\"\n"
            "    def __init__(self, drop_prob=0.0):\n"
            "        super().__init__()\n"
            "        self.drop_prob = drop_prob\n"
            "    def forward(self, x):\n"
            "        if self.drop_prob == 0.0 or not self.training:\n"
            "            return x\n"
            "        keep_prob = 1 - self.drop_prob\n"
            "        shape = (x.shape[0],) + (1,) * (x.ndim - 1)\n"
            "        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)\n"
            "        return x / keep_prob * rand.floor()\n\n"
            "# 使用：在残差块的 forward 中\n"
            "# self.drop_path = DropPath(drop_prob)\n"
            "# return x + self.drop_path(self.block(x))"
        ),
    },
]

# ---------------------------------------------------------------------------
# 匹配引擎
# ---------------------------------------------------------------------------

def match_components(
    layer_type: str,
    layer_params: int = 0,
    layer_flops: int = 0,
    context: dict | None = None,
) -> list[dict]:
    """根据瓶颈层特征匹配可替换的组件。

    Args:
        layer_type: 瓶颈层的类型名 (Conv2d, MultiheadAttention, etc.)
        layer_params: 该层参数量
        layer_flops: 该层 FLOPs
        context: 额外上下文（如模型类型、任务类型）

    Returns:
        匹配的组件列表，按推荐度排序
    """
    matches = []
    for comp in COMPONENT_LIBRARY:
        score = 0
        # 类型匹配
        for target in comp.get("replaces", []):
            if target.lower() in layer_type.lower() or layer_type.lower() == target.lower():
                score += 3
                break

        # 高参数层匹配高节省组件
        if layer_params > 1_000_000 and comp.get("params_saving") == "高":
            score += 2
        elif layer_params > 100_000 and comp.get("params_saving") in ("高", "中"):
            score += 1

        # 高 FLOPs 层匹配 FLOPs 节省组件
        if layer_flops > 1_000_000 and comp.get("flops_saving") == "高":
            score += 2
        elif layer_flops > 100_000 and comp.get("flops_saving") in ("高", "中"):
            score += 1

        # 上下文匹配
        if context:
            task = context.get("task_type", "")
            if task in ("detection", "segmentation") and any(
                t in comp.get("tags", []) for t in ("检测", "分割", "空间")
            ):
                score += 1

            num_layers = context.get("num_layers", 0)
            if num_layers >= 12 and "深层" in comp.get("when_to_use", ""):
                score += 1

        if score > 0:
            matches.append({**comp, "_score": score})

    matches.sort(key=lambda x: x["_score"], reverse=True)
    return matches


def get_all_component_names() -> list[str]:
    """返回所有可用组件名。"""
    return [c["name"] for c in COMPONENT_LIBRARY]
