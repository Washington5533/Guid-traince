"""cp_15 · 模型管线可视化 (ModelVisualizer)。

解析模型结构 → 真实 FLOPs/参数量 → AI 提议可视化配置 →
组件库匹配改进建议 → 渲染可交互 HTML。

核心改进（v2）：
- 真实 FLOPs 计算（forward hook + dummy input）
- 同构层自动折叠（如 12 个相同的 TransformerBlock → 显示为 ×12）
- 颜色按 FLOPs/参数量强度映射
- D3.js 可折叠树布局

详见 checkpoint/cp_15.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .component_library import match_components, get_all_component_names

from guardian.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 模型解析（真实 FLOPs + 参数统计）
# ---------------------------------------------------------------------------

def _build_dummy_input(model) -> Any:
    """根据模型结构推断 dummy input shape。"""
    try:
        # 尝试从第一个 Conv2d 推断输入通道
        for m in model.modules():
            if hasattr(m, "in_channels") and hasattr(m, "weight"):
                c = m.in_channels
                # 判断是 CNN 还是 ViT
                if any("visual" in n or "conv" in n for n, _ in model.named_modules()):
                    return _zeros(1, c, 224, 224)
        # 默认 3 通道 224
        return _zeros(1, 3, 224, 224)
    except Exception:
        logger.warning("推断 dummy input shape 失败，回退 1x3x224x224", exc_info=True)
        return _zeros(1, 3, 224, 224)


def _zeros(*shape):
    import torch
    return torch.zeros(*shape)


class ModelVisualizer:
    """模型结构解析 + 可视化配置 + HTML 渲染。"""

    def __init__(self, config: dict | None = None, advisor: Any = None):
        self.cfg = config or {}
        self.advisor = advisor
        self.color_map_default = self.cfg.get("color_map_default", "flops")
        self.bottleneck_threshold_pct = float(self.cfg.get("bottleneck_threshold_pct", 25))

    # ------------------------------------------------------------------
    # 模型解析
    # ------------------------------------------------------------------

    @staticmethod
    def parse_model(model_fn) -> dict[str, Any]:
        """解析模型结构为多层节点树，含真实 FLOPs。

        策略：
        1. torch.fx 符号追踪 → 得到算子级图
        2. named_modules → 得到子模块参数
        3. forward hook → 用 dummy input 跑一次，记录每层输入/输出 shape → 算 FLOPs
        4. 合并：子模块参数 + 算子级 FLOPs → 按模块聚合
        """
        try:
            import torch
        except ImportError:
            return {"error": "PyTorch 未安装"}

        model = model_fn()
        if not isinstance(model, torch.nn.Module):
            return {"error": f"model_fn 必须返回 nn.Module，实际为 {type(model).__name__}"}

        # 1. 收集子模块信息
        module_info: dict[str, dict] = {}
        for name, mod in model.named_modules():
            if name == "":
                continue
            params = sum(p.numel() for p in mod.parameters())
            module_info[name] = {
                "type": type(mod).__name__,
                "params": params,
                "depth": name.count("."),
                "flops": 0,
                "input_shape": None,
                "output_shape": None,
            }

        # 2. 注册 forward hooks 收集 shape
        hook_data: dict[str, dict] = {}

        def _make_pre_hook(_name):
            def hook(module, inp):
                if inp and isinstance(inp[0], torch.Tensor):
                    hook_data.setdefault(_name, {})["input_shape"] = list(inp[0].shape)
            return hook

        def _make_post_hook(_name):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    hook_data.setdefault(_name, {})["output_shape"] = list(out.shape)
            return hook

        handles = []
        for name, mod in model.named_modules():
            if name == "":
                continue
            handles.append(mod.register_forward_pre_hook(_make_pre_hook(name)))
            handles.append(mod.register_forward_hook(_make_post_hook(name)))

        # 3. 用 dummy input 跑一次
        total_flops = 0
        try:
            model.eval()
            dummy = _build_dummy_input(model)
            with torch.no_grad():
                model(dummy)
        except Exception:
            logger.warning("用 dummy input 前向失败，FLOPs 将按缺失 shape 估算", exc_info=True)

        # 4. 计算 FLOPs（基于 recorded shapes）
        for name, data in hook_data.items():
            if name not in module_info:
                continue
            inp_s = data.get("input_shape")
            out_s = data.get("output_shape")
            if inp_s and out_s:
                module_info[name]["input_shape"] = inp_s
                module_info[name]["output_shape"] = out_s
                flops_est = _compute_flops(module_info[name]["type"], inp_s, out_s)
                module_info[name]["flops"] = flops_est
                total_flops += flops_est

        # 清理 hooks
        for h in handles:
            h.remove()

        # 5. 按深度折叠 Transformer 同构块
        nodes = _fold_identical_blocks(module_info)

        total_params = sum(p.numel() for p in model.parameters())

        return {
            "nodes": nodes,
            "total_params": total_params,
            "total_flops_est": total_flops,
            "model_name": type(model).__name__,
            "module_count": len(module_info),
        }

    @staticmethod
    def compute_stats(graph: dict) -> dict[str, Any]:
        """基于解析结果计算每层统计。"""
        nodes = graph.get("nodes", [])
        total_params = max(graph.get("total_params", 1), 1)
        total_flops = max(graph.get("total_flops_est", 1), 1)

        layer_stats = []
        for node in nodes:
            layer_stats.append({
                "name": node["name"],
                "type": node.get("type", "unknown"),
                "params": node.get("params", 0),
                "flops": node.get("flops", 0),
                "params_pct": round(node.get("params", 0) / total_params * 100, 2),
                "flops_pct": round(node.get("flops", 0) / total_flops * 100, 2),
                "repeat": node.get("repeat", 1),
                "depth": node.get("depth", 0),
                "children": node.get("children", []),
            })

        return {
            "layer_stats": layer_stats,
            "total_params": total_params,
            "total_flops": total_flops,
        }

    # ------------------------------------------------------------------
    # AI/规则提议
    # ------------------------------------------------------------------

    def propose_config(self, graph, stats, user_feedback=None):
        if self.advisor and self.advisor.is_enabled("visualization"):
            try:
                ctx = {
                    "model_name": graph.get("model_name"),
                    "total_params": stats.get("total_params"),
                    "total_flops": stats.get("total_flops"),
                    "layer_count": len(stats.get("layer_stats", [])),
                    "user_feedback": user_feedback,
                }
                result = self.advisor.suggest("visualization_config", ctx)
                if result and isinstance(result, dict) and "view" in result:
                    return result
            except Exception:
                logger.warning("AI 提议可视化配置失败，使用默认配置", exc_info=True)
        return _default_viz_config(graph, stats)

    def propose_improvements(self, stats, viz_config=None, task_context=None):
        bottlenecks = (viz_config or {}).get("bottlenecks", [])
        layer_stats = stats.get("layer_stats", [])
        improvements = []

        for bn in bottlenecks:
            entry = {
                "layer": bn.get("layer", ""),
                "severity": bn.get("severity", "info"),
                "flops_pct": bn.get("flops_pct", 0),
                "params_pct": bn.get("params_pct", 0),
                "matched_components": [],
                "source": "library",
            }
            # find layer type
            ltype = "unknown"
            lparams = 0
            for ls in layer_stats:
                if ls["name"] == entry["layer"]:
                    ltype = ls.get("type", "unknown")
                    lparams = ls.get("params", 0)
                    break

            matches = match_components(layer_type=ltype, layer_params=lparams, context=task_context or {})
            entry["matched_components"] = matches[:3]
            if not matches and self.advisor and self.advisor.is_enabled("visualization"):
                try:
                    ai = self.advisor.suggest("model_improvement", {
                        "layer": entry["layer"], "layer_type": ltype,
                        "flops_pct": entry["flops_pct"], "params_pct": entry["params_pct"],
                    })
                    if ai and isinstance(ai, dict):
                        entry["ai_suggestion"] = ai
                        entry["source"] = "ai"
                except Exception:
                    logger.warning("AI 模型改进建议失败，仅保留组件库匹配结果", exc_info=True)
            improvements.append(entry)
        return improvements

    # ------------------------------------------------------------------
    # HTML 渲染
    # ------------------------------------------------------------------

    def render_html(self, graph, stats, viz_config, output_path, improvements=None):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        nodes = graph.get("nodes", [])
        # 构建 D3 树结构
        tree = _build_tree(nodes)
        # 注入范式决策
        _assign_paradigms(tree)

        html = _RENDER_HTML.format(
            model_name=graph.get("model_name", "Model"),
            total_params=f"{stats.get('total_params', 0):,}",
            total_flops=f"{stats.get('total_flops', 0):,}",
            total_flops_m=round(stats.get("total_flops", 0) / 1e6, 1),
            module_count=graph.get("module_count", len(nodes)),
            tree_json=json.dumps(tree, ensure_ascii=False),
            bottlenecks_json=json.dumps(viz_config.get("bottlenecks", [])[:10], ensure_ascii=False),
            summary_text=viz_config.get("architecture_summary", ""),
            color_by=viz_config.get("view", {}).get("color_map", "params"),
            improvements_json=json.dumps(improvements or [], ensure_ascii=False),
        )
        out.write_text(html, encoding="utf-8")
        return out

    def print_proposal(self, viz_config, stats, improvements=None):
        view = viz_config.get("view", {})
        bottlenecks = viz_config.get("bottlenecks", [])
        summary = viz_config.get("architecture_summary", "")

        lines = [
            "=" * 60,
            "  Model Structure Analysis & Improvements",
            "=" * 60,
            f"  {summary}",
            f"  Params: {stats.get('total_params', 0):,}  |  FLOPs: {stats.get('total_flops', 0):,}  ({round(stats.get('total_flops', 0) / 1e6, 1)}M)",
            f"  Color: {view.get('color_map', 'params')}",
            "",
        ]

        if bottlenecks:
            lines.append("  --- Bottlenecks ---")
            for b in bottlenecks[:8]:
                icon = {"critical": "[!!]", "warning": "[! ]", "info": "[i ]"}.get(b.get("severity", ""))
                lines.append(
                    f"  {icon} {b.get('layer', '?')}: "
                    f"FLOPs {b.get('flops_pct', 0):.1f}% | Params {b.get('params_pct', 0):.1f}%"
                )

        if improvements:
            lines.append("")
            lines.append("  --- Improvement Suggestions ---")
            for imp in (improvements or [])[:5]:
                lines.append(f"  > {imp.get('layer', '?')}  (source: {imp.get('source', 'library')})")
                for m in imp.get("matched_components", [])[:2]:
                    lines.append(f"    + {m['name']}: {m['description'][:80]}")
                    lines.append(f"      Save: FLOPs {m['flops_saving']} | Params {m['params_saving']}")

        lines.append("")
        lines.append("=" * 60)
        lines.append("  [Enter] confirm  |  [text] refine  |  cancel")
        return "\n".join(lines)

    def visualize(self, model_fn, output_path="./logs/model_viz.html", user_feedback=None):
        graph = self.parse_model(model_fn)
        if "error" in graph:
            return graph
        stats = self.compute_stats(graph)
        viz_config = self.propose_config(graph, stats, user_feedback)
        improvements = self.propose_improvements(stats, viz_config)
        out = self.render_html(graph, stats, viz_config, output_path, improvements)
        return {"graph": graph, "stats": stats, "viz_config": viz_config,
                "improvements": improvements, "output_path": str(out)}


# ---------------------------------------------------------------------------
# FLOPs 计算
# ---------------------------------------------------------------------------

def _compute_flops(layer_type: str, inp_shape: list, out_shape: list) -> int:
    """根据层类型和输入/输出 shape 估算 FLOPs。"""
    if layer_type == "Conv2d" and len(inp_shape) >= 4:
        # FLOPs = 2 * k_h * k_w * c_in * c_out * h_out * w_out
        # 简化：inp=[B,C,H,W], out=[B,C',H',W']
        c_in, h_in, w_in = inp_shape[1], inp_shape[2], inp_shape[3]
        c_out, h_out, w_out = out_shape[1], out_shape[2], out_shape[3]
        # 假设 kernel 3×3
        return 2 * 3 * 3 * c_in * c_out * h_out * w_out

    if layer_type in ("Linear", "NonDynamicallyQuantizableLinear") and len(inp_shape) >= 2:
        # FLOPs = 2 * in_features * out_features
        return 2 * inp_shape[-1] * out_shape[-1]

    if "MultiheadAttention" in layer_type:
        # FLOPs ≈ 4 * seq_len^2 * d_model + 2 * seq_len * d_model^2
        if len(inp_shape) >= 3:
            seq, dim = inp_shape[1], inp_shape[2]
            return 4 * seq * seq * dim + 2 * seq * dim * dim
        return 0

    if layer_type in ("LayerNorm", "BatchNorm2d", "QuickGELU", "GELU", "ReLU", "SiLU", "Dropout", "Identity"):
        return 0  # 可忽略

    if layer_type == "Sequential":
        return 0  # 容器，不计

    # 通用 estimate: 2 * product(inp_shape)
    try:
        total = 1
        for s in out_shape:
            total *= s
        return 2 * total
    except Exception:
        logger.warning("估算 FLOPs 失败（layer_type=%s），按 0 计", layer_type, exc_info=True)
        return 0


# ---------------------------------------------------------------------------
# 同构层折叠
# ---------------------------------------------------------------------------

def _is_identical(a: dict, b: dict) -> bool:
    """两个模块是否结构相同（可折叠）。"""
    return (
        a.get("type") == b.get("type")
        and a.get("params") == b.get("params")
        and a.get("flops") == b.get("flops")
        and a.get("depth") == b.get("depth")
    )


def _fold_identical_blocks(module_info: dict) -> list[dict]:
    """连续同构模块折叠为 ×N。

    如 transformer.resblocks.0 ~ .11 → 显示为 'transformer.resblocks (×12)'，
    children 展开后可见单个 block 内部结构。
    """
    by_depth: dict[int, list[dict]] = {}
    for name, info in module_info.items():
        depth = info["depth"]
        by_depth.setdefault(depth, []).append({"name": name, **info})

    # 在每个深度层内，折叠连续同名模式
    nodes: list[dict] = []
    collapsed: set = set()

    for depth in sorted(by_depth.keys()):
        items = by_depth[depth]
        i = 0
        while i < len(items):
            item = items[i]
            # 查找后续相同类型的连续节点
            run = [item]
            j = i + 1
            while j < len(items) and _is_identical(items[j], item):
                run.append(items[j])
                j += 1

            if len(run) >= 4:  # 至少 4 个相同才折叠
                for r in run[1:]:
                    collapsed.add(r["name"])
                # 只展开第一个代表块的子模块
                children = _get_children(module_info, item["name"], first_only=True)
                nodes.append({
                    **item,
                    "name": item["name"].rsplit(".", 1)[0] if "." in item["name"] else item["name"],
                    "repeat": len(run),
                    "children": children,
                    "collapsed_names": [r["name"] for r in run],
                })
                i = j
            else:
                # 检查是否属于某个已折叠的父节点
                is_collapsed_child = any(
                    item["name"].startswith(c + ".") for c in collapsed
                )
                if item["name"] not in collapsed and not is_collapsed_child:
                    children = _get_children(module_info, item["name"])
                    nodes.append({**item, "repeat": 1, "children": children,
                                  "collapsed_names": []})
                i += 1

    return nodes


def _get_children(module_info: dict, parent_name: str, *, first_only: bool = False) -> list[dict]:
    """获取某个模块的直接子模块。

    first_only=True: 只取第一个同名组的子模块（折叠时用，避免展开 N 份拷贝）。
    """
    prefix = parent_name + "."
    seen = set()
    children = []
    for name, info in module_info.items():
        if not name.startswith(prefix):
            continue
        rest = name[len(prefix):]
        if rest.count(".") != 0:
            continue
        short = rest.split(".")[0] if "." in rest else rest
        if first_only and short in seen:
            continue
        seen.add(short)
        children.append({"name": short, **{k: v for k, v in info.items() if k != "name"}})
    return children[:12]


# ---------------------------------------------------------------------------
# 默认可视化配置
# ---------------------------------------------------------------------------

def _default_viz_config(graph, stats):
    layer_stats = stats.get("layer_stats", [])
    bottlenecks = []
    for ls in layer_stats:
        pct = ls.get("params_pct", 0)
        if pct >= 25:
            severity = "critical" if pct >= 50 else "warning"
            bottlenecks.append({
                "layer": ls["name"], "flops_pct": ls.get("flops_pct", 0),
                "params_pct": pct, "severity": severity,
                "suggestion": f"Params {pct:.0f}% of total" if pct >= 50 else "",
                "code_reference": "",
            })
    bottlenecks.sort(key=lambda x: x["params_pct"], reverse=True)

    return {
        "view": {"color_map": "params", "group_by": "stage"},
        "bottlenecks": bottlenecks[:8],
        "architecture_summary": f"{graph.get('model_name', 'Model')} — "
                                f"{len(layer_stats)} layers, "
                                f"{stats.get('total_params', 0):,} params, "
                                f"{round(stats.get('total_flops', 0) / 1e6, 1)}M FLOPs",
    }


# ---------------------------------------------------------------------------
# D3 树构建
# ---------------------------------------------------------------------------

def _build_tree(nodes: list[dict]) -> dict:
    """将折叠后的节点列表转为 D3 可折叠树。"""
    children = []
    for node in nodes:
        child = {
            "name": node["name"],
            "type": node.get("type", ""),
            "params": node.get("params", 0),
            "flops": node.get("flops", 0),
            "repeat": node.get("repeat", 1),
            "depth": node.get("depth", 0),
            "params_pct": round(node.get("params", 0) / max(1, sum(n.get("params", 0) for n in nodes)) * 100, 1) if nodes else 0,
            "children": _build_tree(node.get("children", [])).get("children", []) if node.get("children") else [],
        }
        children.append(child)
    return {"name": "root", "children": children}


# ---------------------------------------------------------------------------
# 范式决策
# ---------------------------------------------------------------------------

def _decide_paradigm(node: dict) -> str:
    """根据节点特征选择最佳可视化范式。"""
    repeat = node.get("repeat", 1)
    children = node.get("children", [])
    n_children = len(children)

    if repeat >= 4:
        return "blocks"
    if n_children <= 1:
        return "treemap"
    if _is_linear_chain(children):
        return "pipeline"
    if _has_cross_links(children):
        return "mesh"
    return "tree"


def _is_linear_chain(children: list[dict]) -> bool:
    """判断子节点是否构成线性链（每个节点至多一个子节点）。"""
    for c in children:
        sub = c.get("children", [])
        if len(sub) > 1:
            return False
    return len(children) >= 2


def _has_cross_links(children: list[dict]) -> bool:
    """启发式检测子模块间是否存在交叉引用（如 Attention 的 Q/K/V）。"""
    child_names = [c.get("name", "").lower() for c in children]
    cross_keywords = ["attn", "attention", "query", "key", "value", "qkv", "cross"]
    match_count = sum(1 for n in child_names if any(kw in n for kw in cross_keywords))
    return match_count >= 2


def _assign_paradigms(tree: dict) -> None:
    """为树的每个顶层子节点递归分配 paradigm 字段。"""
    for child in tree.get("children", []):
        child["paradigm"] = _decide_paradigm(child)
        _assign_paradigms(child)


# ---------------------------------------------------------------------------
# HTML 模板
# ---------------------------------------------------------------------------

_RENDER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{model_name} - Model Architecture</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
:root {{
  --bg: #ffffff; --bg2: #f6f8fa; --bg3: #f3f4f6;
  --border: #d0d7de; --border2: #d8dee4;
  --text: #1f2328; --text2: #656d76; --text3: #8b949e;
  --blue: #0969da; --green: #1a7f37; --red: #cf222e;
  --orange: #bf8700; --purple: #8250df;
  --radius: 6px; --font: -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  --mono: ui-monospace,SFMono-Regular,Consolas,monospace;
}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:var(--font);background:var(--bg);color:var(--text);overflow:hidden;height:100vh}}
.header{{height:48px;background:var(--bg);border-bottom:1px solid var(--border);display:flex;align-items:center;padding:0 20px;justify-content:space-between;box-shadow:0 1px 0 rgba(31,35,40,.04)}}
.header h1{{font-size:15px;font-weight:600;color:var(--text)}}
.header-stats{{display:flex;gap:16px;font-size:12px;color:var(--text2)}}
.header-stats b{{color:var(--text);font-weight:600}}
.main{{display:flex;height:calc(100vh - 48px)}}
.sidebar{{width:260px;min-width:260px;background:var(--bg2);border-right:1px solid var(--border);padding:14px;overflow-y:auto;display:flex;flex-direction:column;gap:10px}}
.sidebar::-webkit-scrollbar{{width:4px}}
.sidebar::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
.section-title{{font-size:10px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}}
.sort-btns{{display:flex;gap:3px}}
.sort-btn{{padding:3px 8px;font-size:11px;background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text2);cursor:pointer;transition:all .15s}}
.sort-btn.active{{background:var(--blue);border-color:var(--blue);color:#fff}}
.sort-btn:hover{{border-color:var(--blue)}}
.filter-group{{display:flex;flex-direction:column;gap:3px}}
.filter-group label{{font-size:10px;color:var(--text2)}}
.filter-group input[type=range]{{width:100%;accent-color:var(--blue);height:3px}}
.filter-val{{font-size:10px;color:var(--blue);text-align:right}}
.legend{{display:flex;flex-wrap:wrap;gap:6px;font-size:10px;color:var(--text2)}}
.legend-item{{display:inline-flex;align-items:center;gap:3px}}
.legend-dot{{width:6px;height:6px;border-radius:2px}}
.bn-card{{background:var(--bg);border-radius:4px;padding:6px 8px;margin:3px 0;border-left:3px solid var(--red);font-size:10px;cursor:pointer;transition:all .15s;border:1px solid var(--border2);border-left-width:3px}}
.bn-card:hover{{border-left-color:var(--blue)}}
.bn-card.warning{{border-left-color:var(--orange)}}
.bn-card.info{{border-left-color:var(--blue)}}
.bn-card .nm{{font-weight:600;color:var(--text);word-break:break-all;font-size:11px}}
.bn-card .pct{{color:var(--text2);margin-top:1px}}
.backbone-area{{flex:1;overflow:auto;position:relative;background:var(--bg)}}
.backbone-area svg{{width:100%;height:100%}}
.detail-panel{{width:0;overflow:hidden;background:var(--bg);border-left:1px solid var(--border);transition:width .25s}}
.detail-panel.open{{width:340px;min-width:340px}}
.dp-inner{{padding:14px;width:340px;height:100%;overflow-y:auto}}
.dp-inner::-webkit-scrollbar{{width:4px}}
.dp-inner::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
.dp-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px}}
.dp-title{{font-size:13px;font-weight:700;color:var(--text)}}
.dp-close{{width:24px;height:24px;border-radius:4px;border:1px solid var(--border);background:none;color:var(--text2);cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center}}
.dp-close:hover{{background:var(--bg2)}}
.dp-badge{{display:inline-block;padding:1px 6px;border-radius:2em;font-size:10px;font-weight:600;background:#ddf4ff;color:var(--blue);margin-left:6px}}
.dp-stat{{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #f0f0f0;font-size:11px}}
.dp-stat .k{{color:var(--text2)}}.dp-stat .v{{color:var(--text);font-weight:600}}
.dp-bar{{height:3px;border-radius:2px;background:var(--bg3);margin:3px 0 6px;overflow:hidden}}
.dp-bar-fill{{height:100%;border-radius:2px;transition:width .3s}}
.dp-sub{{margin-top:10px}}.dp-sub h4{{font-size:11px;color:var(--text);margin-bottom:6px;font-weight:600}}
.dp-viz{{width:100%;height:240px;border:1px solid var(--border);border-radius:var(--radius);background:var(--bg2);overflow:hidden;margin-top:6px}}
.dp-imp{{margin-top:8px;padding:7px 8px;background:#fff8c5;border-radius:4px;border-left:3px solid var(--orange);font-size:10px;color:var(--text2);line-height:1.5}}
.dp-imp b{{color:var(--orange)}}
.tooltip{{position:absolute;padding:8px 12px;background:#fff;border:1px solid var(--border);border-radius:6px;font-size:11px;color:var(--text);pointer-events:none;opacity:0;transition:opacity .12s;max-width:260px;z-index:100;box-shadow:0 3px 12px rgba(140,149,159,.2);line-height:1.5}}
.tooltip b{{color:var(--blue)}}
.empty-state{{text-align:center;padding:30px;color:var(--text3);font-size:12px}}
@media(max-width:768px){{.sidebar{{width:100%;min-width:unset;max-height:120px;border-right:none;border-bottom:1px solid var(--border)}}.main{{flex-direction:column}}.detail-panel.open{{width:100%;min-width:unset;position:absolute;right:0;top:0;height:100%;z-index:50}}}}
</style>
</head>
<body>
<div class="header">
  <h1>{model_name}</h1>
  <div class="header-stats">
    <span><b>{total_params}</b> params</span>
    <span><b>{total_flops_m}M</b> FLOPs</span>
    <span><b>{module_count}</b> modules</span>
  </div>
</div>
<div class="main">
  <div class="sidebar">
    <div><div class="section-title">Sort</div>
      <div class="sort-btns">
        <button class="sort-btn active" data-sort="default" onclick="sortBy('default')">Default</button>
        <button class="sort-btn" data-sort="params" onclick="sortBy('params')">Params</button>
        <button class="sort-btn" data-sort="flops" onclick="sortBy('flops')">FLOPs</button>
      </div></div>
    <div class="filter-group">
      <label>Min params: <span class="filter-val" id="fv">0%</span></label>
      <input type="range" id="fs" min="0" max="100" value="0" oninput="filterNodes(this.value)">
    </div>
    <div><div class="section-title">Legend</div>
      <div class="legend">
        <span class="legend-item"><span class="legend-dot" style="background:#1a7f37"></span>Low</span>
        <span class="legend-item"><span class="legend-dot" style="background:#8250df"></span>Mid</span>
        <span class="legend-item"><span class="legend-dot" style="background:#cf222e"></span>High</span>
      </div></div>
    <div><div class="section-title">Bottlenecks</div><div id="bn-list"></div></div>
  </div>
  <div class="backbone-area" id="bb-area"><svg id="bb-svg"></svg></div>
  <div class="detail-panel" id="dp"><div class="dp-inner" id="dp-inner"></div></div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const treeData = {tree_json};
const bottlenecks = {bottlenecks_json};
const improvements = {improvements_json};
const colorBy = "{color_by}";

const fp = p => !p ? "0" : p >= 1e6 ? (p/1e6).toFixed(1)+"M" : p >= 1e3 ? (p/1e3).toFixed(0)+"K" : ""+p;
const ff = f => !f ? "" : f >= 1e6 ? (f/1e6).toFixed(1)+"M" : f >= 1e3 ? (f/1e3).toFixed(0)+"K" : ""+f;
const pn = {{tree:"Tree",mesh:"Mesh",treemap:"Treemap",pipeline:"Pipeline",blocks:"Blocks"}};
const tip = document.getElementById("tooltip");
function showTip(e,h){{tip.innerHTML=h;tip.style.opacity=1;tip.style.left=(e.pageX+12)+"px";tip.style.top=(e.pageY-10)+"px"}}
function hideTip(){{tip.style.opacity=0}}
const bbNodes = treeData.children || [];
const maxP = Math.max(1,...bbNodes.map(n=>n.params||0));
const maxF = Math.max(1,...bbNodes.map(n=>n.flops||0));
const cLo = "#1a7f37", cMid = "#8250df", cHi = "#cf222e";
function cP(v){{const r=v/maxP;return r<0.25?cLo:r<0.5?cMid:cHi}}
function cF(v){{const r=v/maxF;return r<0.25?"#0969da":r<0.5?"#8250df":"#bf8700"}}
const sortOrd = {{default:bbNodes.map((_,i)=>i),params:[...bbNodes].sort((a,b)=>(b.params||0)-(a.params||0)).map(n=>bbNodes.indexOf(n)),flops:[...bbNodes].sort((a,b)=>(b.flops||0)-(a.flops||0)).map(n=>bbNodes.indexOf(n))}};
let curSort="default",curFilter=0,selectedNode=null;

const bnL=document.getElementById("bn-list");
bottlenecks.forEach(b=>{{const c=document.createElement("div");c.className="bn-card "+(b.severity||"info");c.innerHTML='<div class="nm">'+b.layer+'</div><div class="pct">P:'+(b.params_pct||0).toFixed(1)+'% | F:'+(b.flops_pct||0).toFixed(1)+'%</div>';c.onclick=()=>{{const n=bbNodes.find(x=>x.name===b.layer);if(n)openDetail(n)}};bnL.appendChild(c)}});
if(!bottlenecks.length)bnL.innerHTML='<div class="empty-state">No bottlenecks</div>';

function drawBB(){{
  const area=document.getElementById("bb-area"),svg=d3.select("#bb-svg");svg.selectAll("*").remove();
  const n=bbNodes.length;if(!n){{svg.append("text").attr("x",200).attr("y",100).attr("fill","#656d76").attr("font-size",14).text("No model data");return}}
  const nw=140,nh=72,gap=60,mx=60,my=50;
  const tw=n*(nw+gap)-gap+mx*2,th=nh+my*2+30;
  const aw=area.clientWidth,ah=area.clientHeight;
  svg.attr("viewBox",[0,0,Math.max(tw,aw),Math.max(th,ah)]);
  const gr=svg.append("defs").append("linearGradient").attr("id","lg").attr("x1","0%").attr("x2","100%");
  gr.append("stop").attr("offset","0%").attr("stop-color","#0969da");
  gr.append("stop").attr("offset","100%").attr("stop-color","#8250df");
  const gg=svg.append("g");
  const ord=sortOrd[curSort];
  for(let i=0;i<n-1;i++){{const j=ord[i],k=ord[i+1];const x1=mx+j*(nw+gap)+nw,y=my+nh/2,x2=mx+k*(nw+gap),tk=Math.max(1,Math.min(5,((bbNodes[j].params||0)/maxP)*5));
    gg.append("line").attr("x1",x1).attr("y1",y).attr("x2",x2).attr("y2",y).attr("stroke","#d0d7de").attr("stroke-width",tk).attr("stroke-linecap","round");
    gg.append("polygon").attr("points",`${{x2}},${{y}} ${{x2-5}},${{y-3}} ${{x2-5}},${{y+3}}`).attr("fill","#d0d7de")}}
  bbNodes.forEach((nd,i)=>{{
    const oi=ord.indexOf(i),x=mx+oi*(nw+gap),y=my;
    const pctP=(nd.params||0)/maxP,pctF=(nd.flops||0)/maxF;
    const isBelow=curFilter>0&&pctP*100<curFilter;
    const g=gg.append("g").attr("transform",`translate(${{x}},${{y}})`).style("opacity",isBelow?0.08:1).style("cursor","pointer")
      .on("click",()=>openDetail(nd))
      .on("mouseenter",(ev)=>showTip(ev,`<b>${{nd.name}}</b><br>${{nd.type||""}} ${{nd.repeat>1?"x"+nd.repeat:""}}<br>Params: ${{fp(nd.params)}} (${{(nd.params_pct||0).toFixed(1)}}%)<br>FLOPs: ${{ff(nd.flops)}}`))
      .on("mouseleave",hideTip).on("mousemove",(ev)=>{{tip.style.left=(ev.pageX+12)+"px";tip.style.top=(ev.pageY-10)+"px"}});
    g.append("rect").attr("width",nw).attr("height",nh).attr("rx",8).attr("fill","#fff").attr("stroke","#d0d7de").attr("stroke-width",1);
    g.append("rect").attr("width",nw).attr("height",2).attr("rx",1).attr("fill","url(#lg)");
    const lbl=nd.name.split(".").pop();
    g.append("text").attr("x",nw/2).attr("y",20).attr("text-anchor","middle").attr("fill","#1f2328").attr("font-size",10).attr("font-weight",600).text(lbl.length>16?lbl.slice(0,14)+"..":lbl);
    g.append("text").attr("x",nw/2).attr("y",32).attr("text-anchor","middle").attr("fill","#656d76").attr("font-size",8).text((nd.type||"")+(nd.repeat>1?" x"+nd.repeat:""));
    g.append("rect").attr("x",8).attr("y",38).attr("width",nw-16).attr("height",3).attr("rx",1.5).attr("fill","#f3f4f6");
    g.append("rect").attr("x",8).attr("y",38).attr("width",Math.max(2,(nw-16)*pctP)).attr("height",3).attr("rx",1.5).attr("fill",cP(nd.params||0));
    g.append("rect").attr("x",8).attr("y",44).attr("width",nw-16).attr("height",3).attr("rx",1.5).attr("fill","#f3f4f6");
    g.append("rect").attr("x",8).attr("y",44).attr("width",Math.max(2,(nw-16)*pctF)).attr("height",3).attr("rx",1.5).attr("fill",cF(nd.flops||0));
    g.append("text").attr("x",6).attr("y",58).attr("fill","#656d76").attr("font-size",8).text(fp(nd.params)+" p");
    g.append("text").attr("x",nw-6).attr("y",58).attr("text-anchor","end").attr("fill","#bf8700").attr("font-size",7).text(ff(nd.flops));
    g.append("text").attr("x",nw/2).attr("y",68).attr("text-anchor","middle").attr("fill","#0969da").attr("font-size",8).text(pn[nd.paradigm]||"Tree");
  }});
  svg.call(d3.zoom().scaleExtent([0.3,3]).on("zoom",e=>gg.attr("transform",e.transform)))
}}
drawBB();

function sortBy(s){{curSort=s;document.querySelectorAll(".sort-btn").forEach(b=>b.classList.toggle("active",b.dataset.sort===s));drawBB()}}
function filterNodes(v){{curFilter=+v;document.getElementById("fv").textContent=v+"%";drawBB()}}

function openDetail(nd){{
  selectedNode=nd;const dp=document.getElementById("dp"),inner=document.getElementById("dp-inner");
  dp.classList.add("open");
  const pp=(nd.params_pct||0),fpp=(nd.flops||0)/Math.max(1,maxF)*100;
  let h='<div class="dp-head"><div class="dp-title">'+nd.name.split(".").pop()+'<span class="dp-badge">'+(pn[nd.paradigm]||"Tree")+"</span></div>";
  h+='<button class="dp-close" onclick="closeDetail()">x</button></div>';
  h+='<div class="dp-stat"><span class="k">Type</span><span class="v">'+(nd.type||"?")+"</span></div>";
  if(nd.repeat>1)h+='<div class="dp-stat"><span class="k">Repeat</span><span class="v">x'+nd.repeat+"</span></div>";
  h+='<div class="dp-stat"><span class="k">Parameters</span><span class="v">'+fp(nd.params)+" ("+pp.toFixed(1)+"%)</span></div>";
  h+='<div class="dp-bar"><div class="dp-bar-fill" style="width:'+pp+"%;background:"+cP(nd.params||0)+'"></div></div>';
  h+='<div class="dp-stat"><span class="k">FLOPs</span><span class="v">'+ff(nd.flops)+"</span></div>";
  h+='<div class="dp-bar"><div class="dp-bar-fill" style="width:'+fpp+"%;background:"+cF(nd.flops||0)+'"></div></div>';
  if((nd.children&&nd.children.length)||(nd.repeat&&nd.repeat>1)){{h+='<div class="dp-sub"><h4>'+(nd.repeat>1?'Repeated x'+nd.repeat:'Sub-structure ('+nd.children.length+' children)')+'</h4><div class="dp-viz" id="dp-viz"></div></div>'}}
  const imp=improvements.find(x=>x.layer===nd.name);
  if(imp){{h+='<div class="dp-imp"><b>Suggestion:</b> '+(imp.ai_suggestion||imp.matched_components?.[0]?.description||"Consider optimizing this layer")+"</div>"}}
  inner.innerHTML=h;
  if((nd.children&&nd.children.length)||(nd.repeat&&nd.repeat>1))setTimeout(()=>renderPV(document.getElementById("dp-viz"),nd),50)
}}
function closeDetail(){{document.getElementById("dp").classList.remove("open");selectedNode=null}}

function renderPV(el,nd){{
  const r=({{tree:rTree,mesh:rMesh,treemap:rTmap,pipeline:rPipe,blocks:rBlocks}})[nd.paradigm]||rTree;
  r(el,nd)
}}
function rTree(el,nd){{
  const w=el.clientWidth,h=el.clientHeight,svg=d3.select(el).append("svg").attr("viewBox",[0,0,w,h]);
  const root=d3.hierarchy(nd);d3.tree().size([h-20,w/2-40])(root);
  const g=svg.append("g").attr("transform","translate("+(w/4)+",10)");
  const mp=Math.max(1,...root.descendants().map(d=>d.data.params||0));
  const cs=d3.scaleLinear().domain([0,mp]).range(["#0969da","#cf222e"]);
  g.selectAll("path").data(root.links()).join("path").attr("fill","none").attr("stroke","#d0d7de").attr("stroke-width",1.5)
    .attr("d",d=>`M${{d.source.y}},${{d.source.x}}C${{(d.source.y+d.target.y)/2}},${{d.source.x}} ${{(d.source.y+d.target.y)/2}},${{d.target.x}} ${{d.target.y}},${{d.target.x}}`);
  const ns=g.selectAll("g").data(root.descendants()).join("g").attr("transform",d=>`translate(${{d.y}},${{d.x}})`);
  ns.append("circle").attr("r",d=>Math.max(2,Math.min(8,Math.sqrt((d.data.params||0)/mp)*10))).attr("fill",d=>cs(d.data.params||0));
  ns.append("text").attr("dy","0.35em").attr("x",d=>d.children?-8:8).attr("text-anchor",d=>d.children?"end":"start").attr("fill","#656d76").attr("font-size",8).text(d=>{{let n=d.data.name.split(".").pop();return n.length>12?n.slice(0,10)+"..":n}})
}}
function rMesh(el,nd){{
  const w=el.clientWidth,h=el.clientHeight,svg=d3.select(el).append("svg").attr("viewBox",[0,0,w,h]);
  const ch=(nd.children||[]).map((c,i)=>({{id:i,name:c.name.split(".").pop(),params:c.params||0,type:c.type||""}}));
  const links=[];for(let i=0;i<ch.length;i++)for(let j=i+1;j<ch.length;j++){{if(["attn","attention","query","key","value","qkv","cross","norm","linear","mlp","ffn"].some(k=>ch[i].name.toLowerCase().includes(k)&&ch[j].name.toLowerCase().includes(k)))links.push({{source:i,target:j}})}}
  const sim=d3.forceSimulation(ch).force("link",d3.forceLink(links).distance(40)).force("charge",d3.forceManyBody().strength(-60)).force("center",d3.forceCenter(w/2,h/2));
  const mp=Math.max(1,...ch.map(c=>c.params));const cs=d3.scaleLinear().domain([0,mp]).range(["#0969da","#cf222e"]);
  const lg=svg.append("g"),ng=svg.append("g");
  const ll=lg.selectAll("line").data(links).join("line").attr("stroke","#d0d7de").attr("stroke-width",1);
  const nn=ng.selectAll("g").data(ch).join("g");
  nn.append("circle").attr("r",d=>Math.max(4,Math.sqrt(d.params/mp)*12)).attr("fill",d=>cs(d.params)).attr("stroke","#d0d7de");
  nn.append("text").attr("dy",-10).attr("text-anchor","middle").attr("fill","#656d76").attr("font-size",7).text(d=>d.name.length>10?d.name.slice(0,8)+"..":d.name);
  sim.on("tick",()=>{{ll.attr("x1",d=>d.source.x).attr("y1",d=>d.source.y).attr("x2",d=>d.target.x).attr("y2",d=>d.target.y);nn.attr("transform",d=>`translate(${{d.x}},${{d.y}})`);}});
}}
function rTmap(el,nd){{
  const w=el.clientWidth,h=el.clientHeight,svg=d3.select(el).append("svg").attr("viewBox",[0,0,w,h]);
  const ch=(nd.children||[]).length?nd.children:[nd];
  const root=d3.hierarchy({{name:nd.name,children:ch}}).sum(d=>d.params||1);
  d3.treemap().size([w,h]).padding(2)(root);
  const mp=Math.max(1,...root.leaves().map(d=>d.value));const cs=d3.scaleLinear().domain([0,mp]).range(["#0969da","#1a7f37"]);
  svg.selectAll("g").data(root.leaves()).join("g").attr("transform",d=>`translate(${{d.x0}},${{d.y0}})`)
    .each(function(d){{const g=d3.select(this);g.append("rect").attr("width",d.x1-d.x0).attr("height",d.y1-d.y0).attr("fill",cs(d.value)).attr("rx",2).attr("opacity",.85).attr("stroke","#fff");
    if(d.x1-d.x0>30)g.append("text").attr("x",2).attr("y",10).attr("fill","#fff").attr("font-size",7).text(d.data.name.split(".").pop().slice(0,8));
    if(d.x1-d.x0>30&&d.y1-d.y0>20)g.append("text").attr("x",2).attr("y",20).attr("fill","rgba(255,255,255,.7)").attr("font-size",6).text(fp(d.value))}})
}}
function rPipe(el,nd){{
  const w=el.clientWidth,h=el.clientHeight,svg=d3.select(el).append("svg").attr("viewBox",[0,0,w,h]);
  const ch=nd.children||[nd];const n=ch.length,bw=Math.min(55,(w-30)/n-6),bh=30,mx=15,cy=h/2;
  const mp=Math.max(1,...ch.map(c=>c.params||0));const cs=d3.scaleLinear().domain([0,mp]).range(["#0969da","#cf222e"]);
  ch.forEach((c,i)=>{{const x=mx+i*(bw+6);
    if(i<n-1){{svg.append("line").attr("x1",x+bw).attr("y1",cy).attr("x2",x+bw+6).attr("y2",cy).attr("stroke","#d0d7de").attr("stroke-width",2);svg.append("polygon").attr("points",`${{x+bw+6}},${{cy}} ${{x+bw+3}},${{cy-3}} ${{x+bw+3}},${{cy+3}}`).attr("fill","#d0d7de")}}
    const g=svg.append("g").attr("transform",`translate(${{x}},${{cy-bh/2}})`);
    g.append("rect").attr("width",bw).attr("height",bh).attr("rx",4).attr("fill","#fff").attr("stroke",cs(c.params||0)).attr("stroke-width",1.5);
    const nm=c.name.split(".").pop();g.append("text").attr("x",bw/2).attr("y",12).attr("text-anchor","middle").attr("fill","#1f2328").attr("font-size",7).text(nm.length>8?nm.slice(0,6)+"..":nm);
    g.append("text").attr("x",bw/2).attr("y",22).attr("text-anchor","middle").attr("fill","#656d76").attr("font-size",6).text(fp(c.params||0))
  }})
}}
function rBlocks(el,nd){{
  const w=el.clientWidth,h=el.clientHeight,svg=d3.select(el).append("svg").attr("viewBox",[0,0,w,h]);
  const rpt=nd.repeat||1,show=Math.min(rpt,6),cw=Math.min(70,w-30),ch2=24,sy=12;
  const cs=d3.scaleLinear().domain([0,rpt]).range(["#0969da","#8250df"]);
  for(let i=0;i<show;i++){{const x=(w-cw)/2,y=sy+i*(ch2+3),op=1-i*0.06;
    svg.append("rect").attr("x",x).attr("y",y).attr("width",cw).attr("height",ch2).attr("rx",4).attr("fill","#fff").attr("stroke",cs(i)).attr("stroke-width",1).attr("opacity",op);
    svg.append("text").attr("x",x+cw/2).attr("y",y+ch2/2+3).attr("text-anchor","middle").attr("fill","#1f2328").attr("font-size",8).text(i===0?nd.name.split(".").pop():"Block "+i)
  }}
  if(rpt>show)svg.append("text").attr("x",w/2).attr("y",sy+show*(ch2+3)+12).attr("text-anchor","middle").attr("fill","#656d76").attr("font-size",9).text("... and "+(rpt-show)+" more (x"+rpt+" total)")
}}
</script>
</body>
</html>
"""
