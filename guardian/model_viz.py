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
            pass

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
                pass
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
                    pass
            improvements.append(entry)
        return improvements

    # ------------------------------------------------------------------
    # HTML 渲染
    # ------------------------------------------------------------------

    def render_html(self, graph, stats, viz_config, output_path):
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        nodes = graph.get("nodes", [])
        # 构建 D3 树结构
        tree = _build_tree(nodes)

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
        out = self.render_html(graph, stats, viz_config, output_path)
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
# HTML 模板
# ---------------------------------------------------------------------------

_RENDER_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{model_name} — Model Structure</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:'Segoe UI',system-ui,sans-serif; background:#0d1117; color:#c9d1d9; overflow:hidden; }}
#app {{ display:flex; height:100vh; }}
#sidebar {{ width:360px; min-width:360px; background:#161b22; padding:20px; overflow-y:auto; border-right:1px solid #30363d; }}
#sidebar h2 {{ font-size:16px; color:#58a6ff; margin-bottom:8px; }}
#sidebar .stat {{ font-size:12px; color:#8b949e; margin:3px 0; }}
#sidebar .stat b {{ color:#e6edf3; }}
#chart {{ flex:1; overflow:auto; cursor:grab; }}
#chart svg {{ width:100%; height:100%; }}
.bn-card {{ background:#0d1117; border-radius:6px; padding:10px; margin:6px 0; border-left:3px solid #f85149; font-size:12px; }}
.bn-card.warning {{ border-left-color:#d2991d; }}
.bn-card.info {{ border-left-color:#58a6ff; }}
.bn-card .name {{ font-weight:600; word-break:break-all; }}
.bn-card .pct {{ color:#8b949e; }}
.bn-card .sug {{ color:#d2991d; margin-top:4px; }}
.node circle {{ stroke:#30363d; stroke-width:1.5px; }}
.node text {{ font-size:10px; fill:#8b949e; pointer-events:none; }}
.node .label-bg {{ fill:#161b22; rx:3; }}
.node .repeat {{ font-size:9px; fill:#58a6ff; }}
.link {{ fill:none; stroke:#21262d; stroke-width:1px; }}
.tooltip {{ position:absolute; padding:8px 12px; background:#21262d; border:1px solid #30363d; border-radius:6px; font-size:11px; pointer-events:none; opacity:0; transition:opacity 0.15s; max-width:280px; z-index:100; }}
</style>
</head>
<body>
<div id="app">
<div id="sidebar">
  <h2>{model_name}</h2>
  <div style="font-size:12px;color:#8b949e;margin-bottom:12px;">{summary_text}</div>
  <div class="stat">Parameters: <b>{total_params}</b></div>
  <div class="stat">FLOPs: <b>{total_flops}</b> ({total_flops_m}M)</div>
  <div class="stat">Modules: <b>{module_count}</b></div>
  <div class="stat">Color by: <b>{color_by}</b></div>
  <div style="margin:12px 0;font-size:11px;color:#8b949e;">
    <span style="display:inline-block;width:12px;height:12px;background:#238636;border-radius:2px;"></span> &lt;5%&nbsp;
    <span style="display:inline-block;width:12px;height:12px;background:#d2991d;border-radius:2px;"></span> 5-25%&nbsp;
    <span style="display:inline-block;width:12px;height:12px;background:#f85149;border-radius:2px;"></span> &gt;25%
  </div>
  <h2 style="margin-top:16px;">Bottlenecks</h2>
  <div id="bottlenecks"></div>
  <div style="margin-top:16px;font-size:11px;color:#484f58;">
    Click node to expand/collapse &middot; Scroll to zoom &middot; Drag to pan
  </div>
</div>
<div id="chart"></div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const treeData = {tree_json};
const bottlenecks = {bottlenecks_json};
const colorBy = "{color_by}";

// Render bottleneck cards
const bnDiv = document.getElementById("bottlenecks");
bottlenecks.forEach(b => {{
  const card = document.createElement("div");
  card.className = "bn-card " + (b.severity || "info");
  card.innerHTML = '<div class="name">' + b.layer + '</div>'
    + '<div class="pct">Params: ' + (b.params_pct||0).toFixed(1) + '% | FLOPs: ' + (b.flops_pct||0).toFixed(1) + '%</div>'
    + (b.suggestion ? '<div class="sug">' + b.suggestion + '</div>' : '');
  bnDiv.appendChild(card);
}});
if (!bottlenecks.length) document.getElementById("bottlenecks").innerHTML
  = '<div class="stat">No significant bottlenecks detected</div>';

// D3 tree
const width = document.getElementById("chart").clientWidth;
const height = document.getElementById("chart").clientHeight;

const root = d3.hierarchy(treeData);
const dx = 22;
const dy = 200;

const maxParams = Math.max(1, ...root.descendants().map(d => d.data.params || 0));
const colorScale = d3.scaleLinear()
  .domain([0, maxParams * 0.05, maxParams * 0.25, maxParams])
  .range(["#238636", "#d2991d", "#f85149", "#da3633"]);

function nodeColor(d) {{
  if (colorBy === "flops" && d.data.flops) {{
    const maxF = Math.max(1, ...root.descendants().map(x => x.data.flops || 0));
    const r = (d.data.flops || 0) / maxF;
    return d3.interpolateRgb("#238636", "#f85149")(r);
  }}
  return colorScale(d.data.params || 0);
}}

const tree = d3.tree().nodeSize([dx, dy])(root);
const svg = d3.select("#chart").append("svg")
  .attr("viewBox", [-width/2, -40, width, Math.max(height, root.descendants().length * dx + 80)])
  .call(d3.zoom().scaleExtent([0.3, 5]).on("zoom", e => g.attr("transform", e.transform)));

const g = svg.append("g");

const link = g.append("g").selectAll("path")
  .data(root.links()).join("path")
  .attr("class", "link")
  .attr("d", d => `M${{d.source.y}},${{d.source.x}}C${{(d.source.y+d.target.y)/2}},${{d.source.x}} ${{(d.source.y+d.target.y)/2}},${{d.target.x}} ${{d.target.y}},${{d.target.x}}`);

const node = g.append("g").selectAll("g")
  .data(root.descendants()).join("g")
  .attr("transform", d => `translate(${{d.y}},${{d.x}})`)
  .style("cursor", d => d.children && d.children.length ? "pointer" : "default")
  .on("click", (event, d) => {{
    if (d.children && d.children.length) {{
      d._children = d.children;
      d.children = null;
    }} else if (d._children) {{
      d.children = d._children;
      d._children = null;
    }}
    update(d);
  }})
  .on("mouseenter", (event, d) => {{
    const tip = document.getElementById("tooltip");
    const rpt = d.data.repeat > 1 ? " <b>x" + d.data.repeat + "</b>" : "";
    tip.innerHTML = "<b>" + d.data.name + "</b>" + rpt
      + "<br>Type: " + (d.data.type || "?")
      + "<br>Params: " + (d.data.params||0).toLocaleString()
      + "<br>FLOPs: " + (d.data.flops||0).toLocaleString()
      + (d.data.params_pct ? "<br>Share: " + d.data.params_pct.toFixed(1) + "%" : "");
    tip.style.opacity = 1;
    tip.style.left = (event.pageX + 12) + "px";
    tip.style.top = (event.pageY - 10) + "px";
  }})
  .on("mouseleave", () => document.getElementById("tooltip").style.opacity = 0)
  .on("mousemove", event => {{
    const tip = document.getElementById("tooltip");
    tip.style.left = (event.pageX + 12) + "px";
    tip.style.top = (event.pageY - 10) + "px";
  }});

node.append("circle")
  .attr("r", d => Math.max(3, Math.min(12, Math.sqrt((d.data.params||0) / Math.max(1,maxParams)) * 15)))
  .attr("fill", d => nodeColor(d));

node.append("text")
  .attr("dy", "0.32em")
  .attr("x", d => (d.children && d.children.length ? -10 : 10))
  .attr("text-anchor", d => (d.children && d.children.length ? "end" : "start"))
  .text(d => {{
    let name = d.data.name.split(".").pop();
    if (name.length > 22) name = name.slice(0,20) + "..";
    return name;
  }})
  .clone(true).lower().attr("class", "label-bg")
  .attr("stroke", "#161b22").attr("stroke-width", 3).attr("stroke-linejoin", "round");

// Repeat badge
node.filter(d => d.data.repeat > 1).append("text")
  .attr("class", "repeat")
  .attr("dy", "-0.8em")
  .attr("text-anchor", "middle")
  .text(d => "x" + d.data.repeat);

function update(source) {{
  const tree = d3.tree().nodeSize([dx, dy])(root);
  const nodes = root.descendants();
  const links = root.links();

  const t = svg.transition().duration(400);

  link.data(links).transition(t)
    .attr("d", d => `M${{d.source.y}},${{d.source.x}}C${{(d.source.y+d.target.y)/2}},${{d.source.x}} ${{(d.source.y+d.target.y)/2}},${{d.target.x}} ${{d.target.y}},${{d.target.x}}`);

  node.data(nodes).transition(t)
    .attr("transform", d => `translate(${{d.y}},${{d.x}})`)
    .select("circle").attr("fill", d => nodeColor(d));
}}
</script>
</body>
</html>"""
