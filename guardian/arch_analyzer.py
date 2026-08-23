"""cp_16 · 架构图分析 (ArchAnalyzer)。

参考 archify 设计逻辑（parse → analyze → propose → render），
针对 guarftrain + DeepSeek Harness 场景剪枝：
- 去掉范式自动检测（tree/mesh/treemap/pipeline/blocks）
- 去掉组件库匹配（保留在 component_library.py 独立使用）
- 去掉 AI 提议可视化配置（直接用规则默认值）
- 统一为两种范式：Treemap（占比可视化）+ Backbone Flow（流水线视图）

输出：
- JSON tree data（供 D3 渲染）
- 瓶颈列表
- 独立 HTML 文件（treemap / backbone 两种视图）

详见 checkpoint/cp_16.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from guardian.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 模型解析（复用 archify 的 FLOPs + shape hook 逻辑）
# ---------------------------------------------------------------------------

def _build_dummy_input(model) -> Any:
    """根据模型结构推断 dummy input shape，并匹配模型设备。"""
    try:
        import torch
        # 检测模型所在设备（CPU / CUDA / MPS）
        _device = next(model.parameters()).device
    except Exception:
        _device = None

    def _zeros(*shape):
        import torch
        t = torch.zeros(*shape)
        if _device is not None and _device.type != "cpu":
            t = t.to(_device)
        return t

    try:
        for m in model.modules():
            if hasattr(m, "in_channels") and hasattr(m, "weight"):
                c = m.in_channels
                if any("visual" in n or "conv" in n for n, _ in model.named_modules()):
                    return _zeros(1, c, 224, 224)
        return _zeros(1, 3, 224, 224)
    except Exception:
        logger.warning("推断 dummy input shape 失败，回退 1x3x224x224", exc_info=True)
        return _zeros(1, 3, 224, 224)


def _zeros(*shape):
    import torch
    return torch.zeros(*shape)


def _to_int(raw: Any) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _to_float(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


class ArchAnalyzer:
    """模型架构分析器：解析结构 → 计算 FLOPs/参数 → 检测瓶颈 → 生成可视化数据。"""

    def __init__(self, config: dict | None = None):
        self.cfg = config or {}
        self.bottleneck_threshold_pct = float(
            self.cfg.get("bottleneck_threshold_pct", 25)
        )

    # ------------------------------------------------------------------
    # 模型解析
    # ------------------------------------------------------------------

    @staticmethod
    def parse_model(model_fn) -> dict[str, Any]:
        """解析模型结构为节点树，含真实 FLOPs。

        策略（复用 archify）：
        1. named_modules → 子模块参数/深度
        2. forward hooks → 用 dummy input 跑一次，记录每层 input/output shape
        3. 基于 shape 估算 FLOPs
        4. 同构层折叠（≥4 个连续相同模块 → ×N）
        """
        try:
            import torch
        except ImportError:
            return {"error": "PyTorch 未安装"}

        try:
            model = model_fn()
        except TypeError as exc:
            return {
                "error": f"model_fn 需要构造参数: {exc}。"
                "请在 contract.yaml 的 buildable_entry 中"
                "声明一个无参工厂函数。",
            }
        if not isinstance(model, torch.nn.Module):
            return {
                "error": f"model_fn 必须返回 nn.Module，实际为 {type(model).__name__}"
            }

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

        # 2. 注册 forward hooks
        hook_data: dict[str, dict] = {}

        def _make_pre_hook(_name: str):
            def hook(module, inp):
                if inp and isinstance(inp[0], torch.Tensor):
                    hook_data.setdefault(_name, {})["input_shape"] = list(
                        inp[0].shape
                    )
            return hook

        def _make_post_hook(_name: str):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor):
                    hook_data.setdefault(_name, {})["output_shape"] = list(
                        out.shape
                    )
            return hook

        handles = []
        for name, mod in model.named_modules():
            if name == "":
                continue
            handles.append(mod.register_forward_pre_hook(_make_pre_hook(name)))
            handles.append(mod.register_forward_hook(_make_post_hook(name)))

        # 3. 用 dummy input 跑一次
        try:
            model.eval()
            dummy = _build_dummy_input(model)
            with torch.no_grad():
                model(dummy)
        except Exception:
            logger.warning(
                "用 dummy input 前向失败，FLOPs 将按缺失 shape 估算", exc_info=True
            )

        # 4. 计算 FLOPs
        try:
            for name, data in hook_data.items():
                if name not in module_info:
                    continue
                inp_s = data.get("input_shape")
                out_s = data.get("output_shape")
                if inp_s and out_s:
                    module_info[name]["input_shape"] = inp_s
                    module_info[name]["output_shape"] = out_s
                    flops_est = _compute_flops(
                        module_info[name]["type"], inp_s, out_s
                    )
                    module_info[name]["flops"] = flops_est
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass

        # 5. 折叠同构层
        nodes = _fold_identical_blocks(module_info)
        total_params = sum(p.numel() for p in model.parameters())

        return {
            "nodes": nodes,
            "total_params": total_params,
            "total_flops_est": sum(n.get("flops", 0) for n in nodes),
            "model_name": type(model).__name__,
            "module_count": len(module_info),
        }

    # ------------------------------------------------------------------
    # 统计 + 瓶颈检测
    # ------------------------------------------------------------------

    @staticmethod
    def compute_stats(graph: dict[str, Any]) -> dict[str, Any]:
        """从解析结果计算每层统计 + 瓶颈。"""
        nodes = graph.get("nodes", [])
        total_params = max(graph.get("total_params", 1), 1)
        total_flops = max(graph.get("total_flops_est", 1), 1)
        threshold_pct = 25  # 默认阈值

        layer_stats = []
        bottlenecks = []
        for node in nodes:
            params_pct = round(node.get("params", 0) / total_params * 100, 2)
            flops_pct = round(node.get("flops", 0) / total_flops * 100, 2)
            layer_stats.append(
                {
                    "name": node["name"],
                    "type": node.get("type", "unknown"),
                    "params": node.get("params", 0),
                    "flops": node.get("flops", 0),
                    "params_pct": params_pct,
                    "flops_pct": flops_pct,
                    "repeat": node.get("repeat", 1),
                    "depth": node.get("depth", 0),
                }
            )
            if params_pct >= threshold_pct:
                severity = "critical" if params_pct >= 50 else "warning"
                bottlenecks.append(
                    {
                        "layer": node["name"],
                        "type": node.get("type", "unknown"),
                        "params_pct": params_pct,
                        "flops_pct": flops_pct,
                        "severity": severity,
                        "params": node.get("params", 0),
                        "flops": node.get("flops", 0),
                    }
                )

        bottlenecks.sort(key=lambda x: x["params_pct"], reverse=True)
        return {
            "layer_stats": layer_stats,
            "total_params": total_params,
            "total_flops": total_flops,
            "bottlenecks": bottlenecks[:10],
            "bottleneck_count": len(bottlenecks),
        }

    # ------------------------------------------------------------------
    # 树构建
    # ------------------------------------------------------------------

    @staticmethod
    def build_tree(nodes: list[dict]) -> dict:
        """将折叠后的节点列表转为 D3 treemap/backbone 可消费的树结构。"""
        children = []
        for node in nodes:
            child = {
                "name": node["name"],
                "type": node.get("type", ""),
                "params": node.get("params", 0),
                "flops": node.get("flops", 0),
                "repeat": node.get("repeat", 1),
                "depth": node.get("depth", 0),
                "params_pct": round(
                    node.get("params", 0)
                    / max(1, sum(n.get("params", 0) for n in nodes))
                    * 100,
                    1,
                ),
                "children": (
                    ArchAnalyzer.build_tree(node.get("children", [])).get(
                        "children", []
                    )
                    if node.get("children")
                    else []
                ),
            }
            children.append(child)
        return {"name": "root", "children": children}

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def analyze(self, model_fn) -> dict[str, Any]:
        """完整分析流程：解析 → 统计 → 树构建。

        返回可直接 JSON 序列化的字典，供前端 D3 渲染。
        """
        t0 = time.perf_counter()
        graph = self.parse_model(model_fn)
        if "error" in graph:
            return graph

        stats = self.compute_stats(graph)
        tree = self.build_tree(graph.get("nodes", []))

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "ok": True,
            "model_name": graph.get("model_name", "Model"),
            "total_params": graph.get("total_params", 0),
            "total_flops": graph.get("total_flops_est", 0),
            "total_flops_m": round(graph.get("total_flops_est", 0) / 1e6, 1),
            "module_count": graph.get("module_count", 0),
            "layer_count": len(stats.get("layer_stats", [])),
            "bottleneck_count": stats.get("bottleneck_count", 0),
            "bottlenecks": stats.get("bottlenecks", []),
            "layer_stats": stats.get("layer_stats", []),
            "tree": tree,
            "elapsed_ms": elapsed_ms,
        }

    # ------------------------------------------------------------------
    # HTML 渲染（独立页面）
    # ------------------------------------------------------------------

    @staticmethod
    def render_html(
        analysis: dict[str, Any], output_path: str | Path, view: str = "treemap"
    ) -> Path:
        """渲染独立 HTML 文件。

        view: "treemap" | "backbone"
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        tree_json = json.dumps(analysis.get("tree", {}), ensure_ascii=False)
        bottlenecks_json = json.dumps(
            analysis.get("bottlenecks", [])[:10], ensure_ascii=False
        )
        improvements_json = json.dumps([], ensure_ascii=False)

        html = _RENDER_HTML.format(
            model_name=analysis.get("model_name", "Model"),
            total_params=f"{analysis.get('total_params', 0):,}",
            total_flops_m=analysis.get("total_flops_m", 0),
            module_count=analysis.get("module_count", 0),
            tree_json=tree_json,
            bottlenecks_json=bottlenecks_json,
            improvements_json=improvements_json,
            color_by="params",
            view_mode=view,
            elapsed_ms=analysis.get("elapsed_ms", 0),
        )
        out.write_text(html, encoding="utf-8")
        return out


# ---------------------------------------------------------------------------
# 同构层折叠（直接复用 archify 逻辑）
# ---------------------------------------------------------------------------

def _is_identical(a: dict, b: dict) -> bool:
    return (
        a.get("type") == b.get("type")
        and a.get("params") == b.get("params")
        and a.get("flops") == b.get("flops")
        and a.get("depth") == b.get("depth")
    )


def _fold_identical_blocks(module_info: dict) -> list[dict]:
    """连续同构模块折叠为 ×N。"""
    by_depth: dict[int, list[dict]] = {}
    for name, info in module_info.items():
        depth = info["depth"]
        by_depth.setdefault(depth, []).append({"name": name, **info})

    nodes: list[dict] = []
    collapsed: set = set()

    for depth in sorted(by_depth.keys()):
        items = by_depth[depth]
        i = 0
        while i < len(items):
            item = items[i]
            run = [item]
            j = i + 1
            while j < len(items) and _is_identical(items[j], item):
                run.append(items[j])
                j += 1

            if len(run) >= 4:
                for r in run[1:]:
                    collapsed.add(r["name"])
                children = _get_children(module_info, item["name"], first_only=True)
                nodes.append(
                    {
                        **item,
                        "name": item["name"].rsplit(".", 1)[0]
                        if "." in item["name"]
                        else item["name"],
                        "repeat": len(run),
                        "children": children,
                        "collapsed_names": [r["name"] for r in run],
                    }
                )
                i = j
            else:
                is_collapsed_child = any(
                    item["name"].startswith(c + ".") for c in collapsed
                )
                if item["name"] not in collapsed and not is_collapsed_child:
                    children = _get_children(module_info, item["name"])
                    nodes.append(
                        {
                            **item,
                            "repeat": 1,
                            "children": children,
                            "collapsed_names": [],
                        }
                    )
                i += 1

    return nodes


def _get_children(
    module_info: dict, parent_name: str, *, first_only: bool = False
) -> list[dict]:
    """获取某个模块的直接子模块。"""
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
        children.append(
            {"name": short, **{k: v for k, v in info.items() if k != "name"}}
        )
    return children[:12]


# ---------------------------------------------------------------------------
# FLOPs 计算
# ---------------------------------------------------------------------------

def _compute_flops(layer_type: str, inp_shape: list, out_shape: list) -> int:
    """根据层类型和输入/输出 shape 估算 FLOPs。"""
    if layer_type == "Conv2d" and len(inp_shape) >= 4:
        c_in, h_in, w_in = inp_shape[1], inp_shape[2], inp_shape[3]
        c_out, h_out, w_out = out_shape[1], out_shape[2], out_shape[3]
        return 2 * 3 * 3 * c_in * c_out * h_out * w_out

    if layer_type in (
        "Linear",
        "NonDynamicallyQuantizableLinear",
    ) and len(inp_shape) >= 2:
        return 2 * inp_shape[-1] * out_shape[-1]

    if "MultiheadAttention" in layer_type:
        if len(inp_shape) >= 3:
            seq, dim = inp_shape[1], inp_shape[2]
            return 4 * seq * seq * dim + 2 * seq * dim * dim
        return 0

    if layer_type in (
        "LayerNorm",
        "BatchNorm2d",
        "QuickGELU",
        "GELU",
        "ReLU",
        "SiLU",
        "Dropout",
        "Identity",
        "Sequential",
    ):
        return 0

    try:
        total = 1
        for s in out_shape:
            total *= s
        return 2 * total
    except Exception:
        logger.warning(
            "估算 FLOPs 失败（layer_type=%s），按 0 计", layer_type, exc_info=True
        )
        return 0


# ---------------------------------------------------------------------------
# HTML 模板（独立页面）
# ---------------------------------------------------------------------------

_RENDER_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{model_name} - Architecture Analysis</title>
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
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: var(--font); background: var(--bg); color: var(--text); overflow: hidden; height: 100vh; }}
.header {{ height: 48px; background: var(--bg); border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 20px; justify-content: space-between; }}
.header h1 {{ font-size: 15px; font-weight: 600; }}
.header-stats {{ display: flex; gap: 16px; font-size: 12px; color: var(--text2); }}
.header-stats b {{ color: var(--text); font-weight: 600; }}
.main {{ display: flex; height: calc(100vh - 48px); }}
.sidebar {{ width: 260px; min-width: 260px; background: var(--bg2); border-right: 1px solid var(--border); padding: 14px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }}
.section-title {{ font-size: 10px; font-weight: 600; color: var(--text2); text-transform: uppercase; letter-spacing: .5px; margin-bottom: 4px; }}
.sort-btns {{ display: flex; gap: 3px; }}
.sort-btn {{ padding: 3px 8px; font-size: 11px; background: var(--bg); border: 1px solid var(--border); border-radius: 4px; color: var(--text2); cursor: pointer; }}
.sort-btn.active {{ background: var(--blue); border-color: var(--blue); color: #fff; }}
.filter-group {{ display: flex; flex-direction: column; gap: 3px; }}
.filter-group label {{ font-size: 10px; color: var(--text2); }}
.filter-val {{ font-size: 10px; color: var(--blue); text-align: right; }}
.bn-card {{ background: var(--bg); border-radius: 4px; padding: 6px 8px; margin: 3px 0; border-left: 3px solid var(--red); font-size: 10px; cursor: pointer; border: 1px solid var(--border2); transition: all .15s; }}
.bn-card.warning {{ border-left-color: var(--orange); }}
.bn-card.info {{ border-left-color: var(--blue); }}
.bn-card .nm {{ font-weight: 600; color: var(--text); word-break: break-all; font-size: 11px; }}
.bn-card .pct {{ color: var(--text2); margin-top: 1px; }}
.viz-area {{ flex: 1; overflow: auto; position: relative; background: var(--bg); }}
.viz-area svg {{ width: 100%; height: 100%; }}
.detail-panel {{ width: 0; overflow: hidden; background: var(--bg); border-left: 1px solid var(--border); transition: width .25s; }}
.detail-panel.open {{ width: 340px; min-width: 340px; }}
.dp-inner {{ padding: 14px; width: 340px; height: 100%; overflow-y: auto; }}
.dp-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
.dp-title {{ font-size: 13px; font-weight: 700; }}
.dp-close {{ width: 24px; height: 24px; border-radius: 4px; border: 1px solid var(--border); background: none; color: var(--text2); cursor: pointer; }}
.dp-stat {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid #f0f0f0; font-size: 11px; }}
.dp-stat .k {{ color: var(--text2); }}
.dp-stat .v {{ color: var(--text); font-weight: 600; }}
.dp-bar {{ height: 3px; border-radius: 2px; background: var(--bg3); margin: 3px 0 6px; overflow: hidden; }}
.dp-bar-fill {{ height: 100%; border-radius: 2px; transition: width .3s; }}
.tooltip {{ position: absolute; padding: 8px 12px; background: #fff; border: 1px solid var(--border); border-radius: 6px; font-size: 11px; pointer-events: none; opacity: 0; transition: opacity .12s; max-width: 260px; z-index: 100; box-shadow: 0 3px 12px rgba(140,149,159,.2); }}
.empty-state {{ text-align: center; padding: 30px; color: var(--text3); font-size: 12px; }}
</style>
</head>
<body>
<div class="header">
  <h1>{model_name} — Architecture Analysis</h1>
  <div class="header-stats">
    <span><b>{total_params}</b> params</span>
    <span><b>{total_flops_m}M</b> FLOPs</span>
    <span><b>{module_count}</b> modules</span>
    <span><b>{elapsed_ms}ms</b></span>
  </div>
</div>
<div class="main">
  <div class="sidebar">
    <div><div class="section-title">View</div>
      <div class="sort-btns">
        <button class="sort-btn active" data-view="treemap" onclick="switchView('treemap')">Treemap</button>
        <button class="sort-btn" data-view="backbone" onclick="switchView('backbone')">Flow</button>
      </div></div>
    <div><div class="section-title">Color By</div>
      <div class="sort-btns">
        <button class="sort-btn active" data-color="params" onclick="switchColor('params')">Params</button>
        <button class="sort-btn" data-color="flops" onclick="switchColor('flops')">FLOPs</button>
      </div></div>
    <div><div class="section-title">Bottlenecks ({bn_count})</div><div id="bn-list"></div></div>
  </div>
  <div class="viz-area" id="viz-area"></div>
  <div class="detail-panel" id="dp"><div class="dp-inner" id="dp-inner"></div></div>
</div>
<div class="tooltip" id="tooltip"></div>
<script>
const treeData = {tree_json};
const bottlenecks = {bottlenecks_json};
const improvements = {improvements_json};
const colorBy = "{color_by}";
let curView = "{view_mode}", curColor = "params";

const fp = p => !p ? "0" : p >= 1e6 ? (p/1e6).toFixed(1)+"M" : p >= 1e3 ? (p/1e3).toFixed(0)+"K" : ""+p;
const ff = f => !f ? "0" : f >= 1e6 ? (f/1e6).toFixed(1)+"M" : f >= 1e3 ? (f/1e3).toFixed(0)+"K" : ""+f;
const tip = document.getElementById("tooltip");
function showTip(e, h) {{ tip.innerHTML = h; tip.style.opacity = 1; tip.style.left = (e.pageX + 12) + "px"; tip.style.top = (e.pageY - 10) + "px"; }}
function hideTip() {{ tip.style.opacity = 0; }}

const bnL = document.getElementById("bn-list");
bottlenecks.forEach(b => {{
  const c = document.createElement("div");
  c.className = "bn-card " + (b.severity || "info");
  c.innerHTML = '<div class="nm">' + b.layer + '</div><div class="pct">P:' + (b.params_pct || 0).toFixed(1) + '% | F:' + (b.flops_pct || 0).toFixed(1) + '%</div>';
  c.onclick = () => openDetail(b.layer);
  bnL.appendChild(c);
}});
if (!bottlenecks.length) bnL.innerHTML = '<div class="empty-state">No bottlenecks detected</div>';

function switchView(v) {{
  curView = v;
  document.querySelectorAll("[data-view]").forEach(b => b.classList.toggle("active", b.dataset.view === v));
  renderViz();
}}
function switchColor(c) {{
  curColor = c;
  document.querySelectorAll("[data-color]").forEach(b => b.classList.toggle("active", b.dataset.color === c));
  renderViz();
}}

function renderViz() {{
  const area = document.getElementById("viz-area");
  area.innerHTML = "";
  if (curView === "treemap") renderTreemap(area);
  else renderBackbone(area);
}}

function getMaxVal() {{
  const root = treeData.children || [];
  return Math.max(1, ...root.map(n => curColor === "params" ? (n.params || 0) : (n.flops || 0)));
}}

function getColor(v) {{
  const r = v / getMaxVal();
  if (curColor === "params") return r < 0.25 ? "#1a7f37" : r < 0.5 ? "#8250df" : "#cf222e";
  return r < 0.25 ? "#0969da" : r < 0.5 ? "#8250df" : "#bf8700";
}}

function renderTreemap(area) {{
  const w = area.clientWidth || 800, h = area.clientHeight || 600;
  const svg = d3.select(area).append("svg").attr("viewBox", [0, 0, w, h]);
  const ch = (treeData.children || []).map((c, i) => ({{ id: i, name: c.name, value: curColor === "params" ? (c.params || 1) : (c.flops || 1), raw: c }}));
  const root = d3.hierarchy({{ children: ch }}).sum(d => d.value);
  d3.treemap().size([w, h]).padding(3)(root);
  const mp = getMaxVal();
  const leaves = root.leaves();
  const g = svg.selectAll("g").data(leaves).join("g").attr("transform", d => `translate(${{d.x0}},${{d.y0}})`);
  g.append("rect").attr("width", d => d.x1 - d.x0).attr("height", d => d.y1 - d.y0)
    .attr("fill", d => getColor(d.data.raw[curColor === "params" ? "params" : "flops"] || 1))
    .attr("rx", 3).attr("opacity", 0.85).attr("stroke", "#fff").attr("stroke-width", 1)
    .on("click", (ev, d) => openDetail(d.data.raw.name))
    .on("mouseenter", (ev, d) => showTip(ev, `<b>${{d.data.raw.name}}</b><br>Type: ${{d.data.raw.type || ""}} ${{d.data.raw.repeat > 1 ? "x" + d.data.raw.repeat : ""}}<br>Params: ${{fp(d.data.raw.params)}}<br>FLOPs: ${{ff(d.data.raw.flops)}}`))
    .on("mouseleave", hideTip).on("mousemove", (ev) => {{ tip.style.left = (ev.pageX + 12) + "px"; tip.style.top = (ev.pageY - 10) + "px"; }});
  g.filter(d => (d.x1 - d.x0) > 40 && (d.y1 - d.y0) > 20).append("text")
    .attr("x", 4).attr("y", 14).attr("fill", "#fff").attr("font-size", 9).attr("font-weight", 600)
    .text(d => {{ let n = d.data.raw.name.split(".").pop(); return n.length > 12 ? n.slice(0, 10) + ".." : n; }});
  g.filter(d => (d.x1 - d.x0) > 40 && (d.y1 - d.y0) > 34).append("text")
    .attr("x", 4).attr("y", 28).attr("fill", "rgba(255,255,255,0.8)").attr("font-size", 8)
    .text(d => fp(d.data.raw.params));
}}

function renderBackbone(area) {{
  const w = area.clientWidth || 800, h = area.clientHeight || 600;
  const svg = d3.select(area).append("svg").attr("viewBox", [0, 0, w, h]);
  const nodes = treeData.children || [];
  if (!nodes.length) {{ svg.append("text").attr("x", w/2).attr("y", h/2).attr("text-anchor", "middle").attr("fill", "#888").text("No data"); return; }}
  const nw = 140, nh = 72, gap = 60, mx = 60, my = 50;
  const tw = nodes.length * (nw + gap) - gap + mx * 2, th = nh + my * 2 + 30;
  svg.attr("viewBox", [0, 0, Math.max(tw, w), Math.max(th, h)]);
  const gg = svg.append("g");
  for (let i = 0; i < nodes.length - 1; i++) {{
    const x1 = mx + i * (nw + gap) + nw, y = my + nh / 2, x2 = mx + (i + 1) * (nw + gap);
    const tk = Math.max(1, Math.min(5, (nodes[i].params || 0) / getMaxVal() * 5));
    gg.append("line").attr("x1", x1).attr("y1", y).attr("x2", x2).attr("y2", y).attr("stroke", "#d0d7de").attr("stroke-width", tk).attr("stroke-linecap", "round");
    gg.append("polygon").attr("points", `${{x2}},${{y}} ${{x2 - 5}},${{y - 3}} ${{x2 - 5}},${{y + 3}}`).attr("fill", "#d0d7de");
  }}
  nodes.forEach((nd, i) => {{
    const x = mx + i * (nw + gap), y = my;
    const pctP = (nd.params || 0) / getMaxVal(), pctF = (nd.flops || 0) / getMaxVal();
    const g = gg.append("g").attr("transform", `translate(${{x}},${{y}})`).style("cursor", "pointer")
      .on("click", () => openDetail(nd.name))
      .on("mouseenter", (ev) => showTip(ev, `<b>${{nd.name}}</b><br>${{nd.type || ""}} ${{nd.repeat > 1 ? "x" + nd.repeat : ""}}<br>Params: ${{fp(nd.params)}}<br>FLOPs: ${{ff(nd.flops)}}`))
      .on("mouseleave", hideTip).on("mousemove", (ev) => {{ tip.style.left = (ev.pageX + 12) + "px"; tip.style.top = (ev.pageY - 10) + "px"; }});
    g.append("rect").attr("width", nw).attr("height", nh).attr("rx", 8).attr("fill", "#fff").attr("stroke", "#d0d7de");
    g.append("rect").attr("width", nw).attr("height", 2).attr("rx", 1).attr("fill", getColor(curColor === "params" ? nd.params : nd.flops || 0));
    const lbl = nd.name.split(".").pop();
    g.append("text").attr("x", nw / 2).attr("y", 20).attr("text-anchor", "middle").attr("fill", "#1f2328").attr("font-size", 10).attr("font-weight", 600).text(lbl.length > 16 ? lbl.slice(0, 14) + ".." : lbl);
    g.append("text").attr("x", nw / 2).attr("y", 32).attr("text-anchor", "middle").attr("fill", "#656d76").attr("font-size", 8).text((nd.type || "") + (nd.repeat > 1 ? " x" + nd.repeat : ""));
    g.append("rect").attr("x", 8).attr("y", 38).attr("width", nw - 16).attr("height", 3).attr("rx", 1.5).attr("fill", "#f3f4f6");
    g.append("rect").attr("x", 8).attr("y", 38).attr("width", Math.max(2, (nw - 16) * pctP)).attr("height", 3).attr("rx", 1.5).attr("fill", "#1a7f37");
    g.append("rect").attr("x", 8).attr("y", 44).attr("width", nw - 16).attr("height", 3).attr("rx", 1.5).attr("fill", "#f3f4f6");
    g.append("rect").attr("x", 8).attr("y", 44).attr("width", Math.max(2, (nw - 16) * pctF)).attr("height", 3).attr("rx", 1.5).attr("fill", "#0969da");
    g.append("text").attr("x", 6).attr("y", 58).attr("fill", "#656d76").attr("font-size", 8).text(fp(nd.params) + " p");
    g.append("text").attr("x", nw - 6).attr("y", 58).attr("text-anchor", "end").attr("fill", "#bf8700").attr("font-size", 7).text(ff(nd.flops));
  }});
  svg.call(d3.zoom().scaleExtent([0.3, 3]).on("zoom", (e) => gg.attr("transform", e.transform)));
}}

function openDetail(name) {{
  const nd = (treeData.children || []).find(x => x.name === name);
  if (!nd) return;
  const dp = document.getElementById("dp"), inner = document.getElementById("dp-inner");
  dp.classList.add("open");
  const pp = (nd.params_pct || 0), fpp = (nd.flops || 0) / Math.max(1, getMaxVal()) * 100;
  let h = '<div class="dp-head"><div class="dp-title">' + nd.name.split(".").pop() + '</div>';
  h += '<button class="dp-close" onclick="closeDetail()">x</button></div>';
  h += '<div class="dp-stat"><span class="k">Type</span><span class="v">' + (nd.type || "?") + '</span></div>';
  if (nd.repeat > 1) h += '<div class="dp-stat"><span class="k">Repeat</span><span class="v">x' + nd.repeat + '</span></div>';
  h += '<div class="dp-stat"><span class="k">Parameters</span><span class="v">' + fp(nd.params) + " (" + pp.toFixed(1) + "%)</span></div>";
  h += '<div class="dp-bar"><div class="dp-bar-fill" style="width:' + pp + "%;background:#1a7f37\"></div></div>";
  h += '<div class="dp-stat"><span class="k">FLOPs</span><span class="v">' + ff(nd.flops) + '</span></div>';
  h += '<div class="dp-bar"><div class="dp-bar-fill" style="width:' + fpp.toFixed(1) + "%;background:#0969da\"></div></div>";
  if (nd.children && nd.children.length) h += '<div class="dp-stat"><span class="k">Sub-modules</span><span class="v">' + nd.children.length + '</span></div>';
  inner.innerHTML = h;
}}

function closeDetail() {{ document.getElementById("dp").classList.remove("open"); }}

renderViz();
window.addEventListener("resize", renderViz);
</script>
</body>
</html>
"""
