"""Training Guardian Agent — Hugging Face Spaces 演示页。

展示内容：项目概览 / 指标曲线 / Checkpoint 演进 / AI 分析 / MCP 工具 / 架构图 / 模型可视化
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Training Guardian Agent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ROOT = Path(__file__).resolve().parent
_LOGS_DIR = ROOT / "logs"
_DEMO_DIR = ROOT / "demo_logs"
LOGS_DIR = _LOGS_DIR if _LOGS_DIR.is_dir() and any(_LOGS_DIR.iterdir()) else _DEMO_DIR

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def _load_summaries() -> list[dict]:
    """加载所有 summary JSON 文件。"""
    summaries = []
    if LOGS_DIR.is_dir():
        for f in sorted(LOGS_DIR.glob("summary_*.json")):
            try:
                summaries.append(json.loads(f.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError):
                pass
    return summaries


def _load_metrics(experiment_id: str) -> list[dict]:
    """加载某个实验的 metrics.jsonl。"""
    path = LOGS_DIR / experiment_id / "metrics.jsonl"
    if not path.is_file():
        return []
    metrics = []
    try:
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            try:
                metrics.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return metrics


def _list_experiments() -> list[dict]:
    """列出所有有数据的实验。"""
    exps = []
    if not LOGS_DIR.is_dir():
        return exps
    for d in sorted(LOGS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / "meta.json"
        metrics_file = d / "metrics.jsonl"
        if metrics_file.is_file():
            meta = {}
            if meta_file.is_file():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    pass
            # 统计条数
            try:
                n_lines = sum(1 for _ in open(metrics_file, encoding="utf-8"))
            except OSError:
                n_lines = 0
            exps.append({
                "id": d.name,
                "name": meta.get("name", d.name),
                "status": meta.get("status", "unknown"),
                "command": meta.get("command", ""),
                "metrics_count": n_lines,
            })
    return exps


def _load_anomalies(experiment_id: str) -> list[dict]:
    """加载异常事件。"""
    path = LOGS_DIR / experiment_id / "anomalies.jsonl"
    if not path.is_file():
        return []
    items = []
    try:
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    except OSError:
        pass
    return items


def _get_model_viz_htmls() -> list[dict]:
    """列出可用的模型可视化 HTML。"""
    viz_dir = LOGS_DIR / "viz"
    htmls = []
    if viz_dir.is_dir():
        for f in sorted(viz_dir.glob("*.html")):
            htmls.append({"name": f.stem, "path": str(f)})
    return htmls


# ---------------------------------------------------------------------------
# 页面渲染
# ---------------------------------------------------------------------------

# --- 标题区 ---
col1, col2 = st.columns([1, 6])
with col1:
    st.markdown("")
    st.markdown("")
    st.markdown("### 🛡️")
with col2:
    st.title("Training Guardian Agent · 训练守护智能体")
    st.markdown(
        "> **一行命令，训练脚本零行改动，获得完整守护能力。**  "
        "One command. Zero changes to your training script. Full guardian capabilities."
    )
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-仓库-blue?logo=github)]"
        "(https://github.com/Washington5533/guarftrain)  "
        "![Version](https://img.shields.io/badge/version-0.2.0-blue)  "
        "![Python](https://img.shields.io/badge/python-3.10%2B-blue)  "
        "![MCP](https://img.shields.io/badge/MCP-35_tools-green)"
    )

st.code("guarftrain init && guarftrain watch -- python train.py --epochs 20", language="bash")

st.divider()

# --- Tab 导航 ---
tab_overview, tab_metrics, tab_faults, tab_checkpoints, tab_ai, tab_mcp, tab_viz, tab_arch = st.tabs([
    "📋 项目概览", "📊 指标曲线", "🛡️ 故障处理", "🏆 Checkpoint", "🤖 AI 分析", "🔧 MCP 工具", "🧠 模型可视化", "🏗️ 架构"
])

# ===== Tab 1: 项目概览 =====
with tab_overview:
    st.subheader("五大阶段能力")

    st.markdown("""
| 阶段 | 能力 | 方式 |
|------|------|------|
| **训练前** Pre-flight | GPU 显存预估 + batch 推荐 | `guarftrain preflight` |
| **训练中** During | GPU+Loss 监控告警 / 崩溃自动恢复 / LLM 智能决策 | `guarftrain watch` |
| **训练后** Post | 摘要+AI 解读 / Checkpoint 分析 / 模型可视化 / 推理 | `guarftrain summarize` |
| **跨实验** Cross | 自然语言查询 / 实验对比 / 外部数据导入 | `guarftrain query "best lr?"` |
| **外部接入** External | MCP 35 工具 + Dashboard 远程配置 + Agent 图表推荐 | `guarftrain start` |
""")

    st.subheader("决策分层架构")
    st.markdown("""
```text
┌─ Contract ──→ 硬边界，人工定义（训练脚本四项契约）
├─ Agent ─────→ LLM 决策层（可选，在允许的动作空间内）
├─ Rules ─────→ 确定性规则（始终在线的兜底）
├─ MCP ──────→ 外部 Agent 接入（双模式委托）
└─ Dashboard ─→ 远程配置 + dirty-flag 用户保护
```
""")

    # 实验列表
    st.subheader("已记录实验")
    exps = _list_experiments()
    if exps:
        st.dataframe(
            [{k: v for k, v in e.items() if k != "command"} for e in exps],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("暂无实验数据（logs/ 目录为空或不存在）")

    st.subheader("CLI 命令速查")
    cmds = [
        ("`init`", "自动扫描项目 + 生成 contract.yaml"),
        ("`check`", "环境自检：Python/GPU/依赖"),
        ("`watch`", "守护任意训练命令"),
        ("`start`", "Dashboard + MCP 一键启动"),
        ("`serve`", "独立 MCP server"),
        ("`preflight`", "GPU 显存预估 + batch 推荐"),
        ("`analyze`", "扫描已有 checkpoint"),
        ("`experiments`", "列出所有历史实验"),
        ("`query`", "自然语言查询（\"best lr?\"）"),
        ("`compare`", "对比两个实验"),
        ("`visualize`", "模型结构可视化（D3.js HTML）"),
        ("`infer`", "用 checkpoint 跑推理"),
        ("`gallery`", "图片筛选 + 展示"),
    ]
    cols = st.columns(3)
    for i, (cmd, desc) in enumerate(cmds):
        with cols[i % 3]:
            st.markdown(f"{cmd} — {desc}")

# ===== Tab 2: 指标曲线 =====
with tab_metrics:
    exps = _list_experiments()
    if not exps:
        st.info("暂无实验数据。将 logs/ 目录中的 metrics.jsonl 放入对应子目录即可展示。")
    else:
        # 选择实验
        exp_names = [f"{e['name']} ({e['metrics_count']} 条)" for e in exps]
        selected_idx = st.selectbox("选择实验", range(len(exp_names)), format_func=lambda i: exp_names[i])
        exp = exps[selected_idx]
        metrics = _load_metrics(exp["id"])

        if not metrics:
            st.warning("该实验无指标数据")
        else:
            st.caption(f"共 {len(metrics)} 条记录")

            # 提取可用的数值字段
            numeric_fields = []
            for k in metrics[-1].keys():
                if k in ("raw", "step", "epoch", "timestamp"):
                    continue
                for m in metrics[-50:]:
                    if k in m and isinstance(m.get(k), (int, float)):
                        numeric_fields.append(k)
                        break

            # Loss 相关
            loss_fields = [f for f in numeric_fields if "loss" in f.lower()]
            acc_fields = [f for f in numeric_fields if any(w in f.lower() for w in ("acc", "map", "f1", "miou"))]
            lr_fields = [f for f in numeric_fields if "lr" in f.lower()]
            other_fields = [f for f in numeric_fields if f not in set(loss_fields + acc_fields + lr_fields)]

            # 画 Loss 图
            if loss_fields:
                st.subheader("📉 Loss")
                loss_data = {}
                for f in loss_fields:
                    loss_data[f] = [m.get(f) for m in metrics if m.get(f) is not None]
                if loss_data:
                    st.line_chart(loss_data, use_container_width=True, height=300)

            # 画 Accuracy 图
            if acc_fields:
                st.subheader("📈 Accuracy / 指标")
                acc_data = {}
                for f in acc_fields:
                    acc_data[f] = [m.get(f) for m in metrics if m.get(f) is not None]
                if acc_data:
                    st.line_chart(acc_data, use_container_width=True, height=300)

            # 画 LR 图
            if lr_fields:
                st.subheader("🔬 Learning Rate")
                lr_data = {}
                for f in lr_fields:
                    lr_data[f] = [m.get(f) for m in metrics if m.get(f) is not None]
                if lr_data:
                    st.line_chart(lr_data, use_container_width=True, height=250)

            # 其他字段
            if other_fields:
                st.subheader("📐 其他指标")
                other_data = {}
                for f in other_fields[:6]:
                    other_data[f] = [m.get(f) for m in metrics if m.get(f) is not None]
                if other_data:
                    st.line_chart(other_data, use_container_width=True, height=250)

# ===== Tab 3: 故障处理展示 =====
with tab_faults:
    st.subheader("故障处理能力演示")
    st.markdown("Guardian 覆盖训练全生命周期的 **12 种故障场景**，以下是基于 1000 epoch 仿真数据的完整演示。")

    # 加载异常数据
    anoms = _load_anomalies("mnist-guardian")

    # 从 summary 加载 restarts
    summary_path = LOGS_DIR / "summary_demo.json"
    summary_data = {}
    if summary_path.is_file():
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    restarts_data = summary_data.get("restarts", [])
    anomaly_data = summary_data.get("anomaly_events", anoms)

    if not anomaly_data and not restarts_data:
        st.info("暂无故障数据。运行 gen_demo_data.py 生成演示数据。")
    else:
        # --- 故障类型统计 ---
        st.subheader("故障类型覆盖")
        fault_types = {}
        for a in anomaly_data:
            t = a.get("type", "unknown")
            fault_types[t] = fault_types.get(t, 0) + 1

        cols = st.columns(6)
        type_labels = {
            "loss_spike": "📈 Loss 突增",
            "gpu_idle": "💤 GPU 空转",
            "oom_recovery": "💥 OOM 恢复",
            "gpu_temp": "🌡️ GPU 过热",
            "agent_restart_lr": "🧠 Agent 降 LR",
            "loss_stagnation": "📏 Loss 停滞",
            "crash_recovery": "🔫 崩溃续训",
            "nan_inf": "☠️ NaN/Inf",
            "code_error": "🐛 代码错误",
        }
        for i, (t, count) in enumerate(sorted(fault_types.items())):
            with cols[i % 6]:
                label = type_labels.get(t, t)
                st.metric(label, count)

        # --- 异常事件时间线 ---
        st.subheader("异常事件时间线")
        st.caption(f"共 {len(anomaly_data)} 个异常事件，按 epoch 排列")

        severity_color = {
            "error": "red",
            "warning": "orange",
            "info": "blue",
        }
        for a in anomaly_data:
            ep = a.get("epoch", "?")
            t = a.get("type", "?")
            sev = a.get("severity", "info")
            detail = a.get("detail", "")
            resp = a.get("response", {})
            action = resp.get("action", "?")
            source = resp.get("source", "?")
            restart = "🔄 重启" if resp.get("restart") else "📢 告警"
            color = severity_color.get(sev, "gray")

            emoji = {
                "loss_spike": "📈", "gpu_idle": "💤", "oom_recovery": "💥",
                "gpu_temp": "🌡️", "agent_restart_lr": "🧠", "loss_stagnation": "📏",
                "crash_recovery": "🔫", "nan_inf": "☠️", "code_error": "🐛",
            }.get(t, "❓")

            with st.expander(f"{emoji} Epoch {ep} — {t} — {restart}"):
                st.markdown(f":{color}[**{sev.upper()}**] | 响应: `{action}` | 来源: `{source}`")
                st.markdown(detail)

        # --- 重启历史 ---
        if restarts_data:
            st.subheader("重启/恢复记录")
            st.caption(f"共 {len(restarts_data)} 次重启")
            rows = []
            for r in restarts_data:
                rows.append({
                    "Epoch": r.get("epoch"),
                    "触发原因": r.get("trigger"),
                    "从 ckpt 恢复": r.get("resumed_from", "无"),
                    "浪费 Epoch": r.get("wasted_epochs", 0),
                    "成功": "✅" if r.get("success") else "❌",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

        # --- 故障场景对照表 ---
        st.subheader("全部故障场景与 Guardian 响应")
        scenarios = [
            ("loss_spike", "Loss 突增", "滑动窗口均值对比，超出阈值 → 告警", "Agent 可选 restart_with_lower_lr / ignore", "📈"),
            ("loss_stagnation", "Loss 停滞", "连续 N 步下降 < 阈值 → 告警", "Agent 可选 suggest_lr_increase", "📏"),
            ("nan_inf", "NaN/Inf Loss", "即时检测，不进滑动窗口", "rollback_to_last_ckpt + 降 LR", "☠️"),
            ("oom_recovery", "OOM 崩溃", "退出码 / 日志检测", "batch_size 减半 + 从 ckpt 恢复", "💥"),
            ("crash_recovery", "进程崩溃 (kill -9)", "进程退出检测", "参数不变 + 从 ckpt 续训", "🔫"),
            ("code_error", "代码错误 (TypeError)", "退出码 / 日志分析", "0 次重启，直接判定不可恢复", "🐛"),
            ("gpu_idle", "GPU 空转", "nvidia-smi 利用率 < 20% 连续 5 次", "告警：数据加载瓶颈", "💤"),
            ("gpu_temp", "GPU 过热", "nvidia-smi 温度 > 85°C", "告警：硬件安全不交给 Agent", "🌡️"),
            ("agent_decision", "Agent 主动决策", "LLM 分析 loss/acc 趋势", "降 LR / 建议调参 / 忽略", "🧠"),
            ("batch_floor", "Batch 下限保护", "减到 min_batch_size 仍 OOM", "停止重试，不再无限重启", "🛑"),
            ("contract_missing", "契约缺失降级", "无可续训入口", "不重启，只告警", "⚠️"),
            ("mcp_delegated", "MCP 委托模式", "外部 Agent 连接", "内置 Agent 进入 provisional，可被覆盖", "🔗"),
        ]
        scols = st.columns(2)
        for i, (typ, name, detect, response, emoji) in enumerate(scenarios):
            with scols[i % 2]:
                st.markdown(f"{emoji} **{name}** (`{typ}`)")
                st.markdown(f"- 检测: {detect}")
                st.markdown(f"- 响应: {response}")
                st.markdown("")


# ===== Tab 4: Checkpoint =====
with tab_checkpoints:
    summaries = _load_summaries()
    found = False
    for s in summaries:
        ckpt = s.get("checkpoints", {})
        ckpts = ckpt.get("checkpoints", [])
        if not ckpts:
            continue
        found = True
        st.subheader(f"实验: {s.get('experiment_id', '?')}")
        st.caption(f"共 {ckpt.get('total', len(ckpts))} 个 checkpoint  |  评估指标: **{ckpt.get('metric', 'accuracy')}**")

        rows = []
        for c in ckpts:
            m = c.get("metrics", {})
            rows.append({
                "Epoch": c.get("epoch"),
                "Train Loss": m.get("train/loss", m.get("loss")),
                "Val Accuracy": m.get("val/accuracy", m.get("val_acc")),
                "LR": m.get("lr"),
                "Batch Size": m.get("batch_size"),
            })
        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

            # 简易柱状图
            epochs = [r["Epoch"] for r in rows]
            accs = [r["Val Accuracy"] for r in rows if r["Val Accuracy"] is not None]
            if accs and len(accs) == len(epochs):
                st.subheader("验证准确率趋势")
                st.bar_chart(
                    dict(zip([str(e) for e in epochs], accs)),
                    use_container_width=True,
                    height=250,
                )

    if not found:
        st.info("暂无 Checkpoint 汇总数据。运行训练后 summary JSON 会自动包含 checkpoint 信息。")

# ===== Tab 4: AI 分析 =====
with tab_ai:
    summaries = _load_summaries()
    found = False
    for s in summaries:
        narrative = s.get("ai_narrative", "")
        if not narrative:
            continue
        found = True
        exp_id = s.get("experiment_id", "?")
        status = s.get("status", "?")
        duration = s.get("duration", "?")
        with st.expander(f"📝 {exp_id} — 状态: {status} (耗时 {duration})", expanded=(len(summaries) <= 2)):
            st.markdown(narrative)

            # 显示资源信息
            resources = s.get("resources", {})
            if resources:
                rcols = st.columns(5)
                rcols[0].metric("GPU 利用率均值", f"{resources.get('gpu_util_avg', '?')}%")
                rcols[1].metric("显存峰值", f"{resources.get('gpu_mem_peak_mb', '?')} MB")
                rcols[2].metric("GPU 数", resources.get("gpu_count", "?"))
                rcols[3].metric("温度均值", f"{resources.get('gpu_temp_avg', '?')}°C")
                rcols[4].metric("GPU 时", f"{resources.get('gpu_hours', '?')} h")

    if not found:
        st.info("暂无 AI 分析数据。启用 --agent 标志训练后会自动生成 AI 解读。")

    # Agent 决策边界说明
    st.subheader("AI 决策边界")
    st.markdown("""
| 阶段 | 规则 | Agent 角色 |
|------|------|------------|
| **训练中** | 规则判定异常 + 恢复可行性（零延迟，确定性） | Agent 选择应对策略（预设动作集内，失败回退默认） |
| **训练后** | 规则提供基础数据 | Agent 主导：定义策略、匹配组件库、编写优化方案 |
| **MCP 模式** | 外部 Claude Code 决策，内置 Agent 让位 | 数据提供者 + 安全执行器 |

> **不变式**: Agent 的自由度永远是人显式授予的。任何一层失效都退回上一层的确定性行为。
""")

# ===== Tab 5: MCP 工具 =====
with tab_mcp:
    st.subheader("MCP 工具清单")
    st.markdown("共 **35 个工具** (24 只读 + 11 受限写)，覆盖 Guardian 全部功能模块。")

    # 只读工具
    st.markdown("#### 只读工具（24 个，无需鉴权）")
    readonly_tools = [
        # 训练监控 (6)
        ("`get_training_status`", "当前训练状态：epoch/step、loss/accuracy、GPU 状态"),
        ("`get_metrics_history`", "指标时间序列（分页+聚合统计）"),
        ("`get_anomaly_history`", "全部异常事件 + 应对来源"),
        ("`get_recovery_history`", "全部重启记录 + 作废 epoch + 参数变更"),
        ("`get_agent_decision_log`", "Agent 全部 LLM 调用记录 + source/延迟/降级原因"),
        ("`get_summary`", "训练摘要（结构化 + AI 解读）"),
        # 实验查询 (3)
        ("`list_experiments`", "列出所有历史实验摘要"),
        ("`query_experiment`", "自然语言查询实验"),
        ("`compare_experiments`", "对比两个实验的参数、指标、异常"),
        # Checkpoint (2)
        ("`list_checkpoints`", "列出所有 checkpoint + best/top_k 标记"),
        ("`compare_checkpoints`", "对比两个 checkpoint 的指标差异"),
        # 模型与配置 (5)
        ("`get_model_structure`", "模型结构 JSON（节点/边/FLOPs/参数量）"),
        ("`get_guardian_mode`", "当前模式：standalone / mcp_delegated"),
        ("`get_gallery_config`", "图片筛选策略配置"),
        ("`get_import_format`", "Guardian 导入格式规范（JSON Schema）"),
        ("`inspect_source`", "采样外部数据文件前 N 行"),
        # 日志与清单 (3)
        ("`get_training_log`", "训练日志尾部（支持 grep 过滤）"),
        ("`get_post_training_checklist`", "训练结束后可执行的操作清单"),
        ("`get_pending_decisions`", "MCP 模式下待处理的 provisional 决策"),
        # 契约 (2)
        ("`get_contract_status`", "契约四项各自的开启/降级状态"),
        ("`list_contract_proposals`", "Agent 提议记录（pending/approved/rejected）"),
        # Dashboard 配置 (3)
        ("`get_dashboard_config`", "获取 Dashboard 当前配置"),
        ("`recommend_charts`", "AI Agent 分析训练状态，推荐应关注的图表组"),
        ("`list_dashboard_templates`", "列出可用 Dashboard 布局模板"),
    ]
    rcols = st.columns(2)
    for i, (name, desc) in enumerate(readonly_tools):
        with rcols[i % 2]:
            st.markdown(f"{name}: {desc}")

    # 写工具
    st.markdown("#### 受限写工具（11 个，需 token 鉴权 + 训练阶段保护）")
    write_tools = [
        # 训练控制 (4)
        ("`trigger_recovery`", "⚠️ 手动触发恢复重启，回滚到最近 ckpt"),
        ("`restart_with_params`", "⚠️ 调整参数后重启（受白名单约束）"),
        ("`stop_training`", "⚠️ 停止训练子进程并终止看护"),
        ("`resolve_decision`", "🟡 批准或覆盖一条待处理决策"),
        # Dashboard 配置 (1)
        ("`set_dashboard_config`", "🟢 设置 Dashboard 图表选择/面板/模板"),
        # 契约管理 (2)
        ("`approve_contract_proposal`", "🟡 批准 Agent 的契约扩展提议"),
        ("`reject_contract_proposal`", "🟢 拒绝并归档契约提议"),
        # 训练后功能 (4)
        ("`run_visualization`", "生成模型管线可视化 HTML（交互式 D3.js）"),
        ("`set_gallery_config`", "更新图片筛选策略配置"),
        ("`run_inference`", "用指定 ckpt 跑推理（分类/检测/分割）"),
        ("`submit_import`", "提交外部训练数据（WandB/TensorBoard/CSV）"),
    ]
    wcols = st.columns(2)
    for i, (name, desc) in enumerate(write_tools):
        with wcols[i % 2]:
            st.markdown(f"{name}: {desc}")

    st.divider()
    st.markdown("""
**安全设计：**
- 🔒 写工具默认关闭，需显式配置 `enable_write_tools: true`
- 🔑 口令鉴权，token 从环境变量读取（不写入配置文件）
- 🛡️ 训练后专用工具在训练中调用返回错误
- 📝 所有调用记录到 `mcp_access_log.json`（不记录 token）
- 🔁 写工具 5 分钟内相同 request_id 幂等去重
- 🔗 MCP 崩溃/断连不影响训练，训练与守护照常运行
""")

# ===== Tab 6: 模型可视化 =====
with tab_viz:
    viz_htmls = _get_model_viz_htmls()
    if not viz_htmls:
        st.info("暂无模型可视化 HTML。运行 `guarftrain visualize --model train:build_model` 生成。")
    else:
        selected = st.selectbox("选择可视化", viz_htmls, format_func=lambda x: x["name"])
        if selected:
            try:
                html_content = Path(selected["path"]).read_text(encoding="utf-8")
                st.caption(f"文件: {selected['path']}")
                st.components.v1.html(html_content, height=700, scrolling=True)
            except OSError as e:
                st.error(f"无法加载: {e}")

# ===== Tab 7: 架构 =====
with tab_arch:
    st.subheader("系统架构")
    st.markdown("""
```text
┌─ Guardian Process (sidecar) ────────────────────────────────────┐
│                                                                  │
│  CLI (16 commands)                                               │
│  ├─ watch ──→ Watchdog: Popen + crash recovery + CLI rewrite    │
│  │             └─ Monitor: log tail + GPU poll + anomaly detect  │
│  │                  └─ AgentAdvisor: LLM decide → intervene       │
│  ├─ serve ──→ MCP Server: 35 tools (24 read + 11 write)         │
│  ├─ start ──→ Dashboard + MCP one-click                        │
│  └─ experiments / query / compare ──→ Cross-experiment analysis  │
│                                                                  │
│  Decision Layers:                                                │
│  ┌─ Contract (hard boundary, human-defined)                     │
│  ├─ Agent (LLM, optional, within action space)                  │
│  ├─ Rules (deterministic, always-on fallback)                   │
│  ├─ MCP (external agent access, dual-mode delegation)           │
│  └─ Dashboard (remote config, dirty-flag user protection)       │
│                                                                  │
│  Training Process: python train.py (0 changes required)          │
└──────────────────────────────────────────────────────────────────┘
```
""")

    st.subheader("双模式架构")
    st.markdown("""
```text
┌─ Standalone (autonomous) ──────────────────────┐
│ guardian AgentAdvisor 自主决策                   │
│ 训练中：预设动作集，失败回退规则默认               │
│ 训练后：创造性策略（需用户确认）                   │
├─ MCP Delegated (provisional) ──────────────────┤
│ 外部 Claude Code 连接时，内置 agent 进入           │
│ provisional 模式：照常决策但标记为"临时"            │
│ → 外部 agent 可调用 resolve_decision 批准或覆盖   │
│ → 超时/断连 → 自动批准，恢复 autonomous            │
└──────────────────────────────────────────────────┘
```
""")

    st.subheader("配置三层结构")
    st.code("""DEFAULTS  <  guardian.yaml  <  GUARDIAN_* env vars  <  CLI flags""", language="text")

    st.subheader("MCP 远程训练场景")
    st.markdown("""
```text
┌─ 远程训练服务器 ───────────────────────┐
│  guarftrain serve --transport sse     │
│    → MCP server 监听 127.0.0.1:8766  │
└───────────────────────────────────────┘
         │ SSH 隧道
         │
┌─ 本地机器 ────────────────────────────┐
│  Claude Code → localhost:8766/sse    │
│  "查看当前训练状态"                      │
│    → get_training_status              │
│    → "epoch 47/100, loss=0.12"        │
└───────────────────────────────────────┘
```
""")

# ---------------------------------------------------------------------------
# 底部
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Training Guardian Agent v0.2.0 · "
    "16 模块 · ~10,500 行生产代码 · 221 测试 · 35 MCP 工具 · MIT License"
)
