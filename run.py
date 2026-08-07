"""cp_8 · CLI 入口。

默认路径是 watch——包装任意训练命令做进程外守护，被守护的脚本不需要
import guardian。`--` 之后的内容原样透传给训练命令，guardian 不解析，
只在重启时按 contract.cli_mappings 追加/替换需要调整的参数。
详见 checkpoint/cp_8.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from guardian.checkpoint_analyzer import CheckpointAnalyzer
from guardian.config import ConfigError, load_config
from guardian.logging_config import configure, get_logger
from guardian.monitor import TrainingMonitor
from guardian.notifier import Notifier, ensure_utf8_stdout
from guardian.resource_estimator import ResourceEstimator
from guardian.summary import SummaryGenerator
from guardian.task_contract import ContractError, TaskContract


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """按 `--` 切分：前面是 guardian 参数，后面是训练命令（原样透传）。"""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run.py",
        description="Training Guardian — sidecar-first 训练守护（训练脚本 0 行改动）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python run.py watch -- python train.py --epochs 20   # 默认路径\n"
            "  python run.py contract check\n"
            "  python run.py analyze\n"
            "  python run.py project init    # 自动探测项目路径"
        ),
    )
    p.add_argument("--config", default="configs/guardian.yaml")
    p.add_argument("--contract", default=None, help="默认取 config 里的 contract.path")
    p.add_argument("--project-dir", default=None, help="项目根目录（自动探测 checkpoints/logs/data 路径）")
    sub = p.add_subparsers(dest="command", required=True)

    w = sub.add_parser("watch", help="守护任意训练命令（默认主路径）")
    w.add_argument("--config", default="configs/guardian.yaml", help="guardian 配置文件路径")
    w.add_argument("--contract", default=None, help="契约文件路径")
    w.add_argument("--strict-contract", action="store_true", help="契约缺项即拒绝启动")
    w.add_argument("--no-monitor", action="store_true")
    w.add_argument("--max-retries", type=int, default=None)
    w.add_argument("--agent", action="store_true", help="启用 agent 决策层（需配置 API key）")
    w.add_argument("--with-mcp", action="store_true", help="watch 的同时后台启动 MCP server")
    w.add_argument("--with-dashboard", action="store_true", help="watch 的同时后台启动 Web 控制面板")

    c = sub.add_parser("contract", help="契约校验与审核")
    c.add_argument("action", choices=["check", "review"])

    a = sub.add_parser("analyze", help="分析已有 checkpoint（独立扫描，不需要训练进程）")
    a.add_argument("--metric", default="val/accuracy")
    a.add_argument("--lower-better", action="store_true")

    pf = sub.add_parser("preflight", help="训练前资源预检（需 buildable_entry 契约）")
    pf.add_argument("--device", default="cuda")
    pf.add_argument("--total-samples", type=int, default=60000, help="训练集样本数")
    pf.add_argument("--epochs", type=int, default=20)
    pf.add_argument("--target-batch-size", type=int, default=None)

    sv = sub.add_parser("serve", help="单独启动 MCP server（独立进程，跨进程读盘）")
    sv.add_argument("--transport", default="stdio",
                    choices=["stdio", "sse", "http", "tcp"],
                    help="stdio=标准输入输出 | sse=SSE over HTTP | http=Streamable HTTP | tcp=兼容旧名，等同 sse")
    sv.add_argument("--host", default="127.0.0.1", help="HTTP 监听地址（sse/http/tcp 时生效）")
    sv.add_argument("--port", type=int, default=None, help="HTTP 端口（默认取配置 mcp.tcp_port=8766）")

    # ---- start：一键启动 ----
    st = sub.add_parser("start", help="一键启动：Dashboard + MCP server（可选附带训练守护）",
                        epilog="示例:\n"
                               "  python run.py start\n"
                               "  python run.py start -- python train.py --epochs 20\n"
                               "  python run.py start --dash-port 8767 --mcp-port 8768 -- python train2.py")
    st.add_argument("--config", default="configs/guardian.yaml")
    st.add_argument("--contract", default=None)
    st.add_argument("--strict-contract", action="store_true")
    st.add_argument("--agent", action="store_true", help="启用 agent 决策层（转发给 watch）")
    st.add_argument("--no-monitor", action="store_true")
    st.add_argument("--max-retries", type=int, default=None)
    st.add_argument("--project-dir", default=None)
    st.add_argument("--host", default="127.0.0.1", help="服务监听地址")
    st.add_argument("--dash-port", type=int, default=None, help="面板端口，默认取配置 dashboard.port=8765")
    st.add_argument("--mcp-port", type=int, default=None, help="MCP HTTP 端口，默认取配置 mcp.tcp_port=8766")
    st.add_argument("--mcp-transport", choices=["sse", "http"], default="sse")
    st.add_argument("--no-dashboard", action="store_true")
    st.add_argument("--no-mcp", action="store_true")

    # ---- v2 子命令 (F3/F4/F7/F10) ----

    q = sub.add_parser("query", help="自然语言查询实验记录")
    q.add_argument("question", nargs="+", help="自然语言问题")
    q.add_argument("--agent", action="store_true", help="启用 AI 翻译")
    q.add_argument("--log-dir", default="./logs", help="日志目录（summary_*.json 所在位置）")
    q.add_argument("--name", default=None, help="手动设置实验名称前缀")
    q.add_argument("--project-dir", default=None, help="项目根目录")

    cmp = sub.add_parser("compare", help="对比两个实验")
    cmp.add_argument("id_a", help="第一个实验 ID")
    cmp.add_argument("id_b", help="第二个实验 ID")
    cmp.add_argument("--agent", action="store_true")
    cmp.add_argument("--log-dir", default="./logs", help="日志目录")
    cmp.add_argument("--name", default=None, help="手动设置实验名称前缀")
    cmp.add_argument("--project-dir", default=None, help="项目根目录")

    exp = sub.add_parser("experiments", help="列出所有历史实验")
    exp.add_argument("--limit", type=int, default=20)
    exp.add_argument("--log-dir", default="./logs", help="日志目录")
    exp.add_argument("--name", default=None, help="手动设置实验名称（覆盖 summary 自带名称）")
    exp.add_argument("--project-dir", default=None, help="项目根目录（自动读 .guardian-project.yaml）")

    viz = sub.add_parser("visualize", help="模型管线可视化（解析模型结构 → 生成 HTML）")
    viz.add_argument("--ckpt", type=int, help="checkpoint epoch")
    viz.add_argument("--model", help="模型入口，如 train:build_model")
    viz.add_argument("--output", default="./logs/model_viz.html")
    viz.add_argument("--agent", action="store_true", help="启用 AI 瓶颈分析和建议")

    gal = sub.add_parser("gallery", help="图片筛选与展示")
    gal.add_argument("--ckpt", type=int, required=True, help="checkpoint epoch")
    gal.add_argument("--data", default="./data/test", help="数据源路径")
    gal.add_argument("--config", help="复用已有 gallery_config.json")
    gal.add_argument("--output", default="./logs/gallery")
    gal.add_argument("--agent", action="store_true", help="启用 AI 策略提议")
    gal.add_argument("--ckpt-dir", default="./checkpoints", help="checkpoint 目录")
    gal.add_argument("--project-dir", default=None, help="项目根目录")

    inf = sub.add_parser("infer", help="模型推理测试（固定脚本，不生成代码）")
    inf.add_argument("--ckpt", type=int, help="checkpoint epoch（不指定则用 best）")
    inf.add_argument("--inputs", default=None, help="输入数据路径（默认取项目的 data_dir）")
    inf.add_argument("--task", choices=["classification", "detection", "segmentation"])
    inf.add_argument("--output", default="./logs/inference")
    inf.add_argument("--agent", action="store_true")
    inf.add_argument("--ckpt-dir", default="./checkpoints", help="checkpoint 目录")
    inf.add_argument("--project-dir", default=None, help="项目根目录")

    # ---- project 子命令 ----
    dash = sub.add_parser("dashboard", help="启动 Web 控制面板（独立进程）")
    dash.add_argument("--port", type=int, default=8765, help="HTTP 端口，默认 8765")
    dash.add_argument("--host", default="127.0.0.1")

    proj = sub.add_parser("project", help="项目上下文管理（自动探测路径）")
    proj.add_argument("action", choices=["init", "show", "scan", "fill"],
                      help="init=自动探测并保存 | show=显示当前 | scan=仅扫描 | fill=AI补全")
    proj.add_argument("path", nargs="?", default=".", help="项目路径（默认当前目录）")
    proj.add_argument("--agent", action="store_true", help="启用 AI 补全缺失项")

    return p


def _load(args) -> tuple[dict, TaskContract]:
    cfg = load_config(args.config)
    configure(cfg)  # 初始化全局日志配置（控制台 + 可选文件）
    for warn in cfg.get("_warnings", []):
        print(f"[配置] {warn}", flush=True)
    contract_path = args.contract or cfg["contract"].get("path")
    if getattr(args, "strict_contract", False):
        cfg["contract"]["strict_mode"] = True
    contract = TaskContract(cfg["contract"], contract_path)
    return cfg, contract


def cmd_contract(args) -> int:
    if args.action == "check":
        cfg, contract = _load(args)
        status = contract.validate_script_contract(ckpt_dir=cfg["project"]["ckpt_dir"])
        print(status.render(), flush=True)
        return 0
    if args.action == "review":
        return cmd_contract_review(args)
    return 2


def cmd_analyze(args) -> int:
    cfg, contract = _load(args)
    analyzer = CheckpointAnalyzer(
        {**cfg["checkpoint"], "stability_checks": 1},
        ckpt_dir=cfg["project"]["ckpt_dir"],
        contract=contract,
    )
    analyzer.poll()
    report = analyzer.report(args.metric, higher_better=not args.lower_better)
    if not report["total"]:
        print(f"未在 {cfg['project']['ckpt_dir']} 找到任何 checkpoint。", flush=True)
        return 1
    print(f"共 {report['total']} 个 checkpoint，最新 {report['latest']}", flush=True)
    print(f"判定指标: {report['metric']}", flush=True)
    for item in report["checkpoints"]:
        val = item["metrics"].get(args.metric)
        shown = f"{val:.4f}" if isinstance(val, (int, float)) else "-"
        mark = " <- best" if report["best"] and item["epoch"] == report["best"]["epoch"] else ""
        print(f"  cp_{item['epoch']:<5} {args.metric}={shown}{mark}", flush=True)
    return 0


def cmd_contract_review(args) -> int:
    """审核 agent 提议的注册表/白名单扩展条目（v1）。"""
    cfg, contract = _load(args)
    proposals = contract.list_proposals(status="pending") if hasattr(contract, "list_proposals") else []
    if not proposals:
        print("没有待审核的提议。", flush=True)
        return 0
    print(f"共 {len(proposals)} 条待审核提议：\n", flush=True)
    for p in proposals:
        pid = p.get("id", "?")
        kind = p.get("kind", "?")
        entry = p.get("entry", {})
        evidence = p.get("evidence", "")
        print(f"  [{pid}] {kind}: {json.dumps(entry, ensure_ascii=False)}", flush=True)
        if evidence:
            print(f"       依据: {evidence}", flush=True)
    print("\n使用 MCP 写工具 approve_contract_proposal / reject_contract_proposal 审核。", flush=True)
    return 0


def cmd_preflight(args) -> int:
    """训练前资源预检（cp_1）。依赖 contract.buildable_entry 契约项。"""
    cfg, contract = _load(args)
    status = contract.validate_script_contract(ckpt_dir=cfg["project"]["ckpt_dir"])
    if not status.is_ok("buildable_entry"):
        print("错误: 契约未声明 buildable_entry（model_fn / dataloader_fn），"
              "preflight 无法执行。", flush=True)
        print("在 configs/contract.yaml 的 script_contract.buildable_entry 中"
              "声明后可用的 import 入口。", flush=True)
        return 1

    entry = contract.script.get("buildable_entry") or {}
    model_ref = entry.get("model_fn", "")
    loader_ref = entry.get("dataloader_fn", "")

    # 从 train.py import
    try:
        mod_path, func_name = model_ref.split(":", 1) if ":" in model_ref else ("train", model_ref)
        import importlib
        mod = importlib.import_module(mod_path)
        model_fn = getattr(mod, func_name)
    except Exception as exc:
        print(f"错误: 无法 import {model_ref}: {exc}", flush=True)
        return 1

    try:
        mod_path, func_name = loader_ref.split(":", 1) if ":" in loader_ref else ("train", loader_ref)
        import importlib
        mod = importlib.import_module(mod_path)
        dataloader_fn = getattr(mod, func_name)
    except Exception as exc:
        print(f"错误: 无法 import {loader_ref}: {exc}", flush=True)
        return 1

    # 白名单上限
    upper = None
    for p in (contract.adjustable_paths or []):
        if p.get("path") == "dataloader.batch_size" and "max" in p:
            upper = int(p["max"])

    estimator = ResourceEstimator(cfg.get("preflight"))
    report = estimator.preflight_check(
        model_fn, dataloader_fn,
        device=args.device,
        total_samples=args.total_samples,
        epochs=args.epochs,
        target_batch_size=args.target_batch_size,
        batch_upper_bound=upper,
    )
    estimator.print_report(report)
    return 0


def cmd_serve(args) -> int:
    """独立 MCP server 进程（cp_10）。"""
    from guardian.mcp_server import GuardianMCPServer

    available, err = GuardianMCPServer.is_available()
    if not available:
        print(f"错误: {err}", flush=True)
        print("请 pip install -r requirements-mcp.txt 后重试。", flush=True)
        return 1

    cfg, contract = _load(args)
    server = GuardianMCPServer(
        cfg, mode="standalone",
        state_dir=cfg["project"]["log_dir"],
        task_contract=contract,
    )
    if args.transport in ("sse", "http", "tcp"):
        _port = args.port or int(cfg["mcp"]["tcp_port"])
        _path = "/sse" if args.transport in ("sse", "tcp") else "/mcp"
        print(f"[MCP] 监听 http://{args.host}:{_port}{_path}（Ctrl+C 退出）", flush=True)
    result = server.start(transport=args.transport, host=args.host, port=args.port)
    if result:
        print(result, flush=True)
    return 0


# ---------------------------------------------------------------------------
# 一键启动（cp_10 + dashboard）
# ---------------------------------------------------------------------------

def _port_in_use(host: str, port: int) -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def _print_start_banner(host: str, dash_port: int, mcp_port: int,
                        mcp_started: bool, dash_status: str,
                        train_cmd: list[str] | None) -> None:
    """dash_status: "started" | "reused" | "failed" """
    print("=" * 60, flush=True)
    print("  Training Guardian", flush=True)
    print("=" * 60, flush=True)
    if dash_status == "started":
        print(f"  Dashboard  : http://{host}:{dash_port}", flush=True)
    elif dash_status == "reused":
        print(f"  Dashboard  : http://{host}:{dash_port} （复用已有）", flush=True)
    else:
        print(f"  Dashboard  : 未启动（端口被占或依赖缺失）", flush=True)
    if mcp_started:
        print(f"  MCP (SSE)  : http://{host}:{mcp_port}/sse", flush=True)
    else:
        print(f"  MCP        : 未启动（端口被占或依赖缺失）", flush=True)
    if train_cmd:
        print(f"  训练命令    : {' '.join(train_cmd)}", flush=True)
    print("", flush=True)
    print("  本地接入（在新终端执行）：", flush=True)
    print(f"    ssh -L {dash_port}:127.0.0.1:{dash_port} -L {mcp_port}:127.0.0.1:{mcp_port} user@这台机器", flush=True)
    print("", flush=True)
    print("  然后：", flush=True)
    print(f"    浏览器打开  http://127.0.0.1:{dash_port}", flush=True)

    # 自动打开 Dashboard 前端页面
    if dash_status in ("started", "reused"):
        import webbrowser
        webbrowser.open(f"http://{host}:{dash_port}")
    print(f"    Claude Code 连接 http://127.0.0.1:{mcp_port}/sse", flush=True)
    print("", flush=True)
    print("  Ctrl+C 停止所有服务", flush=True)
    print("=" * 60, flush=True)


def cmd_start(args, train_cmd: list[str]) -> int:
    """一键启动：Dashboard + MCP server（standalone/SSE），可选附带训练守护。"""
    cfg, contract = _load(args)
    host = args.host
    dash_port = args.dash_port or int(cfg.get("dashboard", {}).get("port", 8765))
    mcp_port = args.mcp_port or int(cfg.get("mcp", {}).get("tcp_port", 8766))

    # ---- 依赖与端口检查（逐项降级，绝不崩溃）----
    mcp_ok = True
    if not args.no_mcp:
        from guardian.mcp_server import GuardianMCPServer
        mcp_ok, mcp_err = GuardianMCPServer.is_available()
        if not mcp_ok:
            print(f"[MCP] {mcp_err}", flush=True)
            print("[MCP] pip install -r requirements-mcp.txt 后即可启用，本次跳过。", flush=True)
        elif _port_in_use(host, mcp_port):
            print(f"[MCP] 端口 {host}:{mcp_port} 已被占用，MCP 未启动。", flush=True)
            print(f"[MCP] 请停止占用进程，或用 --mcp-port 指定其他端口。", flush=True)
            mcp_ok = False

    dash_ok = True
    if not args.no_dashboard:
        try:
            import fastapi, uvicorn  # noqa: F401
        except ImportError:
            print("[Dashboard] fastapi/uvicorn 未安装，面板跳过。", flush=True)
            print("[Dashboard] pip install -r requirements-dashboard.txt 后即可启用。", flush=True)
            dash_ok = False
        if dash_ok and _port_in_use(host, dash_port):
            print(f"[Dashboard] 端口 {host}:{dash_port} 已有服务，复用现有面板（不重复启动）。", flush=True)

    if not mcp_ok and not dash_ok and not train_cmd:
        print("错误: 没有任何服务可启动。请按上述提示安装依赖后重试。", flush=True)
        return 1

    # ---- 启动 Dashboard（后台线程）----
    dash_status = "failed"
    dash_reused = dash_ok and _port_in_use(host, dash_port)
    if dash_ok and not dash_reused:
        try:
            from guardian.dashboard import DashboardServer
            dash = DashboardServer(config=cfg, port=dash_port, host=host)
            dash.start(blocking=False)
            import time as _time
            _time.sleep(0.5)  # 等面板绑定端口
            dash_status = "started"
        except Exception as exc:
            print(f"[Dashboard] 启动失败: {exc}", flush=True)
    elif dash_reused:
        dash_status = "reused"

    # ---- 启动 MCP（standalone 跨进程读盘 + SSE）----
    mcp_started = False
    if mcp_ok and not _port_in_use(host, mcp_port):
        try:
            from guardian.mcp_server import GuardianMCPServer
            srv = GuardianMCPServer(
                cfg, mode="standalone",
                state_dir=cfg["project"]["log_dir"],
                task_contract=contract,
            )
            th = srv.start_in_background(transport=args.mcp_transport,
                                         host=host, port=mcp_port)
            if th is not None:
                mcp_started = True
        except Exception as exc:
            print(f"[MCP] 启动失败: {exc}", flush=True)

    _print_start_banner(host, dash_port, mcp_port, mcp_started, dash_status,
                        train_cmd if train_cmd else None)

    # ---- 可选：附带训练守护（复用 cmd_watch，面板走"复用+注册"路径）----
    if train_cmd:
        import types
        watch_args = types.SimpleNamespace(
            config=args.config, contract=args.contract, project_dir=args.project_dir,
            strict_contract=args.strict_contract, no_monitor=args.no_monitor,
            max_retries=args.max_retries, agent=args.agent,
            with_mcp=False,        # 已有独立 SSE MCP，避免第二个 stdio MCP 抢终端
            with_dashboard=True,   # cmd_watch 检测到端口占用即复用并注册进程
            dash_port=dash_port,
        )
        return cmd_watch(watch_args, train_cmd)

    # ---- 前台常驻，Ctrl+C 优雅退出 ----
    try:
        while True:
            import time as _time
            _time.sleep(1)
    except KeyboardInterrupt:
        print("\n[守护] 收到中断，服务已停止。", flush=True)
    return 0


def cmd_watch(args, train_cmd: list[str]) -> int:
    if not train_cmd:
        print("用法: python run.py watch -- <训练命令>\n"
              "例如: python run.py watch -- python train.py --epochs 20", flush=True)
        return 2

    cfg, contract = _load(args)

    # ---- 启动前守卫：必须有显式配置或项目上下文 ----
    from guardian.project_context import ProjectContext
    ctx = ProjectContext(args.project_dir or ".")
    has_explicit_config = args.config != "configs/guardian.yaml" or args.contract is not None
    has_project_file = ctx.detected_by != "none"

    if not has_explicit_config and not has_project_file:
        print("=" * 56, flush=True)
        print("  错误: 未找到配置，无法启动守护训练。", flush=True)
        print("", flush=True)
        print("  请选择以下方式之一：", flush=True)
        print("", flush=True)
        print("  1. 初始化项目（推荐）：", flush=True)
        print("     python run.py project init <项目路径>", flush=True)
        print("", flush=True)
        print("  2. 显式指定配置：", flush=True)
        print("     python run.py watch --config <config.yaml> --contract <contract.yaml> -- <训练命令>", flush=True)
        print("", flush=True)
        print("  3. 在已有项目目录中运行：", flush=True)
        print("     cd <项目目录> && python ../guarftrain/run.py watch -- <训练命令>", flush=True)
        print("=" * 56, flush=True)
        return 1

    project = cfg["project"]

    # 项目上下文自动补全路径
    if has_project_file:
        if ctx.ckpt_dir and project.get("ckpt_dir") == "./checkpoints":
            project["ckpt_dir"] = ctx.ckpt_dir
        if ctx.log_dir and project.get("log_dir") == "./logs":
            project["log_dir"] = ctx.log_dir
        # 项目有 contract 配置则优先
        proj_contract = ctx.data.get("contract", {}).get("path")
        if proj_contract and not args.contract:
            cfg["contract"]["path"] = proj_contract

    ckpt_dir = project["ckpt_dir"]

    # --with-mcp：后台启动 MCP server
    mcp_thread = None
    if args.with_mcp:
        from guardian.mcp_server import GuardianMCPServer
        available, err = GuardianMCPServer.is_available()
        if not available:
            print(f"[MCP] {err}", flush=True)
            print("[MCP] 训练照常进行，仅外部 agent 接入不可用。", flush=True)
        else:
            print("[MCP] 将在 watchdog 就绪后后台启动 ...", flush=True)

    # 启动前校验契约，逐项打印开启/降级状态
    try:
        status = contract.validate_script_contract(train_cmd=train_cmd, ckpt_dir=ckpt_dir)
    except ContractError as exc:
        print(str(exc), flush=True)
        return 1
    print(status.render(), flush=True)
    print(flush=True)

    # 加载凭据（JSON 文件或环境变量）
    _load_creds(args)

    # agent 决策层（v1）
    advisor = None
    if args.agent:
        from guardian.agent_advisor import AgentAdvisor
        # 强制启用配置节，让 AgentAdvisor 的 _has_credentials() 做凭据检测
        cfg["agent"]["enabled"] = True
        advisor = AgentAdvisor(cfg["agent"])
        if not advisor.is_enabled():
            print("[agent] 决策层未启用：未检测到 API 凭据。"
                  "请设置 ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN 环境变量。",
                  flush=True)
            print("[agent] 训练将以纯规则模式继续。", flush=True)
            advisor = None
        else:
            print(f"[agent] 决策层已启用（provider={advisor.provider}, "
                  f"model={advisor._get_model_id()}）", flush=True)

    notifier = Notifier(cfg["notifier"])
    monitor = None
    if not args.no_monitor and cfg["monitor"].get("enabled", True):
        monitor = TrainingMonitor(cfg["monitor"], notifier, contract=contract, advisor=advisor,
                                  on_intervention=None)  # 下文 watchdog 创建后回填
        if not monitor.enabled:
            print("[监控] 指标通道不可用，退化为进程级看护（存活 + 崩溃恢复）", flush=True)

    analyzer = CheckpointAnalyzer(cfg["checkpoint"], ckpt_dir=ckpt_dir, contract=contract)

    wd_cfg = dict(cfg["watchdog"])
    if args.max_retries is not None:
        wd_cfg["max_retries"] = args.max_retries

    from guardian.watchdog import TrainingWatchdog
    watchdog = TrainingWatchdog(
        wd_cfg, notifier, contract=contract, ckpt_dir=ckpt_dir,
        advisor=advisor,
        # 让 watchdog 能算出重启作废了多少 epoch（无指标通道时为 None，不猜）
        progress_fn=(monitor.current_step if monitor is not None else None),
    )

    # 回填 monitor → watchdog 的干预通道：agent 决策的重启式动作经此转发
    if monitor is not None and advisor is not None:
        monitor.on_intervention = watchdog.request_intervention

    summary_gen = SummaryGenerator(project, monitor, analyzer, watchdog, advisor=advisor)

    # --with-mcp：在 watchdog 就绪后后台启动
    if args.with_mcp:
        from guardian.mcp_server import GuardianMCPServer
        _avail, _err = GuardianMCPServer.is_available()
        if _avail:
            mcp_srv = GuardianMCPServer(
                cfg, monitor=monitor, ckpt_analyzer=analyzer,
                watchdog=watchdog, summary_gen=summary_gen, advisor=advisor,
                task_contract=contract,
                mode="shared",
            )
            mcp_thread = mcp_srv.start_in_background(transport="stdio")
            if mcp_thread is not None:
                print("[MCP] 已在后台线程启动，外部 agent 客户端可接入。", flush=True)
        # 不可用时已在上面打印过提示，此处不再重复

    # --with-dashboard：后台启动面板（如未运行）并注册当前进程
    dash_server = None
    dash_url = None
    if args.with_dashboard:
        import socket as _socket
        import urllib.request as _ur
        dash_port = getattr(args, "dash_port", 8765)
        # 检测面板是否已在运行
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        port_in_use = s.connect_ex(('127.0.0.1', dash_port)) == 0
        s.close()
        if not port_in_use:
            try:
                from guardian.dashboard import DashboardServer
                dash_server = DashboardServer(config=cfg, port=dash_port, host="127.0.0.1")
                dash_server.start(blocking=False)
                import time as _time
                _time.sleep(0.5)  # 等面板启动
                print(f"[Dashboard] http://127.0.0.1:{dash_port}", flush=True)
            except Exception as exc:
                print(f"[Dashboard] 启动失败: {exc}", flush=True)
                dash_server = None
        else:
            print(f"[Dashboard] 复用已有面板 http://127.0.0.1:{dash_port}", flush=True)

        # 向面板注册当前进程（无论面板是新启动还是已有）
        dash_url = f"http://127.0.0.1:{dash_port}"
        process_id = project.get("name", "guardian-run")
        try:
            import json as _json
            req = _ur.Request(
                f"{dash_url}/api/register",
                data=_json.dumps({
                    "process_id": process_id,
                    "name": project.get("name", "guardian-run"),
                    "status": "starting",
                    "command": " ".join(train_cmd),
                    "model_entry": contract.script.get("buildable_entry", {}).get("model_fn", ""),
                    "project_dir": str(contract.path.parent.parent.resolve()) if contract.path else "",
                    "extra_paths": getattr(ctx, 'extra_paths', []),
                    "log_file": str(Path(project["log_dir"]) / "train.log"),
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            _ur.urlopen(req, timeout=3)
        except Exception:
            pass  # 面板不可达，不影响训练

    def on_tick(_wd, _proc) -> None:
        if monitor is not None:
            monitor.poll_metrics()
            # Dashboard: push via HTTP
            if dash_url and monitor.enabled:
                try:
                    import json as _json, urllib.request as _ur
                    hist = monitor.get_metrics_history()
                    if hist:
                        _ur.urlopen(_ur.Request(
                            f"{dash_url}/api/process/{process_id}/push",
                            data=_json.dumps({"type": "metrics", "data": hist[-1],
                                "patch": {"latest_metrics": hist[-1],
                                    "epoch": hist[-1].get("epoch") or hist[-1].get("step"),
                                    "anomaly_count": len(monitor.get_anomaly_history())}}).encode(),
                            headers={"Content-Type": "application/json"},
                        ), timeout=2)
                    gpu_hist = getattr(monitor, "get_gpu_history", lambda: [])()
                    if gpu_hist:
                        _ur.urlopen(_ur.Request(
                            f"{dash_url}/api/process/{process_id}/push",
                            data=_json.dumps({"patch": {"latest_gpu": gpu_hist[-1]}}).encode(),
                            headers={"Content-Type": "application/json"},
                        ), timeout=2)
                except Exception:
                    pass
        analyzer.poll()

    if dash_url:
        try:
            import json as _json, urllib.request as _ur
            _ur.urlopen(_ur.Request(
                f"{dash_url}/api/process/{process_id}/push",
                data=_json.dumps({"patch": {"status": "running"}}).encode(),
                headers={"Content-Type": "application/json"},
            ), timeout=3)
        except Exception:
            pass

    print(f"[守护] {' '.join(train_cmd)}", flush=True)
    try:
        result = watchdog.run(train_cmd, on_tick=on_tick)
    except KeyboardInterrupt:
        watchdog.stop()
        result = {"status": "stopped", "exit_code": None}
        print("\n[守护] 收到中断信号", flush=True)

    if monitor is not None:
        monitor.poll_metrics()
    analyzer.poll()

    if dash_url:
        try:
            import json as _json, urllib.request as _ur
            _ur.urlopen(_ur.Request(
                f"{dash_url}/api/process/{process_id}/push",
                data=_json.dumps({"patch": {"status": "completed" if result.get("status") == "completed" else "failed"}}).encode(),
                headers={"Content-Type": "application/json"},
            ), timeout=3)
        except Exception:
            pass

    summary = summary_gen.generate(result)
    print(flush=True)
    summary_gen.print_summary(summary)
    try:
        jpath, _ = summary_gen.save_summary(summary, project["log_dir"])
        print(f"摘要已保存: {jpath}", flush=True)
    except OSError as exc:
        print(f"摘要保存失败: {exc}", flush=True)

    return 0 if result.get("status") == "completed" else 1


# ---------------------------------------------------------------------------
# v2 子命令 (F3/F4/F7/F10)
# ---------------------------------------------------------------------------

def _get_log_dir(args) -> str:
    """智能获取 log_dir：CLI 参数 > 项目配置 > 默认。"""
    from guardian.project_context import ProjectContext
    start = getattr(args, "project_dir", None) or "."
    ctx = ProjectContext(start)
    cli_val = getattr(args, "log_dir", "./logs")
    # CLI 显式覆盖或非默认值时优先
    if cli_val != "./logs":
        return cli_val
    if ctx.log_dir and ctx.log_dir not in ("./checkpoints", "./logs"):
        return ctx.log_dir
    if ctx.detected_by != "none":
        return ctx.log_dir
    return cli_val


def _get_ckpt_dir(args) -> str:
    """智能获取 ckpt_dir。"""
    from guardian.project_context import ProjectContext
    start = getattr(args, "project_dir", None) or "."
    ctx = ProjectContext(start)
    cli_val = getattr(args, "ckpt_dir", "./checkpoints")
    if cli_val != "./checkpoints":
        return cli_val
    if ctx.ckpt_dir and ctx.ckpt_dir not in ("./checkpoints", "./logs"):
        return ctx.ckpt_dir
    if ctx.detected_by != "none":
        return ctx.ckpt_dir
    return cli_val


def cmd_experiments(args) -> int:
    """列出所有历史实验。"""
    from guardian.experiment_query import ExperimentQuery
    log_dir = _get_log_dir(args)
    cfg = {"log_dir": log_dir}
    if args.name:
        cfg["name"] = args.name
    eq = ExperimentQuery(cfg)
    exps = eq.list_experiments(limit=args.limit)
    if not exps:
        print("暂无实验记录。", flush=True)
        return 0
    print(f"{'ID':<30} {'状态':<12} {'最佳指标':<20} {'用时':<12}")
    print("-" * 74)
    for e in exps:
        metric = ""
        if e.get("best_metric_name") and e.get("best_metric_value") is not None:
            metric = f"{e['best_metric_name']}={e['best_metric_value']}"
        print(f"{e['experiment_id']:<30} {e['status']:<12} {metric:<20} {e.get('duration', '-'):<12}")
    return 0


def cmd_query(args) -> int:
    """自然语言查询实验记录。"""
    from guardian.experiment_query import ExperimentQuery
    from guardian.agent_advisor import AgentAdvisor

    question = " ".join(args.question)

    advisor = None
    if args.agent:
        cfg = load_config(args.config)
        cfg["agent"]["enabled"] = True
        advisor = AgentAdvisor(cfg["agent"])
        if not advisor.is_enabled():
            print("[agent] AI 未启用（无凭据），使用模板查询。", flush=True)
            advisor = None

    log_dir = _get_log_dir(args)
    cfg = {"log_dir": log_dir}
    if args.name:
        cfg["name"] = args.name
    eq = ExperimentQuery(cfg, advisor=advisor)
    result = eq.query(question)

    print(f"问题: {result['question']}", flush=True)
    if result.get("interpretation"):
        print(f"理解: {result['interpretation']}", flush=True)
    print(f"来源: {result['source']}", flush=True)
    print(flush=True)
    print(result["answer"], flush=True)

    if result.get("results"):
        print(flush=True)
        for r in result["results"]:
            print(json.dumps(r, ensure_ascii=False, indent=2), flush=True)

    return 0


def cmd_compare(args) -> int:
    """对比两个实验。"""
    from guardian.experiment_query import ExperimentQuery
    from guardian.agent_advisor import AgentAdvisor

    advisor = None
    if args.agent:
        cfg = load_config(args.config)
        cfg["agent"]["enabled"] = True
        advisor = AgentAdvisor(cfg["agent"])
        if not advisor.is_enabled():
            advisor = None

    log_dir = _get_log_dir(args)
    cfg = {"log_dir": log_dir}
    if args.name:
        cfg["name"] = args.name
    eq = ExperimentQuery(cfg, advisor=advisor)
    result = eq.compare(args.id_a, args.id_b)

    if "error" in result:
        print(f"错误: {result['error']}", flush=True)
        return 1

    print(f"实验 A: {result['experiment_a']}", flush=True)
    print(f"实验 B: {result['experiment_b']}", flush=True)
    print(flush=True)

    if result.get("diffs"):
        print("指标差异:", flush=True)
        for d in result["diffs"]:
            direction = "↑" if d["delta"] > 0 else "↓" if d["delta"] < 0 else "="
            print(f"  {d['field']}: {d['a']} → {d['b']} ({direction}{abs(d['delta'])})", flush=True)

    if result.get("param_diffs"):
        print(flush=True)
        print("参数差异:", flush=True)
        for k, v in result["param_diffs"].items():
            print(f"  {k}: {v['a']} → {v['b']}", flush=True)

    if result.get("analysis"):
        print(flush=True)
        print(f"AI 分析: {result['analysis']}", flush=True)

    return 0


def cmd_visualize(args) -> int:
    """模型管线可视化。"""
    from guardian.model_viz import ModelVisualizer
    from guardian.agent_advisor import AgentAdvisor

    advisor = None
    if args.agent:
        cfg = load_config(args.config)
        cfg["agent"]["enabled"] = True
        advisor = AgentAdvisor(cfg["agent"])
        if not advisor.is_enabled():
            print("[agent] AI 未启用，使用默认可视化配置。", flush=True)
            advisor = None

    mv = ModelVisualizer(advisor=advisor)

    # 获取 model_fn
    model_fn = None
    if args.model:
        try:
            mod_path, fn_name = args.model.split(":", 1)
            # 确保 scripts/ 在 sys.path 中
            import sys as _sys
            _scripts_dir = str(Path(mod_path).parent) if "/" in mod_path or "\\" in mod_path else None
            if _scripts_dir and _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            # 将路径转为模块名（如 scripts/clip_adapter → clip_adapter）
            if "/" in mod_path or "\\" in mod_path:
                mod_path = mod_path.replace("\\", "/").replace("/", ".").removesuffix(".py")
            import importlib
            mod = importlib.import_module(mod_path)
            model_fn = getattr(mod, fn_name)
        except Exception as exc:
            print(f"错误: 无法 import {args.model}: {exc}", flush=True)
            return 1
    else:
        # 尝试从 train.py 加载
        try:
            from train import build_model
            model_fn = build_model
        except Exception:
            print("错误: 需要 --model 参数指定模型入口，如 --model train:build_model", flush=True)
            return 1

    # 解析 + 统计
    graph = mv.parse_model(model_fn)
    if "error" in graph:
        print(f"错误: {graph['error']}", flush=True)
        return 1

    stats = mv.compute_stats(graph)

    # AI 提议
    viz_config = mv.propose_config(graph, stats)
    improvements = mv.propose_improvements(stats, viz_config)
    print(mv.print_proposal(viz_config, stats, improvements), flush=True)

    # 用户确认
    print(flush=True)
    try:
        user_input = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n已取消。", flush=True)
        return 0

    if user_input.lower() in ("cancel", "c", "no", "n"):
        print("已取消。", flush=True)
        return 0

    if user_input and user_input.lower() not in ("", "y", "yes", "ok"):
        # NL 修正 → 重新提议
        print(f"[agent] 根据反馈重新生成配置 ...", flush=True)
        viz_config = mv.propose_config(graph, stats, user_feedback=user_input)
        print(flush=True)
        print(mv.print_proposal(viz_config, stats), flush=True)
        try:
            user_input2 = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。", flush=True)
            return 0
        if user_input2.lower() in ("cancel", "c", "no", "n"):
            print("已取消。", flush=True)
            return 0

    # 渲染
    output = mv.render_html(graph, stats, viz_config, args.output)
    print(f"HTML 已生成: {output}", flush=True)

    # 尝试打开浏览器
    try:
        import webbrowser
        webbrowser.open(f"file://{output.resolve()}")
    except Exception:
        pass

    return 0


def cmd_gallery(args) -> int:
    """图片筛选与展示。"""
    from guardian.gallery import GalleryManager
    from guardian.inference import InferenceRunner
    from guardian.agent_advisor import AgentAdvisor

    advisor = None
    if args.agent:
        cfg = load_config(args.config)
        cfg["agent"]["enabled"] = True
        advisor = AgentAdvisor(cfg["agent"])
        if not advisor.is_enabled():
            print("[agent] AI 未启用，使用默认筛选策略。", flush=True)
            advisor = None

    gm = GalleryManager(advisor=advisor)

    # 加载或生成策略
    if args.config:
        strategies = gm.load_config(args.config)
        if strategies is None:
            print(f"错误: 配置文件不存在或无效: {args.config}", flush=True)
            return 1
        print(f"已加载配置: {args.config}", flush=True)
    else:
        task_type = gm.infer_task_type()
        print(f"推断任务类型: {task_type}", flush=True)
        strategies = gm.propose_strategies(task_type)
        print(flush=True)
        print(gm.render_proposal(strategies), flush=True)

        # 确认交互
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已取消。", flush=True)
            return 0

        if user_input.lower() in ("cancel", "c"):
            print("已取消。", flush=True)
            return 0

        if user_input.lower() in ("export", "e"):
            path = args.output + "/gallery_config.json"
            gm.export_config(strategies, path)
            print(f"配置已导出: {path}", flush=True)
            return 0

        if user_input and user_input.lower() not in ("", "y", "yes"):
            # NL 修正
            print(f"[agent] 根据反馈重新生成策略 ...", flush=True)
            strategies = gm.propose_strategies(
                task_type, user_feedback=user_input,
            )
            print(flush=True)
            print(gm.render_proposal(strategies), flush=True)
            try:
                user_input2 = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n已取消。", flush=True)
                return 0
            if user_input2.lower() in ("cancel", "c"):
                print("已取消。", flush=True)
                return 0
            if user_input2.lower() in ("export", "e"):
                path = args.output + "/gallery_config.json"
                gm.export_config(strategies, path)
                print(f"配置已导出: {path}", flush=True)
                return 0

    # 保存配置
    config_path = args.output + "/gallery_config.json"
    gm.export_config(strategies, config_path)

    # 执行
    print(flush=True)
    print("正在执行推理 + 筛选 ...", flush=True)

    ckpt_path = f"{args.ckpt_dir}/cp_{args.ckpt}/model.pth"
    if not Path(ckpt_path).exists():
        print(f"错误: checkpoint 不存在: {ckpt_path}", flush=True)
        return 1

    ir = InferenceRunner()
    results = gm.execute(ckpt_path, strategies, args.data, inference_runner=ir)

    if "error" in results:
        print(f"错误: {results['error']}", flush=True)
        return 1

    # 摘要
    print(flush=True)
    print("=" * 56)
    print("  筛选完成")
    for name, images in results.items():
        print(f"  {name}: {len(images)} 张")
    print("=" * 56)

    # 保存结果
    result_path = Path(args.output) / "gallery_results.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"结果已保存: {result_path}", flush=True)
    print(f"配置已保存: {config_path}", flush=True)

    return 0


def cmd_infer(args) -> int:
    """推理测试（固定脚本）。"""
    from guardian.inference import InferenceRunner
    from guardian.checkpoint_analyzer import CheckpointAnalyzer
    from guardian.agent_advisor import AgentAdvisor

    advisor = None
    if args.agent:
        cfg = load_config(args.config)
        cfg["agent"]["enabled"] = True
        advisor = AgentAdvisor(cfg["agent"])
        if not advisor.is_enabled():
            advisor = None

    ir = InferenceRunner()

    # 确定 checkpoint
    ckpt_epoch = args.ckpt
    ckpt_dir = Path(_get_ckpt_dir(args))

    # inputs 默认取项目 data_dir
    inputs = args.inputs
    if inputs is None:
        from guardian.project_context import ProjectContext
        ctx = ProjectContext(getattr(args, "project_dir", None) or ".")
        inputs = ctx.data_dir
        print(f"输入路径（自动）: {inputs}", flush=True)

    if ckpt_epoch is None:
        # 自动选 best
        analyzer = CheckpointAnalyzer({}, ckpt_dir=str(ckpt_dir))
        analyzer.poll()
        best_epoch = ir.recommend_checkpoint(analyzer, advisor)
        if best_epoch is None:
            print("错误: 无法确定 checkpoint。请用 --ckpt 指定。", flush=True)
            return 1
        ckpt_epoch = best_epoch
        print(f"使用 best checkpoint: cp_{ckpt_epoch}", flush=True)

    ckpt_path = f"{args.ckpt_dir}/cp_{ckpt_epoch}/model.pth"
    if not Path(ckpt_path).exists():
        print(f"错误: checkpoint 不存在: {ckpt_path}", flush=True)
        return 1

    # 确定任务类型
    task_type = args.task
    if task_type is None:
        # 尝试自动检测
        try:
            from train import build_model
            task_type = ir.detect_task_type(build_model)
        except Exception:
            task_type = "classification"
        print(f"推断任务类型: {task_type}", flush=True)

    print(f"Checkpoint: cp_{ckpt_epoch}", flush=True)
    print(f"任务类型: {task_type}", flush=True)
    print(f"输入路径: {inputs}", flush=True)
    print(flush=True)

    result = ir.run(
        checkpoint_path=ckpt_path,
        task_type=task_type,
        inputs=inputs,
        output_dir=args.output,
    )

    if result.get("status") == "completed":
        print(f"推理完成: {result.get('num_inputs', '?')} 张图片", flush=True)
        if result.get("results_file"):
            print(f"结果文件: {result['results_file']}", flush=True)
        return 0
    else:
        print(f"推理失败: {result.get('error', result.get('stderr', '未知错误'))}", flush=True)
        return 1


def cmd_dashboard(args) -> int:
    """启动 Web 控制面板（独立进程）。"""
    from guardian.dashboard import DashboardServer
    cfg = load_config(args.config)
    ds = DashboardServer(config=cfg, port=args.port, host=args.host)
    print(f"Dashboard: http://{args.host}:{args.port}", flush=True)
    ds.start(blocking=True)
    return 0


def cmd_project(args) -> int:
    """项目上下文管理。"""
    from guardian.project_context import ProjectContext
    from guardian.agent_advisor import AgentAdvisor

    start_dir = args.path or "."

    if args.action == "show":
        ctx = ProjectContext(start_dir)
        print(ctx.status(), flush=True)
        return 0

    if args.action == "scan":
        ctx = ProjectContext(start_dir)
        detected = ctx._scan(Path(start_dir))
        if detected:
            print("自动探测结果:", flush=True)
            print(yaml.safe_dump(detected, allow_unicode=True), flush=True)
        else:
            print("未发现标准项目结构（无 checkpoints/logs/data 目录）。", flush=True)
        return 0

    if args.action == "init":
        # 强制重扫（不用已有配置）
        ctx = ProjectContext(start_dir)
        scanned = ctx._scan(Path(start_dir).resolve())
        if scanned:
            ctx.data = scanned

        # AI 补全
        if args.agent:
            advisor = _make_advisor(args)
            if advisor:
                ctx.advisor = advisor
                filled = ctx.fill_with_agent()
                if filled:
                    print("[agent] AI 已补全缺失字段", flush=True)

        # 保存
        if ctx.detected_by == "none":
            print("未发现项目结构，创建最小配置 ...", flush=True)
            ctx.data["project"] = {"name": Path(start_dir).resolve().name,
                                    "ckpt_dir": "./checkpoints",
                                    "log_dir": "./logs",
                                    "data_dir": "./data"}
            ctx.data["_detected_by"] = "user"

        path = ctx.save()
        print(f"项目配置已保存: {path}", flush=True)
        print(ctx.status(), flush=True)
        return 0

    if args.action == "fill":
        ctx = ProjectContext(start_dir)
        advisor = _make_advisor(args) if args.agent else None
        if advisor:
            ctx.advisor = advisor
            if ctx.fill_with_agent():
                print("[agent] AI 已补全:\n" + ctx.status(), flush=True)
            else:
                print("无缺失项或 AI 不可用。", flush=True)
        else:
            print("需要 --agent 且配置 API key。", flush=True)
        return 0

    return 2


def _load_creds(args):
    """加载凭据文件并写入环境变量。"""
    from guardian.credentials import load_credentials, apply_credentials
    start = getattr(args, "project_dir", None) or "."
    cred = load_credentials(start)
    if cred:
        apply_credentials(cred)
        return True
    return False


def _apply_project_paths(args):
    """加载项目上下文并补全 sys.path（使 model_entry 可导入）。"""
    from guardian.project_context import ProjectContext
    start = getattr(args, "project_dir", None) or "."
    ctx = ProjectContext(start)
    if ctx.detected_by != "none":
        ctx.apply_paths()
        return ctx
    return None


def _make_advisor(args):
    """从 args 构建 advisor（复用 watch 的模式）。"""
    _load_creds(args)
    from guardian.agent_advisor import AgentAdvisor
    cfg = load_config(args.config)
    cfg["agent"]["enabled"] = True
    advisor = AgentAdvisor(cfg["agent"])
    if not advisor.is_enabled():
        print("[agent] 未检测到 API 凭据，AI 不可用。", flush=True)
        return None
    print(f"[agent] 已启用（provider={advisor.provider}）", flush=True)
    return advisor


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ensure_utf8_stdout()
    raw = list(sys.argv[1:] if argv is None else argv)
    guardian_argv, train_cmd = split_argv(raw)

    parser = build_parser()
    args = parser.parse_args(guardian_argv)

    try:
        if args.command == "watch":
            return cmd_watch(args, train_cmd)
        if args.command == "contract":
            return cmd_contract(args)
        if args.command == "analyze":
            return cmd_analyze(args)
        if args.command == "preflight":
            return cmd_preflight(args)
        if args.command == "serve":
            return cmd_serve(args)
        if args.command == "start":
            return cmd_start(args, train_cmd)
        # v2
        if args.command == "experiments":
            return cmd_experiments(args)
        if args.command == "query":
            return cmd_query(args)
        if args.command == "compare":
            return cmd_compare(args)
        if args.command == "visualize":
            return cmd_visualize(args)
        if args.command == "gallery":
            return cmd_gallery(args)
        if args.command == "infer":
            return cmd_infer(args)
        if args.command == "project":
            return cmd_project(args)
        if args.command == "dashboard":
            return cmd_dashboard(args)
    except (ConfigError, ContractError) as exc:
        print(f"错误: {exc}", flush=True)
        return 1
    parser.error(f"未知子命令 {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
