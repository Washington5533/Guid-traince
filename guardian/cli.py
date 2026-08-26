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

import yaml

from . import __version__
from .checkpoint_analyzer import CheckpointAnalyzer
from .config import ConfigError, load_config
from .logging_config import configure, get_logger
from .monitor import TrainingMonitor
from .notifier import Notifier, ensure_utf8_stdout
from .resource_estimator import ResourceEstimator
from .summary import SummaryGenerator
from .task_contract import ContractError, TaskContract


def split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """按 `--` 切分：前面是 guardian 参数，后面是训练命令（原样透传）。"""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, []


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="guarftrain",
        description="Training Guardian — sidecar-first 训练守护（训练脚本 0 行改动）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  guarftrain init                                # 初始化项目\n"
            "  guarftrain watch -- python train.py --epochs 20 # 守护训练\n"
            "  guarftrain start                               # Dashboard + MCP\n"
            "  guarftrain check                               # 环境检查\n"
            "\n"
            "兼容旧用法: python run.py <command> ..."
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
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
    w.add_argument("--autonomy", choices=["supervised", "auto", "full"], default="supervised",
                    help="Sub-agent 自主权限：supervised=高风险需审批 | auto=自动调整参数 | full=全自动")
    w.add_argument("--remote", action="store_true", help="启动远程通信服务（算力服务器端，供 PC 端连接）")
    w.add_argument("--remote-port", type=int, default=8765, help="远程通信端口，默认 8765")
    w.add_argument("--remote-host", default="0.0.0.0", help="远程监听地址，默认 0.0.0.0")
    w.add_argument("--remote-auth", default=None, help="远程通信鉴权 token")
    w.add_argument("--remote-keepalive", type=int, default=60,
                   help="训练结束后远程服务保持在线秒数（PC 端补拉最终状态），默认 60")
    w.add_argument("--attach-pid", type=int, default=None,
                   help="附加到已有训练进程 PID（不自行启动）")
    w.add_argument("--screen", default=None, metavar="SESSION",
                   help="在 screen 会话中启动训练（SSH 断开后继续守护）")
    w.add_argument("--tmux", default=None, metavar="SESSION",
                   help="在 tmux 会话中启动训练")
    w.add_argument("--persist-agent", default=None, metavar="PATH",
                   help="SubAgent 状态持久化文件路径（断点续守）")

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

    arch = sub.add_parser("analyze_architecture",
                          help="模型架构分析（FLOPs/参数量/瓶颈层/D3 数据）")
    arch.add_argument("--model", help="模型入口，如 train:build_model（默认从 contract 解析）")
    arch.add_argument("--project-dir", default=None, help="项目根目录（Python 导入路径）")
    arch.add_argument("--output", default=None, help="将分析结果 JSON 写入指定文件")
    arch.add_argument("--html", default=None, help="生成 D3 可视化 HTML（如 ./logs/arch.html）")

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

    rem = sub.add_parser("remote", help="启动远程通信服务（算力服务器端，供 PC 端 Dashboard 连接）")
    rem.add_argument("--port", type=int, default=8765, help="HTTP 端口，默认 8765")
    rem.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0（局域网可访问）")
    rem.add_argument("--auth", default=None, help="鉴权 token（可选）")
    rem.add_argument("--config", default="configs/guardian.yaml")

    proj = sub.add_parser("project", help="项目上下文管理（自动探测路径）")
    proj.add_argument("action", choices=["init", "show", "scan", "fill"],
                      help="init=自动探测并保存 | show=显示当前 | scan=仅扫描 | fill=AI补全")
    proj.add_argument("path", nargs="?", default=".", help="项目路径（默认当前目录）")
    proj.add_argument("--agent", action="store_true", help="启用 AI 补全缺失项")

    # ---- init：project init 的顶级别名 ----
    ini = sub.add_parser("init", help="初始化项目（自动扫描训练脚本，生成配置）")
    ini.add_argument("path", nargs="?", default=".", help="项目路径（默认当前目录）")
    ini.add_argument("--agent", action="store_true", help="启用 AI 补全缺失项")

    # ---- check：环境就绪检查 ----
    sub.add_parser("check", help="检查环境就绪状态（依赖、GPU、项目配置）")

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


def _load_contract(args) -> TaskContract:
    """仅加载 TaskContract（不初始化日志），供 _resolve_model_fn 等辅助函数使用。"""
    cfg = load_config(args.config)
    contract_path = args.contract or cfg["contract"].get("path")
    return TaskContract(cfg["contract"], contract_path)


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
    from .mcp_server import GuardianMCPServer

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
    # CPU 模式检测
    _cpu_mode = False
    try:
        import torch
        _cpu_mode = not torch.cuda.is_available()
    except ImportError:
        _cpu_mode = True  # torch 未安装 → 必然是 CPU

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
    if _cpu_mode:
        print("", flush=True)
        print("  ⚠ CPU 模式：未检测到 NVIDIA GPU，训练将在 CPU 上运行。", flush=True)
        print("    训练曲线（loss / accuracy / lr）仍可正常显示。", flush=True)
        print("    GPU 监控面板将不可用（需要 nvidia-smi + CUDA）。", flush=True)
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
        from .mcp_server import GuardianMCPServer
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
            from .dashboard import DashboardServer
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
            from .mcp_server import GuardianMCPServer
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
            remote=False, remote_port=8765, remote_host="0.0.0.0",
            remote_auth=None, remote_keepalive=60,
            autonomy="supervised",
            attach_pid=None, screen=None, tmux=None, persist_agent=None,
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
    if not train_cmd and not args.attach_pid:
        print("用法: python run.py watch -- <训练命令>\n"
              "例如: python run.py watch -- python train.py --epochs 20\n"
              "      python run.py watch --attach-pid 12345", flush=True)
        return 2

    cfg, contract = _load(args)

    # ---- 进程模式互斥校验 ----
    _process_modes = sum(bool(x) for x in [args.attach_pid, args.screen, args.tmux])
    if _process_modes > 1:
        print("错误: --attach-pid / --screen / --tmux 只能选一个", flush=True)
        return 1

    # ---- 启动前守卫：自动探测项目结构 ----
    from .project_context import ProjectContext
    ctx = ProjectContext(args.project_dir or ".")
    has_project_file = ctx.detected_by != "none"

    if not has_project_file:
        print("=" * 56, flush=True)
        print("  未检测到训练项目结构。", flush=True)
        print("", flush=True)
        print("  请先初始化（推荐）：", flush=True)
        print("     guarftrain init", flush=True)
        print("", flush=True)
        print("  或显式指定配置：", flush=True)
        print("     guarftrain watch --config <config.yaml> -- <训练命令>", flush=True)
        print("", flush=True)
        print("  或在训练项目目录中运行：", flush=True)
        print("     cd <训练项目> && guarftrain watch -- python train.py", flush=True)
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
        from .mcp_server import GuardianMCPServer
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
        from .agent_advisor import AgentAdvisor
        # 强制启用配置节，让 AgentAdvisor 的 _has_credentials() 做凭据检测
        cfg["agent"]["enabled"] = True
        # 持久化决策日志到 logs/ 目录
        log_dir = cfg.get("project", {}).get("log_dir", "./logs")
        cfg["agent"]["decision_log_path"] = str(Path(log_dir) / "decisions.jsonl")
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

    from .watchdog import TrainingWatchdog
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

    # ---- 进程模式选择 + ProcessAdapter ----
    from .process_adapter import (
        AttachedProcessAdapter,
        ScreenProcessAdapter,
        TmuxProcessAdapter,
    )
    import time as _time_mod

    log_dir = Path(project["log_dir"])
    process_mode = "spawn"
    process_adapter = None
    if args.attach_pid:
        process_mode = "attach"
        process_adapter = AttachedProcessAdapter(
            args.attach_pid, log_file=log_dir / "train.log",
        )
        print(f"[进程] 附加模式: PID={args.attach_pid}", flush=True)
    elif args.screen:
        process_mode = "screen"
        process_adapter = ScreenProcessAdapter(args.screen)
        # screen 启动在后面（需要 train_cmd）
    elif args.tmux:
        process_mode = "tmux"
        process_adapter = TmuxProcessAdapter(args.tmux)
        # tmux 启动在后面（需要 train_cmd）

    # ---- SubAgent 实例化 + 持久化 ----
    persist_path: Path | None = None
    if args.persist_agent:
        persist_path = Path(args.persist_agent)
    elif process_mode != "spawn" and advisor is not None:
        persist_path = log_dir / "sub_agent_state.json"

    sub_agent = None
    if advisor is not None:
        from .sub_agent import SubAgent, default_registry
        _sa_cfg = dict(cfg.get("sub_agent") or cfg.get("agent", {}))
        _sa_cfg["autonomy"] = args.autonomy

        if persist_path and persist_path.exists() and process_mode == "attach":
            # 断点续守：恢复 SubAgent 状态
            sub_agent = SubAgent.restore_from(persist_path, _sa_cfg, default_registry())
            print(f"[SubAgent] 从 {persist_path} 恢复"
                  f"（{len(sub_agent.memory)} 条记忆）", flush=True)
        else:
            sub_agent = SubAgent(config=_sa_cfg, tool_registry=default_registry())
            sub_agent._advisor = advisor  # LLM 回调复用
            if persist_path:
                sub_agent._persist_path = persist_path
            sub_agent.spawn({
                "command": " ".join(train_cmd),
                "total_epochs": contract.script.get("epochs", 0) if hasattr(contract, 'script') else 0,
                "model_entry": (contract.script.get("buildable_entry", {}).get("model_fn", "")
                                if hasattr(contract, 'script') else ""),
                "project_dir": str(Path(args.project_dir or ".").resolve()),
                "log_file": str(log_dir / "train.log"),
            })
            print(f"[SubAgent] 已启动（autonomy={args.autonomy}）", flush=True)

    # 将 SubAgent 引用注入 watchdog（崩溃记忆同步）
    watchdog.sub_agent = sub_agent

    # --remote：启动 SSE/REST 远程通信服务（PC 端 DSH 插件 / Dashboard 连接）。
    # 事件由 on_tick 在每轮 watchdog tick 中推送（metrics/anomaly/gpu/decision）。
    remote_server = None
    remote_state: dict[str, Any] = {
        "sid": project.get("name", "guardian-run"),
        "status": "running",
        "last_step": None,
        "decision_count": 0,
    }
    if args.remote:
        import time

        from .remote.server import RemoteServer
        from .remote.persistence import PersistenceManager

        log_dir = Path(project["log_dir"])
        persist = PersistenceManager(log_dir)
        decision_log = log_dir / "decisions.jsonl"

        def _read_decisions() -> list[dict]:
            if not decision_log.is_file():
                return []
            out: list[dict] = []
            try:
                for line in decision_log.read_text(encoding="utf-8").splitlines():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            except OSError:
                pass
            return out

        def _gpu_payload() -> dict:
            """最新 GPU 快照，字段按 DSH 插件 GpuTab 期望的形状映射。"""
            hist = monitor.get_gpu_history() if monitor is not None else []
            latest_by_id: dict[int, dict] = {}
            for rec in hist:
                latest_by_id[int(rec.get("gpu_id", rec.get("index", 0)))] = rec
            gpus = []
            for rec in latest_by_id.values():
                gpus.append({
                    "index": rec.get("gpu_id", 0),
                    "name": rec.get("name", ""),
                    "utilization": rec.get("utilization"),
                    "temperature": rec.get("temperature_c"),
                    "vram_used_mb": rec.get("memory_used_mb"),
                    "vram_total_mb": rec.get("memory_total_mb"),
                    "timestamp": rec.get("timestamp"),
                })
            return {"gpu_count": len(gpus), "gpus": gpus, "timestamp": time.time()}

        class _WatchRemoteHandler:
            """把 monitor / watchdog 的实时状态翻译成 RemoteServer 的查询接口。"""

            def get_training_status(self, session_id: str) -> dict:
                proc = watchdog.proc
                pid = None
                if proc is not None:
                    pid = proc.get_pid() if hasattr(proc, 'get_pid') else getattr(proc, 'pid', None)
                return {
                    "session_id": session_id,
                    "status": remote_state["status"],
                    "pid": pid,
                    "current_step": monitor.current_step() if monitor else None,
                }

            def get_metrics_history(self, session_id: str, limit: int = 200, offset: int = 0) -> dict:
                hist = monitor.get_metrics_history() if monitor else []
                page = hist[offset:offset + limit]
                return {"total": len(hist), "returned": len(page), "metrics": page}

            def get_gpu_status(self) -> dict:
                return _gpu_payload()

            def approve_action(self, session_id: str, action_id: str) -> dict:
                if sub_agent and sub_agent.is_spawned:
                    result = sub_agent.approve(action_id)
                    return {"success": result.success, "tool": result.tool_name,
                            "error": result.error or None}
                return {"error": "SubAgent 未启用"}

            def reject_action(self, session_id: str, action_id: str, reason: str = "") -> dict:
                if sub_agent and sub_agent.is_spawned:
                    result = sub_agent.reject(action_id, reason)
                    return {"success": True, "rejected": result.rejected}
                return {"error": "SubAgent 未启用"}

            def get_decision_log(self, session_id: str, limit: int = 50) -> list[dict]:
                return _read_decisions()[-limit:]

            def get_anomaly_history(self, session_id: str, limit: int = 50) -> list[dict]:
                hist = monitor.get_anomaly_history() if monitor else []
                return hist[-limit:]

            def get_training_log(self, session_id: str, lines: int = 100, grep: str = "") -> list[str]:
                log_file = log_dir / "train.log"
                if not log_file.is_file():
                    return []
                try:
                    with open(log_file, encoding="utf-8", errors="replace") as f:
                        all_lines = f.readlines()
                    result = all_lines[-lines:] if not grep else [l for l in all_lines if grep in l][-lines:]
                    return result
                except OSError:
                    return []

            def get_device_info(self) -> dict:
                try:
                    import platform
                    import psutil

                    return {
                        "hostname": platform.node(),
                        "os": platform.system(),
                        "cpu_count": psutil.cpu_count(),
                        "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
                        "memory_used_pct": psutil.virtual_memory().percent,
                    }
                except Exception:
                    return {"hostname": "unknown"}

            def get_pending_actions(self, session_id: str) -> list[dict]:
                if sub_agent and sub_agent.is_spawned:
                    return sub_agent.get_pending_actions()
                return []

            def trigger_recovery(self, session_id: str, action: str, params: dict) -> dict:
                return {"error": "watch 模式下不提供恢复接口"}

        try:
            remote_server = RemoteServer(
                _WatchRemoteHandler(),
                port=args.remote_port,
                host=args.remote_host,
                auth_token=args.remote_auth,
                persist_dir=log_dir / "remote",
                agent_advisor=advisor,
            )
            remote_server.start()
            remote_server.register_session(remote_state["sid"], {
                "name": project.get("name", "guardian-run"),
                "command": " ".join(train_cmd),
                "log_file": str(log_dir / "train.log"),
                "model_entry": contract.script.get("buildable_entry", {}).get("model_fn", ""),
                "project_dir": str(Path(args.project_dir or ".").resolve()),
            })
            remote_server.push_event(remote_state["sid"], "training_start", {
                "command": " ".join(train_cmd),
            })
            print("=" * 56, flush=True)
            print("  Guardian Remote Server (watch --remote)", flush=True)
            print(f"  SSE/REST : http://{args.remote_host}:{args.remote_port}", flush=True)
            print(f"  session  : {remote_state['sid']}", flush=True)
            if args.remote_auth:
                print("  auth     : 已启用（X-Auth-Token / ?token=）", flush=True)
            print("=" * 56, flush=True)
        except Exception as exc:
            print(f"[remote] 远程服务启动失败，训练照常进行: {exc}", flush=True)
            remote_server = None

    # --with-mcp：在 watchdog 就绪后后台启动
    if args.with_mcp:
        from .mcp_server import GuardianMCPServer
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
                from .dashboard import DashboardServer
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
        # MCP server 需要 dash_url 以调用 Dashboard 配置工具
        if mcp_thread is not None:
            mcp_srv.dash_url = dash_url
        process_id = project.get("name", "guardian-run")
        # 从 contract 提取 Dashboard 初始配置
        dash_config = contract.script.get("dashboard") if hasattr(contract, "script") else None
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
                    "dash_config": dash_config,
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            _ur.urlopen(req, timeout=3)
        except Exception:
            pass  # 面板不可达，不影响训练

    def on_tick(_wd, _proc) -> None:
        if monitor is not None:
            tick_events = monitor.poll_metrics()
            # Remote server: push metrics / anomalies / gpu / decisions via SSE
            if remote_server is not None:
                hist = monitor.get_metrics_history()
                if hist:
                    last = hist[-1]
                    step = last.get("step")
                    if step is not None and step != remote_state["last_step"]:
                        remote_state["last_step"] = step
                        remote_server.push_event(remote_state["sid"], "metrics", last)
                for ev in tick_events:
                    remote_server.push_event(remote_state["sid"], "anomaly", ev.to_dict())
                gpu = _gpu_payload()
                if gpu.get("gpu_count"):
                    remote_server.push_event(remote_state["sid"], "gpu_status", gpu)
                decisions = _read_decisions()
                if len(decisions) > remote_state["decision_count"]:
                    for d in decisions[remote_state["decision_count"]:]:
                        remote_server.push_event(remote_state["sid"], "decision", d)
                    remote_state["decision_count"] = len(decisions)
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

        # ---- SubAgent: 自主异常检测 + LLM 决策 ----
        if sub_agent is not None and sub_agent.is_spawned:
            hist = monitor.get_metrics_history() if monitor else []
            sa_metrics = hist[-1] if hist else {}
            # GPU 数据：直接从 monitor 获取，不依赖 _gpu_payload
            sa_gpu = None
            if monitor is not None:
                gpu_hist = getattr(monitor, "get_gpu_history", lambda: [])()
                if gpu_hist:
                    sa_gpu = gpu_hist
            sa_actions = sub_agent.on_tick(sa_metrics, sa_gpu)
            import uuid as _uuid
            for item in sa_actions:
                action = item["action"]
                if not item["needs_approval"]:
                    # 立即执行
                    result_sa = sub_agent.tools.execute(action, sub_agent.autonomy)
                    sub_agent.tools.record_result(result_sa)
                else:
                    # 入 pending 队列等待审批
                    aid = f"sa_{_uuid.uuid4().hex[:12]}"
                    sub_agent._pending_actions.append({
                        "action_id": aid,
                        "action": action,
                        "status": "pending",
                        "created_at": _time_mod.time(),
                        "priority": item.get("priority", "normal"),
                    })
                    # 推送 SSE decision 事件
                    if remote_server:
                        remote_server.push_event(remote_state["sid"], "decision", {
                            "action_id": aid,
                            "tool": action.tool_name,
                            "params": action.params,
                            "reason": action.reason,
                            "source": "sub_agent",
                            "priority": item.get("priority", "normal"),
                        })
            # 处理已审批动作（来自 PC 端 approve）
            for approved_action in sub_agent.drain_approved():
                result_sa = sub_agent.tools.execute(approved_action, sub_agent.autonomy)
                sub_agent.tools.record_result(result_sa)

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

    # ---- screen/tmux 启动 ----
    if process_mode == "screen" and process_adapter is not None:
        try:
            process_adapter.start(train_cmd)
            print(f"[screen] 会话 '{args.screen}' 已启动", flush=True)
        except Exception as exc:
            print(f"[screen] 启动失败: {exc}", flush=True)
            return 1
    elif process_mode == "tmux" and process_adapter is not None:
        try:
            process_adapter.start(train_cmd)
            print(f"[tmux] 会话 '{args.tmux}' 已启动", flush=True)
        except Exception as exc:
            print(f"[tmux] 启动失败: {exc}", flush=True)
            return 1

    # ---- watchdog 运行模式分发 ----
    try:
        if process_mode == "spawn":
            result = watchdog.run(train_cmd, on_tick=on_tick)
        else:
            result = watchdog.run_attach(process_adapter, on_tick=on_tick)
    except KeyboardInterrupt:
        watchdog.stop()
        result = {"status": "stopped", "exit_code": None}
        print("\n[守护] 收到中断信号", flush=True)

    # ---- SubAgent shutdown ----
    if sub_agent is not None and sub_agent.is_spawned:
        final_metrics = monitor.get_metrics_history()[-1] if monitor else None
        sa_summary = sub_agent.shutdown(final_metrics)
        if persist_path:
            persist_path.unlink(missing_ok=True)  # 清理持久化文件
        if remote_server:
            remote_server.push_event(remote_state["sid"], "decision", {
                "type": "sub_agent_shutdown",
                "narrative": sa_summary.get("narrative"),
                "stats": sa_summary.get("stats"),
            })
        print(f"[SubAgent] 已关闭"
              f"（{sa_summary.get('stats', {}).get('total_decisions', 0)} 次决策）", flush=True)

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

    if remote_server is not None:
        remote_state["status"] = result.get("status", "stopped")
        remote_server.push_event(remote_state["sid"], "training_end", {
            "status": remote_state["status"],
            "exit_code": result.get("exit_code"),
        })
        keepalive = int(getattr(args, "remote_keepalive", 60) or 0)
        if keepalive > 0:
            print(f"[remote] 训练已结束，服务保持在线 {keepalive}s 供 PC 端补拉最终状态 ...", flush=True)
            import time as _time

            _time.sleep(keepalive)
        remote_server.stop()

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
    from .project_context import ProjectContext
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
    from .project_context import ProjectContext
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
    from .experiment_query import ExperimentQuery
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
    from .experiment_query import ExperimentQuery
    from .agent_advisor import AgentAdvisor

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
    from .experiment_query import ExperimentQuery
    from .agent_advisor import AgentAdvisor

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


def _resolve_model_fn(contract, project_dir=None):
    """从 contract.buildable_entry 解析 model_fn。

    优先使用 contract 声明的入口，回退到项目自动扫描。
    返回 (model_fn, error_msg)，失败时 model_fn 为 None。
    """
    import importlib
    import sys

    # 确保项目根目录在 sys.path 中
    if project_dir:
        proj_root = str(Path(project_dir).resolve())
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)

    # 1. contract 声明
    entry = (contract.script or {}).get("buildable_entry", {})
    model_ref = entry.get("model_fn", "")

    # 2. 回退：project context 扫描
    if not model_ref:
        from .project_context import ProjectContext
        ctx = ProjectContext(project_dir or ".")
        ctx.apply_paths()
        model_ref = ctx.model_entry or ""

    if not model_ref:
        return None, "未找到模型入口。请在 contract.yaml 中设置 buildable_entry.model_fn，或传 --model 参数。"

    try:
        mod_path, fn_name = model_ref.split(":", 1) if ":" in model_ref else ("train", model_ref)
        mod = importlib.import_module(mod_path)
        model_fn = getattr(mod, fn_name)
        return model_fn, None
    except Exception as exc:
        return None, f"无法 import {model_ref}: {exc}"


def cmd_visualize(args) -> int:
    """模型管线可视化。"""
    from .model_viz import ModelVisualizer
    from .agent_advisor import AgentAdvisor

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
            import sys as _sys
            _scripts_dir = str(Path(mod_path).parent) if "/" in mod_path or "\\" in mod_path else None
            if _scripts_dir and _scripts_dir not in _sys.path:
                _sys.path.insert(0, _scripts_dir)
            if "/" in mod_path or "\\" in mod_path:
                mod_path = mod_path.replace("\\", "/").replace("/", ".").removesuffix(".py")
            import importlib
            mod = importlib.import_module(mod_path)
            model_fn = getattr(mod, fn_name)
        except Exception as exc:
            print(f"错误: 无法 import {args.model}: {exc}", flush=True)
            return 1
    else:
        # 从 contract / project context 解析（不再硬编码 from train import build_model）
        model_fn, err = _resolve_model_fn(
            _load_contract(args),
            project_dir=getattr(args, "project_dir", None),
        )
        if model_fn is None:
            print(f"错误: {err}", flush=True)
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


def cmd_analyze_architecture(args) -> int:
    """模型架构分析：FLOPs / 参数量 / 瓶颈层 / D3 tree data。"""
    import importlib

    from .arch_analyzer import ArchAnalyzer

    if args.model:
        try:
            mod_path, fn_name = args.model.split(":", 1)
            if "/" in mod_path or "\\" in mod_path:
                _dir = str(Path(mod_path).parent)
                if _dir not in sys.path:
                    sys.path.insert(0, _dir)
                mod_path = mod_path.replace("\\", "/").replace("/", ".").removesuffix(".py")
            mod = importlib.import_module(mod_path)
            model_fn = getattr(mod, fn_name)
        except Exception as exc:
            print(f"错误: 无法 import {args.model}: {exc}", flush=True)
            return 1
    else:
        model_fn, err = _resolve_model_fn(
            _load_contract(args),
            project_dir=getattr(args, "project_dir", None),
        )
        if model_fn is None:
            print(f"错误: {err}", flush=True)
            return 1

    analyzer = ArchAnalyzer()
    result = analyzer.analyze(model_fn)
    if result.get("error"):
        print(f"错误: {result['error']}", flush=True)
        return 1

    print(f"模型: {result['model_name']}", flush=True)
    print(f"参数量: {result['total_params']:,}", flush=True)
    print(f"FLOPs: {result['total_flops_m']}M", flush=True)
    print(f"模块数: {result['module_count']}  层数: {result['layer_count']}  耗时: {result['elapsed_ms']}ms", flush=True)
    if result["bottlenecks"]:
        print(f"瓶颈层 ({result['bottleneck_count']}):", flush=True)
        for b in result["bottlenecks"][:5]:
            print(f"  - [{b['severity']}] {b['layer']}: "
                  f"参数 {b['params_pct']}% / FLOPs {b['flops_pct']}%", flush=True)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 已写入: {out}", flush=True)

    if args.html:
        html_path = ArchAnalyzer.render_html(result, args.html)
        print(f"D3 可视化 HTML: {html_path}", flush=True)

    return 0


def cmd_gallery(args) -> int:
    """图片筛选与展示。"""
    from .gallery import GalleryManager
    from .inference import InferenceRunner
    from .agent_advisor import AgentAdvisor

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
    from .inference import InferenceRunner
    from .checkpoint_analyzer import CheckpointAnalyzer
    from .agent_advisor import AgentAdvisor

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
        from .project_context import ProjectContext
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
        # 从 contract / project context 解析 model_fn 后自动检测
        model_fn, err = _resolve_model_fn(
            _load_contract(args),
            project_dir=getattr(args, "project_dir", None),
        )
        if model_fn is not None:
            try:
                task_type = ir.detect_task_type(model_fn)
            except Exception:
                task_type = "classification"
        else:
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
    from .dashboard import DashboardServer
    cfg = load_config(args.config)
    ds = DashboardServer(config=cfg, port=args.port, host=args.host)
    print(f"Dashboard: http://{args.host}:{args.port}", flush=True)
    ds.start(blocking=True)
    return 0


def cmd_remote(args) -> int:
    """启动远程通信服务（算力服务器端）。"""
    from guardian.remote.server import RemoteServer
    from guardian.remote.persistence import PersistenceManager
    from guardian.gpu_monitor import GpuMonitor
    from guardian.sub_agent import default_registry

    cfg = load_config(args.config)

    # 持久化目录
    persist_root = cfg.get("project", {}).get("log_dir", "./logs")
    persist = PersistenceManager(persist_root)

    # GPU 监控
    gpu_monitor = GpuMonitor(poll_interval=5.0, persist_dir=Path(persist_root) / "gpu")
    gpu_monitor.load_from_disk()
    gpu_monitor.start()

    # 工具注册表（无实际 handler，仅做状态查询）
    tools = default_registry()

    # RemoteHandler 实现
    class _Handler:
        def get_training_status(self, session_id: str) -> dict:
            meta = persist.read_meta(session_id)
            return meta or {"error": "not_found"}

        def get_metrics_history(self, session_id: str, limit: int = 200, offset: int = 0) -> dict:
            metrics = persist.read_metrics(session_id)
            total = len(metrics)
            page = metrics[offset:offset + limit]
            return {"total": total, "returned": len(page), "metrics": page}

        def get_gpu_status(self) -> dict:
            summaries = gpu_monitor.get_all_summaries(window_minutes=60)
            return {
                "gpu_count": len(summaries),
                "gpus": [s.to_dict() for s in summaries],
                "timestamp": time.time(),
            }

        def approve_action(self, session_id: str, action_id: str) -> dict:
            return {"error": "standalone remote mode does not support approve"}

        def reject_action(self, session_id: str, action_id: str, reason: str = "") -> dict:
            return {"error": "standalone remote mode does not support reject"}

        def get_decision_log(self, session_id: str, limit: int = 50) -> list[dict]:
            return persist.read_decisions(session_id)[-limit:]

        def get_anomaly_history(self, session_id: str, limit: int = 50) -> list[dict]:
            events = persist.read_events(session_id)
            anomalies = [e for e in events if e.get("type") == "anomaly"]
            return anomalies[-limit:]

        def get_training_log(self, session_id: str, lines: int = 100, grep: str = "") -> list[str]:
            import os
            meta = persist.read_meta(session_id)
            if not meta:
                return []
            log_file = meta.get("log_file", "")
            if not log_file or not os.path.exists(log_file):
                return []
            try:
                with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                    all_lines = f.readlines()
                result = all_lines[-lines:] if not grep else [l for l in all_lines if grep in l][-lines:]
                return result
            except Exception:
                return []

        def get_device_info(self) -> dict:
            try:
                import platform, psutil
                return {
                    "hostname": platform.node(),
                    "os": platform.system(),
                    "cpu_count": psutil.cpu_count(),
                    "memory_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
                    "memory_used_pct": psutil.virtual_memory().percent,
                }
            except Exception:
                return {"hostname": "unknown"}

        def get_pending_actions(self, session_id: str) -> list[dict]:
            return []

        def trigger_recovery(self, session_id: str, action: str, params: dict) -> dict:
            return {"error": "standalone remote mode does not support recovery"}

    handler = _Handler()

    # 可选 AgentAdvisor（架构图 AI 解读）
    advisor = None
    agent_cfg = cfg.get("agent", {})
    if agent_cfg.get("enabled", False):
        from .agent_advisor import AgentAdvisor
        advisor = AgentAdvisor(agent_cfg)
        if not advisor.is_enabled():
            advisor = None

    server = RemoteServer(handler, port=args.port, host=args.host, auth_token=args.auth,
                          persist_dir=Path(persist_root) / "remote",
                          agent_advisor=advisor)

    # 注册已有会话
    for session in persist.list_sessions():
        server.register_session(session.get("session_id", session.get("experiment_id", "")), session)

    print("=" * 56, flush=True)
    print("  Guardian Remote Server", flush=True)
    print("=" * 56, flush=True)
    print(f"  HTTP/SSE : http://{args.host}:{args.port}", flush=True)
    print(f"  SSE 端点 : http://{args.host}:{args.port}/sse", flush=True)
    print(f"  持久化   : {persist_root}", flush=True)
    gpu_count = gpu_monitor.gpu_count
    print(f"  GPU 设备 : {gpu_count} 个（监控已启动）", flush=True)
    if gpu_count == 0:
        print("", flush=True)
        print("  ⚠ CPU 模式：未检测到 NVIDIA GPU。", flush=True)
        print("    训练曲线仍可正常显示；GPU 监控面板不可用。", flush=True)
    print("", flush=True)
    print("  PC 端连接:", flush=True)
    print(f"    url = \"http://<server-ip>:{args.port}\"", flush=True)
    print("", flush=True)
    print("  Ctrl+C 停止", flush=True)
    print("=" * 56, flush=True)

    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[Remote] 收到中断信号", flush=True)
    finally:
        gpu_monitor.stop()
        server.stop()
    return 0


def cmd_check(args) -> int:
    """环境就绪检查：依赖、GPU、项目配置。有严重问题时返回非零退出码。"""
    from .project_context import ProjectContext
    from .config import load_config, ConfigError

    issues = 0  # 问题计数

    def _ok(msg: str) -> None:
        print(f"  ✓ {msg}", flush=True)

    def _warn(msg: str) -> None:
        nonlocal issues
        issues += 1
        print(f"  ⚠ {msg}", flush=True)

    def _fail(msg: str) -> None:
        nonlocal issues
        issues += 1
        print(f"  ✗ {msg}", flush=True)

    print(f"Training Guardian v{__version__}", flush=True)
    print("=" * 50, flush=True)

    # 1. Python 版本
    py_ver = sys.version.split()[0]
    if sys.version_info >= (3, 9):
        _ok(f"Python {py_ver}")
    else:
        _fail(f"Python {py_ver} — 需要 >= 3.9")

    # 2. 核心依赖
    core_deps = {
        "yaml": "pyyaml", "GPUtil": "GPUtil", "requests": "requests", "numpy": "numpy",
    }
    for mod, pkg in core_deps.items():
        try:
            __import__(mod)
            _ok(pkg)
        except ImportError:
            _fail(f"{pkg} 未安装 (pip install {pkg})")

    # 3. 可选依赖
    print("", flush=True)
    print("  可选依赖:", flush=True)
    opt_deps = {
        "anthropic": ("agent", "AI 决策层"),
        "openai": ("agent-openai", "OpenAI 备选"),
        "mcp": ("mcp", "MCP 外部接入"),
        "fastapi": ("dashboard", "Web 面板"),
        "torch": ("viz", "可视化/推理"),
    }
    for mod, (extra, desc) in opt_deps.items():
        try:
            __import__(mod)
            print(f"    {mod:<12}: ✓ ({desc})", flush=True)
        except ImportError:
            print(f"    {mod:<12}: — (pip install guarftrain[{extra}])", flush=True)

    # 4. GPU 可用性
    print("", flush=True)
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            for g in gpus:
                _ok(f"GPU: {g.name} ({g.memoryTotal}MB)")
        else:
            print("  GPU         : 未检测到（CPU 模式运行）", flush=True)
    except Exception:
        print("  GPU         : 检测失败", flush=True)

    # Also check nvidia-smi (runtime monitoring uses it, not GPUtil)
    import shutil
    if shutil.which("nvidia-smi"):
        _ok("nvidia-smi 可用")

    # 5. 项目上下文
    print("", flush=True)
    ctx = ProjectContext(getattr(args, "project_dir", None) or ".")
    print(f"  项目探测    : {ctx.detected_by}", flush=True)
    if ctx.detected_by != "none":
        print(f"    name      : {ctx.name}", flush=True)
        print(f"    ckpt_dir  : {ctx.ckpt_dir}", flush=True)
        print(f"    log_dir   : {ctx.log_dir}", flush=True)
        if ctx.model_entry:
            print(f"    model     : {ctx.model_entry}", flush=True)
    else:
        _warn("未探测到训练项目结构 — 执行 `guarftrain init` 初始化")

    # 6. 配置文件校验
    config_path = getattr(args, "config", "configs/guardian.yaml")
    print("", flush=True)
    if Path(config_path).exists():
        try:
            cfg = load_config(config_path)
            _ok(f"配置 {config_path} 语法有效")
            # 检查关键配置
            project = cfg.get("project", {})
            ckpt = project.get("ckpt_dir", "./checkpoints")
            if not Path(ckpt).exists():
                _warn(f"ckpt_dir '{ckpt}' 目录不存在（首次运行自动创建）")
        except ConfigError as exc:
            _fail(f"配置 {config_path} 错误: {exc}")
        except Exception as exc:
            _fail(f"配置 {config_path} 解析失败: {exc}")
    else:
        print(f"  配置        : {config_path} 不存在（使用内置默认值）", flush=True)

    # 7. contract.yaml 校验
    contract_path = Path("configs/contract.yaml")
    if contract_path.exists():
        try:
            contract_text = contract_path.read_text(encoding="utf-8")
            yaml.safe_load(contract_text)
            _ok(f"契约 {contract_path} 语法有效")
        except Exception as exc:
            _fail(f"契约 {contract_path} 解析失败: {exc}")
    else:
        _warn(f"契约 {contract_path} 不存在 — 运行 `guarftrain init` 生成")

    print("", flush=True)
    print("=" * 50, flush=True)
    if issues:
        print(f"  发现 {issues} 个问题（标记 ⚠/✗）", flush=True)
    else:
        print(f"  环境就绪 ✓", flush=True)
    print("=" * 50, flush=True)
    return 1 if issues > 0 else 0


def cmd_init(args) -> int:
    """`guarftrain init` — project init 的顶级别名。"""
    import types
    # 构造与 project init 相同的 args
    proj_args = types.SimpleNamespace(
        action="init",
        path=args.path,
        agent=args.agent,
        config="configs/guardian.yaml",
        contract=None,
        project_dir=None,
    )
    return cmd_project(proj_args)


def cmd_project(args) -> int:
    """项目上下文管理。"""
    from .project_context import ProjectContext
    from .agent_advisor import AgentAdvisor

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

        # 自动生成 contract.yaml
        contract_path = ctx.generate_contract()
        if contract_path:
            print(f"契约配置已生成: {contract_path}", flush=True)
            ctx.save()  # 重新保存以包含 contract_path 引用
        else:
            existing = ctx.start_dir / "configs" / "contract.yaml"
            if existing.exists():
                print(f"契约配置已存在: {existing}（跳过生成）", flush=True)

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
    from .credentials import load_credentials, apply_credentials
    start = getattr(args, "project_dir", None) or "."
    cred = load_credentials(start)
    if cred:
        apply_credentials(cred)
        return True
    return False


def _apply_project_paths(args):
    """加载项目上下文并补全 sys.path（使 model_entry 可导入）。"""
    from .project_context import ProjectContext
    start = getattr(args, "project_dir", None) or "."
    ctx = ProjectContext(start)
    if ctx.detected_by != "none":
        ctx.apply_paths()
        return ctx
    return None


def _make_advisor(args):
    """从 args 构建 advisor（复用 watch 的模式）。"""
    _load_creds(args)
    from .agent_advisor import AgentAdvisor
    cfg = load_config(args.config)
    cfg["agent"]["enabled"] = True
    log_dir = cfg.get("project", {}).get("log_dir", "./logs")
    cfg["agent"]["decision_log_path"] = str(Path(log_dir) / "decisions.jsonl")
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
        if args.command == "analyze_architecture":
            return cmd_analyze_architecture(args)
        if args.command == "gallery":
            return cmd_gallery(args)
        if args.command == "infer":
            return cmd_infer(args)
        if args.command == "project":
            return cmd_project(args)
        if args.command == "dashboard":
            return cmd_dashboard(args)
        if args.command == "remote":
            return cmd_remote(args)
        if args.command == "init":
            return cmd_init(args)
        if args.command == "check":
            return cmd_check(args)
    except (ConfigError, ContractError) as exc:
        print(f"错误: {exc}", flush=True)
        return 1
    parser.error(f"未知子命令 {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
