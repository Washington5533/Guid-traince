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
        ),
    )
    p.add_argument("--config", default="configs/guardian.yaml")
    p.add_argument("--contract", default=None, help="默认取 config 里的 contract.path")
    sub = p.add_subparsers(dest="command", required=True)

    w = sub.add_parser("watch", help="守护任意训练命令（默认主路径）")
    w.add_argument("--strict-contract", action="store_true", help="契约缺项即拒绝启动")
    w.add_argument("--no-monitor", action="store_true")
    w.add_argument("--max-retries", type=int, default=None)
    w.add_argument("--agent", action="store_true", help="启用 agent 决策层（需配置 API key）")
    w.add_argument("--with-mcp", action="store_true", help="watch 的同时后台启动 MCP server")

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
    sv.add_argument("--transport", default="stdio", choices=["stdio", "tcp"])

    return p


def _load(args) -> tuple[dict, TaskContract]:
    cfg = load_config(args.config)
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
    result = server.start(transport=args.transport)
    if result:
        print(result, flush=True)
    return 0


def cmd_watch(args, train_cmd: list[str]) -> int:
    if not train_cmd:
        print("用法: python run.py watch -- <训练命令>\n"
              "例如: python run.py watch -- python train.py --epochs 20", flush=True)
        return 2

    cfg, contract = _load(args)
    project = cfg["project"]
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

    # --with-mcp：在 watchdog 就绪后后台启动
    if args.with_mcp:
        from guardian.mcp_server import GuardianMCPServer
        _avail, _err = GuardianMCPServer.is_available()
        if _avail:
            mcp_srv = GuardianMCPServer(
                cfg, monitor=monitor, ckpt_analyzer=analyzer,
                watchdog=watchdog, summary_gen=None, advisor=advisor,
                task_contract=contract,
                mode="shared",
            )
            mcp_thread = mcp_srv.start_in_background(transport="stdio")
            if mcp_thread is not None:
                print("[MCP] 已在后台线程启动，外部 agent 客户端可接入。", flush=True)
        # 不可用时已在上面打印过提示，此处不再重复
    summary_gen = SummaryGenerator(project, monitor, analyzer, watchdog, advisor=advisor)

    def on_tick(_wd, _proc) -> None:
        if monitor is not None:
            monitor.poll_metrics()
        analyzer.poll()

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

    summary = summary_gen.generate(result)
    print(flush=True)
    summary_gen.print_summary(summary)
    try:
        jpath, _ = summary_gen.save_summary(summary, project["log_dir"])
        print(f"摘要已保存: {jpath}", flush=True)
    except OSError as exc:
        print(f"摘要保存失败: {exc}", flush=True)

    return 0 if result.get("status") == "completed" else 1


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
    except (ConfigError, ContractError) as exc:
        print(f"错误: {exc}", flush=True)
        return 1
    parser.error(f"未知子命令 {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
