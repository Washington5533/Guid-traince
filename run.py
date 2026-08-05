"""cp_8 · CLI 入口。

默认路径是 watch——包装任意训练命令做进程外守护，被守护的脚本不需要
import guardian。`--` 之后的内容原样透传给训练命令，guardian 不解析，
只在重启时按 contract.cli_mappings 追加/替换需要调整的参数。
详见 checkpoint/cp_8.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from guardian.checkpoint_analyzer import CheckpointAnalyzer
from guardian.config import ConfigError, load_config
from guardian.monitor import TrainingMonitor
from guardian.notifier import Notifier, ensure_utf8_stdout
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

    c = sub.add_parser("contract", help="契约校验")
    c.add_argument("action", choices=["check"])

    a = sub.add_parser("analyze", help="分析已有 checkpoint（独立扫描，不需要训练进程）")
    a.add_argument("--metric", default="val/accuracy")
    a.add_argument("--lower-better", action="store_true")

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
    cfg, contract = _load(args)
    status = contract.validate_script_contract(ckpt_dir=cfg["project"]["ckpt_dir"])
    print(status.render(), flush=True)
    return 0


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


def cmd_watch(args, train_cmd: list[str]) -> int:
    if not train_cmd:
        print("用法: python run.py watch -- <训练命令>\n"
              "例如: python run.py watch -- python train.py --epochs 20", flush=True)
        return 2

    cfg, contract = _load(args)
    project = cfg["project"]
    ckpt_dir = project["ckpt_dir"]

    # 启动前校验契约，逐项打印开启/降级状态
    try:
        status = contract.validate_script_contract(train_cmd=train_cmd, ckpt_dir=ckpt_dir)
    except ContractError as exc:
        print(str(exc), flush=True)
        return 1
    print(status.render(), flush=True)
    print(flush=True)

    notifier = Notifier(cfg["notifier"])
    monitor = None
    if not args.no_monitor and cfg["monitor"].get("enabled", True):
        monitor = TrainingMonitor(cfg["monitor"], notifier, contract=contract)
        if not monitor.enabled:
            print("[监控] 指标通道不可用，退化为进程级看护（存活 + 崩溃恢复）", flush=True)

    analyzer = CheckpointAnalyzer(cfg["checkpoint"], ckpt_dir=ckpt_dir, contract=contract)

    wd_cfg = dict(cfg["watchdog"])
    if args.max_retries is not None:
        wd_cfg["max_retries"] = args.max_retries

    from guardian.watchdog import TrainingWatchdog
    watchdog = TrainingWatchdog(
        wd_cfg, notifier, contract=contract, ckpt_dir=ckpt_dir,
        # 让 watchdog 能算出重启作废了多少 epoch（无指标通道时为 None，不猜）
        progress_fn=(monitor.current_step if monitor is not None else None),
    )
    summary_gen = SummaryGenerator(project, monitor, analyzer, watchdog)

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
    except (ConfigError, ContractError) as exc:
        print(f"错误: {exc}", flush=True)
        return 1
    parser.error(f"未知子命令 {args.command!r}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
