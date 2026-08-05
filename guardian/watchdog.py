"""cp_3 · 进程守护与恢复：sidecar 形态的核心执行路径。

包装训练命令、以子进程方式拉起并全程看护；崩溃恢复与 cp_2 的主动干预
汇入同一条重启路径——因为在训练进程之外，重启是唯一可用的干预手段。
详见 checkpoint/cp_3.md
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RECOVERABLE = "recoverable"
UNRECOVERABLE = "unrecoverable"

# stderr 文本 -> 可恢复类型。顺序有意义：OOM 优先于泛化的 RuntimeError
RECOVERABLE_PATTERNS: list[tuple[str, str]] = [
    ("oom", r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED"),
    ("oom", r"DefaultCPUAllocator: can't allocate memory|Cannot allocate memory"),
    ("network", r"ConnectionError|ConnectionResetError|Timeout(Error)?\b|NCCL.*(timeout|unhandled)"),
]

# 明确不可恢复的报错（代码错误 / 数据问题），命中即停止重试
UNRECOVERABLE_PATTERNS: list[tuple[str, str]] = [
    ("data", r"FileNotFoundError|EOFError|UnpicklingError"),
    ("code", r"TypeError|AttributeError|NameError|ImportError|ModuleNotFoundError"
             r"|SyntaxError|KeyError|IndexError|ValueError|AssertionError"),
]

# 被信号杀死：-9/-15 (POSIX subprocess) 与 137/143 (128+N, shell 风格)
SIGNAL_EXIT_CODES = {-9: "sigkill", 137: "sigkill", -15: "sigterm", 143: "sigterm"}


@dataclass
class CrashInfo:
    """一次子进程异常退出的分类结果。"""

    verdict: str          # recoverable / unrecoverable
    kind: str             # oom / sigkill / network / code / data / unknown
    exit_code: int | None
    detail: str = ""

    @property
    def recoverable(self) -> bool:
        return self.verdict == RECOVERABLE


def classify_crash(exit_code: int | None, stderr_tail: str = "") -> CrashInfo:
    """纯规则判定"能不能恢复"——永远不过 LLM。

    sidecar 下只有退出码 + stderr 文本可用。**保守优先**：匹配不到任何
    已知可恢复模式时判为不可恢复，宁可停下等人看，也不对真 bug 反复重启。
    """
    text = stderr_tail or ""

    # OOM 判定优先于信号：cgroup OOM-killer 会以 137 退出，但 stderr 常有线索
    for kind, pattern in RECOVERABLE_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return CrashInfo(RECOVERABLE, kind, exit_code, f"stderr 命中 {kind} 模式")

    # 明确的代码/数据错误
    for kind, pattern in UNRECOVERABLE_PATTERNS:
        if re.search(pattern, text):
            return CrashInfo(UNRECOVERABLE, kind, exit_code, f"stderr 命中 {kind} 模式")

    # 被信号杀死且 stderr 无更多线索：视为外部中断，参数不变续训
    if exit_code in SIGNAL_EXIT_CODES:
        return CrashInfo(
            RECOVERABLE, "sigkill", exit_code,
            f"退出码 {exit_code}（{SIGNAL_EXIT_CODES[exit_code]}），按外部中断处理",
        )

    return CrashInfo(
        UNRECOVERABLE, "unknown", exit_code,
        "无法识别的失败原因，保守判定为不可恢复（不反复重启）",
    )


def _flag_value(cmd: list[str], flag: str) -> str | None:
    """从命令行里读某个 flag 当前的值，支持 `--k v` 与 `--k=v` 两种写法。"""
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            return cmd[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def _set_flag(cmd: list[str], flag: str, value: str) -> list[str]:
    """只替换、不追加：已存在的 flag 原地替换，避免同名参数出现两次。"""
    out = list(cmd)
    for i, tok in enumerate(out):
        if tok == flag and i + 1 < len(out):
            out[i + 1] = value
            return out
        if tok.startswith(flag + "="):
            out[i] = f"{flag}={value}"
            return out
    out.extend([flag, value])
    return out


def _has_flag(cmd: list[str], flag: str) -> bool:
    return any(tok == flag or tok.startswith(flag + "=") for tok in cmd)


@dataclass
class RestartRecord:
    """一次重启的完整轨迹，供摘要（cp_5）与 MCP（cp_10）查询。"""

    trigger: str                 # crash / intervention / hang
    reason: str
    resumed_from: str | None
    cmd_before: list[str]
    cmd_after: list[str]
    wasted_epochs: int | None = None
    applied: dict[str, Any] = field(default_factory=dict)
    skipped: str | None = None   # 动作被放弃的原因（取不到当前值、映射缺失等）
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trigger": self.trigger,
            "reason": self.reason,
            "resumed_from": self.resumed_from,
            "cmd_before": " ".join(self.cmd_before),
            "cmd_after": " ".join(self.cmd_after),
            "wasted_epochs": self.wasted_epochs,
            "applied": self.applied,
            "skipped": self.skipped,
            "timestamp": self.timestamp,
        }


class ActionNotApplicable(Exception):
    """动作在 sidecar 下无法落地（映射缺失 / 取不到当前值 / 语义不明）。"""


_CP_RE = re.compile(r"^cp_(\d+)$")


def find_latest_checkpoint(
    ckpt_dir: str | Path,
    required_keys: list[str] | None = None,
) -> tuple[Path | None, int | None]:
    """扫描 ckpt_dir，按 epoch 降序返回第一个有效 checkpoint。

    有效性：目录存在、含权重文件、（若能加载）含契约要求的必需键。
    损坏的自动跳过，选下一个。返回 (路径, epoch)。
    """
    d = Path(ckpt_dir)
    if not d.exists():
        return None, None

    candidates: list[tuple[int, Path]] = []
    for child in d.iterdir():
        if not child.is_dir():
            continue
        m = _CP_RE.match(child.name)
        if m:
            candidates.append((int(m.group(1)), child))
    candidates.sort(key=lambda pair: pair[0], reverse=True)

    for epoch, path in candidates:
        if _ckpt_is_valid(path, required_keys):
            return path, epoch
    return None, None


def _ckpt_is_valid(path: Path, required_keys: list[str] | None) -> bool:
    """非空 + 可加载 + 含必需键。torch 不可用时退化为只检查文件非空。"""
    weights = [p for p in path.glob("*.pth") if p.stat().st_size > 0]
    if not weights:
        return False
    if not required_keys:
        return True
    try:
        import torch
    except ImportError:  # pragma: no cover - 没装 torch 时不做深检
        return True
    for cand in weights:
        try:
            obj = torch.load(cand, map_location="cpu", weights_only=False)
        except Exception:
            continue  # 损坏，试下一个文件
        if isinstance(obj, dict) and all(k in obj for k in required_keys):
            return True
    return False


class TrainingWatchdog:
    """包装训练命令做进程外看护；崩溃恢复与主动干预共用同一条重启路径。"""

    def __init__(
        self,
        config: dict | None = None,
        notifier: Any = None,
        contract: Any = None,
        advisor: Any = None,
        ckpt_dir: str | Path = "./checkpoints",
        progress_fn: Any = None,
    ):
        """progress_fn: 无参可调用，返回训练当前已跑到的 epoch（或 None）。

        通常由 cp_2 的 monitor 提供。用于算重启作废了多少算力——
        没有它就只能记 None，无法回答"这次干预值不值"。
        """
        self.cfg = config or {}
        self.notifier = notifier
        self.contract = contract
        self.advisor = advisor          # v1；v0 恒为 None，全走规则默认策略
        self.ckpt_dir = Path(ckpt_dir)
        self.progress_fn = progress_fn

        self.max_retries = int(self.cfg.get("max_retries", 3))
        self.restart_delay = float(self.cfg.get("restart_delay", 10))
        self.oom_ratio = float(self.cfg.get("oom_batch_reduce_ratio", 0.5))
        self.min_batch = int(self.cfg.get("min_batch_size", 8))
        self.sigterm_grace = float(self.cfg.get("sigterm_grace", 30))

        # 挂起检测（cp_3 第三类故障：不退出的故障）
        raw_timeout = self.cfg.get("no_progress_timeout", 1800)
        self.no_progress_timeout = float(raw_timeout) if raw_timeout else None
        raw_kill = self.cfg.get("no_progress_kill_after")
        # None = 永不因挂起自动重启，只告警。guardian 不猜这个阈值
        self.no_progress_kill_after = float(raw_kill) if raw_kill else None

        self.retry_count = 0
        self.restarts: list[RestartRecord] = []
        self.proc: subprocess.Popen | None = None
        self._intervention: dict | None = None
        self._stop = False
        self._last_progress: Any = None
        self._progress_at: float = 0.0
        self._hang_warned = False

    # --- 契约能力 ---------------------------------------------------

    def can_restart(self) -> bool:
        """没有可续训入口就没有任何干预手段——只能告警。"""
        if self.contract is None:
            return False
        resume_flag, ckpt_flag = self.contract.resume_flags()
        return bool(resume_flag and ckpt_flag)

    def should_retry(self) -> bool:
        return self.retry_count < self.max_retries

    # --- 动作执行层 -------------------------------------------------

    def apply_action(self, cmd: list[str], action: str, param: Any = None) -> tuple[list[str], dict]:
        """把一个动作落到命令行上。返回 (新命令, 实际应用的调整)。

        不可落地时抛 ActionNotApplicable，由调用方回退到 resume_unchanged。
        规则见 cp_3.md「动作执行层」：只替换不追加、不认识的参数原样保留、
        取不到当前值就放弃、grad_accum 必须成对改写。
        """
        if action in ("resume_unchanged", "rollback_to_last_ckpt", "alert_only"):
            return list(cmd), {}

        if action in ("reduce_batch", "restart_with_lower_lr", "enable_grad_accum"):
            return self._apply_param_action(cmd, action, param)

        raise ActionNotApplicable(f"未知动作 {action!r}")

    def _resolve(self, path: str) -> str:
        """查可调路径到命令行参数的映射，查不到即不可落地。"""
        flag = self.contract.resolve_cli_mapping(path) if self.contract else None
        if not flag:
            raise ActionNotApplicable(
                f"契约 cli_mappings 未声明 {path!r} 的命令行参数，sidecar 下该路径不可调"
            )
        return flag

    def _current_number(self, cmd: list[str], flag: str, label: str) -> float:
        """读当前值。取不到就放弃动作，不猜默认值。"""
        raw = _flag_value(cmd, flag)
        if raw is None:
            raise ActionNotApplicable(
                f"命令行未显式指定 {flag}，无法计算{label}的新值（不猜默认值）"
            )
        try:
            return float(raw)
        except ValueError:
            raise ActionNotApplicable(f"{flag} 的当前值 {raw!r} 不是数字")

    def _apply_param_action(self, cmd: list[str], action: str, param: Any) -> tuple[list[str], dict]:
        if action == "restart_with_lower_lr":
            flag = self._resolve("optimizer.lr")
            cur = self._current_number(cmd, flag, "学习率")
            ratio = float(param if param is not None else 0.5)
            new = cur * ratio
            return _set_flag(cmd, flag, f"{new:g}"), {"optimizer.lr": {"from": cur, "to": new}}

        # 以下两个动作都要动 batch_size
        if self.contract is not None and not self.contract.batch_adjustable():
            raise ActionNotApplicable(
                "多进程启动（torchrun/accelerate）且契约未声明 batch_semantics，"
                "v0 不调整 batch_size，只做原样重启"
            )
        bs_flag = self._resolve("dataloader.batch_size")
        cur_bs = int(self._current_number(cmd, bs_flag, "batch size"))

        if action == "reduce_batch":
            ratio = float(param if param is not None else self.oom_ratio)
            new_bs = max(int(cur_bs * ratio), self.min_batch)
            if new_bs >= cur_bs:
                raise ActionNotApplicable(
                    f"batch_size 已降至下限 {self.min_batch}，无法继续减小"
                )
            return (
                _set_flag(cmd, bs_flag, str(new_bs)),
                {"dataloader.batch_size": {"from": cur_bs, "to": new_bs}},
            )

        # enable_grad_accum：必须成对改写，否则不是等价变换
        steps = int(param if param is not None else 2)
        if steps < 2:
            raise ActionNotApplicable("grad_accum steps 必须 >= 2")
        ga_flag = self._resolve("dataloader.grad_accum_steps")
        new_bs = max(cur_bs // steps, self.min_batch)
        if new_bs >= cur_bs:
            raise ActionNotApplicable(f"batch_size 已降至下限 {self.min_batch}")
        out = _set_flag(cmd, bs_flag, str(new_bs))
        out = _set_flag(out, ga_flag, str(steps))
        return out, {
            "dataloader.batch_size": {"from": cur_bs, "to": new_bs},
            "dataloader.grad_accum_steps": {"from": None, "to": steps},
        }

    def default_strategy(self, crash: CrashInfo) -> tuple[str, Any]:
        """规则默认恢复策略（advisor 不可用时的行为，也是 v0 的唯一行为）。"""
        if crash.kind == "oom":
            return "reduce_batch", self.oom_ratio
        if crash.kind == "network":
            return "resume_after_delay", self.restart_delay
        return "resume_unchanged", None

    def _decide_recovery(self, crash: CrashInfo) -> tuple[str, Any, str]:
        """可恢复中断确认后，决定'怎么恢复'。

        advisor 可用时问 agent，否则/超时走规则默认策略。
        返回 (action, param, source)。
        """
        default_action, default_param = self.default_strategy(crash)
        if self.advisor is None:
            return default_action, default_param, "rule_default"

        action_space = self._recovery_action_space(crash.kind)
        context = {
            "crash_kind": crash.kind,
            "detail": crash.detail,
            "exit_code": crash.exit_code,
            "retry_count": self.retry_count,
            "history": [r.to_dict() for r in self.restarts[-5:]],
        }
        result = self.advisor.decide(
            "watchdog_recovery", context, action_space, default_action,
        )
        source = result.get("source", "rule_default")
        chosen = result.get("action", default_action)
        param = None
        if isinstance(chosen, dict):
            param = {k: v for k, v in chosen.items() if k != "action"}
            chosen = chosen.get("action", default_action)
        # 提取数值参数
        if param and isinstance(param, dict) and "ratio" in param:
            param = param["ratio"]
        elif param and isinstance(param, dict) and "steps" in param:
            param = param["steps"]
        elif param is not None and not isinstance(param, (int, float, str, type(None))):
            param = None
        return chosen, param, source

    @staticmethod
    def _recovery_action_space(kind: str) -> list:
        """每种崩溃类型的可选恢复动作（cp_3.md 有限动作集）。"""
        if kind == "oom":
            return [
                {"action": "reduce_batch", "ratio": {"min": 0.1, "max": 0.9}},
                {"action": "enable_grad_accum", "steps": {"min": 2, "max": 8}},
                {"action": "reduce_batch_and_grad_accum",
                 "ratio": {"min": 0.1, "max": 0.5},
                 "steps": {"min": 2, "max": 4}},
                "resume_unchanged",
            ]
        if kind == "sigkill":
            return [
                "resume_unchanged",
                {"action": "resume_with_reduced_workers",
                 "ratio": {"min": 0.25, "max": 0.75}},
            ]
        if kind == "network":
            return [
                {"action": "resume_after_delay", "seconds": {"min": 5, "max": 300}},
                "resume_unchanged",
            ]
        return ["resume_unchanged"]

    # --- 重启路径 ---------------------------------------------------

    # --- 挂起检测（cp_3 第三类故障：不退出的故障） -------------------

    def _reset_progress(self) -> None:
        self._last_progress = None
        self._progress_at = time.monotonic()
        self._hang_warned = False

    def check_hang(self) -> str | None:
        """判据两条同时成立：指标不再前进 + 进程仍存活。

        默认只告警不动手——"慢"和"挂"从进程外看是一样的，一个 epoch 要 40
        分钟的任务配 30 分钟超时会反复误杀正常训练。只有用户显式配置了
        no_progress_kill_after 才会重启。

        返回 None / "warn" / "kill"。
        """
        if self.no_progress_timeout is None or self.progress_fn is None:
            return None            # 无指标通道时该能力自动关闭
        if self.proc is None or self.proc.poll() is not None:
            return None            # 进程已退出，走正常的崩溃分类路径

        try:
            current = self.progress_fn()
        except Exception:
            return None

        now = time.monotonic()
        if current != self._last_progress:
            self._last_progress = current
            self._progress_at = now
            self._hang_warned = False
            return None

        stalled = now - self._progress_at
        if self.no_progress_kill_after is not None and stalled >= self.no_progress_kill_after:
            return "kill"
        if stalled >= self.no_progress_timeout and not self._hang_warned:
            self._hang_warned = True
            self._notify(
                "疑似无进展",
                f"已 {int(stalled)}s 无新指标，进程仍存活。"
                + ("将按配置 kill 并续训" if self.no_progress_kill_after
                   else "未配置 no_progress_kill_after，仅告警不干预"),
                alert_type="no_progress",
            )
            return "warn"
        return None

    def _wasted_epochs(self, resume_epoch: int | None) -> int | None:
        """回滚作废了多少个 epoch = 训练已达进度 - 恢复起点。

        拿不到当前进度（无指标通道）时返回 None，不猜。
        """
        if self.progress_fn is None or resume_epoch is None:
            return None
        try:
            current = self.progress_fn()
        except Exception:
            return None
        if current is None:
            return None
        return max(0, int(current) - int(resume_epoch))

    def _build_resume_cmd(self, cmd: list[str], ckpt: Path | None) -> list[str]:
        """追加 resume/ckpt 参数。没有可用 checkpoint 时从头开始（不加 resume）。"""
        if ckpt is None or self.contract is None:
            return list(cmd)
        resume_flag, ckpt_flag = self.contract.resume_flags()
        out = list(cmd)
        if resume_flag and not _has_flag(out, resume_flag):
            out.append(resume_flag)
        if ckpt_flag:
            out = _set_flag(out, ckpt_flag, str(ckpt))
        return out

    def _notify(self, title: str, msg: str, alert_type: str, level: str = "warning",
                response: dict | None = None) -> None:
        if self.notifier is not None:
            self.notifier.send(title, msg, alert_type=alert_type, level=level, response=response)
        else:
            print(f"[{level}] {title}: {msg}", flush=True)

    def _terminate(self) -> None:
        """先 SIGTERM 给宽限期（避免半截 checkpoint），超时才强杀。"""
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=self.sigterm_grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def request_intervention(self, action: str, param: Any = None, reason: str = "") -> None:
        """cp_2 主动干预入口：请求以调整后的参数重启（下一个看护周期生效）。"""
        self._intervention = {"action": action, "param": param, "reason": reason}

    def stop(self) -> None:
        self._stop = True

    # --- 主入口 -----------------------------------------------------

    def run(self, train_cmd: list[str], on_tick=None) -> dict:
        """以子进程方式拉起训练命令并全程看护。

        on_tick(watchdog, proc) 每个轮询周期调用一次，供 cp_2 在训练进程外
        读指标、判异常、必要时调 request_intervention()。
        返回本次守护的结果摘要。
        """
        cmd = list(train_cmd)
        required = self.contract.checkpoint_required_keys() if self.contract else []
        poll = 0.5
        final = {"status": "unknown", "exit_code": None, "restarts": self.restarts}

        while True:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
            )
            stderr_chunks: list[str] = []

            # 看护循环：等子进程结束，期间给 cp_2 机会介入
            while True:
                try:
                    self.proc.wait(timeout=poll)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if on_tick is not None:
                    on_tick(self, self.proc)
                if self._stop:
                    self._terminate()
                    final.update(status="stopped", exit_code=self.proc.returncode)
                    return final
                if self._intervention is not None:
                    break

            interv = self._intervention
            self._intervention = None

            # 子进程已经退出时，先补一次 on_tick：崩溃可能发生在第一个轮询
            # 周期之内（进程存活时长 < poll），此时内层循环从未调用过 on_tick，
            # monitor 会错过退出前最后写入的指标行，导致 current_step 落后于
            # 真实进度、wasted_epochs 被算少（见 test_wasted_epochs_is_populated 回归）。
            if self.proc.poll() is not None and on_tick is not None:
                on_tick(self, self.proc)

            # --- 主动干预分支 ---
            if interv is not None and self.proc.poll() is None:
                self._terminate()
                new_cmd, applied, skipped = self._try_action(
                    cmd, interv["action"], interv.get("param")
                )
                ckpt, epoch = find_latest_checkpoint(self.ckpt_dir, required)
                new_cmd = self._build_resume_cmd(new_cmd, ckpt)
                wasted = self._wasted_epochs(epoch)
                rec = RestartRecord(
                    trigger="intervention",
                    reason=interv.get("reason") or interv["action"],
                    resumed_from=str(ckpt) if ckpt else None,
                    cmd_before=cmd, cmd_after=new_cmd,
                    wasted_epochs=wasted,
                    applied=applied, skipped=skipped,
                )
                self.restarts.append(rec)
                self._notify(
                    "主动干预重启", rec.reason, alert_type="intervention",
                    response={"source": "rule_default", "action": interv["action"],
                              "restart": True, "resumed_from": rec.resumed_from,
                              "wasted_epochs": wasted},
                )
                cmd = new_cmd
                time.sleep(min(self.restart_delay, 2))
                continue

            # --- 子进程已退出 ---
            out, err = self.proc.communicate()
            if err:
                stderr_chunks.append(err)
            stderr_tail = "".join(stderr_chunks)[-8000:]
            code = self.proc.returncode

            if code == 0:
                final.update(status="completed", exit_code=0)
                return final

            crash = classify_crash(code, stderr_tail)

            if not crash.recoverable:
                self._notify(
                    "训练不可恢复", f"{crash.kind}: {crash.detail}\n{stderr_tail[-1500:]}",
                    alert_type="unrecoverable", level="error",
                )
                final.update(status="failed", exit_code=code, crash=crash.kind,
                             detail=crash.detail, stderr_tail=stderr_tail[-2000:])
                return final

            if not self.can_restart():
                self._notify(
                    "可恢复中断但无法续训",
                    "契约未声明 resume_flag/ckpt_flag，自动恢复能力已关闭，不盲目重启",
                    alert_type="contract_missing", level="error",
                )
                final.update(status="failed", exit_code=code, crash=crash.kind,
                             detail="contract missing resumable")
                return final

            if not self.should_retry():
                self._notify(
                    "达到最大重试次数", f"已重试 {self.retry_count} 次，停止",
                    alert_type="max_retries", level="error",
                )
                final.update(status="failed", exit_code=code, crash=crash.kind,
                             detail="max_retries exceeded")
                return final

            action, param, decision_source = self._decide_recovery(crash)
            if action == "resume_after_delay":
                time.sleep(min(float(param or 0), 5))
                action, param = "resume_unchanged", None

            new_cmd, applied, skipped = self._try_action(cmd, action, param)
            ckpt, epoch = find_latest_checkpoint(self.ckpt_dir, required)
            new_cmd = self._build_resume_cmd(new_cmd, ckpt)

            self.retry_count += 1
            wasted = self._wasted_epochs(epoch)
            rec = RestartRecord(
                trigger="crash",
                reason=f"{crash.kind}: {crash.detail} -> {action}",
                resumed_from=str(ckpt) if ckpt else None,
                cmd_before=cmd, cmd_after=new_cmd,
                wasted_epochs=wasted,
                applied=applied, skipped=skipped,
            )
            self.restarts.append(rec)
            self._notify(
                f"训练中断，自动恢复（第 {self.retry_count} 次）", rec.reason,
                alert_type=f"crash_{crash.kind}",
                response={"source": "rule_default", "action": action, "restart": True,
                          "resumed_from": rec.resumed_from, "wasted_epochs": wasted},
            )
            cmd = new_cmd
            time.sleep(min(self.restart_delay, 2))

    def _try_action(self, cmd: list[str], action: str, param: Any):
        """执行动作，不可落地时回退为原样重启并记录原因。"""
        try:
            new_cmd, applied = self.apply_action(cmd, action, param)
            return new_cmd, applied, None
        except ActionNotApplicable as exc:
            return list(cmd), {}, str(exc)
