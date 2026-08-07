"""cp_11 · 任务契约：sidecar 与训练脚本之间唯一的接口面。

guardian 在训练进程之外，看不见任何进程内变量，只能依赖这份契约声明的
外部可见接口。v0 只实现硬性契约四项的校验与降级，注册表/白名单/提议
流程属于 v1。详见 checkpoint/cp_11.md
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# 契约四项 -> 缺失时关闭的能力（对应 cp_11.md 的降级表）
CAPABILITY_IMPACT = {
    "resumable": "自动恢复 + 全部重启式干预（cp_2 只剩 alert_only）",
    "checkpoint_schema": "checkpoint 续训与断点分析",
    "metrics_channel": "loss 级异常检测与挂起检测（退化为进程级看护）",
    "buildable_entry": "run.py preflight / analyze 的独立评估",
}


@dataclass
class CheckResult:
    """单项契约的校验结果。"""

    name: str
    ok: bool
    detail: str
    impact: str = ""

    def __str__(self) -> str:
        mark = "OK  " if self.ok else "关闭"
        line = f"[{mark}] {self.name}: {self.detail}"
        if not self.ok and self.impact:
            line += f"\n         -> 已关闭：{self.impact}"
        return line


@dataclass
class ContractStatus:
    """四项契约的整体状态。"""

    results: list[CheckResult] = field(default_factory=list)

    @property
    def capabilities(self) -> dict[str, bool]:
        return {r.name: r.ok for r in self.results}

    @property
    def missing(self) -> list[str]:
        return [r.name for r in self.results if not r.ok]

    def is_ok(self, name: str) -> bool:
        return self.capabilities.get(name, False)

    def render(self) -> str:
        lines = ["训练脚本契约校验（cp_11）", "-" * 52]
        lines += [str(r) for r in self.results]
        if self.missing:
            lines.append("-" * 52)
            lines.append(f"缺失 {len(self.missing)} 项，对应能力已关闭，其余功能照常。")
        return "\n".join(lines)


class ContractError(Exception):
    """strict_mode 下契约缺失。"""


class TaskContract:
    """加载并校验 contract.yaml。

    v0 只做硬性契约四项 + cli_mappings 解析；select_metric /
    select_adjust_path / 提议流程属于 v1。
    """

    def __init__(
        self,
        contract_cfg: dict,
        contract_path: str | Path | None = None,
        base_dir: str | Path | None = None,
        advisor: Any = None,
        project_root: str | Path | None = None,
    ):
        """base_dir：契约里相对路径的基准目录。

        默认取 contract.yaml 所在目录——契约描述的是"这个项目的训练脚本"，
        路径相对契约文件本身最直观，也让同一份契约在任何 cwd 下都能工作。

        project_root：项目根目录，用于解析非契约级路径（如 proposal_log）。
        默认取 cwd。

        advisor：v1 可选注入的 AgentAdvisor，供 select_metric /
        select_adjust_path 使用；None 时（v0 默认）两者恒走规则/fallback 路径。
        """
        self.cfg = contract_cfg or {}
        self.path = Path(contract_path) if contract_path else None
        self.advisor = advisor
        if base_dir is not None:
            self.base_dir = Path(base_dir)
        elif self.path is not None:
            self.base_dir = self.path.parent
        else:
            self.base_dir = Path.cwd()
        self.raw: dict[str, Any] = {}
        self.script: dict[str, Any] = {}
        self._load()

        self.metric_registry: dict[str, Any] = self.raw.get("metric_registry") or {}
        self.adjustable_paths: list[dict[str, Any]] = list(self.raw.get("adjustable_paths") or [])

        self.project_root = Path(project_root) if project_root else Path.cwd()

        proposal_log = self.cfg.get("proposal_log", "logs/contract_proposals.json")
        # proposal_log 相对于项目根目录，而非契约文件目录
        p = Path(proposal_log)
        self.proposal_log_path = p if p.is_absolute() else (self.project_root / p).resolve()
        self.agent_can_propose = bool(self.cfg.get("agent_can_propose", True))

    def resolve_path(self, raw: str | Path | None) -> Path | None:
        """把契约里的路径解析为绝对路径（相对 base_dir）。"""
        if not raw:
            return None
        p = Path(raw)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def _load(self) -> None:
        if self.path is None or not self.path.exists():
            return
        if yaml is None:  # pragma: no cover
            raise ContractError("需要 pyyaml 才能解析 contract.yaml")
        loaded = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ContractError(f"{self.path} 顶层必须是映射")
        self.raw = loaded
        self.script = loaded.get("script_contract") or {}

    # --- 契约四项校验 -------------------------------------------------

    def _check_resumable(self, train_cmd: list[str] | None) -> CheckResult:
        block = self.script.get("resumable") or {}
        resume_flag = block.get("resume_flag")
        ckpt_flag = block.get("ckpt_flag")
        if not resume_flag or not ckpt_flag:
            return CheckResult(
                "resumable", False,
                "contract 未声明 resume_flag / ckpt_flag",
                CAPABILITY_IMPACT["resumable"],
            )
        # 有训练命令时，进一步用 --help 验证脚本真的认这两个参数
        if train_cmd:
            supported = self._probe_help(train_cmd)
            if supported is not None:
                missing = [f for f in (resume_flag, ckpt_flag) if f not in supported]
                if missing:
                    return CheckResult(
                        "resumable", False,
                        f"脚本 --help 未出现 {' / '.join(missing)}（声明了但脚本似乎不支持）",
                        CAPABILITY_IMPACT["resumable"],
                    )
        return CheckResult("resumable", True, f"{resume_flag} / {ckpt_flag}")

    def _check_checkpoint_schema(self, ckpt_dir: str | Path) -> CheckResult:
        block = self.script.get("checkpoint_schema") or {}
        required = block.get("required_keys") or []
        if not required:
            return CheckResult(
                "checkpoint_schema", False,
                "contract 未声明 required_keys",
                CAPABILITY_IMPACT["checkpoint_schema"],
            )
        d = Path(ckpt_dir)
        if not d.exists() or not any(d.glob("cp_*")):
            # 训练还没产出 checkpoint，此时只能确认声明存在
            return CheckResult(
                "checkpoint_schema", True,
                f"已声明 {required}（尚无 checkpoint 可实测，将在首个 cp 出现时校验）",
            )
        return CheckResult("checkpoint_schema", True, f"已声明 {required}")

    def _check_metrics_channel(self) -> CheckResult:
        block = self.script.get("metrics_channel") or {}
        ch_type = block.get("type")
        ch_path = block.get("path")
        if not ch_type or not ch_path:
            return CheckResult(
                "metrics_channel", False,
                "contract 未声明 type / path",
                CAPABILITY_IMPACT["metrics_channel"],
            )
        if ch_type == "log_file" and not block.get("log_pattern"):
            return CheckResult(
                "metrics_channel", False,
                "type=log_file 但未声明 log_pattern（无法解析指标）",
                CAPABILITY_IMPACT["metrics_channel"],
            )
        return CheckResult("metrics_channel", True, f"{ch_type} @ {ch_path}")

    def _check_buildable_entry(self) -> CheckResult:
        block = self.script.get("buildable_entry") or {}
        model_fn = block.get("model_fn")
        loader_fn = block.get("dataloader_fn")
        if not model_fn or not loader_fn:
            return CheckResult(
                "buildable_entry", False,
                "contract 未声明 model_fn / dataloader_fn",
                CAPABILITY_IMPACT["buildable_entry"],
            )
        return CheckResult("buildable_entry", True, f"{model_fn} / {loader_fn}")

    @staticmethod
    def _probe_help(train_cmd: list[str]) -> str | None:
        """跑一次 `<cmd> --help` 拿到支持的参数文本。失败返回 None（不据此判负）。"""
        exe = train_cmd[0]
        if not (Path(exe).exists() or shutil.which(exe)):
            return None
        try:
            proc = subprocess.run(
                [*train_cmd, "--help"],
                capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return (proc.stdout or "") + (proc.stderr or "")

    def validate_script_contract(
        self,
        train_cmd: list[str] | None = None,
        ckpt_dir: str | Path = "./checkpoints",
    ) -> ContractStatus:
        """校验四项契约，逐项返回满足/降级状态。"""
        status = ContractStatus(results=[
            self._check_resumable(train_cmd),
            self._check_checkpoint_schema(ckpt_dir),
            self._check_metrics_channel(),
            self._check_buildable_entry(),
        ])
        if self.cfg.get("strict_mode") and status.missing:
            raise ContractError(
                "strict_mode 下契约必须完整，缺失：" + ", ".join(status.missing)
                + "\n" + status.render()
            )
        return status

    def get_capability_status(self, **kwargs) -> dict[str, bool]:
        return self.validate_script_contract(**kwargs).capabilities

    # --- 供 cp_3 重启改写用 -------------------------------------------

    def resume_flags(self) -> tuple[str | None, str | None]:
        block = self.script.get("resumable") or {}
        return block.get("resume_flag"), block.get("ckpt_flag")

    def resolve_cli_mapping(self, path: str) -> str | None:
        """可调路径 -> 训练脚本命令行参数；查不到返回 None（该路径不可调）。"""
        return (self.script.get("cli_mappings") or {}).get(path)

    def metrics_channel(self) -> dict[str, Any]:
        """指标通道声明，path 已解析为绝对路径（相对 base_dir）。"""
        block = dict(self.script.get("metrics_channel") or {})
        if block.get("path"):
            resolved = self.resolve_path(block["path"])
            if resolved is not None:
                block["path"] = str(resolved)
        return block

    def checkpoint_required_keys(self) -> list[str]:
        block = self.script.get("checkpoint_schema") or {}
        return list(block.get("required_keys") or [])

    def batch_adjustable(self) -> bool:
        """torchrun 等多进程启动且未声明 batch_semantics 时，v0 不调 batch。"""
        launcher = (self.script.get("launcher") or "python").lower()
        if launcher in ("python", "python3"):
            return True
        return self.script.get("batch_semantics") is not None

    # ------------------------------------------------------------------
    # v1: 指标选择（agent 在注册表内自适应）
    # ------------------------------------------------------------------

    def select_metric(self, task_context: dict | None = None) -> dict[str, Any]:
        """从注册表选出本次训练的'最优模型判定指标'。

        Returns: {"metric": name, "direction": "max"/"min", "source": "config_explicit"/"agent_inferred"/"fallback"}
        """
        ctx = task_context or {}
        registry = self.metric_registry
        fallback_name = "val_loss"
        fallback_dir = "min"
        if isinstance(registry, dict) and "_fallback" in registry:
            fb = registry["_fallback"]
            fallback_name = fb.get("name", fallback_name)
            fallback_dir = fb.get("direction", fallback_dir)

        # 1. 显式声明 task_type → 直接用
        task_type = self.script.get("task_type")
        if task_type and isinstance(registry, dict) and task_type in registry:
            entries = registry[task_type]
            if isinstance(entries, list) and entries:
                first = entries[0]
                return {"metric": first["name"], "direction": first.get("direction", "max"),
                        "source": "config_explicit", "task_type": task_type}

        # 2. agent 推断（在注册表条目中选择）
        if self.advisor is not None:
            registered = self._flatten_registry(registry)
            if registered:
                context = {
                    "metrics_seen": ctx.get("metrics_seen", []),
                    "checkpoint_keys": ctx.get("checkpoint_keys", []),
                    "log_snippets": ctx.get("log_snippets", []),
                }
                result = self.advisor.decide(
                    "select_metric", context, registered,
                    {"metric": fallback_name, "direction": fallback_dir},
                )
                action = result.get("action", {})
                if isinstance(action, dict) and "metric" in action:
                    return {"metric": action["metric"],
                            "direction": action.get("direction", fallback_dir),
                            "source": result.get("source", "agent_inferred"),
                            "task_type": action.get("task_type")}

        # 3. 规则推断：从指标键名推断任务类型
        inferred = self._infer_from_keys(ctx.get("metrics_seen", []))
        if inferred:
            return {**inferred, "source": "agent_inferred"}

        # 4. fallback
        return {"metric": fallback_name, "direction": fallback_dir, "source": "fallback"}

    @staticmethod
    def _flatten_registry(registry: dict) -> list:
        """把分组注册表展开为 agent 可选的条目列表。"""
        out = []
        if not isinstance(registry, dict):
            return out
        for task_type, entries in registry.items():
            if task_type.startswith("_"):
                continue
            if isinstance(entries, list):
                for e in entries:
                    if isinstance(e, dict):
                        out.append({"action": {**e, "task_type": task_type}})
        return out

    @staticmethod
    def _infer_from_keys(keys: list[str]) -> dict | None:
        """从指标键名推断任务类型和最优指标。"""
        joined = " ".join(keys).lower()
        # 检测任务专用指标
        if any(k in joined for k in ("map", "ap50", "ap75", "coco")):
            return {"metric": "mAP50", "direction": "max", "task_type": "detection"}
        if any(k in joined for k in ("miou", "dice", "iou")):
            return {"metric": "mIoU", "direction": "max", "task_type": "segmentation"}
        if any(k in joined for k in ("rmse", "mae", "mse")):
            return {"metric": "rmse", "direction": "min", "task_type": "regression"}
        if any(k in joined for k in ("accuracy", "acc", "f1")):
            return {"metric": "accuracy", "direction": "max", "task_type": "classification"}
        return None

    # ------------------------------------------------------------------
    # v1: 可调路径选择
    # ------------------------------------------------------------------

    def select_adjust_path(
        self, decision_point: str, context: dict | None = None,
    ) -> list:
        """从白名单选出当前决策点可用的调整路径及幅度范围。

        sidecar 下多一道过滤：路径必须能映射到命令行参数。
        返回可直接作为 cp_9 action_space 的列表。
        """
        ctx = context or {}
        paths = list(self.adjustable_paths or [])

        # sidecar 过滤：只保留有 cli_mappings 的路径
        cli = self.script.get("cli_mappings") or {}
        reachable = [p for p in paths if isinstance(p, dict)
                     and p.get("path", "") in cli]
        if not reachable:
            # 无可调路径 → action_space 只有无参动作
            return ["resume_unchanged", "alert_only"]

        # 构建候选动作列表
        candidates = []
        for p in reachable:
            name = p["path"]
            flag = cli[name]
            entry: dict[str, Any] = {"action": f"adjust:{name}"}
            if "max_delta_ratio" in p:
                entry["ratio"] = {"min": -p["max_delta_ratio"], "max": p["max_delta_ratio"]}
            if "min_value" in p:
                entry["min_value"] = p["min_value"]
            entry["_flag"] = flag
            entry["_path"] = name
            candidates.append(entry)

        # agent 选择
        if self.advisor is not None and len(candidates) > 1:
            result = self.advisor.decide(
                "select_adjust_path",
                {"decision_point": decision_point, **ctx},
                candidates + ["keep_default"],
                "keep_default",
            )
            if result.get("action") != "keep_default" and isinstance(result.get("action"), dict):
                chosen = result["action"]
                return [chosen]

        return candidates

    # ------------------------------------------------------------------
    # v1: 提议审核系统
    # ------------------------------------------------------------------

    def propose_registry_entry(
        self, kind: str, draft_entry: dict, evidence: str = "",
    ) -> dict | None:
        """agent 生成一条注册表扩展提议（不生效），写入 proposal_log。"""
        if not self.agent_can_propose:
            return None

        import uuid as _uuid
        proposal = {
            "id": _uuid.uuid4().hex[:12],
            "kind": kind,           # "metric" / "adjustable_path"
            "entry": draft_entry,
            "evidence": evidence,
            "status": "pending",
            "timestamp": time.time(),
        }
        proposals = self._read_proposals()
        proposals.append(proposal)
        self._write_proposals(proposals)
        return proposal

    def approve_proposal(self, proposal_id: str) -> dict:
        """批准一条提议，写入正式注册表。"""
        proposals = self._read_proposals()
        for p in proposals:
            if p.get("id") == proposal_id and p.get("status") == "pending":
                p["status"] = "approved"
                p["approved_at"] = time.time()
                self._write_proposals(proposals)
                self._apply_proposal(p)
                return {"status": "approved", "id": proposal_id}
        return {"status": "not_found", "id": proposal_id}

    def reject_proposal(self, proposal_id: str, reason: str = "") -> dict:
        """拒绝一条提议并归档。"""
        proposals = self._read_proposals()
        for p in proposals:
            if p.get("id") == proposal_id and p.get("status") == "pending":
                p["status"] = "rejected"
                p["rejected_at"] = time.time()
                p["reject_reason"] = reason
                self._write_proposals(proposals)
                return {"status": "rejected", "id": proposal_id}
        return {"status": "not_found", "id": proposal_id}

    def list_proposals(self, status: str | None = None) -> list[dict]:
        """列出提议，可按状态筛选。"""
        proposals = self._read_proposals()
        if status:
            return [p for p in proposals if p.get("status") == status]
        return list(proposals)

    def _read_proposals(self) -> list[dict]:
        try:
            if self.proposal_log_path.exists():
                data = json.loads(self.proposal_log_path.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
        except (ValueError, OSError):
            pass
        return []

    def _write_proposals(self, proposals: list) -> None:
        self.proposal_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.proposal_log_path.write_text(
            json.dumps(proposals, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _apply_proposal(self, proposal: dict) -> None:
        """批准后把提议条目写入正式注册表/白名单。"""
        kind = proposal.get("kind", "")
        entry = proposal.get("entry") or {}
        if kind == "metric" and isinstance(self.metric_registry, dict):
            task_type = entry.get("task_type", "inferred")
            if task_type not in self.metric_registry:
                self.metric_registry[task_type] = []
            self.metric_registry[task_type].append({
                "name": entry.get("name", ""),
                "direction": entry.get("direction", "max"),
            })
        elif kind == "adjustable_path":
            if not isinstance(self.adjustable_paths, list):
                self.adjustable_paths = []
            self.adjustable_paths.append({
                "path": entry.get("path", ""),
                "max_delta_ratio": entry.get("max_delta_ratio", 0.5),
            })
