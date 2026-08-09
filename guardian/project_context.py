"""项目上下文自动探测 (ProjectContext)。

解决跨项目使用时路径硬编码问题：
1. 扫描当前目录 → 父目录 → 自动发现 .guardian-project.yaml
2. 未找到则自动探测：checkpoints/、logs/、data/ 等标准目录
3. AI 可用时让其补全缺失项并写入配置文件
4. 配置一经写入，后续所有命令自动继承

优先级: 配置文件 > 自动探测 > AI 推断 > 硬编码默认值
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

from guardian.logging_config import get_logger

logger = get_logger(__name__)

# 项目配置模板
PROJECT_TEMPLATE = {
    "project": {
        "name": "",
        "ckpt_dir": "",
        "log_dir": "",
        "data_dir": "",
    },
    "model": {
        "entry": "",
        "task_type": "classification",
    },
    "paths": {
        "extra_sys_paths": [],     # 需要额外加入 sys.path 的目录（如 CLIP 源码）
        "contract_path": "",        # contract.yaml 位置
    },
    "_detected_by": "",  # "user" | "scanner" | "agent" | "none"
}

# 要扫描的目录相对路径
_SCAN_PATTERNS = [
    ("checkpoints", ["checkpoints/cp_*", "checkpoints/*/model.pth"]),
    ("logs", ["logs/summary_*.json", "logs/train.log"]),
    ("data", ["data/**/*.jpg", "data/**/*.png", "data/*/images"]),
]


class ProjectContext:
    """项目上下文：自动探测 + AI 补全 + 配置文件管理。"""

    def __init__(self, start_dir: str | Path | None = None, advisor: Any = None):
        self.start_dir = Path(start_dir) if start_dir else Path.cwd()
        self.advisor = advisor
        self.config_path: Path | None = None
        self.data: dict[str, Any] = {}
        self._load_or_detect()

    # ------------------------------------------------------------------
    # 加载 / 探测
    # ------------------------------------------------------------------

    def _load_or_detect(self) -> None:
        """按优先级加载配置。

        搜索顺序：
        1. start_dir 向上 4 层找 .guardian-project.yaml
        2. CWD 向上 4 层找 .guardian-project.yaml（解决跨目录调用）
        3. start_dir 自动扫描
        4. CWD 自动扫描
        5. 回退默认
        """
        # 1. start_dir → 父目录
        for root in (self.start_dir, Path.cwd()):
            for parent in [root] + list(root.parents)[:4]:
                candidate = parent / ".guardian-project.yaml"
                if candidate.exists():
                    self.config_path = candidate
                    self.start_dir = parent  # 重新锚定到配置所在目录
                    self.data = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
                    self.data.setdefault("_detected_by", "user")
                    return

        # 2. 自动扫描
        for root in (self.start_dir, Path.cwd()):
            detected = self._scan(root)
            if detected:
                self.data = detected
                return

        # 3. 回退
        self.data = dict(PROJECT_TEMPLATE)
        self.data["project"]["name"] = self.start_dir.name
        self.data["_detected_by"] = "none"

    @staticmethod
    def _scan(root: Path) -> dict | None:
        """扫描目录，自动探测项目结构。包括模型入口和额外路径。"""
        ckpt_dir = None
        log_dir = None
        data_dir = None
        extra_paths = []
        model_entries = []
        contract_path = None

        for pattern in ["checkpoints", "checkpoint"]:
            p = root / pattern
            if p.is_dir():
                ckpt_dir = str(p)
                break

        for pattern in ["logs", "log"]:
            p = root / pattern
            if p.is_dir():
                log_dir = str(p)
                break

        for pattern in ["data", "dataset", "datasets"]:
            p = root / pattern
            if p.is_dir():
                data_dir = str(p)
                break

        # 扫描 contract.yaml
        for loc in ["configs/contract.yaml", "contract.yaml"]:
            p = root / loc
            if p.exists():
                contract_path = str(p)
                break

        # 扫描 Python 文件，找 build_model 和 sys.path.insert
        for py_file in sorted(root.glob("*.py")):
            try:
                text = py_file.read_text(encoding="utf-8", errors="replace")
                lines = text.splitlines()
            except Exception:
                logger.warning("读取 Python 文件失败，跳过: %s", py_file, exc_info=True)
                continue

            # 找 build_model / get_dataloaders 函数
            for i, line in enumerate(lines):
                if "def build_model" in line or "def get_model" in line:
                    fn_name = line.strip().split("(")[0].replace("def ", "")
                    model_entries.append(f"{py_file.stem}:{fn_name}")

            # 找 sys.path.insert 调用 → 推断额外路径
            for line in lines:
                if "sys.path.insert" in line or "sys.path.append" in line:
                    # 尝试提取路径字符串
                    import re
                    m = re.search(r"""["']([^"']+)["']""", line)
                    if m:
                        p_str = m.group(1)
                        # 解析相对路径
                        candidate = (py_file.parent / p_str).resolve()
                        if candidate.exists() and candidate.is_dir():
                            extra_paths.append(str(candidate))

        # 清理重复
        extra_paths = list(dict.fromkeys(extra_paths))
        model_entries = model_entries[:5]  # 最多 5 个候选

        if ckpt_dir or log_dir or data_dir or model_entries:
            return {
                "project": {
                    "name": root.name,
                    "ckpt_dir": ckpt_dir or str(root / "checkpoints"),
                    "log_dir": log_dir or str(root / "logs"),
                    "data_dir": data_dir or str(root / "data"),
                },
                "model": {
                    "entry": model_entries[0] if model_entries else "",
                    "entry_candidates": model_entries,
                    "task_type": "classification",
                },
                "paths": {
                    "extra_sys_paths": extra_paths,
                    "contract_path": contract_path or "",
                },
                "_detected_by": "scanner",
            }

        return None

    # ------------------------------------------------------------------
    # AI 补全
    # ------------------------------------------------------------------

    def fill_with_agent(self) -> bool:
        """用 AI 补全缺失项并保存。返回 True 表示有变更。"""
        if self.advisor is None or not self.advisor.is_enabled():
            return False

        missing = self._missing_fields()
        if not missing:
            return False

        try:
            ctx = {
                "current_config": self.data,
                "missing_fields": missing,
                "found_files": self._list_relevant_files(),
            }
            result = self.advisor.suggest("project_config", ctx)
            if result and isinstance(result, dict):
                changed = False
                for section in ("project", "model"):
                    if section in result:
                        for k, v in result[section].items():
                            if v and not self.data.get(section, {}).get(k):
                                self.data.setdefault(section, {})[k] = v
                                changed = True
                if changed:
                    self.data["_detected_by"] = "agent"
                    self.save()
                    return True
        except Exception:
            logger.warning("AI 补全项目配置失败，保持当前探测结果", exc_info=True)

        return False

    def _missing_fields(self) -> list[str]:
        """返回缺失的关键字段。"""
        missing = []
        proj = self.data.get("project", {})
        if not proj.get("ckpt_dir"):
            missing.append("project.ckpt_dir")
        if not proj.get("name") or proj.get("name") == Path.cwd().name:
            missing.append("project.name")
        if not self.data.get("model", {}).get("entry"):
            missing.append("model.entry")
        return missing

    def _list_relevant_files(self) -> list[str]:
        """列出项目中的相关文件供 AI 分析。"""
        files = []
        root = self.start_dir
        for pattern in ["*.py", "configs/*.yaml", "**/checkpoints/cp_*/metrics.json"]:
            for p in list(root.glob(pattern))[:10]:
                files.append(str(p.relative_to(root)))
        return files

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------

    def save(self, path: str | Path | None = None) -> Path:
        """保存配置到 .guardian-project.yaml。"""
        target = Path(path) if path else (self.config_path or self.start_dir / ".guardian-project.yaml")
        target.write_text(
            yaml.safe_dump(self.data, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )
        self.config_path = target
        return target

    # ------------------------------------------------------------------
    # Contract 自动生成
    # ------------------------------------------------------------------

    def generate_contract(self, overwrite: bool = False) -> Path | None:
        """扫描训练脚本，自动生成最小 contract.yaml。

        返回生成的文件路径，如果已存在且不覆盖则返回 None。
        """
        import re

        contract_dir = self.start_dir / "configs"
        contract_path = contract_dir / "contract.yaml"

        if contract_path.exists() and not overwrite:
            return None

        # 扫描训练脚本
        resume_flag = "--resume"
        ckpt_flag = "--ckpt"
        model_fn = ""
        dataloader_fn = ""
        log_pattern = r"epoch (\d+) loss ([\d.naN]+)"
        train_module = ""

        for py_file in sorted(self.start_dir.glob("*.py")):
            try:
                text = py_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            # 找 argparse 的 resume/ckpt 参数
            for match in re.finditer(r'add_argument\([\'"]([^"\']+)[\'"]\s*', text):
                flag = match.group(1)
                if "resume" in flag.lower():
                    resume_flag = flag
                if "ckpt" in flag.lower() or "checkpoint" in flag.lower():
                    ckpt_flag = flag

            # 找 build_model / get_dataloaders
            for func_match in re.finditer(r'def (build_model|get_model|get_dataloaders|get_loaders)\b', text):
                fn_name = func_match.group(1)
                module = py_file.stem
                if "model" in fn_name:
                    model_fn = f"{module}:{fn_name}"
                elif "loader" in fn_name or "data" in fn_name:
                    dataloader_fn = f"{module}:{fn_name}"

            # 找结构化日志模式
            log_match = re.search(r'epoch.*?\{[^}]*\}.*?loss.*?\{[^}]*\}', text)
            if log_match:
                # 用户已有 f-string 格式的日志，推断默认模式
                pass

            if model_fn and dataloader_fn:
                train_module = py_file.stem
                break

        # 确定 entry
        entry = "cli"
        for py_file in sorted(self.start_dir.glob("*.py")):
            try:
                text = py_file.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            if "if __name__" in text and "argparse" in text:
                entry = "cli"
                break

        # 生成 YAML 内容
        contract_content = f"""# 自动生成 by guarftrain init
# 训练脚本接口契约，guardian 在训练进程之外只依赖这里声明的内容。
# 根据实际情况修改以下值。

script_contract:
  resumable:
    entry: {entry}
    resume_flag: "{resume_flag}"
    ckpt_flag: "{ckpt_flag}"

  checkpoint_schema:
    required_keys: [epoch, model_state_dict, optimizer_state_dict]

  metrics_channel:
    type: log_file
    path: ../logs/train.log
    log_pattern: "{log_pattern}"

  buildable_entry:
    model_fn: "{model_fn}"
    dataloader_fn: "{dataloader_fn}"

  cli_mappings:
    optimizer.lr: "--lr"
    dataloader.batch_size: "--batch_size"

  launcher: python
  batch_semantics: null

metric_registry:
  classification:
    - {{name: accuracy, direction: max}}
  detection:
    - {{name: mAP50, direction: max}}
  segmentation:
    - {{name: mIoU, direction: max}}
  _fallback: {{name: val_loss, direction: min}}

adjustable_paths:
  - path: "optimizer.lr"
    max_delta_ratio: 0.5
  - path: "dataloader.batch_size"
    min_value: 8
    max_delta_ratio: 0.5
"""

        # 写入文件
        contract_dir.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(contract_content, encoding="utf-8")

        # 更新 project context 的 contract_path 引用
        self.data.setdefault("paths", {})["contract_path"] = str(
            contract_path.relative_to(self.start_dir) if contract_path.is_relative_to(self.start_dir) else contract_path
        )

        return contract_path

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def ckpt_dir(self) -> str:
        return self.data.get("project", {}).get("ckpt_dir", "./checkpoints")

    @property
    def log_dir(self) -> str:
        return self.data.get("project", {}).get("log_dir", "./logs")

    @property
    def data_dir(self) -> str:
        return self.data.get("project", {}).get("data_dir", "./data")

    @property
    def name(self) -> str:
        return self.data.get("project", {}).get("name", "unnamed")

    @property
    def model_entry(self) -> str | None:
        return self.data.get("model", {}).get("entry")

    @property
    def task_type(self) -> str:
        return self.data.get("model", {}).get("task_type", "classification")

    @property
    def extra_paths(self) -> list[str]:
        return self.data.get("paths", {}).get("extra_sys_paths", [])

    @property
    def contract_path(self) -> str:
        return self.data.get("paths", {}).get("contract_path", "")

    def apply_paths(self) -> None:
        """将项目所需路径加入 sys.path（使 model_entry 可导入）。"""
        proj_root = str(self.start_dir.resolve())
        if proj_root not in sys.path:
            sys.path.insert(0, proj_root)
        for p in self.extra_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

    @property
    def detected_by(self) -> str:
        return self.data.get("_detected_by", "none")

    def status(self) -> str:
        """终端友好的状态输出。"""
        lines = [
            f"Project: {self.name}",
            f"Config:  {self.config_path or 'not saved'}",
            f"Source:  {self.detected_by}",
            f"  ckpt_dir: {self.ckpt_dir}",
            f"  log_dir:  {self.log_dir}",
            f"  data_dir: {self.data_dir}",
        ]
        if self.model_entry:
            lines.append(f"  model:    {self.model_entry}")
        candidates = self.data.get("model", {}).get("entry_candidates", [])
        if len(candidates) > 1:
            lines.append(f"  alt:      {', '.join(candidates[1:4])}")
        extra = self.extra_paths
        if extra:
            lines.append(f"  paths:    {len(extra)} extra sys.path entries")
        if self.contract_path:
            lines.append(f"  contract: {self.contract_path}")
        lines.append(f"  task:     {self.task_type}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI 辅助：解析所有路径相关参数
# ---------------------------------------------------------------------------

def resolve_paths(args, project: ProjectContext | None = None) -> ProjectContext:
    """从 CLI args 解析项目上下文。

    如果 args 中有显式路径则覆盖探测值。
    没有则从当前目录自动探测。
    """
    start_dir = Path.cwd()
    if project is None:
        project = ProjectContext(start_dir)

    # CLI 覆盖
    for attr, arg_name in [
        ("ckpt_dir", "ckpt_dir"),
        ("log_dir", "log_dir"),
        ("data_dir", "data"),
    ]:
        val = getattr(args, arg_name, None)
        if val and val != project.data.get("project", {}).get(arg_name):
            project.data.setdefault("project", {})[arg_name] = str(val)

    if getattr(args, "name", None):
        project.data.setdefault("project", {})["name"] = args.name

    return project
