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
from pathlib import Path
from typing import Any

import yaml

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
        """扫描目录，自动探测项目结构。空目录也算（用户可能刚建好）。"""
        ckpt_dir = None
        log_dir = None
        data_dir = None

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

        if ckpt_dir or log_dir or data_dir:
            return {
                "project": {
                    "name": root.name,
                    "ckpt_dir": ckpt_dir or str(root / "checkpoints"),
                    "log_dir": log_dir or str(root / "logs"),
                    "data_dir": data_dir or str(root / "data"),
                },
                "model": {},
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
            pass

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
