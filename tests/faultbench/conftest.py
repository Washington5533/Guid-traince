"""cp_12 · faultbench 公共装置：临时目录、契约、watchdog 工厂。

每个场景跑在独立临时目录（自己的 checkpoints/ 与 logs/），互不干扰。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

FAKE_TRAIN = Path(__file__).resolve().parent / "fake_train.py"

# 完整契约：四项齐备 + cli_mappings，对应 fake_train.py 实际支持的参数
FULL_CONTRACT = {
    "script_contract": {
        "resumable": {"entry": "cli", "resume_flag": "--resume", "ckpt_flag": "--ckpt"},
        "checkpoint_schema": {
            "required_keys": ["epoch", "model_state_dict", "optimizer_state_dict"],
            "recommended_keys": ["rng_state"],
        },
        "metrics_channel": {
            "type": "log_file",
            "path": "./logs/train.log",
            "log_pattern": r"epoch (\d+) loss ([\d.naN]+)",
        },
        "buildable_entry": {
            "model_fn": "fake_train:build_model",
            "dataloader_fn": "fake_train:get_dataloaders",
        },
        "cli_mappings": {
            "optimizer.lr": "--lr",
            "dataloader.batch_size": "--batch_size",
            "dataloader.grad_accum_steps": "--grad_accum_steps",
            "dataloader.num_workers": "--num_workers",
        },
        "launcher": "python",
    }
}


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "logs").mkdir()
    return tmp_path


@pytest.fixture
def contract_file(workdir: Path):
    """写出 contract.yaml，返回一个可定制的工厂。"""
    import copy
    import yaml

    def _make(**overrides) -> Path:
        data = copy.deepcopy(FULL_CONTRACT)
        sc = data["script_contract"]
        for key, val in overrides.items():
            if val is None:
                sc.pop(key, None)
            else:
                sc[key] = val
        path = workdir / "contract.yaml"
        path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
        return path

    return _make


def train_cmd(workdir: Path, *extra: str) -> list[str]:
    """构造 fake_train 命令，指向本场景的临时目录。"""
    return [
        sys.executable, str(FAKE_TRAIN),
        "--ckpt_dir", str(workdir / "checkpoints"),
        "--log_file", str(workdir / "logs" / "train.log"),
        *extra,
    ]


def make_watchdog(workdir: Path, contract_path: Path, **cfg):
    from guardian.notifier import Notifier
    from guardian.task_contract import TaskContract
    from guardian.watchdog import TrainingWatchdog

    base = {"max_retries": 6, "restart_delay": 0, "sigterm_grace": 5,
            "min_batch_size": 8, "oom_batch_reduce_ratio": 0.5}
    base.update(cfg)
    contract = TaskContract({}, contract_path)
    return TrainingWatchdog(
        base, Notifier({"channels": []}), contract=contract,
        ckpt_dir=workdir / "checkpoints",
    )
