"""cp_4 · Checkpoint 发现、并发安全、top-k 与 keep_recent 保护。"""

from __future__ import annotations

import json
from pathlib import Path

from guardian.checkpoint_analyzer import CheckpointAnalyzer
from guardian.task_contract import TaskContract


def _write_cp(root: Path, epoch: int, acc: float, atomic: bool = True) -> Path:
    """模拟训练脚本写 checkpoint。atomic=True 走 tmp+rename。"""
    final = root / f"cp_{epoch}"
    target = root / f"cp_{epoch}.tmp" if atomic else final
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.pth").write_text(f"weights-{epoch}", encoding="utf-8")
    (target / "metrics.json").write_text(
        json.dumps({"epoch": epoch, "val/accuracy": acc}), encoding="utf-8"
    )
    if atomic:
        target.rename(final)
    return final


def _analyzer(root: Path, **cfg) -> CheckpointAnalyzer:
    base = {"stability_checks": 1, "save_top_k": 5, "keep_recent": 2}
    base.update(cfg)
    return CheckpointAnalyzer(base, ckpt_dir=root)


# ---------- 发现 ----------

def test_polling_discovers_new_checkpoints(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    an = _analyzer(root)
    assert an.poll() == []

    _write_cp(root, 0, 0.50)
    _write_cp(root, 1, 0.60)
    found = an.poll()
    assert [i.epoch for i in found] == [0, 1]
    assert found[1].metrics["val/accuracy"] == 0.60

    assert an.poll() == [], "已发现的不应重复上报"

    _write_cp(root, 2, 0.70)
    assert [i.epoch for i in an.poll()] == [2]


def test_ignores_non_checkpoint_dirs(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "logs").mkdir()
    (root / "cp_bad").mkdir()
    (root / "notes.txt").write_text("x", encoding="utf-8")
    _write_cp(root, 3, 0.8)
    assert [i.epoch for i in _analyzer(root).poll()] == [3]


def test_missing_dir_is_safe(tmp_path):
    assert _analyzer(tmp_path / "nope").poll() == []


# ---------- 并发安全（sidecar 特有） ----------

def test_half_written_checkpoint_skipped_then_picked_up(tmp_path):
    """撞上正在写的目录：本轮跳过，写完后下一轮正常发现。"""
    root = tmp_path / "checkpoints"
    root.mkdir()
    an = _analyzer(root, stability_checks=2)

    partial = root / "cp_0"
    partial.mkdir()
    (partial / "model.pth").write_text("partial", encoding="utf-8")

    assert an.poll() == [], "首次见到应等稳定性确认，不立即采信"

    (partial / "metrics.json").write_text(
        json.dumps({"epoch": 0, "val/accuracy": 0.5}), encoding="utf-8"
    )
    an.poll()                      # 指纹变化，重新计数
    found = an.poll()              # 这次稳定
    assert [i.epoch for i in found] == [0]
    assert found[0].metrics["val/accuracy"] == 0.5


def test_atomic_write_discovered_immediately(tmp_path):
    """原子写的目录首次轮询即可采信，不用多等。"""
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_cp(root, 0, 0.5, atomic=True)
    assert [i.epoch for i in _analyzer(root, stability_checks=1).poll()] == [0]


def test_empty_dir_not_reported(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    (root / "cp_0").mkdir()
    assert _analyzer(root).poll() == [], "空目录不应被当成有效 checkpoint"


def test_required_keys_enforced(tmp_path):
    """契约声明了必需键时，缺键的 checkpoint 不被采信。"""
    root = tmp_path / "checkpoints"
    root.mkdir()
    contract_path = tmp_path / "contract.yaml"
    import yaml
    contract_path.write_text(yaml.safe_dump({
        "script_contract": {"checkpoint_schema": {"required_keys": ["epoch", "model_state_dict"]}}
    }), encoding="utf-8")

    import torch
    bad = root / "cp_0"
    bad.mkdir()
    torch.save({"epoch": 0}, bad / "model.pth")          # 缺 model_state_dict
    good = root / "cp_1"
    good.mkdir()
    torch.save({"epoch": 1, "model_state_dict": {}}, good / "model.pth")

    an = CheckpointAnalyzer(
        {"stability_checks": 1}, ckpt_dir=root,
        contract=TaskContract({}, contract_path),
    )
    assert [i.epoch for i in an.poll()] == [1], "缺必需键的应被跳过"


# ---------- best / 清理 ----------

def test_best_by_metric(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    for ep, acc in [(0, 0.5), (1, 0.9), (2, 0.7)]:
        _write_cp(root, ep, acc)
    an = _analyzer(root)
    an.poll()
    assert an.best().epoch == 1


def test_cleanup_protects_recent(tmp_path):
    """最近 keep_recent 个必须留下，即便指标很差。

    6 个 cp，top_k=2 命中 cp_0/cp_1（指标最好），keep_recent=2 命中 cp_4/cp_5，
    中间的 cp_2/cp_3 既非 top-k 也非最近，应被删除。
    """
    root = tmp_path / "checkpoints"
    root.mkdir()
    for ep, acc in [(0, 0.99), (1, 0.98), (2, 0.10), (3, 0.11), (4, 0.12), (5, 0.13)]:
        _write_cp(root, ep, acc)
    an = _analyzer(root, save_top_k=2, keep_recent=2)
    an.poll()

    removed = an.cleanup()

    survivors = sorted(p.name for p in root.iterdir() if p.is_dir())
    assert "cp_4" in survivors and "cp_5" in survivors, \
        f"最近两个必须保留（训练可能要用），实际: {survivors}"
    assert "cp_0" in survivors and "cp_1" in survivors, "指标最好的应作为 top-k 保留"
    assert sorted(removed) == [2, 3], f"应只删中间的 cp_2/cp_3，实际删了 {removed}"


def test_cleanup_removes_nothing_when_all_protected(tmp_path):
    """top_k + keep_recent 覆盖了全部 checkpoint 时，一个都不该删。"""
    root = tmp_path / "checkpoints"
    root.mkdir()
    for ep, acc in [(0, 0.99), (1, 0.98), (2, 0.10), (3, 0.11)]:
        _write_cp(root, ep, acc)
    an = _analyzer(root, save_top_k=2, keep_recent=2)
    an.poll()
    assert an.cleanup() == []
    assert len([p for p in root.iterdir() if p.is_dir()]) == 4


def test_cleanup_keeps_top_k_when_no_metrics(tmp_path):
    """没有指标可比时按 epoch 新旧保留，不崩溃。"""
    root = tmp_path / "checkpoints"
    root.mkdir()
    for ep in range(5):
        d = root / f"cp_{ep}"
        d.mkdir()
        (d / "model.pth").write_text("w", encoding="utf-8")
    an = _analyzer(root, save_top_k=2, keep_recent=1)
    an.poll()
    an.cleanup()
    survivors = {p.name for p in root.iterdir() if p.is_dir()}
    assert "cp_4" in survivors and "cp_3" in survivors
    assert len(survivors) <= 3


def test_report_shape(tmp_path):
    root = tmp_path / "checkpoints"
    root.mkdir()
    _write_cp(root, 0, 0.5)
    _write_cp(root, 1, 0.8)
    an = _analyzer(root)
    an.poll()
    rep = an.report()
    assert rep["total"] == 2
    assert rep["latest"] == "cp_1"
    assert rep["best"]["epoch"] == 1
    assert len(rep["checkpoints"]) == 2
