"""cp_14 · ExperimentQuery 单元测试。

覆盖 checkpoint/cp_14.md 的快速校验表。
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from guardian.experiment_query import (
    ExperimentQuery,
    _execute_query,
    _template_query,
    _match,
    _format_answer,
)


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------

def _make_summary(exp_id: str, **overrides) -> dict:
    """构建最小合法 summary JSON。"""
    return {
        "experiment_id": exp_id,
        "status": "completed",
        "duration_seconds": 120.0,
        "duration": "2min 0s",
        "exit_code": 0,
        "training": {
            "records": 100,
            "last_step": 100,
            "final_loss": 0.15,
            "min_loss": 0.12,
            "final_val_acc": 0.93,
            "best_val_acc": 0.95,
        },
        "anomaly_events": [],
        "restarts": [],
        "checkpoints": {
            "total": 5,
            "best": {
                "epoch": 20,
                "metrics": {"val/accuracy": 0.95},
            },
            "metric": "val/accuracy",
        },
        "resources": {
            "gpu_util_avg": 85.0,
            "gpu_mem_peak_gb": 8.5,
            "gpu_hours": 0.28,
        },
        "lr_schedule": [{"step": 0, "lr": 0.001}],
        **overrides,
    }


@pytest.fixture
def log_dir(tmp_path):
    """创建含多个 summary JSON 的临时目录。"""
    summaries = [
        _make_summary("exp_1", training={"final_loss": 0.15, "best_val_acc": 0.95},
                      resources={"gpu_util_avg": 85.0, "gpu_mem_peak_gb": 8.5},
                      duration_seconds=120.0, duration="2min 0s"),
        _make_summary("exp_2", training={"final_loss": 0.10, "best_val_acc": 0.97},
                      resources={"gpu_util_avg": 92.0, "gpu_mem_peak_gb": 10.2},
                      duration_seconds=180.0, duration="3min 0s"),
        _make_summary("exp_3", status="failed", exit_code=1,
                      training={"final_loss": 0.50, "best_val_acc": 0.72},
                      restarts=[{"trigger": "crash", "wasted_epochs": 3}]),
    ]
    for s in summaries:
        fname = tmp_path / f"summary_{s['experiment_id']}.json"
        fname.write_text(json.dumps(s, ensure_ascii=False), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# 快速校验
# ---------------------------------------------------------------------------

def test_list_experiments_returns_all(log_dir):
    eq = ExperimentQuery({"log_dir": str(log_dir)})
    exps = eq.list_experiments()
    assert len(exps) == 3


def test_list_experiments_empty_on_no_files(tmp_path):
    eq = ExperimentQuery({"log_dir": str(tmp_path)})
    exps = eq.list_experiments()
    assert exps == []


def test_get_experiment_by_id(log_dir):
    eq = ExperimentQuery({"log_dir": str(log_dir)})
    exp = eq.get_experiment("exp_1")
    assert exp is not None
    assert exp["status"] == "completed"


def test_get_experiment_missing(log_dir):
    eq = ExperimentQuery({"log_dir": str(log_dir)})
    assert eq.get_experiment("nonexistent") is None


# ---------------------------------------------------------------------------
# 查询 - 模板匹配（无 AI）
# ---------------------------------------------------------------------------

def test_query_recent(log_dir):
    eq = ExperimentQuery({"log_dir": str(log_dir)})
    result = eq.query("上次准确率最高的实验")
    assert result["source"] == "template"
    assert len(result["results"]) >= 1
    assert "answer" in result


def test_query_list_all(log_dir):
    eq = ExperimentQuery({"log_dir": str(log_dir)})
    result = eq.query("列出所有实验")
    assert result["source"] == "template"
    assert len(result["results"]) >= 1


def test_query_best(log_dir):
    eq = ExperimentQuery({"log_dir": str(log_dir)})
    result = eq.query("准确率最高的实验")
    assert result["source"] == "template"


def test_query_no_data(tmp_path):
    eq = ExperimentQuery({"log_dir": str(tmp_path)})
    result = eq.query("最好的实验")
    assert "暂无实验记录" in result["answer"]


# ---------------------------------------------------------------------------
# 对比
# ---------------------------------------------------------------------------

def test_compare(log_dir):
    eq = ExperimentQuery({"log_dir": str(log_dir)})
    result = eq.compare("exp_1", "exp_2")
    assert "error" not in result
    assert "diffs" in result
    assert len(result["diffs"]) > 0


def test_compare_missing(log_dir):
    eq = ExperimentQuery({"log_dir": str(log_dir)})
    result = eq.compare("exp_1", "nonexistent")
    assert "error" in result


# ---------------------------------------------------------------------------
# 参数推荐
# ---------------------------------------------------------------------------

def test_suggest_params(log_dir):
    eq = ExperimentQuery({"log_dir": str(log_dir)})
    rec = eq.suggest_params()
    assert "based_on" in rec


def test_suggest_params_no_data(tmp_path):
    eq = ExperimentQuery({"log_dir": str(tmp_path)})
    rec = eq.suggest_params()
    assert "error" in rec


# ---------------------------------------------------------------------------
# 查询执行引擎
# ---------------------------------------------------------------------------

def test_execute_query_no_filters():
    index = [{"name": "a", "value": 10}, {"name": "b", "value": 20}]
    results = _execute_query(index, {"sort_by": "value", "sort_order": "asc",
                                      "fields": ["name", "value"]})
    assert results[0]["value"] == 10


def test_execute_query_filter_eq():
    index = [{"name": "a", "value": 10}, {"name": "b", "value": 20}]
    results = _execute_query(index, {
        "filters": [{"field": "name", "op": "eq", "value": "a"}],
        "fields": ["name", "value"],
    })
    assert len(results) == 1
    assert results[0]["name"] == "a"


def test_execute_query_limit():
    index = [{"name": f"x{i}", "value": i} for i in range(20)]
    results = _execute_query(index, {"sort_by": "value", "limit": 5,
                                      "fields": ["name", "value"]})
    assert len(results) == 5


# ---------------------------------------------------------------------------
# 匹配器
# ---------------------------------------------------------------------------

def test_match_eq():
    assert _match("hello", "eq", "hello")
    assert not _match("hello", "eq", "world")


def test_match_gt():
    assert _match(10, "gt", 5)
    assert not _match(5, "gt", 10)


def test_match_contains():
    assert _match("hello world", "contains", "hello")
    assert not _match("hello", "contains", "world")


def test_match_none():
    assert not _match(None, "eq", "anything")


# ---------------------------------------------------------------------------
# 模板查询
# ---------------------------------------------------------------------------

def test_template_query_recent():
    q = _template_query("上次实验")
    assert q["sort_order"] == "desc"
    assert q["limit"] == 1


def test_template_query_all():
    q = _template_query("列出所有")
    assert q["limit"] >= 10


def test_template_query_best():
    q = _template_query("最高的准确率")
    assert q["sort_order"] == "desc"
    assert q["limit"] == 1


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------

def test_format_answer_single():
    ans = _format_answer("test", [{"experiment_id": "exp_1", "status": "completed",
                                     "final_loss": 0.15}], None, 10)
    assert "exp_1" in ans


def test_format_answer_empty():
    ans = _format_answer("test", [], None, 0)
    assert "未找到" in ans
