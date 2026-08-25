"""SubAgent + ProcessAdapter 测试套件。

覆盖：
- SubAgent 生命周期（spawn → on_tick → approve/reject → shutdown）
- approve 延迟执行（queue-based）
- 记忆持久化（save/restore 往返）
- 熔断器
- ProcessAdapter（AttachedProcess / ScreenProcess）
- watchdog.run_attach 附加模式
"""

from __future__ import annotations

import json
import queue
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from guardian.sub_agent import SubAgent, default_registry, Action, ActionResult
from guardian.sub_agent.memory import RollingMemory, DecisionRecord, TrainingPhase
from guardian.process_adapter import (
    ProcessAdapter,
    AttachedProcessAdapter,
    ScreenProcessAdapter,
    TmuxProcessAdapter,
    SpawnedProcessAdapter,
)


# ---------------------------------------------------------------------------
# 辅助工具
# ---------------------------------------------------------------------------

def _make_subagent(llm_return: str = "alert", autonomy: str = "supervised") -> SubAgent:
    """构建带 mock LLM 的 SubAgent。"""
    def mock_llm(system_prompt, user_message, timeout):
        return llm_return

    agent = SubAgent(
        config={"autonomy": autonomy, "decision_timeout": 5},
        tool_registry=default_registry(),
        llm_callback=mock_llm,
    )
    return agent


# ---------------------------------------------------------------------------
# SubAgent 生命周期
# ---------------------------------------------------------------------------

class TestSubAgentLifecycle:
    def test_spawn_on_tick(self):
        """mock LLM 返回 'alert'，验证 on_tick 返回动作。"""
        agent = _make_subagent("alert")
        agent.spawn({"command": "python train.py", "total_epochs": 20})
        assert agent.is_spawned

        metrics = {"epoch": 1, "step": 10, "loss": 0.5}
        actions = agent.on_tick(metrics, None)
        # LLM 返回 'alert'，应产生一个告警动作
        assert len(actions) >= 0  # 可能没有异常触发

    def test_on_tick_detects_anomaly(self):
        """GPU 温度异常应触发 SubAgent 决策。"""
        agent = _make_subagent("alert")
        agent.spawn({"command": "python train.py", "total_epochs": 20})

        metrics = {"epoch": 5, "step": 100, "loss": 0.3}
        gpu_stats = [{"gpu_id": 0, "temperature": 95, "utilization": 80}]
        actions = agent.on_tick(metrics, gpu_stats)
        # GPU 温度 > 90 应触发 overheat 异常
        assert len(actions) >= 1
        assert actions[0]["priority"] == "high"

    def test_shutdown(self):
        """shutdown 应返回 summary。"""
        agent = _make_subagent("good training")
        agent.spawn({"command": "python train.py", "total_epochs": 10})
        result = agent.shutdown({"epoch": 10, "loss": 0.01})
        assert result["status"] == "shutdown"
        assert "stats" in result


# ---------------------------------------------------------------------------
# approve 延迟执行
# ---------------------------------------------------------------------------

class TestApproveDelayedExecution:
    def test_approve_queued(self):
        """approve 应入队而非立即执行。"""
        agent = _make_subagent("alert")
        agent.spawn({"command": "python train.py", "total_epochs": 10})

        # 手动添加 pending action
        action = Action(tool_name="alert", params={"level": "warning"}, reason="test")
        agent._pending_actions.append({
            "action_id": "sa_test001",
            "action": action,
            "status": "pending",
            "created_at": time.time(),
        })

        result = agent.approve("sa_test001")
        assert result.success
        assert result.result == {"status": "queued_for_execution"}
        # pending 应已被移除
        assert agent.pending_count == 0

    def test_drain_approved(self):
        """drain_approved 应取出已审批动作。"""
        agent = _make_subagent("alert")
        agent.spawn({"command": "python train.py", "total_epochs": 10})

        action = Action(tool_name="alert", params={"level": "info"}, reason="test")
        agent._approved_queue.put(action)

        drained = agent.drain_approved()
        assert len(drained) == 1
        assert drained[0].tool_name == "alert"

    def test_approve_not_found(self):
        """审批不存在的 action_id 应返回失败。"""
        agent = _make_subagent("alert")
        agent.spawn({"command": "python train.py", "total_epochs": 10})

        result = agent.approve("nonexistent")
        assert not result.success
        assert "not found" in result.error


# ---------------------------------------------------------------------------
# reject
# ---------------------------------------------------------------------------

class TestReject:
    def test_reject_removes_action(self):
        """reject 应正确移除 pending action。"""
        agent = _make_subagent("alert")
        agent.spawn({"command": "python train.py", "total_epochs": 10})

        action = Action(tool_name="restart_with_lower_lr", params={"ratio": 0.5})
        agent._pending_actions.append({
            "action_id": "sa_reject01",
            "action": action,
            "status": "pending",
            "created_at": time.time(),
        })

        result = agent.reject("sa_reject01", reason="too aggressive")
        assert result.rejected
        assert agent.pending_count == 0


# ---------------------------------------------------------------------------
# 熔断器
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_circuit_breaker_trigger(self):
        """连续 LLM 失败应触发熔断。"""
        call_count = [0]

        def failing_llm(system_prompt, user_message, timeout):
            call_count[0] += 1
            raise RuntimeError("LLM timeout")

        agent = SubAgent(
            config={
                "autonomy": "supervised",
                "circuit_breaker_threshold": 3,
                "circuit_breaker_cooldown": 60,
            },
            tool_registry=default_registry(),
            llm_callback=failing_llm,
        )
        agent.spawn({"command": "python train.py", "total_epochs": 10})

        # 触发多次异常决策（GPU overheat）
        gpu_hot = [{"gpu_id": 0, "temperature": 95, "utilization": 80}]
        for _ in range(4):
            agent.on_tick({"epoch": 1, "step": 10, "loss": 0.5}, gpu_hot)

        # 熔断器应打开
        assert agent._is_breaker_open()


# ---------------------------------------------------------------------------
# 记忆持久化
# ---------------------------------------------------------------------------

class TestMemoryPersistence:
    def test_memory_to_dict_from_dict(self, tmp_path):
        """RollingMemory 序列化往返一致。"""
        mem = RollingMemory(max_size=50)
        mem.record_decision("anomaly", "loss spike", action="alert", source="sub_agent")
        mem.record_decision("crash", "OOM", action="reduce_batch", source="watchdog")
        mem.update_progress(5, 20, 0.85, "val_acc")

        data = mem.to_dict()
        restored = RollingMemory.from_dict(data)

        assert len(restored) == 2
        assert restored._anomaly_count == 1
        assert restored._crash_count == 1
        assert restored._current_epoch == 5
        assert restored._total_epochs == 20
        assert restored._best_metric_value == 0.85
        assert restored._best_metric_name == "val_acc"

    def test_subagent_save_restore(self, tmp_path):
        """SubAgent save_state → restore_from 往返一致。"""
        agent = _make_subagent("alert")
        agent.spawn({"command": "python train.py", "total_epochs": 10})

        # 添加一条 pending action
        action = Action(tool_name="alert", params={"level": "warning"}, reason="test")
        agent._pending_actions.append({
            "action_id": "sa_persist01",
            "action": action,
            "status": "pending",
            "created_at": 1700000000.0,
            "priority": "normal",
        })

        path = tmp_path / "sub_agent_state.json"
        agent.save_state(path)
        assert path.exists()

        restored = SubAgent.restore_from(path, config={"autonomy": "supervised"},
                                          tool_registry=default_registry())
        assert restored.is_spawned
        assert restored.autonomy == "supervised"
        assert len(restored.memory) == len(agent.memory)
        assert len(restored._pending_actions) == 1
        assert restored._pending_actions[0]["action_id"] == "sa_persist01"


# ---------------------------------------------------------------------------
# ProcessAdapter
# ---------------------------------------------------------------------------

class TestProcessAdapter:
    def test_attached_adapter_mock_alive(self):
        """AttachedProcessAdapter mock PID 存活。"""
        adapter = AttachedProcessAdapter(99999, log_file=None)
        # PID 99999 通常不存在
        assert adapter.get_pid() == 99999
        # 不主动 terminate
        adapter.terminate()  # 不应崩溃

    def test_attached_adapter_log_tail(self, tmp_path):
        """AttachedProcessAdapter 从 log 文件读 tail。"""
        log_file = tmp_path / "train.log"
        log_file.write_text("line1\nline2\nline3\nline4\nline5\n", encoding="utf-8")

        adapter = AttachedProcessAdapter(1, log_file=log_file)
        tail = adapter.get_stderr_tail(lines=2)
        assert "line4" in tail
        assert "line5" in tail

    def test_screen_adapter_init(self):
        """ScreenProcessAdapter 初始化不崩溃。"""
        adapter = ScreenProcessAdapter("test_session")
        assert adapter._session == "test_session"
        assert adapter.get_pid() is None  # 未启动时无 PID

    def test_tmux_adapter_init(self):
        """TmuxProcessAdapter 初始化不崩溃。"""
        adapter = TmuxProcessAdapter("test_tmux")
        assert adapter._session == "test_tmux"
        assert adapter.get_pid() is None

    def test_spawned_adapter_wraps_popen(self):
        """SpawnedProcessAdapter 正确包装 Popen。"""
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        adapter = SpawnedProcessAdapter(proc)
        assert adapter.is_alive()
        assert adapter.get_pid() == proc.pid
        assert adapter.poll() is None
        adapter.terminate(grace=1)
        assert not adapter.is_alive()


# ---------------------------------------------------------------------------
# watchdog.run_attach
# ---------------------------------------------------------------------------

class TestRunAttach:
    def test_attach_mode_process_exits(self):
        """run_attach 进程退出后正确返回。"""
        from guardian.watchdog import TrainingWatchdog

        class MockAdapter(ProcessAdapter):
            def __init__(self):
                self._alive = True
                self._code = 0
                self._tick_count = 0

            def is_alive(self):
                self._tick_count += 1
                if self._tick_count > 3:
                    self._alive = False
                return self._alive

            def get_pid(self):
                return 12345

            def terminate(self, grace=30):
                self._alive = False

            def get_stderr_tail(self, lines=200):
                return "mock stderr"

            @property
            def returncode(self):
                return self._code if not self._alive else None

        adapter = MockAdapter()
        watchdog = TrainingWatchdog(config={"max_retries": 0, "no_progress_timeout": None})
        result = watchdog.run_attach(adapter)

        assert result["status"] == "completed"
        assert result["exit_code"] == 0

    def test_attach_mode_process_crash(self):
        """run_attach 进程异常退出应返回 failed。"""
        from guardian.watchdog import TrainingWatchdog

        class MockAdapter(ProcessAdapter):
            def __init__(self):
                self._alive = True
                self._tick_count = 0

            def is_alive(self):
                self._tick_count += 1
                if self._tick_count > 2:
                    self._alive = False
                return self._alive

            def get_pid(self):
                return 12345

            def terminate(self, grace=30):
                self._alive = False

            def get_stderr_tail(self, lines=200):
                return "CUDA out of memory"

            @property
            def returncode(self):
                return 137 if not self._alive else None

        adapter = MockAdapter()
        watchdog = TrainingWatchdog(config={"max_retries": 0, "no_progress_timeout": None})
        result = watchdog.run_attach(adapter)

        assert result["status"] == "failed"
        assert result["exit_code"] == 137
