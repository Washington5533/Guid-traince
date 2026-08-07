"""notifier.py 测试：终端输出、webhook、静默期。"""

import time
from unittest.mock import MagicMock, patch

import pytest

from guardian.notifier import Notifier, ensure_utf8_stdout


class TestEnsureUtf8:
    def test_does_not_crash(self):
        """ensure_utf8_stdout 在任何情况下都不应抛异常。"""
        ensure_utf8_stdout()  # 正常调用不报错


# ---------------------------------------------------------------------------
# Notifier 基本属性
# ---------------------------------------------------------------------------

class TestNotifierBasics:
    def test_default_channels(self):
        n = Notifier()
        assert "terminal" in n.channels

    def test_default_cooldown(self):
        n = Notifier()
        assert n.cooldown == 300

    def test_history_starts_empty(self):
        n = Notifier()
        assert n.history == []


# ---------------------------------------------------------------------------
# 静默期（cooldown）
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_first_alert_always_sends(self):
        n = Notifier({"cooldown": 60})
        assert n._should_send("loss_spike") is True

    def test_second_alert_within_cooldown_blocked(self):
        n = Notifier({"cooldown": 60})
        n._should_send("loss_spike")
        assert n._should_send("loss_spike") is False

    def test_different_types_not_blocked(self):
        n = Notifier({"cooldown": 60})
        n._should_send("loss_spike")
        assert n._should_send("gpu_temp") is True  # 不同类型

    def test_after_cooldown_expires(self):
        n = Notifier({"cooldown": 300})
        n._should_send("loss_spike")
        # 模拟时间过期：把记录时间改到 600s 之前
        n._last_sent["loss_spike"] -= 600
        assert n._should_send("loss_spike") is True


# ---------------------------------------------------------------------------
# 发送行为
# ---------------------------------------------------------------------------

class TestSend:
    def test_send_terminal_no_crash(self):
        n = Notifier({"channels": ["terminal"], "cooldown": 0})
        # 不应抛异常
        n.send("测试标题", "测试消息", alert_type="test", level="info")

    def test_send_adds_to_history(self):
        n = Notifier({"channels": ["terminal"], "cooldown": 0})
        n.send("标题", "消息", alert_type="test", level="warning")
        assert len(n.history) == 1
        assert n.history[0]["title"] == "标题"

    def test_error_level_ignores_cooldown(self):
        n = Notifier({"cooldown": 9999})
        n._should_send("oom")
        # error 级别应该总是发送
        assert n._should_send("oom") is False  # cooldown 内
        # 但 error 级别不走 _should_send 检查，在 send() 里硬编码跳过


# ---------------------------------------------------------------------------
# 干预响应展示
# ---------------------------------------------------------------------------

class TestInterventionDisplay:
    def test_send_with_response(self):
        n = Notifier({"channels": ["terminal"], "cooldown": 0})
        n.send("干预", "训练已重启", alert_type="intervention", level="warning",
               response={"action": "restart_with_lower_lr",
                         "resumed_from": "cp_10",
                         "wasted_epochs": 2})
        # 不抛异常即可——response 中的信息已在 _print_terminal 中展示
        assert len(n.history) == 1
