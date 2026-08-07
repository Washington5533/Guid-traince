"""cp_6 · 告警推送：终端 / webhook / 邮件，带静默期防刷屏。

架构中立——只负责发消息，不读训练状态也不干预训练。
干预类告警必须同时说明"做了什么"和"代价是什么"。详见 checkpoint/cp_6.md
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

from guardian.logging_config import get_logger

logger = get_logger(__name__)

LEVEL_ICON = {"info": "[i]", "warning": "[!]", "error": "[X]"}


def ensure_utf8_stdout() -> None:
    """Windows 控制台默认 GBK，会把中文输出成乱码。尽量切到 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        reconfig = getattr(stream, "reconfigure", None)
        if reconfig is not None:
            try:
                reconfig(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                # UTF-8 重配置失败不影响后续运行，静默跳过
                pass


class Notifier:
    def __init__(self, config: dict | None = None):
        self.cfg = config or {}
        self.channels = list(self.cfg.get("channels") or ["terminal"])
        self.cooldown = float(self.cfg.get("cooldown", 300))
        self._last_sent: dict[str, float] = {}
        self.history: list[dict[str, Any]] = []
        ensure_utf8_stdout()

    def _should_send(self, alert_type: str) -> bool:
        """静默期检查：同类告警在 cooldown 内不重复推送远程。"""
        now = time.monotonic()
        last = self._last_sent.get(alert_type)
        if last is not None and (now - last) < self.cooldown:
            return False
        self._last_sent[alert_type] = now
        return True

    def send(
        self,
        title: str,
        message: str,
        alert_type: str = "generic",
        level: str = "info",
        response: dict | None = None,
    ) -> dict:
        """发一条告警。response 描述本次应对动作及其代价（见 cp_6.md）。"""
        event = {
            "title": title,
            "message": message,
            "alert_type": alert_type,
            "level": level,
            "timestamp": int(time.time()),
            "response": response or {"source": "rule_default", "action": "alert_only", "restart": False},
        }
        self.history.append(event)

        fresh = self._should_send(alert_type)
        # error 级永不静默：nan_inf 这类底线必须每次都出来
        if level == "error":
            fresh = True

        if "terminal" in self.channels:
            self._print_terminal(event)
        if fresh:
            if "webhook" in self.channels:
                self._send_webhook(event)
            if "email" in self.channels:
                self._send_email(event)
        return event

    def _print_terminal(self, event: dict) -> None:
        icon = LEVEL_ICON.get(event["level"], "[i]")
        resp = event.get("response") or {}
        lines = [
            f"{icon} [{event['level'].upper()}] {event['title']}",
            f"    {event['message']}",
        ]
        action = resp.get("action")
        if action and action != "alert_only":
            src = resp.get("source", "rule_default")
            lines.append(f"    应对: {src} -> {action}")
            if resp.get("restart"):
                frm = resp.get("resumed_from", "?")
                wasted = resp.get("wasted_epochs")
                cost = f"从 {frm} 重启"
                if wasted is not None:
                    cost += f"，作废约 {wasted} epoch"
                lines.append(f"    代价: {cost}")
        logger.info("%s", "\n".join(lines))

    def _send_webhook(self, event: dict) -> None:
        url = os.environ.get(str(self.cfg.get("webhook_url_env", "GUARDIAN_WEBHOOK_URL")) or "")
        if not url:
            return
        try:
            import requests  # 可选依赖
        except ImportError:
            logger.warning("webhook 已配置但未安装 requests，跳过推送")
            return
        try:
            requests.post(
                url,
                data=json.dumps(event, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                timeout=float(self.cfg.get("webhook_timeout", 10)),
            )
        except Exception as exc:  # 推送失败绝不影响看护循环
            logger.warning("webhook 推送失败: %s", exc, exc_info=True)

    def _send_email(self, event: dict) -> None:  # pragma: no cover - 需要 SMTP
        logger.info("email 渠道尚未实现，v0 仅支持 terminal / webhook")
