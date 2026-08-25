"""进程管理抽象层：屏蔽 spawn/attach/screen/tmux 的差异。

四种模式统一为 ProcessAdapter 接口，供 watchdog 的 tick 循环使用。
详见 plan: SubAgent 进程兼容 + 生命周期绑定
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

from guardian.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "ProcessAdapter",
    "SpawnedProcessAdapter",
    "AttachedProcessAdapter",
    "ScreenProcessAdapter",
    "TmuxProcessAdapter",
]

_WINDOWS = sys.platform == "win32"


# ---------------------------------------------------------------------------
# ABC
# ---------------------------------------------------------------------------

class ProcessAdapter(ABC):
    """进程管理统一接口。"""

    @abstractmethod
    def is_alive(self) -> bool:
        ...

    def poll(self) -> int | None:
        """兼容 subprocess.Popen.poll() 语义：存活返回 None，已退出返回 code。"""
        if self.is_alive():
            return None
        return self.returncode

    @abstractmethod
    def get_pid(self) -> int | None:
        ...

    @abstractmethod
    def terminate(self, grace: float = 30) -> None:
        ...

    @abstractmethod
    def get_stderr_tail(self, lines: int = 200) -> str:
        ...

    @property
    @abstractmethod
    def returncode(self) -> int | None:
        ...

    # 兼容 watchdog._terminate 中 proc.communicate() 调用
    def communicate(self, timeout: float = 5) -> tuple[str, str]:
        return ("", "")


# ---------------------------------------------------------------------------
# SpawnedProcessAdapter — 包装 subprocess.Popen
# ---------------------------------------------------------------------------

class SpawnedProcessAdapter(ProcessAdapter):
    """包装已有的 Popen 对象，保持向后兼容。"""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc

    def is_alive(self) -> bool:
        return self._proc.poll() is None

    def get_pid(self) -> int | None:
        return self._proc.pid

    def terminate(self, grace: float = 30) -> None:
        proc = self._proc
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def get_stderr_tail(self, lines: int = 200) -> str:
        # Popen 模式下 stderr 由 watchdog.run() 的 PIPE 收集
        # 此处尝试从 communicate 获取残余
        try:
            if self._proc.stderr:
                _, err = self._proc.communicate(timeout=2)
                return err[-4000:] if err else ""
        except Exception:
            pass
        return ""

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    def communicate(self, timeout: float = 5) -> tuple[str, str]:
        return self._proc.communicate(timeout=timeout)


# ---------------------------------------------------------------------------
# AttachedProcessAdapter — 附加到已有 PID
# ---------------------------------------------------------------------------

class AttachedProcessAdapter(ProcessAdapter):
    """通过 PID 附加到已运行的训练进程。"""

    def __init__(self, pid: int, log_file: str | Path | None = None,
                 allow_terminate: bool = False):
        self._pid = pid
        self._log_file = Path(log_file) if log_file else None
        self._allow_terminate = allow_terminate
        self._exit_code: int | None = None

    def is_alive(self) -> bool:
        if _WINDOWS:
            return self._is_alive_windows()
        try:
            os.kill(self._pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False
        except OSError:
            return False

    def _is_alive_windows(self) -> bool:
        """Windows: 通过 tasklist 检查 PID 是否存在。"""
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {self._pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            return str(self._pid) in result.stdout
        except Exception:
            return False

    def get_pid(self) -> int | None:
        return self._pid

    def terminate(self, grace: float = 30) -> None:
        if not self._allow_terminate:
            logger.info("AttachedProcessAdapter: 不主动终止附加进程 PID=%d", self._pid)
            return
        if _WINDOWS:
            subprocess.run(["taskkill", "/PID", str(self._pid)], capture_output=True, timeout=10)
        else:
            try:
                os.kill(self._pid, 15)  # SIGTERM
                deadline = time.monotonic() + grace
                while time.monotonic() < deadline and self.is_alive():
                    time.sleep(0.5)
                if self.is_alive():
                    os.kill(self._pid, 9)  # SIGKILL
            except (ProcessLookupError, PermissionError):
                pass

    def get_stderr_tail(self, lines: int = 200) -> str:
        if self._log_file and self._log_file.is_file():
            try:
                all_lines = self._log_file.read_text(encoding="utf-8", errors="replace").splitlines()
                return "\n".join(all_lines[-lines:])
            except Exception:
                pass
        return ""

    @property
    def returncode(self) -> int | None:
        if self.is_alive():
            return None
        if self._exit_code is not None:
            return self._exit_code
        # 尝试获取退出码
        if _WINDOWS:
            return self._get_exit_code_windows()
        try:
            pid, status = os.waitpid(self._pid, os.WNOHANG)
            if pid == self._pid:
                self._exit_code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
                return self._exit_code
        except ChildProcessError:
            pass
        return None

    def _get_exit_code_windows(self) -> int | None:
        try:
            result = subprocess.run(
                ["tasklist", "/FI", f"PID eq {self._pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
            )
            if str(self._pid) not in result.stdout:
                return -1  # 进程已退出，无法获取退出码
        except Exception:
            pass
        return None


# ---------------------------------------------------------------------------
# ScreenProcessAdapter — screen 会话中启动训练
# ---------------------------------------------------------------------------

class ScreenProcessAdapter(ProcessAdapter):
    """在 screen 会话中启动训练命令。"""

    def __init__(self, session_name: str):
        self._session = session_name
        self._pid: int | None = None
        self._exit_code: int | None = None

    def start(self, cmd: list[str]) -> None:
        """screen -dmS session_name cmd..."""
        full_cmd = ["screen", "-dmS", self._session] + cmd
        subprocess.run(full_cmd, check=True, timeout=10)
        # 等待会话启动并获取 PID
        time.sleep(0.5)
        self._pid = self._resolve_pid()
        logger.info("Screen 会话 '%s' 已启动 (PID=%s)", self._session, self._pid)

    def _resolve_pid(self) -> int | None:
        """从 screen -ls 解析会话 PID。"""
        try:
            result = subprocess.run(
                ["screen", "-ls"], capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if self._session in line:
                    # 格式: "12345.session_name\t(Detached)"
                    parts = line.strip().split(".")
                    if parts:
                        return int(parts[0].strip())
        except Exception:
            pass
        return None

    def is_alive(self) -> bool:
        try:
            result = subprocess.run(
                ["screen", "-ls"], capture_output=True, text=True, timeout=5,
            )
            return self._session in result.stdout
        except Exception:
            return False

    def get_pid(self) -> int | None:
        if self._pid is None:
            self._pid = self._resolve_pid()
        return self._pid

    def terminate(self, grace: float = 30) -> None:
        try:
            subprocess.run(
                ["screen", "-S", self._session, "-X", "quit"],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass

    def get_stderr_tail(self, lines: int = 200) -> str:
        """通过 screen hardcopy 截取当前屏幕内容。"""
        import tempfile
        cap_file = Path(tempfile.gettempdir()) / f"screen_cap_{self._session}"
        try:
            subprocess.run(
                ["screen", "-S", self._session, "-X", "hardcopy", str(cap_file)],
                capture_output=True, timeout=5,
            )
            time.sleep(0.2)  # screen 写文件需要一点时间
            if cap_file.is_file():
                text = cap_file.read_text(encoding="utf-8", errors="replace")
                all_lines = text.splitlines()
                return "\n".join(all_lines[-lines:])
        except Exception:
            pass
        return ""

    @property
    def returncode(self) -> int | None:
        if self.is_alive():
            return None
        return self._exit_code  # screen 退出后无法可靠获取退出码


# ---------------------------------------------------------------------------
# TmuxProcessAdapter — tmux 会话中启动训练
# ---------------------------------------------------------------------------

class TmuxProcessAdapter(ProcessAdapter):
    """在 tmux 会话中启动训练命令。"""

    def __init__(self, session_name: str):
        self._session = session_name
        self._pid: int | None = None
        self._exit_code: int | None = None

    def start(self, cmd: list[str]) -> None:
        """tmux new-session -d -s session_name cmd..."""
        full_cmd = ["tmux", "new-session", "-d", "-s", self._session] + cmd
        subprocess.run(full_cmd, check=True, timeout=10)
        time.sleep(0.5)
        self._pid = self._resolve_pid()
        logger.info("tmux 会话 '%s' 已启动 (PID=%s)", self._session, self._pid)

    def _resolve_pid(self) -> int | None:
        """tmux list-panes -t session -F '#{pane_pid}'"""
        try:
            result = subprocess.run(
                ["tmux", "list-panes", "-t", self._session, "-F", "#{pane_pid}"],
                capture_output=True, text=True, timeout=5,
            )
            pid_str = result.stdout.strip()
            if pid_str:
                return int(pid_str.splitlines()[0])
        except Exception:
            pass
        return None

    def is_alive(self) -> bool:
        try:
            result = subprocess.run(
                ["tmux", "has-session", "-t", self._session],
                capture_output=True, timeout=5,
            )
            return result.returncode == 0
        except Exception:
            return False

    def get_pid(self) -> int | None:
        if self._pid is None:
            self._pid = self._resolve_pid()
        return self._pid

    def terminate(self, grace: float = 30) -> None:
        try:
            subprocess.run(
                ["tmux", "kill-session", "-t", self._session],
                capture_output=True, timeout=10,
            )
        except Exception:
            pass

    def get_stderr_tail(self, lines: int = 200) -> str:
        """tmux capture-pane -t session -p"""
        try:
            result = subprocess.run(
                ["tmux", "capture-pane", "-t", self._session, "-p"],
                capture_output=True, text=True, timeout=5,
            )
            all_lines = result.stdout.splitlines()
            return "\n".join(all_lines[-lines:])
        except Exception:
            return ""

    @property
    def returncode(self) -> int | None:
        if self.is_alive():
            return None
        return self._exit_code  # tmux 退出后无法可靠获取退出码
