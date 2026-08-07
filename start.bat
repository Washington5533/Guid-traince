@echo off
REM ============================================================================
REM Guardian 本地接入 — Windows 一键启动
REM 在本地 Windows 上执行，建立 SSH 隧道 + 打开 Dashboard
REM
REM 用法：
REM   双击 start.bat
REM   或在终端：start.bat user@your-gpu-server
REM
REM 配置：修改下面的 SERVER 变量，或通过命令行参数传入
REM ============================================================================

setlocal EnableExtensions
chcp 65001 >nul
title Guardian 本地接入

REM === 配置区 ===
if not "%~1"=="" (
    set "SERVER=%~1"
) else (
    set "SERVER=user@your-gpu-server"
)
set "DASH_PORT=8765"
set "MCP_PORT=8766"

echo.
echo ============================================================
echo   Guardian 本地接入
echo ============================================================
echo.
echo   服务器: %SERVER%
echo   Dashboard: http://127.0.0.1:%DASH_PORT%
echo   MCP (SSE): http://127.0.0.1:%MCP_PORT%/sse
echo.

REM === SSH 检查 ===
where ssh >nul 2>nul
if errorlevel 1 (
    echo [错误] 未找到 ssh 命令。请安装 OpenSSH 客户端或 Git for Windows。
    echo        Windows 10/11: 设置 ^> 应用 ^> 可选功能 ^> 添加功能 ^> OpenSSH 客户端
    pause
    exit /b 1
)

REM === 建立 SSH 隧道 ===
echo [1/2] 建立 SSH 隧道（后台窗口）...
start "Guardian-SSH-Tunnel (%SERVER%)" ssh -o ServerAliveInterval=60 -L %DASH_PORT%:127.0.0.1:%DASH_PORT% -L %MCP_PORT%:127.0.0.1:%MCP_PORT% %SERVER% -N

REM === 等待隧道就绪 ===
echo [2/2] 等待隧道就绪...
timeout /t 3 /nobreak >nul

REM === 打开 Dashboard ===
echo.
echo 打开 Dashboard: http://127.0.0.1:%DASH_PORT%
start http://127.0.0.1:%DASH_PORT%

REM === 就绪信息 ===
echo.
echo ============================================================
echo   就绪！
echo ============================================================
echo.
echo   Dashboard : http://127.0.0.1:%DASH_PORT%
echo   MCP (SSE)  : http://127.0.0.1:%MCP_PORT%/sse
echo.
echo   Claude Code MCP 配置：
echo   （添加到 .claude/settings.local.json 的 mcpServers 中）
echo.
echo     "guardian-remote": {
echo       "type": "http",
echo       "url": "http://127.0.0.1:%MCP_PORT%/sse"
echo     }
echo.
echo   关闭此窗口不会断开 SSH 隧道。
echo   要停止隧道，请关闭 "Guardian-SSH-Tunnel" 窗口。
echo ============================================================
echo.
pause
