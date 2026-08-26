#!/bin/bash
# ============================================================================
# Guardian 云端一键启动
# 在 GPU 服务器上执行，同时拉起 Dashboard + MCP + 可选训练守护
#
# 用法：
#   ./start-cloud.sh
#   ./start-cloud.sh -- python train.py --epochs 100
#   ./start-cloud.sh --dash-port 8767 --mcp-port 8768 -- python train2.py
# ============================================================================

set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd ../.. && pwd)"

echo "=== Guardian 云端启动 ==="

# 依赖检查（缺失只提示，不阻断——start 内部也会逐项降级）
echo "[依赖] 检查 Python 环境..."
PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -c "import guardian" 2>/dev/null || {
    echo ""
    echo "  guardian 核心依赖缺失，请先执行："
    echo "    pip install -r \"$ROOT/requirements-core.txt\""
    echo "    pip install -r \"$ROOT/requirements-mcp.txt\"        # MCP 接入（可选）"
    echo "    pip install -r \"$ROOT/requirements-dashboard.txt\"  # Web 面板（可选）"
    exit 1
}

echo "[依赖] OK"
echo ""

exec env PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" python -m guardian start "$@"
