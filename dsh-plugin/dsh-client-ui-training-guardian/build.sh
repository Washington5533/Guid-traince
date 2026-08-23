#!/bin/bash
# WSL build script — run from WSL bash
# Usage: bash build.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "[build.sh] TypeScript type check..."
pnpm exec tsc --noEmit

echo "[build.sh] tsdown bundle..."
pnpm exec tsdown

echo "[build.sh] Done."
echo ""
echo "Output:"
echo "  Host (ESM):  dist/index.mjs"
echo "  Client (CJS): dist-client/client/index.js  (wrapped in __ModuleLoader__.load)"
