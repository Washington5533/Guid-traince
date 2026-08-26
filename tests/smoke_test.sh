#!/bin/bash
# Guardian Dashboard 快速功能测试
# 用法: bash tests/smoke_test.sh

HOST="http://127.0.0.1:8765"
PASS=0
FAIL=0

check() { local desc="$1" url="$2" expect="$3"
  code=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
  if [ "$code" = "$expect" ]; then
    echo "  ✅ $desc"
    ((PASS++))
  else
    echo "  ❌ $desc (期望 $expect, 实际 $code)"
    ((FAIL++))
  fi
}

check_json() { local desc="$1" url="$2" key="$3"
  val=$(curl -s "$url" 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('$key',''))" 2>/dev/null)
  if [ -n "$val" ]; then
    echo "  ✅ $desc → $key=$val"
    ((PASS++))
  else
    echo "  ❌ $desc (key '$key' 为空)"
    ((FAIL++))
  fi
}

echo "========================================"
echo "  Guardian Dashboard 功能测试"
echo "  $HOST"
echo "========================================"

echo ""
echo "--- 基础端点 ---"
check   "健康检查"       "$HOST/health"            "200"
check   "首页"           "$HOST/"                  "200"
check   "进程列表"       "$HOST/api/processes"     "200"
check   "历史进程列表"   "$HOST/api/history"       "200"

echo ""
echo "--- 历史数据 ---"
PID=$(curl -s "$HOST/api/history" | python -c "import sys,json; h=json.load(sys.stdin).get('history',[]); print(h[0]['process_id'] if h else '')" 2>/dev/null)
if [ -n "$PID" ]; then
  check     "历史详情 $PID"        "$HOST/api/history/$PID"          "200"
  check     "历史指标 $PID"        "$HOST/api/history/$PID/metrics?limit=10"  "200"
  check     "历史AI分析 $PID"      "$HOST/api/history/$PID/ai/analyze" "200"  # POST 但 curl 用 GET 测连通性，实际需 POST
  actual_code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$HOST/api/history/$PID/ai/analyze" -H "Content-Type: application/json" -d '{}')
  if [ "$actual_code" = "200" ]; then
    echo "  ✅ 历史AI分析 POST $PID"
    ((PASS++))
  else
    echo "  ❌ 历史AI分析 POST $PID (期望 200, 实际 $actual_code)"
    ((FAIL++))
  fi
else
  echo "  ⚠️  无历史进程，跳过历史数据测试"
fi

echo ""
echo "--- Live 进程 ---"
LIVE=$(curl -s "$HOST/api/processes" | python -c "import sys,json; p=json.load(sys.stdin).get('processes',[]); live=[x for x in p if x.get('source')=='live']; print(live[0]['process_id'] if live else '')" 2>/dev/null)
if [ -n "$LIVE" ]; then
  check     "进程详情 $LIVE"       "$HOST/api/process/$LIVE"         "200"
  check     "进程指标 $LIVE"       "$HOST/api/process/$LIVE/metrics?limit=10" "200"
else
  echo "  ⚠️  无 live 进程（正常——需要先 guarftrain watch 启动训练）"
fi

echo ""
echo "--- 独立工具 ---"
check     "导入格式"       "$HOST/api/history/nonexistent"  "404"  # 测试 404 处理

echo ""
echo "========================================"
echo "  结果: $PASS 通过, $FAIL 失败"
echo "========================================"
exit $FAIL
