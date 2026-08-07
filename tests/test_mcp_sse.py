#!/usr/bin/env python
"""MCP SSE 协议端到端测试。

先启动 MCP server:
    python run.py serve --transport sse

然后运行本脚本:
    python tests/test_mcp_sse.py
"""
import socket
import json
import re
import threading
import time
import sys

BASE_HOST = "127.0.0.1"
BASE_PORT = 8766
MSG_PATH = "/messages/"

# 共享状态
sse_sock = [None]  # SSE 连接的 socket
session_id = [None]
lock = threading.Lock()
sse_events = []  # 从 SSE 流收到的所有事件
event_ready = threading.Event()


def sse_listener():
    """后台线程：维持 SSE 连接 + 读取返回的事件。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(60)
    try:
        sock.connect((BASE_HOST, BASE_PORT))
        sock.sendall(
            f"GET /sse HTTP/1.1\r\nHost: {BASE_HOST}:{BASE_PORT}\r\n"
            f"Accept: text/event-stream\r\nConnection: keep-alive\r\n\r\n"
            .encode()
        )
        # 读 HTTP 头
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += sock.recv(1)
        print("[SSE] HTTP 连接已建立")

        # 读取 SSE 事件流
        line_buf = b""
        event_data = {}
        while True:
            line_buf += sock.recv(1)
            if b"\n" in line_buf:
                line_bytes, line_buf = line_buf.split(b"\n", 1)
                line = line_bytes.decode("utf-8").rstrip("\r")

                if line == "":
                    # 空行 = 事件结束
                    if event_data:
                        with lock:
                            sse_events.append(dict(event_data))
                        event_ready.set()
                        # 提取 session_id（从 endpoint 事件）
                        if "data" in event_data and "session_id=" in str(event_data["data"]):
                            m = re.search(r"session_id=([a-zA-Z0-9\-]+)",
                                          str(event_data["data"]))
                            if m:
                                session_id[0] = m.group(1)
                                print(f"[SSE] session_id = {session_id[0]}")
                        # 打印工具调用响应
                        if "data" in event_data:
                            try:
                                body = json.loads(str(event_data["data"]))
                                if "result" in body:
                                    rid = body.get("id", "?")
                                    res = body.get("result", {})
                                    content = res.get("content", [{}])
                                    text = content[0].get("text", "") if content else str(res)
                                    print(f"[SSE] id={rid}: {text[:150]}...")
                            except (json.JSONDecodeError, ValueError):
                                pass
                        event_data = {}
                elif ":" in line:
                    field, _, value = line.partition(":")
                    field = field.strip()
                    value = value.strip()
                    if field in ("event", "data", "id", "retry"):
                        event_data[field] = value
    except Exception as e:
        print(f"[SSE] 异常: {e}")
    finally:
        sock.close()


def mcp_send(payload: dict) -> int | None:
    """向 MCP 发送 JSON-RPC POST 请求，返回 HTTP 状态码。"""
    sid = session_id[0]
    if not sid:
        print("[POST] 尚无 session_id")
        return None
    body = json.dumps(payload).encode("utf-8")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect((BASE_HOST, BASE_PORT))
        request = (
            f"POST {MSG_PATH}?session_id={sid} HTTP/1.1\r\n"
            f"Host: {BASE_HOST}:{BASE_PORT}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n"
        ).encode() + body
        sock.sendall(request)
        # 读 HTTP 响应
        buf = b""
        while b"\r\n\r\n" not in buf:
            buf += sock.recv(1)
        header = buf.decode("utf-8")
        code = int(header.split(" ")[1])
        # 读 body（如有）
        hdr_end = buf.index(b"\r\n\r\n") + 4
        remaining = buf[hdr_end:]
        content_len = 0
        for h in header.split("\r\n"):
            if h.lower().startswith("content-length:"):
                content_len = int(h.split(":")[1].strip())
        while len(remaining) < content_len:
            remaining += sock.recv(content_len - len(remaining))
        if content_len > 0 and remaining.strip():
            try:
                resp_body = json.loads(remaining.decode("utf-8"))
                result = resp_body.get("result", {})
                content = result.get("content", [{}])
                text = content[0].get("text", "") if content else str(resp_body)[:300]
                print(f"[HTTP] 直接响应: {text[:200]}")
            except (json.JSONDecodeError, ValueError):
                pass  # MCP SSE 通常返回 202 + 空 body，响应经 SSE 通道
        return code
    except Exception as e:
        print(f"[POST] 异常: {e}")
        return None
    finally:
        sock.close()


# ---- 主流程 ----
print("=" * 60)
print("  MCP SSE 协议测试")
print(f"  目标: http://{BASE_HOST}:{BASE_PORT}")
print("=" * 60)

# 1. 启动 SSE 监听
print("\n--- Step 1: SSE 连接 ---")
t = threading.Thread(target=sse_listener, daemon=True)
t.start()
for _ in range(30):  # 等最多 3 秒
    if session_id[0]:
        break
    time.sleep(0.1)
if not session_id[0]:
    print("错误: 获取 session_id 超时。MCP server 是否运行？")
    sys.exit(1)
time.sleep(0.3)

# 2. initialize
print("\n--- Step 2: initialize ---")
code = mcp_send({
    "jsonrpc": "2.0", "id": 1, "method": "initialize",
    "params": {
        "protocolVersion": "2024-11-05",
        "capabilities": {},
        "clientInfo": {"name": "test", "version": "1.0"}
    }
})
print(f"  POST 状态码: {code}")
time.sleep(0.3)

# 3. tools/list
print("\n--- Step 3: tools/list ---")
code = mcp_send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
print(f"  POST 状态码: {code}")
time.sleep(0.5)

# 打印 SSE 收到的工具列表
with lock:
    for ev in sse_events:
        data = ev.get("data", "")
        try:
            body = json.loads(data)
            if body.get("id") == 2:
                tools = body.get("result", {}).get("tools", [])
                print(f"\n  共 {len(tools)} 个工具:")
                for tool in tools:
                    print(f"    - {tool['name']}: {tool.get('description', '')[:70]}")
        except (json.JSONDecodeError, ValueError):
            pass

# 4. get_training_status
print("\n--- Step 4: get_training_status ---")
prev_count = len(sse_events)
code = mcp_send({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "get_training_status", "arguments": {"kwargs": {}}}})
print(f"  POST 状态码: {code}")
time.sleep(0.5)

# 打印响应
with lock:
    for ev in sse_events[prev_count:]:
        data = ev.get("data", "")
        try:
            body = json.loads(data)
            if body.get("id") == 3:
                result = body.get("result", {})
                content = result.get("content", [{}])
                text = content[0].get("text", "") if content else str(result)
                # 格式化 JSON
                try:
                    obj = json.loads(text)
                    text = json.dumps(obj, indent=2, ensure_ascii=False)
                except (json.JSONDecodeError, ValueError):
                    pass
                print(f"\n  响应:\n{text[:600]}")
        except (json.JSONDecodeError, ValueError):
            pass

# 5. get_metrics_history
print("\n--- Step 5: get_metrics_history ---")
prev_count = len(sse_events)
code = mcp_send({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                 "params": {"name": "get_metrics_history", "arguments": {"kwargs": {"limit": 5}}}})
print(f"  POST 状态码: {code}")
time.sleep(0.5)

with lock:
    for ev in sse_events[prev_count:]:
        data = ev.get("data", "")
        try:
            body = json.loads(data)
            if body.get("id") == 4:
                result = body.get("result", {})
                content = result.get("content", [{}])
                text = content[0].get("text", "") if content else str(result)
                print(f"  响应: {text[:400]}")
        except (json.JSONDecodeError, ValueError):
            pass

print("\n" + "=" * 60)
print("  MCP SSE 协议测试完成")
print("=" * 60)
