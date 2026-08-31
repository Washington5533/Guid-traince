# 包分发与配置指南 · Packages & Integration

本页汇总 Training Guardian 全部发布产物的**存储位置、安装方式与使用信息**，以及 **DSH 插件**和**外部 AI Agent（MCP）**两类接入方的配置方案。

## 一、包总览

| 产物 | 渠道 | 当前版本 | 地址 | 安装 / 获取 |
|------|------|---------|------|-------------|
| 核心包 `guarftrain` | PyPI | 0.3.0 | <https://pypi.org/project/guarftrain/> | `pip install guarftrain` |
| DSH 面板插件 `@rrrelink/dsh-client-ui-training-guardian` | npm | 0.1.2 | <https://www.npmjs.com/package/@rrrelink/dsh-client-ui-training-guardian> | `dsh plugin add @rrrelink/dsh-client-ui-training-guardian --profile web` |
| 构建产物（wheel + sdist） | GitHub Release | v0.3.0 | <https://github.com/Washington5533/Guid-traince/releases/tag/v0.3.0> | Release Assets 下载 |
| 源码仓库 | GitHub | main | <https://github.com/Washington5533/Guid-traince> | `git clone` |
| 在线演示 | Streamlit Cloud | — | <https://guarftrain-azvjiidegvdmnnkhczmfq2.streamlit.app/> | 浏览器直接访问 |

可选依赖组（按需安装，核心仅 ~2MB）：

| 组 | 内容 | 安装 |
|----|------|------|
| `agent` | AI 决策层（anthropic SDK） | `pip install guarftrain[agent]` |
| `mcp` | MCP 外部 Agent 接入 | `pip install guarftrain[mcp]` |
| `dashboard` | Web 控制面板 | `pip install guarftrain[dashboard]` |
| `full` | 以上全部 | `pip install guarftrain[full]` |

## 二、核心包（PyPI）使用信息

```bash
pip install guarftrain                 # Python 3.10+
cd /path/to/your-project
guarftrain init                        # 自动扫描训练脚本，生成 configs/contract.yaml
guarftrain watch -- python train.py    # 纯规则守护（零外部依赖）
```

常用启动形态：

```bash
# 全功能守护：AI 决策 + Dashboard + MCP
export ANTHROPIC_API_KEY=your-key
guarftrain watch --agent --with-dashboard --with-mcp -- python train.py

# 一键启动 Dashboard + MCP（不启动训练）
guarftrain start

# 算力服务器端远程通信服务（供 DSH 插件 / 远程 Dashboard 消费）
guarftrain remote --port 8765 --auth <your-token>

# 训练同时开启 remote
guarftrain watch --remote --remote-auth <your-token> -- python train.py
```

源码安装（等价于 PyPI 版）：

```bash
git clone https://github.com/Washington5533/Guid-traince.git
cd Guid-traince
pip install .
```

## 三、DSH 插件配置方案

插件在 DSH Web GUI 侧栏提供六标签页 Training Guardian 面板（概览 / 设备 / 异常 / 决策 / 架构 / 历史），通过 SSE + REST 消费 `guarftrain remote` 服务。

### 1. 安装（二选一）

```bash
# 方式 A：官方 dsh 命令（推荐）
dsh plugin add @rrrelink/dsh-client-ui-training-guardian --profile web

# 方式 B：本项目自带 CLI（无 dsh 命令时）
python scripts/dsh_plugin_cli.py add @rrrelink/dsh-client-ui-training-guardian --profile web
python scripts/dsh_plugin_cli.py validate @rrrelink/dsh-client-ui-training-guardian --profile web
```

开发安装（从源码构建）：

```bash
git clone https://github.com/Washington5533/Guid-traince.git
cd Guid-traince/dsh-plugin/dsh-client-ui-training-guardian
pnpm install && pnpm build
dsh plugin add . --profile web
```

### 2. 训练机侧启动数据源

```bash
guarftrain remote --port 8765 --auth <your-token>   # 默认端口 8765，监听 0.0.0.0；--auth 可选
guarftrain watch -- python train.py                 # 或合并：watch --remote --remote-auth <token>
```

### 3. 面板连接配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| 服务器地址 | `http://localhost:8765` | 填 `RemoteServer` 所在训练机的地址 |
| 鉴权 Token | *（空）* | 与 `guarftrain remote --auth` 保持一致 |

跨机器访问时确保训练机防火墙放行对应端口；面板提示 `auth failed` 即 Token 不一致，提示"无法连接"即地址/端口/防火墙问题。完整说明见[插件说明书](../dsh-plugin/dsh-client-ui-training-guardian/README.zh.md)。

## 四、外部 Agent 接入方案（MCP）

`guarftrain[mcp]` 提供 36 个工具（25 只读 + 11 写），写操作需 `GUARDIAN_MCP_TOKEN` 鉴权。

### 1. 启动 MCP 服务（三选一）

```bash
guarftrain watch --with-mcp -- python train.py        # A. 训练时同启（实时控制）
guarftrain serve --transport stdio                    # B. 独立 stdio 进程（本地 Agent 直连）
guarftrain start                                      # C. Dashboard + MCP 一键启动
```

### 2. Claude Code / Cursor 等 stdio 客户端

在 `~/.claude/mcp.json`（或项目 `.claude/settings.local.json`）添加：

```json
{
  "mcpServers": {
    "guardian": {
      "command": "guarftrain",
      "args": ["serve", "--transport", "stdio"],
      "cwd": "/path/to/your-project"
    }
  }
}
```

### 3. 远程 SSE 接入（跨机器）

```bash
# 服务器端
guarftrain serve --transport sse --port 8766          # HTTP 端口默认 8766

# 本地端口转发
ssh -L 8766:127.0.0.1:8766 user@your-server
```

客户端配置：

```json
{
  "mcpServers": {
    "guardian-remote": {
      "type": "http",
      "url": "http://127.0.0.1:8766/sse"
    }
  }
}
```

### 4. 写工具鉴权

```bash
export GUARDIAN_MCP_TOKEN=your-secret   # 服务端设置；外部 Agent 调用 11 个写工具时携带同一 token
```

只读工具无需鉴权；写工具另受训练阶段门控（如训练结束后不可再触发干预）。委托模式：外部 Agent 连接时内置 Agent 自动进入 provisional 模式，决策可被覆盖。

## 五、默认端口速查

| 服务 | 默认端口 | 修改方式 |
|------|---------|---------|
| Dashboard | 8765 | `guarftrain start --dash-port` / 配置 `dashboard.port` |
| Remote 通信 | 8765 | `guarftrain remote --port` / `watch --remote-port` |
| MCP HTTP/SSE | 8766 | `guarftrain serve --port` / 配置 `mcp.tcp_port` |

> Dashboard 与 Remote 默认端口相同（8765），同机同时启用时请错开。

## 相关链接

- 仓库：<https://github.com/Washington5533/Guid-traince> · [README](../README.md) · [部署手册](DEPLOYMENT.md) · [MCP API 参考](MCP_API_REFERENCE.md) · [MCP 五分钟接入](MCP_QUICKSTART.md)
