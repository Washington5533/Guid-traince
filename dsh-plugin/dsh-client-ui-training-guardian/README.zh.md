# dsh-client-ui-training-guardian

DSH Web GUI 插件，在侧边栏增加 **Training Guardian** 面板，实时展示训练指标、
GPU 设备状态、异常事件流，以及子 Agent 决策审批工作流。

它通过 SSE + REST 接口与 `guarftrain` 的 `guardian.remote.RemoteServer` 通信，
因此任何启动了 guardian 服务的训练任务都可以使用本插件。

## 功能

| 标签页 | 内容 |
|--------|------|
| **概览** | Epoch、Step、Loss、Accuracy、Learning Rate、状态 |
| **设备** | 每张 GPU 的利用率、温度、显存、功耗 + 迷你进度条 |
| **异常** | 异常事件实时流，带严重程度标签 |
| **决策** | 子 Agent 提出的动作，支持行内批准 / 驳回 |

**设置** 卡片（位于 DSH 插件设置区）允许用户配置 guardian 服务器地址、
鉴权 Token 和训练会话 ID，支持自动连接。

## 安装

```bash
# 在 DSH Web GUI 部署目录中：
cd dsh-web-ui/plugins
pnpm add @linxin666/dsh-client-ui-training-guardian
```

重启 DSH 主机即可，插件通过 `cordis.patch.yml` 自动注册。

## 配置

打开 **设置 → 插件 → Training Guardian**，填写：

- **Guardian 服务器地址** — 例如 `http://192.168.1.100:8765`（`RemoteServer` 地址）。
- **鉴权 Token** — 可选，必须与服务器 `--remote-auth` 参数一致。
- **训练会话 ID** — 可选，留空则自动订阅第一个活跃训练会话。
- **自动连接** — 面板打开时自动建立 SSE 连接。

## 依赖要求

- DSH Web GUI `>= 0.1.1-rc.1`
- Node.js `^22.19.0 || >=24.0.0`
- 运行中的 `guarftrain` guardian 服务，且已启用 `--remote`。

## 开发

```bash
pnpm install
pnpm typecheck   # tsc -b
pnpm build       # tsc -b && tsdown
```

## 许可证

Apache-2.0
