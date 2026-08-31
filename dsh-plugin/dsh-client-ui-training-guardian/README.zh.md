# dsh-client-ui-training-guardian

DSH Web GUI 插件，在侧边栏增加 **Training Guardian** 面板：实时训练指标、
GPU 设备状态、异常事件流、子 Agent 决策审批、模型架构分析，以及离线历史回放。

它是 [`guarftrain`](https://github.com/Washington5533/Guid-traince)（Training
Guardian Agent）的前端伴侣。面板消费 `guardian.remote.RemoteServer` 暴露的
SSE + REST 接口，因此**任何**由 guardian 服务守护的训练任务都能直接使用 ——
训练脚本本身零改动。

> 插件独立镜像仓库：<https://github.com/Washington5533/Guid-traince>

## 功能

六个标签页，一条连接：

| 标签页 | 内容 |
|--------|------|
| **概览** | Epoch、Step、Loss、Accuracy、学习率、训练状态，以及实时双轴 loss/accuracy 曲线 |
| **设备** | 每张 GPU 的利用率、温度、显存、功耗（迷你进度条）；纯 CPU 主机自动降级提示 |
| **异常** | 异常事件实时流，带严重程度标签（loss 尖刺 / NaN / OOM 风险 / 停滞 …） |
| **决策** | 子 Agent 提出的动作（调整学习率、早停、回滚 checkpoint …），支持行内**批准 / 驳回** |
| **架构** | guardian 提供的模型架构分析：参数量、FLOPs、瓶颈层 |
| **历史** | 浏览历史训练会话 —— 在线时读取服务器，离线时回退到浏览器 `localStorage` 缓存 |

其他能力：

- **会话头部按钮** — 会话头部提供 Training Guardian 快捷按钮，一键打开/停靠面板。
- **三态连接** — `空闲 → 连接中 → 已连接`，连接失败给出明确提示（鉴权失败 /
  服务器不可达 / 重试耗尽），并自动退避重连。
- **离线历史持久化** — 会话期间收到的指标会缓存到 `localStorage`
  （每会话最多 2000 个点，最多保留 50 个会话），刷新页面或服务器离线后
  历史标签页依然可用。
- **国际化** — 支持英文与简体中文。
- **Skill 集成** — 注册 `training-guardian` skill，当你询问训练状态、GPU、
  loss 曲线、异常或决策时，Agent 可直接打开面板。

## 依赖要求

- DSH Web GUI `>= 0.1.1-rc.1`
- Node.js `^22.19.0 || >=24.0.0`
- 一个正在运行的 `guarftrain` guardian 服务，且已启用远程 API（见下文）

## 安装

### 方式 A — 从社区 registry 安装（推荐）

```bash
dsh plugin add @rrrelink/dsh-client-ui-training-guardian --profile web
```

或使用 guarftrain 自带的 `dsh-plugin` 辅助工具：

```bash
python scripts/dsh_plugin_cli.py add @rrrelink/dsh-client-ui-training-guardian --profile web
python scripts/dsh_plugin_cli.py validate @rrrelink/dsh-client-ui-training-guardian --profile web
```

### 方式 B — 从源码安装（开发用）

```bash
git clone https://github.com/Washington5533/Guid-traince.git
cd Guid-traince/dsh-plugin/dsh-client-ui-training-guardian
pnpm install
pnpm build                 # tsc --noEmit && tsdown
dsh plugin add . --profile web   # 或把构建产物复制到 ~/.dsh/profiles/web/node_modules/
```

重启 / 刷新 DSH Web GUI。插件通过 `cordis.patch.yml` 自动发现，注入两个槽位：

- `conversation.session.header.actions` — 面板开关按钮
- `settings.plugin.item` — 设置卡片

## 启动 guardian 后端

插件只负责渲染数据，数据来自 guardian 远程服务。在训练机上执行：

```bash
pip install guarftrain          # 或在仓库根目录 pip install .

# 启动远程 API 服务（默认端口 8765，监听 0.0.0.0）
guarftrain remote --port 8765 --auth <your-token>     # --auth 可选

# 另开一个终端，让训练在 guardian 守护下运行（零代码改动）
guarftrain watch -- python train.py --epochs 50
```

事件会以 JSONL 形式持久化到 guardian 日志目录，历史 API（`/api/history/...`）
即基于这些文件提供回放能力。

## 配置

打开 **设置 → 插件 → Training Guardian**，填写：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| **服务器地址** | `http://localhost:8765` | `RemoteServer` 所在训练机的地址 |
| **鉴权 Token** | *(空)* | 服务器启用 `guarftrain remote --auth` 时必须一致 |
| **训练会话 ID** | *(空)* | 留空则自动订阅第一个活跃训练会话 |
| **自动连接** | `开` | 面板打开时自动建立 SSE 连接 |
| **模型入口** | *(空)* | 如 `train:build_model`，供架构分析标签页使用 |
| **项目目录** | *(空)* | 训练项目根目录，用于解析架构分析的导入 |
| **Dashboard 地址** | *(空)* | 可选，指向独立 guardian dashboard 的链接 |

配置通过 DSH 设置体系持久化，修改后使用面板头部的 **连接 / 断开** 按钮
（重新）连接即可生效。

## 使用指南

1. **连接** — 配置服务器地址（及 Token，如需要），从会话头部打开面板，
   等待状态徽标变绿。
2. **监控** — 概览页实时展示 `metrics` 事件；曲线图左轴（蓝）为 loss，
   右轴（绿）为 accuracy。
3. **关注异常** — 异常页按严重程度汇总告警，配合服务端规则引擎使用。
4. **审批决策** — 子 Agent 在 `supervised` 自治模式下提出动作时，决策页会
   出现审批卡片，点击 **批准** 或 **驳回**。审批为会话级作用域。
5. **查看历史** — 历史页列出过往会话：在线时读取服务器
   （`GET /api/history/sessions`），离线时回退到浏览器缓存；选中会话可查看
   摘要、指标趋势、异常与决策记录。

### 常见问题

| 现象 | 可能原因 |
|------|----------|
| 连接提示 `auth failed` | Token 与 `--auth` 不一致，或漏填 |
| 重试耗尽、`服务器不可达` | 服务未启动、URL/端口错误，或 8765 端口被防火墙拦截 |
| 面板显示无活跃会话 | 训练未用 `guarftrain watch` 启动，或配置了已结束的会话 ID |
| 离线时历史页为空 | 尚无缓存 —— 缓存仅在连接且收到指标时构建 |
| 架构页只显示估算值 | 模型前向不可用（dummy input 失败），guardian 按设计降级为静态估算 |

## 开发

```bash
pnpm install
pnpm typecheck   # tsc --noEmit
pnpm test        # vitest run（9 个测试套件）
pnpm build       # tsc --noEmit && tsdown
```

发布已自动化：GitHub 工作流 `plugin-publish.yml` 在版本 tag 上触发 ——
校验插件（28 项检查）、运行测试、构建、发布到 npm 并更新 `registry/plugins.json`。

## 许可证

MIT — 见 [LICENSE](LICENSE)。
