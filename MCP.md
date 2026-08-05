# Training Guardian Agent — MCP 说明书

## 1. 概述

Guardian 通过 MCP (Model Context Protocol) 把全部观测与操作能力暴露为标准工具，供 Claude Code、OpenClaw 等外部 agent 客户端接入。你可以在本地 IDE 里直接查看远程服务器上正在跑的训练状态、给守护 agent 下达指令。

**关键设计原则：**

- **读操作不受限**：任何连接的客户端都可以查看训练状态
- **写操作默认关闭**：需要显式配置开启 + 口令鉴权
- **MCP 故障不影响训练**：MCP 奔溃/断连只影响外部接入，训练和守护照常运行
- **可事后补挂**：训练已经在跑的情况下，随时可以启动 MCP 接入，不需要重启训练进程

## 2. 安装

```bash
# MCP 依赖是独立叠加层，任何时间都可以安装
pip install -r requirements-mcp.txt
```

验证：

```bash
python -c "import mcp; print('MCP OK')"
```

## 3. 启动方式

### 3.1 方式一：同进程后台线程（推荐）

```bash
python run.py watch --with-mcp -- python train.py --epochs 20
```

MCP server 在 guardian 进程内的独立线程运行，**直接共享内存中的 monitor/watchdog 状态**——实时性最好，不需要读盘。

### 3.2 方式二：独立进程（事后补挂）

```bash
# 训练已经在跑（可能已经跑了几小时）
python run.py serve --transport stdio
```

独立进程定期读盘刷新状态（默认每 5 秒），可以对着一个已经在跑的 `watch` 补挂，**不需要重启训练**。

### 3.3 方式三：TCP 端口

```bash
python run.py serve --transport tcp
```

MCP server 监听 `127.0.0.1:8765`（默认端口），通过 SSH 隧道从本地连接远程服务器：

```bash
# 在本地机器上建立 SSH 隧道
ssh -L 8765:127.0.0.1:8765 user@your-training-server

# 隧道建立后，本地 Claude Code 连接 localhost:8765
```

## 4. Claude Code 接入

### 项目级配置（随仓库走，团队共享）

在项目根目录创建 `.mcp.json`：

```json
{
  "mcpServers": {
    "guardian": {
      "command": "python",
      "args": ["-m", "guardian.mcp_server", "--transport", "stdio"]
    }
  }
}
```

或在 `run.py` 同级目录：

```json
{
  "mcpServers": {
    "guardian": {
      "command": "python",
      "args": ["run.py", "serve", "--transport", "stdio"]
    }
  }
}
```

### CLI 注册

```bash
claude mcp add guardian -- python run.py serve --transport stdio
```

### 验证连接

在 Claude Code 对话中：

```
请列出 guardian 的所有可用工具
```

或直接查询：

```
查看当前训练状态
```

## 5. 工具清单

### 5.1 只读工具（始终可用，无需鉴权）

| 工具名 | 功能 | 返回内容 |
|--------|------|----------|
| `get_training_status` | 查看当前训练状态 | 最新 epoch/step、loss/accuracy、GPU 状态 |
| `get_metrics_history` | 查看指标历史 | 完整时间序列（支持分页） |
| `list_checkpoints` | 列出所有 checkpoint | 每个 cp 的路径、指标、是否 best/top_k |
| `compare_checkpoints` | 对比两个 checkpoint | 指标差值 |
| `get_anomaly_history` | 查看异常事件 | 全部异常 + 应对来源（agent/rule_default） |
| `get_recovery_history` | 查看恢复记录 | 全部重启事件 + trigger + 作废 epoch 数 |
| `get_summary` | 查看训练摘要 | 结构化摘要 + AI 解读 |
| `get_agent_decision_log` | 查看 Agent 决策日志 | 全部 LLM 调用记录 + source/延迟/降级原因 |
| `get_contract_status` | 查看契约状态 | 契约四项各自的开启/降级状态 |
| `list_contract_proposals` | 查看契约提议 | Agent 提议记录（pending/approved/rejected） |

### 5.2 受限写工具（默认关闭，需鉴权）

| 工具名 | 功能 | 风险等级 | 代价说明 |
|--------|------|----------|----------|
| `trigger_recovery` | 手动触发恢复重启 | ⚠️ 高 | 回滚到最近 ckpt，作废其后算力 |
| `restart_with_params` | 调整参数后重启 | ⚠️ 高 | 同上；参数越界会被拒绝 |
| `stop_training` | 停止训练 | ⚠️ 高 | 训练中止，需人工重新拉起 |
| `trigger_full_validate` | 触发完整校验 | 🟡 中 | 占用算力，可能与训练争 GPU |
| `approve_contract_proposal` | 批准契约扩展提议 | 🟡 中 | 扩大 agent 后续可自主选择的空间 |
| `reject_contract_proposal` | 拒绝契约扩展提议 | 🟢 低 | 无副作用 |

## 6. 写工具鉴权

### 启用写工具

在 `configs/guardian.yaml` 中：

```yaml
mcp:
  enabled: true
  enable_write_tools: true       # 开启写工具（默认关闭）
  write_token_env: GUARDIAN_MCP_TOKEN  # 口令从环境变量读取
```

设置口令：

```bash
export GUARDIAN_MCP_TOKEN=your-secret-token-here
```

### 调用示例

在 Claude Code 对话中调用写工具时，传入 token：

```
请用 token "your-secret-token-here" 调用 restart_with_params，
把 batch_size 改为 32
```

未授权或 token 错误时，写工具返回明确的鉴权失败错误（不是静默失败）。

## 7. 工具详细说明

### 7.1 get_metrics_history — 分页拉取

训练几十万 step 时，全量返回会塞爆 agent 上下文。默认只返回最近 200 条 + 聚合统计。

```
# 查看最近 100 条
get_metrics_history(limit=100)

# 分页拉取完整历史
get_metrics_history(limit=100, cursor=0)   # 最近 100 条
get_metrics_history(limit=100, cursor=100) # 再往前 100 条
```

返回格式：

```json
{
  "total": 50000,
  "returned": 100,
  "cursor": 0,
  "limit": 100,
  "aggregates": {
    "loss_min": 0.023,
    "loss_max": 2.15,
    "loss_avg": 0.341
  },
  "metrics": [
    {"step": 49900, "loss": 0.025, "val_acc": 0.987, "lr": 0.0001},
    ...
  ]
}
```

### 7.2 restart_with_params — 参数重启

**重要：此操作会 kill 当前训练子进程并从 checkpoint 重启，作废最近 checkpoint 之后的所有算力。**

```json
{
  "action": "reduce_batch",
  "param": 0.5,
  "request_id": "unique-id-123"
}
```

支持的 action：
- `reduce_batch(ratio)`: 减半 batch_size（如 64 → 32）
- `restart_with_lower_lr(ratio)`: 降低学习率（如 0.0001 → 0.0005）
- `enable_grad_accum(steps)`: 梯度累积（成对调整 batch_size + grad_accum_steps）

参数必须在 `contract.yaml` 的 `adjustable_paths` 白名单范围内，越界会被拒绝。

`request_id` 用于幂等保证：相同 `request_id` 在 5 分钟内重复调用只执行一次。

### 7.3 approve_contract_proposal — 审核提议

Agent 在注册表中找不到匹配当前任务的条目时，会生成一条提议（不生效）。人工审核通过后才写入正式注册表。

```json
{
  "proposal_id": "abc123def456"
}
```

建议审核时连 `evidence` 字段一起看：

```
先列出 pending 的提议
list_contract_proposals(status="pending")

查看某条提议的详细 evidence
（返回中包含 evidence 字段，说明 agent 为什么建议这个条目）
```

### 7.4 访问日志

所有写工具调用（无论成功失败）都会记录到 `logs/mcp_access_log.json`：

```json
{
  "tool": "restart_with_params",
  "client_id": "claude-code@user-machine",
  "params": {"action": "reduce_batch", "param": 0.5},
  "success": true,
  "deduplicated": false,
  "timestamp": 1754303422
}
```

## 8. 安全说明

| 安全措施 | 说明 |
|----------|------|
| 默认只读 | 写工具默认关闭，需显式配置 `enable_write_tools: true` |
| 口令鉴权 | 写工具调用必须携带 token，从环境变量读取（不写入配置文件） |
| 仅绑本地 | TCP 模式默认只绑定 `127.0.0.1`，远程访问需走 SSH 隧道 |
| 访问日志 | 所有写调用记录到 `mcp_access_log.json`（不记录 token 本身） |
| 幂等保证 | 写工具支持 `request_id`，5 分钟内重复调用不重复执行 |
| 非阻塞 | MCP 奔溃/断连只影响外部接入，训练与守护照常运行 |

## 9. 远程训练机场景

```
┌─ 远程训练服务器 ──────────────────────────┐
│                                            │
│  guardian watch -- python train.py         │
│    └─ MCP server (tcp://127.0.0.1:8765)   │
│                                            │
└────────────────────────────────────────────┘
         │
         │ SSH 隧道: ssh -L 8765:127.0.0.1:8765 user@server
         │
┌─ 本地机器 ─────────────────────────────────┐
│                                            │
│  Claude Code                               │
│    └─ MCP client → localhost:8765          │
│                                            │
│  对话: "看一下训练到哪了"                     │
│    → get_training_status                   │
│    → "epoch 47/100, loss=0.12, GPU 94%"    │
│                                            │
│  "GPU 温度有点高，降一下 lr 试试"            │
│    → restart_with_params(                  │
│         action="restart_with_lower_lr",    │
│         param=0.5,                         │
│         token="..."                        │
│       )                                    │
│    → "已重启训练，lr 减半至 5e-4"            │
│                                            │
└────────────────────────────────────────────┘
```

## 10. 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `mcp 包未安装` | 未装 requirements-mcp.txt | `pip install -r requirements-mcp.txt` |
| `写工具未启用` | `enable_write_tools: false` | 在 guardian.yaml 中设为 true |
| `鉴权失败` | token 不匹配或未设置 | 检查 `GUARDIAN_MCP_TOKEN` 环境变量 |
| `端口被占用` | tcp_port 已被占用 | 改端口或 kill 占用进程 |
| MCP 启动但 Claude Code 连不上 | transport 或端口配置不一致 | 检查 `.mcp.json` 的 args 和端口 |

MCP 启动失败时 guardian 只打印一条 warning，训练和守护照常运行——这是设计行为，不是 bug。
