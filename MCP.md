# Training Guardian Agent — MCP 说明书

## 1. 概述

Guardian 通过 MCP (Model Context Protocol) 把全部观测与操作能力暴露为标准工具，供 Claude Code、OpenClaw 等外部 agent 客户端接入。28 个工具覆盖 Guardian 全部 16 个功能模块（cp_1 ~ cp_16）。

**关键设计原则：**

- **读操作不受限**：任何连接的客户端都可以查看训练状态（18 个只读工具）
- **写操作默认关闭**：需要显式配置开启 + 口令鉴权（10 个受限写工具）
- **MCP 故障不影响训练**：MCP 崩溃/断连只影响外部接入，训练和守护照常运行
- **可事后补挂**：训练已经在跑的情况下，随时可以启动 MCP 接入，不需要重启训练进程
- **双模式架构**：MCP 客户端连接时 guardian 内置 agent 自动让位，断开后自动恢复自主决策

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

### 3.1 方式一：同进程后台线程（推荐，实时性最好）

```bash
python run.py watch --with-mcp -- python train.py --epochs 20
```

MCP server 在 guardian 进程内的独立线程运行，**直接共享内存中的 monitor/watchdog 状态**，不需要读盘。

### 3.2 方式二：独立进程（事后补挂）

```bash
# 训练已经在跑（可能已经跑了几小时）
python run.py serve --transport stdio
```

独立进程定期读盘刷新状态（默认每 5 秒），可以对着一个已经在跑的 `watch` 补挂，**不需要重启训练**。

### 3.3 方式三：一键启动 Dashboard + MCP

```bash
python run.py start
```

同时启动 Dashboard 控制面板 (http://127.0.0.1:8765) 和 MCP SSE 端点 (http://127.0.0.1:8766/sse)，可选附带训练守护：

```bash
python run.py start -- python train.py --epochs 20
```

### 3.4 传输方式对比

| transport | 适用场景 | 说明 |
|-----------|----------|------|
| `stdio` | Claude Code 子进程接入 | 标准输入输出，Claude Code 自动管理进程生命周期 |
| `sse` | 远程 SSH 隧道接入 | SSE over HTTP，需 uvicorn |
| `http` | 远程 Streamable HTTP | 同 sse，走新协议 |
| `tcp` | 旧名别名 | 等同 `sse` |

## 4. Claude Code 接入

### 4.1 项目级配置（推荐）

在项目 `.claude/settings.local.json` 中：

```json
{
  "mcpServers": {
    "guardian": {
      "command": "/d/anaconda/envs/DL_gpu/python.exe",
      "args": ["run.py", "serve", "--transport", "stdio"],
      "cwd": "C:\\Users\\wst\\Desktop\\anytries\\guarftrain"
    }
  }
}
```

### 4.2 远程接入（SSH 隧道）

在远程服务器启动后，本机建立隧道：

```bash
ssh -L 8766:127.0.0.1:8766 user@your-training-server
```

然后在本机 `.claude/settings.local.json` 中：

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

### 4.3 用户级配置

在 `~/.claude/mcp.json` 中（全局生效）：

```json
{
  "mcpServers": {
    "guardian": {
      "command": "python",
      "args": ["run.py", "serve", "--transport", "stdio"],
      "cwd": "C:\\Users\\wst\\Desktop\\anytries\\guarftrain"
    }
  }
}
```

### 4.4 验证连接

在 Claude Code 对话中：

```
请用 get_training_status 查看当前训练状态
```

## 5. 工具清单（28 个）

### 5.1 只读工具 — 训练监控（6 个）

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `get_training_status` | 当前训练状态：epoch/step、loss/accuracy、GPU 状态 | 无 |
| `get_metrics_history` | 指标时间序列（分页+聚合统计） | `limit`(默认200), `cursor`(偏移量) |
| `get_anomaly_history` | 全部异常事件 + 应对来源 | 无 |
| `get_recovery_history` | 全部重启记录 + 作废 epoch + 参数变更 | 无 |
| `get_agent_decision_log` | agent 全部 LLM 调用记录 + source/延迟/降级原因 | 无 |
| `get_summary` | 训练摘要（结构化 + AI 解读） | 无 |

### 5.2 只读工具 — 实验查询（3 个）

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `list_experiments` | 列出所有历史实验摘要 | `limit`(默认50) |
| `query_experiment` | 自然语言查询实验 | `question`(必填) |
| `compare_experiments` | 对比两个实验的参数、指标、异常 | `id_a`, `id_b`(必填) |

### 5.3 只读工具 — Checkpoint 管理（2 个）

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `list_checkpoints` | 列出所有 checkpoint + best/top_k 标记 | `metric`(默认val/accuracy) |
| `compare_checkpoints` | 对比两个 checkpoint 的指标差异 | `epoch_a`, `epoch_b`(必填) |

### 5.4 只读工具 — 模型与配置（5 个）

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `get_model_structure` | 模型结构 JSON（节点/边/FLOPs/参数量） | `model_entry`(如 train:build_model) |
| `get_guardian_mode` | 当前模式：standalone / mcp_delegated | 无 |
| `get_gallery_config` | 图片筛选策略配置 | 无 |
| `get_import_format` | Guardian 导入格式规范（JSON Schema） | 无 |
| `inspect_source` | 采样外部数据文件前 N 行 | `file_path`(必填), `lines`(默认20，上限100) |

### 5.5 只读工具 — 契约（2 个）

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `get_contract_status` | 契约四项各自的开启/降级状态 | 无 |
| `list_contract_proposals` | agent 提议记录（pending/approved/rejected） | `status`(筛选) |

### 5.6 受限写工具 — 训练控制（4 个，训练中可用）

| 工具名 | 功能 | 风险 | 关键参数 |
|--------|------|------|----------|
| `trigger_recovery` | 手动触发恢复重启，回滚到最近 ckpt | ⚠️ 高 | `request_id`(幂等键) |
| `restart_with_params` | 调整参数后重启（受白名单约束） | ⚠️ 高 | `action`, `param`, `request_id` |
| `stop_training` | 停止训练子进程并终止看护 | ⚠️ 高 | `request_id` |
| `trigger_full_validate` | 对指定 ckpt 跑完整校验 | 🟡 中 | `epoch`(必填), `request_id` |

### 5.7 受限写工具 — 契约管理（2 个，训练中可用）

| 工具名 | 功能 | 风险 | 关键参数 |
|--------|------|------|----------|
| `approve_contract_proposal` | 批准 agent 的契约扩展提议 | 🟡 中 | `proposal_id`(必填), `request_id` |
| `reject_contract_proposal` | 拒绝并归档契约提议 | 🟢 低 | `proposal_id`(必填), `request_id` |

### 5.8 受限写工具 — 训练后功能（4 个，仅训练结束后可用）

| 工具名 | 功能 | 关键参数 |
|--------|------|----------|
| `run_visualization` | 生成模型管线可视化 HTML（交互式 D3.js） | `model_entry`(必填), `output_path`, `request_id` |
| `set_gallery_config` | 更新图片筛选策略配置，触发重新筛选 | `strategies`(必填), `checkpoint_epoch`(必填), `data_source`(必填), `request_id` |
| `run_inference` | 用指定 ckpt 跑推理（分类/检测/分割） | `checkpoint_epoch`(必填), `task_type`(必填), `inputs`(必填), `request_id` |
| `submit_import` | 提交外部训练数据（WandB/TensorBoard/CSV） | `meta`(必填，含name), `metrics_path`或`metrics`(二选一), `request_id` |

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

### 训练阶段保护

`run_visualization`、`set_gallery_config`、`run_inference` 三个工具在训练进行中调用会返回：

```json
{"error": "工具 'run_visualization' 仅在训练结束后可用。当前训练仍在进行中。"}
```

## 7. 工具详细说明

### 7.1 get_metrics_history — 分页拉取

```json
// 最近 200 条 + 聚合统计（默认）
get_metrics_history()

// 自定义条数和偏移
get_metrics_history(limit=100, cursor=0)   // 最近 100 条
get_metrics_history(limit=100, cursor=100) // 再往前 100 条
```

返回：

```json
{
  "total": 50000,
  "returned": 200,
  "cursor": 0,
  "limit": 200,
  "aggregates": {"loss_min": 0.023, "loss_max": 2.15, "loss_avg": 0.341},
  "metrics": [{"step": 49900, "loss": 0.025, "val_acc": 0.987}, ...]
}
```

### 7.2 restart_with_params — 参数重启

⚠️ **此操作会 kill 当前训练子进程并从 checkpoint 重启，作废最近 checkpoint 之后的所有算力。**

```json
{
  "action": "reduce_batch",
  "param": 0.5,
  "request_id": "unique-id-for-idempotency"
}
```

支持的 action（受 `contract.yaml` 白名单约束）：

| action | 效果 | param |
|--------|------|-------|
| `reduce_batch` | 减半 batch_size | ratio（如 0.5） |
| `restart_with_lower_lr` | 降低学习率 | ratio（如 0.5） |
| `enable_grad_accum` | 梯度累积（成对调整 batch_size + grad_accum_steps） | steps（整数） |

### 7.3 query_experiment — 自然语言查询

```json
{
  "question": "上次 mAP 最高的那次实验，lr 和 batch_size 分别是多少？"
}
```

支持的查询类型：最高/最低指标、参数查询、实验列表、时间范围等。

### 7.4 submit_import — 导入外部训练数据

方式 A：大文件路径

```json
{
  "meta": {"name": "WandB实验", "source": "wandb"},
  "metrics_path": "./logs/wandb_export.jsonl"
}
```

方式 B：直接传内容（小数据）

```json
{
  "meta": {"name": "手动记录", "source": "csv"},
  "metrics": [
    {"step": 0, "loss": 2.1, "acc": 0.12},
    {"step": 1, "loss": 1.8, "acc": 0.23}
  ]
}
```

校验规则：
- `meta.name` 必须存在（字符串）
- `metrics` 每条为 dict，至少含一个数值字段
- `metrics_path` 存在且为合法 JSONL
- 单次上限 100,000 条

## 8. 幂等保证

所有写工具支持 `request_id` 参数。相同 `request_id` 在 5 分钟（默认）内重复调用返回首次结果，不重复执行：

```json
// 第一次调用：执行
restart_with_params(action="reduce_batch", param=0.5, request_id="abc-123")
// → {"status": "requested", "action": "reduce_batch", "param": 0.5}

// 5 分钟内重复调用：返回首次结果
restart_with_params(action="reduce_batch", param=0.5, request_id="abc-123")
// → {"deduplicated": true, "status": "requested", "action": "reduce_batch", "param": 0.5}
```

## 9. 访问日志

所有工具调用（无论成功失败）记录到 `logs/mcp_access_log.json`：

```json
{
  "tool": "restart_with_params",
  "client_id": null,
  "params": {"action": "reduce_batch", "param": 0.5, "request_id": "abc-123"},
  "success": true,
  "deduplicated": false,
  "timestamp": 1754303422.123
}
```

**注意：`_token` 不记入日志。**

## 10. 安全说明

| 安全措施 | 说明 |
|----------|------|
| 默认只读 | 写工具默认关闭，需显式配置 `enable_write_tools: true` |
| 口令鉴权 | 写工具调用必须携带 token，从环境变量读取（不写入配置文件） |
| 阶段保护 | 训练后专用工具在训练中调用返回错误 |
| 仅绑本地 | TCP/SSE 模式默认只绑定 `127.0.0.1`，远程访问需走 SSH 隧道 |
| 参数白名单 | `restart_with_params` 参数必须在 `contract.yaml` 的 `adjustable_paths` 范围内 |
| 访问日志 | 所有调用记录到 `mcp_access_log.json`（不记录 token 本身） |
| 幂等保证 | 写工具 5 分钟内相同 request_id 不重复执行 |
| 非阻塞 | MCP 崩溃/断连只影响外部接入，训练与守护照常运行 |

## 11. 远程训练机场景

```
┌─ 远程训练服务器 ──────────────────────────────────┐
│                                                    │
│  python run.py serve --transport sse --port 8766   │
│    → MCP server 监听 127.0.0.1:8766               │
│                                                    │
└────────────────────────────────────────────────────┘
         │
         │ SSH 隧道: ssh -L 8766:127.0.0.1:8766 user@server
         │
┌─ 本地机器 ─────────────────────────────────────────┐
│                                                    │
│  Claude Code                                       │
│    └─ MCP client → localhost:8766/sse             │
│                                                    │
│  "查看当前训练状态"                                  │
│    → get_training_status                           │
│    → "epoch 47/100, loss=0.12, GPU 94%"            │
│                                                    │
│  "GPU 温度有点高，降 lr 试试"                       │
│    → restart_with_params(                          │
│         action="restart_with_lower_lr",            │
│         param=0.5,                                 │
│         request_id="..."                           │
│       )                                            │
│    → "已重启训练，lr 减半至 5e-4"                   │
│                                                    │
└────────────────────────────────────────────────────┘
```

## 12. 双模式架构

```
┌─ Standalone ───────────────────────────────────┐
│ guardian AgentAdvisor 自主决策                   │
│ 训练中：预设动作集，失败回退规则默认               │
│ 训练后：创造性策略（需用户确认）                   │
├─ MCP Delegated ────────────────────────────────┤
│ 外部 Claude Code 决策，guardian agent 让位       │
│ guardian 角色：数据提供者 + 安全执行器            │
│ Claude Code 断开 → 自动恢复 standalone          │
└────────────────────────────────────────────────┘
```

模式切换过程：
1. MCP 客户端连接 → `on_client_connect()` → `advisor.set_delegated(True)`
2. MCP 客户端断开 → `on_client_disconnect()` → `advisor.set_delegated(False)` → 恢复自主决策

可通过 `get_guardian_mode` 工具随时查询当前模式。

## 13. 故障排查

| 症状 | 原因 | 解决 |
|------|------|------|
| `mcp 包未安装` | 未装 requirements-mcp.txt | `pip install -r requirements-mcp.txt` |
| `写工具未启用` | `enable_write_tools: false` | 在 guardian.yaml 中设为 true |
| `鉴权失败` | token 不匹配或未设置 | 检查 `GUARDIAN_MCP_TOKEN` 环境变量 |
| `仅训练结束后可用` | 训练仍在进行中 | 等训练结束，或用训练中可用的写工具 |
| `端口被占用` | port 已被占用 | 改端口或 kill 占用进程 |
| Claude Code 连不上 | transport 或端口配置不一致 | 检查 mcpServers 配置的 args 和端口 |
| SSE 模式报 uvicorn 缺失 | 未装 dashboard 依赖 | `pip install -r requirements-dashboard.txt` |

MCP 启动失败时 guardian 只打印一条 warning，训练和守护照常运行——这是设计行为，不是 bug。
