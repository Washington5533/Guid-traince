# Guardian MCP — API 参考

> 完整工具参数说明 · 35 个工具（24 只读 + 11 写）· 版本 0.2.0

---

## 只读工具（24 个）

### get_training_status

返回当前训练状态快照。

```
输入：无
```

**返回示例：**

```json
{
  "latest_metrics": {"epoch": 47, "loss": 0.12, "val_acc": 0.94, "lr": 0.0001},
  "total_records": 4700,
  "latest_gpu": {"utilization": 94, "memory_used_mb": 8192, "temp_c": 72},
  "anomaly_count": 3
}
```

---

### get_metrics_history

分页拉取指标时间序列。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `limit` | integer | 否 | 200 | 返回条数上限 |
| `cursor` | integer | 否 | 0 | 偏移量，0=最新 |

**返回：**

```json
{
  "total": 50000,
  "returned": 200,
  "cursor": 0,
  "limit": 200,
  "aggregates": {"loss_min": 0.023, "loss_max": 2.150, "loss_avg": 0.341},
  "metrics": [...]
}
```

---

### list_checkpoints

列出所有 checkpoint。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `metric` | string | 否 | `val/accuracy` | 排序指标 |

**返回：**

```json
{
  "total": 50,
  "latest": {"epoch": 50, "path": "checkpoints/cp_50"},
  "best": {"epoch": 47, "metric_val": 0.94, "metric": "val/accuracy"},
  "checkpoints": [
    {"epoch": 50, "path": "checkpoints/cp_50", "best": false, "top_k": true,
     "metrics": {"loss": 0.08, "val_acc": 0.92}}
  ]
}
```

---

### compare_checkpoints

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `epoch_a` | integer | **是** | 第一个 checkpoint 的 epoch |
| `epoch_b` | integer | **是** | 第二个 checkpoint 的 epoch |

**返回：**

```json
{
  "epoch_a": 47, "epoch_b": 50,
  "diffs": {
    "val_acc": {"cp_a": 0.94, "cp_b": 0.92, "delta": -0.02},
    "loss": {"cp_a": 0.10, "cp_b": 0.08, "delta": -0.02}
  }
}
```

---

### get_anomaly_history

全部异常事件 + 每次事件的应对来源。

```
输入：无
```

**返回：** `AnomalyEvent[]`

```json
[{
  "type": "loss_spike",
  "detail": "loss 从 0.15 升至 0.39 (+157%)",
  "level": "warning",
  "step": 1234,
  "epoch": 12,
  "metrics": {"loss": 0.39, "val_acc": 0.72},
  "response": {"action": "restart_with_lower_lr", "source": "agent"}
}]
```

---

### get_recovery_history

全部重启记录。

```
输入：无
```

**返回：** `RestartRecord[]`

```json
[{
  "trigger": "crash",
  "reason": "OOM (exit_code=137)",
  "resumed_from": {"epoch": 10, "path": "checkpoints/cp_10"},
  "cmd_before": "python train.py --batch_size 64 --epochs 20",
  "cmd_after": "python train.py --batch_size 32 --epochs 20 --resume --ckpt checkpoints/cp_10",
  "wasted_epochs": 2,
  "applied": {"dataloader.batch_size": 32},
  "skipped": [],
  "timestamp": 1754303422
}]
```

---

### get_summary

已生成的训练摘要（结构化 + AI 解读）。

```
输入：无
```

**返回：**

```json
{
  "experiment_id": "...",
  "status": "completed",
  "duration": {"start": 1754300000, "end": 1754310000, "total_seconds": 10000},
  "training": {"total_epochs": 50, "best_accuracy": 0.94, "final_loss": 0.08},
  "anomaly_events": [...],
  "restarts": [...],
  "checkpoints": {"total": 50, "latest_epoch": 50, "best_epoch": 47},
  "resources": {"gpu_util_avg": 0.88, "gpu_mem_peak_gb": 10.2, "gpu_hours": 2.5},
  "lr_schedule": [{"step": 0, "lr": 0.001}, {"step": 10000, "lr": 0.0005}],
  "ai_narrative": "训练整体稳定，第12个epoch出现loss突增..."
}
```

---

### get_agent_decision_log

全部 agent LLM 调用记录。

```
输入：无
```

**返回：**

```json
[{
  "decision_point": "monitor_response",
  "action": "restart_with_lower_lr",
  "source": "agent",
  "latency_ms": 823,
  "context_summary": "loss_spike: loss +157% at step 1234"
}]
```

`source` 可能值：`agent` | `rule_default` | `disabled` | `timeout` | `error` | `invalid_output`

---

### get_contract_status

契约四项各自的开/关状态。

```
输入：无
```

**返回：**

```json
{
  "capabilities": {
    "resumable": true,
    "checkpoint_schema": true,
    "metrics_channel": true,
    "buildable_entry": false
  },
  "missing": ["buildable_entry"]
}
```

---

### list_contract_proposals

agent 的契约扩展提议记录。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `status` | string | 否 | 筛选：`pending` / `approved` / `rejected` |

---

### list_experiments

所有历史实验摘要。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `limit` | integer | 否 | 50 | 返回条数上限 |

**返回：**

```json
{
  "total": 12,
  "experiments": [
    {"experiment_id": "...", "status": "completed", "best_metric": 0.94, "timestamp": "..."}
  ]
}
```

---

### query_experiment

自然语言查询实验记录。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | **是** | 自然语言问题 |

**示例：**

```json
{"question": "上次 mAP 最高的那次实验，lr 和 batch_size 分别是多少？"}
```

---

### compare_experiments

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `id_a` | string | **是** | 第一个实验 ID |
| `id_b` | string | **是** | 第二个实验 ID |

---

### get_model_structure

返回模型结构 JSON（节点/边/参数量/FLOPs），供外部 agent 分析瓶颈。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_entry` | string | 否 | 如 `train:build_model`，未填从 contract 读取 |

**返回：**

```json
{
  "nodes": [
    {"id": "conv1", "type": "Conv2d", "params": 9408, "flops": 235200,
     "input_shape": [1,3,224,224], "output_shape": [1,64,112,112]}
  ],
  "edges": [{"from": "conv1", "to": "bn1"}],
  "total_params": 25678901,
  "total_flops": 4200000000,
  "layer_stats": [
    {"name": "conv1", "type": "Conv2d", "params_pct": 0.04, "severity": "normal"}
  ]
}
```

---

### get_guardian_mode

```
输入：无
```

**返回：**

```json
{"mode": "standalone", "description": "agent 自主决策中"}
// 或
{"mode": "mcp_delegated", "description": "外部 Claude Code 决策中，内置 agent 已让位"}
```

---

### get_gallery_config

当前图片筛选策略配置（如已生成）。

```
输入：无
```

---

### get_import_format

Guardian 导入格式规范（JSON Schema）。

```
输入：无
```

**返回：**

```json
{
  "meta": {"required": ["name"], "optional": ["command", "source", "project_dir"]},
  "metrics": {
    "format": "JSONL（每行一个 JSON 对象）",
    "required": "至少含一个数值类型的 key",
    "example": {"step": 0, "loss": 2.1, "acc": 0.12, "lr": 0.001}
  },
  "submit_import": {
    "校验规则": ["meta 必须含 name", "metrics 每条至少含一个数值字段", "单次上限 100000 条"]
  }
}
```

---

### inspect_source

采样外部数据文件前 N 行，用于导入前格式探测。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file_path` | string | **是** | — | 外部数据文件路径 |
| `lines` | integer | 否 | 20 | 采样行数（上限 100） |

**返回：**

```json
{
  "file_path": "/path/to/wandb_export.csv",
  "total_lines_sampled": 20,
  "lines": ["step,loss,acc", "0,2.1,0.12", "1,1.8,0.23", ...],
  "format_hints": ["检测到 CSV 格式（逗号分隔）"]
}
```

---

### get_training_log

读取训练日志文件的尾部内容，用于排查训练错误、查看崩溃前的日志、检查输出。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `lines` | integer | 否 | 100 | 返回行数（上限 1000） |
| `offset` | integer | 否 | 0 | 偏移量（从末尾倒数），0=最新 |
| `grep` | string | 否 | — | 过滤关键字（可选），如 `Error`、`epoch` |

**返回：**

```json
{
  "log_file": "logs/train.log",
  "total_lines": 12034,
  "returned": 100,
  "lines": ["...", "..."],
  "offset": 0,
  "grep": null
}
```

---

### get_post_training_checklist

训练结束后的待办清单：哪些 checkpoint 可用、可以生成什么（可视化/推理/图库/摘要）、每个操作的推荐命令和参数。训练结束后应优先调用此工具，然后按清单逐项执行。

```
输入：无
```

**返回：**

```json
{
  "training_active": false,
  "total_items": 6,
  "available_now": 5,
  "checklist": [
    {"category": "checkpoint", "title": "最佳 Checkpoint 分析", "available": true,
     "detail": "共 50 个 checkpoint，最佳 epoch=47",
     "suggested_tool": "list_checkpoints", "suggested_args": {}},
    {"category": "analysis", "title": "模型结构可视化", "available": true,
     "detail": "生成交互式 D3.js 模型管线图（FLOPs + 瓶颈标注 + 改进建议）",
     "suggested_tool": "run_visualization", "suggested_args": {"model_entry": "train:build_model"}}
  ],
  "hint": "训练已结束，建议按顺序执行以上待办项。先调用 get_summary 获取概览。"
}
```

---

### get_pending_decisions

获取所有待处理的 provisional 决策（MCP 模式下 agent 继续做决策，但标记为可覆盖）。每条决策含 id、决策点、临时动作、超时剩余秒数。外部 agent 审核后调用 `resolve_decision` 批准或覆盖；超时未处理自动转为 `approved`。

```
输入：无
```

**返回：**

```json
{
  "mode": "mcp_delegated",
  "count": 1,
  "pending": [
    {"id": "dec_1234", "decision_point": "monitor_response",
     "provisional_action": "restart_with_lower_lr",
     "context_summary": "loss_spike: loss +157% at step 1234",
     "created_at": 1754300000, "ttl": 120, "remaining_seconds": 45}
  ]
}
```

---

### get_dashboard_config

获取 Dashboard 当前配置：启用的图表组、面板显隐、平滑开关、布局模板。只读，无副作用。需 Dashboard 启用（`--with-dashboard`）。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `process_id` | string | 否 | 当前活动进程 | 训练进程 ID |

**返回：**

```json
{
  "template": "training",
  "charts": {"default_groups": ["loss", "accuracy"], "smoothing": false, "range_mode": "auto"},
  "panels": {"cursor_info": true, "logs": true, "ai_chat": false}
}
```

---

### recommend_charts

让 AI agent 分析当前训练状态（指标趋势、异常数量、训练阶段），推荐 Dashboard 应重点关注的图表组和显示配置（是否开平滑等）。需 `--agent` 且配置 API key。

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `process_id` | string | 否 | 当前活动进程 | 训练进程 ID |

**返回：**

```json
{
  "recommendation": {"groups": ["loss", "accuracy", "gpu"], "smoothing": true,
                     "reason": "训练进入后期，建议开启平滑观察趋势"},
  "source": "agent"
}
```

agent 未启用或推荐失败时降级返回：

```json
{"error": "agent 推荐失败，使用默认配置",
 "fallback": {"groups": ["loss", "accuracy"], "smoothing": false}}
```

---

### list_dashboard_templates

列出可用的 Dashboard 布局模板。

```
输入：无
```

**返回：**

```json
{
  "templates": [
    {"name": "training", "description": "训练监控：图表区（loss/accuracy/lr/gpu）+ 坐标信息 + 日志 + AI 对话",
     "panels": {"cursor_info": true, "logs": true, "ai_chat": true}},
    {"name": "comparison", "description": "实验对比：多个进程的图表并列 + 指标对比表格",
     "panels": {"cursor_info": false, "logs": false, "ai_chat": true}},
    {"name": "minimal", "description": "最小面板：仅图表区，适合嵌入或低带宽环境",
     "panels": {"cursor_info": false, "logs": false, "ai_chat": false}}
  ],
  "default": "training"
}
```

---

## 受限写工具（11 个）

### trigger_recovery

⚠️ 手动触发恢复重启，kill 训练子进程并回滚到最近 checkpoint。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token`

---

### restart_with_params

⚠️ 用调整后的参数重启训练。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `action` | string | **是** | `reduce_batch` / `restart_with_lower_lr` / `enable_grad_accum` |
| `param` | any | 否 | 动作参数：ratio / steps |
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token`  
**约束：** 参数必须在 `contract.yaml` 的 `adjustable_paths` 白名单范围内

**参数越界返回：**
```json
{"error": "参数越界：batch_size 最小值为 8，当前请求为 4"}
```

---

### stop_training

停止训练子进程并终止看护。已停止时重复调用无副作用。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token`

---

### approve_contract_proposal

批准一条 agent 的契约扩展提议。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `proposal_id` | string | **是** | 提议 ID |
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token`

---

### reject_contract_proposal

拒绝并归档一条契约扩展提议。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `proposal_id` | string | **是** | 提议 ID |
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token`

---

### run_visualization

生成模型管线可视化 HTML（交互式 D3.js 可折叠树）。

> ⚠️ 仅训练结束后可用

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `model_entry` | string | **是** | 如 `train_clip:build_model` |
| `output_path` | string | 否 | 输出路径（默认 `./logs/model_viz.html`） |
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token` + 训练结束

---

### set_gallery_config

更新图片筛选策略配置并触发重新筛选。

> ⚠️ 仅训练结束后可用

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `strategies` | object | **是** | 筛选策略 JSON（与 `propose_strategies` 输出一致） |
| `checkpoint_epoch` | integer | **是** | 用于推理的 checkpoint epoch |
| `data_source` | string | **是** | 数据源路径 |
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token` + 训练结束

---

### run_inference

使用指定 checkpoint 对输入数据跑推理。

> ⚠️ 仅训练结束后可用

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `checkpoint_epoch` | integer | **是** | checkpoint epoch |
| `task_type` | string | **是** | `classification` / `detection` / `segmentation` |
| `inputs` | string | **是** | 输入数据路径 |
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token` + 训练结束

---

### submit_import

提交外部训练数据导入。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `meta` | object | **是** | 元信息，必须含 `name` 字段 |
| `meta.name` | string | **是** | 实验名称 |
| `meta.source` | string | 否 | 数据来源（如 `wandb`, `tensorboard`, `csv_export`） |
| `meta.command` | string | 否 | 原始训练命令 |
| `metrics_path` | string | 否* | 本地 JSONL 文件路径（与 `metrics` 二选一） |
| `metrics` | array | 否* | 直接传指标列表（与 `metrics_path` 二选一，单次上限 100,000 条） |
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token`

**返回：**
```json
{"process_id": "import_a1b2c3d4", "records": 5000, "status": "imported"}
```

---

### set_dashboard_config

设置 Dashboard 配置：图表组选择、面板显隐、平滑开关、布局模板。Dashboard 前端通过 WebSocket 实时收到变更。用户的本地操作（checkbox/滑块）优先级始终最高，不受此工具覆盖。需 Dashboard 启用（`--with-dashboard`）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `process_id` | string | 否 | 训练进程 ID，默认当前活动进程 |
| `charts` | object | 否 | 图表配置：`{"default_groups": ["loss","accuracy"], "smoothing": true, "range_mode": "auto"}` |
| `panels` | object | 否 | 面板显隐：`{"cursor_info": true, "logs": true, "ai_chat": false}` |
| `template` | string | 否 | 布局模板：`training` / `comparison` / `minimal` |
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token`

**返回：**

```json
{
  "ok": true,
  "config": {
    "template": "training",
    "charts": {"default_groups": ["loss", "accuracy"], "smoothing": true, "range_mode": "auto"},
    "panels": {"cursor_info": true, "logs": true, "ai_chat": false}
  }
}
```

---

### resolve_decision

处理一条待定的 provisional 决策（来自 `get_pending_decisions`）。

- `override=false`：认可当前 provisional 决策，标记为 `approved`。
- `override=true`：用新的 `action` 覆盖。若覆盖动作是 `restart_with_lower_lr` / `reduce_batch` / `enable_grad_accum`，会立即执行重启式干预（kill 训练进程 + 回滚 checkpoint）；`stop_training` 会停止训练。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `decision_id` | string | **是** | 待处理决策 ID（从 `get_pending_decisions` 获取） |
| `override` | boolean | 否 | 是否覆盖（false=批准，true=覆盖） |
| `action` | string | 否* | 覆盖时的动作名（`override=true` 时必填） |
| `param` | any | 否 | 动作参数（ratio 或 steps，视动作类型而定） |
| `request_id` | string | 否 | 幂等键 |

**鉴权：** 需要 `write_token`

**返回：**

```json
{"status": "approved", "id": "dec_1234", "provisional_action": "restart_with_lower_lr",
 "corrective_needed": false}
```

`status` 可能值：`approved` | `overridden` | `not_found` | `already_resolved`

---

## 错误响应格式

所有工具在出错时返回：

```json
{"error": "错误描述"}
```

或在有详细信息时：

```json
{"error": "错误描述", "detail": "详细说明"}
```

---

## 鉴权错误码

| 错误 | 含义 |
|------|------|
| `写工具未启用（enable_write_tools=false）` | 配置中未开启写工具 |
| `未配置 write_token_env 环境变量` | 环境变量未设置 |
| `鉴权失败：token 不匹配` | 提供的 token 与环境变量不一致 |
| `工具仅在训练结束后可用` | 训练中调用了训练后专用工具 |
