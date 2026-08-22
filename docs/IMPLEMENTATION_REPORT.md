# Training Guardian Agent — 实现对照报告

> 对照 `checkpoint/` 设计文档，逐模块评估实现完成度。
> 生成时间：2026-08-09 · 提交范围：`c5e5763` → `dcfb42d`（27 次提交）· 版本 v0.2.0

---

## 总览

| 指标 | 值 |
|------|-----|
| 代码行数 | ~12500 行（含测试 ~3800 行 + 前端 SPA ~2100 行） |
| 源文件 | 19 个 cp 模块（cp_1~cp_19）+ run.py + train.py + Dashboard SPA |
| 测试 | 223 个，全通过 |
| 版本 | v0.2.0 |
| MCP 工具 | 35 个（24 只读 + 11 受限写） |
| v0.2 完成度 | **92%** |

---

## 逐模块对照

### cp_3 · 进程守护与恢复（watchdog.py, 670 行）

**设计文档**: `checkpoint/cp_3.md`  
**关键方法**: `run()`, `classify_crash()`, `apply_action()`, `_decide_recovery()`, `check_hang()`

| 设计项 | 状态 | 说明 |
|--------|------|------|
| 子进程包装与看护 | ✅ | `subprocess.Popen` + 轮询循环，SIGTERM 优雅退出 |
| 崩溃分类（纯规则） | ✅ | 退出码 + stderr 正则，OOM/sigkill/network/code/data/unknown 六类 |
| 从 checkpoint 重启 | ✅ | `find_latest_checkpoint()` 按 epoch 降序扫描，损坏跳过 |
| 命令行改写执行层 | ✅ | `apply_action()`: reduce_batch / restart_with_lower_lr / enable_grad_accum，只替换不追加 |
| 挂起检测 | ✅ | `check_hang()`: 指标停滞 + 进程存活双判据，默认只告警 |
| Agent 恢复策略决策 | ✅ | `_decide_recovery()`: OOM/sigkill/network 各有 action_space |
| 规则默认策略 | ✅ | `default_strategy()`: OOM→reduce_batch, network→delay, 其他→unchanged |
| 主动干预汇入重启路径 | ✅ | `request_intervention()` → 下一个看护周期 kill + 重启 |
| 恢复记录（RestartRecord） | ✅ | trigger/intervention/crash/hang, wasted_epochs, cmd_before/after |
| 不删最近 checkpoint | ✅ | keep_recent 保护，`cleanup()` 跳过 |
| DDP 场景保守处理 | ⚠️ | `batch_adjustable()` 检测，v0 只做原样重启，不调 batch |
| 嵌入模式 | ❌ | 未实现，设计文档列为可选升级 |

**完成度: 92%**

---

### cp_2 · 训练监控（monitor.py, 571 行）

**设计文档**: `checkpoint/cp_2.md`  
**关键方法**: `poll_metrics()`, `_check()`, `_check_loss_spike()`, `_emit()`, `_decide_response()`

| 设计项 | 状态 | 说明 |
|--------|------|------|
| 增量日志 tail | ✅ | `LogFileChannel`: 字节偏移追踪，支持截断/轮转检测 |
| 滑动窗口异常检测 | ✅ | Loss spike（3+样本均值比较）、stagnation（N步降幅阈值）|
| NaN/Inf 检测 | ✅ | 不进滑动窗口，避免污染 |
| GPU 硬件轮询 | ✅ | `poll_gpu()`: nvidia-smi CSV 解析，温度/利用率/显存/功耗 |
| GPU 空转检测 | ✅ | 连续 5 次利用率 < 阈值 → 告警 |
| GPU 温度检测 | ✅ | 超阈值即告警，无自动降频（硬件安全不交 agent） |
| Agent 异常应对决策 | ✅ | `_decide_response()`: loss_spike/nan_inf/gpu_idle 各有 action_space |
| 有限动作集 | ✅ | sidecar 下不含 skip_batch/lower_lr，只含重启式动作 |
| 干预回调 | ✅ | `on_intervention` → watchdog.request_intervention |
| wandb/tensorboard 通道 | ✅/❌ | wandb ✅（本地 run 目录增量读取）；tensorboard ❌ |
| 嵌入模式 GuardianCallback | ❌ | 设计文档列为可选升级 |

**完成度: 88%**

---

### cp_6 · 告警推送（notifier.py, 121 行）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| 终端输出 | ✅ | 带图标 + 颜色，UTF-8 兼容 Windows |
| Webhook 推送 | ✅ | HTTP POST JSON，可选 requests 依赖，10s 超时 |
| 静默期 | ✅ | cooldown 秒内同类告警不重复推送；error 级永不静默 |
| 干预代价可见 | ✅ | response 字段含 action/resumed_from/wasted_epochs |
| Email 渠道 | ❌ | stub，打印"尚未实现" |

**完成度: 80%**

---

### cp_4 · 断点分析（checkpoint_analyzer.py, 251 行）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| 目录轮询发现 | ✅ | `poll()`: 扫描 cp_*/ 目录，epoch 排序 |
| 写入完成判定 | ✅ | 稳定性判据（连续 N 次轮询 size/mtime 不变）+ 可加载性检查 |
| 契约必需键校验 | ✅ | `_has_required_keys()`: torch.load + 键检查 |
| Top-k 管理 + keep_recent | ✅ | `cleanup()`: 保护最近 N 个，其余只保留 top-k |
| 报告生成 | ✅ | `report()`: total/latest/best/metric/checkpoints |
| 快速校验/完整校验 | ❌ | 需 buildable_entry 契约（v1），当前只读 metrics.json |
| best.pth 软链接 | ❌ | 未实现 |
| per-class 指标/混淆矩阵 | ❌ | 需完整校验支持 |
| metric_source 记录 | ✅ | report() 自动调 contract.select_metric() |

**完成度: 75%**

---

### cp_5 · 日志摘要（summary.py, 279 行）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| 结构化摘要生成 | ✅ | experiment_id/status/duration/training/best_model/anomaly_events/restarts/checkpoints |
| 终端渲染 | ✅ | `render()`: 表格格式，含异常事件、重启记录 |
| JSON + TXT 保存 | ✅ | `save_summary()`: 双格式落盘 |
| 重启记录 | ✅ | trigger（crash/intervention/hang）、wasted_epochs、参数变更 |
| GPU 资源统计 | ✅ | `_collect_resource_usage()`: 利用率均值/峰值显存/GPU时数/温度 |
| LR 调度历史 | ✅ | `_collect_lr_schedule()`: 跨重启拼接 |
| AI 自然语言解读 | ✅ | `_generate_ai_narrative()` → advisor.narrate() → DeepSeek |
| 最优模型指标口径标注 | ❌ | 需 cp_11 select_metric（v1） |

**完成度: 85%**

---

### cp_7 · 参考训练脚本（train.py, 225 行）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| SimpleCNN 模型 | ✅ | ~600K 参数，MNIST 分类 |
| 契约四项满足 | ✅ | --resume/--ckpt、checkpoint_schema、log_file、buildable_entry |
| 原子写 checkpoint | ✅ | tmp 目录 + rename |
| 结构化日志 | ✅ | `epoch {n} loss {v} val_acc {v} lr {v}` 格式，可被 cp_2 解析 |

**完成度: 95%**

---

### cp_8 · CLI 入口（cli.py, 1419 行；run.py 已改为 shell 包装）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| watch 子命令 | ✅ | 守护任意训练命令，`--` 透传 |
| contract check 子命令 | ✅ | 契约四项逐项校验 + 降级说明 |
| contract review 子命令 | ✅ | 列出待审核提议，经 MCP approve/reject 审核（cp_11 已完成） |
| init 子命令 | ✅ | 初始化项目：自动扫描训练脚本 → 生成 guardian.yaml + contract.yaml |
| check 子命令 | ✅ | 环境就绪检查：Python 版本 / 依赖 / GPU / 项目配置 |
| start 子命令 | ✅ | 一键启动 Dashboard + MCP server（可选附带训练守护） |
| analyze 子命令 | ✅ | 独立扫描 checkpoint 目录 |
| preflight 子命令 | ✅ | cp_1 资源预检入口 |
| serve 子命令 | ✅ | cp_10 MCP server 独立进程（stdio/sse/http/tcp） |
| --agent / --with-mcp / --with-dashboard | ✅ | watch 同时启用 agent 决策层 / 后台 MCP server / Web 面板 |
| --version | ✅ | `guarftrain --version`（0.2.0） |
| 入口形态 | ✅ | run.py 改为 shell 包装（转发 guardian.cli.main）；pip 安装后 `guarftrain` 全局可用 |

**完成度: 90%**

---

### cp_11 · 任务契约（task_contract.py, 514 行）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| contract.yaml 加载 | ✅ | YAML 解析 + base_dir 相对路径解析 |
| 契约四项校验 | ✅ | resumable（含 --help 探测）/ checkpoint_schema / metrics_channel / buildable_entry |
| 降级表 | ✅ | 每项缺失对应关闭的能力明确列出 |
| strict_mode | ✅ | 缺失即拒绝启动 |
| cli_mappings 解析 | ✅ | 抽象路径 → 命令行 flag |
| batch_adjustable() | ✅ | launcher 检测，DDP 下 v0 不调 batch |
| metric_registry | ✅ | 5 类任务 + fallback |
| select_metric | ✅ | 4 层推断: config_explicit → agent_decide → key_infer → fallback |
| select_adjust_path | ✅ | 白名单 + cli_mappings 过滤 + agent 选择 |
| 提议审核系统 | ✅ | propose/approve/reject/list_proposals，JSON 落盘 |

**完成度: 90%**

---

### cp_9 · Agent 决策封装（agent_advisor.py, 810 行）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| decide() 统一入口 | ✅ | 超时/异常/disabled/invalid_output 全返回合法动作 |
| narrate() 文本生成 | ✅ | 2x timeout，失败返回 None |
| suggest() 提议生成 | ✅ | 同步无超时，失败返回 None |
| Anthropic SDK 接入 | ✅ | `_call_anthropic()`: base_url + auth_token + 安全文本提取 |
| OpenAI 备选 | ✅ | `_call_openai()`: 同接口 |
| 第三方兼容 API | ✅ | ANTHROPIC_BASE_URL / ANTHROPIC_AUTH_TOKEN / ANTHROPIC_MODEL 环境变量 |
| 熔断机制 | ✅ | 连续 5 次失败 → 600s cooldown → 自动恢复 |
| 决策日志 | ✅ | decision_point / action / source / latency_ms / context |
| Prompt 构建器 | ✅ | decide/narrate/suggest 三种场景各有 system + user |
| 响应解析 | ✅ | `_parse_llm_response()`: JSON/纯文本/混合提取 |
| 动作校验 | ✅ | `_validate_action()`: 参数范围 + 动作集白名单 |
| per-point 开关 | ✅ | decision_points 逐点启用/关闭 |
| 图表推荐 | ✅ | `recommend_charts()` + `chart_selection` 决策点，详见 cp_18 |

**完成度: 90%**

---

### cp_1 · 资源预估（resource_estimator.py, 483 行）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| GPU 信息获取 | ✅ | torch.cuda / GPUtil 回退 |
| 显存测量 | ✅ | 小 batch 前向+反向 → 峰值记录 |
| 线性回归 | ✅ | y=a*x+b, R² 评估 |
| batch 推荐 | ✅ | memory_margin + cp_11 白名单上限约束 |
| 时间预估 | ✅ | 单 step 测量 → 外推总时长 |
| 终端输出 | ✅ | `print_report()` |
| 测量无副作用 | ✅ | 独立进程，退出即清理 |
| 多卡支持 | ⚠️ | 当前仅报告第一张卡 |

**完成度: 85%**

---

### cp_10 · MCP 工具层（mcp_server.py, 2150 行）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| 10 只读工具 | ✅ | status/history/checkpoints/anomalies/recovery/summary/log/contract |
| 8 v2 只读工具 | ✅ | experiments/query/compare/model_structure/mode/gallery/import_format/inspect_source |
| 6 受限写工具 | ✅ | recovery/restart_with_params/stop/validate/approve/reject |
| 4 v2 受限写工具 | ✅ | run_visualization/set_gallery_config/run_inference/submit_import |
| 4 Dashboard 远程配置工具 | ✅ | get/set_dashboard_config + recommend_charts + list_dashboard_templates（需 dash_url，详见 cp_19） |
| 共 35 个 MCP 工具 | ✅ | 24 只读 + 11 受限写（v0.1 为 28 个：18 只读 + 10 写） |
| MCP annotations | ✅ | readOnlyHint/destructiveHint/idempotentHint |
| 幂等保证 | ✅ | request_id + dedup window |
| 访问日志 | ✅ | mcp_access_log.json |
| 写工具鉴权 | ✅ | write_token 口令校验 |
| 训练阶段保护 | ✅ | 训练后专用工具在训练中调用返回错误 |
| 双模式架构 | ✅ | Standalone / MCP_Delegated 自动切换 |
| 非阻塞保证 | ✅ | 后台线程、崩溃不影响训练 |
| 跨进程状态快照 | ✅ | standalone 模式定期读盘 |
| 传输方式 | ✅ | stdio / SSE / HTTP / TCP |
| SDK 版本兼容 | ✅ | mcp<2.0 和 mcp>=1.0 双路径兼容 |
| 工具 Schema 注入 | ✅ | _SchemaInjectedMCPServer 补全 SDK v2 schema |
| MCP SSE 测试 | ✅ | test_mcp_sse.py：原始 socket 协议测试 |
| 真实 Claude Code 接入 | ✅ | 已通过项目级 + 用户级 mcp.json 配置验证；另有 test_mcp_all_tools.py / test_dashboard_mcp.py 全工具 E2E 脚本 |

**完成度: 92%**（sys import bug 已修复，35 工具全覆盖，文档已补写）

---

### cp_12 · 故障注入测试（tests/faultbench/, 6 文件）

| 设计项 | 状态 | 说明 |
|--------|------|------|
| fake_train.py | ✅ | 可编程失败：oom/type_error/oom_if_batch_gt/self_kill/hang/nan/loss_spike |
| S1: OOM 救回 | ✅ | 条件 OOM → 减 batch → 跑完全部 epoch |
| S2: kill -9 续训 | ✅ | SIGKILL → 参数不变从 ckpt 续训 |
| S3: 代码错误不重启 | ✅ | TypeError → 0 次重启 |
| 命令行改写验证 | ✅ | reduce_batch 后 --batch_size 正确替换 |
| 环境层故障 | ❌ | cgroup OOM / mock nvidia-smi / 磁盘写满 / LLM 不可用 未覆盖 |

**完成度: 75%**

---

### cp_17 · Dashboard 远程配置（dashboard/server.py, 1241 行）

**设计文档**: `checkpoint/cp_17.md`（待补写，INDEX 已登记 v0.2）  
**关键方法**: `_build_app()`, `_broadcast_process()`, `get/set dashboard-config` 端点, `bind_guardian()`

| 设计项 | 状态 | 说明 |
|--------|------|------|
| dashboard-config 端点 | ✅ | GET/POST `/api/process/{id}/dashboard-config`：GET 拉取，POST 深度合并（template/charts/panels 顶层 key，charts/panels 整体替换） |
| WebSocket 实时推送 | ✅ | `/ws/process/{id}` 单进程流 + `/ws` 全局事件流；配置变更广播 `dashboard-config` 事件（携带 source） |
| dirty flag 用户保护 | ✅ | 前端 `_dashDirty` 记录用户手动改过的项，远程配置不覆盖；reset 时 `_dashDirty.clear()` 并重应用远程配置 |
| 来源标记 | ✅ | `_source` 区分 user / api / mcp_agent，用户本地改动不回灌 |
| 前端 SPA | ✅ | `guardian/dashboard/static/index.html`（~2100 行）：`loadDashboardConfig()` / `applyRemoteDashConfig()` / `markDashDirty()` |
| 契约初始化配置 | ✅ | contract.yaml `dashboard` 段（template/charts/panels）随 `/api/register` 注入；无配置时用 DASH_DEFAULTS |
| 默认配置 | ✅ | template=training，charts=[loss, accuracy]，smoothing=false，range_mode=auto |

**完成度: 90%**

---

### cp_18 · Agent 图表推荐（agent_advisor.py, 810 行）

**设计文档**: `checkpoint/cp_18.md`（待补写，INDEX 已登记 v0.2）  
**关键方法**: `recommend_charts()`, `_call_llm_chart_recommend()`, `SYSTEM_CHART_RECOMMEND`

| 设计项 | 状态 | 说明 |
|--------|------|------|
| recommend_charts() | ✅ | 输入 process_id / metrics_summary / anomaly_count / training_phase，返回推荐配置 dict；失败返回 None（降级为配置不变） |
| chart_selection 决策点 | ✅ | 新决策点，per-point 开关可关，关闭时直接返回 None |
| SYSTEM_CHART_RECOMMEND 提示词 | ✅ | 规则化推荐：loss 异常或早期必选 loss；中后期加 accuracy；GPU 异常加 gpu；临近结束开 smoothing |
| 输出约束 | ✅ | groups 必须是 available_groups 子集；smoothing 布尔；附一句话中文理由 |
| 超时与熔断 | ✅ | 复用 decision_timeout + 失败计数/熔断，LLM 失败不中断训练与面板流程 |

**完成度: 90%**

---

### cp_19 · MCP Dashboard 工具（mcp_server.py, 2150 行）

**设计文档**: `checkpoint/cp_19.md`（待补写，INDEX 已登记 v0.2）  
**关键方法**: `_dash_request()`, `_handle_get_dashboard_config()`, `_handle_set_dashboard_config()`, `_handle_recommend_charts()`, `_handle_list_dashboard_templates()`

| 设计项 | 状态 | 说明 |
|--------|------|------|
| dash_url 参数 | ✅ | 构造参数 `dash_url`（watch 自动注入），未启用 Dashboard 时工具返回明确错误 |
| get_dashboard_config | ✅ | 只读：拉取指定进程的 Dashboard 配置 |
| set_dashboard_config | ✅ | 受限写（write_token 鉴权）：更新配置并经 WebSocket 广播前端 |
| recommend_charts | ✅ | 只读：拉取最近 200 条指标 + 当前配置 → `advisor.recommend_charts()` → 返回推荐配置；agent 未启用/失败时返回 fallback |
| list_dashboard_templates | ✅ | 只读：列出可用模板（training / comparison / minimal） |
| 鉴权/幂等/日志复用 | ✅ | 沿用 write_token 鉴权 + request_id 幂等 + mcp_access_log.json |

**完成度: 90%**

---

## Agent 决策回路验证

```
[E2E 实测] fake_train --epochs 1 --agent

loss_spike 检测 → agent 决策:
  +157% → agent -> restart_with_lower_lr (重启式干预)
  +117% → agent -> ignore              (智能忽略)
  +96%  → agent -> restart_with_lower_lr
  +84%  → agent -> restart_with_lower_lr
  +76%  → agent -> restart_with_lower_lr
  +70%  → agent -> restart_with_lower_lr

AI 解读（DeepSeek, ~250字）:
  "训练虽跑完，但自救机制未真正执行（fake_train 不支持真实重启），
   最终模型表现远低于最佳水平，建议排查学习率调整与 Loss 突增的根因，
   并考虑改用最佳检查点恢复部署。"
```

---

## 设计文档覆盖度

| 文档 | 行数 | 覆盖状态 |
|------|------|----------|
| overview.md | 126 | ✅ 全部模块已实现或标注为 v1 延后 |
| functional_overview.md | 224 | ✅ v0 全实现 + v1 部分实现 |
| configuration.md | 189 | ✅ 全部默认值在 config.py DEFAULTS 中 |
| requirements.md | 60 | ✅ requirements-core.txt（v0.2 已去 torch/anthropic，改为 pip extras 按需安装）+ requirements-mcp.txt |
| cp_1.md | 109 | ✅ resource_estimator.py 已实现 |
| cp_2.md | 176 | ✅ 88%，缺 tensorboard 通道和嵌入模式 |
| cp_3.md | 265 | ✅ 92%，缺嵌入模式 |
| cp_4.md | 198 | ⚠️ 75%，缺独立评估 |
| cp_5.md | 180 | ✅ 85%，缺指标口径标注 |
| cp_6.md | 105 | ⚠️ 80%，缺 email |
| cp_7.md | 190 | ✅ 95% |
| cp_8.md | 216 | ✅ 90% |
| cp_9.md | 133 | ✅ 90%，LLM 全通 |
| cp_10.md | 229 | ✅ 92%，真实 MCP 客户端测试已完成（SSE 原始 socket + Claude Code 实测 + E2E 脚本） |
| cp_11.md | 225 | ✅ 90%，注册表 + 提议审核已完成 |
| cp_12.md | 145 | ⚠️ 75%，缺环境层故障 |
| cp_17.md | — | ⚠️ 实现完成（server.py），设计文档待补写 |
| cp_18.md | — | ⚠️ 实现完成（agent_advisor.py），设计文档待补写 |
| cp_19.md | — | ⚠️ 实现完成（mcp_server.py），设计文档待补写 |

---

## 已知差距排序

| 优先级 | 差距 | 影响 |
|--------|------|------|
| 🔴 | cp_4 quick/full validate | checkpoint 不做独立评估，只读训练脚本自报指标（v1 最大功能缺口） |
| 🟡 | cp_2 tensorboard 通道 | 只支持 log_file / metrics_json / wandb，tensorboard 用户需自行转存 |
| 🟡 | cp_17/18/19 设计文档待补写 | 三个 v0.2 新模块实现完成，checkpoint/ 下仅 INDEX 登记无正文 |
| 🟢 | cp_6 email | terminal + webhook 已覆盖主要场景 |
| 🟢 | cp_4 best.pth 软链接 | 不影响 best 判定逻辑 |
| 🟢 | cp_12 环境层故障 | 核心验收标准已覆盖，SSE/HTTP 全工具 E2E 脚本已补真实客户端路径 |
| 🟢 | Dashboard 图表推荐仅 MCP 侧触发 | recommend_charts 无前端按钮入口；用户手动改动受 dirty flag 保护 |
