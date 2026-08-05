# cp_9 · Agent 决策封装 (AgentAdvisor)

**文件**: `guardian/agent_advisor.py`
**阶段**: 全程
**核心目标**: 所有"中层 agent 化"决策点的统一出入口，负责调用 LLM、裁剪动作空间、强制超时降级，确保规则引擎兜底不被破坏

> **架构基线：sidecar-first**（见 [functional_overview.md](functional_overview.md#架构基线sidecar-first)）。advisor 与所有调用方一样跑在 guardian 进程里，**不在训练进程内**，这带来两个直接后果：
> - **`decision_timeout` 不占用训练时间**：等待 LLM 的 8 秒里训练子进程照常往前跑。代价是决策生效有延迟（训练在这期间又训了若干步），但不存在"主循环被网络请求卡死"的风险。嵌入模式下相反，`decide()` 是循环内同步调用，建议把超时压到 2 秒内或只在 epoch 边界触发。
> - **`action_space` 由调用方按当前模式裁剪后传入**：sidecar 下 [cp_2.md](cp_2.md) 不会把 `skip_batch` 放进来，[cp_11.md](cp_11.md) 会过滤掉映射不到命令行参数的路径。advisor 本身不判断模式，只校验"返回值是否在传入的 action_space 内"——模式差异对这一层是透明的。

---

## 为什么需要这一层

monitor（cp_2）和 watchdog（cp_3）里各有一个决策点想接入 agent：异常怎么应对、崩溃怎么恢复。如果让每个模块自己写"调 LLM、解析返回、超时怎么办"，会出现三个问题：

1. 超时/降级逻辑重复写三遍，容易有的地方漏掉降级，训练可能被一次卡住的网络请求拖死
2. 每个模块自己定义 prompt 格式和返回解析，非法输出的处理标准不统一
3. 没有统一的决策记录，summary（cp_5）想回溯"这次是 agent 决定的还是规则默认的"会很麻烦

`AgentAdvisor` 把这些逻辑收敛到一处：谁要用 agent 决策，都通过它的 `decide()` 调用，输入输出格式固定，超时/异常处理只写一次。

---

## 关键类与方法

### `AgentAdvisor`

| 方法 | 说明 |
|------|------|
| `__init__(config)` | 从 `config.agent` 读取 provider（anthropic/openai/自定义）、api_key、decision_timeout、是否启用 |
| `decide(decision_point, context, action_space, default_action)` | 主入口：组装 prompt，调用 LLM，校验返回，超时/失败/非法输出时返回 `default_action` |
| `narrate(structured_data, prompt_template)` | 纯文本生成：用于 summary 的自然语言解读，无动作约束，失败返回 `None` |
| `suggest(kind, context, registry_snapshot)` | 只读式建议生成：用于 cp_11 的 `propose_registry_entry`，输出一条候选条目 + 依据，不生效、不走超时降级语义（本身就是"仅供人工参考"），失败返回 `None` |
| `_build_prompt(decision_point, context, action_space)` | 拼接结构化上下文和动作集为 LLM 可理解的 prompt |
| `_call_llm(prompt, timeout)` | 实际网络调用，带超时控制 |
| `_validate_action(raw_output, action_space)` | 校验 LLM 返回是否为动作集内的合法动作（含参数范围检查） |
| `_log_decision(decision_point, context, chosen_action, source, latency_ms)` | 记录每次决策：来源（agent/rule_default/timeout/invalid_output）、耗时、依据 |
| `is_enabled()` | 返回当前是否可用（配置检查 + 上次调用是否连续失败触发熔断） |

---

## 调用契约

```
decide(decision_point, context, action_space, default_action)
  |
  |- is_enabled() == False（未配置/熔断中）
  |    `- 直接返回 default_action，source="disabled"
  |
  |- 调用 LLM，等待 decision_timeout 秒
  |    |- 超时
  |    |    `- 返回 default_action，source="timeout"，记录告警（不推送，仅日志）
  |    |- 网络/API 错误
  |    |    `- 返回 default_action，source="error"
  |    `- 正常返回
  |         |- _validate_action() 通过 -> 返回 agent 选择的动作，source="agent"
  |         `- _validate_action() 未通过（不在动作集内/参数越界）
  |              `- 返回 default_action，source="invalid_output"
  |
  `- 无论哪条分支，_log_decision() 记录一次，供 summary 和 MCP 工具查询
```

**调用方永远拿到一个合法动作**，不需要自己处理"LLM 没返回怎么办"——这是这一层存在的核心价值。

---

## 熔断机制

```
连续 consecutive_failure_threshold（默认 5）次调用超时/报错
  -> 自动熔断，is_enabled() 返回 False
  -> 后续 circuit_breaker_cooldown（默认 600 秒）内不再尝试调用 LLM，直接走默认动作
  -> cooldown 结束后自动恢复尝试
```

避免网络间歇性故障时，训练主循环反复等待相同的超时时间。

---

## 配置示例

```yaml
agent:
  enabled: true
  provider: anthropic          # anthropic / openai / custom
  model: claude-sonnet-4-5
  api_key_env: ANTHROPIC_API_KEY
  decision_timeout: 8          # 秒，超过直接降级
  consecutive_failure_threshold: 5
  circuit_breaker_cooldown: 600
  decision_points:             # 逐个决策点开关，可单独关闭
    monitor_response: true
    watchdog_recovery: true
    summary_narrative: true
    select_metric: true         # cp_11：best-model 判定指标选择
    select_adjust_path: true    # cp_11：可调路径与幅度选择
```

`select_metric` / `select_adjust_path` 是 cp_11 新增的两个决策点，走的是完全相同的 `decide()` 契约（超时/熔断/非法输出回退逻辑不变），只是 `action_space` 换成了 cp_11 的指标注册表 / 可调路径白名单而已——AgentAdvisor 本身不需要为它们新增任何特殊逻辑。`suggest()` 是唯一的例外：它服务于"agent 觉得注册表/白名单不够用，想建议新增一条"这种场景，产出的内容不生效、不需要超时降级（找不到就是找不到，不影响当次训练），因此不走 `decide()`，详见 [cp_11.md](cp_11.md)。

`agent.enabled: false` 或未配置 `api_key_env` 对应的环境变量时，`is_enabled()` 恒为 False，整个 agent 层零成本降级为原有规则引擎行为——这是离线/无网络服务器环境的默认路径，不需要额外代码分支。

---

## ✅ 快速校验

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| 类可实例化 | `AgentAdvisor(config)` | 无异常 |
| 未配置降级 | `agent.enabled: false` 调用 `decide()` | 立即返回 default_action，source="disabled" |
| 正常决策 | mock LLM 返回合法动作 | 返回该动作，source="agent" |
| 非法输出降级 | mock LLM 返回不在 action_space 内的值 | 返回 default_action，source="invalid_output" |
| narrate 基本可用 | 传入结构化数据 | 返回一段文本，不报错 |

---

## ✅ 完整校验

| 检查项 | 验证方式 | 通过标准 |
|--------|----------|----------|
| 超时降级 | mock LLM 调用阻塞超过 decision_timeout | 在超时时刻立即返回 default_action，不再等待 |
| 熔断触发 | 连续 5 次超时/报错 | 第 6 次调用不再发起网络请求，直接返回默认值 |
| 熔断恢复 | cooldown 结束后再次调用 | 恢复正常尝试 LLM 调用 |
| 参数范围校验 | agent 返回 `restart_with_lower_lr(ratio=5.0)`（超出 cp_11 白名单的 max_delta_ratio） | 视为非法输出，降级 |
| 决策记录完整 | 连续触发 10 次不同 decision_point | 每次都有日志，含 source/latency/context 摘要 |
| narrate 失败降级 | mock LLM 报错 | 返回 None，调用方（summary）不崩溃 |
| 并发调用安全 | monitor 和 watchdog 同时触发 decide() | 互不阻塞，各自独立超时 |
| 单点开关生效 | `decision_points.watchdog_recovery: false` | watchdog 直接走默认策略，monitor 不受影响 |
| select_metric 走既有契约 | `decision_points.select_metric: false` | cp_11 的 `select_metric` 直接回退 `_fallback` 指标，与其他决策点降级路径一致 |
| suggest 不影响主流程 | mock `suggest()` 抛异常 | 返回 `None`，cp_11 的 `propose_registry_entry` 记录"本次未生成建议"，训练/决策流程不受影响 |
| 超时不影响训练（sidecar） | `guardian watch` 运行中，mock LLM 阻塞满 decision_timeout | 训练子进程在这 8 秒内持续产出新指标/step，未被暂停或减速 |
| action_space 由调用方裁剪 | sidecar 下 cp_2 传入不含 `skip_batch` 的 action_space，mock LLM 仍返回 `skip_batch` | 判为 invalid_output 并降级——advisor 不需要知道当前是哪种模式 |
