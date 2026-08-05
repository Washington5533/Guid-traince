# 配置参考 — guardian.yaml / contract.yaml

两份配置文件，职责不同，不要混：

| 文件 | 内容 | 谁会改 |
|------|------|--------|
| `configs/guardian.yaml` | **guardian 自己怎么工作**：轮询间隔、检测阈值、重试次数、agent/MCP 开关 | 调优时改 |
| `configs/contract.yaml` | **被守护的训练脚本长什么样**：可续训入口、指标通道、命令行映射、可调路径白名单 | 接入一个新项目时改一次，详见 [cp_11.md](cp_11.md) |

分开的理由：guardian.yaml 是 guardian 的调优参数，换项目基本不用动；contract.yaml 是"这个训练脚本的接口描述"，每个项目一份。混在一起会导致换项目时要在一个大文件里挑挑拣拣。

---

## v0 最小配置

sidecar 守护闭环真正必需的只有这些，其余全部有合理默认值：

```yaml
# configs/guardian.yaml
project:
  name: my-experiment
  ckpt_dir: ./checkpoints
  log_dir: ./logs

watchdog:
  max_retries: 3

monitor:
  poll_interval: 10

notifier:
  channels: [terminal]
```

```yaml
# configs/contract.yaml
script_contract:
  resumable:
    resume_flag: "--resume"
    ckpt_flag: "--ckpt"
  metrics_channel:
    type: log_file
    path: ./logs/train.log
    log_pattern: "epoch (\\d+).*loss ([\\d.]+)"
  cli_mappings:
    dataloader.batch_size: "--batch_size"
    optimizer.lr: "--lr"
```

这两份文件 + `guardian watch -- python train.py` 就是完整的 v0。`checkpoint_schema` / `buildable_entry` 未声明时对应能力自动关闭（见 cp_11 降级表），不阻断启动。

---

## guardian.yaml 完整参考

### project

| 键 | 默认 | 说明 |
|----|------|------|
| `name` | `guardian-run` | 实验名，用于摘要与推送标题 |
| `ckpt_dir` | `./checkpoints` | cp_3 找恢复点、cp_4 轮询发现都用这个目录 |
| `log_dir` | `./logs` | 摘要、决策日志、访问日志的落盘位置 |
| `device` | `auto` | 仅影响 cp_1 预检与 cp_4 校验；训练用什么设备由训练脚本自己决定 |

### watchdog（cp_3，v0 核心）

| 键 | 默认 | 说明 |
|----|------|------|
| `max_retries` | `3` | 连续失败上限，达到后停止重试并推送最终诊断 |
| `restart_delay` | `10` | 秒，重启前等待，避免瞬时资源未释放 |
| `oom_batch_reduce_ratio` | `0.5` | OOM 时规则默认的 batch 缩减比例 |
| `min_batch_size` | `8` | batch 下限，减到此值仍 OOM 则停止 |
| `sigterm_grace` | `30` | 秒，主动干预时先 SIGTERM 等待这么久，超时才 SIGKILL（避免半截 checkpoint） |
| `no_progress_timeout` | `1800` | 秒，多久没有新指标算"疑似无进展"，触发告警 |
| `no_progress_kill_after` | `null` | 秒，挂起多久后 kill 重启；**`null` = 永不自动处理，只告警**。必须按自己任务的 epoch 时长设定，guardian 不猜 |
| `keep_training_on_exit` | `true` | guardian 自己退出时是否留下训练子进程继续跑 |

### monitor（cp_2，v0 核心）

| 键 | 默认 | 说明 |
|----|------|------|
| `poll_interval` | `10` | 秒，读取指标通道的间隔。**这是 sidecar 下检测延迟的下限** |
| `hardware_poll_interval` | `30` | 秒，轮询 nvidia-smi 的间隔 |
| `sliding_window` | `50` | 滑动窗口大小，用于自适应阈值 |
| `loss_spike_ratio` | `0.5` | 超过窗口均值这个比例判为突增 |
| `loss_stagnation_steps` | `500` | 连续多少步无改善判为停滞 |
| `loss_stagnation_threshold` | `0.001` | 该区间内的最小可接受降幅 |
| `gpu_idle_threshold` | `20` | GPU 利用率低于此值（连续 5 次采样）判为空转 |
| `gpu_temp_threshold` | `85` | 摄氏度，超过则告警（**无自动降频，硬件安全不交给 agent**） |

### notifier（cp_6，v0 核心）

| 键 | 默认 | 说明 |
|----|------|------|
| `channels` | `[terminal]` | `terminal` / `webhook` / `email`，可多选 |
| `cooldown` | `300` | 秒，同类告警静默期，防刷屏 |
| `webhook_url_env` | `GUARDIAN_WEBHOOK_URL` | 从环境变量读取，不写进配置文件 |
| `webhook_timeout` | `10` | 秒，推送超时后放弃，不阻塞看护循环 |
| `smtp_*` | - | 邮件配置；密码走环境变量 |

### checkpoint（cp_4，v0 核心）

| 键 | 默认 | 说明 |
|----|------|------|
| `poll_interval` | `30` | 秒，扫描 ckpt_dir 发现新目录的间隔 |
| `save_top_k` | `5` | 保留最优的 k 个 |
| `keep_recent` | `2` | **额外无条件保留最近 N 个 epoch 的目录**，避免删掉训练脚本下次 resume 要用的文件 |
| `quick_val_sample_ratio` | `0.05` | 快速校验的采样比例 |
| `full_val_every_n` | `5` | 每 N 个 checkpoint 做一次完整校验 |
| `stability_checks` | `2` | 判定"写完了"需要连续几次轮询 size/mtime 不变 |

### preflight（cp_1，v1）

| 键 | 默认 | 说明 |
|----|------|------|
| `enabled` | `true` | 需要 `buildable_entry` 契约项，缺失则自动跳过 |
| `test_batch_sizes` | `[1, 2, 4]` | 实测采样点，用于线性回归外推 |
| `memory_margin` | `0.2` | 预留显存比例 |

### agent（cp_9，v1）

| 键 | 默认 | 说明 |
|----|------|------|
| `enabled` | `false` | **默认关闭**。未配置或无 API key 时全层零成本降级为纯规则 |
| `provider` | `anthropic` | `anthropic` / `openai` / `custom` |
| `model` | - | 模型 id |
| `api_key_env` | `ANTHROPIC_API_KEY` | 走环境变量 |
| `decision_timeout` | `8` | 秒。sidecar 下不占训练时间；**嵌入模式建议压到 2 以内**（见 cp_9） |
| `consecutive_failure_threshold` | `5` | 连续失败几次触发熔断 |
| `circuit_breaker_cooldown` | `600` | 秒，熔断冷却 |
| `decision_points.*` | 全 `true` | 逐点开关：`monitor_response` / `watchdog_recovery` / `summary_narrative` / `select_metric` / `select_adjust_path` |

### mcp（cp_10，v1）

| 键 | 默认 | 说明 |
|----|------|------|
| `enabled` | `false` | 需要 `requirements-mcp.txt`；未装包只警告不报错 |
| `transport` | `stdio` | `stdio` / `tcp` |
| `tcp_port` | `8765` | **仅绑 127.0.0.1**，远程访问走 SSH 隧道 |
| `enable_write_tools` | `false` | 默认关闭全部受限写工具 |
| `write_token_env` | `GUARDIAN_MCP_TOKEN` | 写工具口令，走环境变量 |
| `state_refresh_interval` | `5` | 秒，`run.py serve` 跨进程读盘刷新间隔 |
| `dedup_window` | `300` | 秒，相同 `request_id` 在此窗口内视为重复调用（幂等保证） |
| `default_result_limit` | `200` | 只读工具默认返回条数上限，防止塞爆 agent 上下文 |

### contract（指向 contract.yaml，cp_11）

| 键 | 默认 | 说明 |
|----|------|------|
| `path` | `configs/contract.yaml` | 契约文件位置 |
| `strict_mode` | `false` | `true` 时任一必需契约项缺失即阻止启动 |
| `agent_can_propose` | `true` | 是否允许 agent 生成注册表扩展提议（不生效，需人工审核） |
| `proposal_log` | `logs/contract_proposals.json` | 提议记录落盘位置 |

---

## 配置优先级

```
命令行参数  >  环境变量 GUARDIAN_*  >  配置文件  >  内置默认值
```

环境变量命名用双下划线表示层级：`GUARDIAN_WATCHDOG__MAX_RETRIES=5` 覆盖 `watchdog.max_retries`。

**secrets 一律只走环境变量**，不接受写在 yaml 里：`api_key_env` / `write_token_env` / `webhook_url_env` / smtp 密码。配置文件里存的是"环境变量的名字"，不是值本身。

---

## 几个容易设错的参数

- **`monitor.poll_interval` 决定检测延迟下限**。设成 60 秒就意味着 NaN 最坏情况过一分钟才被发现。但设得太小（<5 秒）在指标文件很大时会有 I/O 开销，且训练脚本本身的输出频率是更硬的上限——脚本每个 epoch 才打一行，poll_interval 设成 1 秒也没有意义。
- **`no_progress_kill_after` 默认 `null` 是刻意的**。"慢"和"挂"从进程外看不出区别，一个 epoch 要 40 分钟的任务配上 30 分钟的超时会导致 guardian 反复误杀正常训练。除非你清楚自己 epoch 的时长，不要开这个。
- **`checkpoint.keep_recent` 不要设成 0**。guardian 在进程外不知道训练脚本下一次 resume 会读哪个文件，清理掉最近的 checkpoint 可能让恢复失败。
- **`watchdog.save_every` 不是 guardian 的参数**。checkpoint 保存频率由训练脚本自己决定，但它直接决定重启式干预的算力代价（见 [cp_3.md](cp_3.md)）——间隔越大，每次干预作废的 epoch 越多。
- **`agent.decision_timeout` 在两种模式下的含义不同**。sidecar 下这段等待不占训练时间；嵌入模式下它就是卡住训练主循环的时长。同一个数字，两种代价。

---

## ✅ 快速校验

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| 最小配置可用 | 只写本文档"v0 最小配置"两段 | `guardian watch` 正常启动并守护 |
| 全默认值可用 | 删除 guardian.yaml | 用内置默认值启动，打印提示而非报错 |
| 优先级正确 | 同一键在文件/环境变量/命令行都设值 | 命令行生效 |
| 环境变量层级 | `GUARDIAN_WATCHDOG__MAX_RETRIES=5` | `watchdog.max_retries` 变为 5 |
| secrets 不落盘 | 在 yaml 里直接写 api_key 字面值 | 明确拒绝并提示改用 `api_key_env` |
| 未知键提示 | 写一个拼错的键名 | 启动时 warning 列出未识别的键，不静默忽略 |
| 契约文件缺失 | 删除 contract.yaml | 四项契约全判为缺失，对应能力关闭，进程级看护仍可用 |
