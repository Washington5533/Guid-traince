# cp_6 · 告警推送 (Notifier)

**文件**: `guardian/notifier.py`
**阶段**: 全程
**核心目标**: 统一告警推送，支持终端/Webhook/邮件，带静默期防刷屏

> **架构中立**：本模块在 guardian 进程内运行，只被其他模块调用去发消息，不读训练状态也不干预训练。sidecar 与嵌入模式下行为完全一致，无需区分。唯一相关的一点：sidecar 下告警文本里的干预动作应写明是重启式的（如"已降 lr 至 5e-4 并从 cp_10 重启，作废 2 epoch"），让收到推送的人知道付出了什么代价。

---

## 关键类与方法

### `Notifier`

| 方法 | 说明 |
|------|------|
| `__init__(config)` | 从 config.notifier 读取推送渠道配置和 cooldown |
| `send(title, message, alert_type, level)` | 主入口：终端输出 + 按条件推送远程渠道 |
| `_should_send(alert_type)` | 静默期检查：同类告警在 cooldown 内不重复推送 |
| `_print_terminal(title, message, level)` | 终端格式化输出（带图标和边框） |
| `_send_webhook(title, message, level)` | Webhook HTTP POST 推送 |
| `_send_email(title, message, level)` | SMTP 邮件推送 |

---

## 推送渠道

### 终端（始终启用）
```
⚠️ [WARNING] Loss 突增
╔══════════════════════════════════════════════════╗
║ Loss 突增 +42%，当前 0.38，窗口均值 0.27        ║
║ Epoch: 23 | Step: 4520                          ║
║ GPU: RTX4090 | 温度 82°C | 显存 11.2/24GB       ║
║ 应对: agent 决策 → 降 lr 至 5e-4 并从 cp_22 重启 ║
║ 代价: 作废约 1 epoch                            ║
╚══════════════════════════════════════════════════╝
```

干预类告警必须同时说明"做了什么"和"代价是什么"——sidecar 下所有参数干预都是重启式的（见 [cp_3.md](cp_3.md)），收到推送的人应当能立刻判断这次干预是否划算，而不是只看到"已自动处理"。

### Webhook
```json
{
  "title": "Loss 突增",
  "message": "Loss 突增 +42%...",
  "level": "warning",
  "timestamp": 1754303422,
  "response": {
    "source": "agent",
    "action": "restart_with_lower_lr(0.5)",
    "restart": true,
    "resumed_from": "checkpoints/cp_22",
    "wasted_epochs": 1
  }
}
```

`response` 字段在纯告警（`alert_only`）时为 `{"source": "...", "action": "alert_only", "restart": false}`，字段结构保持一致，方便下游 webhook 消费方统一解析。

### 邮件
```
Subject: [Guardian WARNING] Loss 突增
Body: 完整告警信息
```

---

## 静默期机制

```
alert_type: "loss_spike"
cooldown: 300 秒

第一次触发 → 立即推送 → 记录时间戳
第二次触发 (+30s)  → 静默，跳过
第三次触发 (+310s) → 推送（超过 cooldown）
```

---

## ✅ 快速校验

| 检查项 | 验证方式 | 预期结果 |
|--------|----------|----------|
| 类可实例化 | `Notifier(config)` | 无异常 |
| 终端输出 | `send("test", "msg")` | 终端显示格式化告警 |
| 静默期 | 连续 send 3 次同类型 | 仅第 1 次推送远程 |
| 不同类不静默 | send("a", ...) + send("b", ...) | 两次均推送 |

---

## ✅ 完整校验

| 检查项 | 验证方式 | 通过标准 |
|--------|----------|----------|
| Webhook 投递 | 配置 httpbin.org 接收 | 收到正确 JSON payload |
| Webhook 超时 | URL 为不可达地址 | 10s 超时，不阻塞 guardian 看护循环，更不影响训练子进程 |
| 邮件发送 | 配置测试 SMTP | 收件箱收到正确主题和内容 |
| 邮件认证失败 | 错误密码 | 捕获异常，输出 warning，不崩溃 |
| disabled 模式 | `enabled: false` | 仅终端输出，不推送任何远程 |
| level 图标 | info/warning/error | 终端显示不同图标 |
| 消息过长 | message 超过 500 字 | 终端换行正确，Webhook 不截断 |
| 干预代价可见 | 一次重启式干预触发的告警 | 终端与 webhook 均含 action / resumed_from / wasted_epochs |
| response 结构一致 | 一条 alert_only + 一条重启式干预 | 两者 `response` 字段结构相同，仅 `restart` 布尔值与附加字段不同 |
