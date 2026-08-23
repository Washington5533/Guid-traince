"""Sub-agent System Prompts。

三个核心场景的 prompt 模板：
1. anomaly_response — 异常检测后的应对决策
2. crash_recovery  — 崩溃后的恢复策略
3. training_summary — 训练结束后的 AI 解读
"""

from __future__ import annotations

import json

__all__ = [
    "SYSTEM_ANOMALY_RESPONSE",
    "SYSTEM_CRASH_RECOVERY",
    "SYSTEM_TRAINING_SUMMARY",
    "build_anomaly_prompt",
    "build_crash_prompt",
    "build_summary_prompt",
]

SYSTEM_ANOMALY_RESPONSE = (
    "你是一个训练守护智能体 (Training Guardian Sub-Agent)。"
    "你运行在算力服务器上，独立于任何外部 agent 平台。"
    "训练监控系统已通过规则引擎确认异常存在——你需要选择'怎么应对'。\n\n"
    "规则：\n"
    "1. 你必须从提供的 action_space 中选择一个动作，不能自己发明\n"
    "2. 如果动作带有参数（如 ratio），参数必须在声明的范围内\n"
    "3. 考虑训练阶段：早期(前10%)应更激进，后期(后20%)应更保守\n"
    "4. 考虑近期决策历史：同一问题反复出现时尝试不同策略\n"
    "5. 返回 JSON 格式：{\"action\": \"动作名\", ...参数} 或纯字符串动作名\n"
    "6. 只返回 JSON 或动作名，不要任何解释文字\n"
    "7. 如果无法确定，选择最保守的动作（通常是 ignore 或 alert）"
)

SYSTEM_CRASH_RECOVERY = (
    "你是一个训练守护智能体。训练子进程已崩溃，你需要决定如何恢复。\n\n"
    "崩溃分类：\n"
    "- OOM: 显存不足，可通过减 batch / 梯度累积解决\n"
    "- sigkill: 外部终止，通常可从 checkpoint 续训\n"
    "- code_error: 代码错误（如 TypeError），无法自动恢复\n"
    "- unknown: 未知原因，保守处理\n\n"
    "规则：\n"
    "1. code_error 类型只能选择 stop_training（无法自动修复代码错误）\n"
    "2. OOM 优先选择 reduce_batch 或 enable_grad_accum\n"
    "3. sigkill 优先选择 resume_unchanged（不是代码问题）\n"
    "4. 连续崩溃 3 次以上应建议 stop_training\n"
    "5. 返回 JSON 格式：{\"action\": \"动作名\", ...参数}"
)

SYSTEM_TRAINING_SUMMARY = (
    "你是一个训练分析师。根据结构化训练摘要，用中文生成一段简洁的"
    "自然语言总结（200-400 字）。内容包括：\n"
    "1. 训练是否顺利完成、总耗时、总 epoch 数\n"
    "2. 关键指标（最佳 loss/accuracy、最终指标）\n"
    "3. 异常事件与处理结果（告警/干预/忽略各几次）\n"
    "4. 崩溃与恢复情况\n"
    "5. GPU 资源使用概况\n"
    "6. 整体评价和改进建议\n"
    "不要编造数据，只基于提供的摘要信息。"
)

SYSTEM_HEALTH_CHECK = (
    "你是一个训练健康评估专家。根据当前训练状态，判断训练是否健康运行。\n\n"
    "健康指标：\n"
    "- loss 是否持续下降（允许小幅波动）\n"
    "- GPU 利用率是否合理（>30%）\n"
    "- GPU 温度是否安全（<85°C）\n"
    "- 是否有异常堆积（同一异常 5 分钟内出现 3 次以上）\n\n"
    "返回 JSON：{\"healthy\": true/false, \"concerns\": [\"...\"], \"suggestions\": [\"...\"]}"
)


def build_anomaly_prompt(context: dict, memory_context: str, action_space: list) -> str:
    """构造异常应对决策的 user message。"""
    lines = [
        f"决策点: {context.get('decision_point', 'monitor_response')}",
        f"异常类型: {context.get('anomaly_type', 'unknown')}",
        f"异常描述: {context.get('description', '')}",
        f"严重程度: {context.get('severity', 'medium')}",
        "",
        "训练上下文:",
        f"  当前指标: {json.dumps(context.get('current_metrics', {}), ensure_ascii=False)}",
        f"  训练阶段: {context.get('training_phase', 'unknown')}",
        "",
        "近期记忆:",
        f"  {memory_context}",
        "",
        "可选动作（必须从以下列表中选一个）：",
    ]
    for item in action_space:
        if isinstance(item, str):
            lines.append(f'  - "{item}"')
        elif isinstance(item, dict):
            lines.append(f"  - {json.dumps(item, ensure_ascii=False)}")
    lines.append("")
    lines.append("请返回你的选择：")
    return "\n".join(lines)


def build_crash_prompt(crash_info: dict, memory_context: str, action_space: list) -> str:
    """构造崩溃恢复决策的 user message。"""
    lines = [
        f"崩溃类型: {crash_info.get('crash_type', 'unknown')}",
        f"退出码: {crash_info.get('exit_code', 'N/A')}",
        f"错误信息: {crash_info.get('stderr', '')[:500]}",
        f"已重启次数: {crash_info.get('restart_count', 0)}",
        f"作废 epoch 数: {crash_info.get('wasted_epochs', 0)}",
        "",
        "近期记忆:",
        f"  {memory_context}",
        "",
        "可选动作（必须从以下列表中选一个）：",
    ]
    for item in action_space:
        if isinstance(item, str):
            lines.append(f'  - "{item}"')
        elif isinstance(item, dict):
            lines.append(f"  - {json.dumps(item, ensure_ascii=False)}")
    lines.append("")
    lines.append("请返回你的选择：")
    return "\n".join(lines)


def build_summary_prompt(summary: dict) -> str:
    """构造训练总结的 user message。"""
    return f"请基于以下训练摘要生成自然语言解读：\n\n{json.dumps(summary, ensure_ascii=False, indent=2)}"


def build_health_prompt(status: dict, gpu_status: dict) -> str:
    """构造健康检查的 user message。"""
    return (
        f"训练状态:\n"
        f"  epoch: {status.get('epoch', '?')}/{status.get('total_epochs', '?')}\n"
        f"  loss: {status.get('loss', '?')}\n"
        f"  status: {status.get('status', '?')}\n"
        f"\nGPU 状态:\n"
        f"  {json.dumps(gpu_status, ensure_ascii=False)}\n"
        f"\n请评估训练健康状态（JSON 格式）。"
    )
