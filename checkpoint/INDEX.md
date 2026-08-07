# Checkpoint 设计文档索引

> 训练守护系统（Training Guardian Agent）的模块化设计文档。
> 编号 cp_1 ~ cp_16 对应各个功能模块。

| 编号 | 模块 | 实现文件 | 设计文档 | 状态 |
|------|------|----------|----------|------|
| cp_1 | 资源预估 | `guardian/resource_estimator.py` | 待补写 | ✅ 已实现 |
| cp_2 | 训练监控 | `guardian/monitor.py` | 待补写 | ✅ 已实现 |
| cp_3 | 进程守护与恢复 | `guardian/watchdog.py` | 待补写 | ✅ 已实现 |
| cp_4 | 断点分析 | `guardian/checkpoint_analyzer.py` | 待补写 | ✅ 已实现 |
| cp_5 | 日志摘要 | `guardian/summary.py` | 待补写 | ✅ 已实现 |
| cp_6 | 告警推送 | `guardian/notifier.py` | 待补写 | ✅ 已实现 |
| cp_7 | 参考训练脚本 | `train.py` | 待补写 | ✅ 已实现 |
| cp_8 | CLI 入口 | `run.py` | 待补写 | ✅ 已实现 |
| cp_9 | Agent 决策封装 | `guardian/agent_advisor.py` | 待补写 | ✅ 已实现 |
| cp_10 | MCP 工具层 | `guardian/mcp_server.py` | [cp_10.md](cp_10.md) | ✅ 已实现 |
| cp_11 | 任务契约 | `guardian/task_contract.py` | 待补写 | ✅ 已实现 |
| cp_12 | 故障注入测试 | `tests/faultbench/` | 待补写 | ✅ 已实现 |
| cp_13 | 图片筛选 | `guardian/gallery.py` | 待补写 | ✅ 已实现 |
| cp_14 | 实验查询 | `guardian/experiment_query.py` | 待补写 | ✅ 已实现 |
| cp_15 | 模型可视化 | `guardian/model_viz.py` | 待补写 | ✅ 已实现 |
| cp_16 | 推理运行器 | `guardian/inference.py` | 待补写 | ✅ 已实现 |

## 辅助模块

| 文件 | 用途 |
|------|------|
| `guardian/config.py` | 分层配置加载（DEFAULTS < YAML < ENV < CLI） |
| `guardian/credentials.py` | 凭据加载（`~/.guardian-credentials.json`） |
| `guardian/project_context.py` | 项目上下文（`.guardian-project.yaml`） |
| `guardian/component_library.py` | 经典组件库（11 个优化组件） |
| `guardian/dashboard/server.py` | Dashboard HTTP + WebSocket 服务 |
| `guardian/dashboard/static/index.html` | Dashboard 前端 SPA |
| `guardian/streamlit_app.py` | 图片筛选 Streamlit 展示 |
| `configs/guardian.yaml` | guardian 自身配置 |
| `configs/contract.yaml` | 训练脚本契约声明 |
| `scripts/` | 推理脚本（分类/检测/分割） |

## 顶层文档

| 文件 | 内容 |
|------|------|
| [ARCHITECTURE.md](../ARCHITECTURE.md) | 架构与工作流 |
| [DEPLOYMENT.md](../DEPLOYMENT.md) | 使用说明书 |
| [MCP.md](../MCP.md) | MCP 接入说明书 |
| [MCP_API_REFERENCE.md](../MCP_API_REFERENCE.md) | MCP API 参考 |
| [MCP_QUICKSTART.md](../MCP_QUICKSTART.md) | MCP 5 分钟快速接入 |
| [IMPLEMENTATION_REPORT.md](../IMPLEMENTATION_REPORT.md) | 实现对照报告 |
