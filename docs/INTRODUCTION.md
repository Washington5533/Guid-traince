# Training Guardian Agent · 训练守护智能体

## 一句话介绍

**一行命令、训练脚本零改动，即可获得覆盖训练前、训练中、训练后全流程的 AI 守护能力**——崩了自动续训、异常实时告警、决策交给 LLM，让深度学习训练真正可以"挂机"。

## 项目背景

深度学习训练长期处于“人肉盯盘”状态：一次实验动辄数小时到数天，研究者必须守在机器旁。Loss 爆炸、NaN、OOM、崩溃、GPU 过热随时可能发生，参数凭经验盲调浪费算力，实验日志与检查点散落各处无法复盘。而 TensorBoard、W&B 等现有工具只做“可视化”，看得见问题却不会处理问题。本项目要把“盯盘 + 救火 + 复盘”三件事全部自动化。

## 解决的问题与核心功能

### 解决的问题

> 训练过程无人值守时的故障发现慢、恢复靠人工、经验无法沉淀。

### 核心功能（按训练生命周期）

| 阶段 | 能力 | 命令 |
|------|------|------|
| **训练前** | GPU 显存预估 + batch size 推荐，避免开局即 OOM | `guarftrain preflight` |
| **训练中** | GPU + Loss 实时监控、异常检测告警、崩溃自动续训、OOM 恢复、LLM 智能决策、Sub-agent 自主干预 | `guarftrain watch` |
| **训练后** | 训练摘要 + AI 解读、Checkpoint 对比分析、模型结构可视化、架构分析（FLOPs/瓶颈层） | `guarftrain summarize` |
| **跨实验** | 自然语言查询（"最好的学习率是哪个？"）、实验对比、外部数据导入 | `guarftrain query` |
| **外部接入** | MCP 协议 36 工具（25 只读 + 11 写）、Dashboard 远程配置、DSH Web GUI 插件 | `guarftrain start` |

完整覆盖 **12 类故障场景**：Loss 异常、NaN 回滚、GPU 过热、OOM、进程崩溃、训练停滞等。

### 技术方案

采用 Sidecar 边车架构：守护进程与训练进程完全解耦，通过日志尾部解析 + GPU 轮询感知训练状态，训练脚本一行不用改。内部由 Watchdog（拉起进程、崩溃自动续训、参数重写重启）、Monitor（异常检测）、AgentAdvisor（LLM 决策）、Remote 远程服务、MCP Server（36 工具）与 Dashboard 面板组成。其上构建五层决策架构 Contract→Agent→Sub-agent→Rules→MCP/Dashboard，权责清晰、逐层降级，写操作带 token 鉴权与训练阶段门控。另设四项契约渐进增强：脚本每满足一项即解锁对应能力，全不满足也能正常守护；核心依赖仅 2MB，无 GPU 自动降级运行。

## 创新点

1. **零侵入守护**：不改一行训练代码即获得崩溃续训 + 异常恢复，感知层只依赖日志和 GPU 指标；
2. **五层决策 + 三级自主度**：从纯规则兜底到 LLM 全自主干预，人类始终握有契约硬边界与审批权；
3. **MCP 原生委托**：36 个标准化工具实现“Agent 指挥 Agent 训练”，写操作带鉴权与门控；
4. **架构级洞察**：forward hook 实测 FLOPs + D3 可视化，结构分析成为守护能力的一部分；
5. **全流程闭环**：一个工具贯通训练前预估、训练中干预、训练后解读与跨实验知识沉淀。

## 应用价值

- **个人研究者**：晚上提交训练放心睡觉，崩溃自动续训、异常第二天看 AI 解读；
- **实验室 / 团队**：共享算力服务器上多任务守护，Dashboard + 远程通信让任何人从自己的电脑查看远端训练状态；
- **AI Agent 生态**：通过 MCP 让 Claude Code 等外部 Agent 直接查询与干预训练，构建自动化实验流水线；
- **成本节约**：显存预估避免 OOM 反复试错，崩溃秒级恢复减少算力空转，跨实验查询让历史实验不再白跑。

## 使用说明

```bash
pip install guarftrain          # 核心 ~2MB，可选组件 [agent]/[mcp]/[dashboard]/[full]
guarftrain init                 # 自动扫描训练脚本，生成契约配置
guarftrain watch -- python train.py   # 纯规则守护，零外部依赖
```

watch 追加 `--agent --with-dashboard --with-mcp` 即全功能模式；可选 DSH 面板插件 `@rrrelink/dsh-client-ui-training-guardian`。仓库、PyPI、在线演示链接见 README。要求 Python 3.10+，MIT 许可证。
