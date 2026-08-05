# 依赖清单 — Training Guardian Agent

依赖拆成两份文件，对应两种部署形态：**轻量远程部署**（核心训练 + 规则引擎 + 可选的 agent 决策层）和 **MCP 接入叠加层**（供 Claude Code / OpenClaw 等外部 agent 客户端连接）。二者物理隔离，装 `requirements-core.txt` 时机器上不会出现任何 MCP 相关的包，不会因为"以后可能想接 Claude Code"而让远程训练机的安装变重、攻击面变大。

---

## requirements-core.txt — 轻量远程部署（必需 + 训练侧可选项）

安装: `pip install -r requirements-core.txt`

```
# === 核心依赖（必需） ===
torch>=2.0
torchvision>=0.15
pyyaml>=6.0

# === 硬件监控（可选，无 GPU 时自动降级） ===
psutil>=5.9
GPUtil>=1.4

# === 告警推送（可选，仅终端输出无需） ===
requests>=2.28

# === Agent 决策层（可选，未配置 API key 时自动降级为纯规则）===
anthropic>=0.40         # 或 openai>=1.50，二者择一，取决于 configs/guardian.yaml 的 agent.provider

# === 开发/测试 ===
numpy>=1.24
```

`anthropic`/`openai` 放在这份核心文件里，因为 cp_9 的 agent 决策层是"独立 agent 功能"的核心——即使从来没有任何外部 MCP 客户端连接，训练本身也能用 LLM 做异常应对/恢复策略判断。这份文件装完，`run.py watch --agent -- python train.py` 就是全功能的，只是不能被外部 agent 接入查看/操作。

---

## requirements-mcp.txt — MCP 接入叠加层（可选）

安装: `pip install -r requirements-mcp.txt`（在 core 装完之后，任何时候都可以补装，不需要重启训练）

```
mcp>=1.0
```

只有当你想让 Claude Code / OpenClaw 等外部 MCP 客户端连接查看/操作正在跑的训练（cp_10）时才需要装这份。**不装这份文件，guardian 的其余全部能力（预检/监控/恢复/摘要/agent 决策）都不受影响**——`--with-mcp` 或 `run.py serve` 检测到 `mcp` 包不存在时，只打印一条警告并跳过 MCP 启动，不会报错退出，详见 [cp_8.md](cp_8.md) 和 [cp_10.md](cp_10.md)。

---

## 推荐的部署时序

```
# 1. 远程机器上先用最轻量的方式起训练（sidecar 守护，训练脚本 0 行改动）
pip install -r requirements-core.txt
python run.py contract check                                   # 先确认脚本契约四项
python run.py watch --agent -- python train.py --epochs 20

# 2. 过一阵子想让 Claude Code 看一眼训练状态了，再单独补装、单独起进程
pip install -r requirements-mcp.txt
python run.py serve --transport stdio      # 读盘接入已经在跑的训练，无需重启
```

"要不要能被外部 agent 接入"不是部署时刻就要决定的事，而是同一次部署里可以随时叠加的选项。
