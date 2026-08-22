# Guardian MCP — 5 分钟快速接入

## 第一步：安装依赖

```bash
cd C:\Users\wst\Desktop\anytries\guarftrain
pip install -r requirements-mcp.txt
```

验证：

```bash
python -c "import mcp; print('OK')"
```

## 第二步：启动 MCP 服务

根据你的场景三选一：

### A. 训练时同进程启动（实时性最好）

```bash
python run.py watch --with-mcp -- python train.py --epochs 20
```

### B. 独立 stdio 进程（Claude Code 子进程接入）

```bash
python run.py serve --transport stdio
```

### C. 一键启动 Dashboard + MCP + 浏览器自动打开

```bash
python run.py start
```

## 第三步：配置 Claude Code

### 本机 stdio 接入

在 `~/.claude/mcp.json` 或项目 `.claude/settings.local.json` 中添加：

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

### 远程 SSH 隧道接入

**远程服务器：**
```bash
python run.py serve --transport sse --port 8766
```

**本机：**
```bash
ssh -L 8766:127.0.0.1:8766 user@your-server
```

**本机配置：**
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

## 第四步：验证

在 Claude Code 对话中：

```
查看当前训练状态
```

Claude Code 会自动调用 `get_training_status` 工具并返回结果。

或手动列出所有工具：

```
请列出 guardian 的所有可用工具
```

## 开启写工具（可选）

如果需要从 Claude Code 控制训练（重启、调参、停止等）：

**1. 修改 `configs/guardian.yaml`：**

```yaml
mcp:
  enabled: true
  enable_write_tools: true
```

**2. 设置口令：**

```bash
export GUARDIAN_MCP_TOKEN=your-secret-token
```

**3. 在 Claude Code 中调用写工具时传入 token：**

```
请用 GUARDIAN_MCP_TOKEN 的 token 值调用 restart_with_params，把 batch_size 减半
```

## 常用对话示例

### 查看训练

```
现在训练到多少 epoch 了？loss 和 accuracy 是多少？
GPU 利用率怎么样？
```

### 分析问题

```
训练过程中出现了哪些异常？
agent 做了哪些决策？有没有被降级的情况？
列出所有 checkpoint，哪个最好？
```

### 控制训练（需 write_token）

```
把 learning rate 降到一半重启训练
触发一次恢复流程，回到最近 checkpoint
训练跑完了，对 checkpoint 17 跑一下推理
```

### 查询实验

```
列出所有历史实验
上次准确率最高的实验，lr 和 batch_size 是多少？
对比 exp_a 和 exp_b 的指标差异
```

### 模型分析

```
分析一下模型的 FLOPs 分布，哪些层是瓶颈？
生成模型结构可视化 HTML
```

### 数据导入

```
这份 WandB 数据的格式是什么？帮我采样前 20 行看看
把 ./logs/wandb_export.jsonl 导入到 Guardian Dashboard
```

## 下一步

- 完整工具列表 → [docs/MCP.md](MCP.md)
- 详细参数说明 → [docs/MCP_API_REFERENCE.md](MCP_API_REFERENCE.md)
- 安全配置 → [docs/MCP.md § 10](MCP.md#10-安全说明)
- 故障排查 → [docs/MCP.md § 13](MCP.md#13-故障排查)
