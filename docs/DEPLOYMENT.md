# Training Guardian · 部署指南

## 目录

- [部署模式总览](#部署模式总览)
- [模式 A：本地开发（WSL + dsh-wsl）](#模式-a本地开发wsl--dsh-wsl)
- [模式 B：Docker 容器](#模式-bdocker-容器)
- [模式 C：云端服务器（systemd + SSH tunnel）](#模式-c云端服务器systemd--ssh-tunnel)
- [模式 D：远程 Dashboard（PC 浏览器连接算力服务器）](#模式-d远程-dashboardpc-浏览器连接算力服务器)
- [网络拓扑](#网络拓扑)
- [故障排查](#故障排查)

---

## 部署模式总览

| 模式 | 适用场景 | 复杂度 |
|------|----------|--------|
| **A. WSL 本地开发** | 在 Windows 上跑 dsh-wsl + 插件 | ⭐ |
| **B. Docker** | 单容器打包 DSH + guardian server | ⭐⭐ |
| **C. 云端服务器** | 算力服务器上跑 guardian + systemd | ⭐⭐⭐ |
| **D. 远程 Dashboard** | PC 浏览器连接远程 guardian | ⭐⭐⭐ |

---

## 模式 A：本地开发（WSL + dsh-wsl）

### 前提

- Windows 11 + WSL2（推荐 Ubuntu 22.04+）
- Node.js ≥ 22（WSL 内）
- pnpm ≥ 9
- Python ≥ 3.10（WSL 内）

### 步骤

```bash
# 1. 克隆 dsh-wsl
git clone https://github.com/DeepSeek-ai/dsh-wsl.git ~/dsh-wsl
cd ~/dsh-wsl
pnpm install
pnpm build

# 2. 安装 Training Guardian 插件（本地开发模式，用 file: 链接）
pnpm dsh plugin --profile web add /path/to/guarftrain/dsh-plugin/dsh-client-ui-training-guardian

# 3. 启动 DSH web
pnpm dsh web
# → http://127.0.0.1:3080
```

### 验证

1. Windows 浏览器打开 http://127.0.0.1:3080
2. 登录 DSH 后进入任意会话
3. 会话头部应看到 **"Training Guardian"** 按钮
4. 点击按钮弹出监控面板（5 个 tab）

### 插件热更新

插件源码改完后重新 build：

```bash
cd /path/to/guarftrain/dsh-plugin/dsh-client-ui-training-guardian
pnpm build
# dsh-wsl 的 dev 模式会自动 reload client bundle
```

---

## 模式 B：Docker

### Dockerfile

```dockerfile
# 多阶段构建
FROM node:22-slim AS dsh-builder
WORKDIR /dsh-wsl
RUN git clone https://github.com/DeepSeek-ai/dsh-wsl.git . \
    && pnpm install \
    && pnpm build

FROM python:3.11-slim AS guardian-runtime
RUN pip install guarftrain[full]

FROM node:22-slim
WORKDIR /app

# 层 1: DSH web
COPY --from=dsh-builder /dsh-wsl /dsh-wsl
WORKDIR /dsh-wsl

# 安装插件
COPY dsh-plugin/package.json dsh-plugin/
COPY dsh-plugin/cordis.patch.yml dsh-plugin/
RUN pnpm add ./dsh-plugin

# 层 2: guardian runtime
COPY --from=guardian-runtime /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY guardian/ /guardian/

EXPOSE 3080 8765

CMD ["pnpm", "dsh", "web"]
```

### 运行

```bash
docker build -t guarftrain-dsh .
docker run -p 3080:3080 -p 8765:8765 guarftrain-dsh
```

### docker-compose（推荐）

```yaml
version: '3.8'
services:
  dsh-web:
    build: .
    ports:
      - "3080:3080"
    environment:
      - DSH_PROFILE=web
    volumes:
      - dsh-data:/root/.dsh
    restart: unless-stopped

volumes:
  dsh-data:
```

---

## 模式 C：云端服务器（systemd + SSH tunnel）

适用：训练跑在远程 Linux 服务器，用 Windows/Mac 浏览器远程查看。

### 1. 服务器端：安装 DSH + 插件

```bash
# 在算力服务器上
ssh gpu-server

# 安装 Node.js 22
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs

# 安装 pnpm
corepack enable && corepack prepare pnpm@11.22.0 --activate

# 克隆 dsh-wsl
git clone https://github.com/DeepSeek-ai/dsh-wsl.git ~/dsh-wsl
cd ~/dsh-wsl && pnpm install && pnpm build

# 安装插件
pnpm dsh plugin --profile web add @linxin666/dsh-client-ui-training-guardian
```

### 2. 配置 Guardian RemoteServer

```bash
# 安装 guarftrain
pip install guarftrain[full]

# 启动训练时开启 remote
guarftrain watch --remote --remote-auth <your-token> -- python train.py
```

### 3. 创建 systemd 服务

新建 `/etc/systemd/system/guarftrain-dsh.service`：

```ini
[Unit]
Description=Guarftrain DSH Web UI
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/dsh-wsl
ExecStart=/usr/bin/pnpm dsh web
Restart=always
RestartSec=5
Environment=DSH_PROFILE=web
Environment=NODE_ENV=production

# 安全加固
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
ProtectHome=read-only

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now guarftrain-dsh
sudo systemctl status guarftrain-dsh
```

### 4. 防火墙配置

```bash
# 仅允许特定 IP 访问（替换 <YOUR_IP>）
sudo ufw allow from <YOUR_IP> to any port 3080
sudo ufw allow from <YOUR_IP> to any port 8765

# 或走 SSH tunnel（更安全，不需要开放端口）
```

### 5. SSH Tunnel（推荐，无需开放端口）

在本地 Windows 上：

```powershell
# PowerShell: 隧道 3080
ssh -L 3080:localhost:3080 ubuntu@gpu-server

# 浏览器访问 http://localhost:3080
```

或用 `autossh` 保持持久隧道：

```bash
sudo apt install autossh
sudo systemctl enable --now autossh@dsh-tunnel
```

---

## 模式 D：远程 Dashboard（PC 浏览器连接算力服务器）

当训练跑在远程服务器时，不需要跑完整 DSH，只需要 guardian server + 插件通过浏览器远程访问。

### 架构

```
┌──────────────┐     HTTPS/SSE      ┌──────────────┐     SSH tunnel     ┌──────────────┐
│  PC 浏览器    │ ◄─────────────────► │ 反向代理      │ ◄───────────────► │ 算力服务器     │
│  localhost:3080│   (nginx/Caddy)    │  443/80       │                   │  guardian:8765 │
└──────────────┘                     └──────────────┘                   └──────────────┘
```

### 方案 1：SSH Tunnel（最简单）

```powershell
# Windows PowerShell — 一条命令
ssh -L 3080:localhost:3080 -L 8765:localhost:8765 ubuntu@gpu-server -N
```

然后：
- DSH Web → http://localhost:3080
- Guardian SSE → ws://localhost:8765（插件自动走 localhost）

### 方案 2：Caddy 反向代理（需要域名）

```bash
# 在算力服务器上安装 Caddy
sudo apt install caddy

# /etc/caddy/Caddyfile
gpu.example.com {
    reverse_proxy localhost:3080
    # TLS 自动由 Let's Encrypt 提供
}

sudo systemctl reload caddy
```

PC 浏览器直接访问 https://gpu.example.com

### 方案 3：Nginx + 自签名证书（内网）

```nginx
# /etc/nginx/sites-available/guarftrain
server {
    listen 443 ssl;
    server_name gpu.internal;

    ssl_certificate     /etc/ssl/guarftrain.crt;
    ssl_certificate_key /etc/ssl/guarftrain.key;

    location / {
        proxy_pass http://localhost:3080;
        proxy_set_header Host $host;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 网络拓扑

```
训练场景 1 — 全部本地
────────────────────
Windows ←WSL mirrored→ WSL(dsh-wsl:3080 + guardian:8765)

训练场景 2 — 训练在服务器，监控在 PC
────────────────────────────────────
gpu-server                PC
├── python train.py       ├── browser (localhost:3080 via SSH tunnel)
├── guardian:8765         │
└── dsh-wsl:3080          │
                          └── SSH tunnel: 3080, 8765

训练场景 3 — 云端部署 + 域名
────────────────────────────
internet ──HTTPS──► Caddy/nginx ──HTTP──► dsh-wsl:3080
                              └──HTTP──► guardian:8765
```

---

## 故障排查

### `pnpm dsh web` 报 "pnpm not found"

```bash
# WSL 里
corepack enable
corepack prepare pnpm@11.22.0 --activate
```

### 浏览器访问 3080 空白页

1. 确认 DSH web 确实在监听：`curl http://127.0.0.1:3080`
2. 确认插件已安装：`pnpm dsh plugin --profile web list`
3. 确认 boot manifest 包含插件：看启动日志里的 bundle 列表
4. 如果是远程访问，检查 tunnel/proxy 是否转发

### SSE 连接失败（401 Unauthorized）

1. 确认插件设置里的 **Auth Token** 和 `--remote-auth` 参数一致
2. 确认 guardian server 的 `--remote` 标志已启用
3. 查看插件设置 → 确认 Server URL 正确（远程场景用公网 IP 或 tunnel 地址）

### Guardian server 端口冲突

```bash
# 换端口
guarftrain watch --remote --remote-port 8766 -- python train.py
# 插件设置里同步改 Server URL
```
