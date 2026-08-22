@echo off
REM === DeepSeek Dashboard 启动 ===
REM 请先设置环境变量: set OPENAI_API_KEY=your-key
REM 或通过 .guardian-credentials.json 配置
set GUARDIAN_AI_PROVIDER=openai
set GUARDIAN_AI_MODEL=deepseek-chat
set OPENAI_BASE_URL=https://api.deepseek.com
python run.py dashboard
