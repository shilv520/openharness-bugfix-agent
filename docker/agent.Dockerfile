FROM python:3.11-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY agent/ ./agent/
COPY mcp_server/ ./mcp_server/
COPY graph/ ./graph/
COPY data/ ./data/
COPY .env .env

# 启动 Bug Fix Agent Worker
WORKDIR /app
CMD ["python", "-m", "agent.bugfix_worker", "dev"]
