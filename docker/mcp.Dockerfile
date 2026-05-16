FROM python:3.11-slim

WORKDIR /app

# 安装项目依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY agent/ ./agent/
COPY mcp_server/ ./mcp_server/
COPY data/ ./data/
COPY .env .env

# MCP Server 以 stdio 模式运行
CMD ["python", "-m", "mcp_server.mcp_server", "--stdio"]
