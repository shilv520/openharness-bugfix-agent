FROM python:3.11-slim

WORKDIR /app

# 安装项目依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码（保持目录结构以匹配 sys.path）
COPY agent/ ./agent/
COPY mcp_server/ ./mcp_server/
COPY graph/ ./graph/
COPY .env .env

# API 入口保持相对路径与本地一致: docker/api/main.py
COPY docker/api/main.py ./docker/api/main.py

# 复制前端构建产物（SPA 服务）
COPY frontend/dist/ ./frontend/dist/

EXPOSE 8000

CMD ["uvicorn", "docker.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
