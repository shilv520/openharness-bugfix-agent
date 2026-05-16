@echo off
REM Redis + ChromaDB 启动脚本 (Windows)
REM =========================

echo ============================================
echo Bug Fix Agent - Redis + ChromaDB 启动
echo ============================================

REM 检查 Docker 是否运行
docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker 未运行，请先启动 Docker Desktop
    exit /b 1
)

echo.
echo 1. 启动 Redis...
docker run -d --name bugfix-redis -p 6379:6379 -v bugfix_redis_data:/data redis:7-alpine redis-server --appendonly yes

echo.
echo 2. 启动 ChromaDB...
docker run -d --name bugfix-chromadb -p 8001:8000 -v bugfix_chroma_data:/chroma/chroma -e IS_PERSISTENT=TRUE -e ANONYMIZED_TELEMETRY=FALSE chromadb/chroma:latest

echo.
echo 3. 等待服务启动...
timeout /t 5 /nobreak >nul

echo.
echo 4. 检查服务状态...
echo.
echo Redis:
docker ps --filter name=bugfix-redis --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo ChromaDB:
docker ps --filter name=bugfix-chromadb --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo.
echo ============================================
echo 服务已启动!
echo ============================================
echo.
echo 连接信息:
echo   Redis: localhost:6379
echo   ChromaDB: localhost:8001
echo.
echo 测试命令:
echo   python data/benchmark/test_redis_chroma_integration.py
echo.
echo 停止命令:
echo   docker stop bugfix-redis bugfix-chromadb
echo   docker rm bugfix-redis bugfix-chromadb

pause