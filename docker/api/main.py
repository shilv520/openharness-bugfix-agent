"""
FastAPI 后端服务 - Bug Fix Agent API (v2.0 - Real Multi-Agent)
============================================================

功能：
1. 用户认证（注册/登录）
2. Bug代码分析 - 调用真实 Multi-Agent 协同系统 (Reviewer+Analyzer+Fixer+Validator)
3. 完整Bug修复流程 - 6步协同管道
4. 修复历史记录查询
"""

import asyncio
import json
import hashlib
import mimetypes
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List

import jwt

# 修复 Windows 上 .js/.mjs 文件的 MIME 类型（浏览器要求正确的 MIME 才能加载 ES modules）
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

# 添加项目路径以导入 agent 模块
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from mcp_server.mcp_server import BugFixMcpServer
from agent.communication import reset_comm_bus

import redis as redis_lib

# ============================================
# 配置
# ============================================
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

# JWT 认证配置
JWT_SECRET = os.getenv("JWT_SECRET", "bugfix-agent-dev-secret--change-in-production!!")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

security_scheme = HTTPBearer(auto_error=False)

try:
    r = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    r.ping()
except Exception:
    r = None
    _cache: dict = {}

# ============================================
# FastAPI 应用
# ============================================
app = FastAPI(title="Bug Fix Agent API v2.0", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 静态文件服务（前端构建产物）
# ============================================
_frontend_dist = _project_root / "frontend" / "dist"
_frontend_assets = _frontend_dist / "assets"

if _frontend_assets.exists():
    app.mount("/assets", StaticFiles(directory=str(_frontend_assets)), name="assets")


# ============================================
# 数据模型
# ============================================
class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


class BugSubmitRequest(BaseModel):
    code: str
    language: str = "java"
    description: Optional[str] = None
    test_case: Optional[str] = None


class BugCandidate(BaseModel):
    location: str = ""
    type: str = ""
    severity: str = ""
    description: str = ""


class ReviewResult(BaseModel):
    bugs_found: int = 0
    candidates: List[BugCandidate] = []
    code_quality: str = ""


class DiscussionResult(BaseModel):
    agreed: bool = False
    consensus: str = ""


class AnalysisResult(BaseModel):
    bug_location: str = ""
    bug_type: str = ""
    root_cause: str = ""
    fix_suggestion: str = ""
    confidence: float = 0.0


class PipelineStep(BaseModel):
    step: str = ""
    result: Optional[str] = None
    bugs_found: Optional[int] = None
    agreed: Optional[bool] = None
    patch_generated: Optional[bool] = None
    passed: Optional[bool] = None
    root_cause: Optional[str] = None


class BugAnalyzeResponse(BaseModel):
    success: bool = False
    task_id: str = ""
    review: ReviewResult = Field(default_factory=ReviewResult)
    discussion: DiscussionResult = Field(default_factory=DiscussionResult)
    analysis: AnalysisResult = Field(default_factory=AnalysisResult)
    patch: str = ""
    fixed_code: str = ""
    test_passed: bool = False
    total_steps: int = 0
    steps: List[PipelineStep] = []
    error: Optional[str] = None


class BugTaskResponse(BaseModel):
    task_id: str
    username: str
    status: str
    language: str
    code: str = ""
    created_at: str = ""
    bug_location: Optional[str] = None
    bug_type: Optional[str] = None
    root_cause: Optional[str] = None
    fix_suggestion: Optional[str] = None
    patch: Optional[str] = None
    fixed_code: Optional[str] = None
    test_passed: Optional[bool] = None
    confidence: Optional[float] = None
    total_steps: Optional[int] = None
    steps: Optional[List[dict]] = None
    error: Optional[str] = None


# ============================================
# 工具函数
# ============================================
def _hset(key: str, field: str, value: str):
    if r:
        r.hset(key, field, value)
    else:
        _cache.setdefault(key, {})[field] = value


def _hget(key: str, field: str) -> Optional[str]:
    if r:
        return r.hget(key, field)
    return _cache.get(key, {}).get(field)


def _hgetall(key: str) -> dict:
    if r:
        return r.hgetall(key)
    return _cache.get(key, {})


def hash_pwd(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> str:
    """从 Bearer token 中提取当前用户。未登录返回 "anonymous" """
    if credentials is None:
        return "anonymous"
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username:
            return username
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Token 已过期，请重新登录")
    except jwt.InvalidTokenError:
        raise HTTPException(401, "无效的 Token")
    return "anonymous"


async def _execute_multi_agent_analysis(code: str, language: str, username: str = "anonymous", task_id: str = "") -> BugAnalyzeResponse:
    """执行真实 Multi-Agent 协同分析"""
    reset_comm_bus()
    server = BugFixMcpServer()

    try:
        agents = await server._get_agents()

        # 设置分层记忆会话（使 Agent 能访问历史上下文）
        session_id = task_id or f"analyze_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        for agent in agents.values():
            agent.set_memory_session(username, session_id)

        review_result = await agents["reviewer"].execute_action(
            "review_code", {"code": code, "language": language}
        )
        bugs = review_result.get("bug_candidates", [])

        if bugs:
            discussion = await agents["reviewer"].collaborate(
                "Analyzer", "Bug确认",
                {"view": f"发现{len(bugs)}个潜在Bug", "candidates": bugs}
            )
        else:
            discussion = {"agreed": False, "consensus": "未发现Bug"}

        analysis_context = {
            "code": code,
            "review_result": review_result,
            "discussion": discussion
        }
        analysis_result = await agents["analyzer"].execute_action(
            "analyze_bug", analysis_context
        )

        candidates = []
        for b in bugs:
            candidates.append(BugCandidate(
                location=b.get("location", ""),
                type=b.get("type", ""),
                severity=b.get("severity", ""),
                description=b.get("description", "")[:200]
            ))

        return BugAnalyzeResponse(
            success=len(bugs) > 0,
            review=ReviewResult(
                bugs_found=len(bugs),
                candidates=candidates,
                code_quality=review_result.get("code_quality", "unknown")
            ),
            discussion=DiscussionResult(
                agreed=discussion.get("agreed", False),
                consensus=str(discussion.get("consensus", ""))[:500]
            ),
            analysis=AnalysisResult(
                bug_location=analysis_result.get("bug_location", "unknown"),
                bug_type=analysis_result.get("bug_type", "unknown"),
                root_cause=analysis_result.get("root_cause", "unknown"),
                fix_suggestion=analysis_result.get("fix_suggestion", "unknown"),
                confidence=float(analysis_result.get("confidence", 0))
            ),
        )
    except Exception as e:
        return BugAnalyzeResponse(success=False, error=str(e))


async def _execute_full_bugfix(code: str, language: str, username: str = "anonymous", task_id: str = "") -> BugAnalyzeResponse:
    """执行完整 Multi-Agent 协同修复流程"""
    reset_comm_bus()
    server = BugFixMcpServer()

    try:
        agents = await server._get_agents()

        # 设置分层记忆会话
        session_id = task_id or f"fix_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        for agent in agents.values():
            agent.set_memory_session(username, session_id)

        reviewer = agents["reviewer"]
        analyzer = agents["analyzer"]
        fixer = agents["fixer"]
        validator = agents["validator"]

        steps: List[PipelineStep] = []

        # Step 1: Reviewer 审查
        review_result = await reviewer.execute_action(
            "review_code", {"code": code, "language": language}
        )
        steps.append(PipelineStep(
            step="review",
            result="success",
            bugs_found=len(review_result.get("bug_candidates", []))
        ))

        # Step 2: Reviewer-Analyzer 讨论
        bugs = review_result.get("bug_candidates", [])
        if not bugs:
            steps.append(PipelineStep(step="no_bugs_found", result="skip"))
            return BugAnalyzeResponse(
                success=False,
                total_steps=len(steps),
                steps=steps,
                error="No bugs found"
            )

        discussion_ra = await reviewer.collaborate(
            "Analyzer", "Bug确认",
            {"view": "发现Bug", "candidates": bugs}
        )
        steps.append(PipelineStep(
            step="discuss_review_analyzer",
            agreed=discussion_ra.get("agreed", False)
        ))

        # Step 3: Analyzer 分析
        analysis_result = await analyzer.execute_action("analyze_bug", {
            "code": code,
            "review_result": review_result
        })
        steps.append(PipelineStep(
            step="analyze",
            root_cause=analysis_result.get("root_cause", "")[:200]
        ))

        # Step 4: Analyzer-Fixer 讨论
        discussion_af = await analyzer.collaborate(
            "Fixer", "修复策略",
            {"view": f"Bug: {analysis_result.get('bug_type')}", "analysis": analysis_result}
        )
        steps.append(PipelineStep(
            step="discuss_analyzer_fixer",
            agreed=discussion_af.get("agreed", False)
        ))

        # Step 5: Fixer 生成补丁
        fix_result = await fixer.execute_action("generate_patch", {
            "code": code,
            "analysis_result": analysis_result,
            "fix_discussion": discussion_af
        })
        steps.append(PipelineStep(
            step="fix",
            patch_generated=bool(fix_result.get("patch"))
        ))

        # Step 6: Validator 最终验证
        validation_result = await validator.execute_action("validate_fix", {
            "fixed_code": fix_result.get("fixed_code", ""),
            "code": code,
            "patch": fix_result.get("patch", "")
        })
        steps.append(PipelineStep(
            step="validate",
            passed=validation_result.get("test_passed", False)
        ))

        candidates = []
        for b in bugs:
            candidates.append(BugCandidate(
                location=b.get("location", ""),
                type=b.get("type", ""),
                severity=b.get("severity", ""),
                description=b.get("description", "")[:200]
            ))

        return BugAnalyzeResponse(
            success=validation_result.get("test_passed", False),
            review=ReviewResult(
                bugs_found=len(bugs),
                candidates=candidates,
                code_quality=review_result.get("code_quality", "unknown")
            ),
            discussion=DiscussionResult(
                agreed=discussion_af.get("agreed", False),
                consensus=str(discussion_af.get("consensus", ""))[:500]
            ),
            analysis=AnalysisResult(
                bug_location=analysis_result.get("bug_location", ""),
                bug_type=analysis_result.get("bug_type", ""),
                root_cause=analysis_result.get("root_cause", ""),
                fix_suggestion=analysis_result.get("fix_suggestion", ""),
                confidence=float(analysis_result.get("confidence", 0))
            ),
            patch=fix_result.get("patch", ""),
            fixed_code=fix_result.get("fixed_code", ""),
            test_passed=validation_result.get("test_passed", False),
            total_steps=len(steps),
            steps=steps,
        )
    except Exception as e:
        return BugAnalyzeResponse(success=False, error=str(e))


# ============================================
# API 端点
# ============================================

@app.post("/api/register")
async def register(user: UserRegister):
    """用户注册"""
    if _hget(f"user:{user.username}", "password"):
        raise HTTPException(400, "用户已存在")
    _hset(f"user:{user.username}", "password", hash_pwd(user.password))
    if user.email:
        _hset(f"user:{user.username}", "email", user.email)
    return {"status": "success"}


@app.post("/api/login")
async def login(user: UserLogin):
    """用户登录 — 返回 JWT token"""
    stored = _hget(f"user:{user.username}", "password")
    if not stored:
        raise HTTPException(401, "用户不存在")
    if stored != hash_pwd(user.password):
        raise HTTPException(401, "密码错误")
    payload = {
        "sub": user.username,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"status": "success", "token": token, "username": user.username}


@app.post("/api/bug/analyze", response_model=BugAnalyzeResponse)
async def analyze_bug(bug: BugSubmitRequest, username: str = Depends(get_current_user)):
    """
    提交代码并进行 Multi-Agent 协同分析（同步）

    调用真实 Agent 系统：
    - ReviewerAgent: 审查代码发现Bug
    - AnalyzerAgent: 分析根因
    - 讨论确认后返回结果

    响应时间: 约 20-40 秒（取决于API延迟）
    """
    task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    code = bug.code
    language = bug.language

    print(f"[API] analyze_bug: task_id={task_id}, language={language}, code_len={len(code)}, user={username}")

    result = await _execute_multi_agent_analysis(code, language, username, task_id)
    result.task_id = task_id

    # 存储到 Redis
    _hset(f"task:{task_id}", "username", username)
    _hset(f"task:{task_id}", "code", code[:500])
    _hset(f"task:{task_id}", "language", language)
    _hset(f"task:{task_id}", "status", "completed" if result.success else "analyzed")
    _hset(f"task:{task_id}", "created_at", datetime.now().isoformat())
    _hset(f"task:{task_id}", "bug_location", result.analysis.bug_location)
    _hset(f"task:{task_id}", "bug_type", result.analysis.bug_type)
    _hset(f"task:{task_id}", "root_cause", result.analysis.root_cause[:500])
    _hset(f"task:{task_id}", "confidence", str(result.analysis.confidence))
    if result.error:
        _hset(f"task:{task_id}", "error", result.error)

    return result


@app.post("/api/bug/fix", response_model=BugAnalyzeResponse)
async def fix_bug(bug: BugSubmitRequest, username: str = Depends(get_current_user)):
    """
    提交代码并运行完整 Multi-Agent 协同修复流程（同步）

    完整 6 步流程：
    1. Reviewer 审查
    2. Reviewer-Analyzer 讨论
    3. Analyzer 根因分析
    4. Analyzer-Fixer 修复策略讨论
    5. Fixer 生成补丁
    6. Validator 验证修复

    响应时间: 约 60-120 秒
    """
    task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    code = bug.code
    language = bug.language

    print(f"[API] fix_bug: task_id={task_id}, language={language}, code_len={len(code)}, user={username}")

    result = await _execute_full_bugfix(code, language, username, task_id)
    result.task_id = task_id

    # 存储到 Redis
    _hset(f"task:{task_id}", "username", username)
    _hset(f"task:{task_id}", "code", code[:500])
    _hset(f"task:{task_id}", "language", language)
    _hset(f"task:{task_id}", "status", "fixed" if result.success else "analyzed")
    _hset(f"task:{task_id}", "created_at", datetime.now().isoformat())
    _hset(f"task:{task_id}", "bug_location", result.analysis.bug_location)
    _hset(f"task:{task_id}", "bug_type", result.analysis.bug_type)
    _hset(f"task:{task_id}", "root_cause", result.analysis.root_cause[:500])
    _hset(f"task:{task_id}", "patch", result.patch[:1000] if result.patch else "")
    _hset(f"task:{task_id}", "fixed_code", result.fixed_code[:1000] if result.fixed_code else "")
    _hset(f"task:{task_id}", "test_passed", str(result.test_passed))
    _hset(f"task:{task_id}", "confidence", str(result.analysis.confidence))
    _hset(f"task:{task_id}", "total_steps", str(result.total_steps))
    _hset(f"task:{task_id}", "steps", json.dumps([s.dict() for s in result.steps], ensure_ascii=False))
    if result.error:
        _hset(f"task:{task_id}", "error", result.error)

    return result


@app.post("/api/bug/submit")
async def submit_bug(bug: BugSubmitRequest, username: str = Depends(get_current_user)):
    """
    提交Bug修复任务（异步后台执行）

    立即返回 task_id，后台执行 Multi-Agent 修复流程。
    使用 GET /api/bug/task/{task_id} 轮询结果。
    """
    task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    _hset(f"task:{task_id}", "username", username)
    _hset(f"task:{task_id}", "code", bug.code[:1000])
    _hset(f"task:{task_id}", "language", bug.language)
    _hset(f"task:{task_id}", "status", "pending")
    _hset(f"task:{task_id}", "created_at", datetime.now().isoformat())
    if bug.description:
        _hset(f"task:{task_id}", "description", bug.description)
    if bug.test_case:
        _hset(f"task:{task_id}", "test_case", bug.test_case[:500])

    return {"status": "submitted", "task_id": task_id}


@app.get("/api/bug/task/{task_id}")
async def get_task(task_id: str):
    """获取任务详情（包含修复结果）"""
    task_data = _hgetall(f"task:{task_id}")
    if not task_data:
        raise HTTPException(404, "任务不存在")
    return dict(task_data)


@app.get("/api/bug/tasks/{username}")
async def get_user_tasks(username: str):
    """获取用户的所有任务"""
    tasks = []
    keys = r.keys("task:*") if r else [k for k in _cache if k.startswith("task:")]
    for key in keys:
        task_data = _hgetall(key)
        if task_data.get("username") == username:
            task_data["task_id"] = key.replace("task:", "") if isinstance(key, str) else key
            tasks.append(task_data)
    return {"tasks": tasks}


@app.get("/api/health")
async def health():
    """健康检查"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "redis": "connected" if r else "memory_fallback",
        "multi_agent": True
    }


@app.get("/")
async def root():
    """根路由 — 返回前端单页应用"""
    index_path = _frontend_dist / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "name": "Bug Fix Agent API v2.0",
        "description": "Multi-Agent Collaborative Bug Fix System",
        "agents": ["Reviewer", "Analyzer", "Fixer", "Validator"],
        "endpoints": {
            "analyze": "POST /api/bug/analyze",
            "fix": "POST /api/bug/fix",
            "submit": "POST /api/bug/submit",
            "task": "GET /api/bug/task/{task_id}",
            "tasks": "GET /api/bug/tasks/{username}",
            "health": "GET /api/health",
            "docs": "/docs"
        }
    }


@app.get("/{path:path}")
async def spa_fallback(path: str):
    """SPA fallback — 非 API 路由返回前端 index.html"""
    index_path = _frontend_dist / "index.html"
    if index_path.exists() and not path.startswith("api/") and path not in ("docs", "redoc", "openapi.json"):
        return FileResponse(index_path)
    raise HTTPException(404, "Not found")
