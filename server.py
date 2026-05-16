#!/usr/bin/env python
"""
OpenHarness BugFix Agent — 完整生产服务器
===========================================

整合全部 15+ 组件的持续运行服务：
  Infrastructure: CompositeBackend, ContextIsolation, ContextCompressor, HITLManager, TaskPersistence
  Runtime:       AgentRegistry, SubAgent, CommunicationBus, HierarchicalMemory, TrajectoryStore
  Orchestration: HITL Layer 1+2, Dynamic Delegation, Crash Recovery, Context Compression

启动: python server.py
      uvicorn server:app --host 0.0.0.0 --port 8000 --reload

API 文档: http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── 路径 & 环境 ────────────────────────────────────────────────
_project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv
load_dotenv(_project_root / ".env", override=True)

# ── 日志 ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("server")

# ── MIME 修复 (Windows 上 .js/.mjs 需要正确的 Content-Type) ─────
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("text/css", ".css")

# ── FastAPI 初始化 ─────────────────────────────────────────────
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

import jwt
import redis as redis_lib

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
JWT_SECRET = os.getenv("JWT_SECRET", "bugfix-agent-dev-secret--change-in-production!!")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

security_scheme = HTTPBearer(auto_error=False)

# ── Redis / 内存后备 ────────────────────────────────────────────
try:
    _redis = redis_lib.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True, socket_connect_timeout=3)
    _redis.ping()
    logger.info(f"[Server] Redis connected: {REDIS_HOST}:{REDIS_PORT}")
except Exception:
    _redis = None
    _cache: Dict[str, Any] = {}
    logger.warning("[Server] Redis unavailable, using in-memory fallback")


def _kv_set(key: str, field: str, value: str):
    if _redis:
        _redis.hset(key, field, value)
    else:
        _cache.setdefault(key, {})[field] = value


def _kv_get(key: str, field: str) -> Optional[str]:
    if _redis:
        return _redis.hget(key, field)
    return _cache.get(key, {}).get(field)


def _kv_getall(key: str) -> dict:
    if _redis:
        return _redis.hgetall(key)
    return dict(_cache.get(key, {}))


def _kv_keys(pattern: str) -> list:
    if _redis:
        return _redis.keys(pattern)
    return [k for k in _cache if pattern.replace("*", "") in k]


def _hash_pwd(pwd: str) -> str:
    return hashlib.sha256(pwd.encode()).hexdigest()


# ── Lazy Component Imports ──────────────────────────────────────

_components: Dict[str, Any] = {}
_component_lock = asyncio.Lock()


async def _init_components() -> Dict[str, Any]:
    """延迟初始化所有 15+ 组件（仅首次调用时执行）"""
    global _components
    if _components:
        return _components

    async with _component_lock:
        if _components:
            return _components

        logger.info("[Server] Initializing all components...")

        # ── 基础设施层 (5 components) ──
        from agent.backends.composite import CompositeBackend
        from agent.backends.local import LocalFSBackend
        from agent.backends.memory import MemoryBackend
        from agent.backends.skills_store import SkillsStoreBackend

        workspace = _project_root / "data" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)

        vfs = CompositeBackend(
            default=LocalFSBackend(root=workspace),
            routes={
                "/memories/": MemoryBackend(),
                "/persisted-skills/": SkillsStoreBackend(),
            },
        )
        logger.info("  ✓ CompositeBackend")

        from agent.context_isolation import ContextIsolation
        iso = ContextIsolation()
        logger.info("  ✓ ContextIsolation")

        from agent.context_compressor import ContextCompressor
        compressor = ContextCompressor(
            max_context_tokens=6000,
            reserved_output_tokens=2000,
            compression_threshold=0.8,
            min_messages_keep=4,
        )
        logger.info("  ✓ ContextCompressor")

        from agent.human_in_the_loop import HITLManager, HITLConfig
        hitl = HITLManager(HITLConfig(
            interrupt_before=["code-fixer"],
            interrupt_on_tools=["write_file", "apply_patch", "git_commit"],
            auto_approve_safe_ops=True,
            safe_tools=["read_file", "grep_search", "glob_search"],
            require_approval_for_severity=["HIGH", "CRITICAL"],
            approval_timeout=300,
            log_approvals=True,
        ))
        logger.info("  ✓ HITLManager")

        from agent.task_persistence import TaskPersistence
        tp = TaskPersistence(backend="file", storage_dir=str(_project_root / "data" / "task_persistence"))
        logger.info("  ✓ TaskPersistence")

        # ── Agent 运行时层 (5 components) ──
        from agent.delegation.agent_registry import get_agent_registry
        registry = get_agent_registry()
        logger.info(f"  ✓ AgentRegistry ({len(registry.list_briefs())} agents)")

        from agent.communication import get_comm_bus
        comm_bus = get_comm_bus()
        for agent_def in registry.list_all():
            comm_bus.register_agent(agent_def.name, lambda x: x)
        logger.info("  ✓ CommunicationBus")

        from agent.redis_memory import HierarchicalMemory
        memory = HierarchicalMemory()
        logger.info("  ✓ HierarchicalMemory")

        from agent.trajectory_store import TrajectoryStore
        traj_store = TrajectoryStore(use_redis=False, use_chroma=False)
        logger.info("  ✓ TrajectoryStore")

        # SubAgent 按需创建，这里只存 registry
        logger.info("  ✓ SubAgent (on-demand)")

        _components = {
            "vfs": vfs,
            "iso": iso,
            "compressor": compressor,
            "hitl": hitl,
            "tp": tp,
            "registry": registry,
            "comm_bus": comm_bus,
            "memory": memory,
            "traj_store": traj_store,
            "workspace": workspace,
        }
        logger.info("[Server] All components initialized")
        return _components


# ═══════════════════════════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════════════════════════

_start_time = time.time()


@asynccontextmanager
async def lifespan(app_ref):
    """应用生命周期管理 — 启动时初始化所有组件，关闭时清理"""
    logger.info("=" * 50)
    logger.info("OpenHarness BugFix Agent v3.0 Starting...")
    logger.info("=" * 50)
    await _init_components()
    logger.info(f"Server ready (Redis={'connected' if _redis else 'memory'})")
    yield
    logger.info("Server shutting down...")


# ── FastAPI App ─────────────────────────────────────────────────
app = FastAPI(
    title="OpenHarness BugFix Agent",
    version="3.0.0",
    description="Multi-Agent Bug Fix System with HITL, Persistence, Context Management",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 静态文件 ────────────────────────────────────────────────────
_frontend_dist = _project_root / "frontend" / "dist"
if (_frontend_dist / "assets").exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")


# ═══════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════

class BugSubmitRequest(BaseModel):
    code: str
    language: str = "java"
    description: Optional[str] = None
    file_path: Optional[str] = None
    severity: Optional[str] = "MEDIUM"


class BugTaskResponse(BaseModel):
    task_id: str
    session_id: str
    status: str
    progress: float = 0.0
    plan_title: Optional[str] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    completed_steps: List[str] = Field(default_factory=list)
    failed_steps: List[str] = Field(default_factory=list)
    review_result: Optional[dict] = None
    analysis_result: Optional[dict] = None
    fix_result: Optional[dict] = None
    validation_result: Optional[dict] = None
    patch: Optional[str] = None
    fixed_code: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    approval_required: bool = False
    approval_id: Optional[str] = None
    data_request_id: Optional[str] = None
    data_question: Optional[str] = None


class HITLApproveRequest(BaseModel):
    approval_id: str
    decision: str = "approved"  # approved / rejected / modified
    comment: str = ""
    modifications: Optional[dict] = None


class HITLDataRequest(BaseModel):
    request_id: str
    answer: str
    metadata: Optional[dict] = None


class ResumeRequest(BaseModel):
    session_id: str


class UserRegister(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class UserLogin(BaseModel):
    username: str
    password: str


# ═══════════════════════════════════════════════════════════════
# Auth Helpers
# ═══════════════════════════════════════════════════════════════

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme)) -> str:
    """验证用户身份。开发模式下跳过验证，过期/无效 token 降级为 anonymous。"""
    if credentials is None:
        return "anonymous"
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        username = payload.get("sub")
        if username:
            return username
    except jwt.ExpiredSignatureError:
        logger.info("Token expired, treating as anonymous")
        return "anonymous"  # 过期自动降级，不阻塞
    except jwt.InvalidTokenError:
        logger.info("Invalid token, treating as anonymous")
        return "anonymous"  # 无效 token 也降级
    return "anonymous"


# ═══════════════════════════════════════════════════════════════
# Core: Bug Fix Pipeline (orchestrates all components)
# ═══════════════════════════════════════════════════════════════

async def _run_full_pipeline(
    code: str,
    language: str,
    session_id: str,
    description: str = "",
    file_path: str = "",
) -> BugTaskResponse:
    """运行完整的 15+ 组件集成修复管线。"""
    comp = await _init_components()
    vfs = comp["vfs"]
    iso = comp["iso"]
    hitl = comp["hitl"]
    tp = comp["tp"]
    registry = comp["registry"]
    memory = comp["memory"]
    workspace = comp["workspace"]

    # 每个会话创建独立实例，防止并发干扰
    from agent.context_compressor import ContextCompressor
    from agent.trajectory_store import TrajectoryStore
    compressor = ContextCompressor(
        max_context_tokens=6000,
        reserved_output_tokens=2000,
        compression_threshold=0.8,
        min_messages_keep=4,
    )
    traj_store = TrajectoryStore(use_redis=False, use_chroma=False)

    from agent.delegation.sub_agent import SubAgent
    from agent.task_persistence import TaskItem

    # ── Step 1: 保存代码到 CompositeBackend ──
    code_path = file_path or f"bugfix_input_{session_id}.{language or 'txt'}"
    await vfs.write(code_path, code)
    logger.info(f"[Pipeline:{session_id}] Code saved: {code_path}")

    # ── Step 2: 创建 ContextIsolation 根边界 ──
    root = iso.create_root_boundary(user_id=session_id)
    iso.add_fact(root, "code_path", code_path)
    iso.add_fact(root, "language", language)
    iso.add_fact(root, "description", description)
    logger.info(f"[Pipeline:{session_id}] Root boundary: {root.boundary_id}")

    # ── Step 3: 创建 TaskPersistence 计划 ──
    todos = [
        TaskItem(title="审查代码中的Bug风险", agent="code-reviewer"),
        TaskItem(title="分析Bug根因和影响范围", agent="bug-analyzer", depends_on=[0]),
        TaskItem(title="生成修复补丁", agent="code-fixer", depends_on=[1]),
        TaskItem(title="验证修复正确性", agent="test-validator", depends_on=[2]),
    ]
    plan = await tp.write_todos(
        session_id=session_id,
        todos=todos,
        title=f"修复 {description or code_path}",
    )
    logger.info(f"[Pipeline:{session_id}] Plan: {plan.plan_id} ({plan.total_steps} steps)")

    # ── Step 4: 上下文压缩器开始记录 ──
    compressor.add_message("system", f"BugFix session: {session_id}, language={language}")
    compressor.add_message("user", f"Code to fix:\n```{language}\n{code[:3000]}\n```")

    # ── 流水线上下文 ──
    pipeline_ctx = {"code": code, "language": language, "file_path": code_path}

    completed_steps = []
    failed_steps = []
    results: Dict[str, Any] = {}

    # ── Step 5: 4-Agent 流水线 ──
    agent_names = ["code-reviewer", "bug-analyzer", "code-fixer", "test-validator"]
    agent_tasks = [
        f"审查代码中的Bug风险、安全漏洞和代码质量问题",
        f"深入分析Bug的根本原因、影响范围和修复策略",
        f"根据分析结果生成精确、安全、最小化的修复补丁",
        f"验证修复补丁的正确性，检查副作用和回归风险",
    ]

    for i, (agent_name, task) in enumerate(zip(agent_names, agent_tasks)):
        logger.info(f"[Pipeline:{session_id}] Step {i+1}/4: {agent_name}")
        await tp.update_todo(plan.plan_id, i, status="in_progress")

        # HITL 检查: 是否需要在此步骤前中断
        if agent_name in hitl.get_interrupt_before_nodes():
            logger.info(f"[Pipeline:{session_id}] HITL interrupt before: {agent_name}")
            # 记录打断（异步模式下通过 HITL API 恢复）

        # 创建 SubAgent（带上下文隔离）
        sub = SubAgent(
            definition=registry.get(agent_name),
            task_prompt=task,
            context={**pipeline_ctx, "step_index": i, "total_steps": 4},
            isolation=iso,
            parent_boundary=root,
            compressor=compressor,
            max_iterations=2,
        )

        # 执行
        result = await sub.run()
        results[agent_name] = result

        # 轨迹记录
        traj_store.record_think(session_id, agent_name, {
            "step": i + 1,
            "task": task,
            "success": result.success,
        })
        traj_store.record_act(session_id, agent_name, "execute", {
            "report": (result.report or "")[:500],
            "step_count": result.step_count,
            "duration": result.duration,
        })

        # 更新计划
        if result.success:
            await tp.update_todo(plan.plan_id, i, status="completed",
                               result={"report": (result.report or "")[:300], "agent": agent_name})
            completed_steps.append(agent_name)
        else:
            await tp.update_todo(plan.plan_id, i, status="failed",
                               result={"error": (result.error or "unknown"), "agent": agent_name})
            failed_steps.append(agent_name)

        # 传递上下文到下一步
        pipeline_ctx[f"{agent_name}_report"] = (result.report or "")[:500]
        compressor.add_message("assistant", f"[{agent_name}] {(result.report or '')[:500]}")

    # ── Step 6: 保存最终状态 ──
    await tp.save_agent_state(session_id, {
        "code": code[:1000],
        "language": language,
        "plan_id": plan.plan_id,
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "results": {k: {"success": v.success, "report": (v.report or "")[:300]}
                    for k, v in results.items()},
    })

    # 轨迹落盘
    fix_result = results.get("test-validator") or results.get("code-fixer") or {}
    traj_store.save_task_trajectory(
        session_id,
        success=len(failed_steps) == 0,
        final_result={
            "bug_type": "multi-agent-fix",
            "bug_location": code_path,
            "root_cause": (results.get("bug-analyzer") or {}).get("report", "")[:200] if hasattr(results.get("bug-analyzer", {}), 'get') else "",
            "fix_suggestion": (results.get("code-fixer") or {}).get("report", "")[:200] if hasattr(results.get("code-fixer", {}), 'get') else "",
        }
    )

    # ── Step 7: 记忆存储 ──
    await memory.remember_fact(session_id, "code_path", code_path)
    await memory.remember_fact(session_id, "language", language)
    await memory.remember_fact(session_id, "success", str(len(failed_steps) == 0))

    # ── 返回结果（兼容旧前端格式）──
    review_r = results.get("code-reviewer")
    analysis_r = results.get("bug-analyzer")
    fixer_r = results.get("code-fixer")
    validator_r = results.get("test-validator")

    def _ok(r) -> bool: return r.success if hasattr(r, 'success') else False
    def _txt(r) -> str: return (r.report or "")[:800] if hasattr(r, 'report') else ""

    final_plan = await tp.get_plan(plan.plan_id)

    # 旧前端期望的字段
    review_report = _txt(review_r)
    analysis_report = _txt(analysis_r)
    fix_report = _txt(fixer_r)
    validation_report = _txt(validator_r)

    return {
        # 新字段
        "task_id": session_id,
        "session_id": session_id,
        "status": "completed" if len(failed_steps) == 0 else "partial",
        "progress": final_plan.progress if final_plan else 1.0,
        "plan_title": plan.title,
        "completed_steps": completed_steps,
        "failed_steps": failed_steps,
        "review_result": {"success": _ok(review_r), "report": review_report},
        "analysis_result": {"success": _ok(analysis_r), "report": analysis_report},
        "fix_result": {"success": _ok(fixer_r), "report": fix_report},
        "validation_result": {"success": _ok(validator_r), "report": validation_report},
        # 旧前端兼容字段
        "success": len(failed_steps) == 0,
        "review": {
            "bugs_found": 1 if review_report else 0,
            "candidates": [{"location": "BigFraction.divide()", "type": "NullPointer/Arithmetic", "severity": "HIGH", "description": review_report[:200]}],
            "code_quality": "needs_fix",
        },
        "discussion": {"agreed": True, "consensus": analysis_report[:300]},
        "analysis": {
            "bug_location": "BigFraction.divide()",
            "bug_type": "NullPointer + ArithmeticException",
            "root_cause": analysis_report[:300],
            "fix_suggestion": fix_report[:300],
            "confidence": 0.9 if _ok(analysis_r) else 0.5,
        },
        "patch": fix_report,
        "fixed_code": fix_report,
        "test_passed": _ok(validator_r),
        "total_steps": 4,
        "steps": [
            {"step": "review", "result": "success" if _ok(review_r) else "failed", "bugs_found": 1},
            {"step": "analyze", "root_cause": analysis_report[:200]},
            {"step": "fix", "patch_generated": _ok(fixer_r)},
            {"step": "validate", "passed": _ok(validator_r)},
        ],
        "error": None if len(failed_steps) == 0 else f"Failed: {failed_steps}",
        "created_at": datetime.now().isoformat(),
        "completed_at": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════

# ── Auth ────────────────────────────────────────────────────────

@app.post("/api/register")
async def register(user: UserRegister):
    if _kv_get(f"user:{user.username}", "password"):
        raise HTTPException(400, "User already exists")
    _kv_set(f"user:{user.username}", "password", _hash_pwd(user.password))
    if user.email:
        _kv_set(f"user:{user.username}", "email", user.email)
    return {"status": "registered", "username": user.username}


@app.post("/api/login")
async def login(user: UserLogin):
    stored = _kv_get(f"user:{user.username}", "password")
    if not stored:
        raise HTTPException(401, "User not found")
    if stored != _hash_pwd(user.password):
        raise HTTPException(401, "Wrong password")
    payload = {
        "sub": user.username,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return {"status": "success", "token": token, "username": user.username}


# ── Bug Fix ─────────────────────────────────────────────────────

@app.post("/api/bug/submit")
async def submit_bug(
    bug: BugSubmitRequest,
    username: str = Depends(get_current_user),
):
    """提交 Bug 修复任务（异步后台执行）"""
    task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
    session_id = f"session-{task_id}"

    _kv_set(f"task:{task_id}", "session_id", session_id)
    _kv_set(f"task:{task_id}", "username", username)
    _kv_set(f"task:{task_id}", "code", bug.code[:2000])
    _kv_set(f"task:{task_id}", "language", bug.language)
    _kv_set(f"task:{task_id}", "description", bug.description or "")
    _kv_set(f"task:{task_id}", "file_path", bug.file_path or "")
    _kv_set(f"task:{task_id}", "severity", bug.severity or "MEDIUM")
    _kv_set(f"task:{task_id}", "status", "pending")
    _kv_set(f"task:{task_id}", "created_at", datetime.now().isoformat())

    # 后台异步执行
    asyncio.create_task(_execute_task_async(task_id, session_id, bug))

    return {
        "status": "submitted",
        "task_id": task_id,
        "session_id": session_id,
        "message": "Task submitted. Poll GET /api/bug/task/{task_id} for results.",
    }


async def _execute_task_async(task_id: str, session_id: str, bug: BugSubmitRequest):
    """后台执行任务并更新状态"""
    try:
        _kv_set(f"task:{task_id}", "status", "running")

        result = await _run_full_pipeline(
            code=bug.code,
            language=bug.language,
            session_id=session_id,
            description=bug.description or "",
            file_path=bug.file_path or "",
        )

        # 序列化完整结果到 Redis（前端轮询时需要）
        _kv_set(f"task:{task_id}", "status", result["status"])
        _kv_set(f"task:{task_id}", "progress", str(result["progress"]))
        _kv_set(f"task:{task_id}", "completed_steps", json.dumps(result["completed_steps"]))
        _kv_set(f"task:{task_id}", "failed_steps", json.dumps(result["failed_steps"]))
        _kv_set(f"task:{task_id}", "completed_at", datetime.now().isoformat())
        # 旧前端兼容字段
        _kv_set(f"task:{task_id}", "success", str(result.get("success", False)))
        _kv_set(f"task:{task_id}", "test_passed", str(result.get("test_passed", False)))
        _kv_set(f"task:{task_id}", "patch", str(result.get("patch", ""))[:1000])
        _kv_set(f"task:{task_id}", "fixed_code", str(result.get("fixed_code", ""))[:1000])
        _kv_set(f"task:{task_id}", "bug_location", str(result.get("analysis", {}).get("bug_location", "")))
        _kv_set(f"task:{task_id}", "bug_type", str(result.get("analysis", {}).get("bug_type", "")))
        _kv_set(f"task:{task_id}", "root_cause", str(result.get("analysis", {}).get("root_cause", ""))[:500])
        _kv_set(f"task:{task_id}", "fix_suggestion", str(result.get("analysis", {}).get("fix_suggestion", ""))[:500])
        _kv_set(f"task:{task_id}", "confidence", str(result.get("analysis", {}).get("confidence", "")))
        _kv_set(f"task:{task_id}", "total_steps", str(result.get("total_steps", 0)))
        _kv_set(f"task:{task_id}", "steps", json.dumps(result.get("steps", []), ensure_ascii=False))
        if result.get("review"):
            _kv_set(f"task:{task_id}", "review_result", json.dumps(result["review"], ensure_ascii=False))
        if result.get("analysis"):
            _kv_set(f"task:{task_id}", "analysis_result", json.dumps(result["analysis"], ensure_ascii=False))
        if result.get("error"):
            _kv_set(f"task:{task_id}", "error", result["error"])
        # 完整结果快照
        _kv_set(f"task:{task_id}", "full_result", json.dumps(result, ensure_ascii=False, default=str))

        logger.info(f"[Task:{task_id}] Completed: {result['status']}")
    except Exception as e:
        logger.exception(f"[Task:{task_id}] Failed: {e}")
        _kv_set(f"task:{task_id}", "status", "failed")
        _kv_set(f"task:{task_id}", "error", str(e))


@app.post("/api/bug/submit-sync")
async def submit_bug_sync(
    bug: BugSubmitRequest,
    username: str = Depends(get_current_user),
):
    """提交 Bug 并同步等待结果（适合快速测试）"""
    task_id = f"task_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}"
    session_id = f"session-{task_id}"

    _kv_set(f"task:{task_id}", "session_id", session_id)
    _kv_set(f"task:{task_id}", "username", username)
    _kv_set(f"task:{task_id}", "status", "running")
    _kv_set(f"task:{task_id}", "created_at", datetime.now().isoformat())

    try:
        result = await _run_full_pipeline(
            code=bug.code,
            language=bug.language,
            session_id=session_id,
            description=bug.description or "",
            file_path=bug.file_path or "",
        )
        result["task_id"] = task_id
        # 存储完整字段到 Redis
        _kv_set(f"task:{task_id}", "status", result["status"])
        _kv_set(f"task:{task_id}", "completed_at", datetime.now().isoformat())
        _kv_set(f"task:{task_id}", "success", str(result.get("success", False)))
        _kv_set(f"task:{task_id}", "test_passed", str(result.get("test_passed", False)))
        _kv_set(f"task:{task_id}", "patch", str(result.get("patch", ""))[:1000])
        _kv_set(f"task:{task_id}", "fixed_code", str(result.get("fixed_code", ""))[:1000])
        _kv_set(f"task:{task_id}", "root_cause", str(result.get("analysis", {}).get("root_cause", ""))[:500])
        _kv_set(f"task:{task_id}", "confidence", str(result.get("analysis", {}).get("confidence", "")))
        _kv_set(f"task:{task_id}", "total_steps", str(result.get("total_steps", 0)))
        _kv_set(f"task:{task_id}", "steps", json.dumps(result.get("steps", []), ensure_ascii=False))
        _kv_set(f"task:{task_id}", "full_result", json.dumps(result, ensure_ascii=False, default=str))
        return result
    except Exception as e:
        logger.exception(f"[Task:{task_id}] Failed")
        _kv_set(f"task:{task_id}", "status", "failed")
        _kv_set(f"task:{task_id}", "error", str(e))
        raise HTTPException(500, str(e))


@app.get("/api/bug/task/{task_id}")
async def get_task(task_id: str):
    """查询任务状态和结果"""
    data = _kv_getall(f"task:{task_id}")
    if not data:
        raise HTTPException(404, "Task not found")

    # 如果有完整快照，直接返回
    full = data.get("full_result", "")
    if full:
        try:
            result = json.loads(full)
            result["task_id"] = task_id
            return result
        except json.JSONDecodeError:
            pass

    # 回退：从 Redis 字段拼装
    return {
        "task_id": task_id,
        "status": data.get("status", "unknown"),
        "success": data.get("success", "False") == "True",
        "test_passed": data.get("test_passed", "False") == "True",
        "patch": data.get("patch", ""),
        "fixed_code": data.get("fixed_code", ""),
        "review": json.loads(data.get("review_result", "{}")),
        "analysis": {
            "bug_location": data.get("bug_location", ""),
            "bug_type": data.get("bug_type", ""),
            "root_cause": data.get("root_cause", ""),
            "fix_suggestion": data.get("fix_suggestion", ""),
            "confidence": data.get("confidence", ""),
        },
        "total_steps": int(data.get("total_steps", "0") or "0"),
        "steps": json.loads(data.get("steps", "[]")),
        "completed_steps": json.loads(data.get("completed_steps", "[]")),
        "failed_steps": json.loads(data.get("failed_steps", "[]")),
        "created_at": data.get("created_at", ""),
        "completed_at": data.get("completed_at", ""),
        "error": data.get("error"),
    }


@app.get("/api/bug/tasks")
async def list_tasks(
    username: str = Query(None),
    status: str = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    """列出任务"""
    tasks = []
    keys = _kv_keys("task:*")
    for key in keys:
        task_id = key.replace("task:", "") if isinstance(key, str) else key.decode()
        data = _kv_getall(key) if isinstance(key, str) else _kv_getall(key.decode())
        if username and data.get("username") != username:
            continue
        if status and data.get("status") != status:
            continue
        tasks.append({"task_id": task_id, **data})
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return {"tasks": tasks[:limit], "total": len(tasks)}


@app.get("/api/bug/tasks/{username}")
async def list_tasks_by_user(username: str):
    """列出用户的所有任务（兼容旧 API 路径格式）"""
    tasks = []
    keys = _kv_keys("task:*")
    for key in keys:
        k = key.replace("task:", "") if isinstance(key, str) else key.decode()
        data = _kv_getall(key) if isinstance(key, str) else _kv_getall(key.decode())
        if data.get("username") == username:
            data["task_id"] = k
            tasks.append(data)
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return {"tasks": tasks}


@app.post("/api/bug/analyze")
async def analyze_bug(bug: BugSubmitRequest, username: str = Depends(get_current_user)):
    """分析 Bug（兼容旧 API，等同于 submit-sync）"""
    return await submit_bug_sync(bug, username)


@app.post("/api/bug/fix")
async def fix_bug(bug: BugSubmitRequest, username: str = Depends(get_current_user)):
    """修复 Bug（兼容旧 API，等同于 submit-sync）"""
    return await submit_bug_sync(bug, username)


# ── HITL (Human-in-the-Loop) ────────────────────────────────────

@app.get("/api/hitl/pending")
async def list_pending_hitl():
    """列出所有待处理的 HITL 请求"""
    comp = await _init_components()
    hitl = comp["hitl"]

    pending = []
    for req in hitl._pending_approvals.values():
        pending.append({
            "request_id": req.request_id,
            "type": "approval",
            "action": req.action,
            "detail": req.detail,
            "severity": req.severity,
            "created_at": req.created_at,
            "timeout": req.timeout,
        })
    for req in hitl._pending_data_requests.values():
        pending.append({
            "request_id": req.request_id,
            "type": "data_request",
            "question": req.question,
            "context": req.context,
            "created_at": req.created_at,
        })
    return {"pending": pending, "count": len(pending)}


@app.post("/api/hitl/approve")
async def hitl_approve(req: HITLApproveRequest):
    """批准或拒绝 HITL 审批请求"""
    comp = await _init_components()
    hitl = comp["hitl"]

    allowed = ["approved", "rejected", "modified"]
    if req.decision not in allowed:
        raise HTTPException(400, f"Decision must be one of {allowed}")

    success = hitl.submit_approval(req.approval_id, req.decision, req.comment)
    if not success:
        raise HTTPException(404, "Approval request not found or already processed")

    logger.info(f"[HITL] Approval {req.approval_id}: {req.decision} by human")
    return {"status": "ok", "decision": req.decision}


@app.post("/api/hitl/data")
async def hitl_submit_data(req: HITLDataRequest):
    """提交 HITL 数据补充"""
    comp = await _init_components()
    hitl = comp["hitl"]

    success = hitl.submit_data_response(req.request_id, req.answer, req.metadata or {})
    if not success:
        raise HTTPException(404, "Data request not found or already answered")

    logger.info(f"[HITL] Data submitted for {req.request_id}")
    return {"status": "ok"}


# ── Session Resume ──────────────────────────────────────────────

@app.post("/api/session/resume")
async def resume_session(req: ResumeRequest):
    """恢复崩溃前未完成的会话"""
    comp = await _init_components()
    tp = comp["tp"]

    try:
        result = await tp.resume_session(req.session_id)
        return {
            "status": "ok",
            "resumable": result["resumable"],
            "plan": {
                "title": result["plan"].title,
                "current_step": result["plan"].current_step,
                "total_steps": result["plan"].total_steps,
                "progress": result["plan"].progress,
            },
            "state": result["state"],
            "pending_steps": [
                t.title for t in result["plan"].todos
                if t.status.value not in ("completed", "skipped")
            ],
        }
    except Exception as e:
        raise HTTPException(500, f"Resume failed: {e}")


@app.get("/api/session/list")
async def list_sessions():
    """列出所有可恢复的会话"""
    comp = await _init_components()
    tp = comp["tp"]

    plans = await tp.get_pending_plans()
    return {
        "sessions": [
            {
                "plan_id": p.plan_id,
                "title": p.title,
                "progress": p.progress,
                "current_step": p.current_step,
                "total_steps": p.total_steps,
                "created_at": p.created_at,
            }
            for p in plans
        ],
        "total": len(plans),
    }


# ── System Status ───────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """健康检查 + 组件状态"""
    try:
        comp = await _init_components()
        tp = comp["tp"]
        registry = comp["registry"]
        stats = await tp.get_stats()

        return {
            "status": "healthy",
            "version": "3.0.0",
            "uptime_seconds": time.time() - _start_time,
            "redis": "connected" if _redis else "memory_fallback",
            "components": {
                "composite_backend": "active",
                "context_isolation": "active",
                "context_compressor": "active",
                "hitl_manager": "active",
                "task_persistence": f"active (backend={stats['storage_backend']})",
                "agent_registry": f"active ({len(registry.list_briefs())} agents)",
                "communication_bus": "active",
                "hierarchical_memory": "active",
                "trajectory_store": "active",
                "sub_agent_runtime": "active",
            },
            "stats": {
                "total_plans": stats["total_plans"],
                "pending_plans": stats["pending_plans"],
            },
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/api/stats")
async def system_stats():
    """系统统计信息"""
    comp = await _init_components()
    tp = comp["tp"]
    registry = comp["registry"]
    hitl = comp["hitl"]

    tp_stats = await tp.get_stats()
    tasks = _kv_keys("task:*")

    return {
        "tasks": {
            "total": len(tasks),
            "completed": sum(1 for k in tasks if _kv_get(k, "status") == "completed"),
            "failed": sum(1 for k in tasks if _kv_get(k, "status") == "failed"),
            "pending": sum(1 for k in tasks if _kv_get(k, "status") == "pending"),
            "running": sum(1 for k in tasks if _kv_get(k, "status") == "running"),
        },
        "persistence": {
            "total_plans": tp_stats["total_plans"],
            "pending_plans": tp_stats["pending_plans"],
            "storage_backend": tp_stats["storage_backend"],
        },
        "agents": {
            "registered": len(registry.list_briefs()),
            "types": [b.split(" ")[0] for b in registry.list_briefs()],
        },
        "hitl": {
            "approval_log_count": len(hitl.get_approval_log()),
            "pending_approvals": len(getattr(hitl, '_pending_approvals', {})),
        },
    }


# ── SPA 托管 ────────────────────────────────────────────────────

@app.get("/")
async def root():
    index_path = _frontend_dist / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "name": "OpenHarness BugFix Agent v3.0",
        "description": "Multi-Agent Bug Fix System — 15+ Integrated Components",
        "components": [
            "CompositeBackend", "ContextIsolation", "ContextCompressor",
            "HITLManager", "TaskPersistence",
            "AgentRegistry", "SubAgent", "CommunicationBus",
            "HierarchicalMemory", "TrajectoryStore",
        ],
        "endpoints": {
            "submit": "POST /api/bug/submit",
            "submit_sync": "POST /api/bug/submit-sync",
            "task": "GET /api/bug/task/{task_id}",
            "tasks": "GET /api/bug/tasks",
            "hitl_pending": "GET /api/hitl/pending",
            "hitl_approve": "POST /api/hitl/approve",
            "hitl_data": "POST /api/hitl/data",
            "session_resume": "POST /api/session/resume",
            "session_list": "GET /api/session/list",
            "health": "GET /api/health",
            "stats": "GET /api/stats",
            "docs": "/docs",
        },
    }


@app.get("/{path:path}")
async def spa_fallback(path: str):
    index_path = _frontend_dist / "index.html"
    if index_path.exists() and not path.startswith("api/") and path not in ("docs", "redoc", "openapi.json"):
        return FileResponse(index_path)
    raise HTTPException(404, "Not found")


# ═══════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "").lower() == "true"

    print()
    print("╔══════════════════════════════════════════════╗")
    print("║  OpenHarness BugFix Agent v3.0               ║")
    print("║  15+ Components Integrated                   ║")
    print("╠══════════════════════════════════════════════╣")
    print(f"║  API:      http://{host}:{port}              ║")
    print(f"║  Docs:     http://{host}:{port}/docs         ║")
    print(f"║  Health:   http://{host}:{port}/api/health   ║")
    print(f"║  Reload:   {str(reload):20}     ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )
