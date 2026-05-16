"""
Task Persistence — 任务规划持久化 & 重启恢复
============================================

PDF Harness Engineering: write_todos + Agent state 持久化

核心能力:
  1. write_todos — 将任务计划结构化写入持久化存储
  2. Agent state 快照 — 保存完整 Agent 状态（plan/current_step/results）
  3. 重启恢复 — 从持久化存储恢复未完成的任务
  4. 检查点 — 与 LangGraph checkpointer 协同工作

与 LangGraph checkpointer 的关系:
  LangGraph checkpointer 保存状态图的状态（状态机快照）
  TaskPersistence 保存业务层任务计划（跨图、跨会话的 TODO 列表）
  两者互补：checkpointer 负责执行流程的恢复，TaskPersistence 负责任务语义的持久化

存储后端:
  优先: Redis (生产环境，支持 TTL、多键查询)
  回退: JSON 文件 (开发环境，零依赖)

用法:
  tp = TaskPersistence(backend="redis")

  # 写入任务
  await tp.write_todos(plan_id="task-001", todos=[
      TaskItem(title="审查代码", agent="code-reviewer", status="pending"),
      TaskItem(title="分析Bug", agent="bug-analyzer", status="pending"),
  ])

  # 更新状态
  await tp.update_todo("task-001", 0, status="in_progress")
  await tp.update_todo("task-001", 0, status="completed", result={"bug_count": 3})

  # 保存完整 Agent 状态
  await tp.save_agent_state("session-001", agent_state_dict)

  # 重启恢复
  pending = await tp.get_pending_plans()
  state = await tp.load_agent_state("session-001")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("task_persistence")


# ── 枚举 ────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"       # 等待人工审批


# ── 数据类 ──────────────────────────────────────────────────

@dataclass
class TaskItem:
    """单个任务项（对应 PDF 项目的 write_todos 中的一条）"""
    title: str
    agent: str                           # 负责的 Agent 类型
    status: TaskStatus = TaskStatus.PENDING
    description: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    depends_on: List[int] = field(default_factory=list)  # 依赖的任务索引
    retry_count: int = 0
    max_retries: int = 3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "agent": self.agent,
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "description": self.description,
            "context": self.context,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "depends_on": self.depends_on,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskItem":
        return cls(
            title=d.get("title", ""),
            agent=d.get("agent", ""),
            status=TaskStatus(d.get("status", "pending")),
            description=d.get("description", ""),
            context=d.get("context", {}),
            result=d.get("result"),
            error=d.get("error"),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            depends_on=d.get("depends_on", []),
            retry_count=d.get("retry_count", 0),
            max_retries=d.get("max_retries", 3),
        )


@dataclass
class TaskPlan:
    """完整任务计划（对应 PDF 项目的 write_todos 输出）"""
    plan_id: str
    session_id: str
    title: str
    todos: List[TaskItem]
    status: TaskStatus = TaskStatus.PENDING
    current_step: int = 0
    total_steps: int = 0
    replan_count: int = 0
    max_replan: int = 3
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None
    parent_plan_id: Optional[str] = None   # 嵌套计划（replan 产生的新计划）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "session_id": self.session_id,
            "title": self.title,
            "todos": [t.to_dict() for t in self.todos],
            "status": self.status.value if isinstance(self.status, TaskStatus) else self.status,
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "replan_count": self.replan_count,
            "max_replan": self.max_replan,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "parent_plan_id": self.parent_plan_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TaskPlan":
        return cls(
            plan_id=d.get("plan_id", ""),
            session_id=d.get("session_id", ""),
            title=d.get("title", ""),
            todos=[TaskItem.from_dict(t) for t in d.get("todos", [])],
            status=TaskStatus(d.get("status", "pending")),
            current_step=d.get("current_step", 0),
            total_steps=d.get("total_steps", len(d.get("todos", []))),
            replan_count=d.get("replan_count", 0),
            max_replan=d.get("max_replan", 3),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            completed_at=d.get("completed_at"),
            parent_plan_id=d.get("parent_plan_id"),
            metadata=d.get("metadata", {}),
        )

    @property
    def progress(self) -> float:
        """完成进度 0.0 - 1.0"""
        if self.total_steps == 0:
            return 0.0
        completed = sum(1 for t in self.todos if t.status == TaskStatus.COMPLETED)
        return completed / self.total_steps

    @property
    def is_completed(self) -> bool:
        return self.status == TaskStatus.COMPLETED or all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for t in self.todos
        )

    @property
    def blocked_tasks(self) -> List[TaskItem]:
        return [t for t in self.todos if t.status == TaskStatus.BLOCKED]


# ── 存储后端接口 ─────────────────────────────────────────────

class StorageBackend:
    """持久化存储后端抽象"""

    async def get(self, key: str) -> Optional[str]:
        raise NotImplementedError

    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        raise NotImplementedError

    async def delete(self, key: str) -> bool:
        raise NotImplementedError

    async def keys(self, pattern: str) -> List[str]:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        raise NotImplementedError


class RedisStorageBackend(StorageBackend):
    """Redis 存储后端"""

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.host = host
        self.port = port
        self.db = db
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                self._redis = await aioredis.from_url(
                    f"redis://{self.host}:{self.port}/{self.db}",
                    encoding="utf-8", decode_responses=True,
                )
                await self._redis.ping()
            except Exception:
                logger.warning("[TaskPersistence] Redis not available, falling back to FileStorage")
                return None
        return self._redis

    async def get(self, key: str) -> Optional[str]:
        r = await self._get_redis()
        if r:
            return await r.get(key)
        return None

    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        r = await self._get_redis()
        if r:
            await r.set(key, value, ex=ttl)
            return True
        return False

    async def delete(self, key: str) -> bool:
        r = await self._get_redis()
        if r:
            await r.delete(key)
            return True
        return False

    async def keys(self, pattern: str) -> List[str]:
        r = await self._get_redis()
        if r:
            return await r.keys(pattern)
        return []

    async def exists(self, key: str) -> bool:
        r = await self._get_redis()
        if r:
            return await r.exists(key) > 0
        return False


class FileStorageBackend(StorageBackend):
    """文件存储后端（回退方案）"""

    def __init__(self, base_dir: str = "data/task_persistence"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        safe_key = key.replace(":", "_").replace("/", "_")
        return self.base_dir / f"{safe_key}.json"

    async def get(self, key: str) -> Optional[str]:
        path = self._key_to_path(key)
        if path.exists():
            return path.read_text(encoding="utf-8")
        return None

    async def set(self, key: str, value: str, ttl: int = None) -> bool:
        path = self._key_to_path(key)
        path.write_text(value, encoding="utf-8")
        return True

    async def delete(self, key: str) -> bool:
        path = self._key_to_path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    async def keys(self, pattern: str) -> List[str]:
        import fnmatch
        results = []
        for path in self.base_dir.glob("*.json"):
            key = path.stem.replace("_", ":").replace("taskpersistence:", "task_persistence:", 1)
            if fnmatch.fnmatch(key, pattern):
                results.append(key)
        return results

    async def exists(self, key: str) -> bool:
        return self._key_to_path(key).exists()


# ── TaskPersistence ───────────────────────────────────────────

class TaskPersistence:
    """任务规划持久化管理器。

    对应 PDF 项目的 write_todos + Agent state 持久化。

    核心功能:
      - write_todos: 写入任务计划
      - update_todo: 更新单个任务状态
      - save_agent_state / load_agent_state: 保存/恢复 Agent 完整状态
      - get_pending_plans: 获取未完成的计划（重启恢复）
      - snapshot / restore: 快照与恢复
      - cleanup: 清理过期任务

    用法:
        tp = TaskPersistence(backend="file")

        # 写入计划
        plan = await tp.write_todos(session_id="s1", todos=[...])

        # 更新步骤
        await tp.update_todo(plan.plan_id, 0, status="completed")

        # 保存 Agent 状态
        await tp.save_agent_state("s1", state_dict)

        # 重启恢复
        pending = await tp.get_pending_plans()
        state = await tp.load_agent_state("s1")
    """

    def __init__(
        self,
        backend: str = "auto",
        redis_host: str = "localhost",
        redis_port: int = 6379,
        storage_dir: str = "data/task_persistence",
        default_ttl: int = 86400 * 7,  # 7 天
    ):
        """
        Args:
            backend: "redis" | "file" | "auto" (先尝试 Redis，失败回退到文件)
            redis_host: Redis 主机
            redis_port: Redis 端口
            storage_dir: 文件存储目录
            default_ttl: Redis 键的默认 TTL（秒）
        """
        self.default_ttl = default_ttl

        if backend == "redis":
            self._storage = RedisStorageBackend(redis_host, redis_port)
        elif backend == "file":
            self._storage = FileStorageBackend(storage_dir)
        else:  # auto
            self._storage = FileStorageBackend(storage_dir)
            # Redis 会在需要时尝试连接

        self._key_prefix = "task_persistence"
        self._redis_available = False

    async def _ensure_storage(self) -> StorageBackend:
        """确保存储可用（auto 模式下尝试升级到 Redis）"""
        if isinstance(self._storage, FileStorageBackend) and not self._redis_available:
            # 尝试连接 Redis
            try:
                import redis.asyncio as aioredis
                test_redis = await aioredis.from_url(
                    "redis://localhost:6379/0",
                    encoding="utf-8", decode_responses=True,
                )
                await test_redis.ping()
                self._storage = RedisStorageBackend()
                self._redis_available = True
                logger.info("[TaskPersistence] Upgraded to Redis backend")
            except Exception:
                pass
        return self._storage

    # ── 键名生成 ─────────────────────────────────────────────

    def _plan_key(self, plan_id: str) -> str:
        return f"{self._key_prefix}:plan:{plan_id}"

    def _session_key(self, session_id: str) -> str:
        return f"{self._key_prefix}:session:{session_id}"

    def _state_key(self, session_id: str) -> str:
        return f"{self._key_prefix}:state:{session_id}"

    def _index_key(self) -> str:
        return f"{self._key_prefix}:plan_index"

    # ── write_todos ─────────────────────────────────────────

    async def write_todos(
        self,
        session_id: str,
        todos: List[TaskItem],
        title: str = "",
        plan_id: str = None,
        parent_plan_id: str = None,
        metadata: Dict = None,
    ) -> TaskPlan:
        """写入任务计划（对应 PDF 项目的 write_todos 工具）。

        Args:
            session_id: 会话 ID
            todos: 任务列表
            title: 计划标题
            plan_id: 计划 ID（不指定则自动生成）
            parent_plan_id: 父计划 ID（replan 产生的新计划）
            metadata: 额外的元数据

        Returns:
            TaskPlan: 创建的完整计划
        """
        storage = await self._ensure_storage()
        plan_id = plan_id or f"plan-{uuid.uuid4().hex[:12]}"

        plan = TaskPlan(
            plan_id=plan_id,
            session_id=session_id,
            title=title or f"BugFix Plan ({len(todos)} steps)",
            todos=todos,
            total_steps=len(todos),
            parent_plan_id=parent_plan_id,
            metadata=metadata or {},
        )

        # 持久化
        key = self._plan_key(plan_id)
        await storage.set(key, json.dumps(plan.to_dict(), ensure_ascii=False), ttl=self.default_ttl)

        # 更新索引
        await self._add_to_index(plan_id)

        # 关联会话 → 计划
        session_key = self._session_key(session_id)
        await storage.set(session_key, plan_id, ttl=self.default_ttl)

        logger.info(f"[TaskPersistence] Plan written: {plan_id} ({len(todos)} todos)")
        return plan

    async def update_todo(
        self,
        plan_id: str,
        todo_index: int,
        status: str = None,
        result: Dict = None,
        error: str = None,
    ) -> Optional[TaskPlan]:
        """更新单个任务状态。

        Args:
            plan_id: 计划 ID
            todo_index: 任务索引
            status: 新状态
            result: 执行结果
            error: 错误信息

        Returns:
            更新后的 TaskPlan，或 None（计划不存在）
        """
        plan = await self.get_plan(plan_id)
        if not plan:
            logger.warning(f"[TaskPersistence] Plan not found: {plan_id}")
            return None

        if todo_index < 0 or todo_index >= len(plan.todos):
            logger.warning(f"[TaskPersistence] Invalid todo index: {todo_index}")
            return None

        todo = plan.todos[todo_index]

        if status:
            todo.status = TaskStatus(status)
        if result is not None:
            todo.result = result
        if error is not None:
            todo.error = error

        # 时间戳
        if status == "in_progress" and not todo.started_at:
            todo.started_at = datetime.now().isoformat()
        if status in ("completed", "failed", "skipped") and not todo.completed_at:
            todo.completed_at = datetime.now().isoformat()

        # 更新计划级状态
        plan.current_step = todo_index + 1
        plan.updated_at = datetime.now().isoformat()

        all_done = all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)
            for t in plan.todos
        )
        if all_done:
            plan.status = TaskStatus.COMPLETED
            plan.completed_at = datetime.now().isoformat()

        # 持久化
        await self._save_plan(plan)
        logger.info(f"[TaskPersistence] Updated todo {todo_index} in {plan_id}: {status or 'no change'}")
        return plan

    async def get_plan(self, plan_id: str) -> Optional[TaskPlan]:
        """获取任务计划"""
        storage = await self._ensure_storage()
        key = self._plan_key(plan_id)
        data = await storage.get(key)
        if data:
            return TaskPlan.from_dict(json.loads(data))
        return None

    async def _save_plan(self, plan: TaskPlan):
        """保存任务计划到存储"""
        storage = await self._ensure_storage()
        key = self._plan_key(plan.plan_id)
        plan.updated_at = datetime.now().isoformat()
        await storage.set(key, json.dumps(plan.to_dict(), ensure_ascii=False), ttl=self.default_ttl)

    async def _add_to_index(self, plan_id: str):
        """添加到计划索引"""
        storage = await self._ensure_storage()
        index_key = self._index_key()
        existing = await storage.get(index_key)
        index = json.loads(existing) if existing else []
        if plan_id not in index:
            index.append(plan_id)
            await storage.set(index_key, json.dumps(index), ttl=self.default_ttl)

    async def _remove_from_index(self, plan_id: str):
        """从计划索引中移除"""
        storage = await self._ensure_storage()
        index_key = self._index_key()
        existing = await storage.get(index_key)
        if existing:
            index = json.loads(existing)
            if plan_id in index:
                index.remove(plan_id)
                await storage.set(index_key, json.dumps(index), ttl=self.default_ttl)

    # ── Agent State 持久化 ───────────────────────────────────

    async def save_agent_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        checkpoint_id: str = None,
    ) -> str:
        """保存完整 Agent 状态（用于重启恢复）。

        Args:
            session_id: 会话 ID
            state: Agent 状态字典（LangGraph State 或其他）
            checkpoint_id: 检查点 ID（与 LangGraph checkpointer 关联）

        Returns:
            state_key: 状态存储键
        """
        storage = await self._ensure_storage()
        key = self._state_key(session_id)

        snapshot = {
            "session_id": session_id,
            "checkpoint_id": checkpoint_id or str(uuid.uuid4().hex[:8]),
            "saved_at": datetime.now().isoformat(),
            "state": state,
        }

        await storage.set(key, json.dumps(snapshot, ensure_ascii=False), ttl=self.default_ttl)
        logger.info(f"[TaskPersistence] Agent state saved: {session_id}")
        return key

    async def load_agent_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """加载 Agent 状态（重启恢复）。

        Returns:
            保存的 Agent 状态快照，或 None（不存在）
        """
        storage = await self._ensure_storage()
        key = self._state_key(session_id)
        data = await storage.get(key)
        if data:
            snapshot = json.loads(data)
            logger.info(f"[TaskPersistence] Agent state loaded: {session_id}")
            return snapshot
        return None

    async def delete_agent_state(self, session_id: str) -> bool:
        """删除 Agent 状态（任务完成后清理）"""
        storage = await self._ensure_storage()
        key = self._state_key(session_id)
        return await storage.delete(key)

    # ── 重启恢复 API ────────────────────────────────────────

    async def get_pending_plans(self) -> List[TaskPlan]:
        """获取所有未完成的计划（用于重启恢复）。

        扫描所有已知计划，返回状态不是 COMPLETED 的。
        """
        storage = await self._ensure_storage()
        index_key = self._index_key()
        index_data = await storage.get(index_key)

        if not index_data:
            return []

        plan_ids = json.loads(index_data)
        pending = []

        for plan_id in plan_ids:
            plan = await self.get_plan(plan_id)
            if plan and not plan.is_completed:
                pending.append(plan)

        # 按创建时间排序（最新的在前）
        pending.sort(key=lambda p: p.created_at, reverse=True)
        return pending

    async def get_session_plan(self, session_id: str) -> Optional[TaskPlan]:
        """获取会话的当前计划"""
        storage = await self._ensure_storage()
        key = self._session_key(session_id)
        plan_id = await storage.get(key)
        if plan_id:
            return await self.get_plan(plan_id)
        return None

    async def resume_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """恢复会话（计划 + 状态）。

        重启时调用此方法，返回：
        {
            "plan": TaskPlan or None,
            "state": Agent state dict or None,
            "resumable": bool   # 是否可以继续执行
        }
        """
        plan = await self.get_session_plan(session_id)
        state_snapshot = await self.load_agent_state(session_id)

        if not plan and not state_snapshot:
            return {"plan": None, "state": None, "resumable": False}

        resumable = (
            plan is not None
            and not plan.is_completed
            and plan.replan_count < plan.max_replan
        )

        return {
            "plan": plan,
            "state": state_snapshot.get("state") if state_snapshot else None,
            "resumable": resumable,
        }

    # ── 快照与恢复 ──────────────────────────────────────────

    async def snapshot(self, session_id: str) -> Dict[str, Any]:
        """创建完整的会话快照（用于检查点）"""
        plan = await self.get_session_plan(session_id)
        state = await self.load_agent_state(session_id)

        return {
            "session_id": session_id,
            "snapshot_time": datetime.now().isoformat(),
            "plan": plan.to_dict() if plan else None,
            "state": state,
        }

    async def restore(
        self,
        session_id: str,
        snapshot: Dict[str, Any],
    ) -> bool:
        """从快照恢复会话"""
        plan_data = snapshot.get("plan")
        state = snapshot.get("state")

        if plan_data:
            plan = TaskPlan.from_dict(plan_data)
            await self._save_plan(plan)
            storage = await self._ensure_storage()
            await storage.set(self._session_key(session_id), plan.plan_id, ttl=self.default_ttl)

        if state:
            await self.save_agent_state(session_id, state.get("state", state))

        logger.info(f"[TaskPersistence] Restored session: {session_id}")
        return True

    # ── 统计与清理 ─────────────────────────────────────────

    async def get_stats(self) -> Dict[str, Any]:
        """获取持久化统计"""
        storage = await self._ensure_storage()
        pending = await self.get_pending_plans()
        index_key = self._index_key()
        index_data = await storage.get(index_key)
        total = len(json.loads(index_data)) if index_data else 0

        return {
            "total_plans": total,
            "pending_plans": len(pending),
            "completed_plans": total - len(pending),
            "storage_backend": type(self._storage).__name__,
        }

    async def cleanup(self, older_than_days: int = 30):
        """清理过期的已完成计划"""
        storage = await self._ensure_storage()
        index_key = self._index_key()
        index_data = await storage.get(index_key)

        if not index_data:
            return

        plan_ids = json.loads(index_data)
        cutoff = datetime.now().isoformat()
        removed = 0

        for plan_id in list(plan_ids):
            plan = await self.get_plan(plan_id)
            if plan and plan.is_completed and plan.completed_at:
                if plan.completed_at < cutoff:
                    await storage.delete(self._plan_key(plan_id))
                    await self._remove_from_index(plan_id)
                    removed += 1

        logger.info(f"[TaskPersistence] Cleaned up {removed} expired plans")


# ── write_todos 工具函数 ────────────────────────────────────

async def write_todos_tool(
    persistence: TaskPersistence,
    session_id: str,
    todos_json: str,
    title: str = "",
) -> str:
    """write_todos 工具（供 Agent 调用）。

    Agent 规划完成后，通过此工具将计划持久化。

    Args:
        persistence: TaskPersistence 实例
        session_id: 会话 ID
        todos_json: JSON 字符串，格式 [{"title": "...", "agent": "...", ...}, ...]
        title: 计划标题

    Returns:
        持久化结果摘要
    """
    try:
        todo_dicts = json.loads(todos_json)
    except json.JSONDecodeError as e:
        return f"错误: todos_json 不是有效的 JSON: {e}"

    todos = []
    for td in todo_dicts:
        todos.append(TaskItem(
            title=td.get("title", ""),
            agent=td.get("agent", "unknown"),
            status=TaskStatus(td.get("status", "pending")),
            description=td.get("description", ""),
            context=td.get("context", {}),
            depends_on=td.get("depends_on", []),
        ))

    plan = await persistence.write_todos(
        session_id=session_id,
        todos=todos,
        title=title,
    )

    return json.dumps({
        "plan_id": plan.plan_id,
        "total_steps": plan.total_steps,
        "message": f"已写入 {len(todos)} 个任务到计划 {plan.plan_id}",
    }, ensure_ascii=False)


# ── 工厂函数 ─────────────────────────────────────────────────

async def create_task_persistence(
    backend: str = "auto",
    storage_dir: str = "data/task_persistence",
) -> TaskPersistence:
    """创建预配置的任务持久化实例"""
    tp = TaskPersistence(
        backend=backend,
        storage_dir=storage_dir,
    )
    # 触发存储初始化
    await tp._ensure_storage()
    return tp
