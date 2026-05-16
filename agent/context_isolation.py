"""
Context Isolation — 子Agent上下文隔离管理器
==============================================

PDF Harness Engineering: 子Agent上下文隔离

核心概念:
  1. Fork — 父Agent派生子Agent时，复制一份干净的上下文
  2. Isolate — 子Agent在自己的沙盒中运行，不污染父Agent上下文
  3. Merge — 子Agent完成后，只将结构化结果合并回父Agent
  4. Clean — 子Agent销毁时，清理所有临时上下文

与 SubAgent 的关系:
  SubAgent 负责 Agent 执行（ReAct循环），ContextIsolation 负责上下文管理。
  两者配合: ContextIsolation.fork() → SubAgent.run() → ContextIsolation.merge()
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("context.isolation")


@dataclass
class ContextBoundary:
    """上下文边界 — 定义一个Agent的上下文作用域。

    每个Agent（主Agent或子Agent）有一个ContextBoundary，
    定义了哪些信息在其上下文中可见。
    """

    boundary_id: str
    parent_id: Optional[str]     # 父边界ID（None=主Agent）
    messages: List[Dict[str, str]] = field(default_factory=list)
    facts: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # 隔离规则
    read_only_keys: List[str] = field(default_factory=list)    # 只读的fact key
    writable_keys: List[str] = field(default_factory=list)     # 可写的fact key
    hidden_keys: List[str] = field(default_factory=list)       # 完全不可见的fact key

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def is_root(self) -> bool:
        return self.parent_id is None

    @property
    def depth(self) -> int:
        """上下文树的深度（0=主Agent）"""
        return 0 if self.is_root else 1  # 简化实现，实际可递归计算


@dataclass
class IsolationResult:
    """上下文隔离操作的结果"""
    child_boundary: ContextBoundary
    snapshot: Dict[str, Any]        # fork时的快照（用于回滚）
    child_messages: List[Dict[str, str]]  # 子Agent产生的消息
    structured_output: Dict[str, Any]     # 子Agent的结构化输出
    merge_messages: List[Dict[str, str]]  # 需要合并回父Agent的消息
    merge_facts: Dict[str, str]           # 需要合并回父Agent的事实


class ContextIsolation:
    """子Agent上下文隔离管理器。

    对应PDF项目中子Agent的上下文隔离机制。

    工作流程:
      1. fork(boundary, agent_type, task) → 创建子边界
      2. 子Agent在子边界中运行
      3. merge(parent, child_result) → 只合并结构化结果
      4. cleanup(child_boundary) → 清理子边界

    用法:
        iso = ContextIsolation(memory_manager)

        # Fork: 创建子Agent上下文
        child_ctx = iso.fork(parent_boundary, "code-reviewer", "审查代码")

        # ... 子Agent运行在 child_ctx 中 ...

        # Merge: 合并结果
        iso.merge(parent_boundary, child_ctx, sub_result)

        # Clean: 清理
        iso.cleanup(child_ctx)
    """

    def __init__(self, memory_manager=None, llm_config: Dict[str, Any] = None):
        """
        Args:
            memory_manager: HierarchicalMemory实例（用于持久化隔离状态）
            llm_config: LLM配置
        """
        self.memory = memory_manager
        self.llm_config = llm_config or {}

        # 活跃边界注册表
        self._boundaries: Dict[str, ContextBoundary] = {}
        self._children: Dict[str, List[str]] = {}  # parent_id → [child_ids]

    # ── 公开 API ──────────────────────────────────────────────

    def create_root_boundary(
        self,
        user_id: str = "default",
        system_prompt: str = "",
    ) -> ContextBoundary:
        """创建主Agent的根上下文边界"""
        boundary = ContextBoundary(
            boundary_id=f"root-{uuid.uuid4().hex[:8]}",
            parent_id=None,
            messages=[{"role": "system", "content": system_prompt}] if system_prompt else [],
            metadata={"user_id": user_id, "type": "root"},
        )
        self._boundaries[boundary.boundary_id] = boundary
        self._children[boundary.boundary_id] = []
        logger.info(f"[Isolation] Created root boundary: {boundary.boundary_id}")
        return boundary

    def fork(
        self,
        parent: ContextBoundary,
        agent_type: str,
        task_prompt: str,
        context: Dict[str, Any] = None,
        inherit_rules: Dict[str, str] = None,
    ) -> ContextBoundary:
        """从父边界Fork一个子边界。

        Args:
            parent: 父上下文边界
            agent_type: 子Agent类型名称
            task_prompt: 分配给子Agent的任务
            context: 任务上下文数据
            inherit_rules: 事实继承规则 {'fact_key': 'read_only'|'writable'|'hidden'}

        Returns:
            子ContextBoundary
        """
        inherit = inherit_rules or {}

        # 过滤父Agent的事实（根据继承规则）
        child_facts = {}
        read_only = []
        writable = []
        hidden = []

        for key, value in parent.facts.items():
            rule = inherit.get(key, "hidden")  # 默认隐藏
            if rule == "hidden":
                hidden.append(key)
            elif rule == "read_only":
                child_facts[key] = value
                read_only.append(key)
            elif rule == "writable":
                child_facts[key] = value
                writable.append(key)

        # 创建子边界
        child = ContextBoundary(
            boundary_id=f"child-{agent_type}-{uuid.uuid4().hex[:8]}",
            parent_id=parent.boundary_id,
            messages=[
                {"role": "system", "content": f"你是一个{agent_type}子Agent。你的任务是: {task_prompt}"},
            ],
            facts=child_facts,
            metadata={
                "agent_type": agent_type,
                "task": task_prompt,
                "context": context or {},
                "type": "child",
                "parent_boundary": parent.boundary_id,
            },
            read_only_keys=read_only,
            writable_keys=writable,
            hidden_keys=hidden,
        )

        # 注册
        self._boundaries[child.boundary_id] = child
        if parent.boundary_id not in self._children:
            self._children[parent.boundary_id] = []
        self._children[parent.boundary_id].append(child.boundary_id)

        logger.info(
            f"[Isolation] Forked {child.boundary_id} from {parent.boundary_id} "
            f"({len(read_only)} read-only, {len(writable)} writable, {len(hidden)} hidden facts)"
        )

        return child

    def merge(
        self,
        parent: ContextBoundary,
        child: ContextBoundary,
        child_result: Dict[str, Any],
        merge_strategy: str = "structured_only",
    ) -> Dict[str, Any]:
        """将子Agent结果合并回父Agent上下文。

        Args:
            parent: 父边界
            child: 子边界
            child_result: 子Agent的执行结果 {"report": ..., "structured_output": {...}}
            merge_strategy:
              - "structured_only": 只合并structured_output（默认，最干净）
              - "summary": 合并report + structured_output
              - "full": 合并所有内容（含消息）

        Returns:
            合并后更新的父边界信息
        """
        report = child_result.get("report", "")
        structured = child_result.get("structured_output", {})

        update_info = {
            "merged_from": child.boundary_id,
            "agent_type": child.metadata.get("agent_type", "unknown"),
            "strategy": merge_strategy,
            "facts_added": 0,
            "messages_added": 0,
        }

        if merge_strategy == "structured_only":
            # 只合并结构化输出 → 最干净
            for key, value in structured.items():
                if key not in parent.facts:
                    parent.facts[f"sub_{child.metadata.get('agent_type', 'unknown')}_{key}"] = (
                        json.dumps(value, ensure_ascii=False)
                        if not isinstance(value, str) else value
                    )
                    update_info["facts_added"] += 1

            # 添加一条合并摘要消息
            parent.messages.append({
                "role": "system",
                "content": f"[子Agent {child.metadata.get('agent_type')} 完成] {report[:200]}",
            })
            update_info["messages_added"] = 1

        elif merge_strategy == "summary":
            # 合并report + structured
            parent.facts[f"sub_{child.metadata.get('agent_type')}_report"] = report
            update_info["facts_added"] += 1

            for key, value in structured.items():
                parent.facts[f"sub_{child.metadata.get('agent_type')}_{key}"] = (
                    json.dumps(value, ensure_ascii=False)
                    if not isinstance(value, str) else value
                )
                update_info["facts_added"] += 1

            parent.messages.append({
                "role": "system",
                "content": f"[子Agent {child.metadata.get('agent_type')} 报告]\n{report[:500]}",
            })
            update_info["messages_added"] = 1

        elif merge_strategy == "full":
            # 合并所有内容
            for key, value in child.facts.items():
                parent.facts[f"child_{key}"] = value
                update_info["facts_added"] += 1

            parent.messages.extend(child.messages)
            update_info["messages_added"] = len(child.messages)

        logger.info(
            f"[Isolation] Merged {child.boundary_id} → {parent.boundary_id} "
            f"({merge_strategy}: {update_info['facts_added']} facts, "
            f"{update_info['messages_added']} messages)"
        )

        return update_info

    def cleanup(self, boundary: ContextBoundary):
        """清理子Agent上下文边界。

        子Agent完成使命后，清理其上下文，释放资源。
        """
        # 从父边界解除注册
        if boundary.parent_id and boundary.parent_id in self._children:
            if boundary.boundary_id in self._children[boundary.parent_id]:
                self._children[boundary.parent_id].remove(boundary.boundary_id)

        # 递归清理子边界
        if boundary.boundary_id in self._children:
            for child_id in list(self._children.get(boundary.boundary_id, [])):
                if child_id in self._boundaries:
                    self.cleanup(self._boundaries[child_id])
            del self._children[boundary.boundary_id]

        # 删除边界
        self._boundaries.pop(boundary.boundary_id, None)
        logger.info(f"[Isolation] Cleaned up boundary: {boundary.boundary_id}")

    def get_boundary(self, boundary_id: str) -> Optional[ContextBoundary]:
        """获取上下文边界"""
        return self._boundaries.get(boundary_id)

    def get_children(self, parent_id: str) -> List[ContextBoundary]:
        """获取父边界的所有子边界"""
        child_ids = self._children.get(parent_id, [])
        return [self._boundaries[cid] for cid in child_ids if cid in self._boundaries]

    def add_message(self, boundary: ContextBoundary, role: str, content: str):
        """向指定边界添加消息"""
        boundary.messages.append({"role": role, "content": content})

    def add_fact(self, boundary: ContextBoundary, key: str, value: str):
        """向指定边界添加事实（遵守可写规则）"""
        if key in boundary.hidden_keys:
            logger.warning(f"[Isolation] Cannot add hidden fact: {key}")
            return
        if key in boundary.read_only_keys:
            logger.warning(f"[Isolation] Cannot modify read-only fact: {key}")
            return
        boundary.facts[key] = value

    def get_stats(self) -> Dict[str, Any]:
        """获取隔离系统统计"""
        roots = [b for b in self._boundaries.values() if b.is_root]
        children = [b for b in self._boundaries.values() if not b.is_root]

        return {
            "total_boundaries": len(self._boundaries),
            "root_boundaries": len(roots),
            "child_boundaries": len(children),
            "active_children": sum(
                len(ids) for ids in self._children.values()
            ),
        }

    def snapshot(self, boundary: ContextBoundary) -> Dict[str, Any]:
        """创建边界的完整快照（用于回滚或检查点）"""
        return {
            "boundary_id": boundary.boundary_id,
            "parent_id": boundary.parent_id,
            "messages": list(boundary.messages),
            "facts": dict(boundary.facts),
            "metadata": dict(boundary.metadata),
            "read_only_keys": list(boundary.read_only_keys),
            "writable_keys": list(boundary.writable_keys),
            "hidden_keys": list(boundary.hidden_keys),
            "created_at": boundary.created_at,
            "snapshot_time": datetime.now().isoformat(),
        }

    def restore(self, boundary: ContextBoundary, snapshot: Dict[str, Any]):
        """从快照恢复边界"""
        boundary.messages = list(snapshot.get("messages", []))
        boundary.facts = dict(snapshot.get("facts", {}))
        boundary.metadata = dict(snapshot.get("metadata", {}))
        boundary.read_only_keys = list(snapshot.get("read_only_keys", []))
        boundary.writable_keys = list(snapshot.get("writable_keys", []))
        boundary.hidden_keys = list(snapshot.get("hidden_keys", []))
        logger.info(f"[Isolation] Restored boundary from snapshot: {boundary.boundary_id}")


# ── 便捷函数 ─────────────────────────────────────────────────

def create_context_isolation(
    memory_manager=None,
    llm_config: Dict = None,
) -> ContextIsolation:
    """创建预配置的上下文隔离管理器"""
    return ContextIsolation(memory_manager=memory_manager, llm_config=llm_config)
