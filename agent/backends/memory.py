"""
MemoryBackend — 记忆存储后端
=============================

PDF Harness Engineering: CompositeBackend routes["/memories/"] = StoreBackend(MongoDB)

本项目用 Redis + ChromaDB 替代 MongoDB，将 /memories/ 路径映射到
agent/redis_memory.py 的 HierarchicalMemory。

路径约定:
  /memories/{user_id}/preferences.md  → 所有用户事实（Markdown 序列化）
  /memories/{user_id}/short-term/     → 当前会话上下文
  /memories/{user_id}/long-term/      → 持久化事实列表
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

from .base import Backend

logger = logging.getLogger("backends.memory")


class MemoryBackend(Backend):
    """将 /memories/ 路径路由到 HierarchicalMemory (Redis + ChromaDB)。"""

    def __init__(self, memory_manager=None):
        """
        Args:
            memory_manager: HierarchicalMemory 实例（可选，延迟加载）
        """
        self._memory = memory_manager
        self._store: Dict[str, str] = {}  # 开发模式回退

    def _get_memory(self):
        """延迟初始化 HierarchicalMemory"""
        if self._memory is not None:
            return self._memory

        try:
            from agent.redis_memory import HierarchicalMemory
            self._memory = HierarchicalMemory()
            logger.info("[MemoryBackend] HierarchicalMemory loaded")
        except Exception as e:
            logger.warning(f"[MemoryBackend] HierarchicalMemory unavailable: {e}")
            self._memory = None

        return self._memory

    def _parse_path(self, path: str) -> Optional[dict]:
        """解析 /memories/{user_id}/{subpath} 为结构化信息。

        Returns:
            {"user_id": str, "file": str}  or None
        """
        clean = path.lstrip("/")
        parts = clean.split("/")
        # parts[0] = "memories", parts[1] = user_id, parts[2:] = subpath
        if len(parts) < 2 or parts[0] != "memories":
            return None
        user_id = parts[1] if len(parts) > 1 else "default"
        filename = parts[-1] if len(parts) > 2 else "preferences.md"
        return {"user_id": user_id, "file": filename}

    # ── preferences.md 序列化 ──────────────────────────────────

    def _facts_to_markdown(self, facts: Dict[str, str]) -> str:
        """将用户事实字典序列化为 Markdown"""
        lines = ["# User Preferences & Facts\n"]
        if not facts:
            lines.append("_(empty)_\n")
        else:
            for key, value in sorted(facts.items()):
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)

    def _markdown_to_facts(self, md: str) -> Dict[str, str]:
        """从 Markdown 解析事实字典

        格式: - **key**: value
        """
        facts: Dict[str, str] = {}
        for line in md.splitlines():
            line = line.strip()
            if line.startswith("- **") and "**: " in line:
                line = line[4:]  # remove "- **"
                key_end = line.find("**")
                if key_end == -1:
                    continue
                key = line[:key_end]
                val = line[key_end + 3:]  # skip "**: "
                facts[key] = val
        return facts

    # ── Backend 接口实现 ──────────────────────────────────────

    async def read(self, path: str) -> str:
        info = self._parse_path(path)
        if not info:
            raise FileNotFoundError(f"[MemoryBackend] invalid path: {path}")

        user_id = info["user_id"]
        filename = info["file"]

        mem = self._get_memory()

        if filename == "preferences.md":
            if mem:
                facts = await mem.recall_all_facts(user_id)
            else:
                facts = {}
            return self._facts_to_markdown(facts)

        if filename == "context.json":
            if mem:
                ctx = await mem.get_context("session:latest", user_id)
            else:
                ctx = {}
            return json.dumps(ctx or {}, ensure_ascii=False, indent=2)

        raise FileNotFoundError(f"[MemoryBackend] unknown file: {filename}")

    async def write(self, path: str, content: str) -> bool:
        info = self._parse_path(path)
        if not info:
            return False

        user_id = info["user_id"]
        filename = info["file"]
        mem = self._get_memory()

        if filename == "preferences.md":
            facts = self._markdown_to_facts(content)
            if mem:
                import asyncio
                for key, val in facts.items():
                    await mem.remember_fact(user_id, key, val)
                # 等待异步写入完成
                await asyncio.sleep(0.5)
            else:
                for key, val in facts.items():
                    self._store[f"fact:{user_id}:{key}"] = val
            logger.info(f"[MemoryBackend] wrote {len(facts)} facts for {user_id}")
            return True

        return False

    async def list_dir(self, path: str) -> List[str]:
        clean = path.lstrip("/")
        parts = clean.split("/")

        if clean == "memories" or clean == "":
            return ["default/"]

        if clean.startswith("memories/") or (len(parts) >= 1 and parts[0] == "memories"):
            return ["preferences.md", "context.json"]

        return []

    async def exists(self, path: str) -> bool:
        info = self._parse_path(path)
        if not info:
            return False
        filename = info["file"]
        return filename in ("preferences.md", "context.json")

    async def delete(self, path: str) -> bool:
        info = self._parse_path(path)
        if not info:
            return False

        user_id = info["user_id"]
        filename = info["file"]
        mem = self._get_memory()

        if filename == "preferences.md" and mem:
            # 清除该用户的所有事实
            facts = await mem.recall_all_facts(user_id)
            for key in facts:
                self._store.pop(f"fact:{user_id}:{key}", None)
            logger.info(f"[MemoryBackend] cleared {len(facts)} facts for {user_id}")
            return True

        return False

    async def mkdir(self, path: str) -> bool:
        # MemoryBackend 不需要真实目录
        return True
