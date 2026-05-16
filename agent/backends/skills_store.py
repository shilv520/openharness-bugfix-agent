"""
SkillsStoreBackend — 技能持久化后端
====================================

PDF Harness Engineering: CompositeBackend routes["/persisted-skills/"] = StoreBackend(MongoDB)

本项目用 ChromaDB + Redis 替代 MongoDB，将 /persisted-skills/ 路径映射到
agent/skills/persistence.py 的 SkillPersistence。

路径约定:
  /persisted-skills/                    → 列出所有 Skills
  /persisted-skills/{skill_name}/       → 列出 Skill 目录内容
  /persisted-skills/{skill_name}/SKILL.md → 读写 Skill 完整内容
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .base import Backend

logger = logging.getLogger("backends.skills_store")


class SkillsStoreBackend(Backend):
    """将 /persisted-skills/ 路径路由到 SkillPersistence (ChromaDB + Redis)。"""

    def __init__(self, skill_manager=None):
        """
        Args:
            skill_manager: SkillManager 实例（可选，延迟加载）
        """
        self._manager = skill_manager
        self._store: Dict[str, str] = {}  # 开发模式回退

    def _get_manager(self):
        """延迟初始化 SkillManager"""
        if self._manager is not None:
            return self._manager

        try:
            from agent.skills.manager import SkillManager
            self._manager = SkillManager()
            self._manager.sync_all()
            logger.info("[SkillsStoreBackend] SkillManager loaded")
        except Exception as e:
            logger.warning(f"[SkillsStoreBackend] SkillManager unavailable: {e}")
            self._manager = None

        return self._manager

    def _parse_path(self, path: str) -> Optional[dict]:
        """解析 /persisted-skills/{skill_name}/{file} 为结构化信息。

        Returns:
            {"skill_name": str, "file": str}  or None (listing root)
        """
        clean = path.lstrip("/")
        parts = clean.split("/")
        if not clean.startswith("persisted-skills"):
            return None

        if len(parts) == 1:
            # "/persisted-skills" → listing root
            return {"skill_name": None, "file": None}

        skill_name = parts[1] if len(parts) > 1 else None
        filename = parts[-1] if len(parts) > 2 else "SKILL.md"
        return {"skill_name": skill_name, "file": filename}

    # ── Backend 接口 ────────────────────────────────────────────

    async def read(self, path: str) -> str:
        info = self._parse_path(path)
        if not info or not info["skill_name"]:
            raise FileNotFoundError(f"[SkillsStore] invalid path: {path}")

        skill_name = info["skill_name"]
        mgr = self._get_manager()

        if mgr:
            content = mgr.load_full_skill(skill_name)
            if content:
                return content

        # Fallback
        fallback_key = f"skill:{skill_name}"
        if fallback_key in self._store:
            return self._store[fallback_key]

        raise FileNotFoundError(f"[SkillsStore] skill not found: {skill_name}")

    async def write(self, path: str, content: str) -> bool:
        info = self._parse_path(path)
        if not info or not info["skill_name"]:
            return False

        skill_name = info["skill_name"]
        mgr = self._get_manager()

        if mgr:
            # 如果 Skill 已存在，更新 frontmatter 后重新持久化
            existing_fm = mgr.get_frontmatter(skill_name)
            if existing_fm:
                from agent.skills.progressive import extract_frontmatter_only
                new_fm = extract_frontmatter_only(content)
                if not new_fm.name:
                    new_fm.name = skill_name
                new_fm.scope = existing_fm.scope
                new_fm.path = existing_fm.path or f"skills/{existing_fm.scope.value}/{skill_name}/SKILL.md"
                mgr.persistence.store_skill_content(new_fm, content)
                mgr._content_cache[skill_name] = content
                logger.info(f"[SkillsStore] updated: {skill_name}")
                return True
            else:
                # 新建 Skill
                from agent.skills.progressive import extract_frontmatter_only, get_body_only
                fm = extract_frontmatter_only(content)
                body = get_body_only(content)
                if not fm.name:
                    fm.name = skill_name
                new_fm = await mgr.create_skill(
                    name=fm.name or skill_name,
                    description=fm.description or f"Skill: {skill_name}",
                    body=body or content,
                    tags=fm.tags,
                )
                return new_fm is not None

        # Fallback: 内存中缓存
        self._store[f"skill:{skill_name}"] = content
        logger.info(f"[SkillsStore] fallback write: {skill_name}")
        return True

    async def list_dir(self, path: str) -> List[str]:
        info = self._parse_path(path)

        if info and info["skill_name"] is None:
            # /persisted-skills/ → 列出所有 Skills
            mgr = self._get_manager()
            if mgr:
                fms = mgr.list_frontmatters()
                return [f"{fm.name}/" for fm in fms]
            return [f"{k.replace('skill:', '')}/" for k in self._store if k.startswith("skill:")]

        if info and info["skill_name"]:
            # /persisted-skills/{name}/ → 列出 Skill 目录内容
            return ["SKILL.md"]

        return []

    async def exists(self, path: str) -> bool:
        info = self._parse_path(path)
        if not info or not info["skill_name"]:
            return False

        mgr = self._get_manager()
        if mgr:
            fm = mgr.get_frontmatter(info["skill_name"])
            return fm is not None

        return f"skill:{info['skill_name']}" in self._store

    async def delete(self, path: str) -> bool:
        info = self._parse_path(path)
        if not info or not info["skill_name"]:
            return False

        skill_name = info["skill_name"]
        mgr = self._get_manager()
        ok = False
        if mgr:
            ok = mgr.persistence.delete_skill(skill_name)
            mgr._content_cache.pop(skill_name, None)
            # 同时删除文件系统中的目录
            import shutil
            for scope_dir in ["main", "user", "downloaded", "skill-management"]:
                skill_dir = mgr.skills_root / scope_dir / skill_name
                if skill_dir.exists():
                    shutil.rmtree(skill_dir)
                    logger.info(f"[SkillsStore] deleted files: {skill_dir}")
            logger.info(f"[SkillsStore] deleted: {skill_name}")
        else:
            self._store.pop(f"skill:{skill_name}", None)
            ok = True

        return ok

    async def mkdir(self, path: str) -> bool:
        # SkillsStoreBackend 不需要真实目录
        return True
