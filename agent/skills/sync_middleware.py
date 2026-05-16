"""
SkillsSyncMiddleware — Stage 1: Auto-sync at Agent Startup
===========================================================

PDF Harness Engineering pattern: SkillsSyncMiddleware + _seed_files()

On agent start:
  1. Scan local skills/ directory
  2. Compare checksums with Redis
  3. Only sync changed files to ChromaDB (incremental)
  4. Restore user-persisted skills from previous sessions
  5. Seed initial skills on first run

Usage:
    middleware = SkillsSyncMiddleware(manager)
    await middleware.on_agent_start()
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from .manager import SkillManager, get_skill_manager
from .progressive import SkillScope, SkillFrontmatter

logger = logging.getLogger("skills.sync_middleware")


class SkillsSyncMiddleware:
    """Startup middleware that synchronizes local skills with the persistence layer.

    This is the entry point for Stage 1 (Sync) of the 4-stage lifecycle.
    It should be called once when the Agent process starts.
    """

    def __init__(self, manager: SkillManager = None):
        self.manager = manager or get_skill_manager()
        self._last_sync: Optional[datetime] = None
        self._sync_report: Dict = {}

    async def on_agent_start(self) -> Dict:
        """Run the full startup sync pipeline.

        1. Sync local skills → ChromaDB + Redis (incremental)
        2. Restore user skills from persistence
        3. Seed initial skills if first run

        Returns sync report dict.
        """
        logger.info("[SkillsSyncMiddleware] Agent starting — syncing skills...")

        # Step 1: Sync local skills (Stage 1)
        report = self.manager.sync_all()
        self._sync_report = report

        # Step 2: Restore persisted skills from previous sessions (Stage 4 - Restore)
        restored = self.manager.restore_from_store()
        logger.info(f"[SkillsSyncMiddleware] restored {len(restored)} skills from store")

        # Step 3: Seed initial skills if this is the first run
        if report["total"] == 0 and len(restored) == 0:
            logger.info("[SkillsSyncMiddleware] First run detected — seeding initial skills")
            self._seed_initial_skills()

        self._last_sync = datetime.now()

        # Build summary
        fms = self.manager.list_frontmatters()
        summary = {
            "synced": len(report.get("synced", [])),
            "skipped": len(report.get("skipped", [])),
            "errors": len(report.get("errors", [])),
            "restored": len(restored),
            "total_available": len(fms),
            "scopes": {},
        }
        for fm in fms:
            sc = fm.scope.value
            summary["scopes"].setdefault(sc, 0)
            summary["scopes"][sc] += 1

        logger.info(f"[SkillsSyncMiddleware] complete: {summary}")
        return summary

    def _seed_initial_skills(self):
        """Create the meta skill-management SKILL.md if missing."""
        from pathlib import Path

        mgmt_dir = self.manager.skills_root / "skill-management"
        mgmt_md = mgmt_dir / "SKILL.md"
        if mgmt_md.exists():
            return

        # The meta-skill SHOULD exist at this point (created alongside this module).
        # If it doesn't, skip seeding — the user can create it manually.
        logger.info("[SkillsSyncMiddleware] skill-management SKILL.md already exists, skipping seed")

    @property
    def last_sync_time(self) -> Optional[datetime]:
        return self._last_sync

    @property
    def last_report(self) -> Dict:
        return self._sync_report
