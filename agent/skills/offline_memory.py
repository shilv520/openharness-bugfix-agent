"""
Offline Memory — SkillDreamer（离线「做梦」机制）
=================================================

灵感来自 Claude Code / Codex 的离线后台总结（见 coding-agent-memory-mechanisms）：

  每日任务结束后，不再有实时请求压力时，系统对 skill 做一次「频率分层」整理：
  - 高频 skill 继续常驻（resident）
  - 低频 skill 归档到 ARCHIVED scope（外置记忆，不参与默认发现，仍可语义检索/按需加载）
  - 归档后重新变热的 skill 恢复到原 scope

「统计机制 = memory」：hit_count / last_hit_at 就是关于 skill 的元记忆，
由本模块在离线时段消费，驱动生命周期升降级，从而给 skill 膨胀找到退路。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from .progressive import SkillFrontmatter, SkillScope

logger = logging.getLogger("skills.offline_memory")


def _parse_ts(ts: str) -> Optional[datetime]:
    """解析 ISO 时间戳，失败返回 None。"""
    if not ts:
        return None
    try:
        # 兼容有无时区的 ISO 字符串
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


class SkillDreamer:
    """离线对 skill 做频率分层整理（archive / restore / keep）。"""

    def __init__(
        self,
        manager=None,
        archive_after_days: int = 7,
        restore_recent_days: int = 3,
    ):
        # 延迟 import，避免循环依赖（manager → offline_memory → manager）
        from .manager import SkillManager

        self.manager: SkillManager = manager or SkillManager()
        self.archive_after_days = archive_after_days
        self.restore_recent_days = restore_recent_days

    def _last_activity(self, fm: SkillFrontmatter) -> Optional[datetime]:
        """最近活动时间 = last_hit_at 优先，其次 created_at，其次 updated_at。"""
        return (
            _parse_ts(fm.last_hit_at)
            or _parse_ts(fm.created_at)
            or _parse_ts(fm.updated_at)
        )

    def dream(self, now: Optional[datetime] = None) -> Dict:
        """执行一次离线整理，返回报告。

        Returns:
            {
              "archived": [name, ...],   # 本次归档的低频 skill
              "restored": [name, ...],   # 本次恢复的重新变热 skill
              "kept": [name, ...],       # 保持常驻的高频 skill
              "summary": "...",
              "memory_key": "skill:dream:...",  # 报告写入 memory 的 key
            }
        """
        now = now or datetime.now(timezone.utc)
        archive_cutoff = now - timedelta(days=self.archive_after_days)
        restore_cutoff = now - timedelta(days=self.restore_recent_days)

        archived: List[str] = []
        restored: List[str] = []
        kept: List[str] = []

        # 一次性取全部（含 ARCHIVED），在内存里分层，避免边改边遍历
        all_skills = self.manager.persistence.list_skills()

        for fm in all_skills:
            if fm.scope == SkillScope.ARCHIVED:
                # ── 归档 → 重新变热：最近被命中则恢复 ──
                last_hit = _parse_ts(fm.last_hit_at)
                if last_hit and last_hit >= restore_cutoff:
                    if self.manager.restore_skill(fm.name):
                        restored.append(fm.name)
                continue

            # ── 常驻 → 低频：活动时间超过阈值则归档 ──
            last_activity = self._last_activity(fm)
            if last_activity is None or last_activity < archive_cutoff:
                if self.manager.archive_skill(fm.name):
                    archived.append(fm.name)
            else:
                kept.append(fm.name)

        summary = (
            f"dream @ {now.isoformat()}: "
            f"archived={len(archived)}, restored={len(restored)}, kept={len(kept)}"
        )
        report = {
            "archived": archived,
            "restored": restored,
            "kept": kept,
            "summary": summary,
            "at": now.isoformat(),
        }

        # 决策报告本身写入 memory（「用 memory 兜底」：可审计、可回滚）
        memory_key = f"skill:dream:{now.strftime('%Y%m%d')}"
        self.manager.persistence.store_memory(memory_key, report)
        report["memory_key"] = memory_key

        logger.info(f"[SkillDreamer] {summary}")
        return report
