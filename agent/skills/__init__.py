"""
Skills 动态加载 + 自我进化模块
==============================

PDF Harness Engineering pattern: 4-Stage Skills Self-Evolution

  Stage 1 (Sync):       SkillsSyncMiddleware — 增量同步
  Stage 2 (Discover):   SkillManager.list_frontmatters() / load_full_skill()
  Stage 3 (Create/DL):  download_skill() / create_skill()
  Stage 4 (Persist):    assign_skill() / restore_from_store()

Legacy components (kept for backward compatibility):
  - SkillDefinition, SkillRegistry, SkillLoader, load_skill_registry
  - IntentMatcher (2-layer routing: Embedding + LLM)
"""

# ── New: 4-Stage Self-Evolution ────────────────────────────────

from .progressive import (
    SkillFrontmatter,
    SkillScope,
    parse_frontmatter,
    extract_frontmatter_only,
    load_full_content,
    get_body_only,
    compute_checksum,
    make_frontmatter_block,
)

from .persistence import (
    SkillPersistence,
)

from .manager import (
    SkillManager,
    get_skill_manager,
)

from .tools import (
    SkillToolsHandler,
    get_tool_definitions,
    build_skill_tools_for_agent,
    run_tool_sync,
)

from .sync_middleware import (
    SkillsSyncMiddleware,
)

from .skill_router import (
    SkillRouter,
    MatchResult,
    get_skill_router,
    AGENT_KEYWORD_TIERS,
)

# ── Legacy: Static Skill Loading (backward compatible) ──────────

from .loader import (
    SkillDefinition,
    SkillRegistry,
    SkillLoader,
    load_skill_registry,
    get_bundled_skills,
    BUNDLED_SKILLS,
)

# ── Legacy: Intent Matching ─────────────────────────────────────

from ..intent_matcher import (
    IntentMatcher,
    EmbeddingMatcher,
    LLMRouter,
)

__all__ = [
    # Progressive disclosure
    "SkillFrontmatter",
    "SkillScope",
    "parse_frontmatter",
    "extract_frontmatter_only",
    "load_full_content",
    "get_body_only",
    "compute_checksum",
    "make_frontmatter_block",
    # Persistence
    "SkillPersistence",
    # Manager (4-stage lifecycle)
    "SkillManager",
    "get_skill_manager",
    # Tools (Agent-callable)
    "SkillToolsHandler",
    "get_tool_definitions",
    "build_skill_tools_for_agent",
    "run_tool_sync",
    # Sync middleware
    "SkillsSyncMiddleware",
    # Skill→Agent Router
    "SkillRouter",
    "MatchResult",
    "get_skill_router",
    "AGENT_KEYWORD_TIERS",
    # Legacy
    "SkillDefinition",
    "SkillRegistry",
    "SkillLoader",
    "load_skill_registry",
    "get_bundled_skills",
    "BUNDLED_SKILLS",
    "IntentMatcher",
    "EmbeddingMatcher",
    "LLMRouter",
]
