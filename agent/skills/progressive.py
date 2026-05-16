"""
Progressive Disclosure for Skills (Stage 2)
===========================================

PDF Harness Engineering pattern: 渐进式披露
- 启动时只读 frontmatter（name + description + tags + version）
- 需要时才加载完整 SKILL.md body
- Token 消耗降低约 90%

Skills are discovered via:
  1. Agent runs list_skills → sees frontmatter only
  2. Agent calls load_skill(name) → gets complete body
  3. System Prompt never carries full skill text
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("skills.progressive")


class SkillScope(str, Enum):
    """Skill scope / ownership domain"""
    MAIN = "main"            # Core system skills (review, analysis, fix, validation)
    USER = "user"            # User-created skills (persisted across sessions)
    DOWNLOADED = "downloaded"  # Downloaded from external sources
    SKILL_MANAGEMENT = "skill-management"  # Meta-skill for self-management


@dataclass
class SkillFrontmatter:
    """Frontmatter metadata extracted from SKILL.md (no body).

    This is what the Agent sees at discovery time.  The body is only
    loaded when load_full_skill() is called.
    """
    name: str
    description: str
    version: str = "1.0.0"
    scope: SkillScope = SkillScope.MAIN
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""
    path: str = ""            # relative path to SKILL.md on disk
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["scope"] = self.scope.value
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> "SkillFrontmatter":
        scope = data.get("scope", "main")
        if isinstance(scope, str):
            scope = SkillScope(scope)
        return cls(
            name=data.get("name", ""),
            description=data.get("description", ""),
            version=data.get("version", "1.0.0"),
            scope=scope,
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            checksum=data.get("checksum", ""),
            path=data.get("path", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @property
    def discovery_text(self) -> str:
        """One-line summary used during skill discovery (progressive disclosure)."""
        tags_str = ", ".join(self.tags) if self.tags else "none"
        return f"[{self.scope.value}] {self.name} v{self.version} — {self.description} (tags: {tags_str})"


# ── Frontmatter Parsing ────────────────────────────────────────

def parse_frontmatter(content: str) -> SkillFrontmatter:
    """Parse YAML frontmatter block from SKILL.md content."""
    fm = SkillFrontmatter(name="", description="")

    if not content.startswith("---"):
        return fm

    end_idx = content.find("\n---", 4)
    if end_idx == -1:
        return fm

    raw = content[4:end_idx]

    try:
        import yaml
        data = yaml.safe_load(raw) or {}
    except Exception:
        logger.warning("YAML frontmatter parse failed, falling back to manual parse")
        data = _manual_parse(raw)

    if not isinstance(data, dict):
        return fm

    fm.name = data.get("name", "")
    fm.description = data.get("description", "")
    fm.version = str(data.get("version", "1.0.0"))
    fm.tags = data.get("tags", []) or []
    fm.dependencies = data.get("dependencies", []) or []
    fm.created_at = str(data.get("created_at", ""))
    fm.updated_at = str(data.get("updated_at", ""))

    scope_raw = data.get("scope", "main")
    try:
        fm.scope = SkillScope(scope_raw)
    except ValueError:
        fm.scope = SkillScope.MAIN

    return fm


def _manual_parse(raw: str) -> Dict:
    """Fallback parser for malformed YAML."""
    result: Dict = {}
    for line in raw.strip().splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.startswith("[") and val.endswith("]"):
                val = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",") if v.strip()]
            result[key] = val
    return result


def extract_frontmatter_only(content: str) -> SkillFrontmatter:
    """Extract ONLY the frontmatter (without body) for progressive disclosure."""
    fm = parse_frontmatter(content)
    fm.checksum = compute_checksum(content)
    return fm


def load_full_content(filepath: Path) -> str:
    """Read the full SKILL.md content from disk."""
    return filepath.read_text(encoding="utf-8")


def compute_checksum(content: str) -> str:
    """SHA256 checksum for change detection (incremental sync)."""
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def get_body_only(content: str) -> str:
    """Return the SKILL.md body (everything after the frontmatter)."""
    if not content.startswith("---"):
        return content
    end_idx = content.find("\n---", 4)
    if end_idx == -1:
        return content
    return content[end_idx + 4:].lstrip("\n")


def make_frontmatter_block(fm: SkillFrontmatter) -> str:
    """Serialize a SkillFrontmatter back to YAML frontmatter block."""
    import yaml
    data = {
        "name": fm.name,
        "description": fm.description,
        "version": fm.version,
        "scope": fm.scope.value,
        "tags": fm.tags,
        "dependencies": fm.dependencies,
        "created_at": fm.created_at or datetime.now().isoformat(),
        "updated_at": fm.updated_at or datetime.now().isoformat(),
    }
    return f"---\n{yaml.safe_dump(data, allow_unicode=True, sort_keys=False)}---\n"
