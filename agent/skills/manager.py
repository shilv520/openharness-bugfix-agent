"""
SkillManager — 4-Stage Skills Self-Evolution Orchestrator
==========================================================

PDF Harness Engineering pattern: Skills 自我进化 4 阶段

  Stage 1 (Sync):         sync_all() — 本地 → 注册表同步
  Stage 2 (Discover):     list_frontmatters() / load_full_skill() — 渐进式披露
  Stage 3 (Create/DL):    download_skill() / create_skill() — 自动获取新技能
  Stage 4 (Assign/Persist): assign_skill() / restore_from_store() — 持久化与恢复
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .progressive import (
    SkillFrontmatter, SkillScope,
    parse_frontmatter, extract_frontmatter_only,
    load_full_content, get_body_only,
    make_frontmatter_block, compute_checksum,
)
from .persistence import SkillPersistence

logger = logging.getLogger("skills.manager")

# ── Default directories ────────────────────────────────────────

_SKILLS_ROOT = Path(__file__).parent.parent.parent / "skills"


def _scope_dir(scope) -> str:
    """Directory name for a scope."""
    return scope.value if hasattr(scope, "value") else str(scope)


class SkillManager:
    """Central Skills lifecycle manager.

    Usage::

        mgr = SkillManager()
        mgr.sync_all()                       # Stage 1
        fms = mgr.list_frontmatters()        # Stage 2 (discovery)
        skill = mgr.load_full_skill("code-review")  # Stage 2 (load)
        mgr.download_skill("https://...")    # Stage 3
        mgr.assign_skill("my-skill", "main")  # Stage 4
    """

    def __init__(
        self,
        skills_root: Path = None,
        persistence: SkillPersistence = None,
        api_key: str = None,
    ):
        self.skills_root = Path(skills_root or _SKILLS_ROOT).resolve()
        self.persistence = persistence or SkillPersistence()
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")

        # In-memory cache of full content (name → full_text)
        self._content_cache: Dict[str, str] = {}

        logger.info(f"[SkillManager] root={self.skills_root}")

    # ═══════════════════════════════════════════════════════════════
    # Stage 1: Sync
    # ═══════════════════════════════════════════════════════════════

    def sync_all(self) -> Dict:
        """Synchronise tous les skills locaux vers la persistence + cache en mémoire.

        Seuls les fichiers modifiés (checksum diff) sont re-synchronisés.
        Retourne un rapport de synchronisation.
        """
        report = {"synced": [], "skipped": [], "errors": [], "total": 0}

        if not self.skills_root.exists():
            logger.warning(f"[SkillManager] Skills root not found: {self.skills_root}")
            return report

        for scope in SkillScope:
            scope_path = self.skills_root / _scope_dir(scope)
            if not scope_path.exists():
                continue

            for child in sorted(scope_path.iterdir()):
                skill_md = child / "SKILL.md" if child.is_dir() else child
                if not child.is_dir():
                    if child.suffix != ".md":
                        continue
                    skill_md = child

                if not skill_md.exists():
                    continue

                report["total"] += 1
                try:
                    content = load_full_content(skill_md)
                    new_fm = extract_frontmatter_only(content)
                    if not new_fm.name:
                        new_fm.name = child.name if child.is_dir() else child.stem
                    new_fm.scope = scope
                    new_fm.path = str(skill_md.resolve().relative_to(Path.cwd()))

                    # Incremental: skip if checksum unchanged
                    existing = self.persistence.get_skill_meta(new_fm.name)
                    if existing and existing.checksum == new_fm.checksum:
                        report["skipped"].append(new_fm.name)
                        # Still cache content
                        self._content_cache[new_fm.name] = content
                        continue

                    self.persistence.store_skill_content(new_fm, content)
                    self._content_cache[new_fm.name] = content
                    report["synced"].append(new_fm.name)
                    logger.info(f"[SkillManager] synced: {new_fm.name} (scope={scope.value})")
                except Exception as e:
                    report["errors"].append({"name": str(child), "error": str(e)})
                    logger.warning(f"[SkillManager] sync error for {child}: {e}")

        logger.info(
            f"[SkillManager] Sync complete: {len(report['synced'])} synced, "
            f"{len(report['skipped'])} skipped, {len(report['errors'])} errors"
        )
        return report

    def restore_from_store(self) -> List[SkillFrontmatter]:
        """Restaurer les skills persistés depuis ChromaDB/Redis après un redémarrage.

        C'est l'étape qui permet à un Agent de retrouver les skills créés
        ou téléchargés lors d'une session précédente (Stage 4 - Restore).
        """
        restored = self.persistence.list_skills()
        for fm in restored:
            if fm.path:
                full_path = Path(fm.path)
                if full_path.exists():
                    try:
                        self._content_cache[fm.name] = load_full_content(full_path)
                    except Exception:
                        pass
        logger.info(f"[SkillManager] restored {len(restored)} skills from persistence")
        return restored

    # ═══════════════════════════════════════════════════════════════
    # Stage 2: Progressive Discovery
    # ═══════════════════════════════════════════════════════════════

    def list_frontmatters(self, scope: str = None) -> List[SkillFrontmatter]:
        """Lists all skills — frontmatter only (name + description + tags + version).

        Agent calls this at startup to discover available skills
        without loading full content into its context window.

        Corresponds to: Agent `ls /skills/{scope}/` in the PDF project.
        """
        fms = self.persistence.list_skills(scope)
        if not fms:
            # Fallback: scan filesystem directly
            fms = self._scan_fs_frontmatters(scope)
        return sorted(fms, key=lambda f: f.name)

    def load_full_skill(self, name: str) -> Optional[str]:
        """Load the complete SKILL.md content for a skill.

        Only called when the Agent actually needs the skill.
        The body is never injected into the System Prompt upfront.

        Corresponds to: Agent `read_file("/skills/{scope}/{name}/SKILL.md")` in the PDF project.
        """
        # Check cache first
        if name in self._content_cache:
            return self._content_cache[name]

        # Check persistence for path
        fm = self.persistence.get_skill_meta(name)
        if fm and fm.path:
            full_path = Path(fm.path)
            if full_path.exists():
                content = load_full_content(full_path)
                self._content_cache[name] = content
                return content

        # Scan filesystem
        for scope in SkillScope:
            scope_path = self.skills_root / _scope_dir(scope)
            # Directory-style
            skill_dir = scope_path / name
            skill_md = skill_dir / "SKILL.md"
            if skill_md.exists():
                content = load_full_content(skill_md)
                self._content_cache[name] = content
                return content
            # File-style
            file_md = scope_path / f"{name}.md"
            if file_md.exists():
                content = load_full_content(file_md)
                self._content_cache[name] = content
                return content

        logger.warning(f"[SkillManager] skill not found: {name}")
        return None

    def get_frontmatter(self, name: str) -> Optional[SkillFrontmatter]:
        """Get frontmatter for a single skill (without loading the body)."""
        return self.persistence.get_skill_meta(name)

    # ═══════════════════════════════════════════════════════════════
    # Stage 3: Create / Download
    # ═══════════════════════════════════════════════════════════════

    async def download_skill(
        self, url: str, scope: SkillScope = SkillScope.DOWNLOADED, skill_name: str = None,
    ) -> Optional[SkillFrontmatter]:
        """Télécharger une compétence depuis une URL (zip ou md brut).

        Flow:
          1. Télécharger le fichier
          2. Si .zip: extraire → valider la présence de SKILL.md
          3. Si .md: utiliser directement
          4. Tester la validité (parse frontmatter)
          5. Persister dans le scope cible

        Corresponds to: `execute("python download_skill.py '{url}'")` in the PDF project.
        """
        import httpx

        logger.info(f"[SkillManager] downloading: {url}")
        download_dir = self.skills_root / _scope_dir(scope)
        download_dir.mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "")

                if url.endswith(".zip") or "zip" in content_type:
                    # Extract ZIP
                    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
                        tmp.write(resp.content)
                        tmp_path = tmp.name

                    try:
                        with zipfile.ZipFile(tmp_path, "r") as zf:
                            # Find SKILL.md or the root dir name
                            names = zf.namelist()
                            skill_md_in_zip = None
                            for n in names:
                                if n.endswith("SKILL.md") or n.endswith("skill.md"):
                                    skill_md_in_zip = n
                                    break

                            if skill_md_in_zip:
                                # Single SKILL.md inside
                                skill_dir_name = skill_name or Path(skill_md_in_zip).parent.name or "downloaded_skill"
                                dest_dir = download_dir / skill_dir_name
                                zf.extractall(dest_dir)
                            else:
                                # Assume root-level extraction
                                skill_dir_name = skill_name or Path(names[0]).parts[0]
                                dest_dir = download_dir / skill_dir_name
                                zf.extractall(dest_dir)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

                    # Locate SKILL.md
                    skill_md = self._find_skill_md(dest_dir)
                    if not skill_md:
                        logger.error("[SkillManager] no SKILL.md found in downloaded zip")
                        return None

                else:
                    # Raw .md content
                    skill_dir_name = skill_name or "downloaded_skill"
                    dest_dir = download_dir / skill_dir_name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    skill_md = dest_dir / "SKILL.md"
                    skill_md.write_bytes(resp.content)

                # Validate & register
                content = load_full_content(skill_md)
                fm = extract_frontmatter_only(content)
                if not fm.name:
                    fm.name = skill_dir_name
                fm.scope = scope
                fm.path = str(skill_md.resolve().relative_to(Path.cwd()))
                fm.created_at = datetime.now().isoformat()

                self.persistence.store_skill_content(fm, content)
                self._content_cache[fm.name] = content
                logger.info(f"[SkillManager] downloaded & registered: {fm.name}")
                return fm

        except Exception as e:
            logger.error(f"[SkillManager] download failed: {e}")
            return None

    async def create_skill(
        self, name: str, description: str, body: str,
        scope: SkillScope = SkillScope.USER, tags: List[str] = None,
    ) -> Optional[SkillFrontmatter]:
        """Créer une nouvelle compétence à partir d'un template.

        Agent calls this when it wants to create a new skill from scratch
        (e.g., after discovering a new bug pattern and formalizing the fix strategy).

        Corresponds to: Agent `edit_file("/skills/user/{name}/SKILL.md")` in the PDF project.
        """
        scope_dir = self.skills_root / _scope_dir(scope)
        scope_dir.mkdir(parents=True, exist_ok=True)

        skill_dir = scope_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)

        fm = SkillFrontmatter(
            name=name,
            description=description,
            version="1.0.0",
            scope=scope,
            tags=tags or [],
            created_at=datetime.now().isoformat(),
        )

        full_content = make_frontmatter_block(fm) + "\n" + body
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(full_content, encoding="utf-8")

        fm.checksum = compute_checksum(full_content)
        fm.path = str(skill_md.resolve().relative_to(Path.cwd()))

        self.persistence.store_skill_content(fm, full_content)
        self._content_cache[name] = full_content

        logger.info(f"[SkillManager] created: {name} (scope={scope.value})")
        return fm

    # ═══════════════════════════════════════════════════════════════
    # Stage 4: Assign & Persist
    # ═══════════════════════════════════════════════════════════════

    def assign_skill(self, name: str, target_scope: SkillScope) -> bool:
        """Déplacer un skill d'un scope à un autre.

        e.g. promote a 'downloaded' skill to 'main' after validation.

        Corresponds to: `assign_skill(skill_name, agent_name)` in the PDF project.
        """
        fm = self.persistence.get_skill_meta(name)
        if fm is None:
            logger.warning(f"[SkillManager] assign: skill not found '{name}'")
            return False

        old_scope = fm.scope
        if old_scope == target_scope:
            return True

        # Move files
        old_dir = self.skills_root / _scope_dir(old_scope) / name
        new_dir = self.skills_root / _scope_dir(target_scope) / name

        if old_dir.exists() and not new_dir.exists():
            new_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old_dir), str(new_dir))

        # Update metadata
        fm.scope = target_scope
        fm.path = str((new_dir / "SKILL.md").resolve().relative_to(Path.cwd()))
        fm.updated_at = datetime.now().isoformat()

        # Remove from old scope index
        if self.persistence._redis:
            try:
                self.persistence._redis.srem(
                    self.persistence._redis_key_scope(old_scope.value), name
                )
            except Exception:
                pass

        self.persistence.store_skill(fm)

        # Evict cache so next load reads new path
        self._content_cache.pop(name, None)

        logger.info(f"[SkillManager] assigned {name}: {old_scope.value} → {target_scope.value}")
        return True

    async def search_skills(
        self, query: str, top_k: int = 5, scope: str = None,
    ) -> List[Tuple[SkillFrontmatter, float]]:
        """Recherche sémantique de skills via ChromaDB."""
        return await self.persistence.search_skills(query, top_k, scope)

    # ═══════════════════════════════════════════════════════════════
    # Helpers
    # ═══════════════════════════════════════════════════════════════

    def _find_skill_md(self, root: Path) -> Optional[Path]:
        """Recursively find SKILL.md inside a directory."""
        for pattern in ["SKILL.md", "skill.md"]:
            for found in root.rglob(pattern):
                return found
        return None

    def _scan_fs_frontmatters(self, scope: str = None) -> List[SkillFrontmatter]:
        """Fallback: scan filesystem directly for frontmatters."""
        fms: List[SkillFrontmatter] = []
        scopes = [SkillScope(s) for s in ([scope] if scope else [s.value for s in SkillScope])]

        for sc in scopes:
            scope_dir = self.skills_root / _scope_dir(sc)
            if not scope_dir.exists():
                continue
            for child in sorted(scope_dir.iterdir()):
                skill_md = child / "SKILL.md" if child.is_dir() else child
                if child.is_dir() and not skill_md.exists():
                    continue
                if not child.is_dir() and child.suffix != ".md":
                    continue
                try:
                    content = load_full_content(skill_md)
                    fm = extract_frontmatter_only(content)
                    if not fm.name:
                        fm.name = child.name if child.is_dir() else child.stem
                    fm.scope = sc
                    fm.path = str(skill_md.resolve().relative_to(Path.cwd()))
                    fms.append(fm)
                except Exception as e:
                    logger.warning(f"[SkillManager] scan failed for {skill_md}: {e}")

        return fms

    def get_stats(self) -> Dict:
        """Return manager statistics."""
        ps = self.persistence.get_stats()
        ps["cached_contents"] = len(self._content_cache)
        ps["scopes"] = {s.value: _scope_dir(s) for s in SkillScope}
        return ps


# ── Module-level convenience ────────────────────────────────────

_global_manager: Optional[SkillManager] = None


def get_skill_manager(skills_root: Path = None) -> SkillManager:
    """Get or create the global SkillManager singleton."""
    global _global_manager
    if _global_manager is None:
        _global_manager = SkillManager(skills_root=skills_root)
    return _global_manager
