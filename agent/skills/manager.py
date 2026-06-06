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
import re
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

    async def update_skill(
        self, name: str, description: str = None, body: str = None,
        tags: List[str] = None,
    ) -> Optional[SkillFrontmatter]:
        """更新已有 Skill 的 body 内容或元数据。

        当 Agent 发现某个 Skill 无法处理新问题，但问题类型与该 Skill 相近时，
        可以扩充 body 内容，而不是每次都创建全新 Skill。

        如果 Skill 不存在，返回 None（调用方应改用 create_skill）。
        """
        existing_fm = self.persistence.get_skill_meta(name)
        if existing_fm is None:
            logger.warning(f"[SkillManager] update_skill: '{name}' not found, use create_skill instead")
            return None

        # 定位 SKILL.md 文件
        scope_dir = self.skills_root / _scope_dir(existing_fm.scope)
        skill_dir = scope_dir / name
        skill_md = skill_dir / "SKILL.md" if skill_dir.is_dir() else scope_dir / f"{name}.md"

        if not skill_md.exists():
            logger.error(f"[SkillManager] update_skill: SKILL.md not found at {skill_md}")
            return None

        # 更新元数据
        if description:
            existing_fm.description = description
        if tags:
            existing_fm.tags = tags
        existing_fm.updated_at = datetime.now().isoformat()

        # 拼接新的完整内容
        new_body = body if body is not None else get_body_only(load_full_content(skill_md))
        full_content = make_frontmatter_block(existing_fm) + "\n" + new_body

        # 写回文件
        skill_md.write_text(full_content, encoding="utf-8")
        existing_fm.checksum = compute_checksum(full_content)
        existing_fm.path = str(skill_md.resolve().relative_to(Path.cwd()))

        # 更新持久化
        self.persistence.store_skill_content(existing_fm, full_content)
        self._content_cache[name] = full_content

        logger.info(f"[SkillManager] updated: {name} (checksum={existing_fm.checksum[:8]})")
        return existing_fm

    # ── 二次判定辅助：从问题描述中提取关键词 ────────────────────

    # 已知中英文领域关键词（从 AGENT_KEYWORD_TIERS 提取，按长度降序避免短词误匹配）
    _DOMAIN_KEYWORDS: Optional[List[str]] = None

    # 中英术语映射：中文问题描述能匹配英文 tags
    _ZH_EN_MAP = {
        "死锁": "deadlock", "活锁": "livelock",
        "空指针": "null pointer", "空指针异常": "null pointer",
        "并发": "concurrency", "多线程": "multithreading",
        "竞态": "race condition", "竞态条件": "race condition",
        "内存泄漏": "memory leak", "内存溢出": "out of memory",
        "线程安全": "thread safety", "线程不安全": "thread unsafe",
        "缓存穿透": "cache coherence", "缓存雪崩": "cache coherence",
        "循环依赖": "circular dependency",
        "根因": "root cause", "根因分析": "root cause analysis",
        "影响": "impact", "影响评估": "impact analysis",
        "代码审查": "code review", "静态分析": "static analysis",
        "安全漏洞": "security vulnerability", "漏洞": "vulnerability",
        "代码质量": "code quality", "代码规范": "code quality",
        "语法错误": "syntax error",
        "补丁": "patch", "补丁生成": "patch generation",
        "修复": "fix", "重构": "refactor",
        "防御性编程": "defensive programming",
        "异常处理": "exception handling",
        "单元测试": "unit test", "回归": "regression",
        "集成测试": "integration test",
        "副作用": "side effect", "覆盖率": "coverage",
        "诊断": "diagnose", "追溯": "trace", "追踪": "trace",
        "分类": "classify", "归类": "classify",
        "安全": "safety",  # "null pointer safety" → "空指针安全"
        "检查": "check", "检测": "detect",
        "分析": "analysis", "验证": "validate", "测试": "test",
    }

    @classmethod
    def _get_domain_keywords(cls) -> List[str]:
        """获取已知领域关键词列表（懒加载，按长度降序）。"""
        if cls._DOMAIN_KEYWORDS is not None:
            return cls._DOMAIN_KEYWORDS
        from agent.skills.skill_router import AGENT_KEYWORD_TIERS
        seen = set()
        for tiers in AGENT_KEYWORD_TIERS.values():
            for tier_name in ("tier1", "tier2"):
                for kw in tiers.get(tier_name, []):
                    kw_lower = kw.lower()
                    if kw_lower not in seen and len(kw_lower) >= 2:
                        seen.add(kw_lower)
        cls._DOMAIN_KEYWORDS = sorted(seen, key=len, reverse=True)
        return cls._DOMAIN_KEYWORDS

    @classmethod
    def _extract_keywords(cls, text: str) -> set:
        """从文本中提取关键词（中英文混合，子串扫描 + 分隔符分词 + 中英互译）。

        中文没有空格分隔，不能用 split 分词。改用三阶段提取：
          1. 子串扫描：用已知领域关键词做子串匹配，并追加其英文对应词
          2. 空格/标点分词：处理英文等有分隔符的语言
        """
        text_lower = text.lower()
        keywords = set()

        # 阶段 1：已知领域词子串扫描 + 中英互译
        for kw in cls._get_domain_keywords():
            if kw in text_lower:
                keywords.add(kw)
        # 把中文领域词对应的英文术语也加入，使中文问题描述能匹配英文 tags
        for zh, en in cls._ZH_EN_MAP.items():
            if zh in text_lower:
                keywords.add(en)
                keywords.add(zh)

        # 阶段 2：分隔符分词（捕获不在领域词表中的新词）
        tokens = re.split(r'[，,、\s。；;：:！!？?\.\+\-\*/=<>\(\)\[\]{}]+', text_lower)
        for t in tokens:
            t = t.strip()
            if not t or len(t) <= 1:
                continue
            keywords.add(t)

        return keywords

    async def _resolve_ambiguous(
        self, problem_description: str,
        search_results: List[tuple],  # [(SkillFrontmatter, similarity), ...]
        top_k: int = 3,
    ) -> Optional[Dict]:
        """对 ambiguous 区间做二次判定，尽量给出确定性的 create/update 建议。

        判定策略：
          1. 从问题描述中提取关键词
          2. 对每个候选 Skill，计算 tags 与关键词的重叠度
          3. tags 重叠 ≥ 2 → 强信号，推荐 update 该 Skill
          4. tags 重叠 = 1 + Skill 描述中包含问题领域词 → 中等信号，推荐 update
          5. 候选 Skill 的 name 本身就含问题关键词 → update
          6. 所有候选 tags 重叠都是 0 → 推荐 create
          7. 以上都不满足 → 返回 None（保持 ambiguous，交给 Agent LLM）

        Returns:
            消歧成功的 Dict，或 None（保持 ambiguous）
        """
        problem_keywords = self._extract_keywords(problem_description)
        if not problem_keywords:
            return None

        # 没有候选 Skill → 直接返回 create
        if not search_results:
            return {
                "action": "create",
                "reason": "二次判定: 无可用的候选 Skill，建议创建新 Skill。",
                "target_skill": None,
                "candidates": [],
                "top_match_similarity": 0.0,
                "resolve_method": "keyword_no_candidates",
            }

        # 对每个候选计算 tags/keywords 重叠度
        scored_candidates = []
        for fm, similarity in search_results:
            # tags 重叠（子串匹配: "null pointer" 包含 "null"）
            tag_overlap = 0
            for tag in fm.tags:
                tag_lower = tag.lower()
                for kw in problem_keywords:
                    if tag_lower in kw or kw in tag_lower:
                        tag_overlap += 1
                        break

            # skill name 中的词是否出现在问题描述中
            name_words = set(fm.name.lower().replace("-", " ").replace("_", " ").split())
            name_overlap = 0
            for nw in name_words:
                for kw in problem_keywords:
                    if nw in kw or kw in nw:
                        name_overlap += 1
                        break

            # skill description 中的词是否出现在问题描述中
            desc_words = self._extract_keywords(fm.description)
            desc_overlap = 0
            for dw in desc_words:
                for kw in problem_keywords:
                    if dw in kw or kw in dw:
                        desc_overlap += 1
                        break

            total_overlap = tag_overlap + name_overlap + desc_overlap
            scored_candidates.append((fm, similarity, {
                "tag_overlap": tag_overlap,
                "name_overlap": name_overlap,
                "desc_overlap": desc_overlap,
                "total_overlap": total_overlap,
            }))

        # 按 total_overlap 降序排序
        scored_candidates.sort(key=lambda x: x[2]["total_overlap"], reverse=True)
        best_fm, best_sim, best_scores = scored_candidates[0]

        logger.info(
            f"[SkillManager] _resolve_ambiguous: problem_keywords={problem_keywords}, "
            f"best={best_fm.name} tag_overlap={best_scores['tag_overlap']} "
            f"name_overlap={best_scores['name_overlap']} "
            f"desc_overlap={best_scores['desc_overlap']}"
        )

        # ── 判定逻辑 ──

        # 强信号：tags 重叠 ≥ 2，或者 name 直接命中
        if best_scores["tag_overlap"] >= 2 or best_scores["name_overlap"] >= 1:
            return {
                "action": "update",
                "reason": (
                    f"二次判定: Skill '{best_fm.name}'(相似度{best_sim:.0%}) 的 tags/名称 "
                    f"与问题关键词高度重叠 (tags:{best_scores['tag_overlap']}, "
                    f"name:{best_scores['name_overlap']})，建议扩充此 Skill。"
                ),
                "target_skill": best_fm.name,
                "candidates": [
                    {"name": fm.name, "similarity": round(sim, 3)}
                    for fm, sim in search_results
                ],
                "top_match_similarity": round(best_sim, 3),
                "resolve_method": "keyword_overlap_strong",
            }

        # 中等信号：tags 重叠 = 1 且总重叠 ≥ 2
        if best_scores["tag_overlap"] >= 1 and best_scores["total_overlap"] >= 2:
            return {
                "action": "update",
                "reason": (
                    f"二次判定: Skill '{best_fm.name}'(相似度{best_sim:.0%}) 的 "
                    f"tags+描述与问题关键词有部分重叠 (tags:{best_scores['tag_overlap']}, "
                    f"total:{best_scores['total_overlap']})，建议扩充此 Skill。"
                ),
                "target_skill": best_fm.name,
                "candidates": [
                    {"name": fm.name, "similarity": round(sim, 3)}
                    for fm, sim in search_results
                ],
                "top_match_similarity": round(best_sim, 3),
                "resolve_method": "keyword_overlap_moderate",
            }

        # 弱信号：所有候选的 tags 重叠都是 0
        if best_scores["total_overlap"] == 0:
            return {
                "action": "create",
                "reason": (
                    f"二次判定: 所有候选 Skill 的 tags/名称/描述与问题关键词均无重叠，"
                    f"建议创建新 Skill 而非拼凑。"
                ),
                "target_skill": None,
                "candidates": [
                    {"name": fm.name, "similarity": round(sim, 3)}
                    for fm, sim in search_results
                ],
                "top_match_similarity": round(best_sim, 3),
                "resolve_method": "keyword_no_overlap",
            }

        # 无法确定 → 返回 None，保持 ambiguous
        logger.info(
            f"[SkillManager] _resolve_ambiguous: cannot resolve, "
            f"best={best_fm.name} total_overlap={best_scores['total_overlap']}"
        )
        return None

    async def decide_create_or_update(
        self, problem_description: str, top_k: int = 3,
    ) -> Dict:
        """判断一个问题应该创建新 Skill 还是扩充已有 Skill。

        流程：
          1. 用 ChromaDB 语义搜索最相似的已有 Skill
          2. 相似度 > 0.7 → 推荐 update_skill（扩充）
          3. 相似度 0.3-0.7 → 列出候选，交给 Agent 自己判断
          4. 相似度 < 0.3 或没结果 → 推荐 create_skill（新建）

        Returns:
            {
                "action": "update" | "create" | "ambiguous",
                "reason": "决策理由",
                "target_skill": "要更新的 Skill 名称（action=update 时）",
                "candidates": [{"name": ..., "similarity": ...}, ...],  # 候选列表
                "top_match_similarity": 0.85,  # 最佳匹配的相似度
            }
        """
        # 先语义搜索
        search_results = await self.search_skills(problem_description, top_k=top_k)

        if not search_results:
            return {
                "action": "create",
                "reason": "没有找到语义相近的已有 Skill，建议创建新 Skill",
                "target_skill": None,
                "candidates": [],
                "top_match_similarity": 0.0,
            }

        top_fm, top_similarity = search_results[0]

        if top_similarity >= 0.7:
            return {
                "action": "update",
                "reason": (
                    f"已有 Skill '{top_fm.name}' 相似度 {top_similarity:.0%}，"
                    f"描述: {top_fm.description}。建议扩充此 Skill 的 body 内容，"
                    f"追加针对'{problem_description[:50]}'的处理步骤。"
                ),
                "target_skill": top_fm.name,
                "candidates": [
                    {"name": fm.name, "similarity": round(sim, 3)}
                    for fm, sim in search_results
                ],
                "top_match_similarity": round(top_similarity, 3),
            }

        if top_similarity >= 0.3:
            # ── 二次判定：不直接返回 ambiguous，先尝试用 tags/keywords 消歧 ──
            resolved = await self._resolve_ambiguous(
                problem_description, search_results, top_k
            )
            if resolved is not None:
                return resolved

            # 消歧失败，回退到交给 Agent LLM 判断
            return {
                "action": "ambiguous",
                "reason": (
                    f"最佳匹配 '{top_fm.name}' 相似度仅 {top_similarity:.0%}，"
                    f"不够确定。候选: {[fm.name for fm, _ in search_results]}。"
                    f"请 Agent 根据具体情况判断是扩充其中一个还是新建。"
                ),
                "target_skill": None,
                "candidates": [
                    {"name": fm.name, "similarity": round(sim, 3)}
                    for fm, sim in search_results
                ],
                "top_match_similarity": round(top_similarity, 3),
            }

        return {
            "action": "create",
            "reason": (
                f"最佳匹配 '{top_fm.name}' 相似度仅 {top_similarity:.0%}，"
                f"差距过大，建议创建新 Skill 而非拼凑。"
            ),
            "target_skill": None,
            "candidates": [
                {"name": fm.name, "similarity": round(sim, 3)}
                for fm, sim in search_results
            ],
            "top_match_similarity": round(top_similarity, 3),
        }

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
