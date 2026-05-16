"""
Skill Persistence Layer — ChromaDB + Redis
==========================================

PDF Harness Engineering pattern: Stage 4 (Assign & Persist)

ChromaDB:  skill vector index for semantic search
Redis:     skill metadata + scope indexes (fast lookups)

Reuses patterns from agent/redis_memory.py for ChromaDB + Redis integration.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .progressive import SkillFrontmatter, SkillScope, compute_checksum

logger = logging.getLogger("skills.persistence")

# ── Optional imports ───────────────────────────────────────────

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# ── Embedding API config ───────────────────────────────────────

EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")


class SkillPersistence:
    """ChromaDB + Redis persistence for Skills.

    Redis schema:
      skill:meta:{name}   → Hash {version, scope, checksum, path, created_at, updated_at, description, tags}
      skill:scope:{scope} → Set  {skill_names}

    ChromaDB schema:
      Collection: skill_embeddings
      IDs:        skill:{name}
      Documents:  {name}: {description} (text used for embedding)
      Metadata:   {name, version, scope, tags_json, checksum, created_at}
    """

    def __init__(
        self,
        redis_host: str = None,
        redis_port: int = None,
        chroma_host: str = None,
        chroma_port: str = None,
    ):
        self._redis: "redis.Redis | None" = None
        self._chroma_client = None
        self._collection = None

        # ── Redis ──────────────────────────────────────────
        if REDIS_AVAILABLE:
            host = redis_host or os.environ.get("REDIS_HOST", "localhost")
            port = redis_port or int(os.environ.get("REDIS_PORT", 6379))
            try:
                self._redis = redis.Redis(
                    host=host, port=port,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                self._redis.ping()
                logger.info(f"[SkillPersist] Redis connected: {host}:{port}")
            except Exception as e:
                logger.warning(f"[SkillPersist] Redis unavailable: {e}")
                self._redis = None
        else:
            logger.warning("[SkillPersist] redis module not installed")
            self._redis = None

        # ── ChromaDB ───────────────────────────────────────
        if CHROMADB_AVAILABLE:
            chroma_host = chroma_host or os.environ.get("CHROMA_HOST", "localhost")
            chroma_port = chroma_port or int(os.environ.get("CHROMA_PORT", "8001"))
            try:
                self._chroma_client = chromadb.HttpClient(
                    host=chroma_host, port=int(chroma_port),
                )
                self._collection = self._chroma_client.get_or_create_collection(
                    name="skill_embeddings",
                    metadata={"description": "Skills vector index for semantic search"},
                )
                logger.info(
                    f"[SkillPersist] ChromaDB connected: {chroma_host}:{chroma_port} "
                    f"(skills: {self._collection.count()})"
                )
            except Exception as e:
                logger.warning(f"[SkillPersist] ChromaDB unavailable: {e}")
                self._chroma_client = None
                self._collection = None
        else:
            logger.warning("[SkillPersist] chromadb module not installed")
            self._collection = None

        # Fallback in-memory store
        self._fallback: Dict[str, SkillFrontmatter] = {}

    @property
    def is_ready(self) -> bool:
        return self._redis is not None and self._collection is not None

    # ── Redis helpers ──────────────────────────────────────────

    def _redis_key_meta(self, name: str) -> str:
        return f"skill:meta:{name}"

    def _redis_key_scope(self, scope: str) -> str:
        return f"skill:scope:{scope}"

    # ── Store ───────────────────────────────────────────────────

    def store_skill(self, fm: SkillFrontmatter) -> bool:
        """Store skill metadata in Redis + ChromaDB."""
        name = fm.name
        if not name:
            return False

        now = datetime.now().isoformat()
        if not fm.created_at:
            fm.created_at = now
        fm.updated_at = now

        # Redis Hash
        if self._redis:
            try:
                meta = {
                    "version": fm.version,
                    "scope": fm.scope.value,
                    "description": fm.description[:500],
                    "tags": json.dumps(fm.tags),
                    "dependencies": json.dumps(fm.dependencies),
                    "checksum": fm.checksum,
                    "path": fm.path,
                    "created_at": fm.created_at,
                    "updated_at": fm.updated_at,
                }
                self._redis.hset(self._redis_key_meta(name), mapping=meta)
                self._redis.sadd(self._redis_key_scope(fm.scope.value), name)
                logger.info(f"[SkillPersist] Redis stored: {name} (scope={fm.scope.value})")
            except Exception as e:
                logger.warning(f"[SkillPersist] Redis store failed for {name}: {e}")

        # ChromaDB vector
        if self._collection:
            try:
                doc_id = f"skill:{name}"
                doc_text = f"{name}: {fm.description}"
                # Remove old entry if exists
                try:
                    self._collection.delete(ids=[doc_id])
                except Exception:
                    pass
                self._collection.add(
                    ids=[doc_id],
                    documents=[doc_text],
                    metadatas=[{
                        "name": name,
                        "version": fm.version,
                        "scope": fm.scope.value,
                        "tags": json.dumps(fm.tags),
                        "checksum": fm.checksum,
                        "created_at": fm.created_at,
                    }],
                )
                logger.info(f"[SkillPersist] ChromaDB indexed: {name}")
            except Exception as e:
                logger.warning(f"[SkillPersist] ChromaDB store failed for {name}: {e}")

        # Fallback
        self._fallback[name] = fm
        return True

    def store_skill_content(self, fm: SkillFrontmatter, content: str) -> bool:
        """Store skill frontmatter + persist full content checksum."""
        fm.checksum = compute_checksum(content)
        return self.store_skill(fm)

    # ── Retrieve ────────────────────────────────────────────────

    def get_skill_meta(self, name: str) -> Optional[SkillFrontmatter]:
        """Get skill frontmatter by name (from Redis or fallback)."""
        if self._redis:
            try:
                data = self._redis.hgetall(self._redis_key_meta(name))
                if data:
                    tags = json.loads(data.get("tags", "[]"))
                    deps = json.loads(data.get("dependencies", "[]"))
                    scope = data.get("scope", "main")
                    try:
                        scope = SkillScope(scope)
                    except ValueError:
                        scope = SkillScope.MAIN
                    return SkillFrontmatter(
                        name=name,
                        description=data.get("description", ""),
                        version=data.get("version", "1.0.0"),
                        scope=scope,
                        tags=tags,
                        dependencies=deps,
                        checksum=data.get("checksum", ""),
                        path=data.get("path", ""),
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                    )
            except Exception as e:
                logger.warning(f"[SkillPersist] Redis get failed for {name}: {e}")

        return self._fallback.get(name)

    def list_skills(self, scope: str = None) -> List[SkillFrontmatter]:
        """List all skills (optionally filtered by scope)."""
        result: List[SkillFrontmatter] = []

        if self._redis:
            try:
                if scope:
                    names = self._redis.smembers(self._redis_key_scope(scope))
                else:
                    # Gather from all scope sets concurrently is overkill;
                    # scan keys instead.
                    names = set()
                    for sc in SkillScope:
                        names |= self._redis.smembers(self._redis_key_scope(sc.value))

                for name in names:
                    fm = self.get_skill_meta(name)
                    if fm:
                        result.append(fm)
            except Exception as e:
                logger.warning(f"[SkillPersist] list failed: {e}")

        if not result:
            result = list(self._fallback.values())
            if scope:
                result = [fm for fm in result if fm.scope.value == scope]

        return result

    def list_frontmatters(self, scope: str = None) -> List[SkillFrontmatter]:
        """Alias: list skills with frontmatter-only data (progressive disclosure)."""
        return self.list_skills(scope)

    # ── Semantic Search (via ChromaDB) ──────────────────────────

    async def search_skills(
        self, query: str, top_k: int = 5, scope: str = None,
    ) -> List[Tuple[SkillFrontmatter, float]]:
        """Semantic search for skills using ChromaDB vector similarity.

        Returns list of (SkillFrontmatter, similarity_score).
        """
        if not self._collection or not query.strip():
            logger.warning("[SkillPersist] ChromaDB not available for search")
            return []

        # Compute embedding for query
        embedding = await self._compute_embedding(query)
        if not embedding:
            return []

        try:
            where_filter = None
            if scope:
                where_filter = {"scope": scope}

            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k * 2, self._collection.count()),
                where=where_filter,
            )

            out: List[Tuple[SkillFrontmatter, float]] = []
            if results and results.get("metadatas") and results["metadatas"][0]:
                for i, meta in enumerate(results["metadatas"][0]):
                    name = meta.get("name", "")
                    fm = self.get_skill_meta(name)
                    if fm is None:
                        fm = SkillFrontmatter(
                            name=name,
                            description=meta.get("description", ""),
                            version=meta.get("version", "1.0.0"),
                            tags=json.loads(meta.get("tags", "[]")),
                        )
                    distance = results["distances"][0][i] if results.get("distances") else 0.0
                    similarity = 1.0 - distance
                    out.append((fm, similarity))

            out.sort(key=lambda x: x[1], reverse=True)
            return out[:top_k]
        except Exception as e:
            logger.warning(f"[SkillPersist] search failed: {e}")
            return []

    async def _compute_embedding(self, text: str) -> List[float]:
        """Compute text embedding via API."""
        import httpx

        if not EMBEDDING_API_KEY or not EMBEDDING_API_URL:
            return []

        base = EMBEDDING_API_URL.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{base}/embeddings",
                    headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
                    json={"model": EMBEDDING_MODEL, "input": text},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("data"):
                        return data["data"][0].get("embedding", [])
        except Exception as e:
            logger.warning(f"[SkillPersist] embedding failed: {e}")
        return []

    # ── Delete ───────────────────────────────────────────────────

    def delete_skill(self, name: str) -> bool:
        """Remove a skill from Redis and ChromaDB."""
        fm = self.get_skill_meta(name)
        if fm is None:
            logger.warning(f"[SkillPersist] delete: skill not found: {name}")
            return False

        if self._redis:
            try:
                self._redis.delete(self._redis_key_meta(name))
                self._redis.srem(self._redis_key_scope(fm.scope.value), name)
            except Exception as e:
                logger.warning(f"[SkillPersist] Redis delete failed: {e}")

        if self._collection:
            try:
                self._collection.delete(ids=[f"skill:{name}"])
            except Exception as e:
                logger.warning(f"[SkillPersist] ChromaDB delete failed: {e}")

        self._fallback.pop(name, None)
        logger.info(f"[SkillPersist] deleted: {name}")
        return True

    # ── Seed / Bulk Load ────────────────────────────────────────

    def seed_from_directory(self, root: Path, scope: SkillScope) -> int:
        """Bulk-load all SKILL.md files from a directory tree.

        Scans <root>/<skill_dir>/SKILL.md and stores metadata only
        (progressive disclosure — full content loaded on demand)."""
        from .progressive import extract_frontmatter_only, load_full_content

        count = 0
        if not root.exists():
            return 0

        for child in sorted(root.iterdir()):
            skill_md = child / "SKILL.md" if child.is_dir() else child
            if child.is_dir() and not skill_md.exists():
                continue
            if not child.is_dir():
                # Direct .md file
                if child.suffix != ".md":
                    continue
                skill_md = child

            try:
                content = load_full_content(skill_md)
                fm = extract_frontmatter_only(content)
                if not fm.name:
                    fm.name = child.name if child.is_dir() else child.stem
                fm.scope = scope
                fm.path = str(skill_md.resolve().relative_to(Path.cwd()))
                if not fm.created_at:
                    fm.created_at = datetime.now().isoformat()
                self.store_skill(fm)
                count += 1
                logger.info(f"[SkillPersist] seed: {fm.name} (scope={scope.value})")
            except Exception as e:
                logger.warning(f"[SkillPersist] seed failed for {skill_md}: {e}")

        return count

    def get_stats(self) -> Dict:
        """Return persistence statistics."""
        chroma_count = 0
        if self._collection:
            try:
                chroma_count = self._collection.count()
            except Exception:
                pass

        redis_count = 0
        if self._redis:
            try:
                for sc in SkillScope:
                    redis_count += self._redis.scard(self._redis_key_scope(sc.value))
            except Exception:
                pass

        return {
            "chromadb_skills": chroma_count,
            "redis_skills": redis_count,
            "fallback_skills": len(self._fallback),
        }
