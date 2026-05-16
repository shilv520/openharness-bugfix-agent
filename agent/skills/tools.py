"""
Agent-Callable Skill Management Tools
=====================================

PDF Harness Engineering pattern: MCP-style tools that the Agent invokes
to manage the skill lifecycle:

  list_skills      → progressive discovery (frontmatter only)
  load_skill       → load full SKILL.md for a specific skill
  search_skills    → semantic search via ChromaDB
  download_skill   → Stage 3: download + validate + persist
  create_skill     → Stage 3: create new skill from template
  assign_skill     → Stage 4: move skill between scopes

These are designed to be registered as MCP tools or called directly
by an agent loop.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .progressive import SkillScope, SkillFrontmatter
from .manager import SkillManager, get_skill_manager

logger = logging.getLogger("skills.tools")

# ── Tool Definitions ────────────────────────────────────────────

SKILL_TOOLS_DEFINITIONS = [
    {
        "name": "list_skills",
        "description": "列出所有可用 Skills（渐进式披露：仅返回 name + description + tags + version，不返回完整内容）",
        "parameters": {
            "type": "object",
            "properties": {
                "scope": {
                    "type": "string",
                    "description": "Filter by scope: main, user, downloaded, skill-management. 不指定则返回所有.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "load_skill",
        "description": "加载指定 Skill 的完整 SKILL.md 内容（渐进式披露：仅在需要时加载）",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Skill name, e.g. 'code-review', 'bug-analysis'",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "search_skills",
        "description": "语义搜索现有 Skills（使用 ChromaDB 向量相似度）",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query, e.g. 'concurrency bug detection'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                    "default": 5,
                },
                "scope": {
                    "type": "string",
                    "description": "Optional scope filter",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "download_skill",
        "description": "从 URL 下载新 Skill → 自动解压(zip) → 验证 SKILL.md → 持久化到 ChromaDB + Redis",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to download the skill from (zip or raw .md)",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Optional custom name for the downloaded skill",
                },
                "scope": {
                    "type": "string",
                    "description": "Target scope (default: downloaded)",
                    "default": "downloaded",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "create_skill",
        "description": "基于模板创建新 Skill → 写入 SKILL.md → 注册到 ChromaDB + Redis",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Unique skill name (kebab-case recommended)",
                },
                "description": {
                    "type": "string",
                    "description": "Short description (1-2 sentences)",
                },
                "body": {
                    "type": "string",
                    "description": "Full SKILL.md body content (everything after the frontmatter)",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags for discovery (e.g. ['java', 'concurrency'])",
                },
                "scope": {
                    "type": "string",
                    "description": "Target scope (default: user)",
                    "default": "user",
                },
            },
            "required": ["name", "description", "body"],
        },
    },
    {
        "name": "assign_skill",
        "description": "将 Skill 从一个 scope 移动到另一个 scope（例如将 downloaded skill 提升为 main）",
        "parameters": {
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to reassign",
                },
                "target_scope": {
                    "type": "string",
                    "description": "Target scope: main, user, downloaded",
                },
            },
            "required": ["skill_name", "target_scope"],
        },
    },
]


def get_tool_definitions() -> List[Dict]:
    """Return MCP-compatible tool definitions for registration."""
    return SKILL_TOOLS_DEFINITIONS


# ── Tool Handler ────────────────────────────────────────────────

class SkillToolsHandler:
    """Async handler that dispatches tool calls to SkillManager."""

    def __init__(self, manager: SkillManager = None):
        self.manager = manager or get_skill_manager()

    async def handle(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a tool call by name.  Returns MCP-style result dict."""
        try:
            if tool_name == "list_skills":
                return self._list(arguments)
            elif tool_name == "load_skill":
                return self._load(arguments)
            elif tool_name == "search_skills":
                return await self._search(arguments)
            elif tool_name == "download_skill":
                return await self._download(arguments)
            elif tool_name == "create_skill":
                return await self._create(arguments)
            elif tool_name == "assign_skill":
                return self._assign(arguments)
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        except Exception as e:
            logger.exception(f"[SkillTools] {tool_name} failed")
            return {"success": False, "error": str(e)}

    # ── Implementations ─────────────────────────────────────

    def _list(self, args: Dict) -> Dict:
        scope = args.get("scope")
        fms = self.manager.list_frontmatters(scope)
        return {
            "success": True,
            "count": len(fms),
            "skills": [
                {
                    "name": fm.name,
                    "description": fm.description,
                    "version": fm.version,
                    "scope": fm.scope.value,
                    "tags": fm.tags,
                    "discovery_text": fm.discovery_text,
                }
                for fm in fms
            ],
        }

    def _load(self, args: Dict) -> Dict:
        name = args["name"]
        content = self.manager.load_full_skill(name)
        if content is None:
            return {"success": False, "error": f"Skill not found: {name}"}
        fm = self.manager.get_frontmatter(name)
        return {
            "success": True,
            "name": name,
            "content": content,
            "frontmatter": fm.to_dict() if fm else None,
        }

    async def _search(self, args: Dict) -> Dict:
        query = args["query"]
        top_k = args.get("top_k", 5)
        scope = args.get("scope")
        results = await self.manager.search_skills(query, top_k, scope)
        return {
            "success": True,
            "query": query,
            "results": [
                {
                    "name": fm.name,
                    "description": fm.description,
                    "version": fm.version,
                    "scope": fm.scope.value,
                    "tags": fm.tags,
                    "similarity": round(score, 3),
                }
                for fm, score in results
            ],
        }

    async def _download(self, args: Dict) -> Dict:
        url = args["url"]
        skill_name = args.get("skill_name")
        scope_str = args.get("scope", "downloaded")
        try:
            scope = SkillScope(scope_str)
        except ValueError:
            scope = SkillScope.DOWNLOADED

        fm = await self.manager.download_skill(url, scope, skill_name)
        if fm is None:
            return {"success": False, "error": "Download failed (check URL and format)"}
        return {
            "success": True,
            "skill": {
                "name": fm.name,
                "description": fm.description,
                "version": fm.version,
                "scope": fm.scope.value,
                "tags": fm.tags,
                "path": fm.path,
            },
        }

    async def _create(self, args: Dict) -> Dict:
        name = args["name"]
        description = args["description"]
        body = args["body"]
        tags = args.get("tags", [])
        scope_str = args.get("scope", "user")
        try:
            scope = SkillScope(scope_str)
        except ValueError:
            scope = SkillScope.USER

        fm = await self.manager.create_skill(name, description, body, scope, tags)
        if fm is None:
            return {"success": False, "error": "Create failed"}
        return {
            "success": True,
            "skill": {
                "name": fm.name,
                "description": fm.description,
                "version": fm.version,
                "scope": fm.scope.value,
                "path": fm.path,
            },
        }

    def _assign(self, args: Dict) -> Dict:
        name = args["skill_name"]
        target = args["target_scope"]
        try:
            scope = SkillScope(target)
        except ValueError:
            return {"success": False, "error": f"Invalid scope: {target}"}

        ok = self.manager.assign_skill(name, scope)
        if not ok:
            return {"success": False, "error": f"Skill not found: {name}"}
        return {"success": True, "skill_name": name, "new_scope": scope.value}


# ── Synchronous convenience wrapper ─────────────────────────────

def run_tool_sync(tool_name: str, arguments: Dict[str, Any], manager: SkillManager = None) -> Dict[str, Any]:
    """Synchronous wrapper for test/debug contexts."""
    import asyncio
    handler = SkillToolsHandler(manager)
    return asyncio.run(handler.handle(tool_name, arguments))
