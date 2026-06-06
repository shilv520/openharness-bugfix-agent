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
        "name": "update_skill",
        "description": "更新已有 Skill 的 body 或元数据。用于 Agent 发现新问题后扩充现有 Skill 的能力，而非每次都创建新 Skill。Skill 不存在时应改用 create_skill。",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "要更新的 Skill 名称",
                },
                "description": {
                    "type": "string",
                    "description": "新的描述（可选，不传则保持原描述）",
                },
                "body": {
                    "type": "string",
                    "description": "新的 SKILL.md body 内容（可选，不传则保持原 body）",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "新的标签列表（可选，用于后续 Skill→Agent 自动匹配）",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "recommend_skill_action",
        "description": "判断一个问题应该创建新 Skill 还是扩充已有 Skill。Agent 在决定 create_skill 还是 update_skill 之前应先调用此工具获取建议。",
        "parameters": {
            "type": "object",
            "properties": {
                "problem": {
                    "type": "string",
                    "description": "问题描述，例如 'Java多线程死锁检测'、'Python内存泄漏诊断'",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的候选 Skill 数量（默认 3）",
                    "default": 3,
                },
            },
            "required": ["problem"],
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
            elif tool_name == "recommend_skill_action":
                return await self._recommend(arguments)
            elif tool_name == "update_skill":
                return await self._update(arguments)
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

    async def _recommend(self, args: Dict) -> Dict:
        problem = args["problem"]
        top_k = args.get("top_k", 3)
        decision = await self.manager.decide_create_or_update(problem, top_k)
        decision["success"] = True
        return decision

    async def _update(self, args: Dict) -> Dict:
        name = args["name"]
        description = args.get("description")
        body = args.get("body")
        tags = args.get("tags")

        fm = await self.manager.update_skill(
            name=name, description=description, body=body, tags=tags,
        )
        if fm is None:
            return {"success": False, "error": f"Skill not found: {name}. Use create_skill to create a new one."}
        return {
            "success": True,
            "skill": {
                "name": fm.name,
                "description": fm.description,
                "version": fm.version,
                "scope": fm.scope.value,
                "tags": fm.tags,
                "updated_at": fm.updated_at,
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


# ── SubAgent 工具包装器 ──────────────────────────────────────────
# 把 SkillToolsHandler 的方法包装为 SubAgent.tools 可调用的函数
# 每个函数签名与 MCP 工具的参数一致，内部委托给 handler.handle()


def build_skill_tools_for_agent(
    handler: SkillToolsHandler = None,
) -> Dict[str, callable]:
    """构建 SubAgent 可用的 Skill 管理工具字典。

    返回 {tool_name: async_callable}，可以直接传给 SubAgent(tools=...).
    每个工具函数的 __doc__ 会被 SubAgent 用来自动生成工具描述。

    Usage:
        handler = SkillToolsHandler(manager)
        skill_tools = build_skill_tools_for_agent(handler)
        sub = SubAgent(..., tools={**base_tools, **skill_tools})
    """
    h = handler or SkillToolsHandler()

    async def search_skills(query: str, top_k: int = 5, scope: str = None) -> dict:
        """搜索现有 Skills。当你需要判断一个问题是否能被已有 Skill 覆盖时先调用此工具。
参数: query(问题描述), top_k(返回数量,默认5), scope(可选: main/user/downloaded)"""
        return await h.handle("search_skills", {
            "query": query, "top_k": top_k, "scope": scope,
        })

    async def load_skill(name: str) -> dict:
        """加载指定 Skill 的完整内容。当你需要查看某个 Skill 的详细步骤时调用。
参数: name(Skill名称,如'code-review')"""
        return await h.handle("load_skill", {"name": name})

    async def recommend_skill_action(problem: str, top_k: int = 3) -> dict:
        """判断一个问题是应该创建新 Skill 还是扩充已有 Skill。在决定 create_skill 还是 update_skill 之前必须先调用此工具！
参数: problem(问题描述), top_k(候选数量,默认3)。返回 action字段: 'create'/'update'/'ambiguous'"""
        return await h.handle("recommend_skill_action", {
            "problem": problem, "top_k": top_k,
        })

    async def create_skill(name: str, description: str, body: str, tags: list = None, scope: str = "user") -> dict:
        """创建全新的 Skill。当 recommend_skill_action 返回 action='create' 时调用。
参数: name(kebab-case命名), description(一句话描述), body(完整Markdown内容), tags(标签列表), scope(默认user)"""
        return await h.handle("create_skill", {
            "name": name, "description": description, "body": body,
            "tags": tags or [], "scope": scope,
        })

    async def update_skill(name: str, description: str = None, body: str = None, tags: list = None) -> dict:
        """更新已有 Skill。当 recommend_skill_action 返回 action='update' 且 target_skill 不为空时调用。
参数: name(要更新的Skill名), description(新描述,可选), body(新内容,可选), tags(新标签,可选)"""
        return await h.handle("update_skill", {
            "name": name, "description": description,
            "body": body, "tags": tags,
        })

    async def list_skills(scope: str = None) -> dict:
        """列出所有可用 Skills（仅返回名称和描述，不返回完整内容）。
参数: scope(可选: main/user/downloaded)"""
        return await h.handle("list_skills", {"scope": scope})

    return {
        "search_skills": search_skills,
        "load_skill": load_skill,
        "recommend_skill_action": recommend_skill_action,
        "create_skill": create_skill,
        "update_skill": update_skill,
        "list_skills": list_skills,
    }


# ── Synchronous convenience wrapper ─────────────────────────────

def run_tool_sync(tool_name: str, arguments: Dict[str, Any], manager: SkillManager = None) -> Dict[str, Any]:
    """Synchronous wrapper for test/debug contexts."""
    import asyncio
    handler = SkillToolsHandler(manager)
    return asyncio.run(handler.handle(tool_name, arguments))
