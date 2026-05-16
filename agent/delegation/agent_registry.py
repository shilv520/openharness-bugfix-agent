"""
AgentRegistry — 子Agent类型注册中心
=====================================

PDF Harness Engineering: Agent Registry管理所有可用的子Agent类型

功能:
  - 从YAML目录加载所有AgentDefinition
  - 按名称/标签/能力搜索可用的Agent类型
  - 提供给主Agent的task工具选择子Agent类型
  - 支持热加载（运行时添加新的Agent类型）
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .agent_definition import AgentDefinition, load_all_definitions

logger = logging.getLogger("delegation.registry")

# ── 全局单例 ─────────────────────────────────────────────────

_registry: Optional["AgentRegistry"] = None


def get_agent_registry(config_dir: str = None) -> "AgentRegistry":
    """获取全局AgentRegistry单例"""
    global _registry
    if _registry is None:
        _registry = AgentRegistry(config_dir=config_dir)
    return _registry


# ── AgentRegistry ─────────────────────────────────────────────

class AgentRegistry:
    """子Agent类型注册中心。

    用法:
        registry = AgentRegistry()
        registry.load()

        # 搜索
        ad = registry.get("code-reviewer")
        ads = registry.search("review")

        # 列表（渐进式披露，只返回摘要）
        for brief in registry.list_briefs():
            print(brief)
    """

    def __init__(self, config_dir: str = None):
        self._definitions: Dict[str, AgentDefinition] = {}
        self._config_dir = config_dir
        self._loaded = False

    @property
    def agent_types(self) -> List[str]:
        """所有已注册的Agent类型名称"""
        if not self._loaded:
            self.load()
        return list(self._definitions.keys())

    @property
    def count(self) -> int:
        if not self._loaded:
            self.load()
        return len(self._definitions)

    def load(self, config_dir: str = None):
        """从YAML目录加载所有Agent定义"""
        directory = config_dir or self._config_dir
        definitions = load_all_definitions(directory)

        self._definitions.clear()
        for ad in definitions:
            self._definitions[ad.name] = ad

        self._loaded = True
        logger.info(f"[Registry] Loaded {len(self._definitions)} agent types: {list(self._definitions.keys())}")

    def get(self, name: str) -> Optional[AgentDefinition]:
        """按名称获取Agent定义"""
        if not self._loaded:
            self.load()
        return self._definitions.get(name)

    def search(
        self,
        query: str = None,
        tags: List[str] = None,
        capability: str = None,
    ) -> List[AgentDefinition]:
        """搜索匹配的Agent定义。

        Args:
            query: 在name和description中搜索的关键词
            tags: 必须匹配的标签列表
            capability: 必须匹配的能力

        Returns:
            匹配的AgentDefinition列表
        """
        if not self._loaded:
            self.load()

        results = list(self._definitions.values())

        if query:
            q = query.lower()
            results = [
                ad for ad in results
                if q in ad.name.lower() or q in ad.description.lower()
            ]

        if tags:
            results = [
                ad for ad in results
                if all(t in ad.tags for t in tags)
            ]

        if capability:
            cap = capability.lower()
            results = [
                ad for ad in results
                if any(cap in c.lower() for c in ad.capabilities)
            ]

        return results

    def list_briefs(self) -> List[str]:
        """渐进式披露：只返回摘要列表（节省Token）"""
        if not self._loaded:
            self.load()
        return [ad.brief for ad in self._definitions.values()]

    def list_all(self) -> List[AgentDefinition]:
        """返回所有Agent定义"""
        if not self._loaded:
            self.load()
        return list(self._definitions.values())

    def register(self, ad: AgentDefinition):
        """动态注册新的Agent类型（运行时扩展）"""
        self._definitions[ad.name] = ad
        logger.info(f"[Registry] Registered: {ad.display_name}")

    def unregister(self, name: str) -> bool:
        """移除Agent类型"""
        if name in self._definitions:
            del self._definitions[name]
            return True
        return False
