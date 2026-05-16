"""
虚拟文件系统 (VirtualFS) — CompositeBackend 模块
================================================

PDF Harness Engineering 核心架构: 虚拟文件系统 + 路径路由 + 安全沙箱

    CompositeBackend(
        default=SandboxBackend,     ← Docker 隔离执行 + 本地回退
        routes={
            "/memories/": MemoryBackend,          # Redis + ChromaDB
            "/persisted-skills/": SkillsStoreBackend,  # ChromaDB + Redis
        }
    )

Agent 看到一棵统一的虚拟文件树:
    /workspace/          → 沙箱隔离（Docker + 本地回退）
    /memories/           → 用户记忆（Redis + ChromaDB）
    /persisted-skills/   → 持久化技能（ChromaDB + Redis）
"""

from .base import Backend, FileInfo
from .local import LocalFSBackend
from .memory import MemoryBackend
from .skills_store import SkillsStoreBackend
from .composite import CompositeBackend, create_default_composite

__all__ = [
    "Backend",
    "FileInfo",
    "LocalFSBackend",
    "MemoryBackend",
    "SkillsStoreBackend",
    "CompositeBackend",
    "create_default_composite",
]
