"""
CompositeBackend — 虚拟文件系统路径路由器
===========================================

PDF Harness Engineering 的核心架构模式:

    backend = lambda rt: CompositeBackend(
        default=sandbox_backend,
        routes={
            "/memories/": StoreBackend(runtime=rt, namespace=...),
            "/persisted-skills/": StoreBackend(runtime=rt, namespace=...),
        }
    )

Agent 只看到一棵统一的虚拟文件树，CompositeBackend 根据路径前缀
透明路由到不同的存储后端（本地磁盘 / Redis+ChromaDB）。
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from .base import Backend

logger = logging.getLogger("backends.composite")


class CompositeBackend(Backend):
    """根据路径前缀将请求路由到不同的 Backend。

    路由规则: 最长前缀匹配。未匹配则回退到 default Backend。

    用法:
        vfs = CompositeBackend(
            default=LocalFSBackend(root="./workspace"),
            routes={
                "/memories/": MemoryBackend(),
                "/persisted-skills/": SkillsStoreBackend(),
            },
        )
        content = await vfs.read("/memories/user1/preferences.md")
        #       → MemoryBackend.read("/memories/user1/preferences.md")
        content = await vfs.read("/workspace/code.java")
        #       → LocalFSBackend.read("/workspace/code.java")
    """

    def __init__(self, default: Backend, routes: Dict[str, Backend] = None):
        """
        Args:
            default: 默认 Backend（未匹配任何路由时使用）
            routes:  路径前缀 → Backend 映射
        """
        self.default = default
        self.routes = routes or {}
        # 按前缀长度降序排列，确保最长匹配优先
        self._sorted_prefixes = sorted(self.routes.keys(), key=len, reverse=True)
        logger.info(
            f"[CompositeBackend] default={type(default).__name__}, "
            f"routes={list(self.routes.keys())}"
        )

    def _resolve(self, path: str) -> Tuple[Backend, str]:
        """根据路径前缀找到对应 Backend。

        Returns:
            (backend, path_passed_to_backend)
        """
        # 规范化路径
        normalized = "/" + path.lstrip("/")

        for prefix in self._sorted_prefixes:
            norm_prefix = "/" + prefix.strip("/") + "/"
            if normalized.startswith(norm_prefix) or normalized == norm_prefix.rstrip("/"):
                return self.routes[prefix], normalized

        # 未匹配 → default
        return self.default, normalized

    # ── 路由代理 ──────────────────────────────────────────────

    async def read(self, path: str) -> str:
        backend, resolved_path = self._resolve(path)
        logger.debug(f"[Composite] read {path} → {type(backend).__name__}({resolved_path})")
        return await backend.read(resolved_path)

    async def write(self, path: str, content: str) -> bool:
        backend, resolved_path = self._resolve(path)
        logger.debug(f"[Composite] write {path} → {type(backend).__name__}({resolved_path})")
        return await backend.write(resolved_path, content)

    async def list_dir(self, path: str) -> List[str]:
        backend, resolved_path = self._resolve(path)
        logger.debug(f"[Composite] list {path} → {type(backend).__name__}({resolved_path})")
        return await backend.list_dir(resolved_path)

    async def delete(self, path: str) -> bool:
        backend, resolved_path = self._resolve(path)
        logger.debug(f"[Composite] delete {path} → {type(backend).__name__}({resolved_path})")
        return await backend.delete(resolved_path)

    async def exists(self, path: str) -> bool:
        backend, resolved_path = self._resolve(path)
        return await backend.exists(resolved_path)

    async def mkdir(self, path: str) -> bool:
        backend, resolved_path = self._resolve(path)
        logger.debug(f"[Composite] mkdir {path} → {type(backend).__name__}({resolved_path})")
        return await backend.mkdir(resolved_path)

    # ── 路由查询 ──────────────────────────────────────────────

    def which_backend(self, path: str) -> str:
        """返回给定路径对应的 Backend 名称（用于调试）。"""
        backend, _ = self._resolve(path)
        return type(backend).__name__

    def add_route(self, prefix: str, backend: Backend):
        """动态添加路由"""
        self.routes[prefix] = backend
        self._sorted_prefixes = sorted(self.routes.keys(), key=len, reverse=True)
        logger.info(f"[CompositeBackend] added route: {prefix} → {type(backend).__name__}")


# ── 工厂函数 ──────────────────────────────────────────────────


def create_default_composite(
    workspace_root: str = None,
    use_sandbox: bool = False,
) -> CompositeBackend:
    """创建默认的 CompositeBackend，注册所有子 Backend。

    路由表:
        /workspace/**          → SandboxBackend 或 LocalFSBackend (default)
        /memories/**           → MemoryBackend (Redis + ChromaDB)
        /persisted-skills/**   → SkillsStoreBackend (ChromaDB + Redis)

    Args:
        workspace_root: 默认 Backend 的根目录，默认为项目根
        use_sandbox: 是否使用 SandboxBackend 作为默认后端（Docker 隔离）

    Returns:
        配置完成的 CompositeBackend
    """
    from .local import LocalFSBackend
    from .memory import MemoryBackend
    from .skills_store import SkillsStoreBackend

    memory = MemoryBackend()
    skills = SkillsStoreBackend()

    if use_sandbox:
        from agent.sandbox.sandbox_backend import SandboxBackend
        default = SandboxBackend(root=workspace_root)
    else:
        default = LocalFSBackend(root=workspace_root)

    composite = CompositeBackend(
        default=default,
        routes={
            "/memories": memory,
            "/persisted-skills": skills,
        },
    )

    logger.info(
        "[CompositeBackend] default composite created (default=%s)",
        type(default).__name__,
    )
    return composite


async def create_sandboxed_composite(
    workspace_root: str = None,
    force_docker: bool = False,
) -> CompositeBackend:
    """创建带沙箱隔离的 CompositeBackend。

    default Backend 使用 SandboxBackend（Docker 隔离 + 本地回退）。

    Args:
        workspace_root: 沙箱根目录
        force_docker: 强制使用 Docker（不可用则抛异常）

    Returns:
        配置完成的 CompositeBackend，默认路径走沙箱
    """
    from .memory import MemoryBackend
    from .skills_store import SkillsStoreBackend
    from agent.sandbox.sandbox_backend import create_sandbox_backend

    memory = MemoryBackend()
    skills = SkillsStoreBackend()
    sandbox = await create_sandbox_backend(root=workspace_root, force_docker=force_docker)

    composite = CompositeBackend(
        default=sandbox,
        routes={
            "/memories": memory,
            "/persisted-skills": skills,
        },
    )

    logger.info(
        "[CompositeBackend] sandboxed composite created (sandbox=%s)",
        sandbox.is_sandboxed,
    )
    return composite
