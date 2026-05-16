"""
Backend 抽象基类 — 虚拟文件系统统一接口
=========================================

PDF Harness Engineering: CompositeBackend 的基石

所有存储后端实现同一接口，Agent 对文件系统的一切操作
都通过 Backend 抽象，不感知底层是本地磁盘、Redis、ChromaDB 还是 MongoDB。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FileInfo:
    """文件/目录信息"""
    name: str
    path: str
    is_dir: bool
    size: int = 0


class Backend(ABC):
    """虚拟文件系统后端抽象基类。

    每个 Backend 管理一个命名空间，Agent 通过统一接口
    读写文件，CompositeBackend 根据路径前缀路由到不同 Backend。
    """

    @abstractmethod
    async def read(self, path: str) -> str:
        """Read file content at path. Raises FileNotFoundError if missing."""
        ...

    @abstractmethod
    async def write(self, path: str, content: str) -> bool:
        """Write content to path. Creates parent dirs automatically. Returns True."""
        ...

    @abstractmethod
    async def list_dir(self, path: str) -> List[str]:
        """List file/dir names under path. Returns [] if path doesn't exist."""
        ...

    @abstractmethod
    async def delete(self, path: str) -> bool:
        """Delete file or empty dir at path. Returns True if deleted."""
        ...

    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check whether path exists."""
        ...

    @abstractmethod
    async def mkdir(self, path: str) -> bool:
        """Create directory (recursively). Returns True."""
        ...

    # ── Sync helpers (for non-async contexts like tools) ──────────

    def read_sync(self, path: str) -> str:
        import asyncio
        return asyncio.run(self.read(path))

    def write_sync(self, path: str, content: str) -> bool:
        import asyncio
        return asyncio.run(self.write(path, content))

    def list_dir_sync(self, path: str) -> List[str]:
        import asyncio
        return asyncio.run(self.list_dir(path))

    def exists_sync(self, path: str) -> bool:
        import asyncio
        return asyncio.run(self.exists(path))
