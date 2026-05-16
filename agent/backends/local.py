"""
LocalFSBackend — 本地文件系统后端
==================================

PDF Harness Engineering: CompositeBackend 的 default Backend

所有未匹配路由前缀的路径都落到这里，对应沙箱中的临时文件 / 代码文件。
本项目用本地文件系统替代 OpenSandbox。

安全约束: root 限定在项目目录内，防止路径穿越。
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

from .base import Backend

logger = logging.getLogger("backends.local")


class LocalFSBackend(Backend):
    """本地文件系统后端 — 读写真实的磁盘文件。

    对应 PDF 项目中 CompositeBackend 的 default=sandbox_backend，
    但因本项目无 OpenSandbox，用本地文件系统替代。
    """

    def __init__(self, root: str | Path = None):
        if root is None:
            root = Path(__file__).parent.parent.parent  # 项目根目录
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        logger.info(f"[LocalFSBackend] root={self.root}")

    def _resolve(self, path: str) -> Path:
        """Resolve a virtual path to an absolute path under root.

        安全: real_path 必须在 root 目录内，防止路径穿越。
        """
        clean = path.lstrip("/")
        candidate = (self.root / clean).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise PermissionError(f"Path traversal blocked: {path}")
        return candidate

    async def read(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            raise FileNotFoundError(f"[LocalFS] not found: {p}")
        return p.read_text(encoding="utf-8")

    async def write(self, path: str, content: str) -> bool:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        logger.info(f"[LocalFS] write: {p}")
        return True

    async def list_dir(self, path: str) -> List[str]:
        p = self._resolve(path)
        if not p.exists() or not p.is_dir():
            return []
        return sorted(
            [child.name + ("/" if child.is_dir() else "") for child in p.iterdir()]
        )

    async def delete(self, path: str) -> bool:
        p = self._resolve(path)
        if not p.exists():
            return False
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
        logger.info(f"[LocalFS] delete: {p}")
        return True

    async def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    async def mkdir(self, path: str) -> bool:
        p = self._resolve(path)
        p.mkdir(parents=True, exist_ok=True)
        logger.info(f"[LocalFS] mkdir: {p}")
        return True
