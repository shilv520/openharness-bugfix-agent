"""
SandboxBackend — 沙箱 + 文件系统双层后端
==========================================

PDF Harness Engineering:
    backend = lambda rt: CompositeBackend(
        default=sandbox_backend,   ← 本地沙箱
        routes={...}
    )

SandboxBackend 实现两层架构:
  1. 文件操作 → Docker 容器内执行（隔离读写）
  2. 代码执行 → Docker 容器内执行（隔离运行）

Docker 不可用时自动降级为 LocalFSBackend（开发模式）。
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List

from agent.backends.base import Backend

logger = logging.getLogger("sandbox.backend")


class SandboxBackend(Backend):
    """沙箱后端 — 将文件操作和代码执行隔离在 Docker 容器内。

    对应 PDF 项目中 CompositeBackend 的 default=sandbox_backend。

    用法:
        backend = await create_sandbox_backend(root="./workspace")
        await backend.write("/workspace/test.py", "print('hello')")
        content = await backend.read("/workspace/test.py")
        ret, stdout, stderr = await backend.exec(["python", "/workspace/test.py"])
    """

    def __init__(self, sandbox=None, fallback: Backend | None = None, root: Path | None = None):
        """
        Args:
            sandbox: SecuritySandbox 实例（可选，可后续 start）
            fallback: Docker 不可用时的回退 Backend
            root: 项目根目录
        """
        self._sandbox = sandbox
        self._fallback = fallback
        self.root = Path(root).resolve() if root else Path.cwd()
        self._active = sandbox is not None and sandbox.is_running
        self._started_here = False

    @property
    def is_sandboxed(self) -> bool:
        """是否真正运行在沙箱中"""
        return self._active and self._sandbox is not None

    async def start(self) -> None:
        """启动沙箱（如果没有提供已运行的 SecuritySandbox）"""
        if self._active:
            return

        from agent.sandbox.sandbox import (
            SecuritySandbox,
            SandboxUnavailableError,
            get_docker_availability,
        )

        availability = get_docker_availability()
        if availability.available:
            try:
                self._sandbox = SecuritySandbox(
                    project_root=self.root,
                    auto_build=True,
                )
                await self._sandbox.start()
                self._active = True
                self._started_here = True
                logger.info("[SandboxBackend] Docker sandbox started")
                return
            except SandboxUnavailableError as e:
                logger.warning(f"[SandboxBackend] Docker unavailable: {e}")

        # 降级
        self._active = False
        if self._fallback is None:
            from agent.backends.local import LocalFSBackend
            self._fallback = LocalFSBackend(root=self.root)
            logger.info("[SandboxBackend] Fallback to LocalFSBackend")
        else:
            logger.info("[SandboxBackend] Using provided fallback backend")

    async def stop(self) -> None:
        """停止沙箱（仅当由此实例启动时）"""
        if self._started_here and self._sandbox is not None:
            await self._sandbox.stop()
            self._active = False
            self._started_here = False
            logger.info("[SandboxBackend] Docker sandbox stopped")

    # ── 代码执行接口（Backend ABC 的扩展）──────────────────────

    async def exec(
        self,
        command: list[str],
        *,
        cwd: str | Path | None = None,
        stdin: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str, str]:
        """执行命令（沙箱内或本地）。

        Returns:
            (returncode, stdout, stderr)
        """
        if self.is_sandboxed:
            return await self._sandbox.exec(
                command, cwd=cwd, stdin=stdin, env=env, timeout=timeout,
            )

        # 本地回退
        import subprocess
        try:
            result = subprocess.run(
                command,
                cwd=str(cwd or self.root),
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return (result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            return (-1, "", f"Command timed out after {timeout}s")

    async def exec_python(
        self,
        code: str,
        *,
        timeout: float = 30.0,
    ) -> tuple[int, str, str]:
        """执行 Python 代码"""
        if self.is_sandboxed:
            return await self._sandbox.exec_python(code, timeout=timeout)
        return await self.exec(["python", "-c", code], timeout=timeout)

    async def exec_shell(
        self,
        command: str,
        *,
        timeout: float = 30.0,
    ) -> tuple[int, str, str]:
        """执行 Shell 命令"""
        if self.is_sandboxed:
            return await self._sandbox.exec_shell(command, timeout=timeout)
        return await self.exec(["bash", "-lc", command], timeout=timeout)

    # ── Backend 接口实现 ──────────────────────────────────────

    def _resolve(self, path: str) -> Path:
        """解析虚拟路径到真实路径（含路径穿越保护）"""
        clean = path.lstrip("/")
        candidate = (self.root / clean).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise PermissionError(f"Path traversal blocked: {path}")
        return candidate

    async def read(self, path: str) -> str:
        resolved = self._resolve(path)

        if self.is_sandboxed:
            return await self._sandbox.read_file(resolved)

        if self._fallback:
            return await self._fallback.read(path)
        return resolved.read_text(encoding="utf-8")

    async def write(self, path: str, content: str) -> bool:
        resolved = self._resolve(path)

        if self.is_sandboxed:
            # 确保父目录存在
            parent = str(resolved.parent)
            await self._sandbox.mkdir(parent)
            return await self._sandbox.write_file(resolved, content)

        if self._fallback:
            return await self._fallback.write(path, content)

        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return True

    async def list_dir(self, path: str) -> List[str]:
        resolved = self._resolve(path)

        if self.is_sandboxed:
            return await self._sandbox.list_dir(resolved)

        if self._fallback:
            return await self._fallback.list_dir(path)

        if not resolved.exists() or not resolved.is_dir():
            return []
        return sorted(
            [child.name + ("/" if child.is_dir() else "") for child in resolved.iterdir()]
        )

    async def delete(self, path: str) -> bool:
        resolved = self._resolve(path)

        if self.is_sandboxed:
            return await self._sandbox.delete(resolved)

        if self._fallback:
            return await self._fallback.delete(path)

        if not resolved.exists():
            return False
        import shutil
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        return True

    async def exists(self, path: str) -> bool:
        resolved = self._resolve(path)

        if self.is_sandboxed:
            return await self._sandbox.file_exists(resolved) or await self._sandbox.dir_exists(resolved)

        if self._fallback:
            return await self._fallback.exists(path)

        return resolved.exists()

    async def mkdir(self, path: str) -> bool:
        resolved = self._resolve(path)

        if self.is_sandboxed:
            return await self._sandbox.mkdir(resolved)

        if self._fallback:
            return await self._fallback.mkdir(path)

        resolved.mkdir(parents=True, exist_ok=True)
        return True


# ── 工厂函数 ──────────────────────────────────────────────────

async def create_sandbox_backend(
    root: str | Path = None,
    force_docker: bool = False,
) -> SandboxBackend:
    """创建 SandboxBackend 实例并启动沙箱。

    Args:
        root: 项目根目录
        force_docker: 强制使用 Docker（不可用时抛异常）

    Returns:
        已初始化的 SandboxBackend
    """
    backend = SandboxBackend(root=Path(root).resolve() if root else None)
    await backend.start()

    if force_docker and not backend.is_sandboxed:
        from agent.sandbox.sandbox import SandboxUnavailableError
        raise SandboxUnavailableError("Docker sandbox is required but unavailable")

    return backend
