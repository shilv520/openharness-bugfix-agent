"""
SecuritySandbox — Docker 隔离执行环境
======================================

PDF Harness Engineering 核心安全边界:
  沙箱 = 唯一安全边界 → 所有 Agent 行为在容器内完成

设计思路:
  1. 创建长期运行的 Docker 容器（--network=none + 资源限制）
  2. 项目目录 bind-mount 进容器（Agent 只能访问项目文件）
  3. 所有命令通过 docker exec 在容器内执行
  4. Docker 不可用时自动降级为本地执行（开发模式）
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger("sandbox")

# ── 沙箱镜像 Dockerfile ─────────────────────────────────────────

_SANDBOX_DOCKERFILE = """\
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \\
    ripgrep bash git curl && \\
    rm -rf /var/lib/apt/lists/*
RUN useradd -m -s /bin/bash sandbox
USER sandbox
WORKDIR /workspace
"""

_SANDBOX_IMAGE = "bugfix-sandbox:latest"


# ── 可用性检查 ─────────────────────────────────────────────────

@dataclass(frozen=True)
class SandboxAvailability:
    """沙箱可用性检查结果"""
    enabled: bool
    available: bool
    reason: str | None = None
    command: str | None = None


class SandboxUnavailableError(RuntimeError):
    """沙箱不可用时抛出"""


def get_docker_availability() -> SandboxAvailability:
    """检查 Docker 是否可作为沙箱后端"""
    docker = shutil.which("docker")
    if not docker:
        return SandboxAvailability(
            enabled=True,
            available=False,
            reason="Docker CLI not found; install Docker Desktop or Docker Engine",
        )

    try:
        result = subprocess.run(
            [docker, "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode != 0:
            return SandboxAvailability(
                enabled=True,
                available=False,
                reason="Docker daemon is not running",
                command=docker,
            )
    except (subprocess.TimeoutExpired, OSError):
        return SandboxAvailability(
            enabled=True,
            available=False,
            reason="Docker daemon is not responding",
            command=docker,
        )

    return SandboxAvailability(enabled=True, available=True, command=docker)


# ── 镜像管理 ───────────────────────────────────────────────────

async def _image_exists(image: str) -> bool:
    docker = shutil.which("docker") or "docker"
    proc = await asyncio.create_subprocess_exec(
        docker, "image", "inspect", image,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()
    return proc.returncode == 0


async def build_sandbox_image(image: str = _SANDBOX_IMAGE) -> bool:
    """构建沙箱 Docker 镜像"""
    docker = shutil.which("docker") or "docker"
    logger.info(f"[Sandbox] Building image {image} ...")

    proc = await asyncio.create_subprocess_exec(
        docker, "build", "-t", image, "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate(input=_SANDBOX_DOCKERFILE.encode("utf-8"))

    if proc.returncode == 0:
        logger.info(f"[Sandbox] Image {image} built successfully")
        return True
    logger.warning(f"[Sandbox] Failed to build image {image}: {stderr.decode('utf-8', errors='replace')[:200]}")
    return False


async def ensure_image(image: str = _SANDBOX_IMAGE, auto_build: bool = True) -> bool:
    """确保沙箱镜像存在"""
    if await _image_exists(image):
        return True
    if not auto_build:
        logger.warning(f"[Sandbox] Image {image} not found, auto_build disabled")
        return False
    return await build_sandbox_image(image)


# ── SecuritySandbox ────────────────────────────────────────────

@dataclass
class SecuritySandbox:
    """管理一个长期运行的 Docker 沙箱容器。

    对应 PDF 项目中的 DockerSandboxSession。

    用法:
        sandbox = SecuritySandbox(project_root="/app")
        await sandbox.start()
        result = await sandbox.exec(["python", "script.py"])
        await sandbox.stop()
    """

    project_root: Path
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    image: str = _SANDBOX_IMAGE
    auto_build: bool = True
    network_enabled: bool = False
    cpu_limit: float = 0.0          # CPU 核数限制（0=不限制）
    memory_limit: str = ""           # 内存限制（如 "512m"）
    extra_mounts: list[str] = field(default_factory=list)
    extra_env: dict[str, str] = field(default_factory=dict)

    _container_name: str = field(init=False)
    _running: bool = field(init=False, default=False)

    def __post_init__(self):
        self._container_name = f"bugfix-sandbox-{self.session_id}"

    @property
    def container_name(self) -> str:
        return self._container_name

    @property
    def is_running(self) -> bool:
        return self._running

    def _build_run_argv(self) -> list[str]:
        """构建 docker run 参数"""
        docker = shutil.which("docker") or "docker"
        cwd_str = str(self.project_root.resolve())

        argv = [
            docker, "run", "-d", "--rm",
            "--name", self._container_name,
        ]

        # 网络隔离: 默认禁用网络
        if not self.network_enabled:
            argv.extend(["--network", "none"])
        else:
            # 允许网络但限制 DNS
            argv.extend(["--dns", "8.8.8.8"])

        # 资源限制
        if self.cpu_limit > 0:
            argv.extend(["--cpus", str(self.cpu_limit)])
        if self.memory_limit:
            argv.extend(["--memory", self.memory_limit])

        # Bind-mount 项目目录
        argv.extend(["-v", f"{cwd_str}:{cwd_str}"])
        argv.extend(["-w", cwd_str])

        # 额外挂载
        for mount in self.extra_mounts:
            argv.extend(["-v", mount])

        # 环境变量
        for key, value in self.extra_env.items():
            argv.extend(["-e", f"{key}={value}"])

        argv.extend([self.image, "tail", "-f", "/dev/null"])
        return argv

    async def start(self) -> None:
        """启动沙箱容器"""
        available = await ensure_image(self.image, self.auto_build)
        if not available:
            raise SandboxUnavailableError(
                f"Sandbox image {self.image!r} is not available"
            )

        argv = self._build_run_argv()
        logger.info(f"[Sandbox] Starting: {' '.join(argv)}")

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            msg = stderr.decode("utf-8", errors="replace").strip()
            raise SandboxUnavailableError(f"Failed to start sandbox: {msg}")

        self._running = True
        logger.info(f"[Sandbox] Started: {self._container_name}")

    async def stop(self) -> None:
        """停止沙箱容器"""
        if not self._running:
            return
        docker = shutil.which("docker") or "docker"
        try:
            proc = await asyncio.create_subprocess_exec(
                docker, "stop", "-t", "5", self._container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await asyncio.wait_for(proc.communicate(), timeout=15)
        except (asyncio.TimeoutError, OSError) as exc:
            logger.warning(f"[Sandbox] Error stopping: {exc}")
        finally:
            self._running = False
            logger.info(f"[Sandbox] Stopped: {self._container_name}")

    def stop_sync(self) -> None:
        """同步停止（用于 atexit 回调）"""
        if not self._running:
            return
        docker = shutil.which("docker") or "docker"
        try:
            subprocess.run(
                [docker, "stop", "-t", "3", self._container_name],
                capture_output=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            pass
        finally:
            self._running = False

    async def exec(
        self,
        command: list[str],
        *,
        cwd: str | Path | None = None,
        stdin: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> tuple[int, str, str]:
        """在沙箱内执行命令。

        Args:
            command: 命令及参数列表，如 ["python", "script.py"]
            cwd: 容器内工作目录（默认为 project_root）
            stdin: 标准输入文本
            env: 额外环境变量
            timeout: 超时秒数

        Returns:
            (returncode, stdout, stderr)
        """
        if not self._running:
            raise SandboxUnavailableError("Sandbox is not running")

        docker = shutil.which("docker") or "docker"
        workdir = str(Path(cwd).resolve()) if cwd else str(self.project_root.resolve())

        cmd: list[str] = [docker, "exec", "-w", workdir]

        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.append(self._container_name)
        cmd.extend(command)

        logger.debug(f"[Sandbox] exec: {' '.join(cmd)}")

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE if stdin else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            if timeout:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=stdin.encode("utf-8") if stdin else None),
                    timeout=timeout,
                )
            else:
                stdout_bytes, stderr_bytes = await proc.communicate(
                    input=stdin.encode("utf-8") if stdin else None,
                )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return (-1, "", f"Command timed out after {timeout}s")

        return (
            proc.returncode or 0,
            stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else "",
            stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else "",
        )

    async def exec_python(
        self,
        code: str,
        *,
        timeout: float = 30.0,
    ) -> tuple[int, str, str]:
        """在沙箱内执行 Python 代码。

        Args:
            code: Python 源代码
            timeout: 超时秒数

        Returns:
            (returncode, stdout, stderr)
        """
        return await self.exec(
            ["python", "-c", code],
            timeout=timeout,
        )

    async def exec_shell(
        self,
        command: str,
        *,
        timeout: float = 30.0,
    ) -> tuple[int, str, str]:
        """在沙箱内执行 Shell 命令。

        Args:
            command: Shell 命令字符串
            timeout: 超时秒数

        Returns:
            (returncode, stdout, stderr)
        """
        return await self.exec(
            ["bash", "-lc", command],
            timeout=timeout,
        )

    async def read_file(self, path: str | Path) -> str:
        """从沙箱内读取文件。"""
        ret, stdout, stderr = await self.exec(["cat", str(path)])
        if ret != 0:
            raise FileNotFoundError(f"[Sandbox] read failed: {path} — {stderr}")
        return stdout

    async def write_file(self, path: str | Path, content: str) -> bool:
        """向沙箱内写入文件。"""
        ret, _, stderr = await self.exec(
            ["tee", str(path)],
            stdin=content,
        )
        if ret != 0:
            logger.error(f"[Sandbox] write failed: {path} — {stderr}")
            return False
        return True

    async def list_dir(self, path: str | Path) -> list[str]:
        """列出沙箱内目录内容。"""
        ret, stdout, stderr = await self.exec(["ls", "-1", str(path)])
        if ret != 0:
            return []
        return [line.strip() for line in stdout.splitlines() if line.strip()]

    async def file_exists(self, path: str | Path) -> bool:
        """检查沙箱内文件是否存在。"""
        ret, _, _ = await self.exec(["test", "-f", str(path)])
        return ret == 0

    async def dir_exists(self, path: str | Path) -> bool:
        """检查沙箱内目录是否存在。"""
        ret, _, _ = await self.exec(["test", "-d", str(path)])
        return ret == 0

    async def delete(self, path: str | Path) -> bool:
        """删除沙箱内文件或目录。"""
        ret, _, stderr = await self.exec(["rm", "-rf", str(path)])
        if ret != 0:
            logger.error(f"[Sandbox] delete failed: {path} — {stderr}")
            return False
        return True

    async def mkdir(self, path: str | Path) -> bool:
        """在沙箱内创建目录。"""
        ret, _, stderr = await self.exec(["mkdir", "-p", str(path)])
        if ret != 0:
            logger.error(f"[Sandbox] mkdir failed: {path} — {stderr}")
            return False
        return True
