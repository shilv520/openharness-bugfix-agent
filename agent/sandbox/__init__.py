"""
SecuritySandbox — Agent 行为安全边界
=====================================

PDF Harness Engineering: 沙箱 = 唯一安全边界

所有 Agent 行为（Python 执行 / Shell 命令 / 文件读写 / 网络请求）
全部在 Docker 容器内完成，与宿主机完全隔离。

两层后端:
  - Docker backend: 完整隔离（--network=none + 资源限制）
  - Local fallback:   开发模式，Docker 不可用时自动降级

路径约定:
  /workspace/          → 项目目录（bind-mount 进容器）
  /memories/           → MemoryBackend（不进沙箱）
  /persisted-skills/   → SkillsStoreBackend（不进沙箱）
"""

from .sandbox import (
    SecuritySandbox,
    SandboxAvailability,
    SandboxUnavailableError,
    get_docker_availability,
)
from .sandbox_backend import SandboxBackend, create_sandbox_backend

__all__ = [
    "SecuritySandbox",
    "SandboxAvailability",
    "SandboxUnavailableError",
    "SandboxBackend",
    "create_sandbox_backend",
    "get_docker_availability",
]
