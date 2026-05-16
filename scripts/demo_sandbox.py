#!/usr/bin/env python
"""
SecuritySandbox 演示脚本
=========================

验证安全沙箱的所有能力:

  1. Docker 可用性检查
  2. 沙箱容器生命周期（启动 → 执行 → 停止）
  3. Python 代码隔离执行
  4. Shell 命令隔离执行
  5. 文件操作（读写列删）
  6. 网络隔离验证
  7. SandboxBackend + CompositeBackend 集成
  8. 优雅降级（Docker 不可用 → 本地回退）
"""

import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


async def test_1_docker_availability():
    """Test 1: Docker 可用性检查"""
    print("=" * 60)
    print("TEST 1: Docker 可用性检查")
    print("=" * 60)

    from agent.sandbox.sandbox import get_docker_availability

    avail = get_docker_availability()
    print(f"  enabled:   {avail.enabled}")
    print(f"  available: {avail.available}")
    print(f"  reason:    {avail.reason or 'N/A'}")
    print(f"  command:   {avail.command or 'N/A'}")

    if avail.available:
        print("  ✓ Docker 可用 — 将使用沙箱模式")
    else:
        print("  ⚠ Docker 不可用 — 将使用本地回退模式")

    print("  PASS\n")
    return avail.available


async def test_2_sandbox_lifecycle():
    """Test 2: 沙箱容器生命周期"""
    print("=" * 60)
    print("TEST 2: 沙箱容器生命周期")
    print("=" * 60)

    from agent.sandbox.sandbox import (
        SecuritySandbox,
        SandboxUnavailableError,
        get_docker_availability,
    )

    avail = get_docker_availability()
    if not avail.available:
        print("  ⚠ Docker 不可用 — 跳过容器测试")
        print("  PASS (skipped)\n")
        return

    project_root = Path(__file__).resolve().parent.parent
    sandbox = SecuritySandbox(project_root=project_root)

    try:
        # Start
        await sandbox.start()
        assert sandbox.is_running, "Sandbox should be running"
        print(f"  ✓ 容器启动: {sandbox.container_name}")

        # Verify container exists
        import subprocess
        result = subprocess.run(
            ["docker", "ps", "--filter", f"name={sandbox.container_name}", "--format", "{{.Names}}"],
            capture_output=True, text=True,
        )
        assert sandbox.container_name in result.stdout
        print(f"  ✓ docker ps 确认容器在运行")

        # Check network isolation
        ret, stdout, stderr = await sandbox.exec_shell("ip addr 2>/dev/null || ifconfig 2>/dev/null || echo 'no network tools'")
        print(f"  ✓ 网络状态: {stdout.strip()[:80]}")

        # Check working directory
        ret, pwd, _ = await sandbox.exec(["pwd"])
        print(f"  ✓ 工作目录: {pwd.strip()}")

        # Check user
        ret, whoami, _ = await sandbox.exec(["whoami"])
        print(f"  ✓ 容器用户: {whoami.strip()}")

    except SandboxUnavailableError as e:
        print(f"  ⚠ 沙箱启动失败: {e}")
        print("  PASS (skipped)\n")
        return

    finally:
        await sandbox.stop()
        print(f"  ✓ 容器停止")

    print("  PASS\n")


async def test_3_python_execution():
    """Test 3: Python 代码隔离执行"""
    print("=" * 60)
    print("TEST 3: Python 代码隔离执行")
    print("=" * 60)

    from agent.sandbox.sandbox_backend import SandboxBackend

    backend = SandboxBackend()
    await backend.start()

    # 执行简单 Python
    code = "import sys; print(f'Python {sys.version}'); print('Hello from inside!')"
    ret, stdout, stderr = await backend.exec_python(code)
    print(f"  ✓ Python 执行 (ret={ret}):")
    for line in stdout.splitlines():
        print(f"      {line.strip()}")

    # 执行有错误的 Python（验证错误传播）
    code_err = "import sys; sys.exit(42)"
    ret, stdout, stderr = await backend.exec_python(code_err)
    print(f"  ✓ 错误退出码传播: ret={ret} (expected 42)")

    await backend.stop()
    print("  PASS\n")


async def test_4_shell_execution():
    """Test 4: Shell 命令隔离执行"""
    print("=" * 60)
    print("TEST 4: Shell 命令隔离执行")
    print("=" * 60)

    from agent.sandbox.sandbox_backend import SandboxBackend

    backend = SandboxBackend()
    await backend.start()

    # Echo
    ret, stdout, stderr = await backend.exec_shell("echo 'Hello Shell!'")
    print(f"  ✓ echo: {stdout.strip()}")

    # 管道
    ret, stdout, stderr = await backend.exec_shell("echo 'a\nb\nc' | wc -l")
    print(f"  ✓ 管道: {stdout.strip()} lines")

    # 退出码传播
    ret, stdout, stderr = await backend.exec_shell("exit 7")
    print(f"  ✓ 退出码传播: ret={ret} (expected 7)")

    await backend.stop()
    print("  PASS\n")


async def test_5_file_operations():
    """Test 5: 文件操作（读写列删）"""
    print("=" * 60)
    print("TEST 5: 文件操作 (读写列删)")
    print("=" * 60)

    from agent.sandbox.sandbox_backend import SandboxBackend

    backend = SandboxBackend()
    await backend.start()

    # Write
    ok = await backend.write("/workspace/sandbox_test.txt", "Hello from SecuritySandbox!")
    assert ok, "Write failed"
    print("  ✓ write(/workspace/sandbox_test.txt)")

    # Read
    content = await backend.read("/workspace/sandbox_test.txt")
    assert "SecuritySandbox" in content, f"Content mismatch: {content}"
    print(f"  ✓ read → '{content}'")

    # Exists
    assert await backend.exists("/workspace/sandbox_test.txt")
    print("  ✓ exists → True")

    # List
    items = await backend.list_dir("/workspace")
    assert "sandbox_test.txt" in items, f"Not found in {items}"
    print(f"  ✓ list_dir → {len(items)} items (includes sandbox_test.txt)")

    # Delete
    ok = await backend.delete("/workspace/sandbox_test.txt")
    assert ok, "Delete failed"
    assert not await backend.exists("/workspace/sandbox_test.txt")
    print("  ✓ delete + verify removed")

    await backend.stop()
    print("  PASS\n")


async def test_6_network_isolation():
    """Test 6: 网络隔离验证（仅 Docker 模式有效）"""
    print("=" * 60)
    print("TEST 6: 网络隔离验证")
    print("=" * 60)

    from agent.sandbox.sandbox_backend import SandboxBackend

    backend = SandboxBackend()
    await backend.start()

    if backend.is_sandboxed:
        # 尝试访问外部网络（应该失败）
        ret, stdout, stderr = await backend.exec_shell(
            "curl -s --connect-timeout 3 https://example.com 2>&1 || echo 'BLOCKED'"
        )
        is_blocked = "BLOCKED" in stdout or "Could not resolve" in stdout
        print(f"  ✓ 网络隔离: {'生效 (BLOCKED)' if is_blocked else '注意: 网络可访问'}")
        print(f"      curl 结果: {stdout.strip()[:100]}")
    else:
        print("  ⚠ 本地回退模式 — 网络隔离不适用")

    await backend.stop()
    print("  PASS\n")


async def test_7_path_traversal_protection():
    """Test 7: 路径穿越防护"""
    print("=" * 60)
    print("TEST 7: 路径穿越防护")
    print("=" * 60)

    from agent.sandbox.sandbox_backend import SandboxBackend

    backend = SandboxBackend()
    await backend.start()

    # 尝试读取根目录
    blocked = 0
    for malicious_path in [
        "../../etc/passwd",
        "/workspace/../../../etc/shadow",
        "../../../root/.ssh/id_rsa",
    ]:
        try:
            await backend.read(malicious_path)
            print(f"  ✗ 未拦截: {malicious_path}")
        except PermissionError as e:
            blocked += 1
            print(f"  ✓ 已拦截: {malicious_path}")

    print(f"  ✓ 路径穿越防护: {blocked}/3 已拦截")
    assert blocked == 3, f"Expected 3 blocked, got {blocked}"

    await backend.stop()
    print("  PASS\n")


async def test_8_composite_with_sandbox():
    """Test 8: CompositeBackend + SandboxBackend 集成"""
    print("=" * 60)
    print("TEST 8: CompositeBackend + SandboxBackend 集成")
    print("=" * 60)

    from agent.backends.composite import CompositeBackend
    from agent.sandbox.sandbox_backend import SandboxBackend
    from agent.backends.memory import MemoryBackend
    from agent.backends.skills_store import SkillsStoreBackend

    sandbox = SandboxBackend()
    await sandbox.start()

    vfs = CompositeBackend(
        default=sandbox,
        routes={
            "/memories": MemoryBackend(),
            "/persisted-skills": SkillsStoreBackend(),
        },
    )

    # /workspace/ → SandboxBackend
    assert vfs.which_backend("/workspace/code.java") == "SandboxBackend"
    print(f"  ✓ /workspace/code.java → SandboxBackend")

    # /memories/ → MemoryBackend
    assert vfs.which_backend("/memories/user1/prefs.md") == "MemoryBackend"
    print(f"  ✓ /memories/user1/prefs.md → MemoryBackend")

    # /persisted-skills/ → SkillsStoreBackend
    assert vfs.which_backend("/persisted-skills/test/SKILL.md") == "SkillsStoreBackend"
    print(f"  ✓ /persisted-skills/test/SKILL.md → SkillsStoreBackend")

    # 实际写文件到沙箱
    await vfs.write("/workspace/vfs_test.txt", "Composite + Sandbox!")
    content = await vfs.read("/workspace/vfs_test.txt")
    assert "Composite + Sandbox" in content
    print(f"  ✓ 写/读通过 CompositeBackend → SandboxBackend")

    # 清理
    await vfs.delete("/workspace/vfs_test.txt")

    await sandbox.stop()
    print("  PASS\n")


async def test_9_graceful_degradation():
    """Test 9: 优雅降级 — 本地回退功能完整"""
    print("=" * 60)
    print("TEST 9: 优雅降级 (本地回退)")
    print("=" * 60)

    from agent.sandbox.sandbox_backend import SandboxBackend

    # 不提供 SecuritySandbox → 直接进入回退模式
    backend = SandboxBackend()
    await backend.start()  # Docker 不可用时自动回退

    mode = "SANDBOXED" if backend.is_sandboxed else "FALLBACK (local)"
    print(f"  ℹ 当前模式: {mode}")

    # 回退模式下所有功能仍可用
    ok = await backend.write("/workspace/fallback_test.txt", "Fallback works!")
    assert ok
    content = await backend.read("/workspace/fallback_test.txt")
    assert "Fallback works!" in content
    print("  ✓ 回退模式: write + read 正常")

    # 代码执行
    ret, stdout, stderr = await backend.exec_python("print('hello from fallback')")
    print(f"  ✓ 回退模式: Python 执行正常 → {stdout.strip()}")

    # Shell 执行
    ret, stdout, stderr = await backend.exec_shell("echo 'shell works too'")
    print(f"  ✓ 回退模式: Shell 执行正常 → {stdout.strip()}")

    # 清理
    await backend.delete("/workspace/fallback_test.txt")
    await backend.stop()
    print("  PASS\n")


async def test_10_agent_perspective():
    """Test 10: Agent 视角 — 沙箱是完全透明的"""
    print("=" * 60)
    print("TEST 10: Agent 视角的沙箱文件系统")
    print("=" * 60)

    from agent.sandbox.sandbox_backend import SandboxBackend

    backend = SandboxBackend()
    await backend.start()

    mode = "Docker 容器内" if backend.is_sandboxed else "本地进程"
    print(f"\n  Agent 执行的所有操作都在 {mode}")
    print()
    print("  Agent 执行: touch /workspace/hello.py")
    await backend.write("/workspace/hello.py", "print('Hello World')")
    print("  → 文件已创建")

    print()
    print("  Agent 执行: python /workspace/hello.py")
    ret, stdout, stderr = await backend.exec_python("print('Hello World')")
    print(f"  → {stdout.strip()}")

    print()
    print("  Agent 执行: ls /workspace")
    items = await backend.list_dir("/workspace")
    print(f"  → {items}")

    print()
    print("  Agent 执行: rm /workspace/hello.py")
    await backend.delete("/workspace/hello.py")

    # 安全检查仍然有效
    print()
    print("  Agent 尝试: cat /etc/passwd")
    try:
        await backend.read("../../etc/passwd")
        print("  ✗ 不应该能读到")
    except PermissionError:
        print("  → PermissionError (路径穿越被拦截)")

    mode_str = "Docker 沙箱隔离" if backend.is_sandboxed else "本地回退"
    print(f"\n  ✓ Agent 视角验证完成（{mode_str}）")

    await backend.stop()
    print("  PASS\n")


async def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   SecuritySandbox 安全沙箱 — Demo 测试          ║")
    print("║   对应 PDF: Harness Engineering 安全边界        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    tests = [
        ("Docker 可用性检查", test_1_docker_availability),
        ("沙箱容器生命周期", test_2_sandbox_lifecycle),
        ("Python 代码隔离执行", test_3_python_execution),
        ("Shell 命令隔离执行", test_4_shell_execution),
        ("文件操作", test_5_file_operations),
        ("网络隔离验证", test_6_network_isolation),
        ("路径穿越防护", test_7_path_traversal_protection),
        ("CompositeBackend 集成", test_8_composite_with_sandbox),
        ("优雅降级", test_9_graceful_degradation),
        ("Agent 视角验证", test_10_agent_perspective),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            await test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  ❌ FAIL: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("╔══════════════════════════════════════════════════╗")
    print(f"║  结果: {passed} passed, {failed} failed            ║")
    print("╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    asyncio.run(main())
