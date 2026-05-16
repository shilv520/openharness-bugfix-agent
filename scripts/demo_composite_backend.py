#!/usr/bin/env python
"""
CompositeBackend 演示脚本
==========================

验证虚拟文件系统的所有路由:

  1. LocalFSBackend: 读写本地文件
  2. MemoryBackend: 读写用户记忆
  3. SkillsStoreBackend: 读写持久化技能
  4. CompositeBackend 路由: 同一 vfs 下不同路径走不同 Backend
  5. 回退到 default Backend
"""

import asyncio
import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


async def test_1_local_fs():
    """Test 1: LocalFSBackend 读写本地文件"""
    print("=" * 60)
    print("TEST 1: LocalFSBackend — 读写本地文件")
    print("=" * 60)

    from agent.backends.local import LocalFSBackend

    backend = LocalFSBackend(root=Path(__file__).parent.parent)

    # Write
    ok = await backend.write("/workspace/demo_test.txt", "Hello from CompositeBackend!")
    assert ok, "Write failed"
    print("  ✓ write(/workspace/demo_test.txt)")

    # Read
    content = await backend.read("/workspace/demo_test.txt")
    assert "CompositeBackend" in content, f"Read content mismatch: {content}"
    print(f"  ✓ read(/workspace/demo_test.txt) → '{content}'")

    # List
    items = await backend.list_dir("/workspace")
    assert "demo_test.txt" in items, f"demo_test.txt not found in {items}"
    print(f"  ✓ list_dir(/workspace) → {len(items)} items (includes demo_test.txt)")

    # Exists
    assert await backend.exists("/workspace/demo_test.txt"), "exists failed"
    print("  ✓ exists(/workspace/demo_test.txt) → True")

    # Delete
    ok = await backend.delete("/workspace/demo_test.txt")
    assert ok, "delete failed"
    assert not await backend.exists("/workspace/demo_test.txt"), "should be deleted"
    print("  ✓ delete + verify removed")

    print("  PASS\n")


async def test_2_memory_backend():
    """Test 2: MemoryBackend 读写用户记忆"""
    print("=" * 60)
    print("TEST 2: MemoryBackend — 读写用户记忆")
    print("=" * 60)

    from agent.backends.memory import MemoryBackend

    backend = MemoryBackend()

    # Write preferences
    md_content = """# User Preferences & Facts

- **preferred_language**: Chinese
- **bug_type_focus**: NullPointerException
- **recent_project**: commons-math
"""
    ok = await backend.write("/memories/test_user/preferences.md", md_content)
    assert ok, "Write failed"
    print("  ✓ write(/memories/test_user/preferences.md)")

    # Read back
    content = await backend.read("/memories/test_user/preferences.md")
    assert "NullPointerException" in content, f"Content mismatch: {content}"
    print(f"  ✓ read back → verified 'NullPointerException' in content")

    # Exists
    assert await backend.exists("/memories/test_user/preferences.md")
    print("  ✓ exists → True")

    # List
    items = await backend.list_dir("/memories/test_user")
    assert "preferences.md" in items
    print(f"  ✓ list_dir → {items}")

    print("  PASS\n")


async def test_3_skills_store_backend():
    """Test 3: SkillsStoreBackend 读写持久化技能"""
    print("=" * 60)
    print("TEST 3: SkillsStoreBackend — 读写持久化技能")
    print("=" * 60)

    from agent.backends.skills_store import SkillsStoreBackend

    backend = SkillsStoreBackend()

    # List existing skills
    items = await backend.list_dir("/persisted-skills")
    print(f"  ✓ list_dir(/persisted-skills) → {items}")

    # Write a new skill
    skill_md = """---
name: demo-composite-skill
description: A skill created via CompositeBackend demo
version: 1.0.0
scope: user
tags: [demo, test]
---

# Demo Composite Skill

This skill was written through the CompositeBackend virtual filesystem.
"""
    ok = await backend.write("/persisted-skills/demo-composite-skill/SKILL.md", skill_md)
    print(f"  ✓ write(/persisted-skills/demo-composite-skill/SKILL.md) → {ok}")

    # Read back
    content = await backend.read("/persisted-skills/demo-composite-skill/SKILL.md")
    assert "CompositeBackend" in content, "Content mismatch"
    print(f"  ✓ read back → verified in content ({len(content)} chars)")

    # Exists
    assert await backend.exists("/persisted-skills/demo-composite-skill/SKILL.md")
    print("  ✓ exists → True")

    # Cleanup
    await backend.delete("/persisted-skills/demo-composite-skill/SKILL.md")
    print("  ✓ deleted demo skill")

    print("  PASS\n")


async def test_4_composite_routing():
    """Test 4: CompositeBackend 路由 — 同一 vfs 下不同路径走不同 Backend"""
    print("=" * 60)
    print("TEST 4: CompositeBackend 路由")
    print("=" * 60)

    from agent.backends.composite import create_default_composite

    vfs = create_default_composite()

    # Route 1: /workspace/** → LocalFSBackend
    await vfs.write("/workspace/routing_test.txt", "LocalFS")
    content = await vfs.read("/workspace/routing_test.txt")
    assert content == "LocalFS"
    print(f"  ✓ /workspace/routing_test.txt → LocalFSBackend (read='{content}')")

    # Route 2: /memories/** → MemoryBackend
    await vfs.write("/memories/demo_user/preferences.md",
                     "- **test_key**: test_value\n")
    content = await vfs.read("/memories/demo_user/preferences.md")
    assert "test_key" in content
    print(f"  ✓ /memories/demo_user/preferences.md → MemoryBackend")

    # Route 3: /persisted-skills/** → SkillsStoreBackend
    items = await vfs.list_dir("/persisted-skills")
    print(f"  ✓ /persisted-skills → SkillsStoreBackend ({len(items)} skills)")

    # Cleanup
    await vfs.delete("/workspace/routing_test.txt")
    print("  ✓ cleaned up")

    print("  PASS\n")


async def test_5_unmatched_fallback():
    """Test 5: 未匹配路径回退到 default Backend (LocalFSBackend)"""
    print("=" * 60)
    print("TEST 5: 未匹配路径 → default Backend")
    print("=" * 60)

    from agent.backends.composite import create_default_composite

    vfs = create_default_composite()

    # /some-random/path 不在任何路由中 → LocalFSBackend
    backend_name = vfs.which_backend("/some-random/path/file.txt")
    assert backend_name == "LocalFSBackend", f"Expected LocalFSBackend, got {backend_name}"
    print(f"  ✓ /some-random/path/file.txt → {backend_name} (default)")

    # /workspace 也在默认路由中
    backend_name = vfs.which_backend("/workspace/code.java")
    assert backend_name == "LocalFSBackend"
    print(f"  ✓ /workspace/code.java → {backend_name} (default)")

    # 明确的路由
    backend_name = vfs.which_backend("/memories/user1/prefs.md")
    assert backend_name == "MemoryBackend", f"Expected MemoryBackend, got {backend_name}"
    print(f"  ✓ /memories/user1/prefs.md → {backend_name} (routed)")

    backend_name = vfs.which_backend("/persisted-skills/test/SKILL.md")
    assert backend_name == "SkillsStoreBackend", f"Expected SkillsStoreBackend, got {backend_name}"
    print(f"  ✓ /persisted-skills/test/SKILL.md → {backend_name} (routed)")

    print("  PASS\n")


async def test_6_agent_view():
    """Test 6: Agent 视角 — 就像操作一个普通文件系统"""
    print("=" * 60)
    print("TEST 6: Agent 视角的虚拟文件系统")
    print("=" * 60)

    from agent.backends.composite import create_default_composite

    vfs = create_default_composite()

    # Agent 看到的: 一棵统一文件树
    print()
    print("  Agent 执行: ls /")
    print("  ┌─────────────────────────────────────┐")
    print("  │ /workspace/        ← 本地磁盘        │")
    print("  │ /memories/         ← Redis+ChromaDB  │")
    print("  │ /persisted-skills/ ← ChromaDB+Redis  │")
    print("  └─────────────────────────────────────┘")

    # Agent: 写入记忆
    await vfs.write(
        "/memories/agent_user/preferences.md",
        "- **preferred_language**: Chinese\n- **bug_count**: 42\n"
    )

    # Agent: 读取记忆
    prefs = await vfs.read("/memories/agent_user/preferences.md")
    print(f"\n  Agent 执行: cat /memories/agent_user/preferences.md")
    print(f"  → {prefs.strip()}")

    # Agent: 查看可用技能
    skills = await vfs.list_dir("/persisted-skills")
    print(f"\n  Agent 执行: ls /persisted-skills")
    print(f"  → {len(skills)} skills: {skills}")

    # Agent: 在 workspace 创建文件
    await vfs.write("/workspace/agent_output.txt", "Bug analysis complete.")
    exists = await vfs.exists("/workspace/agent_output.txt")
    print(f"\n  Agent 执行: echo 'Bug analysis complete.' > /workspace/agent_output.txt")
    print(f"  → file exists: {exists}")

    # Cleanup
    await vfs.delete("/workspace/agent_output.txt")
    print(f"\n  ✓ Agent 视角验证完成")
    print("  PASS\n")


async def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   CompositeBackend 虚拟文件系统 — Demo 测试      ║")
    print("║   对应 PDF: Harness Engineering 核心架构         ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    tests = [
        ("LocalFSBackend 读写", test_1_local_fs),
        ("MemoryBackend 读写", test_2_memory_backend),
        ("SkillsStoreBackend 读写", test_3_skills_store_backend),
        ("CompositeBackend 路由", test_4_composite_routing),
        ("未匹配路径回退", test_5_unmatched_fallback),
        ("Agent 视角验证", test_6_agent_view),
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
