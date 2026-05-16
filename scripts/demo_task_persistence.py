#!/usr/bin/env python
"""
任务规划持久化 Demo 脚本
========================

验证 write_todos + Agent state 持久化 + 重启恢复:

  1. write_todos 写入任务计划
  2. update_todo 更新单个任务状态
  3. 任务进度追踪
  4. Agent state 保存/加载
  5. get_pending_plans 获取未完成计划
  6. resume_session 重启恢复
  7. 快照和恢复
  8. 文件存储后端
  9. 嵌套计划（replan 产���新计划）
  10. 端到端: 模拟重启恢复完整流程
"""

import asyncio
import io
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

TEST_DIR = "data/test_persistence_demo"


async def test_1_write_todos():
    """Test 1: write_todos 写入任务计划"""
    print("=" * 60)
    print("TEST 1: write_todos — 写入任务计划")
    print("=" * 60)

    from agent.task_persistence import TaskPersistence, TaskItem

    tp = TaskPersistence(backend="file", storage_dir=TEST_DIR)

    todos = [
        TaskItem(title="审查BigFraction空指针风险", agent="code-reviewer"),
        TaskItem(title="分析除零Bug根因", agent="bug-analyzer", depends_on=[0]),
        TaskItem(title="生成修复补丁", agent="code-fixer", depends_on=[1]),
        TaskItem(title="验证修复正确性", agent="test-validator", depends_on=[2]),
    ]

    plan = await tp.write_todos(
        session_id="demo-session-001",
        todos=todos,
        title="修复BigFraction除零Bug",
    )

    print(f"  计划ID: {plan.plan_id}")
    print(f"  标题: {plan.title}")
    print(f"  步骤数: {plan.total_steps}")
    print(f"  进度: {plan.progress:.0%}")

    assert plan.plan_id.startswith("plan-")
    assert plan.total_steps == 4
    assert plan.status.value == "pending"
    print(f"  ✓ 任务计划写入成功")

    # 验证可读回
    loaded = await tp.get_plan(plan.plan_id)
    assert loaded is not None
    assert loaded.title == plan.title
    print(f"  ✓ 任务计划可读回")

    print("  PASS\n")


async def test_2_update_todo():
    """Test 2: update_todo 更新任务状态"""
    print("=" * 60)
    print("TEST 2: update_todo — 更新任务状态")
    print("=" * 60)

    from agent.task_persistence import TaskPersistence, TaskItem

    tp = TaskPersistence(backend="file", storage_dir=TEST_DIR)

    todos = [
        TaskItem(title="任务A", agent="agent-a"),
        TaskItem(title="任务B", agent="agent-b"),
    ]
    plan = await tp.write_todos(session_id="demo-session-002", todos=todos)

    # 开始执行
    updated = await tp.update_todo(plan.plan_id, 0, status="in_progress")
    assert updated.todos[0].status.value == "in_progress"
    assert updated.todos[0].started_at is not None
    print(f"  ✓ 任务0 → in_progress (started_at: {updated.todos[0].started_at[:19]})")

    # 完成
    updated = await tp.update_todo(
        plan.plan_id, 0, status="completed",
        result={"bug_count": 3, "severity": "HIGH"},
    )
    assert updated.todos[0].status.value == "completed"
    assert updated.todos[0].completed_at is not None
    assert updated.todos[0].result["bug_count"] == 3
    print(f"  ✓ 任务0 → completed (result: {updated.todos[0].result})")

    # 失败
    updated = await tp.update_todo(
        plan.plan_id, 1, status="failed",
        error="LLM API 超时",
    )
    assert updated.todos[1].status.value == "failed"
    assert updated.todos[1].error == "LLM API 超时"
    print(f"  ✓ 任务1 → failed (error: {updated.todos[1].error})")

    print("  PASS\n")


async def test_3_progress_tracking():
    """Test 3: 任务进度追踪"""
    print("=" * 60)
    print("TEST 3: 进度追踪")
    print("=" * 60)

    from agent.task_persistence import TaskPersistence, TaskItem

    tp = TaskPersistence(backend="file", storage_dir=TEST_DIR)

    todos = [
        TaskItem(title="审查", agent="reviewer"),
        TaskItem(title="分析", agent="analyzer"),
        TaskItem(title="修复", agent="fixer"),
        TaskItem(title="验证", agent="validator"),
    ]
    plan = await tp.write_todos(session_id="demo-session-003", todos=todos)

    steps = [
        (0, "completed", {"bugs": 2}),
        (1, "completed", {"root_cause": "null check missing"}),
        (2, "in_progress", None),
    ]

    for idx, status, result in steps:
        await tp.update_todo(plan.plan_id, idx, status=status, result=result)
        current = await tp.get_plan(plan.plan_id)
        print(f"  Step {idx+1}/4 ({status}): 进度 {current.progress:.0%}")

    final = await tp.get_plan(plan.plan_id)
    assert final.progress == 0.5  # 2/4 completed
    assert not final.is_completed
    print(f"  ✓ 进度追踪: {final.progress:.0%} ({final.current_step}/4 steps)")

    # 完成所有
    await tp.update_todo(plan.plan_id, 2, status="completed")
    await tp.update_todo(plan.plan_id, 3, status="completed")
    final = await tp.get_plan(plan.plan_id)
    assert final.is_completed
    assert final.completed_at is not None
    print(f"  ✓ 计划完成: {final.progress:.0%} ({final.completed_at[:19]})")

    print("  PASS\n")


async def test_4_agent_state():
    """Test 4: Agent state 保存/加载"""
    print("=" * 60)
    print("TEST 4: Agent State 保存/加载")
    print("=" * 60)

    from agent.task_persistence import TaskPersistence

    tp = TaskPersistence(backend="file", storage_dir=TEST_DIR)

    # 模拟 Agent 状态
    agent_state = {
        "code": "public class Test { void foo() { a.divide(b); } }",
        "language": "java",
        "plan": ["审查代码", "分析Bug", "修复Bug", "验证修复"],
        "current_step": 2,
        "review_result": {"bugs": [{"type": "ArithmeticException", "line": 5}]},
        "analysis_result": {"root_cause": "除数为零未检查"},
        "fix_result": None,
        "replan_count": 0,
        "done": False,
    }

    key = await tp.save_agent_state("demo-session-004", agent_state)
    print(f"  ✓ Agent 状态已保存: {key}")
    print(f"    状态 keys: {list(agent_state.keys())}")

    # 加载
    loaded = await tp.load_agent_state("demo-session-004")
    assert loaded is not None
    assert loaded["state"]["current_step"] == 2
    assert loaded["state"]["language"] == "java"
    print(f"  ✓ Agent 状态已加载: current_step={loaded['state']['current_step']}")

    # 删除
    await tp.delete_agent_state("demo-session-004")
    deleted = await tp.load_agent_state("demo-session-004")
    assert deleted is None
    print(f"  ✓ Agent 状态已删除")

    print("  PASS\n")


async def test_5_pending_plans():
    """Test 5: get_pending_plans 获取未完成计划"""
    print("=" * 60)
    print("TEST 5: 获取未完成计划（重启恢复）")
    print("=" * 60)

    from agent.task_persistence import TaskPersistence, TaskItem

    tp = TaskPersistence(backend="file", storage_dir=TEST_DIR)

    # 创建几个计划（有的完成，有的未完成）
    for i in range(3):
        todos = [TaskItem(title=f"任务{i}-A", agent="agent-a"),
                 TaskItem(title=f"任务{i}-B", agent="agent-b")]
        plan = await tp.write_todos(session_id=f"pending-test-{i}", todos=todos)

        if i == 1:
            # 只有第 2 个计划完成
            await tp.update_todo(plan.plan_id, 0, status="completed")
            await tp.update_todo(plan.plan_id, 1, status="completed")

    pending = await tp.get_pending_plans()
    print(f"  总计划数: {(await tp.get_stats())['total_plans']}")
    print(f"  未完成: {len(pending)}")
    for p in pending:
        print(f"    - {p.plan_id}: {p.title} ({p.progress:.0%})")

    assert len(pending) >= 2  # test 1-3 可能还有未完成的
    print(f"  ✓ 未完成计划可查询")

    print("  PASS\n")


async def test_6_resume_session():
    """Test 6: resume_session 重启恢复"""
    print("=" * 60)
    print("TEST 6: resume_session — 重启恢复")
    print("=" * 60)

    from agent.task_persistence import TaskPersistence, TaskItem

    tp = TaskPersistence(backend="file", storage_dir=TEST_DIR)

    # 模拟执行到一半的会话
    todos = [
        TaskItem(title="审查代码", agent="code-reviewer", status="completed",
                 result={"bugs": 2}),
        TaskItem(title="分析Bug", agent="bug-analyzer", status="completed",
                 result={"root_cause": "NPE"}),
        TaskItem(title="修复Bug", agent="code-fixer", status="in_progress"),
        TaskItem(title="验证修复", agent="test-validator", status="pending"),
    ]
    plan = await tp.write_todos(
        session_id="resume-test-session",
        todos=todos,
        title="模拟中断的Bug修复",
    )

    agent_state = {
        "code": "a.divide(b)",
        "language": "java",
        "current_step": 2,
        "review_result": {"bugs": 2},
        "analysis_result": {"root_cause": "NPE"},
    }
    await tp.save_agent_state("resume-test-session", agent_state)

    print(f"  [模拟: 进程重启...]")
    print()

    # 重启后恢复
    result = await tp.resume_session("resume-test-session")
    assert result["resumable"]
    assert result["plan"] is not None
    assert result["state"] is not None

    print(f"  可恢复: {result['resumable']}")
    print(f"  计划: {result['plan'].title}")
    print(f"  当前步骤: {result['plan'].current_step}/{result['plan'].total_steps}")
    print(f"  进度: {result['plan'].progress:.0%}")
    print(f"  阻断任务: {[t.title for t in result['plan'].blocked_tasks]}")
    print(f"  状态 current_step: {result['state'].get('current_step')}")
    print(f"  ✓ 会话可完整恢复")

    # 继续执行
    await tp.update_todo(plan.plan_id, 2, status="completed",
                         result={"patch": "添加null检查"})
    await tp.update_todo(plan.plan_id, 3, status="completed",
                         result={"test_passed": True})

    final = await tp.get_plan(plan.plan_id)
    assert final.is_completed
    print(f"  ✓ 恢复后完成执行: {final.progress:.0%}")

    print("  PASS\n")


async def test_7_snapshot_restore():
    """Test 7: 快照和恢复"""
    print("=" * 60)
    print("TEST 7: 快照和恢复")
    print("=" * 60)

    from agent.task_persistence import TaskPersistence, TaskItem

    tp = TaskPersistence(backend="file", storage_dir=TEST_DIR)

    # 创建计划并执行到一半
    todos = [
        TaskItem(title="步骤1", agent="agent-a", status="completed"),
        TaskItem(title="步骤2", agent="agent-b", status="in_progress"),
    ]
    plan = await tp.write_todos(session_id="snapshot-test", todos=todos)

    # 创建快照
    snap = await tp.snapshot("snapshot-test")
    print(f"  ✓ 快照创建: {snap['snapshot_time'][:19]}")
    print(f"    计划: {snap['plan']['title']} ({snap['plan']['current_step']}/{snap['plan']['total_steps']})")

    # 修改计划
    await tp.update_todo(plan.plan_id, 1, status="completed")
    modified = await tp.get_plan(plan.plan_id)
    assert modified.progress == 1.0

    # 恢复到快照
    await tp.restore("snapshot-test", snap)
    restored = await tp.get_plan(plan.plan_id)
    assert restored.progress == 0.5  # 回到快照时的状态
    print(f"  ✓ 恢复后进度: {restored.progress:.0%} (回到快照状态)")

    print("  PASS\n")


async def test_8_file_backend():
    """Test 8: 文件存储后端验证"""
    print("=" * 60)
    print("TEST 8: 文件存储后端")
    print("=" * 60)

    from agent.task_persistence import FileStorageBackend

    be = FileStorageBackend(base_dir=f"{TEST_DIR}/backend_test")

    # CRUD
    await be.set("key1", json.dumps({"data": "test_value"}))
    val = await be.get("key1")
    assert val is not None
    assert "test_value" in val
    print(f"  ✓ 写入/读取: {val[:50]}")

    assert await be.exists("key1")
    print(f"  ✓ exists: True")

    keys = await be.keys("key*")
    assert "key1" in keys
    print(f"  ✓ keys('key*'): {keys}")

    await be.delete("key1")
    assert not await be.exists("key1")
    print(f"  ✓ delete + exists: False")

    # 验证文件实际存在
    path = be._key_to_path("key1")
    assert not path.exists()
    print(f"  ✓ 文件已删除: {path}")

    print("  PASS\n")


async def test_9_nested_plan():
    """Test 9: 嵌套计划（replan 产生新计划）"""
    print("=" * 60)
    print("TEST 9: 嵌套计划（replan 场景）")
    print("=" * 60)

    from agent.task_persistence import TaskPersistence, TaskItem

    tp = TaskPersistence(backend="file", storage_dir=TEST_DIR)

    # 初始计划
    plan1 = await tp.write_todos(
        session_id="replan-test",
        todos=[
            TaskItem(title="审查代码", agent="code-reviewer"),
            TaskItem(title="分析Bug", agent="bug-analyzer"),
        ],
        title="初始修复计划",
    )

    # 执行失败 → replan 产生新计划
    await tp.update_todo(plan1.plan_id, 0, status="failed", error="审查失败")
    await tp.update_todo(plan1.plan_id, 1, status="skipped")

    plan2 = await tp.write_todos(
        session_id="replan-test",
        todos=[
            TaskItem(title="重新审查（更详细）", agent="code-reviewer"),
            TaskItem(title="深入分析", agent="bug-analyzer"),
            TaskItem(title="生成补丁", agent="code-fixer"),
        ],
        title="Replan: 修复计划 v2",
        parent_plan_id=plan1.plan_id,
    )

    print(f"  原始计划: {plan1.plan_id} (progress: {plan1.progress:.0%})")
    print(f"  重规划: {plan2.plan_id}")
    print(f"    父计划: {plan2.parent_plan_id}")
    print(f"    额外步骤: {plan2.total_steps}")

    assert plan2.parent_plan_id == plan1.plan_id
    assert plan2.total_steps == 3
    print(f"  ✓ 嵌套计划创建成功（保持父子关联）")

    # 验证父子关联可追踪
    loaded = await tp.get_plan(plan2.plan_id)
    assert loaded.parent_plan_id == plan1.plan_id
    print(f"  ✓ 父子关联可追踪")

    print("  PASS\n")


async def test_10_full_restart_simulation():
    """Test 10: 端到端 — 模拟重启恢复完整流程"""
    print("=" * 60)
    print("TEST 10: 端到端 — 模拟重启恢复")
    print("=" * 60)

    from agent.task_persistence import TaskPersistence, TaskItem

    tp = TaskPersistence(backend="file", storage_dir=TEST_DIR)

    print()
    print("  ┌──────────────────────────────────────────────┐")
    print("  │  初始执行 (first run)                         │")
    print("  ├──────────────────────────────────────────────┤")

    # Step 1: 写入任务计划
    todos = [
        TaskItem(title="审查代码中的空指针风险", agent="code-reviewer"),
        TaskItem(title="分析空指针根因", agent="bug-analyzer", depends_on=[0]),
        TaskItem(title="生成修复补丁", agent="code-fixer", depends_on=[1]),
        TaskItem(title="验证修复正确性", agent="test-validator", depends_on=[2]),
    ]
    plan = await tp.write_todos(
        session_id="e2e-restart-session",
        todos=todos,
        title="修复 BigFraction 空指针Bug",
    )
    print(f"  │  计划: {plan.plan_id} ({plan.total_steps} steps)")

    # Step 2: 执行前两步
    await tp.update_todo(plan.plan_id, 0, status="in_progress")
    await tp.update_todo(
        plan.plan_id, 0, status="completed",
        result={"bug_candidates": [
            {"location": "line 45", "type": "NullPointerException", "severity": "HIGH"}
        ]}
    )
    print(f"  │  Step 1/4: code-reviewer → completed (1 bug found)")

    await tp.update_todo(plan.plan_id, 1, status="in_progress")
    await tp.update_todo(
        plan.plan_id, 1, status="completed",
        result={"root_cause": "Optional.get() 未检查 isPresent()", "confidence": 0.9}
    )
    print(f"  │  Step 2/4: bug-analyzer → completed (root cause found)")

    # Step 3: 保存 Agent 状态
    agent_state = {
        "code": "Optional<BigDecimal> val = getValue(); return val.get();",
        "language": "java",
        "current_step": 2,
        "review_result": {"bugs": 1},
        "analysis_result": {"root_cause": "Optional.get() without isPresent() check"},
        "fix_result": None,
        "validation_result": None,
        "replan_count": 0,
        "done": False,
    }
    await tp.save_agent_state("e2e-restart-session", agent_state)
    print(f"  │  Agent 状态已保存")

    # Step 4: 开始第 3 步 → 模拟崩溃
    await tp.update_todo(plan.plan_id, 2, status="in_progress")
    print(f"  │  Step 3/4: code-fixer → in_progress")
    print(f"  │  💥 [模拟: 进程崩溃!]")
    print("  ├──────────────────────────────────────────────┤")

    # ── 重启恢复 ──
    print("  │  重启恢复 (resume)                            │")
    print("  ├──────────────────────────────────────────────┤")

    # 重新创建 TaskPersistence（模拟进程重启）
    tp2 = TaskPersistence(backend="file", storage_dir=TEST_DIR)
    result = await tp2.resume_session("e2e-restart-session")

    print(f"  │  可恢复: {result['resumable']}")
    print(f"  │  计划标题: {result['plan'].title}")
    print(f"  │  当前步骤: {result['plan'].current_step}/{result['plan'].total_steps}")
    print(f"  │  已完成: {[t.title for t in result['plan'].todos if t.status.value == 'completed']}")
    print(f"  │  进行中: {[t.title for t in result['plan'].todos if t.status.value == 'in_progress']}")

    assert result["resumable"]
    assert result["state"]["current_step"] == 2
    print(f"  │  ✓ 状态完整恢复")

    # 继续执行
    await tp2.update_todo(
        plan.plan_id, 2, status="completed",
        result={"patch": "添加 isPresent() 检查", "fixed_code": "if (val.isPresent()) return val.get();"}
    )
    print(f"  │  Step 3/4: code-fixer → completed (patch generated)")

    await tp2.update_todo(plan.plan_id, 3, status="in_progress")
    await tp2.update_todo(
        plan.plan_id, 3, status="completed",
        result={"test_passed": True, "validation_reason": "修复后空指针已消除"}
    )
    print(f"  │  Step 4/4: test-validator → completed (tests passed)")

    final = await tp2.get_plan(plan.plan_id)
    print("  ├──────────────────────────────────────────────┤")
    print(f"  │  最终状态: {final.status.value}")
    print(f"  │  总耗时: 在崩溃前已保存 → 重启后无缝继续")
    print(f"  │  丢失的工作: 仅 Step 3 的 in_progress → 需重新执行")
    print("  └──────────────────────────────────────────────┘")

    assert final.is_completed
    stats = await tp2.get_stats()
    print(f"\n  ✓ 端到端重启恢复成功")
    print(f"     总计划数: {stats['total_plans']}")
    print(f"     未完成: {stats['pending_plans']}")

    print("  PASS\n")


async def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   任务规划持久化 — Demo 测试                      ║")
    print("║   write_todos + State Persistence + Restart      ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    # 清理之前的测试数据
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    tests = [
        ("write_todos", test_1_write_todos),
        ("update_todo", test_2_update_todo),
        ("进度追踪", test_3_progress_tracking),
        ("Agent State 保存/加载", test_4_agent_state),
        ("未完成计划查询", test_5_pending_plans),
        ("重启恢复", test_6_resume_session),
        ("快照恢复", test_7_snapshot_restore),
        ("文件后端", test_8_file_backend),
        ("嵌套计划", test_9_nested_plan),
        ("端到端重启模拟", test_10_full_restart_simulation),
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

    # 清理
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)

    print("╔══════════════════════════════════════════════════╗")
    print(f"║  结果: {passed} passed, {failed} failed            ║")
    print("╚══════════════════════════════════════════════════╝")


if __name__ == "__main__":
    asyncio.run(main())
