#!/usr/bin/env python
"""
Human-in-the-Loop 双层中断 Demo
===============================

验证双层人工介入机制:

  1. HITLConfig 配置加载
  2. Layer 1: 数据补充中断（data_interrupt）
  3. Layer 1: 数据补充完整流程（interrupt → 人类响应 → resume）
  4. Layer 2: 审批中断（approval_interrupt）
  5. Layer 2: 审批决策流程（interrupt → approved/rejected）
  6. Layer 2: 审批超时自动拒绝
  7. 确认中断（confirm: 是/否）
  8. 安全操作自动批准
  9. HITL + TaskPersistence 集成
  10. 端到端: LangGraph interrupt_before 模拟
"""

import asyncio
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


async def test_1_hitl_config():
    """Test 1: HITLConfig 配置加载"""
    print("=" * 60)
    print("TEST 1: HITLConfig 配置")
    print("=" * 60)

    from agent.human_in_the_loop import HITLConfig, HITLManager

    config = HITLConfig(
        interrupt_before=["fixer", "validator"],
        interrupt_on_tools=["write_file", "git_commit"],
        approval_timeout=300,
        require_approval_for_severity=["HIGH", "CRITICAL"],
    )

    hitl = HITLManager(config=config)

    print(f"  第 1 层数据工具: {config.data_tools}")
    print(f"  第 2 层 interrupt_before: {config.interrupt_before}")
    print(f"  第 2 层 interrupt_on_tools: {config.interrupt_on_tools}")
    print(f"  审批超时: {config.approval_timeout}s")
    print(f"  需要审批的严重级别: {config.require_approval_for_severity}")

    assert "fixer" in hitl.get_interrupt_before_nodes()
    assert hitl.should_interrupt_tool("write_file")
    assert not hitl.should_interrupt_tool("read_file")  # 安全操作
    assert hitl.should_interrupt_node("fixer")
    assert not hitl.should_interrupt_node("reviewer")

    summary = hitl.get_hitl_summary()
    assert summary["pending_approvals"] == 0
    print(f"  ✓ HITL 配置验证通过")
    print("  PASS\n")


async def test_2_layer1_data_interrupt():
    """Test 2: Layer 1 — 数据补充中断"""
    print("=" * 60)
    print("TEST 2: Layer 1 — 数据补充中断")
    print("=" * 60)

    from agent.human_in_the_loop import HITLManager, create_hitl_manager

    hitl = create_hitl_manager()

    request_id = None

    async def collect_events():
        nonlocal request_id
        events = []
        async for event in hitl.data_interrupt(
            tool_name="request_missing_info",
            question="请提供 Bug 的复现步骤和触发条件",
            expected_format="free_text",
            context={"code": "a.divide(b)", "bug_type": "ArithmeticException"},
            timeout=10,
        ):
            events.append(event)
            if event.event_type == "interrupt":
                request_id = event.data_request.request_id
                # 后台提交响应，避免阻塞
                asyncio.create_task(submit_response())
        return events

    async def submit_response():
        await asyncio.sleep(0.1)
        if request_id:
            hitl.submit_data_response(request_id, "Bug在调用a.divide(b)且b=null时复现")

    events = await collect_events()

    # 应该收到 interrupt + resume 事件
    assert len(events) == 2
    assert events[0].event_type == "interrupt"
    assert events[0].data_request is not None
    assert "复现步骤" in events[0].data_request.question
    print(f"  ✓ 收到中断事件: {events[0].event_type}")
    print(f"    问题: {events[0].data_request.question}")
    print(f"    请求ID: {events[0].data_request.request_id}")
    print(f"  ✓ 收到恢复事件: {events[1].event_type}")
    print(f"    回答: {events[1].data_response.answer}")

    # 验证请求已清空
    pending = hitl.get_pending_data_requests()
    assert len(pending) == 0
    print(f"  ✓ 待处理数据请求: {len(pending)}")

    print("  PASS\n")


async def test_3_layer1_full_flow():
    """Test 3: Layer 1 — 数据补充完整流程"""
    print("=" * 60)
    print("TEST 3: Layer 1 — 数据补充完整流程 (interrupt → response → resume)")
    print("=" * 60)

    from agent.human_in_the_loop import HITLManager

    hitl = HITLManager()

    request_id = None

    async def simulate_agent():
        """模拟 Agent 工具调用"""
        nonlocal request_id
        events = []
        async for event in hitl.data_interrupt(
            tool_name="request_missing_info",
            question="Bug 发生在哪个 Java 版本？",
            expected_format="version_string",
            timeout=10,
        ):
            events.append(event)
            if event.event_type == "interrupt":
                request_id = event.data_request.request_id
                # 模拟人类响应（在另一个 "线程" 中）
                asyncio.create_task(simulate_human_response())
        return events

    async def simulate_human_response():
        """模拟人类提交数据"""
        await asyncio.sleep(0.1)  # 模拟人类思考时间
        if request_id:
            hitl.submit_data_response(
                request_id=request_id,
                answer="Java 17.0.9, OpenJDK, Linux x86_64",
                metadata={"source": "human_operator"},
            )

    events = await simulate_agent()

    assert len(events) == 2
    assert events[0].event_type == "interrupt"
    assert events[1].event_type == "resume"
    assert events[1].data_response is not None
    assert "Java 17" in events[1].data_response.answer
    print(f"  ✓ 完整流程: {events[0].event_type} → {events[1].event_type}")
    print(f"    回答: {events[1].data_response.answer}")

    # 验证 pending 列表已清空
    assert len(hitl.get_pending_data_requests()) == 0
    print(f"  ✓ 待处理请求已清空")

    print("  PASS\n")


async def test_4_layer2_approval_interrupt():
    """Test 4: Layer 2 — 审批中断"""
    print("=" * 60)
    print("TEST 4: Layer 2 — 审批中断")
    print("=" * 60)

    from agent.human_in_the_loop import HITLManager

    hitl = HITLManager()

    approval_id = None

    async def collect_approval():
        nonlocal approval_id
        events = []
        async for event in hitl.approval_interrupt(
            action="write_file",
            detail={
                "path": "/workspace/BigFraction.java",
                "patch": "- break;\n+ if (b != null) break;",
                "reason": "修复除零Bug",
            },
            severity="HIGH",
            timeout=10,
        ):
            events.append(event)
            if event.event_type == "interrupt":
                approval_id = event.request.request_id
                asyncio.create_task(submit_approval())
        return events

    async def submit_approval():
        await asyncio.sleep(0.1)
        if approval_id:
            hitl.submit_approval(approval_id, "approved", "修复方案合理")

    events = await collect_approval()

    assert len(events) == 2
    assert events[0].event_type == "interrupt"
    assert events[0].request is not None
    assert events[0].request.action == "write_file"
    assert events[0].request.severity == "HIGH"
    print(f"  ✓ 审批中断事件: {events[0].event_type}")
    print(f"    操作: {events[0].request.action}")
    print(f"    严重级别: {events[0].request.severity}")
    print(f"  ✓ 审批决定: {events[1].event_type} — {events[1].response.comment}")

    pending = hitl.get_pending_approvals()
    assert len(pending) == 0
    print(f"  ✓ 待审批请求: {len(pending)}")

    print("  PASS\n")


async def test_5_layer2_approval_flow():
    """Test 5: Layer 2 — 审批决策流程"""
    print("=" * 60)
    print("TEST 5: Layer 2 — 审批决策流程 (interrupt → approved)")
    print("=" * 60)

    from agent.human_in_the_loop import HITLManager

    hitl = HITLManager()

    request_id = None

    async def simulate_agent_approval_flow():
        nonlocal request_id
        events = []
        async for event in hitl.approval_interrupt(
            action="apply_patch",
            detail={"patch": "修复空指针", "files_changed": 3},
            severity="HIGH",
            timeout=10,
        ):
            events.append(event)
            if event.event_type == "interrupt":
                request_id = event.request.request_id
                asyncio.create_task(simulate_human_approval())
        return events

    async def simulate_human_approval():
        await asyncio.sleep(0.1)
        if request_id:
            hitl.submit_approval(
                request_id=request_id,
                decision="approved",
                comment="修复方案合理，批准执行",
            )

    events = await simulate_agent_approval_flow()

    assert len(events) == 2
    assert events[0].event_type == "interrupt"
    assert events[1].event_type == "approved"
    assert events[1].response.decision.value == "approved"
    print(f"  ✓ 审批流程: {events[0].event_type} → {events[1].event_type}")
    print(f"    决策: {events[1].response.decision.value}")
    print(f"    评论: {events[1].response.comment}")

    assert len(hitl.get_pending_approvals()) == 0
    print(f"  ✓ 待审批请求已清空")

    # 验证审批日志
    log = hitl.get_approval_log()
    assert len(log) == 1
    print(f"  ✓ 审批日志: {len(log)} 条记录")

    print("  PASS\n")


async def test_6_approval_timeout():
    """Test 6: Layer 2 — 审批超时自动拒绝"""
    print("=" * 60)
    print("TEST 6: Layer 2 — 审批超时自动拒绝")
    print("=" * 60)

    from agent.human_in_the_loop import HITLManager

    hitl = HITLManager()

    events = []
    async for event in hitl.approval_interrupt(
        action="git_commit",
        detail={"message": "fix: 修复空指针Bug", "branch": "main"},
        severity="MEDIUM",
        timeout=1,  # 1 秒超时
    ):
        events.append(event)
        # 不提交响应 → 等待超时

    assert len(events) == 2
    assert events[0].event_type == "interrupt"
    assert events[1].event_type == "timeout"
    assert events[1].response.decision.value == "rejected"
    assert "超时" in events[1].response.comment
    print(f"  ✓ 超时流程: {events[0].event_type} → {events[1].event_type}")
    print(f"    自动决策: {events[1].response.decision.value}")
    print(f"    原因: {events[1].response.comment}")

    print("  PASS\n")


async def test_7_confirm():
    """Test 7: 确认中断（是/否）"""
    print("=" * 60)
    print("TEST 7: 确认中断")
    print("=" * 60)

    from agent.human_in_the_loop import HITLManager

    hitl = HITLManager()

    request_id = None

    async def simulate_confirm_flow():
        nonlocal request_id
        events = []
        async for event in hitl.confirm(
            question="是否继续执行修复操作？",
            timeout=10,
        ):
            events.append(event)
            if event.event_type == "interrupt":
                request_id = event.request.request_id
                asyncio.create_task(simulate_answer())
        return events

    async def simulate_answer():
        await asyncio.sleep(0.1)
        if request_id:
            hitl.answer_confirm(request_id, approved=True, comment="继续")

    events = await simulate_confirm_flow()

    assert len(events) == 2
    assert events[0].event_type == "interrupt"
    assert events[1].event_type == "approved"
    print(f"  ✓ 确认流程: {events[0].event_type} → {events[1].event_type}")

    print("  PASS\n")


async def test_8_auto_approve():
    """Test 8: 安全操作自动批准"""
    print("=" * 60)
    print("TEST 8: 安全操作自动批准")
    print("=" * 60)

    from agent.human_in_the_loop import HITLManager, HITLConfig

    config = HITLConfig(
        auto_approve_safe_ops=True,
        safe_tools=["read_file", "grep_search"],
        require_approval_for_severity=["HIGH", "CRITICAL"],
    )
    hitl = HITLManager(config=config)

    # 安全操作 → 自动批准
    events = []
    async for event in hitl.approval_interrupt(
        action="read_file",
        detail={"path": "/workspace/test.java"},
        severity="LOW",
    ):
        events.append(event)

    assert len(events) == 1
    assert events[0].event_type == "approved"
    assert "自动批准" in events[0].message
    print(f"  ✓ 安全操作自动批准: {events[0].event_type}")
    print(f"    {events[0].message}")

    # 非安全操作 → 需要审批
    assert hitl.should_interrupt_tool("write_file")
    assert not hitl.should_interrupt_tool("read_file")
    print(f"  ✓ 非安全操作仍需审批")

    print("  PASS\n")


async def test_9_hitl_persistence():
    """Test 9: HITL + TaskPersistence 集成"""
    print("=" * 60)
    print("TEST 9: HITL + TaskPersistence 集成")
    print("=" * 60)

    from agent.human_in_the_loop import HITLManager, HITLConfig, ApprovalRequest, ApprovalResponse
    from agent.task_persistence import TaskPersistence, TaskItem, TaskStatus

    hitl = HITLManager(HITLConfig(
        interrupt_before=["fixer"],
        interrupt_on_tools=["apply_patch"],
    ))

    tp = await TaskPersistence(backend="file", storage_dir="data/test_hitl_persistence")._ensure_storage()
    tp = TaskPersistence(backend="file", storage_dir="data/test_hitl_persistence")

    # 模拟：任务被阻断（等待审批）
    todos = [
        TaskItem(title="审查代码", agent="code-reviewer", status="completed"),
        TaskItem(title="分析Bug", agent="bug-analyzer", status="completed"),
        TaskItem(title="修复Bug", agent="code-fixer", status="blocked"),  # ← 等待审批
        TaskItem(title="验证修复", agent="test-validator", status="pending"),
    ]

    plan = await tp.write_todos(
        session_id="test_hitl_session",
        todos=todos,
        title="Bug修复 + HITL审批",
    )
    assert plan.blocked_tasks
    print(f"  ✓ 阻断任务: {[t.title for t in plan.blocked_tasks]}")
    print(f"    计划进度: {plan.progress:.0%}")

    # 模拟审批通过 → 更新任务状态
    await tp.update_todo(plan.plan_id, 2, status="in_progress")
    updated = await tp.update_todo(
        plan.plan_id, 2, status="completed",
        result={"patch": "添加空检查", "files_changed": 1},
    )
    print(f"  ✓ 审批后恢复执行")
    print(f"    新进度: {updated.progress:.0%}")

    # 清理
    import shutil
    shutil.rmtree("data/test_hitl_persistence", ignore_errors=True)

    print("  PASS\n")


async def test_10_end_to_end():
    """Test 10: 端到端 — LangGraph interrupt_before 模拟"""
    print("=" * 60)
    print("TEST 10: 端到端 — 双层中断完整流程")
    print("=" * 60)

    from agent.human_in_the_loop import (
        HITLManager, HITLConfig, create_hitl_manager,
        create_request_missing_info_tool, create_request_approval_tool,
    )

    hitl = create_hitl_manager(
        interrupt_before=["fixer", "validator"],
        interrupt_on_tools=["write_file", "apply_patch"],
    )

    print()
    print("  ┌──────────────────────────────────────────────┐")
    print("  │  双层中断工作流                                │")
    print("  ├──────────────────────────────────────────────┤")

    # Step 1: 数据补充（Layer 1）
    print("  │  Step 1: [Layer 1] Agent 缺少数据 → 请求人类  │")
    layer1_events = []
    request_id = None

    async def l1_collect():
        nonlocal request_id
        async for event in hitl.data_interrupt(
            tool_name="request_missing_info",
            question="Bug 在哪个 Java 版本和操作系统上复现？",
            timeout=10,
        ):
            layer1_events.append(event)
            if event.event_type == "interrupt":
                request_id = event.data_request.request_id
                asyncio.create_task(l1_respond())

    async def l1_respond():
        await asyncio.sleep(0.1)
        if request_id:
            hitl.submit_data_response(request_id, "Java 17.0.9, Windows 11")

    await l1_collect()
    print(f"  │     {layer1_events[0].event_type} → {layer1_events[1].event_type}")
    print(f"  │     回答: {layer1_events[1].data_response.answer}")

    # Step 2: 审批拦截（Layer 2）
    print("  │                                              │")
    print("  │  Step 2: [Layer 2] 执行 fixer → 审批拦截     │")
    assert hitl.should_interrupt_node("fixer")
    print(f"  │     interrupt_before fixer: ✓                │")

    # Step 3: 工具审批
    print("  │                                              │")
    print("  │  Step 3: [Layer 2] write_file → 审批请求     │")
    l2_events = []
    approval_id = None

    async def l2_approve():
        nonlocal approval_id
        async for event in hitl.approval_interrupt(
            action="apply_patch",
            detail={"patch": "修复补丁", "files": 1},
            severity="HIGH",
            timeout=10,
        ):
            l2_events.append(event)
            if event.event_type == "interrupt":
                approval_id = event.request.request_id
                asyncio.create_task(l2_decide())

    async def l2_decide():
        await asyncio.sleep(0.1)
        if approval_id:
            hitl.submit_approval(approval_id, "approved", "补丁审查通过")

    await l2_approve()
    print(f"  │     {l2_events[0].event_type} → {l2_events[1].event_type}")
    print(f"  │     决策: {l2_events[1].response.decision.value}")

    # Step 4: 确认
    print("  │                                              │")
    print("  │  Step 4: 确认最终执行                         │")
    confirm_events = []
    confirm_id = None

    async def do_confirm():
        nonlocal confirm_id
        async for event in hitl.confirm("是否提交修复到代码仓库？", timeout=10):
            confirm_events.append(event)
            if event.event_type == "interrupt":
                confirm_id = event.request.request_id
                asyncio.create_task(answer_confirm())

    async def answer_confirm():
        await asyncio.sleep(0.1)
        if confirm_id:
            hitl.answer_confirm(confirm_id, approved=True, comment="提交")

    await do_confirm()
    print(f"  │     {confirm_events[0].event_type} → {confirm_events[1].event_type}")

    # 最终状态
    print("  ├──────────────────────────────────────────────┤")
    summary = hitl.get_hitl_summary()
    print(f"  │  待审批: {summary['pending_approvals']}                                   │")
    print(f"  │  待数据: {summary['pending_data_requests']}                                   │")
    print(f"  │  日志数: {len(hitl.get_approval_log())}                                   │")
    print("  └──────────────────────────────────────────────┘")

    assert summary["pending_approvals"] == 0
    assert summary["pending_data_requests"] == 0
    assert len(hitl.get_approval_log()) == 1
    print(f"\n  ✓ 端到端双层中断流程验证通过")

    print("  PASS\n")


async def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   Human-in-the-Loop — Demo 测试                  ║")
    print("║   双层中断: 数据补充 + 审批拦截                    ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    tests = [
        ("HITLConfig 配置", test_1_hitl_config),
        ("Layer1 数据中断", test_2_layer1_data_interrupt),
        ("Layer1 完整流程", test_3_layer1_full_flow),
        ("Layer2 审批中断", test_4_layer2_approval_interrupt),
        ("Layer2 审批流程", test_5_layer2_approval_flow),
        ("审批超时拒绝", test_6_approval_timeout),
        ("确认中断", test_7_confirm),
        ("安全操作自动批准", test_8_auto_approve),
        ("HITL+Persistence 集成", test_9_hitl_persistence),
        ("端到端双层中断", test_10_end_to_end),
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
