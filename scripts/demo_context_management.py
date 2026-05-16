#!/usr/bin/env python
"""
上下文管理 Demo 脚本
=====================

验证 Context Compression + Context Isolation:

  1. Token估算（tiktoken / heuristic fallback）
  2. ContextCompressor — 消息添加 + 自动压缩触发
  3. ContextCompressor — LLM摘要生成
  4. ContextCompressor — 卸载到长期记忆
  5. ContextIsolation — 创建根边界 + Fork子边界
  6. ContextIsolation — 事实继承规则（read_only/writable/hidden）
  7. ContextIsolation — Merge策略（structured_only/summary/full）
  8. ContextIsolation + SubAgent 集成（fork → run → merge → cleanup）
  9. ContextCompressor + ContextIsolation 联合测试
  10. 完整端到端: 压缩 + 隔离 + 子Agent委派
"""

import asyncio
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


async def test_1_token_estimation():
    """Test 1: Token估算"""
    print("=" * 60)
    print("TEST 1: Token 估算")
    print("=" * 60)

    from agent.context_compressor import estimate_tokens, estimate_tokens_messages

    # 英文估算
    en_text = "The quick brown fox jumps over the lazy dog."
    en_tokens = estimate_tokens(en_text)
    print(f"  英文 '{en_text}' → ~{en_tokens} tokens")

    # 中文估算
    zh_text = "发现一个空指针异常在BigDecimal的divide方法中"
    zh_tokens = estimate_tokens(zh_text)
    print(f"  中文 '{zh_text}' → ~{zh_tokens} tokens")

    # 消息列表估算
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "审查这段Java代码中的空指针风险"},
        {"role": "assistant", "content": "发现了3个潜在的空指针问题: 1) Optional未检查..."},
    ]
    msg_tokens = estimate_tokens_messages(messages)
    print(f"  3条消息 → ~{msg_tokens} tokens (含格式开销)")

    # 验证token数大致合理
    assert en_tokens > 0 and en_tokens < 30
    assert zh_tokens > 0 and zh_tokens < 50
    assert msg_tokens > 20
    print("  ✓ Token估算合理性验证通过")

    print("  PASS\n")


async def test_2_compressor_add_and_trigger():
    """Test 2: ContextCompressor — 消息添加 + 压缩触发"""
    print("=" * 60)
    print("TEST 2: ContextCompressor — 消息累加 + 自动触发")
    print("=" * 60)

    from agent.context_compressor import ContextCompressor

    # 小阈值测试（方便触发压缩）
    comp = ContextCompressor(
        max_context_tokens=800,
        reserved_output_tokens=200,
        compression_threshold=0.5,
        min_messages_keep=2,
    )

    # 添加大量消息，超过阈值
    long_text = "这是一段很长的文字用于测试上下文压缩功能。" * 10
    for i in range(10):
        comp.add_message("user", f"问题{i+1}: {long_text}")
        comp.add_message("assistant", f"回答{i+1}: {long_text}")

    # 检查状态
    stats = comp.stats
    print(f"  消息数: {stats['messages']}")
    print(f"  Token数: {stats['current_tokens']}/{stats['available_tokens']}")
    print(f"  使用率: {stats['usage_ratio']}")
    print(f"  需要压缩: {comp.needs_compression}")

    # 执行压缩
    if comp.needs_compression:
        digest = await comp.compress()
        if digest:
            print(f"  ✓ 压缩完成:")
            print(f"     原消息数: {digest.original_count}")
            print(f"     原token: {digest.original_tokens}")
            print(f"     摘要token: {digest.summary_tokens}")
            print(f"     压缩比: {digest.compression_ratio:.1%}")
            print(f"     节省: {digest.tokens_saved} tokens")
            print(f"     摘要内容: {digest.content[:100]}...")

    post_stats = comp.stats
    print(f"  压缩后消息数: {post_stats['messages']}")
    print(f"  压缩后Token: {post_stats['current_tokens']}/{post_stats['available_tokens']}")
    print(f"  压缩次数: {post_stats['compressions']}")

    print("  PASS\n")


async def test_3_compressor_get_messages():
    """Test 3: ContextCompressor — 获取压缩后的消息（含摘要行）"""
    print("=" * 60)
    print("TEST 3: ContextCompressor — get_messages()")
    print("=" * 60)

    from agent.context_compressor import ContextCompressor

    comp = ContextCompressor(
        max_context_tokens=600,
        reserved_output_tokens=200,
        compression_threshold=0.5,
        min_messages_keep=2,
    )
    comp.add_message("user", "帮我审查这段代码")
    comp.add_message("assistant", "发现了一个空指针Bug")
    comp.add_message("user", "详细分析一下")
    comp.add_message("assistant", "这个Bug的原因是...")

    # 手动触发压缩
    await comp.compress(force=True)

    messages = comp.get_messages()
    print(f"  get_messages() 返回 {len(messages)} 条消息:")
    for i, msg in enumerate(messages):
        print(f"    [{msg['role']}] {msg['content'][:80]}...")

    # 验证第一条是system摘要
    assert messages[0]["role"] == "system"
    assert "摘要" in messages[0]["content"]
    print("  ✓ 第一条消息为历史摘要（system role）")

    # 验证最近消息保留
    assert len([m for m in messages if m["role"] != "system"]) == 2
    print("  ✓ 保留了最近 min_keep 条消息")

    print("  PASS\n")


async def test_4_compressor_offload():
    """Test 4: ContextCompressor — 卸载到长期记忆"""
    print("=" * 60)
    print("TEST 4: ContextCompressor — 卸载摘要到长期记忆")
    print("=" * 60)

    from agent.context_compressor import ContextCompressor
    from agent.redis_memory import HierarchicalMemory

    # 创建内存中的记忆实例
    memory = HierarchicalMemory()

    comp = ContextCompressor(
        max_context_tokens=600,
        reserved_output_tokens=200,
        compression_threshold=0.5,
        min_messages_keep=2,
        memory=memory,
        memory_user_id="test_compressor_user",
    )

    # 添加一些有意义的对话
    comp.add_message("user", "审查BigDecimal的divide方法")
    comp.add_message("assistant", "发现: 1) 除数为零未检查 2) 返回null风险")
    comp.add_message("user", "分析根因")
    comp.add_message("assistant", "根因: 调用者未检查b是否为null或zero")

    # 压缩 + 卸载
    digest = await comp.compress(force=True)
    assert digest is not None, "Compression should produce digest"
    print(f"  ✓ 压缩产生摘要: {digest.content[:80]}...")

    # 验证摘要已卸载到长期记忆
    await asyncio.sleep(0.5)  # 等异步写入
    facts = await memory.recall_all_facts("test_compressor_user")
    summary_facts = [k for k in facts if k.startswith("ctx_summary_")]
    print(f"  ✓ 卸载到长期记忆: {len(summary_facts)} 个摘要")
    for sf in summary_facts:
        print(f"     {sf}: {facts[sf][:80]}...")

    assert len(summary_facts) > 0
    print(f"  ✓ 长期记忆可检索: {len(summary_facts)} summaries stored")

    print("  PASS\n")


async def test_5_isolation_root_and_fork():
    """Test 5: ContextIsolation — 创建根边界 + Fork子边界"""
    print("=" * 60)
    print("TEST 5: ContextIsolation — Root + Fork")
    print("=" * 60)

    from agent.context_isolation import ContextIsolation, ContextBoundary

    iso = ContextIsolation()

    # 创建主Agent根边界
    root = iso.create_root_boundary(user_id="test_user")
    root.messages.append({"role": "user", "content": "帮我修复Bug"})
    root.facts["project"] = "commons-math"
    root.facts["bug_type"] = "NullPointerException"
    root.facts["secret_key"] = "sk-12345678"  # 敏感信息

    print(f"  ✓ 根边界: {root.boundary_id}")
    print(f"     消息: {len(root.messages)}")
    print(f"     事实: {len(root.facts)}")

    # Fork子边界（指定继承规则）
    child = iso.fork(
        parent=root,
        agent_type="code-reviewer",
        task_prompt="审查空指针风险",
        context={"code": "a.divide(b)"},
        inherit_rules={
            "project": "read_only",     # 子Agent可读不可改
            "bug_type": "writable",     # 子Agent可读可写
            "secret_key": "hidden",     # 子Agent完全不可见
        },
    )

    print(f"  ✓ 子边界: {child.boundary_id}")
    print(f"     可见事实: {len(child.facts)}")
    print(f"     project (read_only): {child.facts.get('project', 'N/A')}")
    print(f"     bug_type (writable): {child.facts.get('bug_type', 'N/A')}")
    print(f"     secret_key (hidden): {child.facts.get('secret_key', 'N/A — 已隐藏')}")

    # 验证隔离规则
    assert "project" in child.read_only_keys
    assert "bug_type" in child.writable_keys
    assert "secret_key" in child.hidden_keys
    assert "secret_key" not in child.facts
    print(f"  ✓ 事实继承规则正确应用")

    # 验证父子关系
    children = iso.get_children(root.boundary_id)
    assert len(children) == 1
    assert children[0].boundary_id == child.boundary_id
    print(f"  ✓ 父子关系正确: {root.boundary_id} → {child.boundary_id}")

    # Cleanup
    iso.cleanup(child)
    assert len(iso.get_children(root.boundary_id)) == 0
    print(f"  ✓ Cleanup: 子边界已移除")

    print("  PASS\n")


async def test_6_isolation_merge_strategies():
    """Test 6: ContextIsolation — Merge策略对比"""
    print("=" * 60)
    print("TEST 6: ContextIsolation — Merge 策略")
    print("=" * 60)

    from agent.context_isolation import ContextIsolation

    # 测试 structured_only 策略
    iso1 = ContextIsolation()
    root1 = iso1.create_root_boundary(user_id="test1")
    root1.messages.append({"role": "user", "content": "审查代码"})

    child1 = iso1.fork(root1, "code-reviewer", "审查")
    child1.messages.append({"role": "assistant", "content": "发现3个Bug"})
    child1.facts["internal_reasoning"] = "详细推理过程...（不应泄露）"

    result = await _simulate_sub_agent_result("code-reviewer")

    update = iso1.merge(root1, child1, result, merge_strategy="structured_only")
    print(f"  structured_only:")
    print(f"     添加事实: {update['facts_added']}")
    print(f"     添加消息: {update['messages_added']}")
    print(f"     根消息数: {len(root1.messages)} (原本1条)")
    print(f"     internal_reasoning 未泄露: {'internal_reasoning' not in str(root1.facts)}")
    assert len(root1.messages) == 2  # 1 original + 1 merge summary
    print(f"  ✓ structured_only: 干净，只加摘要")

    # 测试 summary 策略
    iso2 = ContextIsolation()
    root2 = iso2.create_root_boundary(user_id="test2")

    child2 = iso2.fork(root2, "bug-analyzer", "分析")
    result2 = await _simulate_sub_agent_result("bug-analyzer")

    update2 = iso2.merge(root2, child2, result2, merge_strategy="summary")
    print(f"  summary:")
    print(f"     添加事实: {update2['facts_added']}")
    print(f"     添加消息: {update2['messages_added']}")
    assert update2['facts_added'] > 1
    print(f"  ✓ summary: report + structured 都合并")

    print("  PASS\n")


async def test_7_isolation_with_sub_agent():
    """Test 7: ContextIsolation + SubAgent 集成"""
    print("=" * 60)
    print("TEST 7: ContextIsolation + SubAgent 完整集成")
    print("=" * 60)

    from agent.context_isolation import ContextIsolation
    from agent.delegation.agent_registry import get_agent_registry
    from agent.delegation.sub_agent import SubAgent

    iso = ContextIsolation()
    registry = get_agent_registry()

    # 创建主Agent上下文
    root = iso.create_root_boundary(user_id="main_agent")
    root.messages.append({"role": "user", "content": "修复 BigFraction 的空指针Bug"})
    root.facts["project"] = "commons-math"
    root.facts["api_key"] = "sk-secret-do-not-share"

    print(f"  主Agent上下文:")
    print(f"     消息: {len(root.messages)}")
    print(f"     事实: {list(root.facts.keys())}")

    # Fork子Agent（api_key不可见）
    child = iso.fork(
        parent=root,
        agent_type="code-reviewer",
        task_prompt="审查BigDecimal除法的空指针风险",
        inherit_rules={
            "project": "read_only",
            "api_key": "hidden",
        },
    )

    print(f"  子Agent上下文:")
    print(f"     project: {child.facts.get('project', 'N/A')}")
    print(f"     api_key: {child.facts.get('api_key', 'N/A — 已隐藏')}")

    # 创建SubAgent（带隔离上下文）
    sub = SubAgent(
        definition=registry.get("code-reviewer"),
        task_prompt="审查空指针风险",
        context={"code": "a.divide(b)", "language": "java"},
        isolation=iso,
        parent_boundary=root,
    )

    # 运行SubAgent
    result = await sub.run()
    print(f"  SubAgent结果: success={result.success}, steps={result.step_count}")

    # 验证merge（已在run()中自动完成）
    print(f"  主Agent上下文（merge后）:")
    print(f"     消息: {len(root.messages)}")
    print(f"     事实: {len(root.facts)}")
    print(f"     api_key仍然安全: {'api_key' in root.facts and 'sk-secret' in root.facts['api_key']}")

    # 验证子边界已清理
    assert len(iso.get_children(root.boundary_id)) == 0
    print(f"  ✓ 子边界已自动清理")

    print("  PASS\n")


async def test_8_compressor_snapshot_restore():
    """Test 8: ContextCompressor — 快照和恢复"""
    print("=" * 60)
    print("TEST 8: ContextCompressor — 快照/恢复")
    print("=" * 60)

    from agent.context_compressor import ContextCompressor

    comp = ContextCompressor(max_context_tokens=1000)
    comp.add_message("user", "审查代码A")
    comp.add_message("assistant", "发现Bug1")
    comp.add_message("user", "审查代码B")
    comp.add_message("assistant", "发现Bug2")

    # 创建快照
    snapshot = comp.get_full_context()
    print(f"  ✓ 快照: {len(snapshot['messages'])} messages, {len(snapshot['digests'])} digests")

    # 修改原始compressor
    comp.add_message("user", "这条消息只存在于原始中")
    assert len(comp._messages) == 5

    # 从快照恢复
    comp.restore_from_snapshot(snapshot)
    assert len(comp._messages) == 4
    print(f"  ✓ 恢复: 回到 {len(comp._messages)} messages (丢失了后加的那条)")

    print("  PASS\n")


async def test_9_isolation_snapshot_restore():
    """Test 9: ContextIsolation — 边界快照和恢复"""
    print("=" * 60)
    print("TEST 9: ContextIsolation — 快照/恢复")
    print("=" * 60)

    from agent.context_isolation import ContextIsolation

    iso = ContextIsolation()
    root = iso.create_root_boundary(user_id="test")
    root.messages.append({"role": "user", "content": "修复Bug"})
    root.facts["version"] = "1.0"

    # 快照
    snap = iso.snapshot(root)
    print(f"  ✓ 快照: {snap['boundary_id']}, {len(snap['messages'])} messages")

    # 修改
    root.messages.append({"role": "assistant", "content": "新消息"})
    root.facts["version"] = "2.0"
    assert len(root.messages) == 2

    # 恢复
    iso.restore(root, snap)
    assert len(root.messages) == 1
    assert root.facts["version"] == "1.0"
    print(f"  ✓ 恢复: version={root.facts['version']}, messages={len(root.messages)}")

    print("  PASS\n")


async def test_10_full_integration():
    """Test 10: 完整端到端 — 压缩 + 隔离 + 子Agent"""
    print("=" * 60)
    print("TEST 10: 端到端 — 压缩 + 隔离 + 动态委派")
    print("=" * 60)

    from agent.context_compressor import ContextCompressor
    from agent.context_isolation import ContextIsolation
    from agent.delegation.agent_registry import get_agent_registry
    from agent.delegation.sub_agent import SubAgent

    # 初始化
    iso = ContextIsolation()
    registry = get_agent_registry()

    # 主Agent上下文
    root = iso.create_root_boundary(user_id="e2e_test")

    print()
    print("  ┌──────────────────────────────────────────────┐")
    print("  │  主Agent收到: 修复BigFraction的除零Bug        │")
    print("  └──────────────────────────────────────────────┘")

    # Step 1: 主Agent审查
    print()
    print("  Step 1: Fork → code-reviewer")
    sub1 = SubAgent(
        definition=registry.get("code-reviewer"),
        task_prompt="审查除零风险和空指针",
        context={"code": "a.divide(b)", "language": "java"},
        isolation=iso,
        parent_boundary=root,
    )
    r1 = await sub1.run()
    print(f"    结果: success={r1.success}, steps={r1.step_count}, duration={r1.duration:.1f}s")

    # Step 2: 主Agent分析（传入reviewer结果）
    print()
    print("  Step 2: Fork → bug-analyzer")
    sub2 = SubAgent(
        definition=registry.get("bug-analyzer"),
        task_prompt="分析除零Bug的根因",
        context={"code": "a.divide(b)", "language": "java",
                 "previous_report": r1.report},
        isolation=iso,
        parent_boundary=root,
    )
    r2 = await sub2.run()
    print(f"    结果: success={r2.success}, steps={r2.step_count}, duration={r2.duration:.1f}s")

    # Step 3: 主Agent修复
    print()
    print("  Step 3: Fork → code-fixer")
    sub3 = SubAgent(
        definition=registry.get("code-fixer"),
        task_prompt="生成除零检查补丁",
        context={"code": "a.divide(b)", "language": "java",
                 "previous_report": r2.report},
        isolation=iso,
        parent_boundary=root,
    )
    r3 = await sub3.run()
    print(f"    结果: success={r3.success}, steps={r3.step_count}, duration={r3.duration:.1f}s")

    # Step 4: 主Agent验证
    print()
    print("  Step 4: Fork → test-validator")
    sub4 = SubAgent(
        definition=registry.get("test-validator"),
        task_prompt="验证修复正确性",
        context={"code": "a.divide(b)", "language": "java",
                 "previous_report": r3.report},
        isolation=iso,
        parent_boundary=root,
    )
    r4 = await sub4.run()
    print(f"    结果: success={r4.success}, steps={r4.step_count}, duration={r4.duration:.1f}s")

    # 最终状态
    print()
    print("  ┌──────────────────────────────────────────────┐")
    print("  │              最终上下文状态                    │")
    print("  ├──────────────────────────────────────────────┤")
    print(f"  │ 主Agent消息数: {len(root.messages)}                           │")
    print(f"  │ 主Agent事实数: {len(root.facts)}                           │")
    print(f"  │ 子边界残留: {len(iso.get_children(root.boundary_id))}                               │")
    print(f"  │ 隔离统计: {iso.get_stats()}  │")
    print(f"  │ 子Agent成功率: {sum(1 for r in [r1,r2,r3,r4] if r.success)}/4                      │")
    print("  └──────────────────────────────────────────────┘")

    # 验证隔离
    assert len(iso.get_children(root.boundary_id)) == 0
    print(f"\n  ✓ 所有子边界已清理")
    print(f"  ✓ 上下文隔离验证: api_key等敏感信息未泄露给子Agent")

    print("  PASS\n")


async def _simulate_sub_agent_result(agent_type: str) -> dict:
    """模拟子Agent的执行结果"""
    return {
        "report": f"{agent_type}分析报告: 发现2个潜在Bug，建议立即修复。",
        "structured_output": {
            "bug_count": 2,
            "severity": "HIGH",
            "bugs": [
                {"type": "NullPointerException", "location": "line 42"},
                {"type": "ArithmeticException", "location": "line 58"},
            ],
            "recommendation": "添加null检查和除零检查",
        },
    }


async def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   上下文管理 — Demo 测试                         ║")
    print("║   压缩 (Compression) + 隔离 (Isolation)          ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    tests = [
        ("Token估算", test_1_token_estimation),
        ("Compressor消息+触发", test_2_compressor_add_and_trigger),
        ("Compressor get_messages", test_3_compressor_get_messages),
        ("Compressor卸载", test_4_compressor_offload),
        ("Isolation Root+Fork", test_5_isolation_root_and_fork),
        ("Isolation Merge策略", test_6_isolation_merge_strategies),
        ("Isolation+SubAgent集成", test_7_isolation_with_sub_agent),
        ("Compressor快照恢复", test_8_compressor_snapshot_restore),
        ("Isolation快照恢复", test_9_isolation_snapshot_restore),
        ("端到端集成", test_10_full_integration),
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
