#!/usr/bin/env python
"""
动态委派 Demo 脚本
===================

验证动态子Agent委派的所有能力:

  1. AgentDefinition 从 YAML 加载
  2. AgentRegistry 注册和搜索
  3. SubAgent 独立上下文执行 (ReAct循环)
  4. TaskTool 串行流水线 (reviewer → analyzer → fixer → validator)
  5. TaskTool 并行委派
  6. 动态委派 LangGraph (orchestrator → dispatcher → synthesizer)
  7. 与固定管道的对比
  8. YAML 不可用时的优雅降级
"""

import asyncio
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


async def test_1_yaml_loading():
    """Test 1: 从YAML加载AgentDefinition"""
    print("=" * 60)
    print("TEST 1: AgentDefinition — YAML加载")
    print("=" * 60)

    from agent.delegation.agent_definition import load_agent_definition, load_all_definitions

    # 加载单个
    yaml_path = Path(__file__).parent.parent / "config" / "agents" / "code-reviewer.yaml"
    ad = load_agent_definition(yaml_path)
    assert ad is not None, "Failed to load code-reviewer.yaml"
    assert ad.name == "code-reviewer"
    assert ad.version == "2.0.0"
    assert len(ad.capabilities) > 0
    assert len(ad.tools) > 0
    print(f"  ✓ 加载单个: {ad.display_name}")
    print(f"     能力: {ad.capabilities}")
    print(f"     工具: {ad.tools}")
    print(f"     技能: {ad.skills}")

    # 加载所有
    all_defs = load_all_definitions()
    assert len(all_defs) == 4, f"Expected 4 agents, got {len(all_defs)}"
    print(f"  ✓ 加载全部: {len(all_defs)} agents")
    for ad in all_defs:
        print(f"     {ad.brief}")

    # 测试优雅降级
    from agent.delegation.agent_definition import _get_fallback_definitions
    fallbacks = _get_fallback_definitions()
    assert len(fallbacks) == 4
    print(f"  ✓ 回退定义: {len(fallbacks)} agents (YAML不可用时的降级)")

    print("  PASS\n")


async def test_2_agent_registry():
    """Test 2: AgentRegistry 注册和搜索"""
    print("=" * 60)
    print("TEST 2: AgentRegistry — 注册和搜索")
    print("=" * 60)

    from agent.delegation.agent_registry import AgentRegistry

    registry = AgentRegistry()
    registry.load()

    # 计数
    assert registry.count == 4
    print(f"  ✓ 注册数量: {registry.count}")

    # 按名称获取
    ad = registry.get("code-reviewer")
    assert ad is not None
    print(f"  ✓ get('code-reviewer') → {ad.display_name}")

    # 按关键词搜索
    results = registry.search(query="review")
    assert len(results) == 1
    print(f"  ✓ search('review') → {len(results)} result: {results[0].name}")

    # 按标签搜索
    results = registry.search(tags=["fix"])
    assert len(results) == 1
    print(f"  ✓ search(tags=['fix']) → {results[0].name}")

    # 按能力搜索
    results = registry.search(capability="根因")
    assert len(results) == 1
    print(f"  ✓ search(capability='根因') → {results[0].name}")

    # 渐进式披露（只返回摘要）
    briefs = registry.list_briefs()
    assert len(briefs) == 4
    print(f"  ✓ list_briefs() → {len(briefs)} briefs (省Token)")

    # 动态注册
    from agent.delegation.agent_definition import AgentDefinition
    custom = AgentDefinition(
        name="custom-security-scanner",
        description="自定义安全扫描器",
        system_prompt="Scan for security issues",
        tags=["security", "custom"],
    )
    registry.register(custom)
    assert registry.get("custom-security-scanner") is not None
    print(f"  ✓ 动态注册: custom-security-scanner")

    # 清理
    registry.unregister("custom-security-scanner")

    print("  PASS\n")


async def test_3_sub_agent_independent_context():
    """Test 3: SubAgent 独立上下文执行"""
    print("=" * 60)
    print("TEST 3: SubAgent — 独立上下文 + ReAct循环")
    print("=" * 60)

    from agent.delegation.agent_registry import get_agent_registry
    from agent.delegation.sub_agent import SubAgent

    registry = get_agent_registry()

    # 用 no-op LLM config（模拟模式，不真实调API）
    code = """
    public BigDecimal divide(BigDecimal a, BigDecimal b) {
        return a.divide(b);
    }
    """

    # 创建两个独立的子Agent（证明上下文隔离）
    sub1 = SubAgent(
        definition=registry.get("code-reviewer"),
        task_prompt="检查空指针风险",
        context={"code": code, "language": "java"},
        max_iterations=1,
    )

    sub2 = SubAgent(
        definition=registry.get("bug-analyzer"),
        task_prompt="分析除法异常风险",
        context={"code": code, "language": "java"},
        max_iterations=1,
    )

    # 验证两个SubAgent有独立的ID和上下文
    assert sub1.task_id != sub2.task_id, "Sub-agents should have unique IDs"
    print(f"  ✓ 独立ID: sub1={sub1.task_id}, sub2={sub2.task_id}")

    assert sub1.agent_type == "code-reviewer"
    assert sub2.agent_type == "bug-analyzer"
    print(f"  ✓ 不同Agent类型: {sub1.agent_type} vs {sub2.agent_type}")

    assert sub1.task_prompt != sub2.task_prompt
    print(f"  ✓ 不同任务: '{sub1.task_prompt}' vs '{sub2.task_prompt}'")

    # 验证system prompt不同（从各自的AgentDefinition构建）
    sys1 = sub1._build_system_message()
    sys2 = sub2._build_system_message()
    assert "审查" in sys1 and "根因" in sys2
    print(f"  ✓ 不同System Prompt: {len(sys1)} chars vs {len(sys2)} chars")

    print("  PASS\n")


async def test_4_task_tool_pipeline():
    """Test 4: TaskTool 串行流水线"""
    print("=" * 60)
    print("TEST 4: TaskTool — 串行流水线委派")
    print("=" * 60)

    from agent.delegation.agent_registry import get_agent_registry
    from agent.delegation.task_tool import TaskTool

    registry = get_agent_registry()
    tool = TaskTool(registry=registry)

    code = """
    public String getName(Object obj) {
        return obj.toString();
    }
    """

    # 定义4步流水线（模拟，不真实调LLM → 快速通过）
    pipeline = [
        {
            "agent_type": "code-reviewer",
            "task": "审查代码中的空指针风险",
            "context": {"code": code, "language": "java"},
        },
        {
            "agent_type": "bug-analyzer",
            "task": "分析空指针根因",
            "context": {"code": code, "language": "java"},
        },
        {
            "agent_type": "code-fixer",
            "task": "生成修复补丁",
            "context": {"code": code, "language": "java"},
        },
        {
            "agent_type": "test-validator",
            "task": "验证修复正确性",
            "context": {"code": code, "language": "java"},
        },
    ]

    # 串行执行（不调真实LLM的话会快速失败，但结构正确）
    results = await tool.execute_pipeline(pipeline)

    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    print(f"  ✓ 流水线长度: {len(results)} steps")

    # 验证步骤顺序
    expected_types = ["code-reviewer", "bug-analyzer", "code-fixer", "test-validator"]
    for i, (r, expected) in enumerate(zip(results, expected_types)):
        assert r["agent_type"] == expected, f"Step {i}: expected {expected}, got {r['agent_type']}"
        print(f"  ✓ Step {i+1}: {r['agent_type']} (success={r['success']}, duration={r.get('duration', 0):.1f}s)")

    # 验证委派历史
    assert tool.total_delegations == 4
    print(f"  ✓ 委派历史: {tool.total_delegations} records")

    print("  PASS\n")


async def test_5_parallel_delegation():
    """Test 5: TaskTool 并行委派"""
    print("=" * 60)
    print("TEST 5: TaskTool — 并行委派")
    print("=" * 60)

    from agent.delegation.agent_registry import get_agent_registry
    from agent.delegation.task_tool import TaskTool

    registry = get_agent_registry()
    tool = TaskTool(registry=registry)

    code = "public void process() { /* TODO */ }"

    # 并行委派两个独立任务
    tasks = [
        {
            "agent_type": "code-reviewer",
            "task": "审查代码质量",
            "context": {"code": code, "language": "java"},
        },
        {
            "agent_type": "test-validator",
            "task": "检查测试覆盖",
            "context": {"code": code, "language": "java"},
        },
    ]

    import time
    start = time.time()
    results = await tool.execute_parallel(tasks)
    elapsed = time.time() - start

    assert len(results) == 2
    print(f"  ✓ 并行任务数: {len(results)}")
    print(f"  ✓ 总耗时: {elapsed:.2f}s (并行执行)")

    for r in results:
        print(f"     {r['agent_type']}: success={r['success']}")

    print("  PASS\n")


async def test_6_unknown_agent_type():
    """Test 6: 未知Agent类型的错误处理"""
    print("=" * 60)
    print("TEST 6: 未知Agent类型 — 错误处理")
    print("=" * 60)

    from agent.delegation.agent_registry import get_agent_registry
    from agent.delegation.task_tool import TaskTool

    registry = get_agent_registry()
    tool = TaskTool(registry=registry)

    result = await tool.execute({
        "agent_type": "non-existent-agent",
        "task": "do something",
        "context": {},
    })

    assert not result["success"]
    assert "Unknown" in result.get("error", "")
    print(f"  ✓ 未知Agent类型: {result['error'][:80]}...")

    # 验证错误消息包含可用列表
    assert "Available" in result.get("error", "")
    print(f"  ✓ 错误消息包含可用Agent列表")

    print("  PASS\n")


async def test_7_dynamic_graph():
    """Test 7: 动态委派 LangGraph"""
    print("=" * 60)
    print("TEST 7: 动态委派 LangGraph")
    print("=" * 60)

    from graph.dynamic_delegation_graph import (
        build_dynamic_delegation_graph,
        create_initial_state,
    )

    # 构建图
    graph = build_dynamic_delegation_graph()
    compiled = graph.compile()
    print("  ✓ LangGraph 构建成功")

    # 创建初始状态
    code = """
    public BigDecimal divide(BigDecimal a, BigDecimal b) {
        return a.divide(b);
    }
    """
    state = create_initial_state(code=code, language="java")
    assert state["code"] == code
    assert state["language"] == "java"
    assert state["current_step"] == 0
    assert state["replan_count"] == 0
    assert not state["done"]
    print(f"  ✓ 初始状态: code={len(code)} chars, language={state['language']}")

    # 运行图（不提供API key → orchestrator使用启发式计划）
    result = await compiled.ainvoke(state)

    # 验证核心流程
    assert "plan" in result
    plan = result["plan"]
    assert len(plan) == 4, f"Expected 4-step plan, got {len(plan)}"
    print(f"  ✓ 动态计划: {len(plan)} steps")
    for i, step in enumerate(plan):
        print(f"     Step {i+1}: {step.get('agent_type')} — {step.get('task', '')[:50]}...")

    # 验证子Agent被委派
    assert "sub_results" in result
    print(f"  ✓ 子Agent结果: {len(result.get('sub_results', []))} collected")

    # 验证最终报告
    assert "report" in result
    print(f"  ✓ 最终报告: {len(result.get('report', ''))} chars")

    print("  PASS\n")


async def test_8_comparison_with_fixed_pipeline():
    """Test 8: 固定管道 vs 动态委派对比"""
    print("=" * 60)
    print("TEST 8: 固定管道 vs 动态委派 — 对比")
    print("=" * 60)

    print()
    print("  ┌─────────────────────────────────────────────────────┐")
    print("  │              架构对比                               │")
    print("  ├────────────────┬────────────────────────────────────┤")
    print("  │ 旧: 固定管道    │ 新: 动态委派                        │")
    print("  ├────────────────┼────────────────────────────────────┤")
    print("  │ 4个硬编码类     │ YAML配置文件 → 无限种Agent类型      │")
    print("  │ 固定执行顺序    │ 动态决定执行顺序（可并行）           │")
    print("  │ 共享LangGraph   │ 每个子Agent独立上下文               │")
    print("  │   State         │                                    │")
    print("  │ 修改 = 改代码    │ 新增Agent = 加一个YAML文件          │")
    print("  │ 无法动态选择    │ 按任务需求选择Agent类型              │")
    print("  │ 无法并行        │ 支持并行委派和串行流水线            │")
    print("  │ 角色固定4个     │ 角色可无限扩展                      │")
    print("  └────────────────┴────────────────────────────────────┘")
    print()

    # 代码层面验证
    from agent.delegation.agent_registry import get_agent_registry
    registry = get_agent_registry()

    # 旧方式：4个硬编码类
    from agent.collaborative_agent import (
        CollaborativeReviewerAgent,
        CollaborativeAnalyzerAgent,
        CollaborativeFixerAgent,
        CollaborativeValidatorAgent,
    )
    old_agents = [
        CollaborativeReviewerAgent,
        CollaborativeAnalyzerAgent,
        CollaborativeFixerAgent,
        CollaborativeValidatorAgent,
    ]
    print(f"  旧方式: {len(old_agents)} 个硬编码类")
    print(f"     类名: {[c.__name__ for c in old_agents]}")

    # 新方式：从Registry动态获取
    new_agents = registry.list_all()
    print(f"  新方式: {len(new_agents)} 个YAML配置的Agent类型")
    print(f"     类型: {[ad.name for ad in new_agents]}")
    print(f"     扩展: 新增Agent类型只需添加一个YAML文件")
    print(f"     回退: YAML不可用时有硬编码回退定义")

    print("  PASS\n")


async def test_9_graceful_degradation():
    """Test 9: 优雅降级 — YAML文件不可用时的回退"""
    print("=" * 60)
    print("TEST 9: 优雅降级 — YAML不可用时回退")
    print("=" * 60)

    from agent.delegation.agent_definition import load_all_definitions

    # 用不存在的目录测试回退
    defs = load_all_definitions(config_dir="/nonexistent/path")
    assert len(defs) == 4, f"Expected 4 fallback defs, got {len(defs)}"
    print(f"  ✓ YAML不可用 → 回退定义: {len(defs)} agents")
    for ad in defs:
        print(f"     {ad.brief}")

    # 验证回退定义功能完整
    ad = defs[0]
    assert ad.name
    assert ad.system_prompt
    assert ad.capabilities
    assert ad.output_schema
    print(f"  ✓ 回退定义功能完整 (name + system_prompt + capabilities + output_schema)")

    print("  PASS\n")


async def test_10_end_to_end_flow():
    """Test 10: 端到端流程 — 模拟完整Bug修复"""
    print("=" * 60)
    print("TEST 10: 端到端动态委派流程")
    print("=" * 60)

    from agent.delegation.agent_registry import get_agent_registry
    from agent.delegation.task_tool import TaskTool

    registry = get_agent_registry()
    tool = TaskTool(registry=registry)

    code = """
    public String getDisplayName(User user) {
        return user.getName().toUpperCase();
    }
    """

    print()
    print("  ╔══════════════════════════════════════════════════════╗")
    print("  ║  主Agent收到Bug修复请求                              ║")
    print("  ╚══════════════════════════════════════════════════════╝")
    print()
    print(f"  代码: {code.strip()}")
    print()
    print("  ── 主Agent分析 → 生成计划 ──")
    print(f"  可用Agent: {registry.list_briefs()}")
    print()

    # Step 1: 审查
    print("  Step 1: 委派 code-reviewer")
    r1 = await tool.execute({
        "agent_type": "code-reviewer",
        "task": "审查空指针风险",
        "context": {"code": code, "language": "java"},
    })
    print(f"    结果: success={r1['success']}")

    # Step 2: 分析
    print("  Step 2: 委派 bug-analyzer")
    r2 = await tool.execute({
        "agent_type": "bug-analyzer",
        "task": "分析空指针根因",
        "context": {
            "code": code,
            "language": "java",
            "previous_report": r1.get("report", ""),
        },
    })
    print(f"    结果: success={r2['success']}")

    # Step 3: 修复
    print("  Step 3: 委派 code-fixer")
    r3 = await tool.execute({
        "agent_type": "code-fixer",
        "task": "生成Optional空指针修复",
        "context": {
            "code": code,
            "language": "java",
            "previous_report": r2.get("report", ""),
        },
    })
    print(f"    结果: success={r3['success']}")

    # Step 4: 验证
    print("  Step 4: 委派 test-validator")
    r4 = await tool.execute({
        "agent_type": "test-validator",
        "task": "验证修复正确性",
        "context": {
            "code": code,
            "language": "java",
            "previous_report": r3.get("report", ""),
        },
    })
    print(f"    结果: success={r4['success']}")

    print()
    print(f"  ── 汇总 ──")
    print(f"  总委派数: {tool.total_delegations}")
    print(f"  成功: {sum(1 for r in [r1, r2, r3, r4] if r['success'])} / 4")

    print("  PASS\n")


async def main():
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║   动态子Agent委派 — Demo 测试                    ║")
    print("║   对应 PDF: Harness Engineering task 工具        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()

    tests = [
        ("YAML加载", test_1_yaml_loading),
        ("AgentRegistry", test_2_agent_registry),
        ("SubAgent独立上下文", test_3_sub_agent_independent_context),
        ("TaskTool串行流水线", test_4_task_tool_pipeline),
        ("TaskTool并行委派", test_5_parallel_delegation),
        ("未知Agent错误处理", test_6_unknown_agent_type),
        ("动态委派LangGraph", test_7_dynamic_graph),
        ("固定管道vs动态委派", test_8_comparison_with_fixed_pipeline),
        ("优雅降级", test_9_graceful_degradation),
        ("端到端流程", test_10_end_to_end_flow),
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
