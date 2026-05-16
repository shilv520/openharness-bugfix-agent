"""
动态委派 LangGraph — 主Agent动态委派子Agent
=============================================

PDF Harness Engineering: 主Agent规划 → 动态选择子Agent → 委派 → 汇总

流程:
  1. Orchestrator (主Agent) 制定计划 → 拆解为子任务列表
  2. Dispatcher 根据任务类型选择合适的子Agent类型
  3. 并行/串行委派子Agent执行（每个子Agent独立上下文）
  4. Synthesizer 汇总所有子Agent报告 → 最终输出

改造前 (固定管道):
  planner → reviewer → analyzer → fixer → validator → replanner

改造后 (动态委派):
  orchestrator → dispatcher → [code-reviewer, bug-analyzer, code-fixer, test-validator]
                              ↓ (并行或串行)
                           synthesizer → (replan or END)
"""

from __future__ import annotations

import logging
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

logger = logging.getLogger("graph.dynamic_delegation")


# ── State ─────────────────────────────────────────────────────

class DelegationState(TypedDict, total=False):
    """动态委派的状态定义。

    与旧的BugFixState不同：
      - 不硬编码 review_result / analysis_result / fix_result / validation_result
      - 使用通用的 sub_results 列表存储任意子Agent结果
      - 主Agent可以动态决定委派多少个子Agent
    """
    # 输入
    code: str
    language: str
    test_case: str

    # 编排
    plan: List[Dict[str, Any]]       # [{"agent_type": str, "task": str, "depends_on": int}]
    current_step: int
    sub_results: Annotated[List[Dict[str, Any]], "append"]  # 子Agent结果累积

    # 最终输出
    fixed_code: str
    patch: str
    report: str
    test_passed: bool

    # 控制
    done: bool
    error: str
    replan_count: int
    iteration_count: int

    # LLM 配置
    llm_config: Dict[str, Any]


def create_initial_state(
    code: str,
    language: str = "java",
    test_case: str = "",
    llm_config: Dict[str, Any] = None,
) -> DelegationState:
    """创建初始状态"""
    return DelegationState(
        code=code,
        language=language,
        test_case=test_case,
        plan=[],
        current_step=0,
        sub_results=[],
        fixed_code="",
        patch="",
        report="",
        test_passed=False,
        done=False,
        error="",
        replan_count=0,
        iteration_count=0,
        llm_config=llm_config or {},
    )


# ── Node: Orchestrator ───────────────────────────────────────

async def orchestrator_node(state: DelegationState) -> Dict[str, Any]:
    """主Agent制定工作计划 → 分解为子任务列表。

    对应PDF项目主Agent的第一步：分析任务 → 规划 → 决定委派哪些子Agent。

    与旧planner_node不同：
      - 旧: 固定4步 plan = ["review", "analyze", "fix", "validate"]
      - 新: 动态生成 plan = [{"agent_type": "code-reviewer", "task": "..."}, ...]
    """
    code = state.get("code", "")
    language = state.get("language", "java")
    replan_count = state.get("replan_count", 0)

    if replan_count > 0 and state.get("error"):
        # 重规划模式
        plan = await _replan(state)
    else:
        # 初始规划
        plan = await _generate_plan(code, language, state.get("llm_config", {}))

    logger.info(f"[Orchestrator] Generated plan with {len(plan)} tasks")
    for i, step in enumerate(plan):
        logger.info(f"  Step {i+1}: {step.get('agent_type')} — {step.get('task', '')[:60]}")

    return {
        "plan": plan,
        "current_step": 0,
        "sub_results": [],
        "done": False,
        "error": "",
        "iteration_count": state.get("iteration_count", 0) + 1,
    }


async def _generate_plan(
    code: str,
    language: str,
    llm_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """主Agent分析代码并生成动态任务计划。

    使用LLM决定：
      - 需要哪些类型的子Agent
      - 每个子Agent的具体任务是什么
      - 是否并行还是串行执行
    """
    # 先用启发式规则生成默认计划
    plan = [
        {
            "agent_type": "code-reviewer",
            "task": f"审查以下{language}代码，识别所有潜在Bug、安全漏洞和代码质量问题。输出每个Bug的严重级别、位置和修复建议。",
            "context": {"code": code, "language": language},
            "depends_on": -1,  # 无依赖，可并行
        },
        {
            "agent_type": "bug-analyzer",
            "task": "分析代码审查发现的Bug，判定真假阳性，追溯根因，评估影响范围，推荐修复方案。",
            "context": {"code": code, "language": language},
            "depends_on": 0,   # 依赖 code-reviewer 的结果
        },
        {
            "agent_type": "code-fixer",
            "task": "根据根因分析结果生成精确的代码补丁。遵循最小化修改原则，保持代码风格一致性。",
            "context": {"code": code, "language": language},
            "depends_on": 1,   # 依赖 bug-analyzer 的结果
        },
        {
            "agent_type": "test-validator",
            "task": "验证修复代码的正确性。检查编译、测试通过、无新增Bug、所有已确认Bug已修复。",
            "context": {"code": code, "language": language},
            "depends_on": 2,   # 依赖 code-fixer 的结果
        },
    ]

    # 如果LLM配置可用，尝试LLM动态规划
    if llm_config.get("api_key") or llm_config.get("openai_api_key"):
        try:
            llm_plan = await _llm_plan(code, language, llm_config)
            if llm_plan:
                return llm_plan
        except Exception as e:
            logger.warning(f"[Orchestrator] LLM planning failed, using heuristic: {e}")

    return plan


async def _llm_plan(
    code: str,
    language: str,
    llm_config: Dict[str, Any],
) -> List[Dict[str, Any]] | None:
    """使用LLM进行动态规划"""
    import aiohttp
    import json
    import os

    api_key = llm_config.get("api_key") or llm_config.get("openai_api_key") or os.environ.get("DEEPSEEK_API_KEY")
    base_url = llm_config.get("base_url") or llm_config.get("openai_base_url", "https://api.deepseek.com/v1")
    model = llm_config.get("model", "deepseek-chat")

    prompt = f"""你是一个代码Bug修复任务规划器。你需要为以下代码制定修复计划。

可用的Agent类型:
1. code-reviewer — 代码审查专家，识别潜在Bug
2. bug-analyzer — Bug根因分析专家，追溯根因
3. code-fixer — 代码修复专家，生成补丁
4. test-validator — 测试验证专家，验证修复

语言: {language}
代码:
```
{code[:2000]}
```

请输出JSON格式的任务计划列表，每个任务包含:
- agent_type: 使用的Agent类型
- task: 具体任务描述（中文）
- depends_on: 依赖的前置任务索引（-1表示无依赖）

示例:
[
  {{"agent_type": "code-reviewer", "task": "审查代码中的空指针风险", "depends_on": -1}},
  {{"agent_type": "bug-analyzer", "task": "分析空指针根因", "depends_on": 0}}
]

只输出JSON数组，不要其他内容。"""

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.3,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]

    # 提取JSON数组
    import re
    match = re.search(r'\[[\s\S]*\]', content)
    if match:
        return json.loads(match.group(0))
    return None


async def _replan(state: DelegationState) -> List[Dict[str, Any]]:
    """失败重规划：根据错误调整委派策略"""
    error = state.get("error", "")
    plan = state.get("plan", [])

    if "test_failed" in error or "validation" in error.lower():
        # 测试失败 → 重新委派 fixer 和 validator
        return [
            {"agent_type": "code-fixer", "task": f"修复测试失败: {error}", "context": {"code": state.get("code", "")}, "depends_on": -1},
            {"agent_type": "test-validator", "task": "重新验证修复", "context": {"code": state.get("code", "")}, "depends_on": 0},
        ]
    else:
        # 全面重试
        return await _generate_plan(state.get("code", ""), state.get("language", "java"), state.get("llm_config", {}))


# ── Node: Dispatcher ─────────────────────────────────────────

async def dispatcher_node(state: DelegationState) -> Dict[str, Any]:
    """执行当前步骤的子Agent委派。

    根据plan[current_step]的agent_type和task创建子Agent，
    运行后收集结果。

    这是动态委派的核心：
      - 旧executor_node: 用if/elif分支调用固定函数
      - 新dispatcher_node: 用AgentRegistry动态查找Agent类型 → 创建SubAgent → 执行
    """
    from agent.delegation.agent_registry import get_agent_registry
    from agent.delegation.sub_agent import SubAgent

    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    if current_step >= len(plan):
        return {"done": True, "error": "No more steps"}

    step = plan[current_step]
    agent_type = step.get("agent_type", "")
    task = step.get("task", "")
    context = step.get("context", {})

    # 如果有依赖，合并前置结果
    depends_on = step.get("depends_on", -1)
    if depends_on >= 0:
        sub_results = state.get("sub_results", [])
        if depends_on < len(sub_results):
            prev = sub_results[depends_on]
            context["previous_report"] = prev.get("report", "")
            context["previous_output"] = prev.get("structured_output", {})

    # 确保code在context中
    if "code" not in context:
        context["code"] = state.get("code", "")
    if "language" not in context:
        context["language"] = state.get("language", "java")

    logger.info(f"[Dispatcher] Step {current_step+1}/{len(plan)}: delegating to {agent_type}")

    # 获取Agent定义
    registry = get_agent_registry()
    ad = registry.get(agent_type)
    if not ad:
        return {
            "error": f"Unknown agent type: {agent_type}",
            "done": True,
        }

    # 创建子Agent
    sub = SubAgent(
        definition=ad,
        task_prompt=task,
        context=context,
        llm_config=state.get("llm_config", {}),
        max_iterations=3,
    )

    # 执行
    result = await sub.run()

    # 累积结果
    result_dict = {
        "agent_type": result.agent_type,
        "task_id": result.task_id,
        "success": result.success,
        "report": result.report,
        "structured_output": result.structured_output,
        "duration": result.duration,
        "step_count": result.step_count,
        "error": result.error,
    }

    logger.info(f"[Dispatcher] Step {current_step+1} result: success={result.success}, duration={result.duration:.1f}s")

    return {
        "current_step": current_step + 1,
        "sub_results": [result_dict],
        "done": False,
    }


# ── Node: Synthesizer ─────────────────────────────────────────

async def synthesizer_node(state: DelegationState) -> Dict[str, Any]:
    """汇总所有子Agent报告 → 生成最终输出。

    与旧replanner_node不同：
      - 旧: 只检查test_passed，简单判断
      - 新: 综合所有子Agent的structured_output，生成统一报告
    """
    sub_results = state.get("sub_results", [])
    plan = state.get("plan", [])

    # 检查所有步骤是否成功
    all_success = all(r.get("success", False) for r in sub_results)
    failed_steps = [r for r in sub_results if not r.get("success", False)]

    # 提取最终结果
    test_passed = False
    patch = ""
    fixed_code = ""
    report_sections = []

    for i, (step, result) in enumerate(zip(plan, sub_results)):
        agent_type = step.get("agent_type", "unknown")
        report_sections.append(f"## Step {i+1}: {agent_type}\n\n{result.get('report', 'No report')}")

        # 从 structured_output 提取关键字段
        so = result.get("structured_output", {})

        if agent_type == "test-validator":
            test_passed = so.get("test_passed", False)
        elif agent_type == "code-fixer":
            patch = so.get("patch", "")
            fixed_code = so.get("fixed_code", "")

    report = "\n\n".join(report_sections)

    # 判定是否需要重规划
    if not all_success:
        error_msgs = [f"{r['agent_type']}: {r.get('error', 'Unknown error')}" for r in failed_steps]
        error_str = "; ".join(error_msgs)
        if state.get("replan_count", 0) < 3:
            return {
                "report": report,
                "error": error_str,
                "done": False,
                "replan_count": state.get("replan_count", 0) + 1,
            }
        else:
            return {
                "report": report,
                "fixed_code": fixed_code,
                "patch": patch,
                "test_passed": False,
                "error": f"Max replan reached: {error_str}",
                "done": True,
            }

    # 成功
    return {
        "report": report,
        "fixed_code": fixed_code,
        "patch": patch,
        "test_passed": test_passed,
        "error": "",
        "done": True,
    }


# ── Routing Functions ─────────────────────────────────────────

def should_continue_executing(state: DelegationState) -> str:
    """判断是否继续执行下一步"""
    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    if state.get("done", False):
        return "done"

    if state.get("error") and not state.get("done"):
        return "done"  # 有错误就去synthesizer判定

    if current_step < len(plan):
        return "continue"

    return "done"


def should_replan(state: DelegationState) -> str:
    """判断是否需要重规划"""
    if state.get("done", False) and not state.get("error"):
        return "end"

    if state.get("replan_count", 0) >= 3:
        return "end"

    return "replan"


# ── Graph Builder ─────────────────────────────────────────────

def build_dynamic_delegation_graph() -> StateGraph:
    """构建动态委派LangGraph。

    工作流:
        orchestrator → dispatcher ⟲ (循环执行所有步骤)
                           ↓
                      synthesizer → (replan → orchestrator) or END
    """
    graph = StateGraph(DelegationState)

    # 添加节点
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("dispatcher", dispatcher_node)
    graph.add_node("synthesizer", synthesizer_node)

    # 设置入口
    graph.set_entry_point("orchestrator")

    # 编排 → 委派
    graph.add_edge("orchestrator", "dispatcher")

    # 委派 → 条件: 继续委派 or 去汇总
    graph.add_conditional_edges(
        "dispatcher",
        should_continue_executing,
        {
            "continue": "dispatcher",  # 循环执行下一步
            "done": "synthesizer",     # 所有步骤完成
        },
    )

    # 汇总 → 条件: 结束 or 重新规划
    graph.add_conditional_edges(
        "synthesizer",
        should_replan,
        {
            "end": END,
            "replan": "orchestrator",
        },
    )

    return graph


# ── 便捷函数 ─────────────────────────────────────────────────

async def run_dynamic_delegation(
    code: str,
    language: str = "java",
    test_case: str = "",
    llm_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """运行动态委派修复流程。

    Args:
        code: 待修复的代码
        language: 编程语言
        test_case: 测试用例
        llm_config: LLM配置

    Returns:
        包含report, fixed_code, patch, test_passed的字典
    """
    graph = build_dynamic_delegation_graph()
    compiled = graph.compile()

    initial_state = create_initial_state(
        code=code,
        language=language,
        test_case=test_case,
        llm_config=llm_config,
    )

    result = await compiled.ainvoke(initial_state)

    return {
        "success": result.get("test_passed", False),
        "report": result.get("report", ""),
        "fixed_code": result.get("fixed_code", ""),
        "patch": result.get("patch", ""),
        "test_passed": result.get("test_passed", False),
        "error": result.get("error", ""),
        "sub_results": result.get("sub_results", []),
        "plan": result.get("plan", []),
    }
