"""
Plan-Execute 状态图 - Bug修复Multi-Agent系统
============================================

架构：Planning → Execute → Replan 循环

状态流转：
1. Planning: 分析Bug，生成修复计划步骤
2. Execute: 按步骤执行（Reviewer→Analyzer→Fixer→Validator）
3. Replan: 检查结果，失败则重新规划

基于 LangGraph 实现。
"""

from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
import operator

# ============================================
# 状态定义
# ============================================

class BugFixState(TypedDict):
    """Bug修复状态"""

    # 输入
    code: str                    # 原始代码
    language: str                 # 编程语言
    test_case: Optional[str]      # 测试用例

    # 规划
    plan: List[str]               # 修复计划步骤列表
    current_step: int             # 当前执行步骤索引

    # 执行结果
    step_results: Annotated[List[Dict], operator.add]  # 每步执行结果

    # Agent输出
    review_result: Optional[Dict]     # ReviewerAgent 输出
    analysis_result: Optional[Dict]   # AnalyzerAgent 输出
    fix_result: Optional[Dict]        # FixerAgent 输出
    validation_result: Optional[Dict] # ValidatorAgent 输出

    # 最终输出
    bug_location: Optional[str]       # Bug位置
    bug_type: Optional[str]           # Bug类型
    patch: Optional[str]              # 修复补丁
    fixed_code: Optional[str]         # 修复后代码
    test_passed: Optional[bool]       # 测试是否通过

    # 控制
    replan_count: int                 # 重规划次数（限制3次）
    error: Optional[str]              # 错误信息
    done: bool                        # 是否完成


# ============================================
# 状态图构建
# ============================================

def build_bugfix_graph():
    """构建Bug修复状态图"""

    from agent.planner import planner_node
    from agent.executor import executor_node
    from agent.replanner import replanner_node

    graph = StateGraph(BugFixState)

    # 添加节点
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("replanner", replanner_node)

    # 设置入口
    graph.set_entry_point("planner")

    # 边：planner → executor
    graph.add_edge("planner", "executor")

    # 条件边：executor → executor（继续执行）或 replanner（执行完成）
    graph.add_conditional_edges(
        "executor",
        should_continue_executing,
        {
            "continue": "executor",
            "done": "replanner"
        }
    )

    # 条件边：replanner → planner 或 END
    graph.add_conditional_edges(
        "replanner",
        should_replan,
        {
            "replan": "planner",
            "end": END
        }
    )

    return graph.compile()


def should_continue_executing(state: BugFixState) -> str:
    """判断是否继续执行步骤"""

    current_step = state.get("current_step", 0)
    plan = state.get("plan", [])

    # 如果还有步骤未执行，继续
    if current_step < len(plan):
        return "continue"

    # 所有步骤完成，进入replanner
    return "done"


def should_replan(state: BugFixState) -> str:
    """判断是否需要重规划"""

    # 成功 → 结束
    if state.get("test_passed"):
        return "end"

    # 重规划次数超限 → 结束
    if state.get("replan_count", 0) >= 3:
        return "end"

    # 有错误但可重试 → 重规划
    if state.get("error") and not state.get("done"):
        return "replan"

    # 默认结束
    return "end"


# ============================================
# 运行入口
# ============================================

async def run_bugfix(code: str, language: str = "java", test_case: str = None) -> Dict:
    """运行Bug修复流程"""

    graph = build_bugfix_graph()

    # 初始状态
    initial_state = BugFixState(
        code=code,
        language=language,
        test_case=test_case,
        plan=[],
        current_step=0,
        step_results=[],
        review_result=None,
        analysis_result=None,
        fix_result=None,
        validation_result=None,
        bug_location=None,
        bug_type=None,
        patch=None,
        fixed_code=None,
        test_passed=None,
        replan_count=0,
        error=None,
        done=False
    )

    # 执行状态图（异步）
    final_state = await graph.ainvoke(initial_state)

    return {
        "bug_location": final_state.get("bug_location"),
        "bug_type": final_state.get("bug_type"),
        "patch": final_state.get("patch"),
        "fixed_code": final_state.get("fixed_code"),
        "test_passed": final_state.get("test_passed"),
        "replan_count": final_state.get("replan_count"),
        "step_results": final_state.get("step_results")
    }


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    import asyncio

    test_code = """
public class BigFraction {
    public BigFraction(double value) {
        if (epsilon == 0.0 && FastMath.abs(q1) < maxDenominator) {
            break;
        }
        throw new FractionConversionException(value, p2, q2);
    }
}
"""

    result = asyncio.run(run_bugfix(test_code, "java"))
    print(f"Bug位置: {result['bug_location']}")
    print(f"测试通过: {result['test_passed']}")