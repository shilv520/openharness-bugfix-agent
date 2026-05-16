"""
Multi-Agent协同状态图（真正协作版）
=================================

流程：
1. Reviewer Agent执行ReAct循环，发现Bug
2. Reviewer ↔ Analyzer讨论确认Bug
3. Analyzer执行ReAct循环，分析根因
4. Analyzer ↔ Fixer讨论修复策略
5. Fixer执行ReAct循环，生成补丁
6. Fixer ↔ Validator反馈循环验证
7. Validator验证，反馈给Fixer调整
8. 成功或重规划
"""

import asyncio
import logging
from typing import TypedDict, List, Dict, Optional, Annotated
from langgraph.graph import StateGraph, END
import operator

from agent.communication import get_comm_bus, reset_comm_bus
from agent.collaborative_agent import (
    CollaborativeReviewerAgent,
    CollaborativeAnalyzerAgent,
    CollaborativeFixerAgent,
    CollaborativeValidatorAgent
)

logger = logging.getLogger("collaborative-graph")


# ============================================
# 状态定义
# ============================================

class CollaborativeBugFixState(TypedDict):
    """协同Bug修复状态"""

    # 输入
    code: str
    language: str
    test_case: Optional[str]

    # 协作记录
    review_analyzer_discussion: Optional[Dict]   # Reviewer-Analyzer讨论
    analyzer_fixer_discussion: Optional[Dict]    # Analyzer-Fixer讨论
    fixer_validator_feedback: Optional[Dict]     # Fixer-Validator反馈

    # 各Agent的ReAct结果
    reviewer_react_result: Optional[Dict]
    analyzer_react_result: Optional[Dict]
    fixer_react_result: Optional[Dict]
    validator_react_result: Optional[Dict]

    # 共享上下文
    shared_context: Dict

    # Agent输出
    bug_candidates: Optional[List]
    bug_location: Optional[str]
    bug_type: Optional[str]
    root_cause: Optional[str]
    patch: Optional[str]
    fixed_code: Optional[str]

    # 验证结果
    test_passed: Optional[bool]
    validation_reason: Optional[str]

    # 控制
    current_phase: str          # review/analyze/fix/validate
    iteration_count: int
    replan_count: int
    error: Optional[str]
    done: bool

    # 通信历史
    message_history: Annotated[List[Dict], operator.add]


# ============================================
# 协同节点实现
# ============================================

async def collaborative_reviewer_node(state: Dict) -> Dict:
    """
    协同Reviewer节点

    执行ReAct循环，并与Analyzer讨论确认Bug
    """
    logger.info("[CollaborativeGraph] Reviewer节点开始")

    # 创建协同Agent
    reviewer = CollaborativeReviewerAgent()

    # 准备上下文
    context = {
        "code": state.get("code"),
        "language": state.get("language"),
        "task": "代码审查和Bug识别"
    }

    # 执行ReAct循环
    react_result = await reviewer.react_loop(context, max_iterations=2)

    # 更新状态
    new_state = {
        "reviewer_react_result": react_result,
        "bug_candidates": react_result.get("bug_candidates"),
        "current_phase": "discuss_with_analyzer",
        "iteration_count": state.get("iteration_count", 0) + react_result.get("iterations", 1)
    }

    # 如果有讨论结果
    if react_result.get("discussion"):
        new_state["review_analyzer_discussion"] = react_result["discussion"]

    logger.info(f"[CollaborativeGraph] Reviewer完成，迭代{react_result.get('iterations')}次")

    return new_state


async def collaborative_analyzer_node(state: Dict) -> Dict:
    """
    协同Analyzer节点

    与Reviewer讨论后，执行ReAct分析根因，并与Fixer讨论修复策略
    """
    logger.info("[CollaborativeGraph] Analyzer节点开始")

    # 创建协同Agent
    analyzer = CollaborativeAnalyzerAgent()

    # 准备上下文（包含Reviewer的结果）
    context = {
        "code": state.get("code"),
        "language": state.get("language"),
        "review_result": {
            "bug_candidates": state.get("bug_candidates", []),
            "discussion": state.get("review_analyzer_discussion")
        },
        "task": "Bug根因分析"
    }

    # 执行ReAct循环
    react_result = await analyzer.react_loop(context, max_iterations=2)

    # 更新状态
    new_state = {
        "analyzer_react_result": react_result,
        "bug_location": react_result.get("bug_location"),
        "bug_type": react_result.get("bug_type"),
        "root_cause": react_result.get("root_cause"),
        "current_phase": "discuss_with_fixer",
        "iteration_count": state.get("iteration_count", 0) + react_result.get("iterations", 1)
    }

    # 如果有修复讨论结果
    if react_result.get("fix_discussion"):
        new_state["analyzer_fixer_discussion"] = react_result["fix_discussion"]

    logger.info(f"[CollaborativeGraph] Analyzer完成，定位: {react_result.get('bug_location')}")

    return new_state


async def collaborative_fixer_node(state: Dict) -> Dict:
    """
    协同Fixer节点

    与Analyzer讨论后，执行ReAct生成补丁，并与Validator反馈循环
    """
    logger.info("[CollaborativeGraph] Fixer节点开始")

    # 创建协同Agent
    fixer = CollaborativeFixerAgent()

    # 准备上下文
    context = {
        "code": state.get("code"),
        "language": state.get("language"),
        "analysis_result": {
            "bug_location": state.get("bug_location"),
            "bug_type": state.get("bug_type"),
            "root_cause": state.get("root_cause")
        },
        "fix_discussion": state.get("analyzer_fixer_discussion"),
        "task": "生成修复补丁"
    }

    # 执行ReAct循环
    react_result = await fixer.react_loop(context, max_iterations=2)

    # 更新状态
    new_state = {
        "fixer_react_result": react_result,
        "patch": react_result.get("patch"),
        "fixed_code": react_result.get("fixed_code"),
        "current_phase": "feedback_with_validator",
        "iteration_count": state.get("iteration_count", 0) + react_result.get("iterations", 1)
    }

    # 如果有反馈循环结果
    if react_result.get("feedback_loop"):
        new_state["fixer_validator_feedback"] = react_result["feedback_loop"]

        # 如果反馈后成功，使用最终补丁
        if react_result["feedback_loop"].get("success"):
            final_patch = react_result["feedback_loop"].get("final_patch", {})
            new_state["patch"] = final_patch.get("patch")
            new_state["fixed_code"] = final_patch.get("fixed_code")

    logger.info(f"[CollaborativeGraph] Fixer完成，补丁生成")

    return new_state


async def collaborative_validator_node(state: Dict) -> Dict:
    """
    协同Validator节点

    验证修复，如果失败则反馈给Fixer调整
    """
    logger.info("[CollaborativeGraph] Validator节点开始")

    # 创建协同Agent
    validator = CollaborativeValidatorAgent()

    # 准备上下文
    context = {
        "code": state.get("code"),
        "fixed_code": state.get("fixed_code"),
        "patch": state.get("patch"),
        "task": "验证修复"
    }

    # 执行ReAct循环
    react_result = await validator.react_loop(context, max_iterations=2)

    # 更新状态
    new_state = {
        "validator_react_result": react_result,
        "test_passed": react_result.get("test_passed"),
        "validation_reason": react_result.get("validation_reason"),
        "current_phase": "done",
        "iteration_count": state.get("iteration_count", 0) + react_result.get("iterations", 1)
    }

    # 如果验证失败，标记错误
    if not react_result.get("test_passed"):
        new_state["error"] = f"验证失败: {react_result.get('validation_reason', 'unknown')}"

    logger.info(f"[CollaborativeGraph] Validator完成，结果: {react_result.get('test_passed')}")

    return new_state


async def collaborative_replanner_node(state: Dict) -> Dict:
    """
    协同Replanner节点

    分析失败原因，决定是否重新协作
    """
    logger.info("[CollaborativeGraph] Replanner节点开始")

    test_passed = state.get("test_passed")
    error = state.get("error")
    replan_count = state.get("replan_count", 0)
    iteration_count = state.get("iteration_count", 0)

    # 成功
    if test_passed:
        logger.info("[CollaborativeGraph] 修复成功！")
        return {
            "done": True,
            "replan_count": replan_count
        }

    # 检查重规划次数
    if replan_count >= 3:
        logger.warning("[CollaborativeGraph] 重规划次数超限")
        return {
            "done": True,
            "error": "重规划次数超限",
            "replan_count": replan_count
        }

    # 分析失败原因（使用共享上下文）
    comm_bus = get_comm_bus()
    shared_context = comm_bus.get_shared_context()

    # 重置通信总线（新的协作循环）
    reset_comm_bus()

    # 重规划
    logger.info(f"[CollaborativeGraph] 重规划 (第{replan_count + 1}次)")

    return {
        "done": False,
        "error": error,
        "replan_count": replan_count + 1,
        "current_phase": "review",
        "iteration_count": iteration_count
    }


# ============================================
# 条件判断
# ============================================

def should_continue(state: Dict) -> str:
    """判断是否继续执行"""
    current_phase = state.get("current_phase", "review")

    phase_sequence = [
        "review",
        "discuss_with_analyzer",
        "analyze",
        "discuss_with_fixer",
        "fix",
        "feedback_with_validator",
        "validate",
        "done"
    ]

    if current_phase == "done":
        return "replanner"

    # 根据当前阶段决定下一个节点
    if current_phase == "review" or current_phase == "discuss_with_analyzer":
        return "analyzer"
    elif current_phase == "analyze" or current_phase == "discuss_with_fixer":
        return "fixer"
    elif current_phase == "fix" or current_phase == "feedback_with_validator":
        return "validator"
    else:
        return "replanner"


def should_replan(state: Dict) -> str:
    """判断是否重规划"""
    if state.get("test_passed"):
        return "end"

    if state.get("done"):
        return "end"

    if state.get("replan_count", 0) >= 3:
        return "end"

    return "replan"


# ============================================
# 构建协同状态图
# ============================================

def build_collaborative_graph():
    """构建协同Bug修复状态图"""

    graph = StateGraph(CollaborativeBugFixState)

    # 添加节点
    graph.add_node("reviewer", collaborative_reviewer_node)
    graph.add_node("analyzer", collaborative_analyzer_node)
    graph.add_node("fixer", collaborative_fixer_node)
    graph.add_node("validator", collaborative_validator_node)
    graph.add_node("replanner", collaborative_replanner_node)

    # 设置入口
    graph.set_entry_point("reviewer")

    # 协同流程边
    graph.add_edge("reviewer", "analyzer")
    graph.add_edge("analyzer", "fixer")
    graph.add_edge("fixer", "validator")
    graph.add_edge("validator", "replanner")

    # 条件边：重规划
    graph.add_conditional_edges(
        "replanner",
        should_replan,
        {
            "replan": "reviewer",
            "end": END
        }
    )

    logger.info("[CollaborativeGraph] 状态图构建完成")

    return graph.compile()


# ============================================
# 运行入口
# ============================================

async def run_collaborative_bugfix(code: str, language: str = "java", test_case: str = None) -> Dict:
    """运行协同Bug修复"""

    # 重置通信总线
    reset_comm_bus()

    # 构建状态图
    graph = build_collaborative_graph()

    # 初始状态
    initial_state = CollaborativeBugFixState(
        code=code,
        language=language,
        test_case=test_case,
        review_analyzer_discussion=None,
        analyzer_fixer_discussion=None,
        fixer_validator_feedback=None,
        reviewer_react_result=None,
        analyzer_react_result=None,
        fixer_react_result=None,
        validator_react_result=None,
        shared_context={},
        bug_candidates=None,
        bug_location=None,
        bug_type=None,
        root_cause=None,
        patch=None,
        fixed_code=None,
        test_passed=None,
        validation_reason=None,
        current_phase="review",
        iteration_count=0,
        replan_count=0,
        error=None,
        done=False,
        message_history=[]
    )

    # 执行状态图
    final_state = await graph.ainvoke(initial_state)

    return {
        "bug_location": final_state.get("bug_location"),
        "bug_type": final_state.get("bug_type"),
        "root_cause": final_state.get("root_cause"),
        "patch": final_state.get("patch"),
        "fixed_code": final_state.get("fixed_code"),
        "test_passed": final_state.get("test_passed"),
        "validation_reason": final_state.get("validation_reason"),
        "replan_count": final_state.get("replan_count"),
        "iteration_count": final_state.get("iteration_count"),
        "discussions": {
            "review_analyzer": final_state.get("review_analyzer_discussion"),
            "analyzer_fixer": final_state.get("analyzer_fixer_discussion"),
            "fixer_validator": final_state.get("fixer_validator_feedback")
        },
        "agent_results": {
            "reviewer": final_state.get("reviewer_react_result"),
            "analyzer": final_state.get("analyzer_react_result"),
            "fixer": final_state.get("fixer_react_result"),
            "validator": final_state.get("validator_react_result")
        }
    }


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    test_code = """
public class BigFraction {
    public BigFraction(double value) {
        if (epsilon == 0.0 && FastMath.abs(q1) < maxDenominator) {
            break;  // Bug: 错误的break逻辑
        }
        throw new FractionConversionException(value, p2, q2);
    }
}
"""

    async def test():
        print("\n=== Multi-Agent协同测试 ===")
        result = await run_collaborative_bugfix(test_code, "java")

        print(f"\nBug位置: {result['bug_location']}")
        print(f"Bug类型: {result['bug_type']}")
        print(f"测试通过: {result['test_passed']}")
        print(f"总迭代次数: {result['iteration_count']}")
        print(f"重规划次数: {result['replan_count']}")

        print("\n=== 协作讨论记录 ===")
        if result['discussions']['review_analyzer']:
            print(f"Reviewer↔Analyzer讨论: {result['discussions']['review_analyzer'].get('rounds')}轮")

        if result['discussions']['analyzer_fixer']:
            print(f"Analyzer↔Fixer讨论: {result['discussions']['analyzer_fixer'].get('rounds')}轮")

        if result['discussions']['fixer_validator']:
            print(f"Fixer↔Validator反馈: {result['discussions']['fixer_validator'].get('iterations')}次迭代")

    asyncio.run(test())