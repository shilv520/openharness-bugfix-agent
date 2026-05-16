"""
Replan 节点 - 检查结果并决定是否重规划
=====================================

检查执行结果：
- 成功 → 结束
- 失败 → 重规划（最多3次）
"""

import logging
from typing import Dict

logger = logging.getLogger("replanner")

async def replanner_node(state: Dict) -> Dict:
    """重规划节点：检查结果，决定下一步"""

    test_passed = state.get("test_passed")
    error = state.get("error")
    replan_count = state.get("replan_count", 0)
    step_results = state.get("step_results", [])

    logger.info(f"[Replanner] 检查结果 (test_passed={test_passed}, replan_count={replan_count})")

    # 成功
    if test_passed:
        logger.info("[Replanner] 修复成功！")
        return {
            "done": True,
            "replan_count": replan_count
        }

    # 检查是否需要重规划
    if replan_count >= 3:
        logger.warning("[Replanner] 重规划次数已达上限，终止")
        return {
            "done": True,
            "error": "重规划次数超限",
            "replan_count": replan_count
        }

    # 分析失败原因
    failure_reason = analyze_failure(state)
    logger.info(f"[Replanner] 失败原因: {failure_reason}")

    # 标记需要重规划
    return {
        "done": False,
        "error": failure_reason,
        "replan_count": replan_count + 1
    }


def analyze_failure(state: Dict) -> str:
    """分析失败原因"""

    validation_result = state.get("validation_result") or {}
    fix_result = state.get("fix_result") or {}
    analysis_result = state.get("analysis_result") or {}

    # 测试失败
    if validation_result.get("tests_failed", 0) > 0:
        failed_tests = validation_result.get("failed_test_names") or []
        return f"test_failed: {failed_tests}"

    # 补丁生成失败
    if fix_result.get("status") == "error":
        return f"patch_failed: {fix_result.get('error', 'unknown')}"

    # 分析失败
    if analysis_result.get("status") == "error":
        return f"analysis_failed: {analysis_result.get('error', 'unknown')}"

    # 默认
    return "unknown_failure"


# ============================================
# 统计报告生成
# ============================================

def generate_report(state: Dict) -> Dict:
    """生成修复报告"""

    return {
        "bug_location": state.get("bug_location"),
        "bug_type": state.get("bug_type"),
        "patch": state.get("patch"),
        "test_passed": state.get("test_passed"),
        "replan_count": state.get("replan_count"),
        "steps_executed": len(state.get("step_results", [])),
        "success": state.get("test_passed", False)
    }


if __name__ == "__main__":
    import asyncio

    # 测试成功场景
    success_state = {"test_passed": True, "replan_count": 0}
    result = asyncio.run(replanner_node(success_state))
    print(f"成功场景: {result}")

    # 测试失败场景
    fail_state = {
        "test_passed": False,
        "replan_count": 1,
        "validation_result": {"tests_failed": 2, "failed_test_names": ["test1", "test2"]}
    }
    result = asyncio.run(replanner_node(fail_state))
    print(f"失败场景: {result}")