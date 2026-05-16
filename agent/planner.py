"""
Planning 节点 - 生成Bug修复计划
=============================

分析代码，生成修复步骤计划：
1. 代码审查（ReviewerAgent）
2. Bug分析（AnalyzerAgent）
3. 代码修复（FixerAgent）
4. 测试验证（ValidatorAgent）
"""

import logging
from typing import Dict, List
import json

logger = logging.getLogger("planner")

async def planner_node(state: Dict) -> Dict:
    """规划节点：生成修复计划"""

    code = state.get("code", "")
    language = state.get("language", "java")
    error = state.get("error")
    replan_count = state.get("replan_count", 0)

    logger.info(f"[Planner] 开始规划 (重规划次数: {replan_count})")

    # 根据是否有错误决定计划
    if error and replan_count > 0:
        # 重规划：针对性修复
        plan = generate_replan(state)
    else:
        # 初次规划：完整流程
        plan = generate_initial_plan(code, language)

    logger.info(f"[Planner] 生成计划: {plan}")

    return {
        "plan": plan,
        "current_step": 0,
        "step_results": []
    }


def generate_initial_plan(code: str, language: str) -> List[str]:
    """生成初始修复计划"""

    plan = [
        "step_1_review: 使用ReviewerAgent审查代码，识别潜在Bug位置",
        "step_2_analyze: 使用AnalyzerAgent分析Bug根因，确定Bug类型",
        "step_3_fix: 使用FixerAgent生成修复补丁",
        "step_4_validate: 使用ValidatorAgent运行测试，验证修复"
    ]

    # 根据代码长度调整计划
    if len(code) > 500:
        plan.insert(1, "step_0_preprocess: 预处理大型代码，提取关键函数")

    return plan


def generate_replan(state: Dict) -> List[str]:
    """生成重规划方案"""

    error = state.get("error", "")
    last_result = state.get("validation_result", {})

    plan = []

    # 根据错误类型调整计划
    if "test_failed" in error:
        plan.append("step_1_reanalyze: 重新分析Bug，检查遗漏的边界条件")
        plan.append("step_2_refix: 生成新补丁，修复测试失败的问题")
        plan.append("step_3_validate: 再次运行测试验证")

    elif "patch_failed" in error:
        plan.append("step_1_analyze_patch: 分析补丁失败原因")
        plan.append("step_2_generate_new_patch: 生成新补丁")
        plan.append("step_3_validate: 验证新补丁")

    else:
        # 默认重试完整流程
        plan = generate_initial_plan(state.get("code", ""), state.get("language", "java"))

    return plan


# ============================================
# LLM辅助规划（可选）
# ============================================

async def llm_plan(code: str, api_key: str = None) -> List[str]:
    """使用LLM生成更智能的计划"""

    import httpx
    import os

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return generate_initial_plan(code, "java")

    prompt = f"""分析以下代码，生成Bug修复计划步骤：

代码:
```
{code[:1000]}
```

请输出JSON格式计划列表，如:
["step_1: 描述", "step_2: 描述", ...]

最多4个步骤。"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3
                }
            )

            if response.status_code == 200:
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                # 尝试解析JSON
                try:
                    plan = json.loads(content)
                    return plan[:4]  # 最多4步
                except:
                    pass
    except Exception as e:
        logger.warning(f"[Planner] LLM规划失败: {e}")

    return generate_initial_plan(code, "java")


if __name__ == "__main__":
    import asyncio

    test_code = "public void test() { if (x == null) return; }"

    result = asyncio.run(planner_node({
        "code": test_code,
        "language": "java"
    }))

    print(f"计划: {result['plan']}")