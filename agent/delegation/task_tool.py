"""
TaskTool — 主Agent的动态委派工具
==================================

PDF Harness Engineering: 主Agent有task工具 → 动态创建子Agent

用法:
    主Agent发现需要代码审查时:
      task("code-reviewer", "审查BigFraction.java中的空指针风险", {"code": "..."})

TaskTool是MCP兼容的工具定义，主Agent通过它动态委派任务给子Agent。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List

from .agent_registry import AgentRegistry
from .sub_agent import SubAgent, SubAgentResult

logger = logging.getLogger("delegation.task_tool")


class TaskTool:
    """主Agent的动态委派工具。

    对应PDF项目中主Agent的task工具。
    主Agent在ReAct循环中调用task()来委派工作给子Agent。

    用法:
        tool = TaskTool(registry, llm_config)

        # 主Agent调用（同步接口给MCP工具用）
        result = await tool.execute({
            "agent_type": "code-reviewer",
            "task": "审查这段代码",
            "context": {"code": "..."}
        })
    """

    def __init__(
        self,
        registry: AgentRegistry,
        llm_config: Dict[str, Any] = None,
        max_iterations: int = 3,
    ):
        """
        Args:
            registry: AgentRegistry实例
            llm_config: LLM配置
            max_iterations: 子Agent最大迭代次数
        """
        self.registry = registry
        self.llm_config = llm_config or {}
        self.max_iterations = max_iterations

        # 记录所有委派历史
        self._history: List[SubAgentResult] = []

    @property
    def history(self) -> List[SubAgentResult]:
        return list(self._history)

    @property
    def total_delegations(self) -> int:
        return len(self._history)

    # ── 主执行入口 ────────────────────────────────────────────

    async def execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """执行委派任务。

        Args:
            params: {
                "agent_type": str,   # Agent类型名称
                "task": str,         # 任务描述
                "context": dict,     # 任务上下文（代码等）
            }

        Returns:
            {"success": bool, "report": str, "structured_output": dict, "agent_type": str, "task_id": str}
        """
        agent_type = params.get("agent_type", "")
        task = params.get("task", "")
        context = params.get("context", {})

        if not agent_type or not task:
            return {
                "success": False,
                "error": "Missing required parameters: agent_type and task",
            }

        # 查找Agent定义
        ad = self.registry.get(agent_type)
        if not ad:
            available = self.registry.list_briefs()
            return {
                "success": False,
                "error": f"Unknown agent type: {agent_type}. Available: {available}",
            }

        # 创建并运行子Agent
        sub = SubAgent(
            definition=ad,
            task_prompt=task,
            context=context,
            llm_config=self.llm_config,
            max_iterations=self.max_iterations,
        )

        result = await sub.run()
        self._history.append(result)

        return {
            "success": result.success,
            "report": result.report,
            "structured_output": result.structured_output,
            "agent_type": result.agent_type,
            "task_id": result.task_id,
            "duration": result.duration,
            "step_count": result.step_count,
        }

    # ── 并行委派 ──────────────────────────────────────────────

    async def execute_parallel(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """并行执行多个委派任务。

        Args:
            tasks: [{"agent_type": str, "task": str, "context": dict}, ...]

        Returns:
            结果列表（保持输入顺序）
        """
        coros = [self.execute(t) for t in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                output.append({
                    "success": False,
                    "error": str(result),
                    "agent_type": tasks[i].get("agent_type", "unknown"),
                })
            else:
                output.append(result)

        return output

    # ── 串行委派（流水线）──────────────────────────────────────

    async def execute_pipeline(
        self,
        pipeline: List[Dict[str, Any]],
        merge_context: bool = True,
    ) -> List[Dict[str, Any]]:
        """串行执行委派任务，前一个的输出成为后一个的输入。

        Args:
            pipeline: [{"agent_type": str, "task": str, "context": dict}, ...]
            merge_context: 是否将前一步的输出合并到后一步的context

        Returns:
            结果列表（按执行顺序）
        """
        results = []
        accumulated_context = {}

        for step in pipeline:
            # 合并上下文
            if merge_context and results:
                last = results[-1]
                if last.get("success"):
                    accumulated_context["previous_report"] = last.get("report", "")
                    accumulated_context["previous_output"] = last.get("structured_output", {})

            context = {**step.get("context", {}), **accumulated_context}
            result = await self.execute({
                "agent_type": step["agent_type"],
                "task": step["task"],
                "context": context,
            })
            results.append(result)

            # 如果某步失败，可以选择停止
            if not result.get("success"):
                break

        return results


# ── MCP 工具定义 ──────────────────────────────────────────────

def create_task_tool(
    registry: AgentRegistry,
    llm_config: Dict[str, Any] = None,
) -> tuple[TaskTool, Dict[str, Any]]:
    """创建TaskTool和对应的MCP工具定义。

    Returns:
        (TaskTool实例, MCP工具定义dict)
    """
    tool = TaskTool(registry=registry, llm_config=llm_config)

    tool_def = {
        "name": "task",
        "description": (
            "动态委派任务给专业子Agent。"
            "可用的Agent类型通过 list_agent_types 工具查询。"
            "子Agent有独立上下文，执行完返回报告后自动销毁。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "agent_type": {
                    "type": "string",
                    "description": "子Agent类型名称，如 code-reviewer, bug-analyzer, code-fixer, test-validator",
                },
                "task": {
                    "type": "string",
                    "description": "分配给子Agent的具体任务描述",
                },
                "context": {
                    "type": "object",
                    "description": "任务上下文数据（代码片段、前置分析结果等）",
                    "default": {},
                },
            },
            "required": ["agent_type", "task"],
        },
    }

    return tool, tool_def


def create_list_agents_tool(registry: AgentRegistry) -> Dict[str, Any]:
    """创建列出可用Agent类型的MCP工具定义"""
    return {
        "name": "list_agent_types",
        "description": "列出所有可用的子Agent类型及其能力描述。用于主Agent决定委派任务给哪个子Agent。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "可选搜索关键词，在名称和描述中搜索",
                },
            },
        },
    }
