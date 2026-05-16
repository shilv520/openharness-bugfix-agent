"""
动态子Agent委派系统 (Dynamic Sub-Agent Delegation)
===================================================

PDF Harness Engineering: 主Agent有task工具 → 动态创建子Agent
子Agent有独立上下文 → 执行任务 → 返回报告 → 销毁

改造前: Reviewer → Analyzer → Fixer → Validator (LangGraph固定节点)
改造后: 主Agent根据任务动态选择Agent类型 → 并行/串行委派 → 汇总结果
"""

from .agent_definition import AgentDefinition, load_agent_definition, load_all_definitions
from .agent_registry import AgentRegistry, get_agent_registry
from .sub_agent import SubAgent, SubAgentResult
from .task_tool import TaskTool, create_task_tool

__all__ = [
    "AgentDefinition",
    "AgentRegistry",
    "SubAgent",
    "SubAgentResult",
    "TaskTool",
    "load_agent_definition",
    "load_all_definitions",
    "get_agent_registry",
    "create_task_tool",
]
