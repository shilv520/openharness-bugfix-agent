"""
SubAgent — 独立上下文的子Agent运行时
======================================

PDF Harness Engineering: 子Agent = 独立上下文 + ReAct循环 + 返回报告 → 销毁

每个SubAgent:
  1. 从AgentRegistry获取AgentDefinition（system_prompt + tools + skills）
  2. 创建独立上下文（不与其他子Agent共享）
  3. 执行 ReAct 循环: Think → Act → Observe → (Reflect) → 重复
  4. 返回结构化报告给主Agent
  5. 销毁（释放上下文）

关键区别:
  旧: Reviewer/Analyzer/Fixer/Validator 是 LangGraph 固定节点，共享 state
  新: 每个子Agent是独立实例，由主Agent动态创建，上下文隔离
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .agent_definition import AgentDefinition

logger = logging.getLogger("delegation.sub_agent")


@dataclass
class SubAgentResult:
    """子Agent执行结果"""
    agent_type: str
    task_id: str
    success: bool
    report: str                          # 人类可读的报告
    structured_output: Dict[str, Any]    # 结构化输出（JSON Schema）
    steps: List[Dict[str, Any]]          # ReAct步骤记录
    start_time: float
    end_time: float
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def step_count(self) -> int:
        return len(self.steps)


# ── SubAgent ──────────────────────────────────────────────────

class SubAgent:
    """动态创建、独立执行的子Agent。

    对应PDF项目中主Agent通过task工具创建的临时子Agent。
    每个SubAgent有:
      - 自己的AgentDefinition（角色定义）
      - 自己的上下文（task_prompt + context_data）
      - 自己的ReAct循环
      - 执行完立即销毁

    用法:
        registry = get_agent_registry()
        sub = SubAgent(
            definition=registry.get("code-reviewer"),
            task_prompt="审查这段Java代码中的空指针风险",
            context={"code": java_code, "language": "java"},
            llm_config={"api_key": "...", "base_url": "..."},
        )
        result = await sub.run()
        # sub 在 run() 完成后即可丢弃
    """

    def __init__(
        self,
        definition: AgentDefinition,
        task_prompt: str,
        context: Dict[str, Any] = None,
        llm_config: Dict[str, Any] = None,
        max_iterations: int = 3,
        tools: Dict[str, callable] = None,
        compressor=None,       # ContextCompressor 实例（上下文压缩）
        isolation=None,        # ContextIsolation 实例（上下文隔离）
        parent_boundary=None,  # 父ContextBoundary（Fork来源）
    ):
        """
        Args:
            definition: AgentDefinition（从YAML加载的角色定义）
            task_prompt: 主Agent分配的具体任务
            context: 任务上下文数据（代码、前置结果等）
            llm_config: LLM配置 {'api_key', 'base_url', 'model'}
            max_iterations: ReAct循环最大迭代次数
            tools: 可用的工具函数 {'tool_name': callable}
            compressor: ContextCompressor（自动压缩上下文）
            isolation: ContextIsolation（上下文隔离管理器）
            parent_boundary: 父ContextBoundary（fork来源）
        """
        self.definition = definition
        self.task_prompt = task_prompt
        self.context = context or {}
        self.llm_config = llm_config or {}
        self.max_iterations = max_iterations
        self.tools = tools or {}
        self.compressor = compressor
        self.isolation = isolation
        self.parent_boundary = parent_boundary

        self._task_id = uuid.uuid4().hex[:12]
        self._steps: List[Dict[str, Any]] = []
        self._child_boundary = None  # Fork后的子边界

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def agent_type(self) -> str:
        return self.definition.name

    # ── 公开 API ──────────────────────────────────────────────

    async def run(self) -> SubAgentResult:
        """执行子Agent任务。"""
        start = time.time()
        logger.info(f"[SubAgent:{self.agent_type}:{self._task_id}] Starting task: {self.task_prompt[:80]}...")

        try:
            # 如果配置了上下文隔离，Fork一个子边界
            if self.isolation and self.parent_boundary:
                self._child_boundary = self.isolation.fork(
                    parent=self.parent_boundary,
                    agent_type=self.agent_type,
                    task_prompt=self.task_prompt,
                    context=self.context,
                )
                logger.debug(f"[SubAgent:{self._task_id}] Forked context: {self._child_boundary.boundary_id}")

            steps = await self._run_react_loop()
            report, structured = await self._synthesize(steps)

            # 如果配置了上下文隔离，Merge结果回父Agent
            if self.isolation and self.parent_boundary and self._child_boundary:
                self.isolation.merge(
                    parent=self.parent_boundary,
                    child=self._child_boundary,
                    child_result={"report": report, "structured_output": structured},
                    merge_strategy="structured_only",
                )
                self.isolation.cleanup(self._child_boundary)
                logger.debug(f"[SubAgent:{self._task_id}] Merged & cleaned up context")

            end = time.time()
            logger.info(
                f"[SubAgent:{self.agent_type}:{self._task_id}] "
                f"Completed in {end - start:.1f}s, {len(steps)} steps"
            )

            return SubAgentResult(
                agent_type=self.agent_type,
                task_id=self._task_id,
                success=True,
                report=report,
                structured_output=structured,
                steps=steps,
                start_time=start,
                end_time=end,
            )
        except Exception as e:
            end = time.time()
            logger.error(f"[SubAgent:{self.agent_type}:{self._task_id}] Failed: {e}")
            # 清理子边界
            if self._child_boundary and self.isolation:
                try:
                    self.isolation.cleanup(self._child_boundary)
                except Exception:
                    pass
            return SubAgentResult(
                agent_type=self.agent_type,
                task_id=self._task_id,
                success=False,
                report=f"Task failed: {e}",
                structured_output={},
                steps=self._steps,
                start_time=start,
                end_time=end,
                error=str(e),
            )

    # ── ReAct 循环 ────────────────────────────────────────────

    async def _run_react_loop(self) -> List[Dict[str, Any]]:
        """执行 ReAct 循环: Think → Act → Observe → (Reflect)"""
        steps: List[Dict[str, Any]] = []
        memory: List[Dict[str, str]] = []  # 对话历史

        # 如果配置了compressor，用它管理消息
        # 否则手动管理 memory 列表

        # 构建初始 prompt
        system_msg = self._build_system_message()
        user_msg = self._build_user_message()

        for iteration in range(self.max_iterations):
            logger.debug(f"[SubAgent:{self._task_id}] Iteration {iteration + 1}/{self.max_iterations}")

            # 如果配置了compressor，检查是否需要压缩
            if self.compressor and self.compressor.needs_compression:
                await self.compressor.compress()
                logger.debug(f"[SubAgent:{self._task_id}] Context compressed: {self.compressor.stats}")

            # Step 1: Think — 调用LLM思考下一步
            think_result = await self._think(system_msg, memory, user_msg, iteration)
            if not think_result:
                break

            step = {
                "iteration": iteration + 1,
                "thought": think_result.get("analysis", ""),
                "action": think_result.get("action", "observe"),
                "action_detail": think_result.get("action_detail", ""),
            }
            steps.append(step)

            # Step 2: Act — 执行动作
            action = think_result.get("action", "")
            if action == "finalize":
                step["result"] = think_result.get("final_answer", "")
                break
            elif action in self.tools:
                tool_result = await self._call_tool(action, think_result.get("action_input", {}))
                step["result"] = tool_result
                result_str = json.dumps(tool_result, ensure_ascii=False)
                memory.append({"role": "tool", "content": result_str})
                # 同步更新compressor
                if self.compressor:
                    self.compressor.add_message("tool", result_str)
            elif action == "observe":
                step["result"] = think_result.get("observation", "")
            else:
                step["result"] = f"Unknown action: {action}"

        return steps

    async def _think(
        self,
        system_msg: str,
        memory: List[Dict[str, str]],
        user_msg: str,
        iteration: int,
    ) -> Optional[Dict[str, Any]]:
        """调用LLM进行思考，返回JSON决策"""
        messages = [{"role": "system", "content": system_msg}]

        # 添加对话历史
        for entry in memory[-6:]:  # 只保留最近3轮
            messages.append(entry)

        if iteration == 0:
            messages.append({"role": "user", "content": user_msg})
        else:
            messages.append({
                "role": "user",
                "content": "请继续执行任务。如果已完成，请使用 finalize 动作。使用中文回复。"
            })

        try:
            response = await self._call_llm(messages)
            return self._parse_thinking_response(response)
        except Exception as e:
            logger.error(f"[SubAgent:{self._task_id}] Think failed: {e}")
            return None

    async def _synthesize(self, steps: List[Dict[str, Any]]) -> tuple[str, dict]:
        """汇总ReAct步骤为最终报告"""
        if not steps:
            return "No steps executed.", {}

        # 构建最终综合 prompt
        steps_summary = json.dumps(steps, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": self._build_synthesize_system()},
            {"role": "user", "content": f"任务: {self.task_prompt}\n\n执行步骤:\n{steps_summary}\n\n请根据以上执行步骤生成最终报告。输出JSON格式。"}
        ]

        try:
            response = await self._call_llm(messages)
            return self._parse_synthesize_response(response, steps)
        except Exception:
            # 回退：基于步骤生成简单报告
            report = "\n".join([
                f"## Step {s['iteration']}: {s.get('thought', '')}\n{s.get('result', '')}"
                for s in steps
            ])
            return report, {"steps_count": len(steps)}

    # ── LLM 调用 ──────────────────────────────────────────────

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用LLM API"""
        import aiohttp

        api_key = self.llm_config.get("api_key") or self.llm_config.get("openai_api_key")
        base_url = self.llm_config.get("base_url") or self.llm_config.get("openai_base_url", "https://api.deepseek.com/v1")
        model = self.llm_config.get("model", "deepseek-chat")

        if not api_key:
            # 尝试从环境变量获取
            import os
            api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0.3,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise RuntimeError(f"LLM API error {resp.status}: {text[:200]}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    # ── Prompt 构建 ───────────────────────────────────────────

    def _build_system_message(self) -> str:
        """构建子Agent的system prompt"""
        tools_desc = self._describe_tools()
        schema_desc = json.dumps(self.definition.output_schema, ensure_ascii=False, indent=2)

        return f"""{self.definition.system_prompt}

## 可用工具
{tools_desc}

## 输出格式要求
你必须以JSON格式输出每一步的决策:

{{
  "analysis": "对当前状态的分析（中文）",
  "action": "工具名称 或 observe 或 finalize",
  "action_input": {{"param": "value"}},
  "action_detail": "动作说明（中文）"
}}

如果任务已完成，使用 action="finalize" 并提供:
{{
  "analysis": "任务完成总结",
  "action": "finalize",
  "final_answer": "完整的执行结果报告（中文）"
}}

## 输出Schema
最终报告必须符合以下结构:
{schema_desc}
"""

    def _build_user_message(self) -> str:
        """构建用户任务消息"""
        context_str = json.dumps(self.context, ensure_ascii=False, indent=2)

        return f"""## 任务
{self.task_prompt}

## 上下文
{context_str}

请开始执行任务。先分析当前状态，然后选择下一步动作。使用中文回复。"""

    def _build_synthesize_system(self) -> str:
        schema_desc = json.dumps(self.definition.output_schema, ensure_ascii=False, indent=2)
        return f"""你是{self.definition.description}。
请根据执行步骤生成最终的结构化报告。
输出JSON格式，必须包含以下字段:
{schema_desc}

同时包含:
- "report": 人类可读的完整报告（Markdown格式，中文）
"""

    def _describe_tools(self) -> str:
        """描述可用工具"""
        if not self.tools:
            return "无特殊工具，仅可进行分析和推理。"
        lines = []
        for name, func in self.tools.items():
            doc = getattr(func, "__doc__", "无描述") or "无描述"
            lines.append(f"- **{name}**: {doc.strip().split(chr(10))[0]}")
        return "\n".join(lines)

    # ── 响应解析 ──────────────────────────────────────────────

    def _parse_thinking_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析LLM的思考响应"""
        try:
            # 尝试提取JSON
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            return data
        except (json.JSONDecodeError, ValueError):
            # 回退：文本响应视为finalize
            return {
                "analysis": "Task completed based on text response",
                "action": "finalize",
                "final_answer": response,
            }

    def _parse_synthesize_response(
        self, response: str, steps: List[Dict[str, Any]]
    ) -> tuple[str, dict]:
        """解析综合响应"""
        try:
            json_str = self._extract_json(response)
            data = json.loads(json_str)
            report = data.pop("report", json.dumps(data, ensure_ascii=False, indent=2))
            return report, data
        except (json.JSONDecodeError, ValueError):
            # 回退
            report = response
            return report, {"raw_response": response, "steps_count": len(steps)}

    def _extract_json(self, text: str) -> str:
        """从文本中提取JSON块"""
        # 尝试 ```json ``` 块
        import re
        match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
        if match:
            return match.group(1).strip()
        # 尝试 { } 块
        match = re.search(r'\{[\s\S]*\}', text)
        if match:
            return match.group(0).strip()
        return text

    # ── 工具调用 ──────────────────────────────────────────────

    async def _call_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """调用工具函数"""
        func = self.tools.get(tool_name)
        if not func:
            return {"error": f"Tool not found: {tool_name}"}

        try:
            if asyncio.iscoroutinefunction(func):
                return await func(**tool_input)
            else:
                return func(**tool_input)
        except Exception as e:
            return {"error": f"Tool '{tool_name}' failed: {str(e)}"}
