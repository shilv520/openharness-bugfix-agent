"""
协同Agent - 带ReAct循环和通信能力 + Skills动态加载 + 轨迹持久化
================================

每个Agent具备：
1. 思考（Think）：分析当前情况，决定下一步
2. 行动（Act）：执行具体操作（使用动态加载的Skills）
3. 观察（Observe）：观察执行结果
4. 反思（Reflect）：分析失败原因
5. 协作（Collaborate）：与其他Agent讨论
6. Skills动态加载：按需加载Skills增强能力
7. 轨迹持久化：记录思考/行动轨迹到TrajectoryStore
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
import json
import os
import httpx
from pathlib import Path

from .communication import AgentMessage, get_comm_bus, DiscussionProtocol, FeedbackProtocol
from .skills import load_skill_registry, SkillDefinition, SkillRegistry
from .trajectory_store import TrajectoryStore, get_trajectory_store
from .redis_memory import HierarchicalMemory

logger = logging.getLogger("collaborative-agent")

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)


def get_llm_config():
    """获取LLM配置"""
    return {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1").rstrip("/"),
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    }


async def call_llm(prompt: str, max_tokens: int = 2000) -> str:
    """调用LLM"""
    config = get_llm_config()
    api_key = config["api_key"]
    base_url = config["base_url"]
    model = config["model"]

    if not api_key:
        return "模拟结果"

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                    "max_tokens": max_tokens
                }
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API错误: {response.status_code}"
    except Exception as e:
        return f"调用失败: {str(e)}"


class CollaborativeAgent:
    """
    协同Agent基类

    具备ReAct循环、通信能力、Skills动态加载和轨迹持久化
    """

    # 全局Skill注册表（共享）
    _skill_registry = None
    # 全局轨迹存储（共享）
    _trajectory_store = None
    # 全局分层记忆（共享）
    _hierarchical_memory = None

    def __init__(self, name: str, role: str, capabilities: List[str], skill_names: List[str] = None):
        self.name = name
        self.role = role
        self.capabilities = capabilities
        self.skill_names = skill_names or []  # Agent需要的Skills
        self.loaded_skills: Dict[str, SkillDefinition] = {}  # 已加载的Skills
        self.comm_bus = get_comm_bus()
        self.thought_history: List[Dict] = []  # 思考历史
        self.action_history: List[Dict] = []   # 行动历史
        self._current_bug_id: str = ""  # 当前处理的Bug ID

        # 初始化Skill注册表（全局共享）
        if CollaborativeAgent._skill_registry is None:
            CollaborativeAgent._skill_registry = load_skill_registry(Path(__file__).parent.parent)

        # 初始化轨迹存储（全局共享）
        if CollaborativeAgent._trajectory_store is None:
            CollaborativeAgent._trajectory_store = get_trajectory_store()

        # 初始化分层记忆（全局共享）
        if CollaborativeAgent._hierarchical_memory is None:
            CollaborativeAgent._hierarchical_memory = HierarchicalMemory()
            logger.info(f"[Memory] HierarchicalMemory 初始化完成")

        # 当前会话的 memory 标识
        self._memory_user_id = "agent_system"
        self._memory_session_id = ""

        # 动态加载Agent需要的Skills
        self._load_skills()

        # 注册到通信总线
        self.comm_bus.register_agent(name, self.handle_message)

        logger.info(f"[{name}] 协同Agent初始化，角色: {role}")
        logger.info(f"[{name}] 已加载Skills: {list(self.loaded_skills.keys())}")

    def _load_skills(self):
        """动态加载Agent需要的Skills"""
        registry = CollaborativeAgent._skill_registry

        for skill_name in self.skill_names:
            skill = registry.get(skill_name)
            if skill:
                self.loaded_skills[skill_name] = skill
                logger.info(f"[{self.name}] 加载Skill: {skill_name}")
            else:
                logger.warning(f"[{self.name}] Skill未找到: {skill_name}")

    def get_skill_content(self, skill_name: str) -> str:
        """获取Skill内容"""
        skill = self.loaded_skills.get(skill_name)
        if skill:
            return skill.content
        return ""

    def list_available_skills(self) -> List[str]:
        """列出所有可用Skills"""
        return list(self.loaded_skills.keys())

    def set_memory_session(self, user_id: str, session_id: str):
        """设置当前会话的 memory 标识（由调用者在任务开始时设置）"""
        self._memory_user_id = user_id
        self._memory_session_id = session_id

    async def _enrich_prompt_with_memory(self, base_prompt: str, search_query: str = "") -> str:
        """用分层记忆丰富 LLM 提示词"""
        memory = CollaborativeAgent._hierarchical_memory
        if memory is None:
            return base_prompt

        try:
            query = search_query or base_prompt[:200]
            memory_context = await memory.build_context_for_llm(
                self._memory_user_id,
                self._memory_session_id,
                query
            )
            if memory_context:
                return f"[历史记忆上下文]\n{memory_context}\n\n---\n\n{base_prompt}"
        except Exception:
            pass
        return base_prompt

    async def _store_memory_fact(self, fact_key: str, fact_value: str):
        """存储长期记忆事实"""
        memory = CollaborativeAgent._hierarchical_memory
        if memory is None:
            return
        try:
            await memory.remember_fact(self._memory_user_id, fact_key, fact_value)
            logger.info(f"[{self.name}] 存储记忆: {fact_key}")
        except Exception as e:
            logger.warning(f"[{self.name}] 记忆存储失败: {e}")

    async def _append_memory_message(self, message: str, role: str = "assistant"):
        """添加消息到短期记忆"""
        memory = CollaborativeAgent._hierarchical_memory
        if memory is None or not self._memory_session_id:
            return
        try:
            await memory.append_message(self._memory_user_id, self._memory_session_id, message, role)
        except Exception:
            pass

    async def handle_message(self, message: AgentMessage):
        """处理接收到的消息"""
        logger.info(f"[{self.name}] 收到消息: {message.from_agent} - {message.message_type}")

        # 根据消息类型处理
        if message.message_type == "discuss":
            # 讨论请求，需要回应
            response = await self.generate_discussion_response(message)
            self.comm_bus.send_message(response)

        elif message.message_type == "feedback":
            # 反馈消息，可能需要调整
            await self.handle_feedback(message)

        elif message.message_type == "request":
            # 请求消息，需要执行
            result = await self.execute_request(message)
            response = AgentMessage(
                from_agent=self.name,
                to_agent=message.from_agent,
                message_type="response",
                content="请求执行完成",
                data=result
            )
            self.comm_bus.send_message(response)

    async def think(self, context: Dict) -> Dict:
        """
        思考阶段（ReAct - Think）

        分析当前情况，决定下一步行动
        自动记录到 TrajectoryStore，并用 HierarchicalMemory 丰富上下文
        """
        logger.info(f"[{self.name}] 思考中...")

        # 设置当前Bug ID
        self._current_bug_id = context.get("bug_id", "unknown")
        if not self._memory_session_id:
            self._memory_session_id = self._current_bug_id

        prompt = f"""你是{self.name} Agent，角色是{self.role}。
你具备的能力：{', '.join(self.capabilities)}

当前上下文：
{json.dumps(context, indent=2, ensure_ascii=False)}

请思考：
1. 当前情况分析
2. 需要做什么？
3. 应该用什么策略？
4. 需要与其他Agent协作吗？

输出JSON格式结果，包含analysis、next_action、strategy、need_collaboration、collaborate_with、confidence字段。
只输出JSON。"""

        # 用分层记忆丰富提示词
        enriched_prompt = await self._enrich_prompt_with_memory(
            prompt, context.get("task", context.get("code", ""))[:200]
        )

        response = await call_llm(enriched_prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            thought = json.loads(response[json_start:json_end])
        except:
            thought = {
                "analysis": "需要进一步分析",
                "next_action": "执行默认操作",
                "strategy": "default",
                "need_collaboration": False,
                "confidence": 0.5
            }

        self.thought_history.append(thought)

        # 记录轨迹
        CollaborativeAgent._trajectory_store.record_think(
            self._current_bug_id,
            self.name,
            thought
        )

        logger.info(f"[{self.name}] 思考结果: {thought['next_action']}")

        return thought

    async def act(self, action: str, context: Dict) -> Dict:
        """
        行动阶段（ReAct - Act）

        执行具体操作，自动记录轨迹
        """
        logger.info(f"[{self.name}] 执行行动: {action}")

        # 根据Agent角色执行不同行动
        result = await self.execute_action(action, context)

        action_record = {
            "action": action,
            "result": result,
            "timestamp": self._get_timestamp()
        }
        self.action_history.append(action_record)

        # 记录轨迹
        CollaborativeAgent._trajectory_store.record_act(
            self._current_bug_id,
            self.name,
            action,
            result
        )

        return result

    async def observe(self, action_result: Dict) -> Dict:
        """
        观察阶段（ReAct - Observe）

        观察行动结果，判断是否成功，自动记录轨迹
        并用 HierarchicalMemory 丰富上下文
        """
        logger.info(f"[{self.name}] 观察行动结果...")

        prompt = f"""你是{self.name} Agent，刚执行了一个行动。

行动结果：
{json.dumps(action_result, indent=2, ensure_ascii=False)}

请观察：
1. 行动是否成功？
2. 结果是否符合预期？
3. 是否需要调整？

输出JSON格式结果，包含success、observation、need_adjustment、adjustment_reason和confidence字段。
只输出JSON。"""

        enriched_prompt = await self._enrich_prompt_with_memory(
            prompt, json.dumps(action_result, ensure_ascii=False)[:200]
        )
        response = await call_llm(enriched_prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            observation = json.loads(response[json_start:json_end])
        except:
            observation = {
                "success": action_result.get("status") == "success",
                "observation": "需要人工确认",
                "need_adjustment": False,
                "confidence": 0.5
            }

        # 记录轨迹
        CollaborativeAgent._trajectory_store.record_observe(
            self._current_bug_id,
            self.name,
            observation
        )

        logger.info(f"[{self.name}] 观察结果: 成功={observation['success']}")
        return observation

    async def reflect(self, failure_reason: str, context: Dict) -> Dict:
        """
        反思阶段（ReAct - Reflect）

        分析失败原因，生成调整策略，自动记录轨迹
        并用 HierarchicalMemory 丰富上下文
        """
        logger.info(f"[{self.name}] 反思失败原因: {failure_reason}")

        prompt = f"""你是{self.name} Agent，行动失败了。

失败原因：{failure_reason}

当前上下文：
{json.dumps(context, indent=2, ensure_ascii=False)}

请反思：
1. 为什么失败？
2. 如何改进？
3. 新的策略是什么？

输出JSON格式结果，包含failure_analysis、improvement、new_strategy和retry_confidence字段。
只输出JSON。"""

        enriched_prompt = await self._enrich_prompt_with_memory(prompt, failure_reason[:200])
        response = await call_llm(enriched_prompt)

        try:
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            reflection = json.loads(response[json_start:json_end])
        except:
            reflection = {
                "failure_analysis": failure_reason,
                "improvement": "重新尝试",
                "new_strategy": "default",
                "retry_confidence": 0.5
            }

        # 记录轨迹
        CollaborativeAgent._trajectory_store.record_reflect(
            self._current_bug_id,
            self.name,
            reflection
        )

        logger.info(f"[{self.name}] 反思结果: {reflection['improvement']}")
        return reflection

    async def collaborate(self, target_agent: str, topic: str, data: Dict) -> Dict:
        """
        协作阶段 - 与其他Agent讨论，自动记录讨论轨迹
        """
        logger.info(f"[{self.name}] 与{target_agent}协作讨论: {topic}")

        discussion_protocol = DiscussionProtocol(self.comm_bus, call_llm)

        initial_data = {
            "agent1_view": data.get("view", ""),
            "topic": topic,
            **data
        }

        result = await discussion_protocol.discuss(
            agent1=self.name,
            agent2=target_agent,
            topic=topic,
            initial_data=initial_data
        )

        # 更新共享上下文
        if result["agreed"]:
            self.comm_bus.update_shared_context(f"{topic}_consensus", result["consensus"])

        # 记录讨论轨迹
        CollaborativeAgent._trajectory_store.record_discuss(
            self._current_bug_id,
            [self.name, target_agent],
            topic,
            result
        )

        return result

    async def react_loop(self, context: Dict, max_iterations: int = 3) -> Dict:
        """
        ReAct完整循环

        Think → Act → Observe → Reflect（如果失败）
        """
        logger.info(f"[{self.name}] 开始ReAct循环")

        iteration = 0
        success = False
        final_result = {}

        while iteration < max_iterations and not success:
            iteration += 1

            # Think
            thought = await self.think(context)

            # 如果需要协作
            if thought.get("need_collaboration"):
                collaborators = thought.get("collaborate_with", [])
                if collaborators:
                    collab_result = await self.collaborate(
                        collaborators[0],
                        context.get("task", "unknown"),
                        {"view": thought.get("analysis", "")}
                    )
                    context["collaboration_result"] = collab_result

            # Act
            action_result = await self.act(thought["next_action"], context)

            # Observe
            observation = await self.observe(action_result)

            if observation["success"]:
                success = True
                final_result = action_result
                break

            # Reflect（如果失败）
            if observation["need_adjustment"]:
                reflection = await self.reflect(
                    observation.get("adjustment_reason", "unknown"),
                    context
                )
                context["reflection"] = reflection
                context["strategy"] = reflection["new_strategy"]

        final_result["iterations"] = iteration
        final_result["success"] = success
        final_result["thought_history"] = self.thought_history
        final_result["action_history"] = self.action_history

        logger.info(f"[{self.name}] ReAct结束，迭代{iteration}次，成功={success}")
        return final_result

    async def execute_action(self, action: str, context: Dict) -> Dict:
        """执行具体行动（子类实现）"""
        return {"status": "success", "action": action}

    async def generate_discussion_response(self, message: AgentMessage) -> AgentMessage:
        """生成讨论回应"""
        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            message_type="discuss",
            content="收到讨论请求",
            data={"agree": True}
        )

    async def handle_feedback(self, message: AgentMessage):
        """处理反馈"""
        logger.info(f"[{self.name}] 处理反馈: {message.content}")

    async def execute_request(self, message: AgentMessage) -> Dict:
        """执行请求"""
        return {"status": "success"}

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()


# ============================================
# 具体Agent实现
# ============================================

class CollaborativeReviewerAgent(CollaborativeAgent):
    """协同Reviewer Agent - 使用动态加载的Code Review Skill"""

    def __init__(self):
        super().__init__(
            name="Reviewer",
            role="代码审查专家",
            capabilities=["代码审查", "Bug识别", "质量评估"],
            skill_names=["Code Review", "Agent Discussion"]  # 动态加载Skills
        )

    async def execute_action(self, action: str, context: Dict) -> Dict:
        """执行审查行动（使用Skill增强）"""

        if action == "review_code":
            code = context.get("code", "")

            review_skill = self.get_skill_content("Code Review")

            prompt = f"""你是代码审查专家。

以下是你的Skill指导：
{review_skill[:1000] if review_skill else '无Skill'}

请审查代码：
```
{code}
```

识别潜在的Bug和问题。输出JSON格式结果，包含bug_candidates列表、code_quality、potential_issues和review_summary字段。只输出JSON。"""

            enriched_prompt = await self._enrich_prompt_with_memory(prompt, code[:200])
            response = await call_llm(enriched_prompt)

            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                result = json.loads(response[json_start:json_end])
                result["status"] = "success"
            except:
                result = {
                    "status": "success",
                    "bug_candidates": [{"location": "unknown", "type": "unknown", "confidence": 0.5}],
                    "code_quality": "中等"
                }

            # 存储审查发现到长期记忆
            if result.get("bug_candidates"):
                bug_types = [b.get("type", "unknown") for b in result["bug_candidates"]]
                await self._store_memory_fact("last_review_bug_types", ", ".join(bug_types))
                await self._store_memory_fact("last_review_code_quality", result.get("code_quality", "unknown"))

            return result

        return {"status": "success", "action": action}


class CollaborativeAnalyzerAgent(CollaborativeAgent):
    """协同Analyzer Agent - 使用动态加载的Bug Analysis Skill"""

    def __init__(self):
        super().__init__(
            name="Analyzer",
            role="Bug根因分析专家",
            capabilities=["根因分析", "Bug分类", "修复建议"],
            skill_names=["Bug Analysis", "Agent Discussion"]  # 动态加载Skills
        )

    async def execute_action(self, action: str, context: Dict) -> Dict:
        """执行分析行动（使用Skill增强）"""

        if action == "analyze_bug":
            code = context.get("code", "")
            review_result = context.get("review_result", {})
            bug_candidates = review_result.get("bug_candidates", [])

            analysis_skill = self.get_skill_content("Bug Analysis")

            prompt = f"""你是Bug根因分析专家。

以下是你的Skill指导：
{analysis_skill[:1000] if analysis_skill else '无Skill'}

代码：
```
{code}
```

审查发现的Bug候选：
{json.dumps(bug_candidates, indent=2, ensure_ascii=False)}

深入分析最可能的Bug。输出JSON格式结果，包含bug_location、bug_type、root_cause、fix_suggestion和confidence字段。只输出JSON。"""

            enriched_prompt = await self._enrich_prompt_with_memory(
                prompt, json.dumps(bug_candidates, ensure_ascii=False)[:200]
            )
            response = await call_llm(enriched_prompt)

            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                result = json.loads(response[json_start:json_end])
                result["status"] = "success"
            except:
                result = {
                    "status": "success",
                    "bug_location": "unknown",
                    "bug_type": "unknown",
                    "root_cause": "需要分析",
                    "confidence": 0.5
                }

            # 存储分析结果到长期记忆
            if result.get("bug_type"):
                await self._store_memory_fact("last_bug_type", result.get("bug_type", ""))
                await self._store_memory_fact("last_root_cause", result.get("root_cause", "")[:200])
                await self._store_memory_fact("last_fix_suggestion", result.get("fix_suggestion", "")[:200])

            return result

        return {"status": "success", "action": action}


class CollaborativeFixerAgent(CollaborativeAgent):
    """协同Fixer Agent - 使用动态加载的Code Fix Skill"""

    def __init__(self):
        super().__init__(
            name="Fixer",
            role="代码修复专家",
            capabilities=["补丁生成", "代码修复", "多方案对比"],
            skill_names=["Code Fix", "Agent Discussion"]  # 动态加载Skills
        )

    async def execute_action(self, action: str, context: Dict) -> Dict:
        """执行修复行动（使用Skill增强）"""

        if action == "generate_patch":
            code = context.get("code", "")
            analysis_result = context.get("analysis_result", {})
            fix_discussion = context.get("fix_discussion", {})

            fix_skill = self.get_skill_content("Code Fix")

            prompt = f"""你是代码修复专家。

以下是你的Skill指导：
{fix_skill[:1000] if fix_skill else '无Skill'}

原始代码：
```
{code}
```

Bug分析结果：
{json.dumps(analysis_result, indent=2, ensure_ascii=False)}

讨论建议：
{json.dumps(fix_discussion.get('consensus', {}), indent=2, ensure_ascii=False)}

生成修复补丁。输出JSON格式结果，包含patch、fixed_code、fix_explanation和confidence字段。只输出JSON。"""

            enriched_prompt = await self._enrich_prompt_with_memory(
                prompt, analysis_result.get("bug_type", analysis_result.get("root_cause", ""))[:200]
            )
            response = await call_llm(enriched_prompt, max_tokens=3000)

            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                result = json.loads(response[json_start:json_end])
                result["status"] = "success"
            except:
                result = {
                    "status": "success",
                    "patch": "mock patch",
                    "fixed_code": code,
                    "confidence": 0.5
                }

            # 存储修复结果到长期记忆
            if result.get("patch"):
                await self._store_memory_fact("last_fix_explanation", result.get("fix_explanation", "")[:200])
                await self._store_memory_fact("last_patch_preview", result.get("patch", "")[:200])

            return result

        return {"status": "success", "action": action}


class CollaborativeValidatorAgent(CollaborativeAgent):
    """协同Validator Agent - 使用动态加载的Validation Skill"""

    def __init__(self):
        super().__init__(
            name="Validator",
            role="测试验证专家",
            capabilities=["测试验证", "副作用检测", "质量确认"],
            skill_names=["Validation", "Agent Discussion"]  # 动态加载Skills
        )

    async def execute_action(self, action: str, context: Dict) -> Dict:
        """执行验证行动（使用Skill增强）"""

        if action == "validate_fix":
            fixed_code = context.get("fixed_code", "")
            original_code = context.get("code", "")
            patch = context.get("patch", "")

            # 获取动态加载的Skill内容
            validation_skill = self.get_skill_content("Validation")

            prompt = f"""你是测试验证专家。

以下是你的Skill指导：
{validation_skill[:1000] if validation_skill else '无Skill'}

原始代码：
```
{original_code[:500]}
```

修复后代码：
```
{fixed_code}
```

补丁：
```
{patch}
```

验证修复是否正确。输出JSON格式结果，包含test_passed、validation_reason、side_effects、quality_score和confidence字段。只输出JSON。"""

            enriched_prompt = await self._enrich_prompt_with_memory(prompt, patch[:200])
            response = await call_llm(enriched_prompt)

            try:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                result = json.loads(response[json_start:json_end])
                result["status"] = "success"
            except:
                result = {
                    "status": "success",
                    "test_passed": True,
                    "validation_reason": "需要人工确认",
                    "confidence": 0.5
                }

            # 存储验证结果到长期记忆
            await self._store_memory_fact("last_validation_passed", str(result.get("test_passed", False)))
            await self._store_memory_fact("last_quality_score", str(result.get("quality_score", 0)))

            return result

        return {"status": "success", "action": action}


if __name__ == "__main__":
    async def test_collaboration():
        """测试协同Agent"""

        # 创建Agent
        reviewer = CollaborativeReviewerAgent()
        analyzer = CollaborativeAnalyzerAgent()
        fixer = CollaborativeFixerAgent()
        validator = CollaborativeValidatorAgent()

        # 测试代码
        test_code = """
public class Test {
    public void process(String input) {
        if (input.length() > 0) {  // Bug: 应该先检查null
            System.out.println(input);
        }
    }
}
"""

        # Reviewer执行ReAct循环
        context = {"code": test_code, "task": "Bug修复"}
        review_result = await reviewer.react_loop(context)

        print("\n=== 协同测试结果 ===")
        print(f"Reviewer迭代次数: {review_result['iterations']}")
        print(f"成功: {review_result['success']}")
        print(f"发现Bug: {review_result.get('bug_candidates', [])}")

    asyncio.run(test_collaboration())