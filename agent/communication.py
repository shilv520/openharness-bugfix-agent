"""
Agent通信机制 - Multi-Agent协同核心
=====================================

实现Agent间消息传递、讨论、反馈
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json

logger = logging.getLogger("agent-communication")


@dataclass
class AgentMessage:
    """Agent消息"""
    from_agent: str           # 发送者
    to_agent: str             # 接收者
    message_type: str         # message类型: discuss/feedback/request/response
    content: str              # 消息内容
    data: Dict = field(default_factory=dict)  # 附带数据
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    confidence: float = 1.0   # 置信度


class AgentCommunicationBus:
    """
    Agent通信总线

    实现Agent间消息传递：
    - Reviewer ↔ Analyzer 讨论
    - Fixer ↔ Validator 反馈
    - 广播消息
    """

    def __init__(self):
        self.message_queue: Dict[str, List[AgentMessage]] = {}  # 每个Agent的消息队列
        self.conversation_history: List[AgentMessage] = []      # 所有对话历史
        self.agents: Dict[str, Callable] = {}                   # 注册的Agent
        self.shared_context: Dict = {}                          # 共享上下文

        logger.info("[CommunicationBus] 初始化完成")

    def register_agent(self, agent_name: str, agent_handler: Callable):
        """注册Agent"""
        self.agents[agent_name] = agent_handler
        self.message_queue[agent_name] = []
        logger.info(f"[CommunicationBus] 注册Agent: {agent_name}")

    def send_message(self, message: AgentMessage) -> bool:
        """发送消息"""
        if message.to_agent not in self.message_queue:
            logger.warning(f"[CommunicationBus] 未知的接收者: {message.to_agent}")
            return False

        self.message_queue[message.to_agent].append(message)
        self.conversation_history.append(message)

        logger.info(f"[CommunicationBus] {message.from_agent} → {message.to_agent}: {message.message_type}")
        return True

    def broadcast(self, from_agent: str, message_type: str, content: str, data: Dict = None) -> bool:
        """广播消息给所有Agent"""
        for agent_name in self.agents:
            if agent_name != from_agent:
                msg = AgentMessage(
                    from_agent=from_agent,
                    to_agent=agent_name,
                    message_type=message_type,
                    content=content,
                    data=data or {}
                )
                self.send_message(msg)

        logger.info(f"[CommunicationBus] {from_agent} 广播: {message_type}")
        return True

    def get_messages(self, agent_name: str) -> List[AgentMessage]:
        """获取Agent的所有待处理消息"""
        messages = self.message_queue.get(agent_name, [])
        self.message_queue[agent_name] = []  # 清空队列
        return messages

    def get_conversation_history(self, agent1: str, agent2: str) -> List[AgentMessage]:
        """获取两个Agent之间的对话历史"""
        return [
            msg for msg in self.conversation_history
            if (msg.from_agent == agent1 and msg.to_agent == agent2) or
               (msg.from_agent == agent2 and msg.to_agent == agent1)
        ]

    def update_shared_context(self, key: str, value: any):
        """更新共享上下文"""
        self.shared_context[key] = value
        logger.info(f"[CommunicationBus] 更新共享上下文: {key}")

    def get_shared_context(self) -> Dict:
        """获取共享上下文"""
        return self.shared_context


class DiscussionProtocol:
    """
    Agent讨论协议

    实现Agent间协作讨论：
    - Reviewer发现Bug → 与Analyzer讨论确认
    - Analyzer分析 → 与Fixer讨论修复策略
    """

    def __init__(self, comm_bus: AgentCommunicationBus, llm_caller: Callable):
        self.comm_bus = comm_bus
        self.llm_caller = llm_caller  # LLM调用函数
        self.max_rounds = 3            # 最大讨论轮数

        logger.info("[DiscussionProtocol] 初始化完成")

    async def discuss(self, agent1: str, agent2: str, topic: str, initial_data: Dict) -> Dict:
        """
        两个Agent之间的讨论

        流程：
        1. agent1提出观点
        2. agent2回应/质疑
        3. agent1再回应
        4. 达成共识或保留分歧

        Returns:
            {
                "agreed": True/False,        # 是否达成共识
                "consensus": {...},          # 共识结论
                "disagreement": [...],       # 分歧点
                "discussion_history": [...]  # 讨论记录
            }
        """
        logger.info(f"[Discussion] {agent1} ↔ {agent2} 开始讨论: {topic}")

        discussion_history = []
        round_num = 0
        agreed = False
        consensus = {}
        disagreements = []

        # 第一轮：agent1提出观点
        round_num += 1
        agent1_view = initial_data.get("agent1_view", "")

        msg1 = AgentMessage(
            from_agent=agent1,
            to_agent=agent2,
            message_type="discuss",
            content=f"我的分析结果是：{agent1_view}",
            data=initial_data
        )
        self.comm_bus.send_message(msg1)
        discussion_history.append(msg1)

        # agent2回应
        agent2_response = await self._generate_response(agent2, agent1, msg1, topic)
        discussion_history.append(agent2_response)

        # 检查是否达成共识
        if agent2_response.data.get("agree", False):
            agreed = True
            consensus = agent2_response.data.get("consensus", initial_data)
            logger.info(f"[Discussion] 第{round_num}轮达成共识")

        # 如果未达成共识，继续讨论
        while not agreed and round_num < self.max_rounds:
            round_num += 1

            # agent1回应agent2的质疑
            agent1_response = await self._generate_response(agent1, agent2, agent2_response, topic)
            discussion_history.append(agent1_response)

            # 检查共识
            if agent1_response.data.get("agree", False):
                agreed = True
                consensus = agent1_response.data.get("consensus", {})
                break

            # agent2再次回应
            agent2_response = await self._generate_response(agent2, agent1, agent1_response, topic)
            discussion_history.append(agent2_response)

            if agent2_response.data.get("agree", False):
                agreed = True
                consensus = agent2_response.data.get("consensus", {})
                break

            # 记录分歧
            disagreements.append(agent2_response.data.get("disagreement_point", ""))

        result = {
            "agreed": agreed,
            "consensus": consensus,
            "disagreements": disagreements,
            "rounds": round_num,
            "discussion_history": [
                {"from": msg.from_agent, "to": msg.to_agent, "content": msg.content, "data": msg.data}
                for msg in discussion_history
            ]
        }

        logger.info(f"[Discussion] 结束，共识: {agreed}, 轮数: {round_num}")
        return result

    async def _generate_response(self, responder: str, initiator: str, received_msg: AgentMessage, topic: str) -> AgentMessage:
        """使用LLM生成回应"""

        # 构建prompt
        prompt = f"""你是{responder} Agent，正在与{initiator} Agent讨论：{topic}

{initiator}的观点：
{received_msg.content}

附带数据：
{json.dumps(received_msg.data, indent=2, ensure_ascii=False)}

请分析{initiator}的观点，并决定：
1. 是否同意？如果同意，说明理由
2. 如果不同意，指出分歧点，并提出你的观点
3. 给出你的置信度（0-1）

输出JSON格式结果，包含agree、reason、my_view、consensus、disagreement_point和confidence字段。只输出JSON，不要其他内容。"""

        # 调用LLM
        response_text = await self.llm_caller(prompt)

        # 解析JSON
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                response_data = json.loads(response_text[json_start:json_end])
            else:
                response_data = {"agree": False, "reason": "无法解析", "confidence": 0.5}
        except:
            response_data = {"agree": False, "reason": "解析失败", "confidence": 0.5}

        # 创建回应消息
        content = f"我的回应：{response_data.get('reason', '')}"

        return AgentMessage(
            from_agent=responder,
            to_agent=initiator,
            message_type="discuss",
            content=content,
            data=response_data,
            confidence=response_data.get("confidence", 0.5)
        )


class FeedbackProtocol:
    """
    Agent反馈协议

    实现Fixer ↔ Validator反馈循环：
    - Validator验证 → 反馈给Fixer
    - Fixer根据反馈调整补丁
    """

    def __init__(self, comm_bus: AgentCommunicationBus, llm_caller: Callable):
        self.comm_bus = comm_bus
        self.llm_caller = llm_caller
        self.max_iterations = 3  # 最大反馈迭代次数

        logger.info("[FeedbackProtocol] 初始化完成")

    async def feedback_loop(self, fixer: str, validator: str, initial_patch: Dict) -> Dict:
        """
        Fixer ↔ Validator反馈循环

        流程：
        1. Fixer提交补丁
        2. Validator验证
        3. Validator反馈（如果失败）
        4. Fixer调整补丁
        5. 重复直到成功或达到最大次数

        Returns:
            {
                "success": True/False,
                "final_patch": {...},
                "iterations": N,
                "feedback_history": [...]
            }
        """
        logger.info(f"[Feedback] {fixer} ↔ {validator} 开始反馈循环")

        current_patch = initial_patch
        iteration = 0
        success = False
        feedback_history = []

        while iteration < self.max_iterations and not success:
            iteration += 1

            # Fixer提交补丁给Validator
            submit_msg = AgentMessage(
                from_agent=fixer,
                to_agent=validator,
                message_type="submit_patch",
                content=f"提交补丁（第{iteration}版）",
                data=current_patch
            )
            self.comm_bus.send_message(submit_msg)
            feedback_history.append(submit_msg)

            # Validator验证并反馈
            validation_result = await self._validate_and_feedback(validator, fixer, current_patch)
            feedback_history.append(validation_result)

            # 检查验证结果
            if validation_result.data.get("passed", False):
                success = True
                current_patch["validation_passed"] = True
                logger.info(f"[Feedback] 第{iteration}次迭代成功")
                break

            # 如果失败，Fixer根据反馈调整
            if iteration < self.max_iterations:
                adjusted_patch = await self._adjust_patch(fixer, validator, validation_result, current_patch)
                current_patch = adjusted_patch
                feedback_history.append(AgentMessage(
                    from_agent=fixer,
                    to_agent=validator,
                    message_type="adjusted_patch",
                    content="根据反馈调整补丁",
                    data=current_patch
                ))

        result = {
            "success": success,
            "final_patch": current_patch,
            "iterations": iteration,
            "feedback_history": [
                {"from": msg.from_agent, "to": msg.to_agent, "content": msg.content, "data": msg.data}
                for msg in feedback_history
            ]
        }

        logger.info(f"[Feedback] 结束，成功: {success}, 迭代次数: {iteration}")
        return result

    async def _validate_and_feedback(self, validator: str, fixer: str, patch: Dict) -> AgentMessage:
        """Validator验证并生成反馈"""

        prompt = f"""你是{validator} Agent，正在验证{fixer} Agent提交的补丁。

补丁内容：
{json.dumps(patch, indent=2, ensure_ascii=False)}

请验证补丁是否正确，输出JSON：
{
  "passed": true/false,
  "issues": ["问题1", "问题2"],  // 如果失败，列出问题
  "suggestions": ["建议1", "建议2"],  // 给Fixer的建议
  "confidence": 0.X
}

只输出JSON。"""

        response_text = await self.llm_caller(prompt)

        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response_text[json_start:json_end])
            else:
                result = {"passed": False, "issues": ["解析失败"], "confidence": 0.5}
        except:
            result = {"passed": False, "issues": ["解析失败"], "confidence": 0.5}

        content = f"验证结果：{'通过' if result.get('passed') else '失败'}"
        if result.get("issues"):
            content += f"，问题：{result['issues']}"

        return AgentMessage(
            from_agent=validator,
            to_agent=fixer,
            message_type="feedback",
            content=content,
            data=result,
            confidence=result.get("confidence", 0.5)
        )

    async def _adjust_patch(self, fixer: str, validator: str, feedback: AgentMessage, original_patch: Dict) -> Dict:
        """Fixer根据反馈调整补丁"""

        prompt = f"""你是{fixer} Agent，{validator} Agent给了你反馈：

反馈内容：
{feedback.content}

详细数据：
{json.dumps(feedback.data, indent=2, ensure_ascii=False)}

原始补丁：
{json.dumps(original_patch, indent=2, ensure_ascii=False)}

请根据反馈调整补丁，输出JSON：
{
  "patch": "调整后的补丁内容",
  "fixed_code": "调整后的完整代码",
  "adjustment_reason": "为什么这样调整",
  "confidence": 0.X
}

只输出JSON。"""

        response_text = await self.llm_caller(prompt)

        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                adjusted = json.loads(response_text[json_start:json_end])
            else:
                adjusted = original_patch
        except:
            adjusted = original_patch

        adjusted["adjusted"] = True
        return adjusted


# ============================================
# 全局通信总线（单例）
# ============================================

_global_comm_bus = None


def get_comm_bus() -> AgentCommunicationBus:
    """获取全局通信总线"""
    global _global_comm_bus
    if _global_comm_bus is None:
        _global_comm_bus = AgentCommunicationBus()
    return _global_comm_bus


def reset_comm_bus():
    """重置通信总线"""
    global _global_comm_bus
    _global_comm_bus = AgentCommunicationBus()


if __name__ == "__main__":
    # 测试通信机制
    bus = AgentCommunicationBus()

    # 注册Agent
    bus.register_agent("Reviewer", lambda x: x)
    bus.register_agent("Analyzer", lambda x: x)

    # 发送消息
    msg = AgentMessage(
        from_agent="Reviewer",
        to_agent="Analyzer",
        message_type="discuss",
        content="我发现了一个Bug",
        data={"location": "Line 5", "type": "null_pointer"}
    )
    bus.send_message(msg)

    # 获取消息
    messages = bus.get_messages("Analyzer")
    print(f"Analyzer收到消息: {len(messages)}条")
    for m in messages:
        print(f"  {m.from_agent}: {m.content}")