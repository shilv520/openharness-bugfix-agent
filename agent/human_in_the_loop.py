"""
Human-in-the-Loop — 双层中断机制
================================

PDF Harness Engineering: 双层人工介入

第 1 层: 数据补充 — request_order_info 工具内部 interrupt()
  当 Agent 需要人类提供缺失数据时（如 Bug 复现步骤、环境版本号），
  触发中断 → 等待人类输入 → 数据注入上下文 → 继续执行

第 2 层: 最终审批 — interrupt_on 配置拦截关键操作
  在执行不可逆操作前（如写入代码仓库、执行数据库迁移），
  暂停等待人类审批 → approved/rejected → 继续或回退

用法:
  from agent.human_in_the_loop import HITLManager, ApprovalRequest

  hitl = HITLManager(config=HITLConfig(
      interrupt_before=["fixer", "executor"],
      interrupt_on_tools=["write_file", "git_commit"],
      approval_timeout=300,
  ))

  # Layer 1: 数据补充
  async for event in hitl.request_info("请提供 Bug 的复现步骤"):
      if event.type == "interrupt":
          yield event  # 等待用户输入

  # Layer 2: 审批
  async for event in hitl.request_approval(
      action="write_file", detail={"path": "/workspace/main.py"}
  ):
      if event.type == "interrupt":
          yield event  # 等待审批
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

logger = logging.getLogger("human_in_the_loop")


# ── 枚举 ────────────────────────────────────────────────────

class InterruptType(Enum):
    APPROVAL = "approval"            # 第 2 层: 审批中断
    DATA_REQUEST = "data_request"    # 第 1 层: 数据补充中断
    CONFIRMATION = "confirmation"    # 确认（是/否）

class ApprovalDecision(Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    MODIFIED = "modified"            # 带修改批准


# ── 数据类 ──────────────────────────────────────────────────

@dataclass
class HITLConfig:
    """HITL 配置"""
    # 第 1 层: 数据补充工具
    data_tools: List[str] = field(default_factory=lambda: [
        "request_missing_info",
        "request_reproduction_steps",
        "request_env_version",
    ])

    # 第 2 层: 审批拦截
    interrupt_before: List[str] = field(default_factory=list)   # 在这些节点前中断
    interrupt_after: List[str] = field(default_factory=list)    # 在这些节点后中断
    interrupt_on_tools: List[str] = field(default_factory=lambda: [
        "write_file",
        "git_commit",
        "apply_patch",
    ])

    # 超时与回退
    approval_timeout: int = 300         # 审批超时（秒），超时自动拒绝
    auto_approve_safe_ops: bool = True  # 安全操作自动批准
    safe_tools: List[str] = field(default_factory=lambda: [
        "read_file", "grep_search", "glob_search",
    ])

    # 审批策略
    require_approval_for_severity: List[str] = field(default_factory=lambda: [
        "HIGH", "CRITICAL"
    ])

    # 持久化（审批记录可审计）
    log_approvals: bool = True
    approval_log_path: str = "data/approval_log.jsonl"


@dataclass
class ApprovalRequest:
    """审批请求"""
    request_id: str
    request_type: InterruptType
    action: str                          # 工具名 / 节点名
    detail: Dict[str, Any]               # 请求详情
    severity: str = "MEDIUM"             # HIGH/CRITICAL 必须审批
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    timeout: int = 300

    # 上下文快照
    current_state: Dict[str, Any] = field(default_factory=dict)
    conversation_summary: str = ""


@dataclass
class ApprovalResponse:
    """审批响应"""
    request_id: str
    decision: ApprovalDecision
    comment: str = ""
    modifications: Dict[str, Any] = field(default_factory=dict)
    responder: str = "human"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DataRequest:
    """数据补充请求（第 1 层）"""
    request_id: str
    tool_name: str                       # 触发中断的工具名
    question: str                        # 向人类提出的问题
    expected_format: str                 # 期望的输入格式
    context: Dict[str, Any]              # 上下文（人类需要知道什么）
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DataResponse:
    """数据补充响应（第 1 层）"""
    request_id: str
    answer: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class HITLEvent:
    """HITL 事件（流式输出）"""
    event_type: str                      # "interrupt" | "resume" | "timeout" | "approved" | "rejected"
    request: Optional[ApprovalRequest] = None
    data_request: Optional[DataRequest] = None
    response: Optional[ApprovalResponse] = None
    data_response: Optional[DataResponse] = None
    message: str = ""


# ── HITL Manager ─────────────────────────────────────────────

class HITLManager:
    """双层中断管理器。

    对应 PDF 项目的双层人机协同：
    - 第 1 层: request_* 工具触发数据补充
    - 第 2 层: interrupt_on 配置拦截关键操作

    支持 LangGraph 的 interrupt() 集成和独立使用。

    用法:
        hitl = HITLManager(HITLConfig(
            interrupt_before=["fixer", "validator"],
            interrupt_on_tools=["write_file", "git_commit"],
        ))

        # 注册到 LangGraph 的 interrupt_before
        graph = builder.compile(
            checkpointer=checkpointer,
            interrupt_before=hitl.get_interrupt_before_nodes(),
        )

        # 工具中使用 Layer 1 中断
        async for event in hitl.data_interrupt(
            tool_name="request_missing_info",
            question="请提供 Bug 的复现步骤和触发条件"
        ):
            if event.event_type == "interrupt":
                yield event  # 等待人类输入
    """

    def __init__(
        self,
        config: HITLConfig = None,
        approval_callback: Callable = None,
        data_callback: Callable = None,
    ):
        """
        Args:
            config: HITL 配置
            approval_callback: 审批回调（可选，用于自定义审批逻辑）
            data_callback: 数据补充回调（可选，用于自定义数据收集）
        """
        self.config = config or HITLConfig()
        self._approval_callback = approval_callback
        self._data_callback = data_callback

        # 审批历史（可审计）
        self._approval_log: List[Dict[str, Any]] = []

        # 活跃的请求
        self._pending_approvals: Dict[str, ApprovalRequest] = {}
        self._pending_data_requests: Dict[str, DataRequest] = {}

        # 响应队列（用于异步等待）
        self._response_queues: Dict[str, asyncio.Queue] = {}

    # ── 公开 API：LangGraph 集成 ─────────────────────────────

    def get_interrupt_before_nodes(self) -> List[str]:
        """获取需要在之前中断的节点列表（供 LangGraph compile 使用）"""
        return list(self.config.interrupt_before)

    def get_interrupt_after_nodes(self) -> List[str]:
        """获取需要在之后中断的节点列表"""
        return list(self.config.interrupt_after)

    def should_interrupt_tool(self, tool_name: str, tool_input: Dict = None) -> bool:
        """判断工具调用是否需要审批中断"""
        if tool_name in self.config.safe_tools and self.config.auto_approve_safe_ops:
            return False
        return tool_name in self.config.interrupt_on_tools

    def should_interrupt_node(self, node_name: str) -> bool:
        """判断节点执行前是否需要中断"""
        return node_name in self.config.interrupt_before

    # ── 公开 API：第 1 层 - 数据补充 ─────────────────────────

    async def data_interrupt(
        self,
        tool_name: str,
        question: str,
        expected_format: str = "free_text",
        context: Dict[str, Any] = None,
        timeout: int = 300,
    ) -> AsyncIterator[HITLEvent]:
        """第 1 层中断：向人类请求数据补充。

        用在 Agent 工具内部，当 Agent 发现缺少关键数据时调用。

        Args:
            tool_name: 触发中断的工具名（如 'request_missing_info'）
            question: 向人类提出的问题
            expected_format: 期望的回答格式
            context: 给人类的上下文
            timeout: 超时秒数

        Yields:
            HITLEvent: 先 yield interrupt 事件，等待人类响应后 yield resume 事件
        """
        request = DataRequest(
            request_id=f"data-{uuid.uuid4().hex[:12]}",
            tool_name=tool_name,
            question=question,
            expected_format=expected_format,
            context=context or {},
        )

        self._pending_data_requests[request.request_id] = request
        queue = asyncio.Queue()
        self._response_queues[request.request_id] = queue

        logger.info(f"[HITL:L1] Data request: {tool_name} — {question[:80]}")

        # 发送中断事件
        yield HITLEvent(
            event_type="interrupt",
            data_request=request,
            message=f"🛑 [数据补充] {question}",
        )

        # 等待人类响应（或超时）
        try:
            response: DataResponse = await asyncio.wait_for(
                queue.get(), timeout=timeout
            )
            logger.info(f"[HITL:L1] Data response: {response.answer[:80]}")
            yield HITLEvent(
                event_type="resume",
                data_request=request,
                data_response=response,
                message=f"✓ [数据已补充] {response.answer[:80]}",
            )
        except asyncio.TimeoutError:
            logger.warning(f"[HITL:L1] Data request timed out: {request.request_id}")
            yield HITLEvent(
                event_type="timeout",
                data_request=request,
                message=f"⏰ [数据请求超时] {question[:80]}",
            )
        finally:
            self._pending_data_requests.pop(request.request_id, None)
            self._response_queues.pop(request.request_id, None)

    def submit_data_response(self, request_id: str, answer: str, metadata: Dict = None):
        """人类提交数据补充响应"""
        if request_id in self._response_queues:
            response = DataResponse(
                request_id=request_id,
                answer=answer,
                metadata=metadata or {},
            )
            self._response_queues[request_id].put_nowait(response)
        else:
            logger.warning(f"[HITL] No pending data request: {request_id}")

    # ── 公开 API：第 2 层 - 审批 ─────────────────────────────

    async def approval_interrupt(
        self,
        action: str,
        detail: Dict[str, Any],
        severity: str = "MEDIUM",
        current_state: Dict[str, Any] = None,
        timeout: int = None,
    ) -> AsyncIterator[HITLEvent]:
        """第 2 层中断：在执行关键操作前请求人类审批。

        用在 Agent 执行关键操作前（如写入文件、提交代码）。

        Args:
            action: 操作名（如 'write_file', 'git_commit'）
            detail: 操作详情
            severity: 严重级别（HIGH/CRITICAL 必须审批）
            current_state: 当前 Agent 状态快照
            timeout: 超时秒数

        Yields:
            HITLEvent: interrupt → (wait for human) → approved/rejected
        """
        timeout = timeout or self.config.approval_timeout

        # 如果不需要审批，自动通过
        if severity not in self.config.require_approval_for_severity:
            if self.config.auto_approve_safe_ops and action in self.config.safe_tools:
                yield HITLEvent(
                    event_type="approved",
                    request=ApprovalRequest(
                        request_id=f"auto-{uuid.uuid4().hex[:8]}",
                        request_type=InterruptType.APPROVAL,
                        action=action,
                        detail=detail,
                        severity=severity,
                    ),
                    message=f"✓ [自动批准] {action} (安全操作)",
                )
                return

        request = ApprovalRequest(
            request_id=f"approval-{uuid.uuid4().hex[:12]}",
            request_type=InterruptType.APPROVAL,
            action=action,
            detail=detail,
            severity=severity,
            timeout=timeout,
            current_state=current_state or {},
        )

        self._pending_approvals[request.request_id] = request
        queue = asyncio.Queue()
        self._response_queues[request.request_id] = queue

        logger.info(f"[HITL:L2] Approval requested: {action} [{severity}]")

        # 发送中断事件
        yield HITLEvent(
            event_type="interrupt",
            request=request,
            message=f"🛑 [审批请求] {action} [{severity}]\n{json.dumps(detail, ensure_ascii=False, indent=2)[:500]}",
        )

        # 等待审批决策
        try:
            response: ApprovalResponse = await asyncio.wait_for(
                queue.get(), timeout=timeout
            )
            decision_event_type = response.decision.value
            logger.info(f"[HITL:L2] Approval decision: {response.decision.value} — {response.comment[:80]}")

            # 记录审批日志
            if self.config.log_approvals:
                self._log_approval(request, response)

            yield HITLEvent(
                event_type=decision_event_type,
                request=request,
                response=response,
                message=f"{'✓' if response.decision == ApprovalDecision.APPROVED else '✗'} [{response.decision.value}] {response.comment[:80]}",
            )
        except asyncio.TimeoutError:
            # 超时自动拒绝
            logger.warning(f"[HITL:L2] Approval timed out: {request.request_id}")
            auto_response = ApprovalResponse(
                request_id=request.request_id,
                decision=ApprovalDecision.REJECTED,
                comment="审批超时，自动拒绝",
            )
            if self.config.log_approvals:
                self._log_approval(request, auto_response)
            yield HITLEvent(
                event_type="timeout",
                request=request,
                response=auto_response,
                message="⏰ [审批超时] 自动拒绝",
            )
        finally:
            self._pending_approvals.pop(request.request_id, None)
            self._response_queues.pop(request.request_id, None)

    def submit_approval(
        self,
        request_id: str,
        decision: str,
        comment: str = "",
        modifications: Dict = None,
    ):
        """人类提交审批决策"""
        if request_id in self._response_queues:
            try:
                dec = ApprovalDecision(decision)
            except ValueError:
                dec = ApprovalDecision.REJECTED
            response = ApprovalResponse(
                request_id=request_id,
                decision=dec,
                comment=comment,
                modifications=modifications or {},
            )
            self._response_queues[request_id].put_nowait(response)
        else:
            logger.warning(f"[HITL] No pending approval: {request_id}")

    # ── 公开 API：确认（是/否）───────────────────────────────

    async def confirm(
        self,
        question: str,
        timeout: int = 120,
    ) -> AsyncIterator[HITLEvent]:
        """简单的是/否确认中断"""
        request = ApprovalRequest(
            request_id=f"confirm-{uuid.uuid4().hex[:12]}",
            request_type=InterruptType.CONFIRMATION,
            action="confirm",
            detail={"question": question},
            severity="MEDIUM",
            timeout=timeout,
        )

        queue = asyncio.Queue()
        self._response_queues[request.request_id] = queue

        yield HITLEvent(
            event_type="interrupt",
            request=request,
            message=f"❓ [确认] {question}",
        )

        try:
            response = await asyncio.wait_for(queue.get(), timeout=timeout)
            yield HITLEvent(
                event_type=response.decision.value,
                request=request,
                response=response,
                message=f"{'✓' if response.decision == ApprovalDecision.APPROVED else '✗'} {response.comment}",
            )
        except asyncio.TimeoutError:
            yield HITLEvent(
                event_type="timeout",
                request=request,
                message="⏰ [确认超时]",
            )
        finally:
            self._response_queues.pop(request.request_id, None)

    def answer_confirm(self, request_id: str, approved: bool, comment: str = ""):
        """回答确认请求"""
        self.submit_approval(
            request_id,
            "approved" if approved else "rejected",
            comment,
        )

    # ── 公开 API：LangGraph interrupt() 封装 ─────────────────

    def create_langgraph_interrupt_data(
        self,
        request: ApprovalRequest,
    ) -> Dict[str, Any]:
        """创建 LangGraph interrupt() 可用的数据"""
        return {
            "hitl_request_id": request.request_id,
            "hitl_type": request.request_type.value,
            "hitl_action": request.action,
            "hitl_detail": request.detail,
            "hitl_severity": request.severity,
        }

    def parse_langgraph_resume_data(
        self,
        resume_data: Dict[str, Any],
    ) -> ApprovalResponse:
        """解析 LangGraph Command(resume=...) 的数据"""
        return ApprovalResponse(
            request_id=resume_data.get("request_id", ""),
            decision=ApprovalDecision(resume_data.get("decision", "approved")),
            comment=resume_data.get("comment", ""),
            modifications=resume_data.get("modifications", {}),
        )

    # ── 查询 API ─────────────────────────────────────────────

    def get_pending_approvals(self) -> List[ApprovalRequest]:
        """获取所有待审批的请求"""
        return list(self._pending_approvals.values())

    def get_pending_data_requests(self) -> List[DataRequest]:
        """获取所有待数据补充的请求"""
        return list(self._pending_data_requests.values())

    def get_approval_log(self, limit: int = 100) -> List[Dict]:
        """获取审批历史"""
        return self._approval_log[-limit:]

    def get_hitl_summary(self) -> Dict[str, Any]:
        """获取 HITL 状态摘要"""
        return {
            "pending_approvals": len(self._pending_approvals),
            "pending_data_requests": len(self._pending_data_requests),
            "total_approvals_logged": len(self._approval_log),
            "config": {
                "interrupt_before": self.config.interrupt_before,
                "interrupt_on_tools": self.config.interrupt_on_tools,
                "auto_approve_safe_ops": self.config.auto_approve_safe_ops,
            },
        }

    # ── 内部 ─────────────────────────────────────────────────

    def _log_approval(self, request: ApprovalRequest, response: ApprovalResponse):
        """记录审批日志"""
        entry = {
            "request_id": request.request_id,
            "action": request.action,
            "severity": request.severity,
            "decision": response.decision.value,
            "comment": response.comment,
            "request_time": request.timestamp,
            "response_time": response.timestamp,
        }
        self._approval_log.append(entry)


# ── 预配置的 HITL 工具函数 ─────────────────────────────────

def create_request_missing_info_tool(hitl_manager: HITLManager):
    """创建 'request_missing_info' 工具（第 1 层中断工具）。

    这个工具可以被注册到 Agent 的工具列表中，
    当 Agent 需要人类提供缺失信息时调用。
    """
    async def request_missing_info(question: str, expected_format: str = "free_text") -> str:
        """向人类请求缺失的信息。当代码审查需要更多上下文时使用。"""
        result = None
        async for event in hitl_manager.data_interrupt(
            tool_name="request_missing_info",
            question=question,
            expected_format=expected_format,
        ):
            if event.event_type == "interrupt":
                # 在实际 LangGraph 中，这里会调用 interrupt()
                # 在独立使用中，返回中断事件给调用者
                pass
            elif event.event_type == "resume" and event.data_response:
                result = event.data_response.answer
        return result or "未收到回答"

    request_missing_info.__doc__ = "向人类请求缺失的信息。当分析需要更多上下文时使用此工具。"
    return request_missing_info


def create_request_approval_tool(hitl_manager: HITLManager):
    """创建 'request_approval' 工具（第 2 层中断工具）。"""
    async def request_approval(action: str, detail_json: str, severity: str = "MEDIUM") -> str:
        """在执行不可逆操作前请求人工审批。"""
        detail = json.loads(detail_json) if isinstance(detail_json, str) else detail_json
        result = None
        async for event in hitl_manager.approval_interrupt(
            action=action, detail=detail, severity=severity,
        ):
            if event.event_type in ("approved", "rejected", "timeout"):
                result = event.event_type
        return result or "timeout"

    request_approval.__doc__ = "在执行不可逆操作（写文件、提交代码）前请求人工审批。"
    return request_approval


# ── 工厂函数 ─────────────────────────────────────────────────

def create_hitl_manager(
    interrupt_before: List[str] = None,
    interrupt_on_tools: List[str] = None,
    approval_timeout: int = 300,
) -> HITLManager:
    """创建预配置的 HITL 管理器"""
    config = HITLConfig(
        interrupt_before=interrupt_before or ["fixer", "validator"],
        interrupt_on_tools=interrupt_on_tools or ["write_file", "git_commit", "apply_patch"],
        approval_timeout=approval_timeout,
    )
    return HITLManager(config=config)
