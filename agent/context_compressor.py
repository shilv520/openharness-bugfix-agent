"""
Context Compressor — 上下文压缩与自动卸载
===========================================

PDF Harness Engineering: 自动 offload + summarization

三层压缩策略:
  1. Token计数 — 精确估算上下文窗口使用量
  2. 滑动窗口 — 超出阈值时，丢弃最早的消息（保留最近N条）
  3. 自动摘要 + 卸载 — 最老消息 → LLM摘要 → 存入长期记忆 → 从上下文移除

配置:
  max_context_tokens: 上下文窗口上限（默认 6000）
  reserved_output_tokens: 保留给输出的token（默认 2000）
  compression_threshold: 达到这个百分比时触发压缩（默认 0.8）
  summarization_model: 用于生成摘要的模型（默认同主模型）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("context.compressor")


# ── Token 估算 ───────────────────────────────────────────────

def estimate_tokens(text: str, model: str = "deepseek-chat") -> int:
    """估算文本的token数量。

    优先使用 tiktoken（精确），不可用时使用字符估算（近似）。
    近似公式: 中文 ~1.5 chars/token, 英文 ~4 chars/token
    """
    # 尝试 tiktoken
    try:
        import tiktoken
        # DeepSeek 兼容 OpenAI tokenizer
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except (ImportError, Exception):
        pass

    # 回退: 字符估算
    chinese_chars = len(re.findall(r'[一-鿿]', text))
    other_chars = len(text) - chinese_chars
    # 中文约 1.5 chars/token, 其他约 4 chars/token
    return int(chinese_chars / 1.5 + other_chars / 4.0)


def estimate_tokens_messages(messages: List[Dict[str, str]], model: str = "deepseek-chat") -> int:
    """估算消息列表的总token数（含role/formatting开销）"""
    total = 0
    for msg in messages:
        # 每个消息约 4 tokens 的格式开销
        total += 4
        total += estimate_tokens(msg.get("content", ""), model)
    # 对话格式总开销
    total += 2
    return total


# ── 消息摘要 ────────────────────────────────────────────────

@dataclass
class MessageDigest:
    """消息摘要"""
    content: str                        # 摘要文本
    original_count: int                 # 原文消息数
    original_tokens: int                # 原文token数
    summary_tokens: int                 # 摘要token数
    compression_ratio: float            # 压缩比 (summary/original)
    timestamp: str

    @property
    def tokens_saved(self) -> int:
        return self.original_tokens - self.summary_tokens


class ContextCompressor:
    """带自动摘要的上下文压缩器。

    对应PDF项目的自动 offload + summarization 机制。

    工作流程:
      1. 每次添加消息时检查 token 数
      2. 超过 compression_threshold → 触发压缩
      3. 将最老的 N 条消息移出 → LLM 生成摘要
      4. 摘要存储到长期记忆（可检索但不占上下文）
      5. 上下文只保留: 摘要行 + 最近消息

    用法:
        comp = ContextCompressor(max_context_tokens=6000)
        comp.add_message("user", "帮我找Bug")
        comp.add_message("assistant", "发现3个空指针风险...")
        # ... 消息持续积累 ...
        # 自动触发压缩，最老的消息被摘要 + 卸载

        messages = comp.get_messages()  # 压缩后的消息列表
    """

    def __init__(
        self,
        max_context_tokens: int = 6000,
        reserved_output_tokens: int = 2000,
        compression_threshold: float = 0.8,
        min_messages_keep: int = 4,
        summarization_model: str = "deepseek-chat",
        llm_config: Dict[str, Any] = None,
        memory=None,  # HierarchicalMemory 实例（卸载目的地）
        memory_user_id: str = "default",
    ):
        """
        Args:
            max_context_tokens: 上下文窗口上限
            reserved_output_tokens: 保留给模型输出的token
            compression_threshold: 触发压缩的使用率（0-1）
            min_messages_keep: 压缩后至少保留的消息数
            summarization_model: 生成摘要用的模型
            llm_config: LLM API配置
            memory: HierarchicalMemory实例（卸载摘要用）
            memory_user_id: 卸载时使用的user_id
        """
        self.max_tokens = max_context_tokens
        self.reserved_tokens = reserved_output_tokens
        self.available_tokens = max_context_tokens - reserved_output_tokens
        self.threshold = compression_threshold
        self.min_keep = min_messages_keep
        self.model = summarization_model
        self.llm_config = llm_config or {}
        self.memory = memory
        self.memory_user_id = memory_user_id

        # 内部状态
        self._messages: List[Dict[str, str]] = []
        self._digests: List[MessageDigest] = []   # 历史摘要
        self._total_original_tokens: int = 0
        self._compressions_count: int = 0

    # ── 公开 API ──────────────────────────────────────────────

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def current_tokens(self) -> int:
        """当前上下文token数（含摘要行）"""
        tokens = estimate_tokens_messages(self._messages, self.model)
        for d in self._digests:
            tokens += estimate_tokens(self._digest_to_line(d), self.model)
        return tokens

    @property
    def usage_ratio(self) -> float:
        """上下文使用率 (0-1)"""
        return self.current_tokens / self.available_tokens

    @property
    def needs_compression(self) -> bool:
        """是否需要压缩"""
        return self.usage_ratio >= self.threshold

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "messages": len(self._messages),
            "digests": len(self._digests),
            "current_tokens": self.current_tokens,
            "available_tokens": self.available_tokens,
            "usage_ratio": f"{self.usage_ratio:.1%}",
            "compressions": self._compressions_count,
            "total_saved_tokens": sum(d.tokens_saved for d in self._digests),
        }

    def add_message(self, role: str, content: str):
        """添加一条消息，自动检查是否需要压缩"""
        self._messages.append({"role": role, "content": content})

        if self.needs_compression:
            logger.info(
                f"[Compressor] Triggering compression: "
                f"{self.current_tokens}/{self.available_tokens} tokens "
                f"({self.usage_ratio:.1%})"
            )

    async def compress(self, force: bool = False) -> Optional[MessageDigest]:
        """执行压缩：最老消息 → 摘要 → 卸载。

        Args:
            force: 强制压缩（忽略阈值）

        Returns:
            MessageDigest or None (无需压缩)
        """
        if not force and not self.needs_compression:
            return None

        if len(self._messages) <= self.min_keep:
            logger.debug("[Compressor] Not enough messages to compress")
            return None

        # 选择要压缩的消息（最老的，保留最近 min_keep 条）
        to_compress = self._messages[:-self.min_keep]
        self._messages = self._messages[-self.min_keep:]

        if not to_compress:
            return None

        original_tokens = estimate_tokens_messages(to_compress, self.model)

        # 生成摘要
        summary_text = await self._summarize(to_compress)

        digest = MessageDigest(
            content=summary_text,
            original_count=len(to_compress),
            original_tokens=original_tokens,
            summary_tokens=estimate_tokens(summary_text, self.model),
            compression_ratio=(
                estimate_tokens(summary_text, self.model) / original_tokens
                if original_tokens > 0 else 0
            ),
            timestamp=datetime.now().isoformat(),
        )

        self._digests.append(digest)
        self._compressions_count += 1
        self._total_original_tokens += original_tokens

        # 卸载到长期记忆
        if self.memory:
            await self._offload_digest(digest, to_compress)

        logger.info(
            f"[Compressor] Compressed {len(to_compress)} messages "
            f"({original_tokens} → {digest.summary_tokens} tokens, "
            f"ratio={digest.compression_ratio:.1%}, "
            f"saved={digest.tokens_saved} tokens)"
        )

        return digest

    def get_messages(self) -> List[Dict[str, str]]:
        """获取压缩后的消息列表（含摘要行）。

        这是发送给LLM的最终消息列表。
        """
        result = []

        # 历史摘要（作为system注入）
        if self._digests:
            digest_lines = ["[以下是对之前对话的摘要]"]
            for i, d in enumerate(self._digests):
                digest_lines.append(f"摘要{i+1}: {d.content}")
            result.append({
                "role": "system",
                "content": "\n".join(digest_lines),
            })

        # 最近消息
        result.extend(self._messages)
        return result

    def get_full_context(self) -> Dict[str, Any]:
        """获取完整上下文快照（用于子Agent fork）"""
        return {
            "messages": list(self._messages),
            "digests": [
                {
                    "content": d.content,
                    "original_count": d.original_count,
                    "original_tokens": d.original_tokens,
                    "timestamp": d.timestamp,
                }
                for d in self._digests
            ],
            "stats": self.stats,
        }

    def restore_from_snapshot(self, snapshot: Dict[str, Any]):
        """从快照恢复上下文（用于子Agent merge回父Agent）"""
        self._messages = snapshot.get("messages", [])
        self._digests = [
            MessageDigest(
                content=d["content"],
                original_count=d.get("original_count", 0),
                original_tokens=d.get("original_tokens", 0),
                summary_tokens=estimate_tokens(d["content"], self.model),
                compression_ratio=0,
                timestamp=d.get("timestamp", ""),
            )
            for d in snapshot.get("digests", [])
        ]

    # ── 内部方法 ──────────────────────────────────────────────

    async def _summarize(self, messages: List[Dict[str, str]]) -> str:
        """调用LLM生成消息摘要"""
        # 构建摘要 prompt
        conversation = "\n".join([
            f"{m['role']}: {m['content'][:300]}"
            for m in messages
        ])

        summary_prompt = f"""请用中文对以下对话片段进行摘要（最多200字），保留关键信息:
- 讨论的主要话题
- 达成的结论或决策
- 重要的发现（如Bug类型、修复方案等）
- 待解决的问题

对话:
{conversation}

摘要:"""

        try:
            response = await self._call_llm([
                {"role": "system", "content": "你是一个专业的对话摘要器。请简洁准确地摘要对话内容。"},
                {"role": "user", "content": summary_prompt},
            ])
            return response.strip()
        except Exception as e:
            logger.warning(f"[Compressor] Summarization failed: {e}")
            # 回退: 简单拼接每句话的前50个字符
            fallback = "; ".join([
                m['content'][:50] + "..."
                for m in messages[:3]
            ])
            return f"[摘要生成失败] {fallback}"

    async def _offload_digest(
        self,
        digest: MessageDigest,
        original_messages: List[Dict[str, str]],
    ):
        """将摘要卸载到长期记忆"""
        try:
            # 存储摘要本身
            await self.memory.remember_fact(
                self.memory_user_id,
                f"ctx_summary_{digest.timestamp}",
                digest.content,
            )
            # 存储原始消息数（作为元数据）
            await self.memory.remember_fact(
                self.memory_user_id,
                f"ctx_summary_{digest.timestamp}_meta",
                json.dumps({
                    "original_count": digest.original_count,
                    "original_tokens": digest.original_tokens,
                    "compression_ratio": digest.compression_ratio,
                }, ensure_ascii=False),
            )
            logger.debug(f"[Compressor] Offloaded digest to memory: {digest.timestamp}")
        except Exception as e:
            logger.warning(f"[Compressor] Offload failed: {e}")

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """调用LLM API"""
        import os

        api_key = (
            self.llm_config.get("api_key") or
            self.llm_config.get("openai_api_key") or
            os.environ.get("DEEPSEEK_API_KEY") or
            os.environ.get("OPENAI_API_KEY")
        )
        base_url = (
            self.llm_config.get("base_url") or
            self.llm_config.get("openai_base_url", "https://api.deepseek.com/v1")
        )
        model = self.llm_config.get("model", self.model)

        import aiohttp
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 300,
            "temperature": 0.3,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"LLM API error {resp.status}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

    def _digest_to_line(self, digest: MessageDigest) -> str:
        return f"[历史摘要] {digest.content}"


# ── 便捷函数 ─────────────────────────────────────────────────

def create_compressor(
    max_tokens: int = 6000,
    llm_config: Dict = None,
    memory=None,
) -> ContextCompressor:
    """创建预配置的压缩器"""
    return ContextCompressor(
        max_context_tokens=max_tokens,
        reserved_output_tokens=2000,
        compression_threshold=0.8,
        min_messages_keep=4,
        llm_config=llm_config,
        memory=memory,
    )
