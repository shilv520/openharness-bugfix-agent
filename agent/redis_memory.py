"""
Redis + ChromaDB 分层记忆机制
============================

三层记忆架构：
1. 短期记忆（Short-term Memory）
   - 存储当前会话上下文
   - Redis String，会话结束后清除
   - 轻量化，快速访问

2. 长期记忆（Long-term Memory）
   - 异步写入 ChromaDB 向量库（真正的向量数据库）
   - Redis Hash 持久化事实内容
   - 支持跨会话回忆

3. 记忆检索
   - ChromaDB 向量相似度匹配
   - distance 阈值保精度
   - 只返回相关性高的记忆

配合 LangGraph 状态管理，实现记忆无缝衔接。
"""

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path

# Logger 必须先定义
logger = logging.getLogger("redis-memory")
logging.basicConfig(level=logging.INFO)

# ChromaDB 向量库
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
    logger.info("[ChromaDB] 模块可用")
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("[ChromaDB] 未安装，使用内存存储")

# ============================================
# Redis 连接
# ============================================

try:
    import redis
    REDIS_AVAILABLE = True
    logger.info("[Redis] 模块可用")
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("[Redis] 未安装，使用内存存储（开发模式）")

# API Base URL
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/")

# Embeddings API 配置（与聊天模型分开）
# DeepSeek 不提供 embeddings API，需要单独配置 OpenAI embeddings 或跳过
EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL", "")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

logger.info(f"[Embedding] API URL: {EMBEDDING_API_URL}")


class RedisConnection:
    """Redis 连接管理"""

    def __init__(self, host: str = None, port: int = None, password: str = None):
        self.host = host or os.environ.get("REDIS_HOST", "localhost")
        self.port = port or int(os.environ.get("REDIS_PORT", 6379))
        self.password = password or os.environ.get("REDIS_PASSWORD", None)
        self.client = None

        if REDIS_AVAILABLE:
            try:
                self.client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    password=self.password,
                    decode_responses=True,
                    socket_connect_timeout=5
                )
                self.client.ping()
                logger.info(f"[Redis] 连接成功: {self.host}:{self.port}")
            except redis.ConnectionError as e:
                logger.warning(f"[Redis] 连接失败: {e}")
                self.client = None
        else:
            self.client = None

    def is_connected(self) -> bool:
        return self.client is not None


# ============================================
# 短期记忆（Short-term Memory）
# ============================================

class ShortTermMemory:
    """
    短期记忆 - 当前会话上下文

    特点：
    - Redis String 存储
    - 会话结束后 TTL 过期
    - 轻量化，只存关键信息
    """

    def __init__(self, redis_conn: RedisConnection, ttl: int = 3600):
        self.redis = redis_conn
        self.ttl = ttl  # 默认 1 小时过期
        self._fallback: Dict[str, str] = {}  # 开发模式 fallback

    def _key(self, user_id: str, session_id: str) -> str:
        """生成 Redis Key"""
        return f"short_memory:{user_id}:{session_id}"

    async def store(self, user_id: str, session_id: str, context: Dict[str, Any]) -> bool:
        """
        存储当前会话上下文

        Args:
            user_id: 用户ID
            session_id: 会话ID
            context: 上下文数据（对话历史摘要、当前意图等）
        """
        key = self._key(user_id, session_id)
        data = json.dumps({
            "context": context,
            "timestamp": datetime.now().isoformat()
        })

        if self.redis.client:
            try:
                self.redis.client.set(key, data, ex=self.ttl)
                logger.info(f"[ShortMemory] 存储: {key}")
                return True
            except Exception as e:
                logger.warning(f"[ShortMemory] 存储失败: {e}")

        # Fallback: 内存存储
        self._fallback[key] = data
        logger.info(f"[ShortMemory] Fallback 存储: {key}")
        return True

    async def get(self, user_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        """获取当前会话上下文"""
        key = self._key(user_id, session_id)

        if self.redis.client:
            try:
                data = self.redis.client.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.warning(f"[ShortMemory] 获取失败: {e}")

        # Fallback
        data = self._fallback.get(key)
        if data:
            return json.loads(data)
        return None

    async def clear(self, user_id: str, session_id: str) -> bool:
        """清除会话记忆"""
        key = self._key(user_id, session_id)

        if self.redis.client:
            try:
                self.redis.client.delete(key)
                logger.info(f"[ShortMemory] 清除: {key}")
                return True
            except Exception as e:
                logger.warning(f"[ShortMemory] 清除失败: {e}")

        self._fallback.pop(key, None)
        return True

    async def append_message(self, user_id: str, session_id: str, message: str, role: str = "user") -> bool:
        """
        添加消息到短期记忆

        只保留最近 N 条消息，避免上下文过长
        """
        memory = await self.get(user_id, session_id) or {"context": {"messages": []}}

        if "messages" not in memory["context"]:
            memory["context"]["messages"] = []

        # 添加消息
        memory["context"]["messages"].append({
            "role": role,
            "content": message,
            "timestamp": datetime.now().isoformat()
        })

        # 只保留最近 10 条
        if len(memory["context"]["messages"]) > 10:
            memory["context"]["messages"] = memory["context"]["messages"][-10:]

        return await self.store(user_id, session_id, memory["context"])


# ============================================
# 长期记忆（Long-term Memory）
# ============================================

class LongTermMemory:
    """
    长期记忆 - 跨会话持久化

    特点：
    - Redis Hash 存储用户事实
    - ChromaDB 向量库存储 Embedding（真正的向量数据库）
    - 异步写入，不阻塞主流程
    - distance 阈值保精度
    """

    def __init__(self, redis_conn: RedisConnection, chroma_path: str = None):
        self.redis = redis_conn
        self.embedding_model = EMBEDDING_MODEL  # 使用全局配置
        self._fallback: Dict[str, Dict[str, str]] = {}  # Redis fallback

        # ChromaDB 向量库（真正的向量数据库）
        self.chroma_path = chroma_path or str(Path(__file__).parent.parent / "chroma_data")
        self.chroma_client = None
        self.collection = None
        self._init_chromadb()

        # 异步写入队列
        self._write_queue: asyncio.Queue = None
        self._writer_task: asyncio.Task = None

    def _init_chromadb(self):
        """初始化 ChromaDB 向量库（优先 HTTP Server，回退到嵌入式）"""
        if not CHROMADB_AVAILABLE:
            self.chroma_client = None
            self.collection = None
            return

        chroma_host = os.environ.get("CHROMA_HOST", "localhost")
        chroma_port = os.environ.get("CHROMA_PORT", "8001")
        persist_dir = os.environ.get(
            "CHROMA_PERSIST_DIR",
            str(Path(__file__).parent.parent / "data" / "chroma_db")
        )

        # 方案1: 尝试 HTTP Server（Docker 运行）
        try:
            self.chroma_client = chromadb.HttpClient(
                host=chroma_host,
                port=int(chroma_port),
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                )
            )
            self.chroma_client.heartbeat()
            self.collection = self.chroma_client.get_or_create_collection(
                name="memory_embeddings",
                metadata={"description": "用户长期记忆向量库"}
            )
            logger.info(f"[ChromaDB] HTTP Server: {chroma_host}:{chroma_port} ({self.collection.count()} 条)")
            return
        except Exception:
            logger.info(f"[ChromaDB] HTTP Server 不可用，使用嵌入式模式...")

        # 方案2: 嵌入式 PersistentClient（无需外部服务）
        try:
            os.makedirs(persist_dir, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(
                path=persist_dir,
                settings=chromadb.Settings(
                    anonymized_telemetry=False,
                )
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="memory_embeddings",
                metadata={"description": "用户长期记忆向量库"}
            )
            logger.info(f"[ChromaDB] 嵌入式模式: {persist_dir} ({self.collection.count()} 条)")
        except Exception as e:
            logger.warning(f"[ChromaDB] 嵌入式也失败: {e}")
            self.chroma_client = None
            self.collection = None

    def _key(self, user_id: str) -> str:
        """生成 Redis Key"""
        return f"long_memory:{user_id}"

    async def start_async_writer(self):
        """
        启动异步写入任务

        长期记忆异步写入，不阻塞主流程
        """
        if self._write_queue is None:
            self._write_queue = asyncio.Queue()

        if self._writer_task is None:
            self._writer_task = asyncio.create_task(self._async_writer_loop())
            logger.info("[LongMemory] 异步写入任务启动")

    async def _async_writer_loop(self):
        """异步写入循环"""
        while True:
            try:
                # 从队列获取写入任务
                task = await self._write_queue.get()
                user_id, fact_key, fact_value, embedding = task

                # 写入 Redis Hash（事实内容）
                await self._write_fact(user_id, fact_key, fact_value)

                # 写入 ChromaDB 向量库（Embedding）
                if self.collection and embedding:
                    doc_id = f"{user_id}:{fact_key}"
                    try:
                        # 先删除旧数据（如果存在）
                        try:
                            self.collection.delete(ids=[doc_id])
                        except:
                            pass

                        # 添加新向量
                        self.collection.add(
                            ids=[doc_id],
                            embeddings=[embedding],
                            metadatas=[{
                                "user_id": user_id,
                                "fact_key": fact_key,
                                "fact_value": fact_value,
                                "timestamp": datetime.now().isoformat()
                            }],
                            documents=[fact_value]
                        )
                        logger.info(f"[ChromaDB] 写入向量: {doc_id}")
                    except Exception as e:
                        logger.warning(f"[ChromaDB] 写入失败: {e}")

                logger.info(f"[LongMemory] 异步写入完成: {fact_key}")
                self._write_queue.task_done()

            except asyncio.CancelledError:
                logger.info("[LongMemory] 异步写入任务停止")
                break
            except Exception as e:
                logger.warning(f"[LongMemory] 异步写入失败: {e}")

    async def _write_fact(self, user_id: str, fact_key: str, fact_value: str) -> bool:
        """写入事实到 Redis Hash"""
        key = self._key(user_id)

        if self.redis.client:
            try:
                self.redis.client.hset(key, fact_key, json.dumps({
                    "value": fact_value,
                    "timestamp": datetime.now().isoformat()
                }))
                return True
            except Exception as e:
                logger.warning(f"[LongMemory] 写入失败: {e}")

        # Fallback
        if key not in self._fallback:
            self._fallback[key] = {}
        self._fallback[key][fact_key] = json.dumps({
            "value": fact_value,
            "timestamp": datetime.now().isoformat()
        })
        return True

    async def store_fact(self, user_id: str, fact_key: str, fact_value: str, async_write: bool = True) -> bool:
        """
        存储长期记忆

        Args:
            user_id: 用户ID
            fact_key: 事实类型（如 "name", "preference", "location"）
            fact_value: 事实内容
            async_write: 是否异步写入
        """
        # 计算 Embedding（用于后续检索）
        embedding = await self._compute_embedding(fact_value)

        if async_write and self._write_queue:
            # 异步写入队列
            await self._write_queue.put((user_id, fact_key, fact_value, embedding))
            logger.info(f"[LongMemory] 加入异步队列: {fact_key}")
            return True
        else:
            # 同步写入
            await self._write_fact(user_id, fact_key, fact_value)
            return True

    async def _compute_embedding(self, text: str) -> List[float]:
        """计算文本 Embedding"""
        import httpx

        if not EMBEDDING_API_KEY:
            return []

        base_url = EMBEDDING_API_URL.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{base_url}/embeddings",
                    headers={"Authorization": f"Bearer {EMBEDDING_API_KEY}"},
                    json={"model": EMBEDDING_MODEL, "input": text}
                )

                if response.status_code == 200:
                    data = response.json()
                    if "data" in data and len(data["data"]) > 0:
                        return data["data"][0].get("embedding", [])
        except Exception as e:
            logger.warning(f"[LongMemory] Embedding 计算: {e}")

        return []

    async def get_fact(self, user_id: str, fact_key: str) -> Optional[str]:
        """获取单个事实"""
        key = self._key(user_id)

        if self.redis.client:
            try:
                data = self.redis.client.hget(key, fact_key)
                if data:
                    parsed = json.loads(data)
                    return parsed.get("value")
            except Exception as e:
                logger.warning(f"[LongMemory] 获取失败: {e}")

        # Fallback
        if key in self._fallback and fact_key in self._fallback[key]:
            parsed = json.loads(self._fallback[key][fact_key])
            return parsed.get("value")
        return None

    async def get_all_facts(self, user_id: str) -> Dict[str, str]:
        """获取用户所有事实"""
        key = self._key(user_id)

        if self.redis.client:
            try:
                data = self.redis.client.hgetall(key)
                result = {}
                for k, v in data.items():
                    parsed = json.loads(v)
                    result[k] = parsed.get("value")
                return result
            except Exception as e:
                logger.warning(f"[LongMemory] 获取所有失败: {e}")

        # Fallback
        if key in self._fallback:
            result = {}
            for k, v in self._fallback[key].items():
                parsed = json.loads(v)
                result[k] = parsed.get("value")
            return result
        return {}

    async def search_relevant(self, user_id: str, query: str, distance_threshold: float = 0.3, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """
        搜索相关记忆（使用 ChromaDB 向量检索）

        Args:
            user_id: 用户ID
            query: 查询文本
            distance_threshold: 距离阈值（相似度 > threshold 才返回）
            top_k: 返回数量

        Returns:
            [(fact_key, fact_value, similarity), ...]
        """
        # 计算查询 Embedding
        query_embedding = await self._compute_embedding(query)
        if not query_embedding:
            return []

        results = []

        # 使用 ChromaDB 向量检索
        if self.collection:
            try:
                # ChromaDB 查询
                query_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k * 2,  # 多查一些，后面过滤
                    where={"user_id": user_id}  # 只查该用户的记忆
                )

                # 解析结果
                if query_results and query_results.get('metadatas'):
                    for i, metadata in enumerate(query_results['metadatas'][0]):
                        fact_key = metadata.get('fact_key', '')
                        fact_value = metadata.get('fact_value', '')

                        # ChromaDB 返回的是 distance，转换为 similarity
                        distance = query_results['distances'][0][i] if query_results.get('distances') else 0
                        # distance 越小越相似，转换为 similarity（越大越相似）
                        similarity = 1 - distance

                        # distance 阀值过滤
                        if similarity >= distance_threshold:
                            results.append((fact_key, fact_value, similarity))

                logger.info(f"[ChromaDB] 向量检索返回 {len(results)} 条相关记忆")

            except Exception as e:
                logger.warning(f"[ChromaDB] 检索失败: {e}")

        # Fallback: 如果 ChromaDB 没结果，用传统方法
        if not results:
            all_facts = await self.get_all_facts(user_id)
            for fact_key, fact_value in all_facts.items():
                results.append((fact_key, fact_value, 0.5))  # 默认相似度

        # 按相似度排序
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:top_k]


# ============================================
# 分层记忆管理器（整合短期 + 长期）
# ============================================

class HierarchicalMemory:
    """
    分层记忆管理器

    整合短期记忆 + 长期记忆，配合 LangGraph 状态管理
    """

    def __init__(self, redis_host: str = None, redis_port: int = None):
        self.redis_conn = RedisConnection(redis_host, redis_port)
        self.short_memory = ShortTermMemory(self.redis_conn)
        self.long_memory = LongTermMemory(self.redis_conn)

        logger.info("[HierarchicalMemory] 初始化完成")

    async def start(self):
        """启动记忆系统（启动异步写入任务）"""
        await self.long_memory.start_async_writer()

    # ---- 短期记忆接口 ----

    async def store_context(self, user_id: str, session_id: str, context: Dict) -> bool:
        """存储当前会话上下文"""
        return await self.short_memory.store(user_id, session_id, context)

    async def get_context(self, user_id: str, session_id: str) -> Optional[Dict]:
        """获取当前会话上下文"""
        return await self.short_memory.get(user_id, session_id)

    async def append_message(self, user_id: str, session_id: str, message: str, role: str) -> bool:
        """添加消息到短期记忆"""
        return await self.short_memory.append_message(user_id, session_id, message, role)

    async def clear_session(self, user_id: str, session_id: str) -> bool:
        """清除会话记忆"""
        return await self.short_memory.clear(user_id, session_id)

    # ---- 长期记忆接口 ----

    async def remember_fact(self, user_id: str, fact_key: str, fact_value: str) -> bool:
        """记住用户事实（异步写入）"""
        return await self.long_memory.store_fact(user_id, fact_key, fact_value, async_write=True)

    async def recall_fact(self, user_id: str, fact_key: str) -> Optional[str]:
        """回忆特定事实"""
        return await self.long_memory.get_fact(user_id, fact_key)

    async def recall_all_facts(self, user_id: str) -> Dict[str, str]:
        """回忆所有事实"""
        return await self.long_memory.get_all_facts(user_id)

    async def search_memories(self, user_id: str, query: str, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """搜索相关记忆（distance 阈值过滤）"""
        return await self.long_memory.search_relevant(user_id, query, threshold)

    # ---- LangGraph 状态接口 ----

    async def build_context_for_llm(self, user_id: str, session_id: str, current_input: str) -> str:
        """
        构建 LLM 上下文

        整合短期记忆（当前会话）+ 长期记忆（相关历史）
        """
        context_parts = []

        # 1. 短期记忆：当前会话上下文
        short_ctx = await self.get_context(user_id, session_id)
        if short_ctx and "messages" in short_ctx.get("context", {}):
            messages = short_ctx["context"]["messages"]
            if messages:
                context_parts.append("[当前会话]")
                for msg in messages[-5:]:  # 最近 5 条
                    context_parts.append(f"{msg['role']}: {msg['content']}")

        # 2. 长期记忆：相关历史记忆
        relevant_memories = await self.search_memories(user_id, current_input, threshold=0.4)
        if relevant_memories:
            context_parts.append("\n[用户相关记忆]")
            for fact_key, fact_value, similarity in relevant_memories[:3]:
                context_parts.append(f"- {fact_key}: {fact_value} (相似度: {similarity:.2f})")

        # 3. 用户所有事实（兜底）
        if not relevant_memories:
            all_facts = await self.recall_all_facts(user_id)
            if all_facts:
                context_parts.append("\n[用户信息]")
                for k, v in list(all_facts.items())[:5]:
                    context_parts.append(f"- {k}: {v}")

        return "\n".join(context_parts) if context_parts else ""


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, override=True)

    async def test():
        memory = HierarchicalMemory()
        await memory.start()

        user_id = "test_user"
        session_id = "session_001"

        # 测试短期记忆
        print("\n=== 短期记忆测试 ===")
        await memory.append_message(user_id, session_id, "分析这个Bug", "user")
        await memory.append_message(user_id, session_id, "发现空指针异常", "assistant")

        ctx = await memory.get_context(user_id, session_id)
        print(f"当前会话上下文: {ctx}")

        # 测试长期记忆
        print("\n=== 长期记忆测试 ===")
        await memory.remember_fact(user_id, "project", "commons-math")
        await memory.remember_fact(user_id, "bug_type", "NullPointerException")

        # 等待异步写入完成
        await asyncio.sleep(1)

        all_facts = await memory.recall_all_facts(user_id)
        print(f"所有事实: {all_facts}")

        # 构建 LLM 上下文
        print("\n=== LLM 上下文 ===")
        llm_context = await memory.build_context_for_llm(user_id, session_id, "修复这个空指针")
        print(llm_context)

    asyncio.run(test())