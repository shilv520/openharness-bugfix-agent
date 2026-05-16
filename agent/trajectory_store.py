"""
Agent 轨迹持久化存储
===================

存储 Agent 的思考、行动、观察、反思轨迹
支持：
1. 短期存储（Redis）- 当前任务上下文
2. 长期存储（ChromaDB）- 成功经验向量检索
3. 文件持久化（JSON）- 完整轨迹记录
"""

import json
import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("trajectory-store")

# Trajectory 存储路径
TRAJECTORY_DIR = Path(__file__).parent.parent / "data" / "trajectories"
TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)


class TrajectoryRecord:
    """单条轨迹记录"""

    def __init__(
        self,
        bug_id: str,
        agent_name: str,
        step_type: str,  # think | act | observe | reflect | discuss
        content: Dict[str, Any],
        timestamp: str = None
    ):
        self.bug_id = bug_id
        self.agent_name = agent_name
        self.step_type = step_type
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return {
            "bug_id": self.bug_id,
            "agent": self.agent_name,
            "step": self.step_type,
            "content": self.content,
            "timestamp": self.timestamp
        }


class TrajectoryStore:
    """
    轨迹存储管理器

    三层存储：
    1. 内存缓存（当前任务）
    2. 文件持久化（JSON）
    3. 向量库（ChromaDB）- 可选
    """

    def __init__(self, use_redis: bool = False, use_chroma: bool = False):
        self._current_trajectories: List[TrajectoryRecord] = []
        self._use_redis = use_redis
        self._use_chroma = use_chroma

        # Redis 连接（可选）
        self._redis_client = None
        if use_redis:
            self._init_redis()

        # ChromaDB 连接（可选）
        self._chroma_collection = None
        if use_chroma:
            self._init_chroma()

        logger.info(f"[TrajectoryStore] 初始化完成 (Redis={use_redis}, Chroma={use_chroma})")

    def _init_redis(self):
        """初始化 Redis 连接"""
        try:
            import redis
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            self._redis_client = redis.Redis(
                host=redis_host, port=redis_port,
                decode_responses=True, socket_connect_timeout=5
            )
            self._redis_client.ping()
            logger.info(f"[TrajectoryStore] Redis 连接成功: {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"[TrajectoryStore] Redis 连接失败: {e}")
            self._redis_client = None

    def _init_chroma(self):
        """初始化 ChromaDB 向量库"""
        try:
            import chromadb
            chroma_host = os.environ.get("CHROMA_HOST", "localhost")
            chroma_port = int(os.environ.get("CHROMA_PORT", 8001))

            client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
            self._chroma_collection = client.get_or_create_collection(
                name="bug_fix_trajectories",
                metadata={"description": "Bug修复轨迹向量库"}
            )
            logger.info(f"[TrajectoryStore] ChromaDB 连接成功: {chroma_host}:{chroma_port}")
        except Exception as e:
            logger.warning(f"[TrajectoryStore] ChromaDB 连接失败: {e}")
            self._chroma_collection = None

    # ============================================
    # 轨迹记录接口
    # ============================================

    def record_think(self, bug_id: str, agent: str, thought: Dict) -> TrajectoryRecord:
        """记录思考步骤"""
        record = TrajectoryRecord(bug_id, agent, "think", thought)
        self._add_record(record)
        return record

    def record_act(self, bug_id: str, agent: str, action: str, result: Dict) -> TrajectoryRecord:
        """记录行动步骤"""
        record = TrajectoryRecord(bug_id, agent, "act", {
            "action": action,
            "result": result
        })
        self._add_record(record)
        return record

    def record_observe(self, bug_id: str, agent: str, observation: Dict) -> TrajectoryRecord:
        """记录观察步骤"""
        record = TrajectoryRecord(bug_id, agent, "observe", observation)
        self._add_record(record)
        return record

    def record_reflect(self, bug_id: str, agent: str, reflection: Dict) -> TrajectoryRecord:
        """记录反思步骤"""
        record = TrajectoryRecord(bug_id, agent, "reflect", reflection)
        self._add_record(record)
        return record

    def record_discuss(self, bug_id: str, agents: List[str], topic: str, consensus: Dict) -> TrajectoryRecord:
        """记录讨论步骤"""
        record = TrajectoryRecord(bug_id, "discussion", "discuss", {
            "agents": agents,
            "topic": topic,
            "consensus": consensus
        })
        self._add_record(record)
        return record

    def _add_record(self, record: TrajectoryRecord):
        """添加记录到存储"""
        # 1. 内存缓存
        self._current_trajectories.append(record)

        # 2. Redis（如果启用）
        if self._redis_client:
            key = f"trajectory:{record.bug_id}:{record.timestamp}"
            try:
                self._redis_client.set(key, json.dumps(record.to_dict()), ex=3600)
            except Exception as e:
                logger.warning(f"[TrajectoryStore] Redis 写入失败: {e}")

    # ============================================
    # 任务结束保存
    # ============================================

    def save_task_trajectory(self, bug_id: str, success: bool, final_result: Dict) -> str:
        """
        任务结束后保存完整轨迹

        Returns:
            保存的文件路径
        """
        # 构建完整轨迹
        trajectory_data = {
            "bug_id": bug_id,
            "success": success,
            "final_result": final_result,
            "trajectory": [r.to_dict() for r in self._current_trajectories],
            "stats": {
                "total_steps": len(self._current_trajectories),
                "think_count": sum(1 for r in self._current_trajectories if r.step_type == "think"),
                "act_count": sum(1 for r in self._current_trajectories if r.step_type == "act"),
                "discuss_count": sum(1 for r in self._current_trajectories if r.step_type == "discuss"),
                "reflect_count": sum(1 for r in self._current_trajectories if r.step_type == "reflect"),
            },
            "timestamp": datetime.now().isoformat()
        }

        # 文件持久化
        filename = f"{bug_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = TRAJECTORY_DIR / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        logger.info(f"[TrajectoryStore] 保存轨迹: {filepath}")

        # ChromaDB 向量存储（成功案例）
        if self._chroma_collection and success:
            self._store_successful_case(bug_id, trajectory_data)

        # 清空当前缓存
        self._current_trajectories = []

        return str(filepath)

    def _store_successful_case(self, bug_id: str, trajectory_data: Dict):
        """存储成功案例到向量库"""
        try:
            # 构建摘要文本用于向量检索
            summary = self._build_trajectory_summary(trajectory_data)

            doc_id = f"success:{bug_id}"

            # 删除旧数据（如果存在）
            try:
                self._chroma_collection.delete(ids=[doc_id])
            except:
                pass

            # 添加向量
            self._chroma_collection.add(
                ids=[doc_id],
                documents=[summary],
                metadatas=[{
                    "bug_id": bug_id,
                    "success": True,
                    "bug_type": trajectory_data["final_result"].get("bug_type", ""),
                    "timestamp": trajectory_data["timestamp"]
                }]
            )
            logger.info(f"[TrajectoryStore] ChromaDB 存储成功案例: {bug_id}")
        except Exception as e:
            logger.warning(f"[TrajectoryStore] ChromaDB 存储失败: {e}")

    def _build_trajectory_summary(self, data: Dict) -> str:
        """构建轨迹摘要文本"""
        parts = [
            f"Bug ID: {data['bug_id']}",
            f"Bug Type: {data['final_result'].get('bug_type', 'unknown')}",
            f"Bug Location: {data['final_result'].get('bug_location', 'unknown')}",
            f"Root Cause: {data['final_result'].get('root_cause', 'unknown')}",
            f"Fix Approach: {data['final_result'].get('fix_suggestion', 'unknown')}",
            f"Total Steps: {data['stats']['total_steps']}",
            f"Discussion Rounds: {data['stats']['discuss_count']}"
        ]
        return "\n".join(parts)

    # ============================================
    # 历史检索接口
    # ============================================

    def get_trajectory(self, bug_id: str) -> Optional[Dict]:
        """获取指定Bug的轨迹"""
        # 查找文件
        for file in TRAJECTORY_DIR.glob(f"{bug_id}_*.json"):
            with open(file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_all_trajectories(self) -> List[Dict]:
        """列出所有轨迹"""
        trajectories = []
        for file in TRAJECTORY_DIR.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                trajectories.append(json.load(f))
        return trajectories

    def search_similar_successful_cases(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索相似的成功案例（向量检索）"""
        if not self._chroma_collection:
            logger.warning("[TrajectoryStore] ChromaDB 未启用，无法搜索")
            return []

        try:
            results = self._chroma_collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"success": True}
            )

            cases = []
            if results and results.get('metadatas'):
                for i, metadata in enumerate(results['metadatas'][0]):
                    bug_id = metadata.get('bug_id')
                    trajectory = self.get_trajectory(bug_id)
                    if trajectory:
                        distance = results['distances'][0][i] if results.get('distances') else 0
                        trajectory['similarity'] = 1 - distance
                        cases.append(trajectory)

            logger.info(f"[TrajectoryStore] 搜索返回 {len(cases)} 个相似案例")
            return cases
        except Exception as e:
            logger.warning(f"[TrajectoryStore] 搜索失败: {e}")
            return []

    # ============================================
    # 统计接口
    # ============================================

    def get_stats(self) -> Dict:
        """获取存储统计"""
        trajectories = self.list_all_trajectories()

        total = len(trajectories)
        successful = sum(1 for t in trajectories if t.get("success"))

        return {
            "total_tasks": total,
            "successful_fixes": successful,
            "success_rate": successful / total if total > 0 else 0,
            "total_steps": sum(t["stats"]["total_steps"] for t in trajectories),
            "avg_steps": sum(t["stats"]["total_steps"] for t in trajectories) / total if total > 0 else 0,
        }


# ============================================
# 全局实例
# ============================================

_global_store = None


def get_trajectory_store(use_redis: bool = False, use_chroma: bool = False) -> TrajectoryStore:
    """获取全局轨迹存储实例"""
    global _global_store
    if _global_store is None:
        _global_store = TrajectoryStore(use_redis, use_chroma)
    return _global_store


# ============================================
# 测试
# ============================================

if __name__ == "__main__":
    store = TrajectoryStore()

    # 模拟一次Bug修复任务
    bug_id = "test_bug_001"

    store.record_think(bug_id, "Reviewer", {"analysis": "发现break语句问题", "confidence": 0.8})
    store.record_act(bug_id, "Reviewer", "review_code", {"status": "success", "bug_found": True})
    store.record_discuss(bug_id, ["Reviewer", "Analyzer"], "Bug确认", {"agreed": True})

    store.record_think(bug_id, "Analyzer", {"analysis": "break不在循环中", "confidence": 0.9})
    store.record_act(bug_id, "Analyzer", "analyze_bug", {"status": "success", "root_cause": "语法错误"})
    store.record_discuss(bug_id, ["Analyzer", "Fixer"], "修复策略", {"agreed": True})

    store.record_act(bug_id, "Fixer", "generate_patch", {"status": "success", "patch": "移除break"})
    store.record_observe(bug_id, "Validator", {"success": True, "validation": "语法正确"})

    # 保存轨迹
    filepath = store.save_task_trajectory(bug_id, True, {
        "bug_type": "syntax",
        "bug_location": "line 5",
        "fix_suggestion": "移除break语句"
    })

    print(f"\n保存路径: {filepath}")
    print(f"\n统计: {store.get_stats()}")

    # 读取轨迹
    trajectory = store.get_trajectory(bug_id)
    print(f"\n轨迹内容: {json.dumps(trajectory, indent=2, ensure_ascii=False)}")