"""
Skills 意图匹配器 - Embedding + LLM 路由器
==========================================

两层过滤方案：
1. Embedding 相似度 → 粗筛选（快速，低成本）
2. LLM 路由器 → 精确认定（准确）

用户输入 → Embedding筛选Top-3 → LLM确认 → 最终Skill
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import asyncio

logger = logging.getLogger("intent-matcher")
logging.basicConfig(level=logging.INFO)

# API Base URL（支持代理）
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1/")

# ============================================
# 第一层：Embedding 相似度匹配（粗筛选）
# ============================================

class EmbeddingMatcher:
    """基于 Embedding 相似度的 Skills 粗筛选"""

    def __init__(self, api_key: str = None, cache_path: Path = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.cache_path = cache_path or Path(__file__).parent.parent / ".skill_embeddings.json"
        self.skill_embeddings: Dict[str, List[float]] = {}
        self._load_cache()

        if not self.api_key:
            logger.warning("[Embedding] 未配置 OPENAI_API_KEY")

    def _load_cache(self):
        """加载预计算的 Embedding 缓存"""
        if self.cache_path.exists():
            try:
                self.skill_embeddings = json.loads(self.cache_path.read_text())
                logger.info(f"[Embedding] 加载缓存: {len(self.skill_embeddings)} 个 Skills")
            except Exception as e:
                logger.warning(f"[Embedding] 缓存加载失败: {e}")

    def _save_cache(self):
        """保存 Embedding 缓存"""
        try:
            self.cache_path.write_text(json.dumps(self.skill_embeddings, indent=2))
            logger.info(f"[Embedding] 保存缓存: {len(self.skill_embeddings)} 个 Skills")
        except Exception as e:
            logger.warning(f"[Embedding] 缓存保存失败: {e}")

    async def sync_embeddings(self, skills: List[Dict]) -> Dict[str, List[float]]:
        """
        同步 Embedding 缓存（增量更新）

        - 已缓存的 Skill：直接用缓存
        - 新增的 Skill：计算 Embedding 并加入缓存
        - 已删除的 Skill：从缓存中移除
        """
        import httpx

        if not self.api_key:
            logger.error("[Embedding] 缺少 OPENAI_API_KEY")
            return self.skill_embeddings

        # 当前 Skills 名称列表
        current_skill_names = set(s.get("name", "") for s in skills)

        # 缓存中的 Skills 名称列表
        cached_skill_names = set(self.skill_embeddings.keys())

        # 需要新增的 Skills
        new_skills = current_skill_names - cached_skill_names

        # 需要删除的 Skills（缓存中有但 Skills 目录没有）
        removed_skills = cached_skill_names - current_skill_names

        if new_skills:
            logger.info(f"[Embedding] 发现新 Skills: {new_skills}，开始计算...")

        if removed_skills:
            logger.info(f"[Embedding] 移除已删除 Skills: {removed_skills}")
            for name in removed_skills:
                self.skill_embeddings.pop(name, None)

        # 只计算新 Skills 的 Embedding
        if new_skills:
            base_url = OPENAI_BASE_URL.rstrip("/")
            async with httpx.AsyncClient(timeout=30.0) as client:
                for skill in skills:
                    name = skill.get("name", "")
                    if name not in new_skills:
                        continue  # 已缓存的跳过

                    description = skill.get("description", "")
                    text = f"{name}: {description}"

                    try:
                        response = await client.post(
                            f"{base_url}/embeddings",
                            headers={"Authorization": f"Bearer {self.api_key}"},
                            json={"model": "text-embedding-3-small", "input": text}
                        )

                        if response.status_code == 200:
                            data = response.json()
                            if "data" in data and len(data["data"]) > 0:
                                embedding = data["data"][0].get("embedding", [])
                                if embedding:
                                    self.skill_embeddings[name] = embedding
                                    logger.info(f"[Embedding] 新增 {name}: {len(embedding)} 维")

                    except Exception as e:
                        logger.warning(f"[Embedding] {name} 计算失败: {e}")

            # 保存更新后的缓存
            self._save_cache()

        return self.skill_embeddings

    async def precompute_embeddings(self, skills: List[Dict]) -> Dict[str, List[float]]:
        """预计算所有 Skills 的 Embedding（一次性）"""
        import httpx

        if not self.api_key:
            logger.error("[Embedding] 缺少 OPENAI_API_KEY，无法预计算")
            return {}

        embeddings = {}
        base_url = OPENAI_BASE_URL.rstrip("/")
        async with httpx.AsyncClient(timeout=30.0) as client:
            for skill in skills:
                name = skill.get("name", "")
                description = skill.get("description", "")

                # 用 name + description 作为 Embedding 文本
                text = f"{name}: {description}"

                try:
                    response = await client.post(
                        f"{base_url}/embeddings",
                        headers={"Authorization": f"Bearer {self.api_key}"},
                        json={"model": "text-embedding-3-small", "input": text}
                    )

                    # 检查响应状态
                    if response.status_code != 200:
                        logger.warning(f"[Embedding] {name} API错误: {response.status_code}")
                        continue

                    data = response.json()

                    # 安全访问嵌套数据
                    if "data" in data and len(data["data"]) > 0:
                        embedding = data["data"][0].get("embedding", [])
                        if embedding:
                            embeddings[name] = embedding
                            logger.info(f"[Embedding] 计算 {name}: {len(embedding)} 维")
                        else:
                            logger.warning(f"[Embedding] {name} 无 embedding 数据")
                    else:
                        logger.warning(f"[Embedding] {name} 响应格式错误: {data}")

                except httpx.TimeoutException:
                    logger.warning(f"[Embedding] {name} 超时")
                except Exception as e:
                    logger.warning(f"[Embedding] {name} 计算失败: {e}")

        self.skill_embeddings = embeddings
        self._save_cache()
        return embeddings

    async def get_user_embedding(self, user_input: str) -> List[float]:
        """计算用户输入的 Embedding"""
        import httpx

        if not self.api_key:
            logger.error("[Embedding] 缺少 OPENAI_API_KEY")
            return []

        base_url = OPENAI_BASE_URL.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": "text-embedding-3-small", "input": user_input}
                )

                if response.status_code != 200:
                    logger.warning(f"[Embedding] API错误: {response.status_code}")
                    return []

                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return data["data"][0].get("embedding", [])
                else:
                    logger.warning(f"[Embedding] 响应格式错误")
                    return []

        except httpx.TimeoutException:
            logger.warning("[Embedding] 超时")
            return []
        except Exception as e:
            logger.warning(f"[Embedding] 用户输入计算失败: {e}")
            return []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def match(self, user_input: str, skills: List[Dict], top_k: int = 3) -> List[Tuple[str, float]]:
        """Embedding 相似度匹配，返回 Top-K 候选"""
        user_embedding = await self.get_user_embedding(user_input)
        if not user_embedding:
            logger.warning("[Embedding] 用户 embedding 为空，返回空结果")
            return []

        # 如果没有缓存，先预计算
        if not self.skill_embeddings:
            await self.precompute_embeddings(skills)

        # 计算每个 Skill 的相似度
        scores = []
        for skill in skills:
            name = skill.get("name", "")
            skill_embedding = self.skill_embeddings.get(name)

            if skill_embedding:
                similarity = self.cosine_similarity(user_embedding, skill_embedding)
                scores.append((name, similarity))

        # 按相似度排序，返回 Top-K
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ============================================
# 第二层：LLM 路由器（精确认定）
# ============================================

class LLMRouter:
    """LLM 路由器 - 从候选 Skills 中精确选择"""

    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model

        if not self.api_key:
            logger.warning("[LLM Router] 未配置 OPENAI_API_KEY")

    async def route(self, user_input: str, candidates: List[Tuple[str, float]], skills: List[Dict]) -> Optional[str]:
        """LLM 从候选 Skills 中选择最合适的"""
        import httpx

        if not candidates:
            return None

        if not self.api_key:
            logger.warning("[LLM Router] 缺少 API Key，返回相似度最高的")
            return candidates[0][0]

        # 构建 Skills 描述
        skill_descriptions = []
        for name, score in candidates:
            for skill in skills:
                if skill.get("name") == name:
                    desc = skill.get("description", "")
                    skill_descriptions.append(f"- {name}: {desc} (相似度: {score:.2f})")
                    break

        prompt = f"""用户输入："{user_input}"

候选 Skills（按相似度排序）：
{chr(10).join(skill_descriptions)}

请分析用户意图，从上述候选 Skills 中选择最合适的一个。
只返回 Skill 名称，不要解释。

如果用户意图不属于任何 Skill，返回 "NONE"。"""

        base_url = OPENAI_BASE_URL.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 50
                    }
                )

                if response.status_code != 200:
                    logger.warning(f"[LLM Router] API错误: {response.status_code}")
                    return candidates[0][0]

                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    result = data["choices"][0]["message"]["content"].strip()
                    logger.info(f"[LLM Router] 选择: {result}")
                    return result if result != "NONE" else None
                else:
                    return candidates[0][0]

        except httpx.TimeoutException:
            logger.warning("[LLM Router] 超时，返回相似度最高的")
            return candidates[0][0]
        except Exception as e:
            logger.warning(f"[LLM Router] 失败: {e}")
            return candidates[0][0]


# ============================================
# 混合匹配器：Embedding + LLM 路由器
# ============================================

class IntentMatcher:
    """
    意图匹配器 - Embedding + LLM 路由器两层过滤

    流程：
    1. Embedding 相似度 → Top-3 候选
    2. LLM 路由器 → 精确选择

    优势：
    - Embedding 快速筛选（低成本）
    - LLM 只对少量候选判断（节省 Token）
    - 两层过滤更准确
    """

    def __init__(self, skills: List[Dict], api_key: str = None):
        self.skills = skills
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.embedding_matcher = EmbeddingMatcher(self.api_key)
        self.llm_router = LLMRouter(self.api_key)
        logger.info(f"[IntentMatcher] 初始化: {len(skills)} 个 Skills")

    async def sync_on_init(self):
        """
        启动时自动同步 Embedding 缓存

        - 已缓存的 Skill：直接用缓存（不重新计算）
        - 新增的 Skill：计算 Embedding 并加入缓存
        - 已删除的 Skill：从缓存中移除

        建议：在 Agent 启动时调用此方法
        """
        await self.embedding_matcher.sync_embeddings(self.skills)
        logger.info(f"[IntentMatcher] 缓存同步完成: {len(self.embedding_matcher.skill_embeddings)} 个 Skills")

    async def match(self, user_input: str) -> Tuple[Optional[str], Dict]:
        """
        匹配用户输入到 Skill

        Returns:
            (skill_name, debug_info)
        """
        debug_info = {
            "user_input": user_input,
            "embedding_candidates": [],
            "llm_decision": None,
            "final_skill": None
        }

        # 第一层：Embedding 粗筛选
        candidates = await self.embedding_matcher.match(user_input, self.skills, top_k=3)
        debug_info["embedding_candidates"] = candidates

        if not candidates:
            logger.info(f"[IntentMatcher] 无匹配 Skill")
            return None, debug_info

        # 第二层：LLM 路由器精确认定
        final_skill = await self.llm_router.route(user_input, candidates, self.skills)
        debug_info["llm_decision"] = final_skill
        debug_info["final_skill"] = final_skill

        logger.info(f"[IntentMatcher] 最终选择: {final_skill}")
        return final_skill, debug_info

    async def precompute_embeddings(self):
        """预计算所有 Skills 的 Embedding（建议启动时调用）"""
        await self.embedding_matcher.precompute_embeddings(self.skills)


# ============================================
# 使用示例
# ============================================

if __name__ == "__main__":
    # 从 .env 加载环境变量（强制覆盖系统变量）
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path, override=True)

    # 检查 API Key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误: 未配置 OPENAI_API_KEY")
        print("请在 .env 文件中配置: OPENAI_API_KEY=sk-xxx")
        exit(1)

    print(f"API Key: {api_key[:20]}...")

    # Bug修复 Skills 示例
    example_skills = [
        {"name": "code-review", "description": "代码审查，识别潜在Bug和质量问题"},
        {"name": "bug-analysis", "description": "Bug根因分析，定位Bug类型和具体位置"},
        {"name": "fix-patterns", "description": "代码修复模式，生成修复补丁"},
        {"name": "test-generation", "description": "测试用例生成，编写单元测试"},
    ]

    async def test():
        matcher = IntentMatcher(example_skills)

        # 启动时同步 Embedding（增量更新）
        print("\n同步 Embedding 缓存...")
        await matcher.sync_on_init()

        # 测试匹配
        test_inputs = [
            "这段代码有个空指针异常",
            "帮我review这个Java函数",
            "生成修复补丁",
            "写一个单元测试验证修复",
        ]

        for input_text in test_inputs:
            skill, debug = await matcher.match(input_text)
            print(f"\n输入: {input_text}")
            print(f"  Embedding候选: {debug['embedding_candidates']}")
            print(f"  LLM决策: {debug['llm_decision']}")
            print(f"  最终Skill: {skill}")

    asyncio.run(test())