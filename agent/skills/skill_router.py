"""
Skill → Agent 自动匹配路由器（分级关键词 + 评分制）
====================================================

匹配策略（三层递进）：
  1. 分级关键词评分（零成本）：Tier1 高特异性(+3) / Tier2 中等(+2) / Tier3 通用(+1)
     总分 ≥ 3 且没有被排除 → 直接命中。解决之前 "analysis" 之类泛词导致过度匹配的问题。
  2. LLM 语义匹配（回退）：关键词评分不够时，用小模型做一次轻量分类。
  3. 多信号兜底（离线可用）：LLM 不可用时，用 name/description/tags 中的动词和名词信号推断。

用法:
    router = SkillRouter(llm_config={...})
    result = await router.route(skill_fm)
    # → MatchResult(agents=["bug-analyzer"], confidence=0.85, method="keyword:deadlock")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .progressive import SkillFrontmatter

logger = logging.getLogger("skills.router")

# ═══════════════════════════════════════════════════════════════════
# 分级关键词表（Tier 1=强信号 / Tier 2=中等 / Tier 3=弱信号）
# ═══════════════════════════════════════════════════════════════════

# Tier 1: 高特异性领域词 —— 命中一个就足以分配
# Tier 2: 中等特异性词 —— 需要配合其他信号
# Tier 3: 通用词 —— 单独命中不计分，只做加分项
# exclude: 排除词 —— Skill 含这些词时，明确不应分配给该 Agent

AGENT_KEYWORD_TIERS: Dict[str, Dict[str, List[str]]] = {
    "code-reviewer": {
        "tier1": [
            # 精确 Bug 类型
            "空指针", "null pointer", "null dereference",
            "数组越界", "array index out of bounds", "index out of bounds",
            "类型转换错误", "class cast exception", "type mismatch",
            "资源泄漏", "resource leak", "fd leak", "handle leak",
            "SQL注入", "sql injection", "XSS", "cross site scripting",
            "不安全反序列化", "insecure deserialization",
            "硬编码密码", "hardcoded password", "hardcoded credential",
        ],
        "tier2": [
            "代码审查", "code review", "静态分析", "static analysis",
            "安全漏洞", "security vulnerability", "漏洞扫描",
            "代码质量", "code quality", "lint", "代码规范",
            "bug模式", "bug pattern", "反模式", "anti pattern",
            "语法错误", "syntax error", "编译错误", "compilation error",
            "异常处理不当", "exception misuse", "swallowed exception",
        ],
        "tier3": [
            "审查", "review", "检查", "检测", "check", "detect",
            "安全", "security", "规范", "风格", "style",
        ],
        "exclude": [
            "修复", "fix", "补丁", "patch",  # 属于 fixer
            "生成", "generate",
        ],
    },
    "bug-analyzer": {
        "tier1": [
            "死锁", "deadlock", "活锁", "livelock",
            "竞态条件", "race condition", "数据竞争", "data race",
            "内存泄漏", "memory leak", "OOM", "out of memory",
            "线程安全", "thread safety", "线程不安全", "thread unsafe",
            "缓存一致性问题", "cache coherence", "缓存穿透", "缓存雪崩",
            "循环依赖", "circular dependency",
            "CAS ABA问题", "ABA problem",
            "分布式一致性", "distributed consensus",
        ],
        "tier2": [
            "根因分析", "root cause analysis", "根因", "root cause",
            "影响评估", "impact analysis", "波及范围",
            "并发", "concurrency", "多线程", "multithreading",
            "性能瓶颈", "performance bottleneck", "性能分析",
            "诊断", "diagnose", "诊断工具",
            "调用链分析", "call chain", "堆栈分析", "stack trace",
            "假阳性", "false positive", "误报",
        ],
        "tier3": [
            "分析", "analyze", "分析工具",
            "追溯", "trace", "追踪",
            "分类", "classify", "归类",
            "复杂", "complex", "深层",
        ],
        "exclude": [
            "修复", "fix", "补丁", "patch",  # 属于 fixer
            "测试", "test", "验证", "validate",
        ],
    },
    "code-fixer": {
        "tier1": [
            "空指针修复", "null pointer fix", "null check",
            "死锁修复", "deadlock fix", "deadlock prevention",
            "内存泄漏修复", "memory leak fix",
            "边界检查", "boundary check", "范围检查", "bounds check",
            "防御性编程", "defensive programming", "防御性拷贝",
            "异常处理重构", "exception refactor",
            "资源管理", "resource management", "try-with-resources", "RAII",
        ],
        "tier2": [
            "补丁生成", "patch generation", "代码补丁",
            "修复", "fix", "bug fix", "hotfix",
            "重构", "refactor", "重写", "rewrite",
            "代码生成", "code generation",
            "空安全", "null safety", "optional",
        ],
        "tier3": [
            "补丁", "patch", "修改", "modify", "改", "修",
            "生成", "generate",
            "调整", "adjust", "优化", "optimize",
        ],
        "exclude": [
            "测试", "test", "验证", "validate", "verify",  # 属于 validator
            "检测", "detect", "发现", "扫描",  # 属于 reviewer
        ],
    },
    "test-validator": {
        "tier1": [
            "单元测试", "unit test", "JUnit", "pytest", "testNG",
            "回归测试", "regression test", "回归验证",
            "集成测试", "integration test", "端到端测试", "E2E test",
            "模糊测试", "fuzz testing", "fuzzing",
            "变异测试", "mutation testing",
            "契约测试", "contract test", "PACT",
            "覆盖率验证", "coverage validation",
        ],
        "tier2": [
            "测试用例生成", "test case generation", "测试生成",
            "编译验证", "compile verification", "编译检查",
            "副作用检测", "side effect detection",
            "兼容性测试", "compatibility test",
            "Mock", "stub", "测试替身",
        ],
        "tier3": [
            "验证", "validate", "测试", "test", "确认", "verify",
            "回归", "regression", "编译", "compile",
            "副作用", "side effect", "检查点", "checkpoint",
        ],
        "exclude": [
            "修复", "fix", "补丁", "patch",  # 属于 fixer
            "根因", "root cause",  # 属于 analyzer
        ],
    },
}


@dataclass
class MatchResult:
    """路由匹配结果"""
    agents: List[str] = field(default_factory=list)
    confidence: float = 0.0
    method: str = ""               # "keyword:tier1" / "keyword:tier2+tier3" / "llm" / "fallback"
    scores: Dict[str, int] = field(default_factory=dict)  # 每个 Agent 的评分
    reasoning: str = ""            # 匹配理由（可读）


class SkillRouter:
    """根据 Skill 的特征，分级评分后自动匹配到合适的 Agent。"""

    # ── 评分阈值 ──
    MIN_SCORE = 3          # 至少 3 分才认定匹配
    TIER1_WEIGHT = 3       # 高特异性词权重
    TIER2_WEIGHT = 2       # 中等特异性词权重
    TIER3_WEIGHT = 1       # 通用词权重（单独一个 Tier3 不达标）

    def __init__(self, llm_config: Dict = None):
        self.llm_config = llm_config or {}
        # 预计算所有关键词的小写形式
        self._tiers_lower = self._build_lower_tiers()

    def _build_lower_tiers(self) -> Dict:
        """预计算所有关键词的小写版本，运行时只做一次 tolower。"""
        result = {}
        for agent, tiers in AGENT_KEYWORD_TIERS.items():
            result[agent] = {}
            for tier_name in ("tier1", "tier2", "tier3", "exclude"):
                result[agent][tier_name] = [
                    kw.lower() for kw in tiers.get(tier_name, [])
                ]
        return result

    # New field for keyword tier data:
    # AGENT_KEYWORD_TIERS is accessed via class-level lookup
    # but the lowercased versions are in self._tiers_lower

    # ═══════════════════════════════════════════════════════════════
    # 公开 API
    # ═══════════════════════════════════════════════════════════════

    async def route(
        self,
        skill: SkillFrontmatter,
        available_agents: List[str] = None,
    ) -> MatchResult:
        """将单个 Skill 路由到最匹配的 Agent 列表。

        Returns:
            MatchResult(agents, confidence, method, scores, reasoning)
        """
        agents = available_agents or list(AGENT_KEYWORD_TIERS.keys())

        # ── 第一层：分级关键词评分 ──
        result = self._scored_match(skill, agents)
        if result.agents and result.confidence >= 0.6:
            logger.info(
                f"[SkillRouter] '{skill.name}' → {result.agents} "
                f"(confidence={result.confidence:.0%}, method={result.method})"
            )
            return result

        # ── 第二层：LLM 语义匹配（评分不够时）──
        llm_agents = await self._llm_match(skill, agents)
        if llm_agents:
            result.agents = llm_agents
            result.confidence = 0.7
            result.method = "llm"
            result.reasoning = f"关键词评分不足，LLM 判定 → {llm_agents}"
            logger.info(f"[SkillRouter] '{skill.name}' LLM → {llm_agents}")
            return result

        # ── 第三层：多信号兜底（离线可用）──
        fallback = self._multi_signal_fallback(skill, agents)
        result.agents = [fallback]
        result.confidence = 0.4
        result.method = "fallback"
        result.reasoning = f"LLM 不可用，多信号兜底 → {fallback}"
        logger.info(f"[SkillRouter] '{skill.name}' fallback → {fallback}")
        return result

    async def route_batch(
        self,
        skills: List[SkillFrontmatter],
        available_agents: List[str] = None,
    ) -> Dict[str, List[str]]:
        """批量路由：返回 {agent_name: [skill_name, ...]}。"""
        agents = available_agents or list(AGENT_KEYWORD_TIERS.keys())
        assignment: Dict[str, List[str]] = {a: [] for a in agents}

        for skill in skills:
            result = await self.route(skill, agents)
            for agent in result.agents:
                if agent in assignment:
                    assignment[agent].append(skill.name)

        return assignment

    # ═══════════════════════════════════════════════════════════════
    # 第一层：分级关键词评分
    # ═══════════════════════════════════════════════════════════════

    def _scored_match(
        self, skill: SkillFrontmatter, agents: List[str]
    ) -> MatchResult:
        """对每个 Agent 计算加权得分，达到阈值才算匹配。"""
        skill_text = self._build_skill_text(skill)
        scores: Dict[str, int] = {}
        tier_hits: Dict[str, List[str]] = {}  # 记录每个 Agent 命中了哪些词

        for agent in agents:
            tiers = self._tiers_lower.get(agent, {})
            score = 0
            hits: List[str] = []

            # ── 先检查排除词 ──
            excluded = False
            for kw in tiers.get("exclude", []):
                if kw in skill_text:
                    excluded = True
                    logger.debug(
                        f"[SkillRouter] '{skill.name}' excluded from {agent} "
                        f"(排除词: '{kw}')"
                    )
                    break
            if excluded:
                continue

            # ── Tier 1（权重 3）──
            for kw in tiers.get("tier1", []):
                if kw in skill_text:
                    score += self.TIER1_WEIGHT
                    hits.append(f"T1:{kw}")

            # ── Tier 2（权重 2）──
            for kw in tiers.get("tier2", []):
                if kw in skill_text:
                    score += self.TIER2_WEIGHT
                    hits.append(f"T2:{kw}")

            # ── Tier 3（权重 1，但单独一个 Tier3 不计分）──
            # Tier3 只在已经有 Tier1/Tier2 命中时才加分
            if score > 0:
                for kw in tiers.get("tier3", []):
                    if kw in skill_text:
                        score += self.TIER3_WEIGHT
                        hits.append(f"T3:{kw}")

            scores[agent] = score
            tier_hits[agent] = hits

        # ── 筛选达标的 Agent ──
        matched = [
            agent for agent, score in scores.items()
            if score >= self.MIN_SCORE
        ]
        matched.sort(key=lambda a: scores[a], reverse=True)

        # ── 计算置信度 ──
        if matched:
            top_score = scores[matched[0]]
            # 8+ 分 = 高置信，5-7 = 中等，3-4 = 低但达标
            if top_score >= 8:
                confidence = 0.95
            elif top_score >= 5:
                confidence = 0.80
            else:
                confidence = 0.65

            # 确定匹配方法
            has_tier1 = any(
                any(h.startswith("T1:") for h in tier_hits.get(a, []))
                for a in matched
            )
            method = "keyword:tier1" if has_tier1 else "keyword:tier2+tier3"
        else:
            confidence = 0.0
            method = "none"

        best_reason = ""
        if matched:
            best_reason = (
                f"{matched[0]}(得分{scores[matched[0]]}): "
                f"{', '.join(tier_hits[matched[0]][:5])}"
            )

        return MatchResult(
            agents=matched,
            confidence=confidence,
            method=method,
            scores=scores,
            reasoning=best_reason,
        )

    # ═══════════════════════════════════════════════════════════════
    # 第二层：LLM 语义匹配
    # ═══════════════════════════════════════════════════════════════

    async def _llm_match(
        self, skill: SkillFrontmatter, agents: List[str]
    ) -> List[str]:
        """关键词评分不足时，用 LLM 做一次轻量语义分类。"""
        import httpx

        api_key = (
            self.llm_config.get("api_key")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            return []

        base_url = self.llm_config.get(
            "base_url", "https://api.deepseek.com/v1"
        )
        model = self.llm_config.get("model", "deepseek-chat")

        # 用 Agent 的能力描述而非关键词列表，LLM 理解更准确
        agent_roles = {
            "code-reviewer": "代码审查：发现Bug、安全漏洞、代码质量问题",
            "bug-analyzer": "Bug根因分析：追溯根因、评估影响、推荐修复方案",
            "code-fixer": "代码修复：生成精准补丁、重构、防御性编程",
            "test-validator": "测试验证：验证修复正确性、副作用检测、回归测试",
        }

        agent_lines = "\n".join([
            f"- {a}: {agent_roles.get(a, '')}" for a in agents
        ])

        prompt = f"""根据 Skill 特征分配 Agent。每个 Skill 可以匹配 1-2 个 Agent。

Skill:
  名称: {skill.name}
  描述: {skill.description}
  标签: {', '.join(skill.tags)}

Agent 职责:
{agent_lines}

输出 JSON 数组: ["agent-name"] 或 ["agent-name-1", "agent-name-2"]
只输出 JSON，不要其他内容。"""

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0,
                        "max_tokens": 80,
                    },
                )
                content = resp.json()["choices"][0]["message"]["content"]
                import re
                match = re.search(r"\[[\s\S]*?\]", content)
                if match:
                    result = json.loads(match.group())
                    return [a for a in result if a in agents]
        except Exception as e:
            logger.warning(f"[SkillRouter] LLM match failed: {e}")

        return []

    # ═══════════════════════════════════════════════════════════════
    # 第三层：多信号兜底（离线可用，比之前的 _default_agent 更准）
    # ═══════════════════════════════════════════════════════════════

    def _multi_signal_fallback(
        self, skill: SkillFrontmatter, agents: List[str]
    ) -> str:
        """LLM 不可用时，用 name+description+tags 中多个信号加权推断。

        不再像之前那样用几个模糊词猜，而是检查三类信号：
          动词信号：fix/修复 → fixer, test/测试 → validator
          名词信号：死锁/并发 → analyzer, 空指针/漏洞 → reviewer
          组合信号：动词+名词同时指向同一 Agent → 高置信
        """
        text = (skill.name + " " + skill.description).lower()
        tag_text = " ".join(skill.tags).lower()

        signals: Dict[str, int] = {a: 0 for a in agents}

        # ── 动词信号 ──
        verb_signals = {
            "code-reviewer": ["review", "审查", "检查", "detect", "检测", "发现", "扫描", "scan", "find"],
            "bug-analyzer": ["analyze", "分析", "诊断", "diagnose", "trace", "追溯", "定位", "locate", "理解", "understand"],
            "code-fixer": ["fix", "修复", "修", "改", "补丁", "patch", "refactor", "重构", "生成", "generate"],
            "test-validator": ["test", "测试", "验证", "validate", "verify", "确认", "compile", "编译"],
        }
        for agent, verbs in verb_signals.items():
            for v in verbs:
                if v in text:
                    signals[agent] += 2  # 动词信号权重 2

        # ── 名词信号（领域词，隐含 Agent 归属）──
        noun_signals = {
            "code-reviewer": [
                "空指针", "null pointer", "数组越界", "index out",
                "xss", "sql injection", "漏洞", "vulnerability",
                "语法错误", "syntax error", "代码规范",
            ],
            "bug-analyzer": [
                "死锁", "deadlock", "竞态", "race condition",
                "内存泄漏", "memory leak", "并发", "concurrency",
                "性能", "performance", "根因", "root cause",
                "调用链", "堆栈", "stack trace", "瓶颈", "bottleneck",
            ],
            "code-fixer": [
                "防御性", "defensive", "null check", "异常处理",
                "exception handling", "resource management", "资源管理",
            ],
            "test-validator": [
                "单元测试", "unit test", "回归", "regression",
                "覆盖率", "coverage", "junit", "pytest", "mock",
            ],
        }
        for agent, nouns in noun_signals.items():
            for n in nouns:
                if n in text or n in tag_text:
                    signals[agent] += 3  # 名词信号权重 3（更强）

        # ── 选出最高分 ──
        best_agent = max(signals, key=lambda a: signals[a])
        best_score = signals[best_agent]

        if best_score == 0:
            # 完全没信号 → 默认 bug-analyzer（分析能力最通用）
            return "bug-analyzer"

        # 如果第一名得分明显高于第二名（差距 ≥ 2），直接返回
        sorted_agents = sorted(signals.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_agents) >= 2:
            gap = sorted_agents[0][1] - sorted_agents[1][1]
            if gap >= 2:
                return best_agent

        return best_agent

    # ═══════════════════════════════════════════════════════════════
    # Skill 文本构建
    # ═══════════════════════════════════════════════════════════════

    def _build_skill_text(self, skill: SkillFrontmatter) -> str:
        """构建用于匹配的 Skill 文本（小写）。"""
        parts = [
            skill.name.lower().replace("-", " ").replace("_", " "),
            skill.description.lower(),
            " ".join(skill.tags).lower(),
        ]
        return " ".join(parts)

    # ═══════════════════════════════════════════════════════════════
    # 公开：获取匹配得分（供外部诊断用）
    # ═══════════════════════════════════════════════════════════════

    def get_scores(self, skill: SkillFrontmatter) -> Dict[str, int]:
        """获取 Skill 对各 Agent 的匹配得分（不触发 LLM）。"""
        result = self._scored_match(skill, list(AGENT_KEYWORD_TIERS.keys()))
        return result.scores


# ── 便捷函数 ──────────────────────────────────────────────────────

_global_router: Optional[SkillRouter] = None


def get_skill_router(llm_config: Dict = None) -> SkillRouter:
    """获取全局 SkillRouter 单例。"""
    global _global_router
    if _global_router is None:
        _global_router = SkillRouter(llm_config=llm_config)
    return _global_router
