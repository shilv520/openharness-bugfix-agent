# Skill 动态更新与自动分配机制

> 记录 Skill 系统的四个核心改动：Agent 可调用 Skill 管理工具、更新已有 Skill、创建/更新决策、Skill→Agent 自动匹配。

---

## 一、整体流程

```
Agent 在 ReAct 循环中遇到新问题（如 Java 死锁检测）
    │
    ├─① Agent Think: "现有能力不足以处理死锁"
    │     Act: search_skills("死锁检测")     ← Agent 主动调用 Skill 管理工具
    │     Observe: 返回空或低相似度
    │
    ├─② Act: recommend_skill_action("Java多线程死锁检测")
    │     └─ ChromaDB 语义搜索 → 相似度判定
    │          > 0.7 → "update"
    │          0.3-0.7 → "ambiguous"
    │          < 0.3 → "create"
    │
    ├─③ Act: create_skill(name, body, tags)  或  update_skill(name, body, tags)
    │     └─ SKILL.md 落盘 → ChromaDB 索引 → Redis 持久化
    │
    ├─④ SkillRouter.route(skill)
    │     └─ 分级关键词评分(T1/T2/T3) → 总分≥3 匹配 → LLM兜底 → 多信号推断
    │
    └─⑤ SubAgent 启动时 extra_skills 注入 System Prompt
```

---

## 二、SkillManager 新增方法

### 2.1 `update_skill` — 更新已有 Skill

文件：`agent/skills/manager.py`

```python
async def update_skill(
    self, name: str, description: str = None, body: str = None,
    tags: List[str] = None,
) -> Optional[SkillFrontmatter]:
    """更新已有 Skill 的 body 内容或元数据。

    当 Agent 发现某个 Skill 无法处理新问题，但问题类型与该 Skill 相近时，
    可以扩充 body 内容，而不是每次都创建全新 Skill。

    如果 Skill 不存在，返回 None（调用方应改用 create_skill）。
    """
    existing_fm = self.persistence.get_skill_meta(name)
    if existing_fm is None:
        return None

    # 定位 SKILL.md 文件
    scope_dir = self.skills_root / _scope_dir(existing_fm.scope)
    skill_dir = scope_dir / name
    skill_md = skill_dir / "SKILL.md" if skill_dir.is_dir() else scope_dir / f"{name}.md"

    if not skill_md.exists():
        return None

    # 更新元数据
    if description:
        existing_fm.description = description
    if tags:
        existing_fm.tags = tags
    existing_fm.updated_at = datetime.now().isoformat()

    # 拼接新的完整内容
    new_body = body if body is not None else get_body_only(load_full_content(skill_md))
    full_content = make_frontmatter_block(existing_fm) + "\n" + new_body

    # 写回文件
    skill_md.write_text(full_content, encoding="utf-8")
    existing_fm.checksum = compute_checksum(full_content)
    existing_fm.path = str(skill_md.resolve().relative_to(Path.cwd()))

    # 更新持久化（ChromaDB + Redis）
    self.persistence.store_skill_content(existing_fm, full_content)
    self._content_cache[name] = full_content

    return existing_fm
```

### 2.2 `decide_create_or_update` — 创建/更新决策

文件：`agent/skills/manager.py`

```python
async def decide_create_or_update(
    self, problem_description: str, top_k: int = 3,
) -> Dict:
    """判断一个问题应该创建新 Skill 还是扩充已有 Skill。

    流程：
      1. 用 ChromaDB 语义搜索最相似的已有 Skill
      2. 相似度 > 0.7 → 推荐 update_skill（扩充）
      3. 相似度 0.3-0.7 → 列出候选，交给 Agent 自己判断
      4. 相似度 < 0.3 或没结果 → 推荐 create_skill（新建）
    """
    search_results = await self.search_skills(problem_description, top_k=top_k)

    if not search_results:
        return {
            "action": "create",
            "reason": "没有找到语义相近的已有 Skill，建议创建新 Skill",
            "target_skill": None,
            "candidates": [],
            "top_match_similarity": 0.0,
        }

    top_fm, top_similarity = search_results[0]

    if top_similarity >= 0.7:
        return {
            "action": "update",
            "reason": f"已有 Skill '{top_fm.name}' 相似度 {top_similarity:.0%}，建议扩充此 Skill",
            "target_skill": top_fm.name,
            ...
        }

    if top_similarity >= 0.3:
        return {
            "action": "ambiguous",
            "reason": f"最佳匹配 '{top_fm.name}' 相似度仅 {top_similarity:.0%}，请Agent自行判断",
            "target_skill": None,
            ...
        }

    return {
        "action": "create",
        "reason": f"最佳匹配相似度仅 {top_similarity:.0%}，差距过大，建议创建新 Skill",
        ...
    }
```

### 2.3 `_resolve_ambiguous` — 二次判定消歧

当 ChromaDB 相似度落在 0.3-0.7 区间时，不是直接返回 `"action": "ambiguous"` 交给 Agent LLM 判断，而是先用确定性规则尝试消歧。

**判定策略（四层递进）**：

```
1. 关键词提取
   └─ 子串扫描（已知领域词）："死锁" → 匹配 AGENT_KEYWORD_TIERS 中的 Tier1/Tier2 词
   └─ 中英互译映射："死锁" → 额外加入 "deadlock"；"空指针" → 额外加入 "null pointer"
   └─ 分隔符分词：处理英文等有空格的语言

2. Tags 重叠度计算（子串匹配: "null pointer" 包含 "null"）
   对每个候选 Skill 计算:
     tag_overlap:   tags 与 problem_keywords 的重叠数
     name_overlap:  skill name 与 problem_keywords 的重叠数
     desc_overlap:  skill description 与 problem_keywords 的重叠数
     total_overlap = tag_overlap + name_overlap + desc_overlap

3. 判定
   ├─ tag_overlap ≥ 2 或 name_overlap ≥ 1
   │    → "update" (强信号，关键词高度重叠)     method=keyword_overlap_strong
   │
   ├─ tag_overlap ≥ 1 且 total_overlap ≥ 2
   │    → "update" (中等信号，部分重叠)          method=keyword_overlap_moderate
   │
   ├─ total_overlap == 0
   │    → "create" (无任何重叠，不应拼凑)         method=keyword_no_overlap
   │
   └─ 以上都不满足
        → 返回 None（无法消歧，保持 ambiguous，交给 Agent LLM）

4. 边界：空候选列表 → 直接返回 "create"
```

**测试结果**：

| 问题 | 候选 Skill | tags 重叠 | 结果 | 方法 |
|------|-----------|----------|------|------|
| Java多线程死锁并发 | concurrency-patterns (tags: concurrency, java) | "java"命中 | **update** | keyword_overlap_strong |
| 空指针null安全检测 | null-safety-tools (tags: null, safety) | "null"命中"null pointer" | **update** | keyword_overlap_strong |
| Python GIL解释器锁 | java-tools, cpp-lint | 0 | **create** | keyword_no_overlap |
| 通用描述 | tool-a, tool-b | 部分但不够阈值 | **None**(ambiguous) | — |

**为什么不直接交给 Agent LLM？**

ChromaDB 的语义相似度只衡量向量空间的距离，不关心 tags 是否匹配、领域关键词是否重叠。一个 0.45 相似度的搜索结果可能是"词向量恰好接近但实际上不相关"，也可能是"确实相关但表达方式不同"。二次判定用**确定性规则**（tags 关键词重叠）来验证这个相似度是否可信——tags 重叠了说明语义相似度是可靠的 → update；tags 完全不重叠说明相似度可能是噪音 → create；两者之间就保持 ambiguous 交给 LLM。

代码（`agent/skills/manager.py`）：

```python
async def _resolve_ambiguous(self, problem_description, search_results, top_k=3):
    """对 ambiguous 区间做二次判定，尽量给出确定性的 create/update 建议。"""
    # 提取关键词（子串扫描 + 中英互译 + 分隔符分词）
    problem_keywords = self._extract_keywords(problem_description)

    # 对每个候选计算 tags/name/description 与关键词的重叠度
    scored_candidates = []
    for fm, similarity in search_results:
        tag_overlap = sum(1 for tag in fm.tags
                         if any(tag.lower() in kw or kw in tag.lower()
                               for kw in problem_keywords))
        name_overlap = sum(1 for nw in name_words
                          if any(nw in kw or kw in nw
                                for kw in problem_keywords))
        desc_overlap = sum(1 for dw in desc_words
                          if any(dw in kw or kw in dw
                                for kw in problem_keywords))
        scored_candidates.append((fm, similarity, {
            "tag_overlap": tag_overlap,
            "name_overlap": name_overlap,
            "desc_overlap": desc_overlap,
            "total_overlap": tag_overlap + name_overlap + desc_overlap,
        }))

    best = scored_candidates[0]

    # 强信号: tags 重叠 >= 2 或 name 直接命中
    if best["tag_overlap"] >= 2 or best["name_overlap"] >= 1:
        return {"action": "update", "target_skill": best_fm.name, ...}

    # 中等信号: tags 重叠 >= 1 且 total >= 2
    if best["tag_overlap"] >= 1 and best["total_overlap"] >= 2:
        return {"action": "update", "target_skill": best_fm.name, ...}

    # 弱信号: 完全无重叠 → 新建
    if best["total_overlap"] == 0:
        return {"action": "create", ...}

    # 无法确定 → 保持 ambiguous
    return None
```

**决策流程图**：

```
decide_create_or_update(problem)
    │
    ├─ ChromaDB 搜索 → 无结果 → "create"
    │
    ├─ 相似度 > 0.7 → "update"（确定）
    │
    ├─ 相似度 < 0.3 → "create"（确定）
    │
    └─ 相似度 0.3-0.7 → _resolve_ambiguous()
         │
         ├─ tags强重叠 → "update"（关键词验证通过）
         ├─ 完全无重叠 → "create"（相似度可能是噪音）
         └─ 部分重叠   → None → 返回 "ambiguous" → Agent LLM 最终决定
```

---

## 三、SkillRouter — 分级关键词评分匹配

文件：`agent/skills/skill_router.py`（完全重写）

### 3.1 核心思想：Tier1 高特异性 / Tier2 中等 / Tier3 通用 + 排除词

**之前的问题**：扁平关键词集合，命中一个就分配。"analysis" 这种泛词会导致死锁 Skill 误匹配 code-reviewer。

**解决方案**：关键词分三级 + 不同权重 + 排除词 + 总分门槛。

```
Tier 1（权重 3）：高特异性领域词，命中一个就足以分配
  例: "死锁", "deadlock", "空指针", "null pointer", "SQL注入"

Tier 2（权重 2）：中等特异性，需要配合其他信号
  例: "根因分析", "代码审查", "并发", "补丁生成"

Tier 3（权重 1）：通用词，单独命中不计分，只在已有 Tier1/Tier2 时加分
  例: "分析", "检查", "review", "检测"

Exclude（排除）：含这些词的 Skill 明确不应分配给该 Agent
  例: 修复类 Skill 的 "fix"/"补丁"/"patch" → 排除在 reviewer/analyzer/validator 之外
```

### 3.2 分值门槛

```python
MIN_SCORE = 3      # 至少 3 分才认定匹配
TIER1_WEIGHT = 3   # 一个 Tier1 命中 = 3 分，刚好达标
TIER2_WEIGHT = 2   # 两个 Tier2 = 4 分，达标
TIER3_WEIGHT = 1   # 单个 Tier3 不达标（只有 1 分，且不在 score>0 时不计分）
```

**验证结果**：

```
死锁 Skill:     code-reviewer=0,  bug-analyzer=10,  test-validator=0  → 只匹配 analyzer ✓
空指针审查:     code-reviewer=7,  bug-analyzer=0,   code-fixer=2      → 匹配 reviewer ✓
修复 Skill:     code-fixer=10,    test-validator=0（排除词拦截）        → 只匹配 fixer ✓
单元测试:       test-validator=11                                      → 匹配 validator ✓
```

### 3.3 完整匹配代码

```python
def _scored_match(self, skill, agents) -> MatchResult:
    skill_text = self._build_skill_text(skill)  # name+description+tags 小写拼接
    scores = {}

    for agent in agents:
        tiers = self._tiers_lower.get(agent, {})
        score = 0

        # ── 先检查排除词，命中则跳过该 Agent ──
        excluded = any(kw in skill_text for kw in tiers.get("exclude", []))
        if excluded:
            continue

        # ── Tier 1（权重 3）──
        for kw in tiers.get("tier1", []):
            if kw in skill_text:
                score += self.TIER1_WEIGHT

        # ── Tier 2（权重 2）──
        for kw in tiers.get("tier2", []):
            if kw in skill_text:
                score += self.TIER2_WEIGHT

        # ── Tier 3（权重 1）—— 只在已有 Tier1/Tier2 命中时才加分 ──
        if score > 0:
            for kw in tiers.get("tier3", []):
                if kw in skill_text:
                    score += self.TIER3_WEIGHT

        scores[agent] = score

    # 筛选达标 Agent
    matched = [a for a, s in scores.items() if s >= self.MIN_SCORE]
    matched.sort(key=lambda a: scores[a], reverse=True)
    ...
```

### 3.4 三层兜底链

```python
async def route(self, skill, available_agents=None) -> MatchResult:
    agents = available_agents or list(AGENT_KEYWORD_TIERS.keys())

    # 第一层：分级关键词评分（零成本，80%+ 命中率）
    result = self._scored_match(skill, agents)
    if result.agents and result.confidence >= 0.6:
        return result

    # 第二层：LLM 语义匹配（temperature=0, max_tokens=80, 约 200 tokens/次）
    llm_agents = await self._llm_match(skill, agents)
    if llm_agents:
        result.agents = llm_agents
        result.confidence = 0.7
        result.method = "llm"
        return result

    # 第三层：多信号兜底（离线可用，动词+名词双信号加权）
    fallback = self._multi_signal_fallback(skill, agents)
    # → 检查动词信号(权重2) + 名词信号(权重3)，差距≥2 确定，否则默认 analyzer
    ...
```

---

## 四、SubAgent — 启动时自动注入 Skill

文件：`agent/delegation/sub_agent.py`

### 4.1 `__init__` 新增参数

```python
def __init__(self, ..., extra_skills: Dict[str, str] = None):
    """
    extra_skills: 动态分配的额外 Skill，格式 {skill_name: full_skill_body}
                  由 SkillRouter 匹配后传入，Agent 启动时自动注入 System Prompt
    """
    ...
    self.extra_skills = extra_skills or {}
```

### 4.2 `_build_system_message` 末尾注入

```python
def _build_system_message(self) -> str:
    base = f"""{self.definition.system_prompt}
## 可用工具
{tools_desc}
## 输出格式要求
...
"""

    # ── 动态注入额外 Skill ──
    if self.extra_skills:
        skill_sections = []
        for skill_name, skill_body in self.extra_skills.items():
            # 只取 body 部分（去掉 YAML frontmatter 省 Token）
            body_only = self._extract_skill_body(skill_body)
            # 每个 Skill 最多注入 800 字符
            truncated = body_only[:800]
            if len(body_only) > 800:
                truncated += "\n\n[... 内容已截断，完整内容可通过 load_skill 获取 ...]"
            skill_sections.append(f"### 动态加载 Skill: {skill_name}\n{truncated}")

        base += "\n## 动态加载的扩展技能（根据任务自动匹配）\n\n"
        base += "\n\n".join(skill_sections)
        base += "\n\n请结合以上扩展技能完成当前任务。\n"

    return base

def _extract_skill_body(self, content: str) -> str:
    """从完整 SKILL.md 中提取 body（去掉 YAML frontmatter）。"""
    if not content.startswith("---"):
        return content
    end_idx = content.find("\n---", 4)
    if end_idx == -1:
        return content
    return content[end_idx + 4:].lstrip("\n")
```

---

## 五、Agent 可调用 Skill 管理工具 — `build_skill_tools_for_agent`

文件：`agent/skills/tools.py`

### 5.1 问题

SubAgent 内部有完整的 ReAct 循环（Think → Act → Observe），但如果 `tools={}`（默认空字典），Agent 只能做分析和推理，无法主动搜索 Skill、创建 Skill、更新 Skill。需要把 Skill 管理的 MCP 工具包装为 SubAgent 可直接调用的函数。

### 5.2 实现

`build_skill_tools_for_agent()` 把 `SkillToolsHandler` 的 6 个方法包装为 async 函数，每个函数的签名与 MCP 工具参数一致，`__doc__` 会被 SubAgent 自动用来生成工具描述注入 System Prompt。

```python
def build_skill_tools_for_agent(
    handler: SkillToolsHandler = None,
) -> Dict[str, callable]:
    """构建 SubAgent 可用的 Skill 管理工具字典。

    返回 {tool_name: async_callable}，可以直接传给 SubAgent(tools=...).
    每个工具函数的 __doc__ 会被 SubAgent 用来自动生成工具描述。
    """
    h = handler or SkillToolsHandler()

    async def search_skills(query: str, top_k: int = 5, scope: str = None) -> dict:
        """搜索现有 Skills。当你需要判断一个问题是否能被已有 Skill 覆盖时先调用此工具。"""
        return await h.handle("search_skills", {
            "query": query, "top_k": top_k, "scope": scope,
        })

    async def load_skill(name: str) -> dict:
        """加载指定 Skill 的完整内容。当你需要查看某个 Skill 的详细步骤时调用。"""
        return await h.handle("load_skill", {"name": name})

    async def recommend_skill_action(problem: str, top_k: int = 3) -> dict:
        """判断一个问题是应该创建新 Skill 还是扩充已有 Skill。
        在决定 create_skill 还是 update_skill 之前必须先调用此工具！
        返回 action 字段: 'create'/'update'/'ambiguous'"""
        return await h.handle("recommend_skill_action", {
            "problem": problem, "top_k": top_k,
        })

    async def create_skill(name: str, description: str, body: str,
                           tags: list = None, scope: str = "user") -> dict:
        """创建全新的 Skill。当 recommend_skill_action 返回 action='create' 时调用。"""
        return await h.handle("create_skill", {
            "name": name, "description": description, "body": body,
            "tags": tags or [], "scope": scope,
        })

    async def update_skill(name: str, description: str = None,
                           body: str = None, tags: list = None) -> dict:
        """更新已有 Skill。当 recommend_skill_action 返回 action='update' 时调用。"""
        return await h.handle("update_skill", {
            "name": name, "description": description,
            "body": body, "tags": tags,
        })

    async def list_skills(scope: str = None) -> dict:
        """列出所有可用 Skills（仅返回名称和描述，不返回完整内容）。"""
        return await h.handle("list_skills", {"scope": scope})

    return {
        "search_skills": search_skills,
        "load_skill": load_skill,
        "recommend_skill_action": recommend_skill_action,
        "create_skill": create_skill,
        "update_skill": update_skill,
        "list_skills": list_skills,
    }
```

### 5.3 SubAgent 如何调用这些工具

SubAgent 已有的 `_call_tool` 和 `_describe_tools` 方法无需改动，直接兼容：

```python
# SubAgent._describe_tools() — 自动从 func.__doc__ 生成工具描述注入 System Prompt
# → System Prompt 中会出现:
#   - **search_skills**: 搜索现有 Skills。当你需要判断一个问题是否能被已有 Skill 覆盖时先调用...
#   - **recommend_skill_action**: 判断一个问题是应该创建新 Skill 还是扩充已有 Skill...
#   - **create_skill**: 创建全新的 Skill...
#   ...

# SubAgent._call_tool("search_skills", {"query": "死锁检测"})
# → 内部调用 search_skills(query="死锁检测") → handler.handle("search_skills", ...)
# → ChromaDB 语义搜索 → 返回结果给 Agent 的 Observe 阶段
```

### 5.4 Agent ReAct 循环完整示例

```
Think: "System Prompt 中没有死锁检测步骤，先搜索一下有没有相关 Skill"
    → {"action": "search_skills", "action_input": {"query": "死锁检测"}}

Act:   _call_tool("search_skills", {"query": "死锁检测"})
    → 返回: {"success": true, "results": [], "query": "死锁检测"}

Observe: "没有匹配的 Skill，需要创建新的。先让系统判断是创建还是更新"

Think: "调 recommend_skill_action 获取建议"
    → {"action": "recommend_skill_action", "action_input": {"problem": "Java多线程死锁检测"}}

Act:   _call_tool("recommend_skill_action", {"problem": "Java多线程死锁检测"})
    → 返回: {"success": true, "action": "create", "reason": "没有找到语义相近的已有 Skill"}

Observe: "系统推荐创建新 Skill"

Think: "创建死锁检测 Skill"
    → {"action": "create_skill", "action_input": {"name": "java-deadlock-detect", ...}}

Act:   _call_tool("create_skill", {...})
    → 返回: {"success": true, "skill": {"name": "java-deadlock-detect", ...}}

Observe: "新 Skill 已创建，继续执行原任务"
    → {"action": "finalize", ...}
```

---

## 六、server.py 流水线接入

文件：`server.py`

### 6.1 组件初始化

```python
# _init_components() 中新增
from agent.skills.manager import SkillManager
from agent.skills.skill_router import SkillRouter

skill_manager = SkillManager()
skill_manager.sync_all()           # Stage 1: 同步本地 Skill
skill_manager.restore_from_store() # Stage 4: 恢复持久化 Skill

skill_router = SkillRouter(llm_config={...})

# 启动时对所有 user scope 的 Skill 做预分配
all_fms = skill_manager.list_frontmatters()
_skill_assignments = {}  # {agent_name: [skill_name, ...]}
for fm in all_fms:
    if fm.scope.value in ("user", "downloaded"):
        result = await skill_router.route(fm)
        for agent in result.agents:
            _skill_assignments.setdefault(agent, []).append(fm.name)
```

### 6.2 流水线中传入 Skill 工具 + 注入动态 Skill

```python
# _run_full_pipeline() 中，创建 SubAgent 前

# ── 动态加载匹配的 Skill ──
extra_skills = {}
if skill_manager and skill_router and agent_name in skill_assignments:
    for skill_name in skill_assignments.get(agent_name, []):
        content = skill_manager.load_full_skill(skill_name)
        if content:
            extra_skills[skill_name] = content

# ── 构建 Agent 可调用的 Skill 管理工具 ──
from agent.skills.tools import SkillToolsHandler, build_skill_tools_for_agent
skill_handler = SkillToolsHandler(skill_manager)
skill_tools = build_skill_tools_for_agent(skill_handler)

# 创建 SubAgent（带上下文隔离 + 动态 Skill + Skill 管理工具）
sub = SubAgent(
    definition=registry.get(agent_name),
    task_prompt=task,
    ...
    tools=skill_tools,            # ← 6 个 Skill 管理工具
    extra_skills=extra_skills,    # ← 动态匹配的 Skill body
)
```

### 6.3 关键：所有 Agent 都能调用 Skill 工具

`skill_tools` 对流水线中**每个 Agent** 都生效——不管是 code-reviewer、bug-analyzer 还是 code-fixer，都能在自己的 ReAct 循环中调用 `search_skills`、`recommend_skill_action`、`create_skill` 等工具。Agent 不再是被动的 Skill 消费者，而是能主动扩展自身能力的自进化实体。

---

## 七、MCP 工具注册

文件：`agent/skills/tools.py`

新增 2 个工具定义 + `build_skill_tools_for_agent()`，Agent 可用工具总数 6 个：

| 工具名 | 用途 | 关键参数 |
|--------|------|---------|
| `list_skills` | 列出所有可用 Skill（仅 frontmatter） | scope? |
| `load_skill` | 加载指定 Skill 完整内容 | name |
| `search_skills` | ChromaDB 语义搜索已有 Skill | query, top_k?, scope? |
| `recommend_skill_action` | 判定创建/更新/模糊 | problem, top_k? |
| `create_skill` | 创建全新 Skill | name, description, body, tags?, scope? |
| `update_skill` | 更新已有 Skill | name, description?, body?, tags? |

---

## 八、涉及文件清单

| 文件 | 操作 | 关键改动 |
|------|------|---------|
| `agent/skills/manager.py` | 修改 | + `update_skill()` + `decide_create_or_update()` + **`_resolve_ambiguous()`** + `_extract_keywords()` + 中英术语映射 |
| `agent/skills/skill_router.py` | **重写** | 扁平关键词 → 三级评分 + 排除词 + 多信号兜底 |
| `agent/skills/tools.py` | 修改 | + `update_skill` + `recommend_skill_action` 工具定义+handler + **`build_skill_tools_for_agent()`** |
| `agent/skills/__init__.py` | 修改 | 导出 `SkillRouter`, `MatchResult`, `AGENT_KEYWORD_TIERS`, `build_skill_tools_for_agent` |
| `agent/delegation/sub_agent.py` | 修改 | + `extra_skills` 参数 + `_build_system_message()` 动态注入 + `_extract_skill_body()` |
| `server.py` | 修改 | `_init_components()` 初始化 SkillManager/SkillRouter + 预分配；`_run_full_pipeline()` 构建 skill_tools + 注入 extra_skills |
