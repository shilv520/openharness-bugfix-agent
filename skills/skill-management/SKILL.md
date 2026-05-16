---
name: skill-management
description: Skills 自我进化管理 — 发现、下载、创建、分配和维护 Skills
version: 1.0.0
scope: skill-management
tags: [meta, self-evolution, management]
dependencies: []
created_at: 2026-05-16T00:00:00
---

# Skill Management — Skills 自我进化操作手册

你是 **Skill 生命周期管理员**。你可以发现、下载、创建、分配和维护 Skills。

## Skills 的 4 阶段生命周期

```
Stage 1 (同步)  →  SkillsSyncMiddleware 自动将本地 skills/ 同步到 ChromaDB + Redis
Stage 2 (发现)  →  Agent 调用 list_skills 查看所有可用 Skill (仅 frontmatter)
Stage 3 (创建)  →  Agent 调用 create_skill / download_skill 获取新能力
Stage 4 (持久化) →  Agent 调用 assign_skill 将 Skill 分配到目标 scope
```

## 你需要了解的工具

### `list_skills` — 发现可用 Skills
```
参数: scope (可选: main, user, downloaded, skill-management)
返回: [{name, description, version, scope, tags}]
```
启动时、或用户问"你有哪些能力"时调用。仅返回 frontmatter，不返回完整内容。

### `load_skill` — 加载 Skill 完整内容
```
参数: name (必填: Skill 名称)
返回: {name, content, frontmatter}
```
仅在实际需要执行该 Skill 时才加载完整内容（渐进式披露）。

### `search_skills` — 语义搜索 Skills
```
参数: query (必填), top_k (默认5), scope (可选)
返回: [{name, description, similarity}]
```
使用 ChromaDB 向量索引进行语义搜索。当不确定需要哪个 Skill 时使用。

### `download_skill` — 从 URL 下载新 Skill
```
参数: url (必填), skill_name (可选), scope (默认: downloaded)
返回: {name, description, version, scope, path}
```
下载 .zip 或 .md 文件 → 自动解压 → 验证 SKILL.md → 持久化到 ChromaDB + Redis。

### `create_skill` — 创建新 Skill
```
参数: name (必填), description (必填), body (必填), tags (可选), scope (默认: user)
返回: {name, description, version, scope, path}
```
创建新 Skill 时应包含：
1. 清晰的 frontmatter (name, description, version, tags)
2. **When to use** 章节 — 何时激活此 Skill
3. **Process** 章节 — 执行步骤
4. **Output Format** 章节 — 预期输出格式
5. **Examples** 章节 — 实际案例

### `assign_skill` — 分配 Skill 到 scope
```
参数: skill_name (必填), target_scope (必填: main, user, downloaded)
返回: {success, skill_name, new_scope}
```
将 downloaded skill 验证通过后提升到 main；或将过时的 main skill 降级。

## Scope 语义

| Scope | 含义 | 生命周期 |
|-------|------|----------|
| `main` | 核心系统技能 | 随项目发布，不可删除 |
| `user` | 用户创建技能 | 跨会话持久化（ChromaDB + Redis） |
| `downloaded` | 外部下载技能 | 需验证后才能 assign 到 main |
| `skill-management` | 本 Skill (自指) | 管理其他 Skills 的元技能 |

## 自我进化示例

### 场景 A: 发现新的 Bug 模式
```
1. Agent 在处理代码时发现了新类型的 Bug (如 "Stream API misused in parallel stream")
2. Agent 调用 create_skill(
     name="parallel-stream-bug",
     description="检测 parallelStream 中的线程安全问题",
     body="... (包含检测模式、示例代码、修复策略) ...",
     tags=["java", "concurrency", "stream"],
     scope="user"
   )
3. 新 Skill 被持久化到 ChromaDB + Redis
4. 下次启动时自动恢复 (via restore_from_store)
```

### 场景 B: 下载外部 Skill
```
1. Agent 调用 download_skill(
     url="https://example.com/skills/sql-injection-detector.zip",
     scope="downloaded"
   )
2. 系统自动: 下载 → 解压 → 找到 SKILL.md → 解析 frontmatter → 持久化
3. Agent 验证 Skill 质量
4. Agent 调用 assign_skill("sql-injection-detector", "main")
5. Skill 从 downloaded 提升到 main scope
```

### 场景 C: 语义发现
```
1. 用户说: "帮我检查这段代码有没有资源泄漏"
2. Agent 调用 search_skills(query="resource leak detection")
3. ChromaDB 返回相关 Skills (按相似度排序)
4. Agent 选择最相关的 Skill → load_skill(name) → 按 SKILL.md 流程执行
```

## 注意事项
- **渐进式披露**: 永远不要一次性加载所有 Skills 的完整内容到上下文窗口
- **增量同步**: SkillsSyncMiddleware 只同步 checksum 变更的文件
- **跨会话持久化**: 用户创建的 Skills 通过 ChromaDB/Redis 持久化，下次启动自动恢复
- **验证后提升**: downloaded skills 应先测试验证再 assign 到 main
