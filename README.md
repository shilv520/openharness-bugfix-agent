# OpenHarness BugFix Agent v3.0

> **15+ 组件集成的多智能体代码 Bug 自动修复系统** — 提交代码 → 4 个专业 AI Agent 协作修复 → 返回补丁与验证报告

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61dafb.svg)](https://react.dev/)
[![Redis](https://img.shields.io/badge/Redis-7-red.svg)](https://redis.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-orange.svg)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-24+-2496ed.svg)](https://www.docker.com/)

---

## 项目简介

OpenHarness BugFix Agent 是一个**多智能体协作的自动化 Bug 修复平台**。用户通过 Web 界面提交有 Bug 的代码片段，系统自动调度 4 个专业 AI Agent（代码审查 → 根因分析 → 补丁生成 → 验证测试）按序协作，完成从 Bug 发现到修复验证的全流程。系统集成了 **15+ 企业级组件**，涵盖 Agent 协作、上下文管理、记忆持久化、安全沙箱、人机协同等完整能力。

### 核心工作流

```
用户 Web 端提交 Bug 代码
         │
         ▼
  动态 Agent 规划器 ──→ 按 Bug 描述自动选择 Agent 组合
         │
         ▼
  ┌─ code-reviewer ──► bug-analyzer ──► code-fixer ──► test-validator ─┐
  │   (审查找Bug)        (根因分析)       (生成补丁)      (验证修复)      │
  │       │                  │               │               │         │
  │       └── DiscussionProtocol ─┘          └── FeedbackProtocol ──┘   │
  │         (共识: Bug真假?)                  (反馈: 最多3次迭代)         │
  └───────────────────────────────────────────────────────────────────┘
         │
         ▼
   HITL 审批 ──→ 返回 patch + fixed_code + 验证报告
```

---

## 系统架构

```
┌─────────────────────────────────────────────────────┐
│              第 1 层：Web 服务与 API                  │
│  React SPA  │  FastAPI  │  JWT Auth  │  Redis        │
├─────────────────────────────────────────────────────┤
│            第 2 层：多智能体协作引擎                   │
│  AgentRegistry  │  SubAgent  │  CommunicationBus     │
│  ContextIsolation (Fork-Merge-Clean)                │
│  ContextCompressor (三层压缩)                        │
│  动态 Agent 规划 (关键词启发式 + LLM 分类回退)        │
├─────────────────────────────────────────────────────┤
│            第 3 层：记忆与持久化                      │
│  HierarchicalMemory (Redis + ChromaDB)              │
│  TrajectoryStore (全链路轨迹)                        │
│  TaskPersistence (快照恢复)                          │
│  Skills 系统 (渐进式披露 + 四阶段进化)                │
├─────────────────────────────────────────────────────┤
│            第 4 层：基础设施与安全                    │
│  CompositeBackend  │  SecuritySandbox (Docker)       │
│  HITLManager (双层人机协同)  │  MCP 工具集成          │
└─────────────────────────────────────────────────────┘
```

---

## 技术亮点

### 1. Fork-Merge-Clean 上下文隔离

借鉴操作系统进程 fork 模型，子 Agent 只继承必要的只读事实（代码路径、语言类型、任务描述），父 Agent 的完整历史和兄弟 Agent 的中间结果对子 Agent 完全不可见。执行完成后只合并结构化 JSON 结果，**单任务 Token 消耗从 ~60,000 降至 ~18,000（↓70%）**。

### 2. Skill 系统 — 自进化能力

- **渐进式披露**：启动时只读 Skill 元数据（~50 token/个），按需加载完整内容，启动 Token 节省 **93%**
- **四阶段进化**：Sync → Discover → Create/Download → Assign/Persist
- **Agent 自主管理**：Agent 在 ReAct 循环中可主动调用 `search_skills`、`recommend_skill_action`、`create_skill`、`update_skill` 等工具
- **Skill→Agent 自动匹配**：分级关键词评分（Tier1/Tier2/Tier3 + 排除词）+ LLM 兜底 + 多信号推断
- **创建/更新决策**：ChromaDB 语义搜索 → 相似度阈值判定 → 二次关键词消歧

### 3. 动态 Agent 规划

关键词启发式匹配（零成本、零延迟）→ LLM 轻量分类（回退）→ 智能依赖补全。简单 Bug 只用 2-3 个 Agent，减少 **~45%** 不必要 LLM 调用。

### 4. 全链路容错降级

| 层级 | 故障 | 降级方案 |
|------|------|---------|
| 认证 | JWT 过期/无效 | 自动降级 anonymous，不阻塞 |
| 数据 | Redis 不可用 | Python dict 内存后备 |
| 执行 | Docker 不可用 | LocalFSBackend 本地文件系统 |
| Agent | 单 Agent 失败 | Replanner 重规划（最多 3 次）|
| 进程 | 进程崩溃 | TaskPersistence 快照恢复 + Session Resume API |
| Skill | YAML 不可用 | 硬编码 fallback 定义 |

---

## 快速开始

### 环境要求

- Python 3.10+
- Redis（可选，无 Redis 时自动降级为内存存储）
- Docker（可选，用于安全沙箱执行）
- Node.js 18+（前端）

### 1. 克隆项目

```bash
git clone https://github.com/shilv520/openharness-bugfix-agent.git
cd openharness-bugfix-agent
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少填入 LLM API Key
```

`.env` 必要配置：

```env
DEEPSEEK_API_KEY=sk-your-key-here        # LLM API Key
OPENAI_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat
REDIS_HOST=localhost
JWT_SECRET=your-secret-key-change-in-production
```

### 3. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

### 4. 启动后端

```bash
python server.py
# 或
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

API 文档：http://localhost:8000/docs

### 5. 启动前端（可选）

```bash
cd frontend
npm install
npm run dev
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/bug/submit` | 提交 Bug 修复任务（异步） |
| `POST` | `/api/bug/submit-sync` | 提交 Bug 修复任务（同步） |
| `GET` | `/api/bug/task/{task_id}` | 查询任务状态和结果 |
| `GET` | `/api/bug/tasks` | 列出所有任务 |
| `POST` | `/api/register` | 用户注册 |
| `POST` | `/api/login` | 用户登录 (JWT) |
| `GET` | `/api/hitl/pending` | 列出待审批的 HITL 请求 |
| `POST` | `/api/hitl/approve` | 审批 HITL 请求 |
| `POST` | `/api/session/resume` | 恢复崩溃的会话 |
| `GET` | `/api/health` | 系统健康检查 + 组件状态 |
| `GET` | `/api/stats` | 系统统计信息 |

### 提交 Bug 示例

```bash
curl -X POST http://localhost:8000/api/bug/submit-sync \
  -H "Content-Type: application/json" \
  -d '{
    "code": "public void process() { String s = null; s.length(); }",
    "language": "java",
    "description": "空指针异常风险",
    "severity": "HIGH"
  }'
```

---

## 项目结构

```
openharness-bugfix-agent/
├── server.py                    # FastAPI 主入口，流水线编排
├── agent/
│   ├── backends/                # 虚拟文件系统 (CompositeBackend + LocalFS + Memory)
│   ├── delegation/              # 多 Agent 委派系统
│   │   ├── agent_definition.py  # YAML 驱动的 Agent 定义
│   │   ├── agent_registry.py    # Agent 注册中心（全局单例）
│   │   ├── sub_agent.py         # 子 Agent 运行时 (ReAct 循环)
│   │   └── task_tool.py         # 任务委派工具
│   ├── sandbox/                 # Docker 安全沙箱
│   ├── skills/                  # Skills 技能系统
│   │   ├── manager.py           # SkillManager (4-Stage 生命周期)
│   │   ├── skill_router.py      # Skill→Agent 自动匹配路由器
│   │   ├── progressive.py       # 渐进式披露 + Frontmatter 解析
│   │   ├── persistence.py       # ChromaDB + Redis 持久化
│   │   ├── tools.py             # MCP 工具定义 + Agent 工具包装器
│   │   └── bundled/             # 内置 5 个 Skill
│   ├── context_isolation.py     # 上下文隔离 (Fork-Merge-Clean)
│   ├── context_compressor.py    # 上下文压缩 (三层递进)
│   ├── human_in_the_loop.py     # HITL 人机协同 (双层中断)
│   ├── task_persistence.py      # 任务持久化 + 崩溃恢复
│   ├── communication.py         # Agent 通信总线 + DiscussionProtocol + FeedbackProtocol
│   ├── redis_memory.py          # 分层记忆 (Redis + ChromaDB)
│   ├── trajectory_store.py      # 全链路轨迹记录
│   ├── intent_matcher.py        # 意图匹配引擎 (Embedding + LLM)
│   ├── planner.py               # 任务计划生成
│   ├── executor.py              # Agent 步骤执行器
│   └── replanner.py             # 失败重规划
├── frontend/                    # React SPA 前端
│   └── src/
│       ├── pages/
│       │   ├── Dashboard.jsx    # 控制台
│       │   ├── LoginPage.jsx    # 登录
│       │   └── TaskDetail.jsx   # 任务详情
│       └── components/
│           ├── CodeForm.jsx     # Bug 提交表单
│           ├── PipelineProgress.jsx  # 流水线进度
│           ├── ResultsPanel.jsx # 结果展示
│           ├── StatsCards.jsx   # 统计卡片
│           └── TaskHistory.jsx  # 历史任务
├── scripts/                     # 演示和测试脚本
├── data/benchmark/              # 基准测试套件
└── data/workspace/              # Agent 工作区
```

---

## 15+ 集成组件一览

| 组件 | 文件 | 功能 |
|------|------|------|
| **CompositeBackend** | `backends/composite.py` | 虚拟文件系统路由，透明切换 LocalFS/Redis/ChromaDB |
| **SecuritySandbox** | `sandbox/sandbox.py` | Docker 容器隔离执行，禁网络、限资源 |
| **ContextIsolation** | `context_isolation.py` | Fork-Merge-Clean 上下文隔离 |
| **ContextCompressor** | `context_compressor.py` | Token 计数 + 滑动窗口 + LLM 摘要卸载 |
| **HITLManager** | `human_in_the_loop.py` | 双层人机协同中断 |
| **TaskPersistence** | `task_persistence.py` | 任务快照、崩溃恢复、Session Resume |
| **AgentRegistry** | `delegation/agent_registry.py` | YAML 驱动 Agent 注册中心 |
| **SubAgent** | `delegation/sub_agent.py` | 动态创建/销毁子 Agent，ReAct 推理循环 |
| **CommunicationBus** | `communication.py` | Agent 间消息传递 + 讨论/反馈协议 |
| **HierarchicalMemory** | `redis_memory.py` | 分层记忆（短期 Redis + 长期 ChromaDB） |
| **TrajectoryStore** | `trajectory_store.py` | Think/Act/Observe 全链路轨迹 |
| **SkillManager** | `skills/manager.py` | Skills 四阶段生命周期管理 |
| **SkillRouter** | `skills/skill_router.py` | Skill→Agent 分级关键词自动匹配 |
| **IntentMatcher** | `intent_matcher.py` | Embedding + LLM 两层意图匹配 |
| **Planner/Replanner** | `planner.py` / `replanner.py` | 任务计划生成与失败重规划 |

---

## 技术栈

| 层 | 技术 |
|----|------|
| **后端框架** | FastAPI (异步), Uvicorn |
| **前端** | React 18, Vite, React Router |
| **LLM** | DeepSeek / OpenAI 兼容 API |
| **向量数据库** | ChromaDB（语义搜索 + 经验复用） |
| **缓存/存储** | Redis 7（优先）+ Python dict（回退） |
| **容器化** | Docker（Agent 安全沙箱） |
| **认证** | JWT (HS256, 24h 过期) |
| **协议** | MCP (Model Context Protocol, stdio + HTTP SSE) |

---

## 学习资源

项目中包含详细的架构文档和学习笔记：

- [ARCHITECTURE.md](./ARCHITECTURE.md) — 系统架构详解
- [Skill更新机制改动笔记.md](./Skill更新机制改动笔记.md) — Skill 动态更新与自动分配完整文档
- `LEARNING_NOTES_*.md` — 各组件源码级学习笔记

---

## License

MIT
