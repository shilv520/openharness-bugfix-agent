# 离线 Memory「做梦」机制 —— 实现报告

> 新增于 2026-11-01。目标：给 skill 无限膨胀兜底 —— 以「命中统计 = memory」，
> 在每日任务结束后离线归档低频 skill、自动复活高频 skill。

---

## 一、一句话概括

把 skill 的命中统计（`hit_count` / `last_hit_at`）当作**元记忆**，每天任务结束后
由 `SkillDreamer` 离线「做梦」整理一次：

- **低频 skill** → 归档到 `ARCHIVED` scope（外置记忆，不参与默认发现，仍可语义检索/按需加载）
- **归档后重新变热的 skill** → 自动恢复到原 scope
- **高频 skill** → 保持常驻

对应 Simon Zheng 的指导：**统计机制 = memory，触发时机 = 每日任务结束后**。

---

## 二、新增 / 修改的文件

| 文件 | 类型 | 说明 |
|------|------|------|
| [agent/skills/offline_memory.py](agent/skills/offline_memory.py) | 🆕 新增 | `SkillDreamer.dream()` 离线整理逻辑 |
| [scripts/test_offline_memory.py](scripts/test_offline_memory.py) | 🆕 新增 | 自测脚本（纯 fallback，不依赖 Redis/ChromaDB） |
| [agent/skills/progressive.py](agent/skills/progressive.py) | ✏️ 修改 | `SkillScope` 加 `ARCHIVED`；`SkillFrontmatter` 加 `hit_count` / `last_hit_at` / `archived_from` |
| [agent/skills/persistence.py](agent/skills/persistence.py) | ✏️ 修改 | 读写新字段；新增 `record_usage()`、`store_memory()` / `get_memory()` |
| [agent/skills/manager.py](agent/skills/manager.py) | ✏️ 修改 | `load_full_skill` 命中即记录 + 自动复活；`list_frontmatters` 默认过滤 `ARCHIVED`；新增 `archive_skill()` / `restore_skill()` |
| [agent/skills/__init__.py](agent/skills/__init__.py) | ✏️ 修改 | 导出 `SkillDreamer` |
| [server.py](server.py) | ✏️ 修改 | 挂「每日任务结束后」触发 + `POST /api/skills/dream` 手动入口 |

---

## 三、核心设计

### 1. 统计机制 = memory

每个 skill 新增三个元记忆字段：

```
hit_count      命中次数（累计）
last_hit_at    最近命中时间（ISO）
archived_from  归档前的原 scope（用于恢复，空 = 未归档）
```

`load_full_skill()` 每次命中即 `record_usage()`，把「被用了几次、最近什么时候用」沉淀成记忆。

### 2. 离线「做梦」= 频率分层

`SkillDreamer.dream()` 每日跑一次，时间窗判定（**避免抖动**）：

| 状态 | 条件 | 动作 |
|------|------|------|
| 常驻 skill | last_hit/created 超过 `archive_after_days`（默认 7 天） | → **归档** |
| 归档 skill | last_hit 在 `restore_recent_days`（默认 3 天）内 | → **恢复** |
| 其余 | 活跃 | → **保持** |

> 为什么用时间窗而不是累计 `hit_count` 阈值？累计值会导致「高频过但已冷」的 skill
> 归档后下一轮立刻被恢复，来回抖动。时间窗反映的是**最近**热度。

### 3. 用 memory 兜底

dream 的决策报告本身写入 memory（`skill:dream:YYYYMMDD`），跨会话可查、可审计、可回滚。

### 4. 触发时机

```
bug 任务结束（finally / sync 返回前）
  → _maybe_run_dream()          # 每日守卫：今天跑过就跳过
    → SkillDreamer.dream()       # 线程池执行，不阻塞事件循环
```

- 每日只跑一次（Redis / 内存 fallback 记录上次日期，跨重启不重复）
- dream 失败只打 warning，绝不拖垮主任务
- `POST /api/skills/dream?force=true` 可手动强制触发（调试用）

---

## 四、数据流

```
命中 skill
  → load_full_skill(name)
      ├─ record_usage(name)   # hit_count+1, last_hit_at=now
      └─ restore_skill(name)  # 若已归档 → 复活到原 scope

每日任务结束
  → dream()
      ├─ 冷 skill（>7 天没碰） → archive 到 ARCHIVED
      ├─ 归档 skill（近 3 天命中）→ restore 回原 scope
      └─ 报告写入 memory（可审计）
```

---

## 五、测试验证

`python scripts/test_offline_memory.py`（纯 fallback，无需 Redis/ChromaDB）：

```
summary : dream @ ...: archived=1, restored=1, kept=1
archived: ['stale-skill']      # 30 天没碰 → 归档
restored: ['warm-again']       # 归档后又被命中 → 复活
kept    : ['hot-skill']        # 2 小时前命中 → 常驻
[OK] ALL ASSERTIONS PASSED
```

另验证 `load_full_skill`：归档 skill 加载后 scope 从 `archived` → `user`，`hit_count` 0 → 1。

---

## 六、面试对应点（Q10 / Q12）

| 面试题 | 本机制的答案 |
|--------|--------------|
| Q10：skill 会不会无限膨胀？ | 会。用 memory 兜底：低频 skill 归档到 ARCHIVED（外置记忆），不参与默认发现 |
| Q12：统计机制是不是 memory？触发时机？ | 是。`hit_count`/`last_hit_at` 就是元记忆；触发时机 = 每日任务结束后离线「做梦」 |
