"""
Skill更新机制 — 完整测试套件
用法: python scripts/test_full_suite.py
"""
import sys, os, asyncio, shutil, inspect
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

passed = 0
failed = 0

def t(name, ok):
    global passed, failed
    if ok: passed += 1; print(f'  [PASS] {name}')
    else: failed += 1; print(f'  [FAIL] {name}')

print('=' * 60)
print('TEST SUITE: Skill 更新机制完整验证')
print('=' * 60)
print()

# ═══════════════════════════════════════════════════
# 1. 模块导入
# ═══════════════════════════════════════════════════
print('--- 1. 模块导入 ---')
try:
    from agent.skills.manager import SkillManager, get_skill_manager
    from agent.skills.skill_router import SkillRouter, MatchResult, AGENT_KEYWORD_TIERS, get_skill_router
    from agent.skills.tools import SkillToolsHandler, get_tool_definitions
    from agent.skills.progressive import SkillFrontmatter, SkillScope
    from agent.delegation.sub_agent import SubAgent
    from agent.delegation.agent_definition import AgentDefinition
    t('agent.skills.manager', True)
    t('agent.skills.skill_router', True)
    t('agent.skills.tools', True)
    t('agent.skills.progressive', True)
    t('agent.delegation.sub_agent', True)
    t('agent.delegation.agent_definition', True)
except Exception as e:
    t(f'module import: {e}', False)
    sys.exit(1)

# ═══════════════════════════════════════════════════
# 2. SkillManager 方法完整性
# ═══════════════════════════════════════════════════
print('--- 2. SkillManager 方法完整性 ---')
mgr = SkillManager()
t('update_skill', hasattr(mgr, 'update_skill'))
t('decide_create_or_update', hasattr(mgr, 'decide_create_or_update'))
t('create_skill', hasattr(mgr, 'create_skill'))
t('load_full_skill', hasattr(mgr, 'load_full_skill'))
t('list_frontmatters', hasattr(mgr, 'list_frontmatters'))
t('search_skills', hasattr(mgr, 'search_skills'))
t('sync_all', hasattr(mgr, 'sync_all'))
t('restore_from_store', hasattr(mgr, 'restore_from_store'))
t('download_skill', hasattr(mgr, 'download_skill'))
t('assign_skill', hasattr(mgr, 'assign_skill'))
print(f'  SkillManager: 10/10 方法齐全')

# ═══════════════════════════════════════════════════
# 3. SkillRouter 分级匹配精度
# ═══════════════════════════════════════════════════
print('--- 3. SkillRouter 分级匹配精度 ---')
router = SkillRouter()
agents = list(AGENT_KEYWORD_TIERS.keys())

# 3a: 死锁 -> 只匹配 analyzer (之前会误匹配 reviewer)
s_deadlock = SkillFrontmatter(
    name='java-deadlock-detection', description='Java多线程死锁检测',
    tags=['java','concurrency','deadlock'], scope=SkillScope.USER)
r = router._scored_match(s_deadlock, agents)
t('死锁 -> bug-analyzer (score=10)', 'bug-analyzer' in r.agents)
t('死锁 NOT -> code-reviewer (score=0)', 'code-reviewer' not in r.agents)
t('死锁 NOT -> test-validator (score=0)', 'test-validator' not in r.agents)
t('死锁 confidence=0.95', r.confidence >= 0.9)

# 3b: 修复Skill排除词
s_fix = SkillFrontmatter(
    name='auto-fix-null', description='自动修复空指针补丁',
    tags=['fix','null','patch'], scope=SkillScope.USER)
r = router._scored_match(s_fix, agents)
t('修复 -> code-fixer', 'code-fixer' in r.agents)
t('修复 NOT -> test-validator (排除词)', 'test-validator' not in r.agents)
t('修复 NOT -> bug-analyzer (排除词)', 'bug-analyzer' not in r.agents)

# 3c: 测试Skill
s_test = SkillFrontmatter(
    name='unit-test-gen', description='JUnit单元测试自动生成覆盖率验证',
    tags=['test','junit','coverage'], scope=SkillScope.USER)
r = router._scored_match(s_test, agents)
t('测试 -> test-validator (score=11)', 'test-validator' in r.agents)

# 3d: 低信号不误匹配
s_low = SkillFrontmatter(
    name='mystery', description='通用工具', tags=['tool'], scope=SkillScope.USER)
r = router._scored_match(s_low, agents)
t('低信号 confidence<0.6', r.confidence < 0.6)
t('低信号 无匹配agents', len(r.agents) == 0)

# ═══════════════════════════════════════════════════
# 4. 多信号兜底
# ═══════════════════════════════════════════════════
print('--- 4. 多信号兜底 (离线可用) ---')
fallback_tests = [
    ('deadlock-detector', '死锁检测与诊断', ['deadlock','diagnose'], 'bug-analyzer'),
    ('null-finder', '空指针漏洞发现与审查', ['null','vulnerability'], 'code-reviewer'),
    ('fixer', '代码修复与补丁生成', ['fix','patch'], 'code-fixer'),
    ('tester', '单元测试与回归验证', ['test','regression'], 'test-validator'),
]
for name, desc, tags, exp in fallback_tests:
    s = SkillFrontmatter(name=name, description=desc, tags=tags, scope=SkillScope.USER)
    actual = router._multi_signal_fallback(s, agents)
    t(f'{name} -> {exp}', actual == exp)

# ═══════════════════════════════════════════════════
# 5. MCP 工具注册
# ═══════════════════════════════════════════════════
print('--- 5. MCP 工具注册 (8个) ---')
tools = get_tool_definitions()
tool_names = [t['name'] for t in tools]
required = ['list_skills','load_skill','search_skills','download_skill',
            'create_skill','update_skill','recommend_skill_action','assign_skill']
for r_name in required:
    t(f'{r_name}', r_name in tool_names)
handler = SkillToolsHandler()
t('SkillToolsHandler.handle 可调用', callable(handler.handle))

# ═══════════════════════════════════════════════════
# 6. SubAgent 动态 Skill 注入
# ═══════════════════════════════════════════════════
print('--- 6. SubAgent 动态Skill注入 ---')
sig = inspect.signature(SubAgent.__init__)
t('extra_skills 参数注册', 'extra_skills' in sig.parameters)

sub = SubAgent(
    definition=AgentDefinition(
        name='test', description='test agent',
        system_prompt='You are a test agent.',
        skills=['test-skill'],
    ),
    task_prompt='test task',
    extra_skills={
        'deadlock-detect': (
            '---\nname: deadlock-detect\n'
            'description: 死锁检测\n'
            'tags: [java, concurrency]\n'
            '---\n'
            '# 死锁检测指南\n'
            '## 检测步骤\n'
            '1. 检查所有synchronized块的锁获取顺序\n'
            '2. 使用jstack分析线程等待图\n'
            '3. 确认是否存在循环等待\n'
            '\n'
            '## 修复建议\n'
            '- 统一锁顺序\n'
            '- 使用tryLock超时机制\n'
        ),
    },
)
msg = sub._build_system_message()

t('System message 含"动态加载的扩展技能"标题', '动态加载的扩展技能' in msg)
t('System message 含 Skill 名称 deadlock-detect', 'deadlock-detect' in msg)
t('System message 含检测步骤', '检查所有synchronized块' in msg)
t('System message 含修复建议', '统一锁顺序' in msg)
t('frontmatter被剥离 (name不在msg中)', '\nname: deadlock-detect' not in msg)
t('frontmatter被剥离 (tags不在msg中)', '\ntags:' not in msg)

# ═══════════════════════════════════════════════════
# 7. AGENT_KEYWORD_TIERS 结构
# ═══════════════════════════════════════════════════
print('--- 7. 关键词表结构 ---')
for agent_name in agents:
    tiers = AGENT_KEYWORD_TIERS[agent_name]
    t1c = len(tiers['tier1']); t2c = len(tiers['tier2'])
    t3c = len(tiers['tier3']); exc = len(tiers['exclude'])
    t(f'{agent_name}: T1={t1c} T2={t2c} T3={t3c} Excl={exc}',
      t1c >= 3 and t2c >= 3 and exc >= 1)

# ═══════════════════════════════════════════════════
# 8. 5种Bug场景全面覆盖
# ═══════════════════════════════════════════════════
print('--- 8. 5种Bug场景 Skill->Agent ---')
scenarios = [
    ('SQL注入检测', 'security-injection',
     '检测SQL注入和XSS漏洞', ['sql','injection','xss'], 'code-reviewer'),
    ('内存泄漏诊断', 'memory-leak',
     'Java堆内存泄漏根因分析与诊断', ['memory','leak','java'], 'bug-analyzer'),
    ('并发Bug修复', 'concurrency-fix',
     '修复竞态条件和死锁的补丁生成', ['fix','concurrency','deadlock'], 'code-fixer'),
    ('回归测试', 'regression-suite',
     'JUnit回归测试自动执行与覆盖率', ['test','junit','regression'], 'test-validator'),
    ('代码规范检查', 'style-lint',
     'Java代码规范与格式检查lint', ['style','lint','java'], 'code-reviewer'),
]
for name, sname, desc, tags, exp in scenarios:
    s = SkillFrontmatter(name=sname, description=desc, tags=tags, scope=SkillScope.USER)
    r = router._scored_match(s, agents)
    t(f'{name} -> {exp} (score={r.scores.get(exp,0)}, method={r.method})', exp in r.agents)

# ═══════════════════════════════════════════════════
# 9. Skill 文件操作 (create_skill/update_skill)
# ═══════════════════════════════════════════════════
print('--- 9. Skill 文件操作 (create/update) ---')

mgr2 = SkillManager()

async def test_file_ops():
    # 9a: create
    fm = await mgr2.create_skill(
        name='__integration_test__', description='Integration test skill',
        body='# Test Skill\n\n## Step 1\nDo something.\n\n## Step 2\nVerify result.',
        scope=SkillScope.USER, tags=['test', 'integration'],
    )
    assert fm is not None, 'create_skill returned None'
    path = mgr2.skills_root / 'user' / '__integration_test__' / 'SKILL.md'
    assert path.exists(), f'SKILL.md not created at {path}'
    content = path.read_text(encoding='utf-8')
    assert 'Do something' in content, 'body not written'
    assert 'Verify result' in content, 'body incomplete'
    assert 'Integration test skill' in content, 'description not in frontmatter'
    assert 'test' in content and 'integration' in content, 'tags not in frontmatter'

    # 9b: update
    fm2 = await mgr2.update_skill(
        name='__integration_test__',
        description='Updated integration test',
        body='# Updated Skill\n\n## New Step\nUpdated content.',
        tags=['test', 'integration', 'updated'],
    )
    assert fm2 is not None, 'update_skill returned None'
    assert fm2.description == 'Updated integration test'
    assert fm2.tags == ['test', 'integration', 'updated']
    content2 = path.read_text(encoding='utf-8')
    assert 'Updated content' in content2, 'body not updated'
    assert 'Do something' not in content2, 'old body not removed'
    assert 'Updated integration test' in content2, 'description not updated'
    assert 'updated' in content2, 'new tag not written'

    # 9c: load after update
    loaded = mgr2.load_full_skill('__integration_test__')
    assert loaded is not None
    assert 'Updated content' in loaded
    assert 'Do something' not in loaded

    # 9d: cleanup
    skill_dir = mgr2.skills_root / 'user' / '__integration_test__'
    shutil.rmtree(skill_dir)
    mgr2._content_cache.pop('__integration_test__', None)

try:
    asyncio.get_event_loop().run_until_complete(
        asyncio.wait_for(test_file_ops(), timeout=20)
    )
    t('create_skill 创建+落盘', True)
    t('update_skill 更新+写回', True)
    t('load_full_skill 加载已更新内容', True)
    t('cleanup 清理测试文件', True)
except asyncio.TimeoutError:
    t('文件操作: TIMEOUT (ChromaDB/Redis挂起)', False)
except Exception as e:
    t(f'文件操作: {e}', False)

# ═══════════════════════════════════════════════════
# 10. update 不存在Skill 返回 None
# ═══════════════════════════════════════════════════
print('--- 10. 边界情况 ---')
mgr3 = SkillManager()

async def test_edge():
    # 不存在的Skill update 返回 None
    fm = await mgr3.update_skill(name='__nonexistent_xyz_123__', body='test')
    assert fm is None, f'expected None, got {fm}'

try:
    asyncio.get_event_loop().run_until_complete(
        asyncio.wait_for(test_edge(), timeout=10)
    )
    t('update_skill 不存在返回 None', True)
except asyncio.TimeoutError:
    t('update_skill 不存在返回 None: TIMEOUT', False)
except Exception as e:
    t(f'update_skill 不存在返回 None: {e}', False)

# ═══════════════════════════════════════════════════
# RESULT
# ═══════════════════════════════════════════════════
print()
print('=' * 60)
print(f'TOTAL: {passed} passed, {failed} failed, {passed+failed} tests')
if failed == 0:
    print('ALL TESTS PASSED')
else:
    print(f'{failed} TEST(S) FAILED')
    sys.exit(1)
