"""完整测试 Skill 更新机制所有改动"""
import sys, os, asyncio, shutil, inspect, traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

passed = 0
failed = 0
errors = []
results = []

def test(name, func):
    global passed, failed
    try:
        func()
        results.append(f'  [PASS] {name}')
        print(f'  [PASS] {name}')
    except Exception as e:
        msg = f'  [FAIL] {name}: {e}'
        results.append(msg)
        print(msg)
        errors.append((name, str(e)))
        failed += 1
        return
    passed += 1

print('=' * 60)
print('PHASE 1: Skill 核心模块测试')
print('=' * 60)

# ── 1. 导入检查 ──
def t1():
    from agent.skills.manager import SkillManager, get_skill_manager
    from agent.skills.skill_router import SkillRouter, MatchResult, AGENT_KEYWORD_TIERS, get_skill_router
    from agent.skills.tools import SkillToolsHandler, get_tool_definitions
    from agent.skills.progressive import SkillFrontmatter, SkillScope
    from agent.delegation.sub_agent import SubAgent
    from agent.delegation.agent_definition import AgentDefinition
    print('    All imports OK')
test('所有模块导入成功', t1)

# ── 2. SkillManager 8个方法 ──
def t2():
    from agent.skills.manager import SkillManager
    mgr = SkillManager()
    for m in ['update_skill','decide_create_or_update','create_skill','load_full_skill',
              'list_frontmatters','search_skills','sync_all','restore_from_store']:
        assert hasattr(mgr, m), f'missing {m}'
test('SkillManager 8个核心方法齐全', t2)

# ── 3. SkillRouter 死锁不误匹配 ──
def t3():
    from agent.skills.skill_router import SkillRouter, AGENT_KEYWORD_TIERS
    from agent.skills.progressive import SkillFrontmatter, SkillScope
    router = SkillRouter()
    s = SkillFrontmatter(name='java-deadlock-detection', description='Java多线程死锁检测',
                         tags=['java','concurrency','deadlock'], scope=SkillScope.USER)
    r = router._scored_match(s, list(AGENT_KEYWORD_TIERS.keys()))
    assert 'bug-analyzer' in r.agents, f'deadlock must match analyzer: {r.agents}'
    assert 'code-reviewer' not in r.agents, f'deadlock NOT reviewer: scores={r.scores}'
    assert r.scores['bug-analyzer'] >= 8, f'score too low: {r.scores}'
test('死锁Skill只匹配analyzer(cr=0, ba=10)', t3)

# ── 4. 排除词 ──
def t4():
    from agent.skills.skill_router import SkillRouter, AGENT_KEYWORD_TIERS
    from agent.skills.progressive import SkillFrontmatter, SkillScope
    router = SkillRouter()
    s = SkillFrontmatter(name='auto-fix-null', description='自动修复空指针补丁',
                         tags=['fix','null','patch'], scope=SkillScope.USER)
    r = router._scored_match(s, list(AGENT_KEYWORD_TIERS.keys()))
    assert 'code-fixer' in r.agents
    assert 'test-validator' not in r.agents, f'exclude failed: {r.scores}'
    assert 'bug-analyzer' not in r.agents, f'exclude failed: {r.scores}'
test('修复Skill被validator/analyzer排除词拦截', t4)

# ── 5. 置信度 ──
def t5():
    from agent.skills.skill_router import SkillRouter, AGENT_KEYWORD_TIERS
    from agent.skills.progressive import SkillFrontmatter, SkillScope
    router = SkillRouter()
    s_high = SkillFrontmatter(name='unit-test-gen', description='JUnit单元测试生成覆盖率验证',
                              tags=['test','junit','coverage'], scope=SkillScope.USER)
    s_low = SkillFrontmatter(name='mystery', description='通用工具', tags=['tool'], scope=SkillScope.USER)
    r_high = router._scored_match(s_high, list(AGENT_KEYWORD_TIERS.keys()))
    r_low = router._scored_match(s_low, list(AGENT_KEYWORD_TIERS.keys()))
    assert r_high.confidence >= 0.8, f'high confidence: {r_high.confidence}'
    assert r_low.confidence < 0.6, f'low confidence: {r_low.confidence}'
test('置信度: 高信号>=0.8 低信号<0.6', t5)

# ── 6. 多信号兜底 ──
def t6():
    from agent.skills.skill_router import SkillRouter, AGENT_KEYWORD_TIERS
    from agent.skills.progressive import SkillFrontmatter, SkillScope
    router = SkillRouter()
    agents = list(AGENT_KEYWORD_TIERS.keys())
    tests = [
        ('deadlock-detector', '死锁检测与诊断', ['deadlock','diagnose'], 'bug-analyzer'),
        ('null-finder', '空指针漏洞发现', ['null','vulnerability'], 'code-reviewer'),
        ('fixer', '代码修复补丁生成', ['fix','patch'], 'code-fixer'),
        ('tester', '单元测试回归验证', ['test','regression'], 'test-validator'),
    ]
    for name, desc, tags, expected in tests:
        s = SkillFrontmatter(name=name, description=desc, tags=tags, scope=SkillScope.USER)
        actual = router._multi_signal_fallback(s, agents)
        assert actual == expected, f'{name}: expected {expected}, got {actual}'
test('多信号兜底4种Agent全部正确', t6)

# ── 7. MCP 工具 ──
def t7():
    from agent.skills.tools import get_tool_definitions, SkillToolsHandler
    tools = get_tool_definitions()
    names = [t['name'] for t in tools]
    required = ['list_skills','load_skill','search_skills','download_skill',
                'create_skill','update_skill','recommend_skill_action','assign_skill']
    for r in required:
        assert r in names, f'missing: {r}'
    handler = SkillToolsHandler()
    assert hasattr(handler, 'handle')
test('MCP工具 8个全部注册', t7)

# ── 8. SubAgent extra_skills ──
def t8():
    from agent.delegation.sub_agent import SubAgent
    from agent.delegation.agent_definition import AgentDefinition
    sig = inspect.signature(SubAgent.__init__)
    assert 'extra_skills' in sig.parameters
    sub = SubAgent(
        definition=AgentDefinition(name='test', description='test', system_prompt='You are test.'),
        task_prompt='test',
        extra_skills={'test-skill': '---\nname: test\n---\n# Skill\nStep 1: do X.\nStep 2: verify.'},
    )
    msg = sub._build_system_message()
    assert '动态加载的扩展技能' in msg
    assert 'test-skill' in msg
    assert 'Step 1: do X' in msg
    # frontmatter should be stripped
    assert 'name: test' not in msg
test('SubAgent extra_skills注入 + frontmatter剥离', t8)

# ── 9. create_skill 全流程 ──
def t9():
    from agent.skills.manager import SkillManager
    from agent.skills.progressive import SkillScope
    mgr = SkillManager()
    async def run():
        fm = await mgr.create_skill(
            name='test-skill-temp', description='Test skill',
            body='# Test\nThis is a test.', scope=SkillScope.USER, tags=['test','temp'])
        assert fm is not None and fm.name == 'test-skill-temp'
        skill_path = mgr.skills_root / 'user' / 'test-skill-temp' / 'SKILL.md'
        assert skill_path.exists(), f'not created: {skill_path}'
        content = mgr.load_full_skill('test-skill-temp')
        assert 'This is a test' in content
        shutil.rmtree(skill_path.parent)
        mgr._content_cache.pop('test-skill-temp', None)
    try:
        asyncio.get_event_loop().run_until_complete(asyncio.wait_for(run(), timeout=15))
    except asyncio.TimeoutError:
        raise RuntimeError('create_skill timed out (ChromaDB/Redis unavailable)')
test('create_skill 创建→落盘→加载→清理', t9)

# ── 10. update_skill 全流程 ──
def t10():
    from agent.skills.manager import SkillManager
    from agent.skills.progressive import SkillScope
    mgr = SkillManager()
    async def run():
        fm = await mgr.create_skill(
            name='update-test-temp', description='Old', body='# Old\nOld body.',
            scope=SkillScope.USER, tags=['old'])
        fm2 = await mgr.update_skill(
            name='update-test-temp', description='New desc',
            body='# New\nNew body content.', tags=['new','updated'])
        assert fm2 is not None
        assert fm2.description == 'New desc' and fm2.tags == ['new','updated']
        content = mgr.load_full_skill('update-test-temp')
        assert 'New body' in content and 'Old body' not in content
        skill_dir = mgr.skills_root / 'user' / 'update-test-temp'
        if skill_dir.exists(): shutil.rmtree(skill_dir)
        mgr._content_cache.pop('update-test-temp', None)
    try:
        asyncio.get_event_loop().run_until_complete(asyncio.wait_for(run(), timeout=15))
    except asyncio.TimeoutError:
        raise RuntimeError('update_skill timed out (ChromaDB/Redis unavailable)')
test('update_skill 更新→写回→重载', t10)

# ── 11. update 不存在的Skill ──
def t11():
    from agent.skills.manager import SkillManager
    mgr = SkillManager()
    async def run():
        fm = await mgr.update_skill(name='nonexistent-xyz-123', body='test')
        assert fm is None
    try:
        asyncio.get_event_loop().run_until_complete(asyncio.wait_for(run(), timeout=10))
    except asyncio.TimeoutError:
        raise RuntimeError('update_skill nonexistent timed out')
test('update_skill对不存在Skill返回None', t11)

# ── 12. AGENT_KEYWORD_TIERS 结构检查 ──
def t12():
    from agent.skills.skill_router import AGENT_KEYWORD_TIERS
    for agent in ['code-reviewer','bug-analyzer','code-fixer','test-validator']:
        tiers = AGENT_KEYWORD_TIERS[agent]
        assert len(tiers['tier1']) >= 3, f'{agent} tier1 too few: {len(tiers["tier1"])}'
        assert len(tiers['tier2']) >= 3, f'{agent} tier2 too few'
        assert len(tiers['exclude']) >= 1, f'{agent} must have exclude words'
test('AGENT_KEYWORD_TIERS 4 Agent关键词表结构完整', t12)

# ── 13. 5种Skill场景全覆盖 ──
def t13():
    from agent.skills.skill_router import SkillRouter, AGENT_KEYWORD_TIERS
    from agent.skills.progressive import SkillFrontmatter, SkillScope
    router = SkillRouter()
    agents = list(AGENT_KEYWORD_TIERS.keys())
    scenarios = [
        ('SQL注入检测', 'security-injection', '检测SQL注入和XSS漏洞', ['sql','injection','xss'], 'code-reviewer'),
        ('内存泄漏诊断', 'memory-leak-analyze', 'Java堆内存泄漏根因分析', ['memory','leak','java'], 'bug-analyzer'),
        ('并发Bug修复', 'concurrency-fix', '修复竞态条件和死锁的补丁', ['fix','concurrency','deadlock'], 'code-fixer'),
        ('回归测试套件', 'regression-suite', 'JUnit回归测试自动执行', ['test','junit','regression'], 'test-validator'),
        ('代码规范检查', 'style-lint', 'Java代码规范与格式检查', ['style','lint','java'], 'code-reviewer'),
    ]
    for name, sname, desc, tags, expected in scenarios:
        s = SkillFrontmatter(name=sname, description=desc, tags=tags, scope=SkillScope.USER)
        r = router._scored_match(s, agents)
        assert expected in r.agents, f'{name}: expected {expected}, got {r.agents} (scores={r.scores})'
        print(f'    {name}: {r.agents[0]} (score={r.scores[expected]}, method={r.method})')
test('5种Bug场景Skill→Agent匹配全部正确', t13)

print()
print(f'PHASE 1 DONE: {passed} passed, {failed} failed')
if errors:
    for name, err in errors:
        print(f'  FAILED: {name} -> {err}')
    sys.exit(1)
else:
    print('ALL PASSED')
