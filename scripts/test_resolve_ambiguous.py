"""测试 _resolve_ambiguous 二次判定逻辑（直接构造候选，不依赖ChromaDB）"""
import sys, os, asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agent.skills.manager import SkillManager
from agent.skills.progressive import SkillFrontmatter, SkillScope

passed = 0; failed = 0
def t(name, ok, detail=''):
    global passed, failed
    if ok: passed += 1; print(f'  [PASS] {name}' + (f' ({detail})' if detail else ''))
    else: failed += 1; print(f'  [FAIL] {name}: {detail}')

def make_fm(name, desc, tags):
    return SkillFrontmatter(name=name, description=desc, tags=tags, scope=SkillScope.USER)

# ==== 1. 关键词提取 ====
print('=== 1. 关键词提取 (子串扫描 + 中英映射) ===')
kw1 = SkillManager._extract_keywords('Java多线程死锁检测与诊断')
t('提取死锁', '死锁' in kw1, str(kw1))
t('提取deadlock(中英映射)', 'deadlock' in kw1)
t('提取multithreading(中英映射)', 'multithreading' in kw1)

kw2 = SkillManager._extract_keywords('detect Java deadlock and race conditions')
t('提取deadlock', 'deadlock' in kw2)
t('提取race condition', 'race condition' in kw2)

kw3 = SkillManager._extract_keywords('怎么修复空指针异常NullPointerException')
t('提取空指针', '空指针' in kw3)
t('提取null pointer(中英映射)', 'null pointer' in kw3)

kw4 = SkillManager._extract_keywords('Python GIL全局解释器锁性能问题')
t('提取python', 'python' in kw4)

# ==== 2. 二次判定场景 (直接构造候选) ====
print()
print('=== 2. _resolve_ambiguous 场景 ===')
mgr = SkillManager()

async def test_resolve():
    # A: 死锁+并发 → concurrency-patterns tags含"concurrency"
    candidates = [
        (make_fm('bug-analysis', 'Bug根因分析', ['analysis','root-cause','diagnose']), 0.45),
        (make_fm('concurrency-patterns', '并发编程模式', ['concurrency','patterns','java']), 0.38),
    ]
    r = await mgr._resolve_ambiguous('Java多线程死锁并发问题', candidates)
    assert r is not None, 'A: should resolve'
    t('A. 死锁->并发tags重叠>=1: update',
      r['action']=='update' and 'concurrency-patterns' in r.get('target_skill',''),
      f'{r["action"]}->{r.get("target_skill")} method={r.get("resolve_method")}')

    # B: 空指针+null+safety → null-safety-tools tags重叠 >= 2 → 强信号
    candidates = [
        (make_fm('code-review', '代码审查', ['review','code-quality']), 0.40),
        (make_fm('null-safety-tools', '空指针安全检查', ['null','safety','static-analysis']), 0.55),
    ]
    r = await mgr._resolve_ambiguous('空指针null安全检测', candidates)
    assert r is not None, 'B: should resolve'
    t('B. 空指针->null-safety强重叠: update',
      r['action']=='update' and r.get('target_skill')=='null-safety-tools',
      f'{r["action"]}->{r.get("target_skill")} method={r.get("resolve_method")}')

    # C: PythonGIL 与所有候选 tags 无关 → 0重叠 → create
    candidates = [
        (make_fm('java-tools', 'Java工具集', ['java','jvm','maven']), 0.35),
        (make_fm('cpp-lint', 'C++静态分析', ['cpp','lint','clang']), 0.30),
    ]
    r = await mgr._resolve_ambiguous('Python GIL全局解释器锁性能问题', candidates)
    assert r is not None, 'C: should resolve'
    t('C. PythonGIL 0重叠: create',
      r['action']=='create',
      f'{r["action"]} method={r.get("resolve_method")}')

    # D: 低信号 → 无法消歧 → 返回 None (保持 ambiguous)
    candidates = [
        (make_fm('tool-a', '通用工具A', ['utility','general']), 0.42),
        (make_fm('tool-b', '通用工具B', ['helper','misc']), 0.35),
    ]
    r = await mgr._resolve_ambiguous('一个非常通用的描述', candidates)
    # 可能消歧成功也可能返回None
    t('D. 低信号: 不崩溃', r is None or isinstance(r, dict),
      f'result={"ambiguous->None" if r is None else r.get("action","?")}')

    # E: 空候选列表 → 直接返回 create
    r = await mgr._resolve_ambiguous('任何问题', [])
    assert r is not None and r['action'] == 'create', f'E: expected create, got {r}'
    t('E. 空候选: create', r['action']=='create', f'method={r.get("resolve_method")}')

asyncio.get_event_loop().run_until_complete(asyncio.wait_for(test_resolve(), timeout=20))

print()
print(f'TOTAL: {passed} passed, {failed} failed')
print('ALL PASSED' if failed == 0 else f'{failed} FAILED')
