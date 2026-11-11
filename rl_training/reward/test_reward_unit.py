"""
Task 1.5 — reward 纯逻辑单元测试（三件套的"本地可跑"部分）

覆盖：diff 解析 / 补丁安全校验（防 reward hacking）/ 分级打分 / 测试输出解析 / hash 守卫
不需要 Defects4J 环境，Windows 本地直接跑：
    python rl_training/reward/test_reward_unit.py
（也兼容 pytest：python -m pytest rl_training/reward/test_reward_unit.py）

注意：三件套中"官方 patch → 1.0 / 乱 patch → 0"的端到端验证需要 Linux 环境，
属于 Task 1 校准清单；本文件先把所有纯逻辑分支钉死。
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))         # reward/（模块直导）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # rl_training 根（data.*）

from reward import (  # noqa: E402
    TestOutcome, apply_variants, check_untouched, extract_patch, git_apply_ladder,
    grade_reward, match_failed_triggers, normalize_java_eol,
    parse_failing_tests, touched_paths, validate_patch_safety, validate_paths_safety,
)
from data.prompt_builder import fence_patch  # noqa: E402  标签侧与判定侧必须同一个世界

PATCH_SRC = """--- a/src/main/java/org/example/Foo.java
+++ b/src/main/java/org/example/Foo.java
@@ -10,7 +10,7 @@ public class Foo {
-        if (x < 0) {
+        if (x <= 0) {
"""

PATCH_TESTFILE = """--- a/src/test/java/org/example/FooTest.java
+++ b/src/test/java/org/example/FooTest.java
@@ -20,7 +20,7 @@ public class FooTest {
-        assertEquals(1, result);
+        assertEquals(2, result);
"""


# ---------- extract_patch ----------

def test_extract_fenced():
    out = f"分析如下：\n```diff\n{PATCH_SRC}```\n以上。"
    assert extract_patch(out) == PATCH_SRC.strip()


def test_extract_raw():
    out = f"修复方案：\n{PATCH_SRC}\n希望有帮助"
    got = extract_patch(out)
    assert got is not None and "@@" in got and "希望有帮助" not in got


def test_extract_none():
    assert extract_patch("这段代码没问题") is None
    assert extract_patch("") is None
    assert extract_patch(None) is None


# 程序化构造：末行是"单个空格"的空上下文行（编辑器会静默 trim 行尾空格，不能写死在源码里）
PATCH_TBC = ("--- a/src/main/java/org/example/Bar.java\n"
             "+++ b/src/main/java/org/example/Bar.java\n"
             "@@ -5,4 +5,4 @@ public class Bar {\n"
             "     public void m() {\n"
             "-        old();\n"
             "+        new();\n"
             "     }\n"
             " \n")


def test_extract_preserves_trailing_blank_context():
    # 回归（2026-11-09 Linux 真机校准抓出）：hunk 末尾的空上下文行是"单个空格"，
    # 旧实现整段 .strip() 把它吃掉 → 行数与 @@ 头不符 → git apply "corrupt patch"，
    # 8/8 官方补丁全被误判 patch_apply_failed。修复后：只剥换行，不碰空格独占行。
    assert PATCH_TBC.splitlines()[-1] == " ", "测试夹具本身要保证末行是单空格"
    out = f"fix:\n```diff\n{PATCH_TBC}```"
    got = extract_patch(out)
    assert got is not None and got.splitlines()[-1] == " "
    # 裸 diff 路径同样不能吃掉它
    out2 = f"修复方案：\n{PATCH_TBC}\n以上"
    got2 = extract_patch(out2)
    assert got2 is not None and got2.splitlines()[-1] == " "


def test_extract_normalizes_crlf_and_bom():
    # 回归（对抗评审 S6/S7 实测）：围栏路径曾逐字节保留 \r → CRLF 模型输出的上下文
    # 永远匹配不上 LF 文件，全类假 0；裸路径 splitlines 却会归一 → 同内容两种判分。
    crlf = "```diff\r\n" + PATCH_SRC.replace("\n", "\r\n") + "```"
    got = extract_patch("\ufeff" + crlf)          # 顺手带 BOM
    assert got is not None and "\r" not in got and "\ufeff" not in got
    assert got == PATCH_SRC.strip()


# ---------- apply_variants（git apply 重试阶梯的纯函数部分） ----------

def test_apply_variants():
    p = "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n a\n-b\n+B"
    v = apply_variants(p)
    assert v[0] == p, "原文必须是第一档（末行 ' ' 空上下文是 hunk 内容，不许先剥）"
    assert v[-1] == p + "\n\n", "补空上下文档：落盘后末行必须是空行（双 \\n 收尾）"
    # 尾部空白垃圾行 → 产生剥离变体，且原文仍排第一
    pj = p + "\n \n   "
    vj = apply_variants(pj)
    assert vj[0] == pj
    assert vj[1] == p, "剥掉垃圾行后应回到干净原文"


# ---------- validate_paths_safety（git numstat 第二道闸的规则层） ----------

def test_validate_paths_safety():
    assert validate_paths_safety(["src/main/java/Foo.java"]) == (True, "ok")
    ok, r = validate_paths_safety(["src/test/resources/expected.txt"])
    assert not ok and r.startswith("test_file_touched"), "非 .java 测试资源也必须拦"
    ok2, _ = validate_paths_safety(["src/main/java/Foo.java"], allowed_globs=["src/main/*"])
    assert ok2
    ok3, r3 = validate_paths_safety(["build.xml"], allowed_globs=["src/main/*"])
    assert not ok3 and r3.startswith("path_outside_allowlist")
    assert validate_paths_safety([]) == (False, "no_target_files")


# ---------- git apply 重试阶梯（真实 git 端到端；无 git 环境跳过） ----------

def test_git_apply_ladder_real_git():
    """对抗评审三类实测假 0 的端到端回归（git 2.34/2.53 探针结论一致）。"""
    import shutil as _sh
    import subprocess
    import tempfile

    if not _sh.which("git"):
        print("    (skip: git 不可用)")
        return

    def run_fn(cmd, cwd=None):
        cp = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        return cp.returncode, (cp.stdout or "") + "\n" + (cp.stderr or "")

    def mkws(root, name, files):
        ws = root / name
        ws.mkdir()
        run_fn(["git", "init", "-q", str(ws)])
        run_fn(["git", "config", "core.autocrlf", "false"], cwd=ws)
        for fname, content in files.items():
            # write_bytes：git apply 按字节匹配，Windows 文本模式翻译会毁掉 LF fixture
            (ws / fname).write_bytes(content.encode("utf-8"))
        return ws

    def read(ws, fname):
        return (ws / fname).read_bytes().decode("utf-8")

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        # A: hunk 末空上下文行被写成 ''（被 strip("\n") 吃掉）→ 声明 3 行实际 2 行。
        #    固定行数 corrupt；--recount 因 trailing=0 触发 match_end 规则（hunk 无尾
        #    上下文必须匹配在文件末尾）中途拒绝；补空上下文档救援。
        ws = mkws(root, "A", {"f.txt": "alpha\nbeta\n\ngamma\n"})
        patch = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n alpha\n-beta\n+BETA"
        ok, err = git_apply_ladder(ws, root / "A.patch", patch, run_fn)
        assert ok, f"补空上下文档应救援: {err}"
        assert read(ws, "f.txt") == "alpha\nBETA\n\ngamma\n"

        # B: 真 diff 后跟空白垃圾行（计数正确）→ 固定行数档容忍（声明数满足后忽略）。
        #    --recount 会把它折叠成对不上的空上下文 → 假 0（评审 S2）。
        ws = mkws(root, "B", {"f.txt": "l1\nl2\nl3\n"})
        patch = "--- a/f.txt\n+++ b/f.txt\n@@ -1,3 +1,3 @@\n l1\n-l2\n+L2\n l3\n \n   "
        ok, err = git_apply_ladder(ws, root / "B.patch", patch, run_fn)
        assert ok, f"固定行数档应容忍尾部垃圾行: {err}"
        assert read(ws, "f.txt") == "l1\nL2\nl3\n"

        # C: 多文件 traditional patch（无 'diff --git' 行，LLM 手写常见）→ --recount
        #    把第二个文件的 ---/+++ 头吞进第一个 hunk；固定行数档正确分段（评审高危项）。
        ws = mkws(root, "C", {"f.txt": "a\nb\n", "g.txt": "c\nd\n"})
        patch = ("--- a/f.txt\n+++ b/f.txt\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n"
                 "--- a/g.txt\n+++ b/g.txt\n@@ -1,2 +1,2 @@\n c\n-d\n+D")
        ok, err = git_apply_ladder(ws, root / "C.patch", patch, run_fn)
        assert ok, f"多文件 traditional patch 必须能 apply: {err}"
        assert read(ws, "f.txt") == "a\nB\n"
        assert read(ws, "g.txt") == "c\nD\n"

        # D: LLM @@ 行数算错（原始校准胜利，防回归）→ --recount 档救援。
        ws = mkws(root, "D", {"f.txt": "x\ny\nz\n"})
        patch = "--- a/f.txt\n+++ b/f.txt\n@@ -1,99 +1,99 @@\n x\n-y\n+Y\n z"
        ok, err = git_apply_ladder(ws, root / "D.patch", patch, run_fn)
        assert ok, f"--recount 档应救行数算错: {err}"
        assert read(ws, "f.txt") == "x\nY\nz\n"


# ---------- fence_patch（SFT 标签侧，与判定侧共用 _clean_diff_text） ----------

def test_fence_patch_preserves_blank_context():
    # 评审抓出：fence_patch 曾用整段 .strip() —— golden 以 ' ' 空上下文行结尾时
    # （8/8 官方 diff 如此）标签行数与 @@ 头不符，把「计数错」制度化进 SFT 目标。
    out = fence_patch(PATCH_TBC)
    assert out.startswith("```diff\n") and out.endswith("\n```")
    body = out[len("```diff\n"):-len("\n```")]
    assert body.splitlines()[-1] == " ", "标签里的空上下文行必须活着"
    # 全链路闭环：标签喂回判定侧提取，得到同一个补丁（同一个世界）
    assert extract_patch(out) == PATCH_TBC.rstrip("\n")


# ---------- validate_patch_safety ----------

def test_safety_src_ok():
    ok, reason = validate_patch_safety(PATCH_SRC)
    assert ok and reason == "ok"


def test_safety_testfile_rejected():
    ok, reason = validate_patch_safety(PATCH_TESTFILE)
    assert not ok and reason.startswith("test_file_touched")


def test_safety_allowlist():
    ok, reason = validate_patch_safety(PATCH_SRC, allowed_globs=["src/main/*"])
    assert ok
    ok2, reason2 = validate_patch_safety(PATCH_SRC, allowed_globs=["src/other/*"])
    assert not ok2 and reason2.startswith("path_outside_allowlist")


def test_safety_empty():
    ok, reason = validate_patch_safety("")
    assert not ok and reason == "no_patch"


def test_touched_paths():
    assert touched_paths(PATCH_SRC) == ["src/main/java/org/example/Foo.java"]


# ---------- grade_reward ----------

def _grade(**kw):
    base = dict(parsed=True, applied=True, compiled=True, hacked=False,
                trigger=TestOutcome(3, 0), regression=TestOutcome(40, 0))
    base.update(kw)
    return grade_reward(**base)


def test_grade_full_pass():
    assert _grade() == (1.0, "full_pass")


def test_grade_hacked_wins():
    # hacking 优先级最高：即使测试全过也是 0
    r, stage = _grade(hacked=True)
    assert r == 0.0 and stage == "reward_hacking_test_file_modified"


def test_grade_format_error():
    r, stage = _grade(parsed=False, applied=False, compiled=False,
                      trigger=None, regression=None)
    assert r == -0.1 and stage == "format_error_no_valid_diff"


def test_grade_compile_fail():
    r, stage = _grade(compiled=False, trigger=None, regression=None)
    assert r == 0.0 and stage == "compile_failed"


def test_grade_regression_fail():
    r, stage = _grade(regression=TestOutcome(40, 2))
    assert r == 0.3 and stage == "trigger_pass_regression_fail"


def test_grade_partial_trigger():
    r, _ = _grade(trigger=TestOutcome(4, 2), regression=None)
    assert r == 0.3  # 0.1 + 0.4 * 0.5
    r2, _ = _grade(trigger=TestOutcome(4, 3), regression=None)
    assert r2 == 0.2  # 0.1 + 0.4 * 0.25
    r3, _ = _grade(trigger=TestOutcome(4, 1), regression=None)
    assert r3 == 0.4  # 0.1 + 0.4 * 0.75


def test_grade_all_fail():
    r, stage = _grade(trigger=TestOutcome(3, 3), regression=None)
    assert r == 0.0 and stage == "trigger_all_fail"


# ---------- parse_failing_tests（真实 d4j-test 输出格式） ----------

def test_parse_real_fail():
    out = ('Running ant (compile.tests)...................................................... OK\n'
           'Failing tests: 1\n'
           '  - org.apache.commons.lang3.math.NumberUtilsTest::TestLang747\n')
    assert parse_failing_tests(out) == [
        "org.apache.commons.lang3.math.NumberUtilsTest::TestLang747"]


def test_parse_real_pass():
    assert parse_failing_tests("Running ant (run.dev.tests)... OK\nFailing tests: 0\n") == []


def test_parse_class_level_failure():
    out = "Failing tests: 2\n  - org.example.FooTest\n  - org.example.BarTest::testB\n"
    assert parse_failing_tests(out) == ["org.example.FooTest", "org.example.BarTest::testB"]


def test_parse_garbage_none():
    # 解析不到 'Failing tests:' 行 → None（调用方保守按失败处理，绝不能当全绿）
    assert parse_failing_tests("connection refused") is None


# ---------- hash guard ----------

def test_check_untouched():
    before = {"a/FooTest.java": "h1", "b/BarTest.java": "h2"}
    assert check_untouched(before, dict(before))
    assert not check_untouched(before, {**before, "a/FooTest.java": "CHANGED"})
    assert not check_untouched(before, {**before, "c/NewTest.java": "h3"})
    assert not check_untouched(before, {"a/FooTest.java": "h1"})


# ---------- 快路径：trigger 匹配（批量 -r 输出 → trigger 状态） ----------

def test_match_failed_triggers():
    trig = ["a.b.C::t1", "a.b.D::t2"]
    # 方法级精确匹配
    assert match_failed_triggers(trig, ["a.b.C::t1"]) == ["a.b.C::t1"]
    assert match_failed_triggers(trig, []) == []
    # 类级失败（测试类初始化崩溃，输出只有类名）→ 按类前缀命中该类全部 trigger
    assert match_failed_triggers(trig, ["a.b.D"]) == ["a.b.D::t2"]
    # 无关失败不算 trigger 挂（但会进回归失败 → 0.3 档）
    assert match_failed_triggers(trig, ["x.y.Z::other"]) == []
    # 空 trigger 防御
    assert match_failed_triggers([], ["a.b.C::t1"]) == []


# ---------- EOL 归一（CRLF 混杂仓库 vs 模型纯 LF 补丁） ----------

def test_normalize_java_eol():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "sub").mkdir()
        crlf_java = root / "sub" / "A.java"
        crlf_java.write_bytes(b"class A {\r\n  void m() {}\r\n}\r\n")
        crlf_txt = root / "data.txt"
        crlf_txt.write_bytes(b"keep\r\ncrlf\r\n")
        lf_java = root / "B.java"
        lf_java.write_bytes(b"class B {}\n")
        n = normalize_java_eol(root)
        assert n == 1, f"只应转换 1 个 .java，实际 {n}"
        assert crlf_java.read_bytes() == b"class A {\n  void m() {}\n}\n"
        assert crlf_txt.read_bytes() == b"keep\r\ncrlf\r\n", "非 .java 文件不许动"
        assert lf_java.read_bytes() == b"class B {}\n", "已是 LF 的不重写"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed}/{passed + failed} passed")
    sys.exit(1 if failed else 0)
