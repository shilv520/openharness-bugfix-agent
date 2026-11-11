"""
bug 注入器单元测试。本地直接跑：
    python rl_training/injector/test_injector.py
"""
from __future__ import annotations

import random
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bug_injector import InjectedBug, _is_skippable, inject_bug, walk_java_files  # noqa: E402

SAMPLE = '''public class Range {
    // clamp value into [lo, hi]
    public static int clamp(int v, int lo, int hi) {
        if (v < lo) {
            return lo;
        }
        if (v > hi) {
            return hi;
        }
        return v;
    }

    public static boolean inRange(int v, int lo, int hi) {
        return v >= lo && v <= hi;
    }
}'''

NO_TARGET = '''public class A {
    // if (x < 3) return true;
    String s = "a < b && c > d";
    public void f() { }
}'''


def test_inject_produces_mutation():
    bug = inject_bug(SAMPLE, "Range.java", random.Random(42))
    assert isinstance(bug, InjectedBug)
    assert bug.buggy_src != SAMPLE
    assert "--- a/Range.java" in bug.golden_patch
    assert "+++ b/Range.java" in bug.golden_patch


def test_patch_restores_original():
    """golden patch 方向必须是 buggy(a) → 原码(b)：'+' 行含原码，'-' 行含变异码。"""
    bug = inject_bug(SAMPLE, "Range.java", random.Random(7))
    lines = bug.golden_patch.splitlines()
    plus = [l[1:].strip() for l in lines if l.startswith("+") and not l.startswith("+++")]
    minus = [l[1:].strip() for l in lines if l.startswith("-") and not l.startswith("---")]
    assert bug.original_line.strip() in plus
    assert bug.buggy_line.strip() in minus


def test_deterministic_with_seed():
    b1 = inject_bug(SAMPLE, "Range.java", random.Random(123))
    b2 = inject_bug(SAMPLE, "Range.java", random.Random(123))
    assert (b1.mutation, b1.line_no, b1.buggy_src) == (b2.mutation, b2.line_no, b2.buggy_src)


def test_no_target_returns_none():
    """注释/字符串里的'假算子'不能成为注入点。"""
    assert inject_bug(NO_TARGET, "A.java", random.Random(1)) is None


def test_skippable_lines():
    assert _is_skippable("")
    assert _is_skippable("   // if (a < b)")
    assert _is_skippable(" * javadoc line")
    assert _is_skippable("import java.util.List;")
    assert _is_skippable('String s = "a < b";')
    assert not _is_skippable("        if (a < b) {")


def test_walk_excludes_tests():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "src" / "main").mkdir(parents=True)
        (root / "src" / "test").mkdir(parents=True)
        (root / "src" / "main" / "Foo.java").write_text("class Foo {}")
        (root / "src" / "test" / "FooTest.java").write_text("class FooTest {}")
        got = walk_java_files(root)
        names = [p.name for p in got]
        assert "Foo.java" in names and "FooTest.java" not in names


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
