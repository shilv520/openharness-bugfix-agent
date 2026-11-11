"""
Task 1.6 — Java bug 注入器 v0（合成数据：SFT 主力 + clean 评测集）

原理：对"正确的"Java 源码做单点语义破坏 → 得到 buggy 版本；
      原码即标准答案，golden patch = buggy → 原码 的 unified diff。
      零成本、无限量、零预训练污染。

变异算子（v0）：
  关系运算符: <= ↔ >, >= ↔ <, == ↔ !=, < ↔ >=, > ↔ <=
  逻辑运算符: && ↔ ||
  算术运算符: + ↔ -, * ↔ /（仅带空格的二元形式，降低误伤）
  布尔字面量: true ↔ false
  整数 off-by-one: N → N±1

安全跳过：注释行 / import / package / 含字符串字面量的行（v0 粗过滤）。

⚠️ 质量闭环（必做，Task 1 校准清单）：
  注入产物必须过 reward 执行器筛选——只保留
    ① buggy 版本可编译
    ② 至少存在一个测试对 golden patch 敏感（打上 patch 前后测试结果变化）
  的注入样本。不做这步，合成数据里会混入"无行为差异"的死样本。

用法：
  python bug_injector.py <java_src_dir> --out injected.jsonl --n 200 [--seed 42]
"""
from __future__ import annotations

import argparse
import difflib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# (名称, 正则, 替换) —— 按序尝试，一行最多注入一处
OPERATORS: list[tuple[str, str, str]] = [
    ("le_to_gt",   r"<=",  ">"),
    ("ge_to_lt",   r">=",  "<"),
    ("eq_to_ne",   r"(?<!!)==(?!=)", "!="),
    ("ne_to_eq",   r"!=",  "=="),
    ("and_to_or",  r"&&",  "||"),
    ("or_to_and",  r"\|\|", "&&"),
    ("lt_to_ge",   r"<(?!=)", ">="),
    ("gt_to_le",   r"(?<![<>=!])>(?!=)", "<="),
    ("plus_to_minus", r" \+ ", " - "),
    ("minus_to_plus", r" - ", " + "),
    ("mul_to_div",    r" \* ", " / "),
    ("true_to_false", r"\btrue\b", "false"),
    ("false_to_true", r"\bfalse\b", "true"),
]
INT_RE = re.compile(r"\b(\d{1,4})\b")


@dataclass
class InjectedBug:
    rel_path: str
    mutation: str          # 算子名或 off_by_one
    line_no: int           # 1-based
    buggy_src: str
    golden_patch: str      # buggy → 原码 的 unified diff
    original_line: str
    buggy_line: str


def _is_skippable(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.startswith(("//", "*", "/*", "@Override")):
        return True
    if s.startswith(("import ", "package ")):
        return True
    if '"' in line:          # v0 粗过滤：含字符串字面量的行不动
        return True
    return False


def _mutate_line(line: str, rng: random.Random) -> Optional[tuple[str, str]]:
    """返回 (mutation_name, new_line) 或 None。"""
    candidates: list[tuple[str, str]] = []
    for name, pat, repl in OPERATORS:
        m = re.search(pat, line)
        if m:
            candidates.append((name, re.sub(pat, repl, line, count=1)))
    m = INT_RE.search(line)
    if m:
        n = int(m.group(1))
        delta = rng.choice([-1, 1])
        new_n = n + delta
        if new_n >= 0:
            candidates.append(("off_by_one", line[:m.start(1)] + str(new_n) + line[m.end(1):]))
    if not candidates:
        return None
    return rng.choice(candidates)


def inject_bug(src: str, rel_path: str = "Source.java",
               rng: Optional[random.Random] = None) -> Optional[InjectedBug]:
    """对源码注入单个 bug。返回 InjectedBug 或 None（无可注入点）。"""
    rng = rng or random.Random()
    lines = src.splitlines(keepends=True)

    # 收集所有可注入点，随机选一个（保证多样性，避免总改第一处）
    options: list[tuple[int, str, str]] = []  # (idx, mutation, new_line)
    for idx, line in enumerate(lines):
        if _is_skippable(line):
            continue
        got = _mutate_line(line, rng)
        if got:
            options.append((idx, got[0], got[1]))
    if not options:
        return None

    idx, mutation, new_line = rng.choice(options)
    buggy_lines = list(lines)
    buggy_lines[idx] = new_line
    buggy_src = "".join(buggy_lines)

    # golden patch：buggy(a/) → 原码(b/)，即"正确修复"
    diff = difflib.unified_diff(
        buggy_lines, lines,
        fromfile=f"a/{rel_path}", tofile=f"b/{rel_path}",
    )
    patch = "".join(diff)
    if not patch.strip():
        return None

    return InjectedBug(
        rel_path=rel_path, mutation=mutation, line_no=idx + 1,
        buggy_src=buggy_src, golden_patch=patch,
        original_line=lines[idx].rstrip("\n"), buggy_line=new_line.rstrip("\n"),
    )


def walk_java_files(root: Path) -> list[Path]:
    files = []
    for p in root.rglob("*.java"):
        posix = p.relative_to(root).as_posix()
        if "/test/" in posix or "/tests/" in posix:
            continue
        if p.stat().st_size > 200_000:      # 跳过异常大文件
            continue
        files.append(p)
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src_dir", help="Java 源码目录（如 project_repos 检出的 src/main/java）")
    ap.add_argument("--out", default="injected.jsonl")
    ap.add_argument("--n", type=int, default=100, help="目标注入条数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.src_dir)
    files = walk_java_files(root)
    if not files:
        print(f"[ERROR] {root} 下没有可用 java 文件")
        return
    print(f"扫描到 {len(files)} 个 java 文件，目标注入 {args.n} 条")

    rng.shuffle(files)
    written = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for fp in files:
            if written >= args.n:
                break
            try:
                src = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            bug = inject_bug(src, rel_path=fp.relative_to(root).as_posix(), rng=rng)
            if bug is None:
                continue
            f.write(json.dumps({
                "rel_path": bug.rel_path, "mutation": bug.mutation,
                "line_no": bug.line_no, "buggy_src": bug.buggy_src,
                "golden_patch": bug.golden_patch,
                "original_line": bug.original_line, "buggy_line": bug.buggy_line,
                "seed": args.seed,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"完成：{written} 条 → {args.out}")
    print("⚠️ 下一步（必做）：用 reward 执行器筛掉「不可编译 / 无行为差异」的死样本")


if __name__ == "__main__":
    main()
