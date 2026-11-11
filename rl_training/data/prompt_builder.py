"""
Prompt 构建器 — SFT / RL / 评测三处共用同一模板（严禁漂移！）

计划文档里的铁律：chat template 和任务模板在 SFT、GRPO、评测三个环节必须完全一致，
否则 RL 训出来的能力和评测测的能力对不上。本文件是唯一事实源。

patch 解析部分改编自 data/benchmark/test_defects4j.py::parse_patch_to_code，
增强点：保留 hunk 结构 / 支持多文件 / 区分 buggy 侧与 fixed 侧。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

try:
    from reward.reward import _clean_diff_text   # rl_training 根在 sys.path（脚本约定）
except ImportError:                              # 'reward' 已被当顶层模块导入（单测直跑）
    from reward import _clean_diff_text

# ---------- 统一模板（SFT/RL/评测共用，改这里 = 全链路一起改） ----------

SYSTEM_PROMPT = (
    "You are an expert Java programmer. You fix bugs with minimal, targeted patches.\n"
    "Output exactly ONE unified diff inside a ```diff code fence.\n"
    "Rules:\n"
    "- Only modify source files. NEVER modify test files.\n"
    "- Keep the change minimal; do not refactor unrelated code.\n"
    "- The patch must apply cleanly with `git apply` and compile."
)

USER_TEMPLATE = """## Bug Report
- Bug ID: {bug_id} ({jira})
- Project: {project}

## Failing test(s) that expose the bug
{trigger_tests}

## Classes known to contain the bug (fault localization)
{modified_classes}

## Buggy code
```java
{code}
```

## Task
Write a unified diff patch that fixes the bug so the failing tests pass,
without breaking any existing behavior."""


def fence_patch(patch: str) -> str:
    """把 golden patch 包成模型应输出的标准格式（assistant 目标）。

    ⚠️ 严禁整段 .strip()（对抗评审抓出）：golden patch 常以 ' ' 空上下文行结尾
    （校准实测 8/8 官方 diff 如此），strip 吃掉它 → @@ 头声明的行数与实际内容不符，
    等于在 SFT 标签里把「计数算错」制度化——reward 侧的重试阶梯能救判定，
    但标签本身必须干净。与判定侧共用同一个 _clean_diff_text：标签与判定永远同一个世界。"""
    return "```diff\n" + _clean_diff_text(patch) + "\n```"


def build_prompt(bug: dict, code: str) -> str:
    """bug = split.json 里的一条 record；code = buggy 源码（全文或定位片段）。"""
    triggers = "\n".join(f"- {t}" for t in bug.get("trigger_tests", [])) or "- (unknown)"
    classes = "\n".join(f"- {c}" for c in bug.get("modified_classes", [])) or "- (unknown)"
    return USER_TEMPLATE.format(
        bug_id=bug.get("bug_id", "?"),
        jira=bug.get("jira", "?"),
        project=bug.get("project", "?"),
        trigger_tests=triggers,
        modified_classes=classes,
        code=code.strip(),
    )


def build_sft_pair(bug: dict, code: str, golden_patch: str) -> dict:
    """产出一条 ms-swift SFT 训练样本（messages 格式）。"""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(bug, code)},
            {"role": "assistant", "content": fence_patch(golden_patch)},
        ]
    }


# ---------- patch 结构化解析 ----------

@dataclass
class Hunk:
    header: str = ""
    minus: list = field(default_factory=list)   # buggy 侧（被删的行）
    plus: list = field(default_factory=list)    # fixed 侧（新增的行）
    context: list = field(default_factory=list)


@dataclass
class FilePatch:
    path: str = ""
    hunks: list = field(default_factory=list)


_HUNK_RE = re.compile(r"^@@\s.*?@@")


def parse_src_patch(patch_text: str) -> list[FilePatch]:
    """解析 Defects4J 官方 src.patch（unified diff）→ 结构化。"""
    files: list[FilePatch] = []
    cur_file: FilePatch | None = None
    cur_hunk: Hunk | None = None

    for line in (patch_text or "").splitlines():
        if line.startswith("diff --git"):
            cur_file, cur_hunk = FilePatch(), None
            files.append(cur_file)
        elif line.startswith("--- "):
            if cur_file is None:
                cur_file, cur_hunk = FilePatch(), None
                files.append(cur_file)
            p = line[4:].strip()
            p = p[2:] if p.startswith("a/") else p
            if p != "/dev/null":
                cur_file.path = cur_file.path or p
        elif line.startswith("+++ "):
            p = line[4:].strip().split("\t")[0]
            p = p[2:] if p.startswith("b/") else p
            if p != "/dev/null" and cur_file is not None:
                cur_file.path = p
        elif _HUNK_RE.match(line):
            if cur_file is None:
                cur_file = FilePatch()
                files.append(cur_file)
            cur_hunk = Hunk(header=line)
            cur_file.hunks.append(cur_hunk)
        elif cur_hunk is not None:
            if line.startswith("-"):
                cur_hunk.minus.append(line[1:])
            elif line.startswith("+"):
                cur_hunk.plus.append(line[1:])
            elif line.startswith(" ") or line == "":
                cur_hunk.context.append(line[1:] if line else "")

    return [f for f in files if f.path]


def patch_to_bug_context(patch_text: str, max_lines: int = 60) -> str:
    """无完整 checkout 时的降级方案：从官方 patch 提取 buggy 侧代码片段。
    （展示 hunk 上下文 + buggy 行，即模型将看到的"案发现场"）"""
    out: list[str] = []
    for fp in parse_src_patch(patch_text):
        out.append(f"// File: {fp.path}")
        for h in fp.hunks:
            out.append(h.header)
            for c in h.context[:6]:
                out.append(" " + c)
            for m in h.minus:
                out.append("-" + m)
            if len(out) > max_lines:
                out.append("// ... (truncated)")
                return "\n".join(out)
    return "\n".join(out)
