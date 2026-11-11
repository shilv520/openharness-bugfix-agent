"""
Task 1.4 — Reward 执行器（骨架 v0）

分级奖励表（与计划文档一致）：
  ┌────────────────────────────────────────────┬────────┐
  │ 情况                                        │ reward │
  ├────────────────────────────────────────────┼────────┤
  │ 改动测试文件（reward hacking）               │ 0 强制 │
  │ 解析不出合法 diff（纯格式错）                │ -0.1   │
  │ patch apply 失败 / 编译失败                  │ 0      │
  │ trigger 全过 + 回归全过                      │ 1.0    │
  │ trigger 全过、回归挂                         │ 0.3    │
  │ trigger 部分过                               │ 0.1 + 0.4 × 通过比例 │
  │ trigger 全挂                                 │ 0      │
  └────────────────────────────────────────────┴────────┘

代码结构（刻意分成两半）：
  ▸ 纯函数区：解析 / 校验 / 打分 —— Windows 本地即可单测（test_reward_unit.py）
  ▸ 环境区：checkout / compile / test —— 需要 Linux + Defects4J 已初始化（project_repos 下载完）

TODO(Task1 校准清单，到 Linux 环境后做)：
  [x] 测试输出解析已按 d4j-test 源码校准：'Failing tests: N' + '  - name' 列表
      （parse_failing_tests）；到实例后用真实输出跑一遍复核确认
  [ ] 回归测试范围：v0 跑全量，后续改用 relevant_tests 元数据收窄
  [ ] prepare_workspace 的 copytree 换 git worktree / overlayfs 提速
  [ ] 实测单条 reward 延迟，确认 < 30s（Gate 0）
"""
from __future__ import annotations

import fnmatch
import hashlib
import os
import re
import shutil
import signal
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

# ══════════════════════════════════════════════════════════════
# 纯函数区（可单测，不碰环境）
# ══════════════════════════════════════════════════════════════

FENCE_RE = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL)
_DIFF_LINE_RE = re.compile(r"^(---|\+\+\+|@@|[+\\ ]|-)")
_PATH_RE = re.compile(r"^(?:---|\+\+\+)\s+(?:a/|b/)?(\S+)")
_FAILING_HDR_RE = re.compile(r"^Failing tests:\s*(\d+)\s*$", re.MULTILINE)


def _looks_like_diff(text: str) -> bool:
    lines = text.splitlines()
    has_header = any(l.startswith("--- ") for l in lines) and any(l.startswith("+++ ") for l in lines)
    has_hunk = any(l.startswith("@@") for l in lines)
    return has_header or has_hunk


def _find_raw_diff(text: str) -> Optional[str]:
    """从自由文本中抠出裸 unified diff（无代码围栏的情况）。"""
    lines = text.splitlines()
    for i, l in enumerate(lines):
        if l.startswith("--- ") and any(
            lines[j].startswith("+++ ") for j in range(i + 1, min(i + 4, len(lines)))
        ):
            out = []
            for l2 in lines[i:]:
                if l2.startswith("```"):
                    break
                if l2.strip() == "":
                    out.append(l2)
                    continue
                if not _DIFF_LINE_RE.match(l2):
                    break
                out.append(l2)
            cand = _clean_diff_text("\n".join(out))
            if _looks_like_diff(cand):
                return cand
    return None


def _clean_diff_text(s: str) -> str:
    """清理 diff 边缘：只去首尾换行与开头的纯空白行，但绝不碰"空格独占"的行（' '）——
    那是 hunk 里的空上下文行（Java 方法结尾常见）。整段 .strip() 会把它吃掉，
    导致行数与 @@ 头不符 → git apply 报 corrupt patch。
    （2026-11-09 Linux 真机校准实测踩中：8/8 官方补丁全部误判 patch_apply_failed。）"""
    lines = s.strip("\n").split("\n")
    while lines and lines[0].strip() == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def apply_variants(patch: str) -> list:
    """git apply 重试阶梯的候选文本（纯函数，单测覆盖）。按序：
      1) 原文            —— 末行 ' ' 空上下文必须保留（它是 hunk 内容的一部分）
      2) 剥尾部空白行    —— 模型在 diff 后面吐的 ' '/'   ' 垃圾行，--recount 会把它
                            折叠成空上下文 → 上下文对不上 → 假 0（对抗评审 S2-S5 实测）
      3) 补一个空上下文行 —— hunk 末空上下文被写成 ''（而非 ' '）时被 strip("\\n") 吃掉，
                            行数与 @@ 头不符；git 接受空行作空上下文（apply.c case '\\n'），
                            补回即恢复（对抗评审 Z1 实测）
    每个变体再按「固定行数 → --recount」两档试（见 git_apply_ladder）。
    只在失败后逐级升级 → 本来能 apply 成功的补丁永远不会被阶梯改变结果。"""
    def _strip_tail_ws(text: str) -> str:
        lines = text.split("\n")
        while lines and lines[-1].strip() == "":
            lines.pop()
        return "\n".join(lines)

    stripped = _strip_tail_ws(patch)
    variants = [patch]
    if stripped and stripped != patch:
        variants.append(stripped)
    # 补空上下文档：文本必须以 "\n\n" 收尾——落盘后最后一行才是"空行"，git 才把
    # 它当空上下文行（apply.c case '\n'）。只补一个 "\n" 会被写入端的"保证末尾
    # 换行"折叠掉，等于没补（2026-11-09 单测集成用例 A 实测抓出）。
    variants.extend(v.rstrip("\n") + "\n\n" for v in list(variants))
    return variants


def extract_patch(model_output: Optional[str]) -> Optional[str]:
    """从模型输出中提取 unified diff：优先 ```diff 围栏，其次裸 diff。取不到返回 None。"""
    if not model_output:
        return None
    # 入口统一归一 CRLF/BOM（对抗评审实测）：围栏路径逐字节保留 \r（裸路径 splitlines
    # 会归一）→ CRLF 输出的补丁上下文带 \r，永远匹配不上 LF 文件，同一内容围栏/裸体
    # 判分还不一致。workspace 已经 normalize_java_eol 归一到 LF，补丁必须同一个世界。
    # BOM（U+FEFF）会让 git 直接报 corrupt。
    model_output = model_output.lstrip("﻿").replace("\r\n", "\n").replace("\r", "\n")
    m = FENCE_RE.search(model_output)
    if m:
        cand = _clean_diff_text(m.group(1))
        if _looks_like_diff(cand):
            return cand
    return _find_raw_diff(model_output)


def touched_paths(patch: str) -> list[str]:
    """解析 patch 触及的所有文件路径（去重排序，跳过 /dev/null）。"""
    paths = set()
    for line in (patch or "").splitlines():
        m = _PATH_RE.match(line)
        if m:
            p = m.group(1).strip()
            if p and p != "/dev/null":
                paths.add(p.replace("\\", "/"))
    return sorted(paths)


def is_test_path(path: str) -> bool:
    p = (path or "").replace("\\", "/")
    if "/test/" in p or "/tests/" in p:
        return True
    return p.endswith(("Test.java", "Tests.java", "TestCase.java"))


def validate_paths_safety(paths: list[str],
                          allowed_globs: Optional[list[str]] = None) -> tuple[bool, str]:
    """路径清单安全检查。返回 (ok, reason)。reason 前缀含义：
       test_file_touched → reward hacking（强制 0 分）
       其余 → 按 apply 失败处理（0 分）

    两个入口共用本规则：
      validate_patch_safety（文本正则解析 ---/+++ 头）
      evaluate 内 apply 前的第二道闸（git apply --numstat -z，git 自己解析的路径）
    为什么需要第二道闸（对抗评审实测）：文本正则会被 C 风格八进制引号路径
    （'"a/src/\\164est/..."') 与含空格路径绕过（正则只截到第一个空格），而 git 的
    解析器不会。*.java 测试文件还有 hash 守卫兜底，但 src/test/resources/ 下的
    非 .java 文件对两层文本守卫都不可见——numstat 闸补的就是这个洞。
    """
    if not paths:
        return False, "no_target_files"
    for p in paths:
        if is_test_path(p):
            return False, f"test_file_touched:{p}"
    if allowed_globs:
        for p in paths:
            if not any(fnmatch.fnmatch(p, g) for g in allowed_globs):
                return False, f"path_outside_allowlist:{p}"
    return True, "ok"


def validate_patch_safety(patch: Optional[str],
                          allowed_globs: Optional[list[str]] = None) -> tuple[bool, str]:
    """补丁安全检查（文本层，第一道闸）。"""
    if not patch:
        return False, "no_patch"
    return validate_paths_safety(touched_paths(patch), allowed_globs)


@dataclass
class TestOutcome:
    n_total: int = 0
    n_failed: int = 0
    failed_names: list = field(default_factory=list)

    @property
    def all_pass(self) -> bool:
        return self.n_total > 0 and self.n_failed == 0

    @property
    def pass_fraction(self) -> float:
        return (self.n_total - self.n_failed) / self.n_total if self.n_total else 0.0


def parse_failing_tests(stdout: str) -> Optional[list]:
    """解析 `defects4j test` 的真实输出（对照 framework/bin/d4j/d4j-test 源码）：

         Running ant (run.dev.tests)................ OK
         Failing tests: 1
           - org.apache.commons.lang3.math.NumberUtilsTest::TestLang747

       类级失败（如测试类初始化异常）只有类名、没有 ::method。
       返回失败测试名列表（空列表 = 全绿）；解析不到 'Failing tests:' 行
       （超时 / ant 崩溃 / 环境错误）返回 None，调用方必须保守按失败处理。
       注意：输出不含测试总数，"应跑多少个" 由调用方（trigger 清单）提供。
    """
    text = stdout or ""
    m = _FAILING_HDR_RE.search(text)
    if not m:
        return None
    names = []
    for line in text[m.end():].splitlines():
        s = line.strip()
        if s.startswith("- "):
            names.append(s[2:].strip())
        elif s:
            break                      # 失败列表已结束，后续输出忽略
    return names[: int(m.group(1))]    # 以头部计数为准


def grade_reward(*, parsed: bool, applied: bool, compiled: bool, hacked: bool,
                 trigger: Optional[TestOutcome],
                 regression: Optional[TestOutcome]) -> tuple[float, str]:
    """分级打分（纯函数，单测覆盖全部分支）。返回 (reward, stage)。"""
    if hacked:
        return 0.0, "reward_hacking_test_file_modified"
    if not parsed:
        return -0.1, "format_error_no_valid_diff"
    if not applied:
        return 0.0, "patch_apply_failed"
    if not compiled:
        return 0.0, "compile_failed"
    if trigger is None:
        return 0.0, "trigger_tests_not_run"
    if trigger.all_pass:
        if regression is None or regression.all_pass:
            return 1.0, "full_pass"
        return 0.3, "trigger_pass_regression_fail"
    if trigger.n_total > 0 and trigger.n_failed < trigger.n_total:
        return round(0.1 + 0.4 * trigger.pass_fraction, 3), "partial_trigger_pass"
    return 0.0, "trigger_all_fail"


def check_untouched(before: dict[str, str], after: dict[str, str]) -> bool:
    """测试文件 hash 前后一致（增/删/改任一发生 → False）。"""
    return before == after


def match_failed_triggers(trigger_tests: list, failing_names: list) -> list:
    """从 defects4j test 的失败名单里挑出属于 trigger 的项（纯函数，单测覆盖）。

    失败条目两种形态：
      - 方法级 `class::method` → 与 trigger 精确匹配
      - 类级 `class`（测试类初始化崩溃时没有方法名）→ 按 trigger 的类前缀命中
    """
    fset = set(failing_names or [])
    failed = []
    for t in trigger_tests or []:
        if t in fset or t.split("::")[0] in fset:
            failed.append(t)
    return failed


# ══════════════════════════════════════════════════════════════
# 环境区（需要 Linux + Defects4J 初始化完成）
# ══════════════════════════════════════════════════════════════

@dataclass
class BugMeta:
    bug_id: str                       # 如 "Lang-1"
    project: str                      # 如 "Lang"
    vid: int                          # 如 1
    trigger_tests: list = field(default_factory=list)   # ["class::method", ...]
    allowed_globs: Optional[list] = None                # 如 ["src/main/*"]；None = 只禁测试文件

    @classmethod
    def from_split_record(cls, rec: dict) -> "BugMeta":
        return cls(
            bug_id=rec["bug_id"], project=rec["project"], vid=int(rec["vid"]),
            trigger_tests=list(rec.get("trigger_tests", [])),
            allowed_globs=rec.get("allowed_globs"),
        )


@dataclass
class RewardResult:
    bug_id: str
    reward: float
    stage: str
    detail: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def git_apply_ladder(ws, patch_file, patch: str, run_fn) -> tuple[bool, str]:
    """git apply 重试阶梯（候选文本见 apply_variants，纯函数部分单测覆盖）。

    档位顺序 = 保守解释优先，每档只在失败后升级（git apply 原子：失败不落盘，
    连续尝试安全；成功档的编辑位置都有上下文行逐行匹配背书，不存在假成功）：
      固定行数先试 —— 容忍尾部空白垃圾行（声明数满足后忽略多余行）；多文件
        traditional patch（无 'diff --git' 行，LLM 手写常见形态）正确分段。
        ⚠️ --recount 对这两类是破坏性的：它的 fragment 解析器贪婪吞掉
        第二个文件的 ---/+++ 头（对抗评审实测 rc=1，固定行数 rc=0）。
      --recount 兜底 —— 救 LLM 最高频的格式错误：@@ 行数算错。
    （2026-11-09 三镜头对抗评审驱动；git 2.34/2.53 两版本探针结论一致。）"""
    pfile = Path(patch_file)
    last_out = ""
    for variant in apply_variants(patch):
        for extra in ([], ["--recount"]):
            text = variant if variant.endswith("\n") else variant + "\n"
            # newline="\n"：文本模式默认把 \n 翻译成 os.linesep（Windows → CRLF），
            # git apply 按字节匹配上下文，补丁文件必须纯 LF（Linux 生产是 no-op）
            with open(pfile, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            rc, out = run_fn(["git", "apply", *extra, "--whitespace=nowarn",
                              str(pfile)], cwd=ws)
            if rc == 0:
                return True, ""
            last_out = out
    return False, last_out[-300:]


def normalize_java_eol(workspace: Path) -> int:
    """把 workspace 内所有 .java 文件 CRLF→LF 归一，返回转换文件数。

    为什么必须做（2026-11-09 Chart-10 实测）：Defects4J 老仓库 EOL 混杂——
    Chart-10 目标文件 69/69 行 CRLF，同仓库 Chart-11 的文件却是纯 LF。
    而模型看到的代码经文本模式读取（通用换行翻译已剥掉 \\r），产出的补丁必然纯 LF；
    git apply 按字节匹配上下文、对 CR 零容忍（--ignore-whitespace 与
    core.whitespace=cr-at-eol 实测都救不了）→ CRLF 文件的 bug 会被系统性误判
    patch_apply_failed（reward 恒 0，训练信号全毁）。
    归一到 LF 后，「模型看代码 → 产补丁 → apply」全链路同一个世界。
    javac 对 EOL 不敏感（CR 只是空白字符），编译与测试语义不变；
    非 .java 文件（测试资源、数据文件）一律不动。
    """
    n = 0
    ws = Path(workspace)
    for p in ws.rglob("*.java"):
        try:
            b = p.read_bytes()
        except OSError:
            continue
        if b"\r\n" in b:
            try:
                p.write_bytes(b.replace(b"\r\n", b"\n"))
                n += 1
            except OSError:
                pass
    return n


def snapshot_test_hashes(workspace: Path, max_files: int = 5000) -> dict[str, str]:
    """对 workspace 内所有测试 java 文件做 sha256 快照。"""
    hashes: dict[str, str] = {}
    ws = Path(workspace)
    for p in ws.rglob("*.java"):
        rel = p.relative_to(ws).as_posix()
        if is_test_path(rel):
            try:
                hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
            except OSError:
                pass
        if len(hashes) >= max_files:
            break
    return hashes


class RewardExecutor:
    """单条 (bug, model_output) → RewardResult 的完整判定流水线。

    defects4j CLI 约定（真机校准 2026-11-09 AutoDL + 对照 framework/bin/d4j/* 源码）：
      checkout: defects4j checkout -p <Project> -v <vid>{b|f} -w <workdir>
      compile:  defects4j compile -w <workdir>
      test:     defects4j test -w <workdir>                    # 全量开发者测试（一次 ant）
                defects4j test -w <workdir> -t <class::method> # 单个测试
                defects4j test -w <workdir> -r                 # 只跑 relevant tests（D4J 原生
                                                               #   裁剪：compile_tests + run_relevant_tests）
      export:   defects4j export -w <workdir> -p tests.relevant -o <file>
                                                               # relevant 测试类清单（文件干净、纯 LF）
      ⚠️ -t 只接一个测试名：逗号列表实测报错 "Cannot run tests!"（d4j-test L135），
         也不能传 trigger 文件（Wrong format for single test!）
      ⚠️ 输出只有 'Failing tests: N' + 列表，没有 JUnit 汇总行（见 parse_failing_tests）

    测试阶段策略（Gate 0 延迟驱动）：
      快路径：prepare 阶段 export tests.relevant 并校验「trigger 测试类 ⊆ relevant 测试类」，
             通过才写 .d4j_fastpath_ok → evaluate 只跑一次 `test -r`（编译+trigger+回归
             一次 ant 调用完成），trigger 状态从失败名单里读出（match_failed_triggers）。
      慢路径兜底：标记缺失（旧缓存 / export 失败 / 校验不过）→ 逐条跑 trigger + `-r` 回归
             （无 relevant 清单时退化为全量 `test`）。
      裁剪依据：relevant = 触碰被修改源码的测试类（D4J 定义）。实测 Chart-10 全量 328 个
      测试类 vs relevant 1 个。代价：改坏"无关类"不被回归抓——训练期接受此权衡，
      正式评测（Task 1.7）仍用全量。
    """

    def __init__(self, defects4j_bin: str | Path, cache_dir: str | Path,
                 run_dir: str | Path, timeout_s: int = 300):
        self.bin = Path(defects4j_bin)
        if not self.bin.exists():
            raise EnvironmentError(
                f"defects4j CLI 不存在: {self.bin}\n"
                "请先在 Linux/Docker 完成环境初始化（见 rl_training/README.md「关键前置」）"
            )
        self.cache_dir = Path(cache_dir)
        self.run_dir = Path(run_dir)
        self.timeout_s = timeout_s
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ---- 底层子进程 ----
    def _run(self, cmd: list[str], cwd=None) -> tuple[int, str]:
        # 坑 #39：subprocess.run 超时只杀直接子进程（defects4j 的 sh/perl 壳），
        #   它拉起的 java 孙进程（run.dev.tests / ant）变孤儿继续满核烧 CPU——实例实测
        #   堆积 ~14 个孤儿、load 31–41、CPU 100%。根治：start_new_session 让子进程
        #   自成进程组（pgid==pid），超时 killpg 杀全组，一个都不留。
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
            start_new_session=(os.name == "posix"))
        try:
            out, err = proc.communicate(timeout=self.timeout_s)
            return proc.returncode, (out or "") + "\n" + (err or "")
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            else:  # Windows 本地跑单测：无进程组可杀，退化为杀子进程
                proc.kill()
            try:
                proc.communicate(timeout=30)  # 收尸，防僵尸 + 释放管道
            except subprocess.TimeoutExpired:
                pass
            return -99, "timeout"

    REL_FILE = ".d4j_relevant_tests"        # export tests.relevant 的类清单
    FASTPATH_FILE = ".d4j_fastpath_ok"      # trigger ⊆ relevant 校验通过标记
    VID_SUFFIX = "b"                        # b=buggy（reward 判定世界）；子类可置 "f"
                                            # 构建 fixed 缓存（Task 2.1 合成注入用全绿世界）

    def prepare_workspace(self, bug: BugMeta) -> Path:
        """从预构建缓存复制一份独立 workspace；缓存缺失则现场
        checkout → EOL 归一 → compile → export relevant tests → 回填缓存。"""
        cache_ws = self.cache_dir / bug.bug_id
        ws = self.run_dir / f"{bug.bug_id}_{int(time.time() * 1000)}"
        if cache_ws.exists():
            shutil.copytree(cache_ws, ws)
            return ws
        rc, out = self._run([str(self.bin), "checkout", "-p", bug.project,
                             "-v", f"{bug.vid}{self.VID_SUFFIX}", "-w", str(ws)])
        if rc != 0:
            raise RuntimeError(f"checkout failed for {bug.bug_id}: {out[-500:]}")
        normalize_java_eol(ws)  # CRLF→LF：混杂 EOL 老仓库 vs 模型纯 LF 补丁（坑 #11）
        rc, out = self._run([str(self.bin), "compile", "-w", str(ws)])
        if rc == 0:
            self._export_relevant(bug, ws)  # 失败不阻塞：evaluate 自动走慢路径
            shutil.copytree(ws, cache_ws, dirs_exist_ok=True)  # 回填缓存（只存编译成功的）
        return ws

    def _export_relevant(self, bug: BugMeta, ws: Path) -> None:
        """导出 relevant tests 清单并校验 trigger ⊆ relevant（快路径前提）。"""
        try:
            rc, _ = self._run([str(self.bin), "export", "-w", str(ws),
                               "-p", "tests.relevant", "-o", str(ws / self.REL_FILE)])
            rel = self._relevant_tests(ws) if rc == 0 else []
            trig_classes = {t.split("::")[0] for t in bug.trigger_tests if t}
            if rel and trig_classes and trig_classes <= set(rel):
                (ws / self.FASTPATH_FILE).write_text("OK", encoding="utf-8")
        except OSError:
            pass

    def _relevant_tests(self, ws: Path) -> list:
        f = Path(ws) / self.REL_FILE
        if not f.exists():
            return []
        out = []
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            s = line.strip()
            if s and " " not in s and "." in s:  # 防 ant 进度行混入
                out.append(s)
        return out

    def _fastpath_ok(self, ws: Path) -> bool:
        f = Path(ws) / self.FASTPATH_FILE
        try:
            return f.exists() and f.read_text(encoding="utf-8").strip() == "OK"
        except OSError:
            return False

    def _numstat_paths(self, ws: Path, patch_file: Path) -> Optional[list]:
        """git 自己解析补丁 → 触及路径清单（第二道闸，见 validate_paths_safety）。

        --numstat -z 不碰工作区；NUL 分隔防路径含空格切错。解析失败/对不上返回
        None（补丁本身也 apply 不上，无需重复判罚）。局限：rename 补丁的 -z 记录
        形态不同，对不齐时跳过——模型补丁极罕见 rename，且 *.java 有 hash 守卫兜底。
        """
        rc, out = self._run(["git", "apply", "--numstat", "-z", str(patch_file)], cwd=ws)
        if rc != 0:
            return None
        tokens = out.split("\0")   # 形如 "<add>\t<del>\0<path>\0..."；_run 合并的
        paths = []                 # stdout+"\n"+stderr 尾巴由 fullmatch 守卫滤掉
        for i in range(0, len(tokens) - 1, 2):
            if re.fullmatch(r"[-\d]+\t[-\d]+", tokens[i].strip()):
                paths.append(tokens[i + 1])
        return paths or None

    def _apply_patch(self, ws: Path, patch_file: Path, patch: str) -> tuple[bool, str]:
        return git_apply_ladder(ws, patch_file, patch, self._run)

    def evaluate(self, bug: BugMeta, model_output: str) -> RewardResult:
        t0 = time.time()
        patch = extract_patch(model_output)
        parsed = patch is not None
        hacked = False
        applied = compiled = False
        trigger: Optional[TestOutcome] = None
        regression: Optional[TestOutcome] = None
        detail = ""

        safe, reason = validate_patch_safety(patch, bug.allowed_globs)
        if parsed and not safe:
            if reason.startswith("test_file_touched"):
                hacked = True
            detail = reason

        ws: Optional[Path] = None
        try:
            if parsed and safe:
                ws = self.prepare_workspace(bug)
                before = snapshot_test_hashes(ws)

                patch_file = ws.parent / f"{ws.name}.patch"
                with open(patch_file, "w", encoding="utf-8", newline="\n") as fh:
                    fh.write(patch if patch.endswith("\n") else patch + "\n")

                # 第二道闸（apply 前）：git 解析器复核路径。文本正则可被八进制引号
                # 路径（'"a/src/\164est/..."') / 含空格路径绕过，git 的解析不会。
                git_paths = self._numstat_paths(ws, patch_file)
                if git_paths:
                    ok2, reason2 = validate_paths_safety(git_paths, bug.allowed_globs)
                    if not ok2:
                        if reason2.startswith("test_file_touched"):
                            hacked = True
                        safe = False   # 闸不过 → 不让补丁落盘（与文本闸同语义）
                        detail = detail or reason2

                if safe:
                    # 重试阶梯：固定行数 → --recount，失败才升级（见 git_apply_ladder）
                    applied, apply_err = self._apply_patch(ws, patch_file, patch)
                    if not applied:
                        detail = detail or f"git_apply:{apply_err}"

                    after = snapshot_test_hashes(ws)
                    hacked = hacked or not check_untouched(before, after)

                if applied and not hacked and self._fastpath_ok(ws):
                    # 快路径：一次 `test -r` 完成编译+trigger+回归（前提在 prepare
                    # 阶段已校验 trigger ⊆ relevant，见 _export_relevant）
                    rc, out = self._run([str(self.bin), "test", "-w", str(ws), "-r"])
                    if "Cannot compile" in out:
                        compiled = False
                        detail = detail or f"compile:{out[-300:]}"
                    else:
                        compiled = True  # d4j-test 内部先 compile_tests，跑到出结果即编译通过
                        failing = parse_failing_tests(out) if rc != -99 else None
                        if failing is None:
                            # 超时 / ant 崩溃：无法区分编译与运行失败，保守全挂
                            trigger = TestOutcome(len(bug.trigger_tests),
                                                  len(bug.trigger_tests),
                                                  list(bug.trigger_tests))
                            detail = detail or ("test_run_timeout" if rc == -99
                                                else f"test_run_error:{out[-200:]}")
                        else:
                            failed = match_failed_triggers(bug.trigger_tests, failing)
                            trigger = TestOutcome(len(bug.trigger_tests),
                                                  len(failed), failed)
                            if trigger.all_pass:
                                # relevant 集里的其余失败 = 回归挂（0.3 档）
                                regression = TestOutcome(max(len(failing), 1),
                                                         len(failing), failing)
                elif applied and not hacked:
                    # 慢路径（旧缓存无标记 / trigger⊄relevant）：独立 compile + 逐条 trigger
                    rc, out = self._run([str(self.bin), "compile", "-w", str(ws)])
                    compiled = (rc == 0)
                    if not compiled:
                        detail = detail or f"compile:{out[-300:]}"

                    if compiled:
                        # -t 只接单个测试名（class::method），逐条跑 trigger
                        n_failed = 0
                        failed_names: list = []
                        for t in bug.trigger_tests:
                            rc, out = self._run([str(self.bin), "test", "-w", str(ws),
                                                 "-t", t])
                            failing = parse_failing_tests(out) if rc != -99 else None
                            if failing is None:
                                # 超时 / 命令没跑成 → 保守记失败
                                n_failed += 1
                                failed_names.append(t)
                                detail = detail or ("trigger_timeout" if rc == -99
                                                    else f"trigger_run_error:{out[-200:]}")
                            elif failing:
                                n_failed += 1
                                failed_names.append(t)
                        trigger = TestOutcome(len(bug.trigger_tests), n_failed, failed_names)
                        if trigger.all_pass:
                            # 回归收窄到 relevant tests（一次 ant）；清单缺失退化为全量
                            cmd = [str(self.bin), "test", "-w", str(ws)]
                            if self._relevant_tests(ws):
                                cmd.append("-r")
                            rc2, out2 = self._run(cmd)
                            failing2 = parse_failing_tests(out2) if rc2 != -99 else None
                            if failing2 is None:
                                detail = detail or ("regression_timeout" if rc2 == -99
                                                    else f"regression_run_error:{out2[-200:]}")
                                regression = TestOutcome(1, 1)  # 无法验证 → 保守按回归挂处理
                            else:
                                # 真实输出不含测试总数：n_total 用 max(len,1) 作"已跑成"哨兵，
                                # all_pass 等价于失败列表为空
                                regression = TestOutcome(max(len(failing2), 1),
                                                         len(failing2), failing2)
        except (RuntimeError, OSError) as e:
            detail = detail or f"env_error:{e}"
        finally:
            if ws is not None:
                shutil.rmtree(ws, ignore_errors=True)

        reward, stage = grade_reward(parsed=parsed, applied=applied, compiled=compiled,
                                     hacked=hacked, trigger=trigger, regression=regression)
        return RewardResult(bug_id=bug.bug_id, reward=reward, stage=stage,
                            detail=detail[:500], duration_s=round(time.time() - t0, 2))


def farm_evaluate(executor_kwargs: dict, tasks: list[tuple[dict, str]],
                  max_workers: int = 8) -> list[dict]:
    """并行 reward 判定（进程池）。tasks = [(split_record, model_output), ...]

    v0 简化实现：每个任务独立构造 executor（cache 复用保证不重复 checkout）。
    TODO: 换成持久 worker 池（initializer 里建 executor），减少进程开销。
    """
    results: list[dict] = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_farm_worker, executor_kwargs, rec, out) for rec, out in tasks]
            for fut in futures:
                results.append(fut.result())
    except (OSError, RuntimeError):
        # 串行兜底（Windows spawn 问题 / 资源不足时）
        results = [_farm_worker(executor_kwargs, rec, out) for rec, out in tasks]
    return results


def _farm_worker(executor_kwargs: dict, rec: dict, model_output: str) -> dict:
    ex = RewardExecutor(**executor_kwargs)
    return ex.evaluate(BugMeta.from_split_record(rec), model_output).to_dict()
