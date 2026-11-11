#!/usr/bin/env python3
"""
build_grpo_data.py — 把 syn_staging_train.jsonl 转成 GRPO 训练数据（plan §4，策略 (a) 合成-only）

为什么这么建（关键设计，全部经实例 schema 实测确认）：

  * syn_staging_train.jsonl 的 978 行**不直接带** buggy_src/trigger_tests/golden_patch/vid，
    但每行带一个 `pair = {"messages":[system,user,assistant]}`——这就是 SFT 训练样本本身：
      - user  = build_prompt 产出（含「## Failing test(s)」= trigger_tests、「## Buggy code」= // File+buggy_src）
      - assistant = fence_patch(golden_patch)
    分工：**prompt 与 trigger_tests 从 pair 抽取**（保证模板逐字节同 SFT/评测），
    但 **buggy_src / golden_patch 用 inject_bug(fixed_src, seed_k) 重生成**（不用 prompt 抽取值）：
      (1) GRPO prompt = pair.messages[0:2]（system+user）→ 与 SFT/评测**逐字节同模板**（铁律，plan §4.2）；
      (2) trigger_tests 从 user prompt「## Failing test(s)」解析（= 当初测出并写进 prompt 的名单）；
      (3) **为什么 buggy_src 必须重生成**：build_prompt 对 buggy 代码 .strip() 丢了尾换行，prompt 抽取值
          无尾换行；当 golden 的 hunk 落在文件末尾时，重放「无尾换行」的 buggy_src 会让 git apply 匹配不上
          EOF 上下文 → patch_apply_failed（train golden 正对照实测 Time-26，9/10）。重生成拿到与
          build_sft_data/syn_eval_clean 同口径的**未 strip 精确字节**，并用 .strip() 奇偶校验对齐 prompt；
      (4) GRPO 丢掉 assistant turn（奖励来自执行，不是标签）。

  * trigger_tests 用「解析 user prompt」而非重新测量：解析出的就是当初 attempt_one 测出、
    并被 build_prompt 写进 prompt 的那一份敏感测试名单；用 audit_rec.n_sensitive 交叉校验条数。

  * held-out 硬断言：train 的 (src_bug,seed_k) 与 (file,line_no,mutation) 三元组都不得出现在
    syn_eval_clean(21) 里（坑 #30 曾实证 23/50 syn_eval 与训练标签逐字节相同 → 必须查）。
    实测 62 个 train src_bug 与 7 个 eval src_bug 不相交，但仍逐行断言。

  * golden_patch **不进**训练文件（单独写 grpo_golden.jsonl，只给 pre-flight 正对照用）。
    test-split 官方 patch 永不进任何文件（项目纪律）。

  * 离线 token 过滤：用 sft_merged 的 tokenizer 量 prompt 长度，>max_prompt_len(4096) 丢弃
    （配合训练时 --truncation_strategy delete；绝不 left 截断——那会砍掉 system prompt 里的
    「ONE diff fence / NEVER modify test files」指令，plan §8 pitfall #2）。

  * 冒烟子集 grpo_train_smoke.jsonl = prompt 最长的 8 行（HF generate 把 batch pad 到最长，
    最长 prompt = worst-case VRAM，plan §6）。

产物（默认 /root/autodl-tmp/grpo_data/）：
  grpo_train.jsonl        训练集（messages + 全部 reward 元数据列，无 golden）
  grpo_golden.jsonl       golden 正对照集（pre-flight 用，含 golden_patch）
  grpo_train_smoke.jsonl  最长 8 行（2 步 GPU 冒烟用）
  build_report.json       统计 + 断言结果

跑在 swift venv（需 transformers 量 token）：
  /root/autodl-tmp/envs/swift/bin/python /root/rl_training/rl/build_grpo_data.py
纪律：CPU-only，不碰 GPU；不改 reward.py；无关机/cron/timer。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

# ---- sys.path：injector 在 rl_training/ 下（本文件在 rl_training/rl/）----
_RL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _RL_ROOT not in sys.path:
    sys.path.insert(0, _RL_ROOT)
from injector.bug_injector import inject_bug  # noqa: E402  # 重生成精确 buggy_src/golden

# ---- user prompt 模板锚点（来自 data/prompt_builder.py USER_TEMPLATE，唯一事实源，冻结）----
_FAILING_HDR = "## Failing test(s) that expose the bug"
_CLASSES_HDR = "## Classes known to contain the bug"
_BUGGY_CODE_RE = re.compile(r"## Buggy code\s*```java\n(.*?)\n```", re.DOTALL)
_DIFF_FENCE_RE = re.compile(r"```diff\n(.*?)\n```", re.DOTALL)
_FILE_PREFIX = "// File: "


def _section(text: str, start_hdr: str, end_hdr: str | None = None) -> str:
    if start_hdr not in text:
        raise ValueError(f"prompt 缺锚点: {start_hdr!r}")
    i = text.index(start_hdr) + len(start_hdr)
    j = text.index(end_hdr, i) if end_hdr and end_hdr in text[i:] else len(text)
    return text[i:j].strip("\n")


def parse_trigger_tests(user_text: str) -> list[str]:
    sec = _section(user_text, _FAILING_HDR, _CLASSES_HDR)
    tests = []
    for line in sec.splitlines():
        line = line.strip()
        if line.startswith("- "):
            t = line[2:].strip()
            if t and t != "(unknown)":
                tests.append(t)
    return tests


def parse_code_block(user_text: str) -> str:
    m = _BUGGY_CODE_RE.search(user_text)
    if not m:
        raise ValueError("prompt 里找不到 '## Buggy code' 的 ```java 代码块")
    return m.group(1)


def split_file_and_src(code: str) -> tuple[str, str]:
    """code = '// File: {file_rel}\\n{buggy_src}'（build_prompt 里被 .strip() 过，尾换行已去）。"""
    head, _, rest = code.partition("\n")
    if not head.startswith(_FILE_PREFIX):
        raise ValueError(f"代码块首行不是 '// File: ...': {head[:60]!r}")
    return head[len(_FILE_PREFIX):].strip(), rest


def parse_golden(assistant_text: str) -> str:
    m = _DIFF_FENCE_RE.search(assistant_text)
    return (m.group(1) if m else assistant_text).strip("\n")


def vid_of(src_bug: str) -> int:
    """src_bug='Chart-14' → 14。"""
    return int(src_bug.rsplit("-", 1)[1])


def build_record(row: dict, cache_fixed: Path) -> tuple[dict, dict]:
    """一行 syn_staging_train → (grpo_train_rec, golden_rec)。prompt 抽取 + 精确字节重生成。"""
    pair = row["pair"]
    msgs = pair["messages"]
    assert isinstance(msgs, list) and len(msgs) == 3, f"pair.messages 形态异常: {type(msgs)}"
    sys_m, user_m, asst_m = msgs
    assert sys_m["role"] == "system" and user_m["role"] == "user" and asst_m["role"] == "assistant"
    user_text = user_m["content"]

    src_bug = row["src_bug"]                       # 'Chart-14'
    project = row["project"]                       # 'Chart'
    seed_k = row["seed_k"]
    file_rel = row["file"]                         # 相对路径
    vid = vid_of(src_bug)
    assert src_bug == f"{project}-{vid}", f"src_bug/project/vid 不自洽: {src_bug} {project} {vid}"

    trigger_tests = parse_trigger_tests(user_text)
    code = parse_code_block(user_text)
    code_file_rel, prompt_buggy_src = split_file_and_src(code)   # prompt 里是被 .strip() 过的（丢尾换行）
    ar = row.get("audit_rec") or {}

    # ---- 交叉断言：prompt 抽取自洽（抓模板漂移 / 解析 bug）----
    assert code_file_rel == file_rel, f"代码块 file_rel({code_file_rel}) != row.file({file_rel})"
    n_sensitive = row.get("n_sensitive")
    if n_sensitive is not None:
        assert len(trigger_tests) == n_sensitive, \
            f"trigger_tests 条数 {len(trigger_tests)} != n_sensitive {n_sensitive} ({src_bug}/{seed_k})"
    assert prompt_buggy_src.strip(), f"prompt buggy_src 为空 ({src_bug}/{seed_k})"
    if ar.get("mutation"):
        assert ar["mutation"] == row.get("mutation"), "audit_rec.mutation != row.mutation"

    # ---- 重生成精确字节（修 EOF 坑）----
    # build_prompt 对 buggy 代码做了 .strip()，尾换行丢失；当 golden 的 hunk 落在文件末尾时，
    # 把「无尾换行」的 buggy_src 重放后 git apply 匹配不上 EOF 上下文 → patch_apply_failed
    # （train golden 正对照实测 Time-26 命中，9/10）。故 buggy_src/golden_patch 不用 prompt 抽取值，
    # 而用与 build_sft_data 完全相同的 inject_bug(fixed_src, seed_k) 复现**未 strip** 的精确字节
    # （与 syn_eval_clean 同口径，其 golden 21/21 已证），reward 重放才能逐字节复现当初的测量态。
    fixed_path = cache_fixed / src_bug / file_rel
    if not fixed_path.exists():
        raise ValueError(f"fixed 缓存缺文件: {fixed_path}")
    fixed_src = fixed_path.read_bytes().decode("utf-8")
    inj = inject_bug(fixed_src, rel_path=file_rel, rng=random.Random(seed_k))
    if inj is None:
        raise ValueError(f"inject_bug 无注入点（seed 不复现？）{src_bug}/{seed_k}")
    buggy_src = inj.buggy_src                      # 未 strip 的精确字节（含尾换行）
    golden_patch = inj.golden_patch
    # 奇偶校验：重生成.strip() 必须 == prompt 抽取值（抓缓存漂移 / seed 错位 / 模板改动）
    assert buggy_src.strip() == prompt_buggy_src, \
        f"重生成 buggy_src 与 prompt 抽取不一致 ({src_bug}/{seed_k})"
    assert inj.mutation == row.get("mutation"), \
        f"inj.mutation {inj.mutation} != row.mutation ({src_bug}/{seed_k})"
    assert inj.line_no == row.get("line_no"), \
        f"inj.line_no {inj.line_no} != row.line_no ({src_bug}/{seed_k})"

    bug_id = f"SYN-{src_bug}-{seed_k}"             # 与 eval_clean bug_id 同格式
    allowed_globs = [file_rel]                     # 只准动 buggy 文件（堵 reward-hacking，plan §3.4）

    train_rec = {
        "messages": [{"role": "system", "content": sys_m["content"]},
                     {"role": "user", "content": user_text}],   # 丢 assistant（GRPO 不用标签）
        "bug_id": bug_id, "real_bug_id": src_bug, "project": project, "vid": vid,
        "file_rel": file_rel, "buggy_src": buggy_src, "trigger_tests": trigger_tests,
        "allowed_globs": allowed_globs, "mutation": row.get("mutation"),
        "line_no": row.get("line_no"), "seed_k": seed_k,
    }
    golden_rec = {
        "bug_id": bug_id, "real_bug_id": src_bug, "project": project, "vid": vid,
        "file_rel": file_rel, "buggy_src": buggy_src, "trigger_tests": trigger_tests,
        "allowed_globs": allowed_globs, "golden_patch": golden_patch, "seed_k": seed_k,
    }
    return train_rec, golden_rec


def load_eval_exclusions(eval_clean: Path) -> tuple[set, set, set]:
    """syn_eval_clean(21) 的排除集：(src_bug,seed_k) / (file,line,mutation) / src_bug。"""
    rows = [json.loads(l) for l in eval_clean.read_text(encoding="utf-8").splitlines() if l.strip()]
    seed_set, triple_set, bug_set = set(), set(), set()
    for r in rows:
        src_bug = f"{r['project']}-{r['vid']}"
        bug_set.add(src_bug)
        seed_set.add((src_bug, r["seed_k"]))
        triple_set.add((r.get("file_rel"), r.get("line_no"), str(r.get("mutation"))))
    return seed_set, triple_set, bug_set


def main() -> None:
    ap = argparse.ArgumentParser(description="syn_staging_train → GRPO 训练数据（策略 a 合成-only）")
    ap.add_argument("--train", default="/root/autodl-tmp/sft_data/syn_staging_train.jsonl")
    ap.add_argument("--eval-clean", default="/root/autodl-tmp/sft_data/syn_eval_clean.jsonl")
    ap.add_argument("--val-staging", default="/root/autodl-tmp/sft_data/syn_staging_val.jsonl",
                    help="R4 护栏：从中提取全部 val src_bug 做黑名单（比 eval_clean 反推的 7/14 覆盖全）")
    ap.add_argument("--allow-same-srcbug", action="store_true",
                    help="R4 护栏：允许与 eval_clean 同 src_bug 不同 mutation 的行入池（默认拒绝）")
    ap.add_argument("--cache-fixed", default="/root/autodl-tmp/d4j_cache_fixed",
                    help="fixed 世界缓存（重生成未 strip 的精确 buggy_src/golden 用）")
    ap.add_argument("--out-dir", default="/root/autodl-tmp/grpo_data")
    ap.add_argument("--model-dir", default="/root/autodl-tmp/sft_merged", help="量 token 用的 tokenizer")
    ap.add_argument("--max-prompt-len", type=int, default=4096)
    ap.add_argument("--smoke-n", type=int, default=8)
    ap.add_argument("--no-token-filter", action="store_true", help="跳过 token 长度过滤（无量 tokenizer 时）")
    cfg = ap.parse_args()

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_fixed = Path(cfg.cache_fixed)

    rows = [json.loads(l) for l in Path(cfg.train).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[build] 读入 train 行: {len(rows)}", flush=True)

    # ---- R4 护栏① split 红线：输入必须是 train split（此前只靠「--train 指向 staging_train」的约定）----
    bad_split = [f"{r.get('src_bug')}/{r.get('seed_k')}" for r in rows
                 if r.get("split") not in (None, "train")]
    if bad_split:
        raise SystemExit(f"[build] ❌ split 红线：{len(bad_split)} 行 split != train"
                         f"（如 {bad_split[:3]}）——拒绝构建（防 val/test 行混入训练池）")
    no_split = sum(1 for r in rows if r.get("split") is None)
    if no_split:
        print(f"[build] ⚠️ {no_split} 行缺 split 字段，保守跳过（不入池）", flush=True)

    # ---- R4 护栏② val/test src_bug 全量黑名单 ----
    # 锁定 split 的完整 bug 清单（来源：make_split 锁定切分 + evidence/01_baseline 两份报告的 per_bug 键；
    # split 锁定是红线、清单永不变化，故可硬编码；syn_staging_val 只 stage 了 14 个 val bug 中的 7 个，
    # 单靠它反推会漏掉 Chart-25/Lang-42/Lang-45/Lang-50/Lang-7/Lang-8/Time-11 七个）。
    VAL_SPLIT_BUGS = {"Chart-18", "Chart-25", "Chart-5", "Lang-19", "Lang-26", "Lang-28",
                      "Lang-42", "Lang-45", "Lang-50", "Lang-7", "Lang-8",
                      "Time-11", "Time-2", "Time-8"}
    TEST_SPLIT_BUGS = {"Chart-1", "Chart-21", "Chart-22", "Chart-4", "Chart-8", "Chart-9",
                       "Lang-1", "Lang-11", "Lang-12", "Lang-15", "Lang-20", "Lang-24", "Lang-31",
                       "Lang-41", "Lang-49", "Lang-53", "Lang-55", "Lang-56", "Lang-60", "Lang-63",
                       "Lang-64", "Time-13", "Time-15", "Time-18", "Time-22", "Time-6", "Time-9"}
    val_bugs: set = set(VAL_SPLIT_BUGS | TEST_SPLIT_BUGS)   # 变量名沿用 val_bugs：语义=「held-out 黑名单」
    val_p = Path(cfg.val_staging)
    if val_p.exists():
        staged = {r["src_bug"] for r in
                  (json.loads(l) for l in val_p.read_text(encoding="utf-8").splitlines() if l.strip())
                  if r.get("src_bug")}
        extra = staged - val_bugs
        if extra:   # staging_val 出现锁定清单之外的 bug → split 定义变了？响亮报警而不是静默并入
            print(f"[build] ⚠️⚠️ val staging 含锁定清单外的 src_bug: {sorted(extra)}——请人工核对 split！", flush=True)
        val_bugs |= staged
    print(f"[build] held-out 黑名单: {len(val_bugs)} 个 src_bug（val14+test27+staging 并集；"
          f"2026-11-11 实测当前 train staging 62 bug 与之零交集）", flush=True)

    seed_excl, triple_excl, bug_excl = load_eval_exclusions(Path(cfg.eval_clean))
    print(f"[build] eval_clean 排除集: {len(seed_excl)} (src_bug,seed_k) / {len(triple_excl)} 三元组 / "
          f"{len(bug_excl)} src_bug = {sorted(bug_excl)}", flush=True)

    train_recs, golden_recs = [], []
    stats = Counter()
    overlap_hits = []
    for row in rows:
        stats["total"] += 1
        if row.get("status") not in (None, "ok", "kept", "pass"):
            # staging 里若有非 kept 行（status 字段），保守跳过；实测 status='ok'
            stats[f"status_{row.get('status')}"] += 1
            continue
        if row.get("split") is None:
            stats["no_split_skipped"] += 1        # R4 护栏①：来源不明的行不入池
            continue
        if row.get("src_bug") in val_bugs:
            stats["val_bug_rejected"] += 1        # R4 护栏②：val bug 的任何变体都拒绝
            continue
        try:
            tr, gr = build_record(row, cache_fixed)
        except (ValueError, AssertionError, KeyError) as e:
            stats["build_error"] += 1
            print(f"  ⚠️ 跳过（抽取/断言失败）{row.get('src_bug')}/{row.get('seed_k')}: {e}", flush=True)
            continue
        # held-out 重叠硬检查
        src_bug, seed_k = tr["real_bug_id"], tr["seed_k"]
        if src_bug in val_bugs:                   # R4 护栏②第二层：构建后再验（防 row/构建字段错位）
            stats["val_bug_rejected_postbuild"] += 1
            continue
        triple = (tr["file_rel"], tr.get("line_no"), str(tr.get("mutation")))
        if (src_bug, seed_k) in seed_excl or triple in triple_excl:
            overlap_hits.append((src_bug, seed_k))
            stats["overlap_excluded"] += 1
            continue
        if src_bug in bug_excl:
            # R4 护栏③：同一真实 bug 但不同 mutation——此前默认保留（只计数），D1 复核改为默认拒绝：
            # 合成设定下虽不算标签泄漏（不同 seed_k=不同变异=不同标签），但同文件同上下文与 held-out
            # 相似度过高，扩池时会被放大；确需保留用 --allow-same-srcbug（历史 953 池该计数为 0，行为不变）。
            if cfg.allow_same_srcbug:
                stats["same_srcbug_diff_mutation"] += 1
            else:
                stats["same_srcbug_rejected"] += 1
                continue
        train_recs.append(tr)
        golden_recs.append(gr)

    if overlap_hits:
        print(f"  ⚠️ held-out 重叠剔除 {len(overlap_hits)} 行: {overlap_hits[:10]}", flush=True)

    # ---- token 过滤（用 sft_merged tokenizer 量 prompt 长度）----
    lengths = []                                            # 与（过滤后的）train_recs 严格同序对齐
    if not cfg.no_token_filter:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(cfg.model_dir, trust_remote_code=True)
            kept_with_len = []
            for tr in train_recs:
                ids = tok.apply_chat_template(tr["messages"], add_generation_prompt=True, tokenize=True)
                L = len(ids)
                if L <= cfg.max_prompt_len:
                    kept_with_len.append((tr, L))
                else:
                    stats["token_filtered"] += 1
            dropped = len(train_recs) - len(kept_with_len)
            # golden_recs 与 train_recs 同序同 bug_id，按保留的 bug_id 过滤 golden
            keep_ids = {tr["bug_id"] for tr, _ in kept_with_len}
            golden_recs = [gr for gr in golden_recs if gr["bug_id"] in keep_ids]
            train_recs = [tr for tr, _ in kept_with_len]
            lengths = [L for _, L in kept_with_len]         # 对齐 kept，供 smoke 选最长 prompt
            print(f"[build] token 过滤: 丢 {dropped} 行 (prompt>{cfg.max_prompt_len})", flush=True)
        except Exception as e:                              # noqa: BLE001
            print(f"[build] ⚠️ token 过滤跳过（tokenizer 不可用）: {e}", flush=True)
            stats["token_filter_skipped"] += 1

    # ---- 写产物 ----
    train_p = out / "grpo_train.jsonl"
    golden_p = out / "grpo_golden.jsonl"
    smoke_p = out / "grpo_train_smoke.jsonl"
    with open(train_p, "w", encoding="utf-8") as f:
        for tr in train_recs:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    with open(golden_p, "w", encoding="utf-8") as f:
        for gr in golden_recs:
            f.write(json.dumps(gr, ensure_ascii=False) + "\n")

    # 冒烟子集：prompt 最长的 smoke_n 行（worst-case VRAM）。有 lengths 就按它排，否则按 buggy_src 长度。
    if lengths and len(lengths) == len(train_recs):
        order = sorted(range(len(train_recs)), key=lambda i: lengths[i], reverse=True)
    else:
        order = sorted(range(len(train_recs)),
                       key=lambda i: len(train_recs[i]["buggy_src"]), reverse=True)
    smoke = [train_recs[i] for i in order[: cfg.smoke_n]]
    with open(smoke_p, "w", encoding="utf-8") as f:
        for tr in smoke:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")

    # ---- 报告 ----
    by_bug = Counter(tr["real_bug_id"] for tr in train_recs)
    report = {
        "train_in": len(rows), "train_out": len(train_recs), "golden_out": len(golden_recs),
        "smoke_out": len(smoke), "stats": dict(stats),
        "distinct_src_bug": len(by_bug), "distinct_files": len({tr["file_rel"] for tr in train_recs}),
        "by_project": dict(Counter(tr["project"] for tr in train_recs)),
        "prompt_len": ({
            "n": len(lengths), "min": min(lengths), "max": max(lengths),
            "mean": round(sum(lengths) / len(lengths), 1),
            "p90": sorted(lengths)[int(len(lengths) * 0.9)] if lengths else 0,
        } if lengths else "token_filter_skipped"),
        "overlap_excluded": stats.get("overlap_excluded", 0),
        "eval_src_bugs": sorted(bug_excl),
        "smoke_bug_ids": [tr["bug_id"] for tr in smoke],
        "out_dir": str(out),
    }
    (out / "build_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\n=== build_grpo_data 报告 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\ntrain : {train_p}\ngolden: {golden_p}\nsmoke : {smoke_p}")
    bad = (stats.get("build_error", 0) > 0) or (report["train_out"] == 0)
    print("BUILD_GRPO_DATA_DONE" if not bad else "BUILD_GRPO_DATA_DONE_WITH_WARNINGS")


if __name__ == "__main__":
    main()
