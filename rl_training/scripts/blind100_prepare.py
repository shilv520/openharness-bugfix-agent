#!/usr/bin/env python3
"""blind100_prepare.py — R3 盲测抽样 + 适配器（实例侧运行，预注册见 docs/prereg_blind100_D1.md）。

是什么：从 953 条训练池中复现 prescreen 的 shuffle（random.Random(42).shuffle）定位 575 条
未筛候选，预注册排除 Lang-6/Lang-30 挂死家族后，用固定种子 20261111 抽 100 条，join
grpo_golden.jsonl（953 条全量含 golden_patch），并从训练 prompt 反解 jira/modified_classes，
产出 blind_eval.jsonl + manifest.json。

为什么反解 + 字节断言：eval_synthetic 用 build_prompt(rec) 现场重建 prompt；训练记录只存
messages 成品、不存 jira/modified_classes 字段——缺了它们重建 prompt 会退化成 "(unknown)"，
与训练时不一致，对比失效。故从 messages[1].content 反解这两个字段，并**逐条断言重建 prompt
与训练 prompt 字节级一致**（一致性由构造保证，断言兜底抓解析 bug；任何一条不过 → 整体拒产）。

安全性质：只读数据文件、只写 /root/blind100/；任何断言失败 → 非零退出、不产出半成品。
"""
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/root/rl_training")        # prompt_builder 内部 import reward.reward 需要根在 path
sys.path.insert(0, "/root/rl_training/data")
from prompt_builder import build_prompt  # noqa: E402

GRPO_DATA = Path("/root/autodl-tmp/grpo_data")
TRAIN = GRPO_DATA / "grpo_train.jsonl"
GOLDEN = GRPO_DATA / "grpo_golden.jsonl"
KEPT = GRPO_DATA / "grpo_train_prescreened.jsonl"
SYN_EVAL_CLEAN = Path("/root/autodl-tmp/sft_data/syn_eval_clean.jsonl")
OUT = Path("/root/blind100")

SCREENED_N = 378        # prescreen_report.json: screened=378（shuffle=True seed=42, kept 168 + dropped 210）
PRESCREEN_SEED = 42     # 与 prescreen_grpo.py 默认一致
SAMPLE_N = 100
SAMPLE_SEED = 20261111  # D1 日期，预注册固定
HANG_FAMILY = {"Lang-6", "Lang-30"}   # 预注册排除（Lang-6 预筛熔断实锤前科）

EVAL_FIELDS = ("bug_id", "real_bug_id", "project", "vid", "file_rel", "line_no",
               "mutation", "seed_k", "buggy_src", "trigger_tests", "allowed_globs")

_JIRA_RE = re.compile(r"^- Bug ID: .+ \((.+)\)$", re.MULTILINE)
_CLASSES_HDR = "## Classes known to contain the bug (fault localization)\n"


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def parse_prompt_extras(user_text: str) -> tuple[str, list[str]]:
    """从训练 prompt 反解 (jira, modified_classes)。USER_TEMPLATE 结构固定（prompt_builder.py）。"""
    m = _JIRA_RE.search(user_text)
    assert m, "prompt 里找不到 '- Bug ID: ... (jira)' 行"
    jira = m.group(1)
    i = user_text.index(_CLASSES_HDR) + len(_CLASSES_HDR)
    j = user_text.index("\n\n## Buggy code", i)
    lines = [l[2:] for l in user_text[i:j].splitlines() if l.startswith("- ")]
    classes = [] if lines == ["(unknown)"] else lines
    assert classes, f"modified_classes 解析为空（模板漂移？）: {user_text[i:j]!r}"
    return jira, classes


def main() -> None:
    rows = load_jsonl(TRAIN)
    assert len(rows) == 953, f"train rows {len(rows)} != 953"

    # 1) 复现 prescreen 的 shuffle，定位未筛 575 条（不是文件尾！）
    shuffled = rows[:]
    random.Random(PRESCREEN_SEED).shuffle(shuffled)
    unscreened = shuffled[SCREENED_N:]
    assert len(unscreened) == 575, f"unscreened {len(unscreened)} != 575"

    # 2) 硬断言：与 kept-168 池、与 21 题 held-out 评测集零交集（盲资源纯净性）
    kept_ids = {r["bug_id"] for r in load_jsonl(KEPT)}
    assert len(kept_ids) == 168, f"kept {len(kept_ids)} != 168"
    unscreened_ids = {r["bug_id"] for r in unscreened}
    overlap_kept = unscreened_ids & kept_ids
    assert not overlap_kept, f"与 kept-168 交集非空（shuffle 复现错误？）: {sorted(overlap_kept)[:5]}"
    eval_ids = {r["bug_id"] for r in load_jsonl(SYN_EVAL_CLEAN)}
    overlap_eval = unscreened_ids & eval_ids
    assert not overlap_eval, f"与 syn_eval_clean 交集非空: {sorted(overlap_eval)[:5]}"

    # 3) 预注册排除挂死家族
    family_excluded = [r for r in unscreened if r["real_bug_id"] in HANG_FAMILY]
    eligible = [r for r in unscreened if r["real_bug_id"] not in HANG_FAMILY]
    assert len(eligible) >= SAMPLE_N, f"排除挂死家族后仅剩 {len(eligible)} < {SAMPLE_N}"

    # 4) 固定种子抽样（可复现）
    sample = random.Random(SAMPLE_SEED).sample(eligible, SAMPLE_N)

    # 5) join golden 拿 golden_patch（同 bug_id 必须同 buggy_src，双保险）
    golden_by_id = {r["bug_id"]: r for r in load_jsonl(GOLDEN)}
    assert len(golden_by_id) == 953, f"golden {len(golden_by_id)} != 953"
    eval_recs = []
    for tr in sample:
        g = golden_by_id.get(tr["bug_id"])
        assert g is not None, f"golden 缺 {tr['bug_id']}"
        assert g["buggy_src"] == tr["buggy_src"], f"buggy_src 不一致 {tr['bug_id']}"
        assert g.get("golden_patch"), f"golden_patch 为空 {tr['bug_id']}"
        rec = {k: tr[k] for k in EVAL_FIELDS}
        rec["golden_patch"] = g["golden_patch"]

        # 6) prompt 字节级一致性：反解 jira/modified_classes → build_prompt 重建 → 必须逐字节等于训练 prompt
        train_prompt = tr["messages"][1]["content"]
        jira, classes = parse_prompt_extras(train_prompt)
        rec["jira"] = jira
        rec["modified_classes"] = classes
        code = f"// File: {rec['file_rel']}\n{rec['buggy_src']}"   # eval_synthetic L215 同款
        rebuilt = build_prompt(rec, code)
        assert rebuilt == train_prompt, f"重建 prompt != 训练 prompt（{tr['bug_id']}）"
        eval_recs.append(rec)

    # 7) 产出
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "blind_eval.jsonl").open("w", encoding="utf-8") as f:
        for rec in eval_recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "prereg": "rl_training/docs/prereg_blind100_D1.md",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": str(TRAIN),
        "shuffle_replicated": {"seed": PRESCREEN_SEED, "screened_n": SCREENED_N,
                               "unscreened_n": len(unscreened)},
        "sample": {"n": SAMPLE_N, "seed": SAMPLE_SEED,
                   "hang_family_excluded": sorted(HANG_FAMILY),
                   "family_excluded_n": len(family_excluded),
                   "eligible_n": len(eligible)},
        "prompt_identity": "build_prompt(rec) == train messages[1].content 逐条字节断言通过（100/100）",
        "sampled": [{"bug_id": r["bug_id"], "real_bug_id": r["real_bug_id"],
                     "seed_k": r.get("seed_k"), "mutation": r.get("mutation"),
                     "project": r.get("project")} for r in sample],
        "sample_mutation_dist": dict(Counter(r.get("mutation") for r in sample).most_common()),
        "sample_project_dist": dict(Counter(r.get("project") for r in sample).most_common()),
        # 永久排除清单：全部 575 未筛 bug_id（其中抽中的 100 条跑完后永不进训练池；
        # 未抽中的 475 条仍是干净候选，供未来扩池——此清单用于扩池时对照去重）
        "unscreened_all_bug_ids": sorted(unscreened_ids),
        "family_excluded_bug_ids": sorted(r["bug_id"] for r in family_excluded),
        "assertions_passed": ["train==953", "unscreened==575", "kept==168",
                              "overlap_kept==0", "overlap_syn_eval_clean==0",
                              "eligible>=100", "golden==953", "buggy_src_match",
                              "golden_patch_nonempty", "prompt_byte_identical_x100"],
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[blind100] unscreened=575 (shuffle 复现 seed={PRESCREEN_SEED}, screened={SCREENED_N})")
    print(f"[blind100] 家族排除 {sorted(HANG_FAMILY)}: {len(family_excluded)} 条 → eligible={len(eligible)}")
    print(f"[blind100] 抽样 n={SAMPLE_N} seed={SAMPLE_SEED}")
    print(f"[blind100] prompt 字节级一致性断言 {len(eval_recs)}/{len(eval_recs)} 通过")
    print(f"[blind100] 算子分布: {manifest['sample_mutation_dist']}")
    print(f"[blind100] 项目分布: {manifest['sample_project_dist']}")
    print(f"[blind100] 产出: {OUT}/blind_eval.jsonl ({len(eval_recs)} 行) + manifest.json")
    print("BLIND100_PREPARE_OK")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"BLIND100_PREPARE_ASSERT_FAILED: {e}", file=sys.stderr)
        sys.exit(1)
