#!/usr/bin/env python3
"""analyze_blind100.py — R3 盲测判定（预注册 docs/prereg_blind100_D1.md §4/§5 的实现）。

是什么：读两臂 syneval_blind100_{sft,grpo}.jsonl（+report），按预注册分析计划算：
总 pass@1 + Wilson CI、成对 McNemar exact + 配对 Wald CI、按算子/项目分解与净增益集中度、
stage 分布、golden 汇总，最后对照 §4 判定表打印 GO / NO-GO / GRAY / DEFER。

为什么：判定标准必须先于数据写死、由脚本机械执行——防止看到数字后挪门柱。
怎么用：python scripts/analyze_blind100.py <evidence_dir>
  evidence_dir 需含 syneval_blind100_sft.jsonl 与 syneval_blind100_grpo.jsonl。
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b01: int, b10: int) -> float:
    nd = b01 + b10
    if nd == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(math.comb(nd, i) for i in range(k + 1)) * 0.5 ** nd)


def load_arm(p: Path) -> dict:
    """bug_id -> row；打分口径与 eval_synthetic 汇总一致：剔 env 行与 baseline_failing>0。"""
    rows = {}
    env_ids, excl_ids = [], []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if str(r.get("detail", "")).startswith("env_error:"):
            env_ids.append(r["bug_id"])
            continue
        if r.get("n_baseline_failing", 0) > 0:
            excl_ids.append(r["bug_id"])
            continue
        rows[r["bug_id"]] = r
    return {"rows": rows, "env": env_ids, "excl": excl_ids}


def main() -> None:
    d = Path(sys.argv[1])
    sft = load_arm(d / "syneval_blind100_sft.jsonl")
    grpo = load_arm(d / "syneval_blind100_grpo.jsonl")

    # golden 效度门（预注册 §2：SFT 臂 golden 必须全 1.0；<90 → 作废）
    golden = [r.get("golden_reward") for r in sft["rows"].values() if r.get("golden_reward") is not None]
    g_pass = sum(1 for g in golden if g >= 0.999)
    if golden and g_pass == len(golden):
        gate = "✅ 有效"
    elif g_pass >= 90:
        gate = "⚠️ 部分失败→成对剔除"
    else:
        gate = "❌ <90 本轮作废"
    print(f"== golden 效度门（SFT 臂）: {g_pass}/{len(golden)} ==  {gate}")
    bad_golden = {b for b, r in sft["rows"].items()
                  if r.get("golden_reward") is not None and r["golden_reward"] < 0.999}

    # 成对总体：两臂都 graded 且 golden 非失败
    common = (set(sft["rows"]) & set(grpo["rows"])) - bad_golden
    n = len(common)
    p1 = {b for b in common if sft["rows"][b]["reward"] >= 0.999}   # SFT 通过
    p2 = {b for b in common if grpo["rows"][b]["reward"] >= 0.999}  # GRPO 通过
    k1, k2 = len(p1), len(p2)
    lo1, hi1 = wilson(k1, n)
    lo2, hi2 = wilson(k2, n)
    print(f"\n== 总 pass@1（成对 n={n}，reward==1.0 口径）==")
    print(f"  SFT : {k1}/{n} = {k1/n:.3f}  Wilson95% [{lo1:.3f},{hi1:.3f}]")
    print(f"  GRPO: {k2}/{n} = {k2/n:.3f}  Wilson95% [{lo2:.3f},{hi2:.3f}]")
    if sft["env"] or grpo["env"] or sft["excl"] or grpo["excl"]:
        print(f"  （剔除：env sft={sft['env']} grpo={grpo['env']}；baseline_excl sft={sft['excl']} grpo={grpo['excl']}）")

    b01 = len(p1 - p2)   # SFT 对 GRPO 错（退步）
    b10 = len(p2 - p1)   # SFT 错 GRPO 对（进步）
    p_ex = mcnemar_exact(b01, b10)
    diff = (k2 - k1) / n
    nd = b01 + b10
    se = math.sqrt((nd - (b10 - b01) ** 2 / n)) / n if nd else 0.0
    print(f"\n== 成对检验 ==\n  b01(退)={b01} b10(进)={b10} π_d={nd/n:.3f}")
    print(f"  McNemar exact p={p_ex:.4f}   配对差={diff*100:+.1f}pp  95%CI=[{(diff-1.96*se)*100:+.1f},{(diff+1.96*se)*100:+.1f}]pp")

    # 按算子净增益归因 + 集中度（预注册 §4：单算子 >50% 净增益 → 不给 GO）
    net_by_op = Counter()
    for b in (p2 - p1):
        net_by_op[grpo["rows"][b].get("mutation")] += 1
    for b in (p1 - p2):
        net_by_op[grpo["rows"][b].get("mutation")] -= 1
    net_total = b10 - b01
    print(f"\n== 净增益按算子归因（net={net_total:+d}）==")
    for op, v in net_by_op.most_common():
        share = v / net_total if net_total else float("nan")
        print(f"  {str(op):<22} {v:+d}  占净增益 {share*100:.0f}%")
    _, top_v = (net_by_op.most_common(1)[0] if net_by_op else (None, 0))
    concentration = (top_v / net_total) if net_total > 0 else 0.0

    # 按项目分解
    proj = defaultdict(lambda: [0, 0, 0])   # [n, sft_pass, grpo_pass]
    for b in common:
        pj = grpo["rows"][b].get("project")
        proj[pj][0] += 1
        proj[pj][1] += b in p1
        proj[pj][2] += b in p2
    print("\n== 按项目 ==")
    for pj, (cnt, s, g) in sorted(proj.items(), key=lambda kv: -kv[1][0]):
        print(f"  {str(pj):<10} n={cnt:<3} SFT {s}/{cnt}  GRPO {g}/{cnt}  ({(g-s)/cnt*100:+.0f}pp)")

    # stage 分布
    print("\n== stage 分布 ==")
    for name, arm in (("SFT", sft), ("GRPO", grpo)):
        st = Counter(r["stage"] for b, r in arm["rows"].items() if b in common)
        print(f"  {name:<5} {dict(st.most_common())}")

    # 预注册判定（§4 机械执行）
    print("\n== 预注册判定（docs/prereg_blind100_D1.md §4）==")
    if not golden or g_pass < 90:
        verdict = "INVALID（golden 效度门未过，本轮作废）"
    elif k1 < 5:
        verdict = "DEFER（难度哨兵：SFT <5/100，盲条件落在全失败区，比较失去分辨力）"
    elif diff >= 0.10 and p_ex < 0.1 and concentration <= 0.5:
        verdict = "GO（≥+10pp & McNemar p<0.1 & 净增益不集中单算子 → 扩池立项）"
    elif diff >= 0.10 and p_ex < 0.1 and concentration > 0.5:
        verdict = "GRAY→NO-GO（幅度达标但净增益集中单一算子，预注册不给 GO；按灰区默认处理）"
    elif diff < 0.05:
        verdict = "NO-GO（<+5pp → 池内涨幅≈背题，同语料扩池无益；GPU 转 D2 verl）"
    else:
        verdict = "GRAY（+5~<10pp 或 p≥0.1 → 默认 NO-GO，D2 后用 56 条件冻结评测集复议）"
    print(f"  差={diff*100:+.1f}pp  p={p_ex:.4f}  集中度={concentration*100:.0f}%  SFT={k1}/{n}")
    print(f"  >>> 判定：{verdict}")


if __name__ == "__main__":
    main()
