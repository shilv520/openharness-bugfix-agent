#!/usr/bin/env python3
"""analyze_f56_arms.py — Task 6 终评：frozen56（49 条）三臂对比的独立复算。

是什么：读 evidence/11_f56_arms/ 下三臂 out jsonl + report，独立重算 pass@1/Wilson CI、
三组成对 McNemar exact + 配对差 CI、逐条翻盘清单、按项目/算子分解、stage 分布，
并与冻结集本体（10_frozen56/syn_eval_frozen56.jsonl）对账 bug_id 集合。

为什么：链上的数字是 eval_synthetic 自报的；终评大表进论文级结论前必须有一套
不经链、不经报告、直接从原始逐条记录重算的数（blind100 同款纪律）。
注意：这里不做 GO/NO-GO 判定——扩池裁决已由 blind100 预注册完成（NO-GO）；
本脚本只产出终评大表的三臂描述性数字。49 条 13 bug，Lang-42=0（覆盖度如实报告）。

怎么用：python scripts/analyze_f56_arms.py <11_f56_arms_dir> [frozen56_jsonl]
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
    """bug_id -> row；口径与 eval_synthetic 汇总一致：剔 env 行与 baseline_failing>0。"""
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


def paired(name_a: str, a: dict, name_b: str, b: dict, common: set) -> None:
    pa = {x for x in common if a["rows"][x]["reward"] >= 0.999}
    pb = {x for x in common if b["rows"][x]["reward"] >= 0.999}
    n = len(common)
    b01 = len(pa - pb)   # A 对 B 错
    b10 = len(pb - pa)   # A 错 B 对
    p_ex = mcnemar_exact(b01, b10)
    diff = (len(pb) - len(pa)) / n
    nd = b01 + b10
    se = math.sqrt((nd - (b10 - b01) ** 2 / n)) / n if nd else 0.0
    print(f"\n== {name_b} − {name_a}（成对 n={n}）==")
    print(f"  b01({name_a}对/{name_b}错)={b01}  b10({name_a}错/{name_b}对)={b10}  π_d={nd/n:.3f}")
    print(f"  McNemar exact p={p_ex:.4f}   配对差={diff*100:+.1f}pp  "
          f"95%CI=[{(diff-1.96*se)*100:+.1f},{(diff+1.96*se)*100:+.1f}]pp")
    if pa - pb:
        print(f"  仅{name_a}过: {sorted(pa - pb)}")
    if pb - pa:
        print(f"  仅{name_b}过: {sorted(pb - pa)}")


def main() -> None:
    d = Path(sys.argv[1])
    arms = {a: load_arm(d / f"syneval_f56_{a}.jsonl") for a in ("sft", "grpo", "dapo")}

    # ---- 0. report 对账（链自报 vs 逐条重算）----
    print("== 0. report 对账 ==")
    for a, arm in arms.items():
        rep = json.loads((d / f"syneval_f56_{a}.report.json").read_text(encoding="utf-8"))
        k = sum(1 for r in arm["rows"].values() if r["reward"] >= 0.999)
        n = len(arm["rows"])
        ok_n = rep["n"] == n
        ok_p = abs(rep["pass_at_1"] - (round(k / n, 4) if n else 0)) < 1e-9
        print(f"  {a:5s} model={rep['model']:<12s} date={rep['date']} eval_json={rep['eval_json']}")
        print(f"        自报 n={rep['n']} pass@1={rep['pass_at_1']} | 重算 n={n} pass@1={k/n if n else 0:.4f}"
              f" | env={arm['env']} excl={arm['excl']} | 对账{'✅' if ok_n and ok_p else '❌ 不一致！'}")
        assert ok_n and ok_p, f"{a} 臂 report 与逐条重算不一致"

    # ---- 1. golden 效度门（SFT 臂全开）----
    sft_rep = json.loads((d / "syneval_f56_sft.report.json").read_text(encoding="utf-8"))
    golden = [r.get("golden_reward") for r in arms["sft"]["rows"].values()
              if r.get("golden_reward") is not None]
    g_pass = sum(1 for g in golden if g >= 0.999)
    print(f"\n== 1. golden 效度门: {g_pass}/{len(golden)}"
          f"（report: checked={sft_rep['golden_control_checked']} failed={sft_rep['golden_control_failed']}）==")
    assert len(golden) == 49 and g_pass == 49, "golden 效度门未全过——本轮三臂数字不可信"
    print("  ✅ 49/49 全过（frozen56  worlds 上重建+判定管线正确，残余风险#10 未发生）")

    # ---- 2. 冻结集对账：三臂 bug_id 集合 == 冻结集 49 条 ----
    if len(sys.argv) > 2:
        frozen = {json.loads(l)["bug_id"] for l in
                  Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if l.strip()}
        for a, arm in arms.items():
            assert set(arm["rows"]) == frozen, f"{a} 臂条件集 != 冻结集！"
        print(f"\n== 2. 冻结集对账 ✅ 三臂 bug_id 集合与 syn_eval_frozen56.jsonl（{len(frozen)} 条）全等 ==")

    common = set(arms["sft"]["rows"]) & set(arms["grpo"]["rows"]) & set(arms["dapo"]["rows"])
    n = len(common)

    # ---- 3. 三臂总 pass@1 ----
    print(f"\n== 3. 总 pass@1（n={n}，reward==1.0 口径）==")
    passes = {}
    for a in ("sft", "grpo", "dapo"):
        k = sum(1 for x in common if arms[a]["rows"][x]["reward"] >= 0.999)
        lo, hi = wilson(k, n)
        passes[a] = {x for x in common if arms[a]["rows"][x]["reward"] >= 0.999}
        print(f"  {a.upper():5s} {k}/{n} = {k/n:.4f}  Wilson95% [{lo:.3f},{hi:.3f}]")

    # ---- 4. 三组成对检验 ----
    paired("SFT", arms["sft"], "GRPO", arms["grpo"], common)
    paired("SFT", arms["sft"], "DAPO", arms["dapo"], common)
    paired("GRPO", arms["grpo"], "DAPO", arms["dapo"], common)

    # ---- 5. 按项目 / 按算子 ----
    print("\n== 5. 按项目（pass 数/条数）==")
    proj = defaultdict(lambda: [0, 0, 0, 0])
    for b in common:
        pj = arms["sft"]["rows"][b].get("project")
        proj[pj][0] += 1
        for i, a in enumerate(("sft", "grpo", "dapo")):
            proj[pj][i + 1] += b in passes[a]
    for pj, (cnt, s, g, dp) in sorted(proj.items(), key=lambda kv: -kv[1][0]):
        print(f"  {str(pj):<8} n={cnt:<3} SFT {s}/{cnt}  GRPO {g}/{cnt}  DAPO {dp}/{cnt}")

    print("\n== 5b. 按算子 ==")
    ops = defaultdict(lambda: [0, 0, 0, 0])
    for b in common:
        op = arms["sft"]["rows"][b].get("mutation")
        ops[op][0] += 1
        for i, a in enumerate(("sft", "grpo", "dapo")):
            ops[op][i + 1] += b in passes[a]
    for op, (cnt, s, g, dp) in sorted(ops.items(), key=lambda kv: -kv[1][0]):
        print(f"  {str(op):<20} n={cnt:<3} SFT {s}/{cnt}  GRPO {g}/{cnt}  DAPO {dp}/{cnt}")

    # ---- 6. stage 分布 ----
    print("\n== 6. stage 分布 ==")
    for a in ("sft", "grpo", "dapo"):
        st = Counter(arms[a]["rows"][b]["stage"] for b in common)
        print(f"  {a.upper():5s} {dict(st.most_common())}")

    # ---- 7. 终评大表行（Task 6 直接引用）----
    k1, k2, k3 = (len(passes[a]) for a in ("sft", "grpo", "dapo"))
    print("\n== 7. 终评大表输入（frozen56 合成 held-out，49 条 / 13 val bug）==")
    print(f"  SFT  {k1}/49 = {k1/49:.1%} | GRPO {k2}/49 = {k2/49:.1%} | DAPO {k3}/49 = {k3/49:.1%}")
    print("  （对照：blind100 SFT 75% / GRPO 74%；21 题集 SFT 52.4% / GRPO 47.6% / DAPO 52.4%；")
    print("    真实 test 四臂 0/27——四把尺子分开列，禁止混排）")


if __name__ == "__main__":
    main()
