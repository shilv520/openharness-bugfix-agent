#!/usr/bin/env python3
"""stats_recheck_d1.py — R1 口径修正 + P1/P2 声明复算（简报#2 修正的数据来源）。

复算工作流三代理的关键声明（自己算，不直接信 agent）：
  P2: GRPO「+48%」窗口问题 —— v0/v1/lineage 三种口径的 OLS 斜率 + Welch 检验
  P1: 算子重加权 held-out pass@1（需要池算子频率，见 --pool-freq）
  R1: 21 题逐题配对 McNemar + 配对 Wald CI（SFT/GRPO/DAPO 两两）
数据全在本地 evidence/，scipy 1.10.1 可用。
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from scipy import stats

EV = Path(__file__).resolve().parents[1] / "docs" / "evidence"


def load_reward_rows(p):
    rows = []
    for line in Path(p).open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r.get("reward"), (int, float)):
            # swift 把 step 记成 'global_step/max_steps': 'N/150' 字符串
            gsm = r.get("global_step/max_steps")
            step = int(str(gsm).split("/")[0]) if gsm else None
            if step is None and isinstance(r.get("global_step"), (int, float)):
                step = int(r["global_step"])
            if step is None and isinstance(r.get("epoch"), (int, float)):
                step = round(r["epoch"] * 168)  # 168 prompts, 1 prompt/step
            if step is not None:
                rows.append((step, float(r["reward"])))
    return sorted(rows)


def ols(name, pairs):
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    res = stats.linregress(xs, ys)
    print(f"  {name}: n={len(pairs)} steps {min(xs)}-{max(xs)}  slope={res.slope:+.5f}/step  "
          f"p={res.pvalue:.4f}  总升幅(slope×跨度)={res.slope*(max(xs)-min(xs)):+.3f}  r={res.rvalue:+.3f}")


def welch(name, a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    print(f"  {name}: mean {ma:.4f} vs {mb:.4f}  diff={mb-ma:+.4f}  Welch t={t:.3f} p={p:.4f}")


def trend_report(label, rows):
    ys = [y for _, y in rows]
    print(f"\n== {label} ==")
    ols("OLS 全程", rows)
    welch("首10 vs 末10", ys[:10], ys[-10:])


def mcnemar(name, p1_set, p2_set, all_bugs):
    """p1=旧模型(如 SFT) p2=新模型(如 GRPO)；b01=旧对新错, b10=旧错新对。"""
    b01 = len(p1_set - p2_set)   # 旧会新不会（退步）
    b10 = len(p2_set - p1_set)   # 旧不会新会（进步）
    n = len(all_bugs)
    nd = b01 + b10
    # exact 二项 McNemar
    if nd == 0:
        p_exact = 1.0
    else:
        k = min(b01, b10)
        p_exact = min(1.0, 2 * sum(math.comb(nd, i) for i in range(0, k + 1)) * 0.5 ** nd)
    diff = (len(p2_set) - len(p1_set)) / n
    se = math.sqrt((nd - (b10 - b01) ** 2 / n)) / n if nd else 0.0
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    print(f"  {name}: b01(退)={b01} b10(进)={b10} π_d={nd/n:.3f}  exact p={p_exact:.3f}  "
          f"配对diff={diff*100:+.1f}pp  95%CI=[{lo*100:+.1f},{hi*100:+.1f}]pp")


def per_bug_sets():
    sets = {}
    all_bugs = set()
    paths = {
        "sft": EV / "02_sft" / "syneval_sft.report.json",
        "grpo": EV / "07_grpo_full" / "syneval_grpo.report.json",
        "dapo": EV / "08_dapo_full" / "syneval_dapo.report.json",
    }
    for k, p in paths.items():
        r = json.loads(p.read_text(encoding="utf-8"))
        sets[k] = {b for b, s in r["per_bug"].items() if s >= 1.0}
        all_bugs |= set(r["per_bug"].keys())   # n = 全部评测题（21），不是通过集并集
    return sets, all_bugs


def load_eval_items(tag):
    """syneval jsonl → [{bug_id, mutation, reward}]"""
    p = EV / {"grpo": "07_grpo_full", "dapo": "08_dapo_full", "sft": "02_sft"}[tag] / f"syneval_{tag}.jsonl"
    items = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            items.append({"bug_id": r["bug_id"], "mutation": r.get("mutation"), "reward": r.get("reward", 0.0)})
    return items


def reweight(tag, pool_freq):
    """按训练池算子频率重加权 held-out pass@1；报覆盖率。"""
    items = load_eval_items(tag)
    by_op = defaultdict(list)
    for it in items:
        by_op[it["mutation"]].append(1.0 if it["reward"] >= 1.0 else 0.0)
    total_w = sum(pool_freq.values())
    covered_w = sum(pool_freq.get(op, 0) for op in by_op)
    rw = 0.0
    print(f"  [{tag}] 算子明细（eval 通过率 | 池权重）:")
    for op, vals in sorted(by_op.items()):
        w = pool_freq.get(op, 0) / total_w
        pr = sum(vals) / len(vals)
        rw += w * pr
        flag = "" if op in pool_freq else "  ⚠️ 池频率缺失"
        print(f"    {op:<22} {sum(vals):.0f}/{len(vals)}  池w={pool_freq.get(op,0)}/{total_w} ({w*100:.1f}%){flag}")
    rw_norm = rw / (covered_w / total_w) if covered_w else float("nan")
    plain = sum(1 for it in items if it["reward"] >= 1.0) / len(items)
    print(f"  [{tag}] 未加权 pass@1={plain:.4f}  重加权(归一到覆盖权重)={rw_norm:.4f}  算子覆盖率={covered_w/total_w*100:.1f}%")


def main():
    print("========== P2: GRPO reward 趋势的窗口敏感性 ==========")
    v0 = load_reward_rows(EV / "07_grpo_full" / "logging_v0_firstrun.jsonl")
    v1 = load_reward_rows(EV / "07_grpo_full" / "logging_v1_resume.jsonl")
    dapo = load_reward_rows(EV / "08_dapo_full" / "logging_v0.jsonl")
    v0d, v1d = dict(v0), dict(v1)
    lineage = sorted([(s, v0d[s]) for s in v0d if s <= 80] + [(s, v1d[s]) for s in v1d])
    trend_report("GRPO v0 首跑 (steps 1-113)", v0)
    trend_report("GRPO v1 续跑 (steps 81-150)", v1)
    trend_report("GRPO lineage (v0 1-80 + v1 81-150，最终权重的真实血统)", lineage)
    trend_report("DAPO v0 (steps 1-150)", dapo)

    print("\n========== R1: 21 题逐题配对 McNemar ==========")
    sets, all_bugs = per_bug_sets()
    mcnemar("GRPO − SFT ", sets["sft"], sets["grpo"], all_bugs)
    mcnemar("DAPO − SFT ", sets["sft"], sets["dapo"], all_bugs)
    mcnemar("DAPO − GRPO", sets["grpo"], sets["dapo"], all_bugs)

    print("\n========== P1: 算子重加权 ==========")
    pf_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if pf_path and pf_path.exists():
        pool_freq = Counter(json.loads(pf_path.read_text(encoding="utf-8")))
        reweight("grpo", pool_freq)
        reweight("dapo", pool_freq)
        reweight("sft", pool_freq)
    else:
        print("  用法: stats_recheck_d1.py <pool_op_freq.json>  （池算子频率 Counter，从实例统计后落盘）")


if __name__ == "__main__":
    main()
