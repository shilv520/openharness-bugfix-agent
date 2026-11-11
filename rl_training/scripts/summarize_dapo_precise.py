#!/usr/bin/env python3
"""summarize_dapo_precise.py — 精确口径：按文件分段统计（不拼接、不去重），
只取含 'reward' 数值键的训练行（排除 train_runtime 等 summary 行），
首10/末10 窗口 + frac_reward_zero_std 全分布 + reward_std==0 步数。
简报#2 引用数字以本脚本输出为准。
"""
import json
from collections import Counter
from pathlib import Path

EV = Path(__file__).resolve().parents[1] / "docs" / "evidence"

FILES = {
    "GRPO v0 (steps 1-113, firstrun)": EV / "07_grpo_full" / "logging_v0_firstrun.jsonl",
    "GRPO v1 (steps 81-150, resume)": EV / "07_grpo_full" / "logging_v1_resume.jsonl",
    "DAPO v0 (steps 1-150)": EV / "08_dapo_full" / "logging_v0.jsonl",
}


def load(p):
    rows = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r.get("reward"), (int, float)):
            rows.append(r)
    return rows


def avg(vals):
    return sum(vals) / len(vals) if vals else None


def seg_report(name, rows):
    n = len(rows)
    w = 10
    first, last = rows[:w], rows[-w:]
    print(f"\n### {name}  (训练行 {n})")
    for label, seg in [("首10步", first), ("末10步", last)]:
        print(f"  {label}: reward={avg([r['reward'] for r in seg]):.4f}  "
              f"reward_std={avg([r['reward_std'] for r in seg]):.4f}  "
              f"frac_zero_std={avg([r.get('frac_reward_zero_std', float('nan')) for r in seg]):.4f}  "
              f"len={avg([r.get('completions/mean_length', float('nan')) for r in seg]):.1f}  "
              f"kl={avg([r['kl'] for r in seg if isinstance(r.get('kl'), (int, float))]) if any('kl' in r for r in seg) else '无kl键'}")
    fr = [r.get("frac_reward_zero_std") for r in rows if isinstance(r.get("frac_reward_zero_std"), (int, float))]
    dist = Counter(round(v, 3) for v in fr)
    full_zero = sum(1 for v in fr if v == 1.0)
    any_zero = sum(1 for v in fr if v > 0)
    std0 = sum(1 for r in rows if r.get("reward_std") == 0)
    print(f"  全程: reward均值={avg([r['reward'] for r in rows]):.4f}  "
          f"frac_zero_std均值={avg(fr):.4f}  分布={dict(sorted(dist.items()))}")
    print(f"  含零信号组的步数={any_zero}/{n} ({any_zero/n*100:.0f}%)  全零信号步(frac==1.0)={full_zero}  reward_std==0步={std0}")
    steps = [r.get("global_step") for r in rows if isinstance(r.get("global_step"), (int, float))]
    if steps and full_zero:
        fz = [r.get("global_step") for r in rows if r.get("frac_reward_zero_std") == 1.0]
        print(f"  frac==1.0 出现在 steps: {fz}")


def main():
    for name, p in FILES.items():
        if p.exists():
            seg_report(name, load(p))
        else:
            print(f"\n### {name}: 文件缺失 {p}")


if __name__ == "__main__":
    main()
