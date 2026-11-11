#!/usr/bin/env python3
"""summarize_grpo_run.py — GRPO 全量训练收尾摘要（D0 验收简报#1 的数字来源）。

在实例上跑：/root/miniconda3/bin/python3 /root/rl_training/scripts/summarize_grpo_run.py
输出：① v1 logging.jsonl 训练曲线首尾对比（reward/std/zero-std 占比/kl/loss/长度/clip）
     ② 合成评测 per-bug diff（GRPO vs SFT，谁新对谁丢）
只读，不碰任何训练产物。
"""
import json
from pathlib import Path

LOG = "/root/autodl-tmp/grpo_out/v1-20261110-231659/logging.jsonl"
SFT = "/root/autodl-tmp/baseline_results/syneval_sft.report.json"
GRPO = "/root/autodl-tmp/baseline_results/syneval_grpo.report.json"

rows = [json.loads(l) for l in Path(LOG).read_text().splitlines() if l.strip()]
rows = [r for r in rows if "loss" in r]


def avg(rs, k):
    v = [r[k] for r in rs if k in r and isinstance(r.get(k), (int, float))]
    return round(sum(v) / len(v), 5) if v else None


first, last = rows[:10], rows[-10:]
print("n_logs", len(rows))
print("step range:", rows[0].get("global_step/max_steps"), "->", rows[-1].get("global_step/max_steps"))
for k in ["reward", "reward_std", "frac_zero_std", "kl", "loss",
          "completions/mean_length", "clip_ratio"]:
    print(f"{k:26s} first10={avg(first, k)}  last10={avg(last, k)}")

s = json.loads(Path(SFT).read_text())["per_bug"]
g = json.loads(Path(GRPO).read_text())["per_bug"]
gain = sorted(b for b in g if g[b] > s.get(b, 0))
loss = sorted(b for b in s if s[b] > g.get(b, 0))
print("GRPO 新增做对:", gain)
print("GRPO 丢掉的:  ", loss)
print(f"SFT {sum(s.values()):.0f}/21 -> GRPO {sum(g.values()):.0f}/21")
