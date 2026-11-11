#!/usr/bin/env python3
"""summarize_dapo_run.py — GRPO vs DAPO 同口径训练曲线 + 逐题评测对比（简报#2 的数据来源）。

只用本地 evidence 文件（无 SSH 依赖）：
  - 07_grpo_full/logging_v0_firstrun.jsonl + logging_v1_resume.jsonl（按 step 去重拼接）
  - 08_dapo_full/logging_v0.jsonl
  - 01/02/07/08 各 syneval_*.report.json 的 per_bug
通用做法：不猜 key 名——对 logging.jsonl 里所有数值型 key 都算 首10步/末10步 均值，
两臂共同 key 并排打印；reward_std==0 的步数单独统计（dynamic sampling 实效的核心证据）。
"""
import json
from pathlib import Path

EV = Path(__file__).resolve().parents[1] / "docs" / "evidence"


def load_jsonl(p: Path):
    rows = []
    for line in p.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def step_key(row):
    for k in row:
        if "step" in k.lower() and isinstance(row[k], (int, float)):
            return k
    return None


def merge_grpo():
    """v0(1-113) + v1(81-150) 按 step 去重（重叠段以 v1 为准），无 step 键则直接拼接。"""
    v0 = load_jsonl(EV / "07_grpo_full" / "logging_v0_firstrun.jsonl")
    v1 = load_jsonl(EV / "07_grpo_full" / "logging_v1_resume.jsonl")
    if v0 and step_key(v0[0]):
        sk = step_key(v0[0])
        by_step = {}
        for r in v0 + v1:
            if sk in r:
                by_step[r[sk]] = r
        return [by_step[s] for s in sorted(by_step)]
    return v0 + v1


def numeric_stats(rows):
    keys = set()
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                keys.add(k)
    out = {}
    n = len(rows)
    w = min(10, max(1, n // 2))
    first, last = rows[:w], rows[-w:]
    for k in sorted(keys):
        def avg(rs):
            vals = [r[k] for r in rs if isinstance(r.get(k), (int, float))]
            return sum(vals) / len(vals) if vals else None
        out[k] = (avg(first), avg(last), n)
    return out


def zero_std_count(rows):
    """reward_std==0（全对/全错组，零优势信号）出现的步数与最后一次出现的 step。"""
    sk = step_key(rows[0]) if rows else None
    cnt, last_step = 0, None
    for r in rows:
        v = r.get("reward_std")
        if v is None:
            for k in r:
                if k.endswith("reward_std"):
                    v = r[k]
                    break
        if isinstance(v, (int, float)) and v == 0:
            cnt += 1
            last_step = r.get(sk) if sk else None
    return cnt, last_step


def fmt(x):
    if x is None:
        return "—"
    return f"{x:.4f}" if abs(x) < 100 else f"{x:.1f}"


def main():
    grpo = merge_grpo()
    dapo = load_jsonl(EV / "08_dapo_full" / "logging_v0.jsonl")
    print(f"GRPO 有效步数: {len(grpo)}   DAPO 有效步数: {len(dapo)}")

    gs, ds = numeric_stats(grpo), numeric_stats(dapo)
    common = sorted(set(gs) & set(ds))
    print(f"\n=== 数值 key 首10步 vs 末10步 均值（两臂共同 key，共 {len(common)} 个）===")
    print(f"{'key':<38} {'GRPO首':>9} {'GRPO末':>9} {'DAPO首':>9} {'DAPO末':>9}")
    for k in common:
        g = gs[k]
        d = ds[k]
        print(f"{k:<38} {fmt(g[0]):>9} {fmt(g[1]):>9} {fmt(d[0]):>9} {fmt(d[1]):>9}")
    only_g = sorted(set(gs) - set(ds))
    only_d = sorted(set(ds) - set(gs))
    if only_g:
        print(f"\n仅 GRPO 有的 key: {only_g}")
    if only_d:
        print(f"仅 DAPO 有的 key: {only_d}")

    gc, gl = zero_std_count(grpo)
    dc, dl = zero_std_count(dapo)
    print(f"\n=== reward_std==0 步数（零信号组）===")
    print(f"GRPO: {gc}/{len(grpo)} 步, 最后一次 step={gl}")
    print(f"DAPO: {dc}/{len(dapo)} 步, 最后一次 step={dl}")

    print(f"\n=== 合成 held-out 逐题对比（pass=score>=1.0）===")
    reports = {
        "base": EV / "01_baseline" / "syneval_base.report.json",
        "sft": EV / "02_sft" / "syneval_sft.report.json",
        "grpo": EV / "07_grpo_full" / "syneval_grpo.report.json",
        "dapo": EV / "08_dapo_full" / "syneval_dapo.report.json",
    }
    passes = {}
    for name, p in reports.items():
        if not p.exists():
            print(f"{name}: 报告缺失 {p}")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        pb = r.get("per_bug", {})
        passes[name] = {b for b, s in pb.items() if isinstance(s, (int, float)) and s >= 1.0}
        print(f"{name}: {len(passes[name])}/{r.get('n', '?')} pass@1={r.get('pass_at_1')}")
    if "sft" in passes and "grpo" in passes and "dapo" in passes:
        print(f"\nDAPO vs GRPO 新增做对: {sorted(passes['dapo'] - passes['grpo'])}")
        print(f"DAPO vs GRPO 丢掉:     {sorted(passes['grpo'] - passes['dapo'])}")
        print(f"DAPO vs SFT  新增做对: {sorted(passes['dapo'] - passes['sft'])}")
        print(f"DAPO vs SFT  丢掉:     {sorted(passes['sft'] - passes['dapo'])}")
        print(f"GRPO 丢的题 DAPO 是否找回: {sorted(passes['sft'] - passes['grpo'])} -> "
              f"找回 {sorted((passes['sft'] - passes['grpo']) & passes['dapo'])}")

    rt = EV / "08_dapo_full" / "local_dapo_merged_test.report.json"
    if rt.exists():
        r = json.loads(rt.read_text(encoding="utf-8"))
        print(f"\n=== 真实 test 27 题 ===\nfixed={r.get('fixed')}/{r.get('n')} "
              f"fix_rate={r.get('fix_rate')} stage={r.get('stage_distribution')}")


if __name__ == "__main__":
    main()
