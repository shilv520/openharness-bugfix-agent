#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stage_frozen56.py — R2 冻结评测集：14 val bug × 4 合成变体 = 56 条件（Task 6 终评基建）

是什么：复用 sft/build_sft_data.py 的单条流水线 attempt_one（fixed 世界副本 → inject_bug
→ test -r 找敏感失败测试 → golden 审计 → eval_rec），为 val split 的 14 个 bug 每个生成
PER_BUG=4 条全新合成条件（seed_k 从 9000000 起、逐条 golden 审计 audit=True）。产物三件：
frozen56_staging.jsonl（append-only，可断点续跑）+ syn_eval_frozen56.jsonl（终评输入，
eval_synthetic.py --eval-json 直接用）+ frozen56_report.json（达成表+断言清单+sha256）。
**绝不**追加/修改已冻结的 syn_staging_*.jsonl / syn_eval*.jsonl（运行前记 sha256、
结束复验，被改动 → FROZEN56_FAIL）。

为什么：① 21 条 clean 评测只覆盖 14 个 val bug 中的 7 个、每题值 4.8pp，统计上不可判定
（坑 #30）——56 条件全覆盖，作终评大表三臂（SFT/GRPO/DAPO）的统一尺子；② blind100 已
裁决 NO-GO（扩池不立项，prereg §9），本集定位从「扩池前置门禁」转为「终评基建」；
③ val 14 bug 被 R4 护栏（build_grpo_data.py 黑名单）永久排除出训练池 → 56 条件天然
与训练隔离，建成即冻结。

怎么做：import build_sft_data 复用机械函数（attempt_one / enumerate_candidates /
_detect_resources），自写「每 bug 定额」滚动提交循环（原 phase_synthetic 是全局 quota，
不保证逐 bug 覆盖）。三重去重防线：
  ① run 内 (src_bug, file, line_no, mutation) 同键不收第二遍；
  ② 对既有全部合成条件（staging_val/staging_train/syn_eval/syn_eval_clean/blind_eval）
     同键去重——撞上改判 dead_duplicate；
  ③ seed_k 全局唯一：收集既有所有 seed_k（实证有冲突源：旧 val 流 seed_k=9000069 恰落
     9000xxx 段），分配器撞位跳号。bug_id = SYN-<bug>-<seed_k> 因此全局唯一。
确定性：候选文件洗牌 / pass 轮转全部由 SEED_BASE=9000000 驱动；断点续跑吃 staging 落盘
事实（kept 计数 + 已烧 seed 集合），续跑不重复不遗漏。
纪律：纯 CPU 不碰 GPU；无关机/定时逻辑；资源闸门同 build_sft_data（坑 #22）；
断言 A2-A8 任一失败 → 拒绝产出最终 jsonl，FROZEN56_FAIL rc=1（loud-fail）。

用法（实例，nohup，纯 CPU ~1.5-2.5h）：
  cd /root/rl_training/rl && nohup /root/miniconda3/bin/python3 stage_frozen56.py \
      > /root/frozen56.log 2>&1 &
  预检（不跑 java，秒级）：--dry-run
标记：FROZEN56_DONE / FROZEN56_DONE_WITH_WARNINGS / FROZEN56_FAIL（断言拒产出）/
FROZEN56_FAIL_UNCAUGHT（未捕获崩溃；grep FROZEN56_FAIL 前缀同样命中）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

HERE = Path(__file__).resolve().parent            # rl_training/rl
REPO = HERE.parent                                # rl_training
for _p in (str(REPO), str(REPO / "sft"), str(REPO / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import build_sft_data as bsd  # noqa: E402  复用 attempt_one/enumerate_candidates/_detect_resources

SFT_DATA = Path("/root/autodl-tmp/sft_data")
SEED_BASE = 9000000           # frozen56 专用 seed_k 段起点（9000xxx）
PER_BUG = 4

# 冻结证据（只读；sha256 前记后验）：任何被改动 = 管线洞，立即 FAIL
FROZEN_FILES = (
    SFT_DATA / "syn_staging_val.jsonl",
    SFT_DATA / "syn_staging_train.jsonl",
    SFT_DATA / "syn_eval.jsonl",
    SFT_DATA / "syn_eval_clean.jsonl",
    Path("/root/blind100/blind_eval.jsonl"),
)
# 既有条件来源（去重防线②③的取数面）。
# 红线：本 run 自己的 staging 绝不能进 ex_* 比对面——否则终检 A2/A3/A4 拿 kept_rows 与
# 「含自己的快照」求交，任何断点续跑必然自撞 FROZEN56_FAIL（对抗审查实机复现证实）。
# 自我去重另有一套：seen_keys / burned_seeds / 循环内 demote，与 ex_* 无关。
# 注意不在这里删掉 staging 项：--out-dir 非默认时，默认位置的 frozen56_staging.jsonl 是
# 「他人产物」，理应参与去重——调用点按 stage_file 精确排除的才是本 run 自己的。
EXISTING_SOURCES = FROZEN_FILES + (SFT_DATA / "frozen56_staging.jsonl",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "MISSING"


def _rows(path: Path) -> list:
    """逐行 JSON；坏行跳过但必须吵闹（静默吞行 = 冻结集悄悄少一条，违反 loud-fail）。"""
    if not path.exists():
        return []
    out, n_bad = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                n_bad += 1
    if n_bad:
        print(f"[frozen56] [WARN] {path.name} 有 {n_bad} 行无法解析（截断/融合行？），已跳过", flush=True)
    return out


def _src_of(r: dict):
    """任意来源行 → 真实 src bug（staging 行有 src_bug；eval 行从 bug_id/real_bug_id 反解）。"""
    if r.get("src_bug"):
        return r["src_bug"]
    if r.get("real_bug_id"):
        return r["real_bug_id"]
    bid = r.get("bug_id") or ""
    parts = bid.split("-")
    if bid.startswith("SYN-") and len(parts) >= 4:
        return f"{parts[1]}-{parts[2]}"
    return None


def collect_existing(sources) -> tuple:
    """(seed_k 集合, 变异位置键集合, bug_id 集合, 逐源计数)——覆盖调用方给的取数面。
    键 = (src_bug, file, line_no, mutation)；只有 kept/eval 行有完整键，dead 行只贡献 seed。
    sources 由调用点传入且必须已排除本 run 自己的 staging（见 EXISTING_SOURCES 红线注释）。"""
    seeds, keys, bids, counts = set(), set(), set(), {}
    for path in sources:
        rows = _rows(path)
        counts[path.name] = len(rows)
        for r in rows:
            if isinstance(r.get("seed_k"), int):
                seeds.add(r["seed_k"])
            bid = r.get("bug_id")
            src = _src_of(r)
            if bid and src:
                bids.add(bid)
            was_kept = r.get("status") == "kept" or bid  # eval 行没有 status，本身就是条件
            if was_kept and src and r.get("line_no") is not None and r.get("mutation"):
                keys.add((src, r.get("file") or r.get("file_rel"), r.get("line_no"), r["mutation"]))
            ev = r.get("eval_rec") or {}
            if ev.get("line_no") is not None and ev.get("mutation") and ev.get("bug_id"):
                bids.add(ev["bug_id"])
                if isinstance(ev.get("seed_k"), int):
                    seeds.add(ev["seed_k"])
                keys.add((_src_of({**r, **ev}) or src, ev.get("file_rel"), ev["line_no"], ev["mutation"]))
    return seeds, keys, bids, counts


class SeedAllocator:
    """从 base 起顺序发号，撞上既有/已烧 seed 跳位；发出的号即时登记（防 run 内重复）。"""

    def __init__(self, base: int, taken: set):
        self.k = base
        self.taken = taken
        self.skipped = 0

    def next(self) -> int:
        while self.k in self.taken:
            self.k += 1
            self.skipped += 1
        s = self.k
        self.k += 1
        self.taken.add(s)
        return s


def build_stream(val_recs: list, per_bug_files: dict, max_passes: int) -> list:
    """(rec, file) 尝试流：pass 间洗牌轮转 bug（定额在 bug 间均匀铺开），
    文件按 (pass+序) 轮转（同 bug 不同尝试尽量换文件 → 变异点多样）。"""
    active = [r for r in val_recs if per_bug_files.get(r["bug_id"])]
    stream = []
    for p in range(max_passes):
        order = list(active)
        random.Random(SEED_BASE + 7919 * p).shuffle(order)
        for i, rec in enumerate(order):
            fs = per_bug_files[rec["bug_id"]]
            stream.append((rec, fs[(p + i) % len(fs)]))
    return stream


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # nohup 下标记永不因编码丢失
    ap = argparse.ArgumentParser(description="R2 冻结评测集生成（14 val bug × 4）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--per-bug", type=int, default=PER_BUG)
    ap.add_argument("--max-passes", type=int, default=200, help="轮转上限（防死循环；14 bug × 200 = 2800 尝试）")
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--max-code-lines", type=int, default=300)
    ap.add_argument("--min-code-lines", type=int, default=20)
    ap.add_argument("--split-json", default=str(REPO / "data" / "split.json"))
    ap.add_argument("--out-dir", default=str(SFT_DATA))
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的 syn_eval_frozen56.jsonl（默认拒绝，R4 纪律）")
    ap.add_argument("--dry-run", action="store_true", help="只建流+去重预检+断言，不跑 java")
    cfg = ap.parse_args()
    t_start = time.time()

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage_file = out_dir / "frozen56_staging.jsonl"
    final_file = out_dir / "syn_eval_frozen56.jsonl"
    report_file = out_dir / "frozen56_report.json"
    if final_file.exists() and not cfg.force and not cfg.dry_run:
        print(f"FROZEN56_FAIL 拒绝覆盖已存在的 {final_file}（冻结纪律；确要重建加 --force 并说明理由）")
        return 1

    # ---- 半行防护（续跑前置）----
    # 上一轮若被 kill，staging 末行可能截断；直接 append 会把半行与本 run 第一条新行「融合」
    # 成一条坏行、两条一起丢（其中可能有 kept 行 → A1 静默欠额且不可恢复）。回退到最后完整行。
    if stage_file.exists() and stage_file.stat().st_size > 0:
        with open(stage_file, "rb+") as g:
            g.seek(-1, os.SEEK_END)
            if g.read(1) != b"\n":
                g.seek(0)
                data = g.read()
                cut = data.rfind(b"\n") + 1
                print(f"[frozen56] [WARN] staging 末行截断，回退到 {cut} 字节"
                      f"（丢弃 {len(data) - cut} 字节半行）", flush=True)
                g.seek(cut)
                g.truncate()

    # ---- 冻结证据 sha256 前记 ----
    sha_before = {str(p): sha256(p) for p in FROZEN_FILES}
    print(f"[frozen56] 冻结证据 sha256 已记录（{len(FROZEN_FILES)} 件）", flush=True)

    # ---- 资源闸门（坑 #22 同款）----
    cores, mem_gb = bsd._detect_resources()
    print(f"[frozen56] 有效 CPU={cores:.1f} 核 / 内存={mem_gb:.1f}GB", flush=True)
    if not cfg.dry_run:
        if cores < 2 or mem_gb < 3:
            print("FROZEN56_FAIL 资源不足（无卡模式跑不了编译+测试，同 build_sft_data 闸门）")
            return 1
        cfg.workers = max(1, min(cfg.workers, int(cores), int(mem_gb / 1.5)))

    # ---- split 纪律自检 ----
    data = json.loads(Path(cfg.split_json).read_text(encoding="utf-8"))
    val_recs = data["val"]
    val_ids = [r["bug_id"] for r in val_recs]
    test_ids = {r["bug_id"] for r in data.get("test", [])}
    train_ids = {r["bug_id"] for r in data["train"]}
    assert len(val_ids) == 14, f"val split 应为 14 bug，实际 {len(val_ids)}"
    assert not (set(val_ids) & test_ids), "val/test 交集非空——split 损坏"
    assert not (set(val_ids) & train_ids), "val/train 交集非空——split 损坏"

    # ---- fixed 世界基线（复用既有测量，全部 green 才继续）----
    bl_file = out_dir / "fixed_baseline.json"
    if not bl_file.exists():
        print(f"FROZEN56_FAIL 缺 {bl_file}（应先跑过 build_sft_data 的 baseline phase）")
        return 1
    baselines = json.loads(bl_file.read_text(encoding="utf-8"))
    # 只放行「全绿且失败列表为空」：eval_synthetic.py L183-184 对 baseline_failing 非空的记录
    # 加载即剔除——冻结件若混进这类条件，终评当天会静默少于 56 条而本脚本照样报成功。
    # 评测集等级不给 build_sft_data「做差消噪」留余地（今天 14/14 全 green，此闸为焊死不变量）。
    bad_bl = {b: baselines.get(b, {}).get("status") for b in val_ids
              if baselines.get(b, {}).get("status") != "green"
              or baselines.get(b, {}).get("baseline")}
    if bad_bl:
        print(f"FROZEN56_FAIL val bug 基线不可用: {bad_bl}")
        return 1
    missing_ws = [b for b in val_ids if not (Path(bsd.CACHE_FIXED) / b).exists()]
    if missing_ws:
        print(f"FROZEN56_FAIL d4j_cache_fixed 缺 fixed 世界: {missing_ws}")
        return 1

    # ---- 断点续跑状态：吃 staging 落盘事实（先于 ex_* 面读取，供自指防护断言用）----
    done_rows = _rows(stage_file)
    kept = [r for r in done_rows if r.get("status") == "kept"]
    kept_per_bug = Counter(r["src_bug"] for r in kept)
    seen_keys = {(r["src_bug"], r.get("file"), r.get("line_no"), r.get("mutation")) for r in kept}
    stats = Counter(r["status"] for r in done_rows)
    burned_seeds = {r["seed_k"] for r in done_rows if isinstance(r.get("seed_k"), int)}

    # ---- 既有条件收集（去重防线②③取数面；排除本 run 自己的 staging，防自指）----
    ex_sources = [p for p in EXISTING_SOURCES if p.resolve() != stage_file.resolve()]
    assert all(p.resolve() != stage_file.resolve() for p in ex_sources), \
        "本 run 的 staging 混进了比对面——续跑会自撞 A2/A3/A4"
    ex_seeds, ex_keys, ex_bids, ex_counts = collect_existing(ex_sources)
    # 自指防护（--dry-run 也走到这里 → 秒级预检即可抓住这类回归）：
    assert not (ex_seeds & burned_seeds), (
        f"collect_existing 吃到了自己的 staging —— ex_* 面自指污染: "
        f"{sorted(ex_seeds & burned_seeds)[:5]}")
    print(f"[frozen56] 既有条件: seeds={len(ex_seeds)} keys={len(ex_keys)} bug_ids={len(ex_bids)} 逐源={ex_counts}", flush=True)

    seeds = SeedAllocator(SEED_BASE, ex_seeds | burned_seeds)
    if done_rows:
        print(f"[frozen56] 续跑：staging 已有 {len(done_rows)} 行（kept {len(kept)}），"
              f"per_bug={dict(kept_per_bug)}", flush=True)

    # ---- 候选文件（洗牌种子与旧 run 错开 → 文件顺序不同，撞旧条件概率低）----
    per_bug_files = {}
    for i, rec in enumerate(val_recs):
        per_bug_files[rec["bug_id"]] = bsd.enumerate_candidates(
            rec, cfg.max_code_lines, cfg.min_code_lines, SEED_BASE + i)
    n_files = {b: len(v) for b, v in per_bug_files.items()}
    print(f"[frozen56] 候选文件/bug: {n_files}（总 {sum(n_files.values())}）", flush=True)
    assert all(v > 0 for v in n_files.values()), f"有 bug 无可注入文件: {n_files}"

    target_total = len(val_ids) * cfg.per_bug

    def n_capped():
        return sum(min(kept_per_bug[b], cfg.per_bug) for b in val_ids)

    if cfg.dry_run:
        stream = build_stream(val_recs, per_bug_files, 3)   # 预检只看流头
        head = [(r["bug_id"], f["rel"], f["tier"], seeds.next()) for r, f in stream[:20]]
        print("[frozen56][dry-run] 流头 20 条 (bug, file, tier, seed_k):")
        for h in head:
            print("   ", h)
        print(f"[frozen56][dry-run] 既有 seed 冲突跳号演示: allocator.taken 含 {len(seeds.taken)} 个")
        print(f"[frozen56][dry-run] sha_before={ {k: v[:12] for k, v in sha_before.items()} }")
        print("FROZEN56_DRYRUN_OK")
        return 0

    # ---- 生成主循环：滚动提交 + 每 bug 定额 + 三重去重 ----
    if n_capped() >= target_total:
        print(f"[frozen56] staging 已达成 {n_capped()}/{target_total}，跳过生成", flush=True)
    else:
        stream = build_stream(val_recs, per_bug_files, cfg.max_passes)
        it = iter(stream)
        fh = open(stage_file, "a", encoding="utf-8")
        t0, n_att = time.time(), len(done_rows)
        capped0 = n_capped()   # 续跑时已达成数——ETA 速率只用本 run 增量，否则续跑速率虚高
        try:
            with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
                futs = set()

                def submit_next() -> bool:
                    for rec, f in it:
                        if kept_per_bug[rec["bug_id"]] >= cfg.per_bug:
                            continue                     # 该 bug 已满额，烧号跳过
                        a = {"rec": rec, "file_rel": f["rel"], "tier": f.get("tier", 0),
                             "seed_k": seeds.next(),
                             "baseline": baselines[rec["bug_id"]]["baseline"],
                             "audit": True,               # 评测集等级：逐条 golden 审计
                             "timeout_s": cfg.timeout_s, "split_tag": "val"}
                        futs.add(pool.submit(bsd.attempt_one, a))
                        return True
                    return False

                for _ in range(cfg.workers * 2):
                    if not submit_next():
                        break
                while futs:
                    done_f, futs = wait(futs, return_when=FIRST_COMPLETED)
                    for fut in done_f:
                        r = fut.result()
                        n_att += 1
                        if r.get("status") == "kept":
                            key = (r["src_bug"], r.get("file"), r.get("line_no"), r.get("mutation"))
                            if key in seen_keys or key in ex_keys or kept_per_bug[r["src_bug"]] >= cfg.per_bug:
                                r = {**r, "status": "dead_duplicate"}   # 防线①②/超额落袋
                            else:
                                seen_keys.add(key)
                                kept_per_bug[r["src_bug"]] += 1
                        stats[r["status"]] += 1
                        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                        fh.flush()
                    if n_att % 10 == 0 or n_capped() >= target_total:
                        # 每尝试保留率 × 剩余条数 = 剩余尝试数；再乘「本 run 每尝试分钟数」。
                        # （旧版丢了 ÷att_run，ETA 虚高百倍——nohup 下这是操作者唯一的去留信号；
                        #   本 run 尚无 kept 时 rate=0，老实打 -- 而不是 1e-9 下限的天文数字）
                        kept_run = n_capped() - capped0
                        if kept_run > 0:
                            # = 剩余 kept 数 × 本 run 平均每 kept 分钟数
                            eta = f"{(target_total - n_capped()) * (time.time() - t0) / 60 / kept_run:.0f}min"
                        else:
                            eta = "--(尚无kept)"
                        print(f"  [frozen56] attempts={n_att} kept={n_capped()}/{target_total} "
                              f"per_bug={dict(kept_per_bug)} dead={dict(stats)} eta≈{eta}", flush=True)
                    if n_capped() >= target_total:
                        for fut in futs:
                            fut.cancel()
                        break
                    while len(futs) < cfg.workers * 2:
                        if not submit_next():
                            break
        finally:
            fh.close()
        print(f"[frozen56] 生成结束 kept={n_capped()}/{target_total} attempts={n_att} "
              f"分布={dict(stats)} 耗时 {(time.time() - t0) / 60:.1f}min", flush=True)

    # ---- 终检断言（从磁盘重读 = 只认落盘事实）----
    rows = _rows(stage_file)
    kept_rows = [r for r in rows if r.get("status") == "kept"]
    per_bug_final = Counter(r["src_bug"] for r in kept_rows)
    under = {b: per_bug_final.get(b, 0) for b in val_ids if per_bug_final.get(b, 0) < cfg.per_bug}
    over = {b: c for b, c in per_bug_final.items() if c > cfg.per_bug}
    ks = [r["seed_k"] for r in kept_rows]
    bids = [f"SYN-{r['src_bug']}-{r['seed_k']}" for r in kept_rows]
    mkeys = [(r["src_bug"], r.get("file"), r.get("line_no"), r.get("mutation")) for r in kept_rows]
    sha_after = {str(p): sha256(p) for p in FROZEN_FILES}

    assertions = {
        "A1_per_bug_exact": (not under and not over,
                             f"under={under} over={over}"),
        "A2_seed_band_unique_disjoint": (
            all(SEED_BASE <= k for k in ks) and len(set(ks)) == len(ks)
            and not (set(ks) & ex_seeds), f"n={len(ks)} 撞既有={sorted(set(ks) & ex_seeds)[:5]}"),
        "A3_bugid_unique_disjoint": (
            len(set(bids)) == len(bids) and not (set(bids) & ex_bids),
            f"撞既有={sorted(set(bids) & ex_bids)[:5]}"),
        "A4_mutkey_unique_disjoint": (
            len(set(mkeys)) == len(mkeys) and not (set(mkeys) & ex_keys),
            f"撞既有={sorted(set(mkeys) & ex_keys)[:3]}"),
        "A5_all_audited": (all(r.get("audit") == "pass" for r in kept_rows),
                           f"n_pass_audit={sum(1 for r in kept_rows if r.get('audit') == 'pass')}/{len(kept_rows)}"),
        "A6_eval_rec_complete": (
            all(r.get("eval_rec", {}).get("trigger_tests")
                and r.get("eval_rec", {}).get("golden_patch")
                and r.get("eval_rec", {}).get("buggy_src")
                and r["eval_rec"].get("file_rel") for r in kept_rows), "trigger/golden/buggy_src/file_rel 非空"),
        "A7_val_isolation": (set(per_bug_final) <= set(val_ids) and not (set(val_ids) & test_ids),
                             f"src_bugs={sorted(set(per_bug_final))}"),
        "A8_frozen_untouched": (sha_before == sha_after,
                                f"changed={[k for k in sha_before if sha_before[k] != sha_after[k]]}"),
        # A9：冻结集与消费者口径对齐——eval_synthetic 加载即剔除 baseline_failing 非空记录，
        # 这类条件若被冻进来，终评当天尺子会静默变短（入口闸已收紧，此断言焊死第二道）。
        "A9_baseline_green": (
            all(not (r.get("eval_rec") or {}).get("baseline_failing") for r in kept_rows),
            f"n_nonempty={sum(1 for r in kept_rows if (r.get('eval_rec') or {}).get('baseline_failing'))}"
            " —— eval_synthetic 会加载即剔除，冻结集将静默少于目标条数"),
    }
    hard_fail = {k: v[1] for k, v in assertions.items() if not v[0] and k != "A1_per_bug_exact"}
    a1_ok = assertions["A1_per_bug_exact"][0]

    # ---- 定额确定性截取：跨 cfg 续跑（先前 per-bug 更大等）时 kept_rows 可能超额，
    # 最终产物恒取每 bug 按 seed_k 升序的前 per_bug 条 → 条数 == per_bug × 覆盖 bug 数。
    sel, dropped_over = [], {}
    for b in val_ids:
        rb = sorted((r for r in kept_rows if r["src_bug"] == b), key=lambda r: r["seed_k"])
        sel.extend(rb[: cfg.per_bug])
        if len(rb) > cfg.per_bug:
            dropped_over[b] = len(rb) - cfg.per_bug
    eval_recs = [r["eval_rec"] for r in sorted(sel, key=lambda r: (r["src_bug"], r["seed_k"]))]

    report = {
        "config": {k: v for k, v in vars(cfg).items()},
        "seed_base": SEED_BASE, "per_bug": cfg.per_bug, "target_total": target_total,
        "kept_total": len(kept_rows), "per_bug_kept": dict(per_bug_final),
        "under_quota_bugs": under, "over_quota_dropped": dropped_over,
        "attempts": len(rows), "status_dist": dict(stats),
        "staging_rows_before_resume": len(done_rows), "kept_before_resume": len(kept),
        "seed_collisions_skipped": seeds.skipped,
        "existing_counts": ex_counts,
        "cfg_fingerprint": {k: vars(cfg)[k] for k in
                            ("per_bug", "max_code_lines", "min_code_lines", "split_json", "workers")},
        "assertions": {k: {"ok": v[0], "detail": v[1]} for k, v in assertions.items()},
        "sha256_before": sha_before, "sha256_after": sha_after,
        "staging_sha256": sha256(stage_file),
        "elapsed_min": round((time.time() - t_start) / 60, 1),
        "eval_usage": ("终评必须用专属 tag：--tag f56_<arm>（out → syneval_f56_<arm>.jsonl）。"
                       "禁止复用 --tag base/sft/grpo/dapo——那四个 out 文件是 Task 2.2/3/5 冻结证据；"
                       "eval_synthetic 已加 foreign-row fail-fast（SYNEVAL_FAIL）拒绝混池。示例："
                       f"python3 scripts/eval_synthetic.py --model-dir <merged> --tag f56_sft "
                       f"--eval-json {final_file}"),
    }

    failed = bool(hard_fail) or not kept_rows
    report["failed"] = failed
    if failed:
        # 失败报告分流到 .FAILED.json——绝不覆盖上一次成功 run 的 provenance 报告
        failed_report = report_file.with_name("frozen56_report.FAILED.json")
        failed_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[frozen56] [FAIL] 硬断言失败——拒绝产出最终 jsonl:", flush=True)
        for k, v in (hard_fail or {"zero_kept": "staging 无任何 kept 行"}).items():
            print(f"  - {k}: {v}", flush=True)
        if any("撞既有" in str(v) and "SYN-" in str(v) for v in hard_fail.values()):
            print("  提示: 若「撞既有」列表全是本 run 自己的 bug_id/seed（SYN-*-9000xxx），"
                  "是比对面吃到了自己 staging（自指），不是真污染——检查 ex_sources 过滤", flush=True)
        print(f"  失败报告: {failed_report}（成功 run 的 provenance 未被触碰）", flush=True)
        print("FROZEN56_FAIL", flush=True)
        return 1

    # ---- 产出最终评测集：原子写（崩溃只会留 .tmp，绝不留半截冻结集被当完整集评测）----
    tmp_file = final_file.with_suffix(".jsonl.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for ev in eval_recs:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_file, final_file)
    report["final_rows"] = len(eval_recs)
    report["final_sha256"] = sha256(final_file)
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[frozen56] syn_eval_frozen56.jsonl 写出 {len(eval_recs)} 条 → {final_file} "
          f"(sha256={report['final_sha256'][:12]})", flush=True)
    if dropped_over:
        print(f"[frozen56] [WARN] 超定额行已按 seed_k 升序截取丢弃（staging 混入其他 cfg 产物？）: "
              f"{dropped_over}", flush=True)
    if not a1_ok or dropped_over:
        print(f"[frozen56] [WARN] 未足额 bug（如实降级，终评按实际条数报告）: {under}", flush=True)
        print("FROZEN56_DONE_WITH_WARNINGS", flush=True)
    else:
        print("FROZEN56_DONE", flush=True)
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception:      # BrokenProcessPool(worker 被 OOM-kill) / OSError(ENOSPC) / KeyError...
        # 怎么死都要留标记：marker-grep 型监控靠它区分「还在跑」vs「已死」（loud-fail 红线）
        import traceback
        traceback.print_exc()
        print("FROZEN56_FAIL_UNCAUGHT", flush=True)
        rc = 1
    sys.exit(rc)
