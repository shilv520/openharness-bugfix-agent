#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 2.1 — SFT 数据构建：官方 train 对 + 合成注入对 + clean 合成评测集

设计原则：每一条样本都经过真实 reward 世界验证，绝不"看起来对"就收。

三路数据来源：
  A. 官方对（~72 条）：train split 真实 bug。prompt = 真实 record + 工作区 buggy
     源码（load_buggy_code，真实路径标签，坑 #19）；golden = 官方修复 diff
     （src.patch 反向 apply 得 buggy→fixed，复用 reward_calib_linux 实测逻辑）。
     质量闸：golden 过 RewardExecutor.evaluate 必须 == 1.0 才收。
  B. 合成 SFT 对（默认 2000 条）：在 train split 的 FIXED 工作区（全绿世界）注入
     单点变异 → test -r 跑出真实失败测试 = 天然 trigger_tests（与 D4J 全分布对齐）
     → 只保留「可编译 + 有敏感失败测试」的样本；golden = 变异回滚 diff（difflib
     产出，行数天然正确）。抽样过 golden 审计：apply 阶梯打回 golden + 复跑
     test -r，敏感测试必须回绿（防 flaky / 逻辑洞混入训练集）。
     为什么用 fixed 世界而不是 buggy 世界：buggy 世界里有背景 D4J bug，回归检查
     被既有失败污染（golden 最多拿 0.3）；fixed 世界基线全绿，语义干净。
  C. clean 合成评测集（默认 50 条）：同 B 管线但语料 = val split（训练不可见），
     零预训练污染；syn_eval.jsonl 带 buggy_src，评测器可精确重建变异工作区。

纪律：test split 永不参与任何数据生成；官方 src.patch 只以 train split 身份
     进 golden（数据集①），合成对与官方 patch 无关（数据集②）。
确定性：--seed 固定则候选枚举顺序 / 注入 rng / 审计抽样全部可复现；
     断点续跑：合成对逐条 append 到 staging jsonl，重跑自动跳过已收 (bug, seed)。

用法（AutoDL，无卡模式即可，纯 CPU；预计 fixed 缓存 ~20min + 合成 2000 条 3-6h）：
  nohup /root/miniconda3/bin/python /root/rl_training/sft/build_sft_data.py \
      > /root/sft_data.log 2>&1 &
  标记: SFT_DATA_DONE / SFT_DATA_FAIL
产物（默认 /root/autodl-tmp/sft_data/）：
  sft_train.jsonl    ms-swift messages 格式（官方+合成，seed 洗牌）
  sft_audit.jsonl    逐样本元数据（与 sft_train 行序对齐：来源/变异点/敏感测试数）
  syn_eval.jsonl     clean 合成评测集（含 buggy_src/golden_patch/baseline）
  build_report.json  计数、死活分布、变异算子分布、耗时
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent            # rl_training/sft
REPO = str(HERE.parent)                           # rl_training（服务器/本地同一份代码）
sys.path.insert(0, REPO)
sys.path.insert(0, str(HERE.parent / "scripts"))

D4J = Path("/root/defects4j")
BIN = str(D4J / "framework/bin/defects4j")        # main() 里可被 --d4j-bin 覆盖（fork 继承）
CACHE_BUGGY = "/root/autodl-tmp/d4j_cache"        # Task 1.3 的 buggy 缓存（105/105）
CACHE_FIXED = "/root/autodl-tmp/d4j_cache_fixed"  # 本脚本 Phase 0 构建（全绿世界）
RUN = "/root/autodl-tmp/sft_run"
OUTDIR = Path("/root/autodl-tmp/sft_data")

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")
os.environ["PATH"] = (f"{D4J / 'framework/bin'}:{os.environ['JAVA_HOME']}/bin:"
                      + os.environ.get("PATH", ""))

from reward.reward import (BugMeta, RewardExecutor, git_apply_ladder,  # noqa: E402
                           is_test_path, parse_failing_tests)
from data.prompt_builder import build_sft_pair, fence_patch            # noqa: E402
from injector.bug_injector import inject_bug, walk_java_files          # noqa: E402


class FixedExecutor(RewardExecutor):
    """fixed 版本工作区执行器：prepare_workspace checkout {vid}f（全绿世界）。
    只借它的 checkout→EOL归一→compile→缓存回填流水线；判定仍走真实 RewardExecutor。"""
    VID_SUFFIX = "f"


# ══════════════════════════════════════════════════════════════
# 纯工具（可单测）
# ══════════════════════════════════════════════════════════════

def file_to_fqcn(rel_posix: str) -> str:
    """工作区相对路径 → 类 FQCN（prompt 的 modified_classes 展示用）。
    ⚠️ JFreeChart 在 source/ 前缀（非标准 Maven，坑 #19 同源知识）；
    Lang/Time 在 src/main/java/。都对不上时按最长已知根退化处理。"""
    for root in ("src/main/java/", "source/", "src/"):
        if rel_posix.startswith(root):
            rest = rel_posix[len(root):]
            return rest[:-5].replace("/", ".") if rest.endswith(".java") else rest
    return rel_posix[:-5].replace("/", ".") if rel_posix.endswith(".java") else rel_posix


_SKIP_SEGMENTS = ("/.git/", "/build/", "/target/", "/bin/", "/classes/", "/javadoc/")


def candidate_files(ws_root: Path, max_lines: int, min_lines: int) -> list:
    """可注入文件清单：非测试、非构建产物、严格 UTF-8、行数在 prompt 预算内。
    ⚠️ >max_lines 不收：prompt 装不下全文（与 load_buggy_code 300 行预算同口径），
    模型看不到变异点 = 教不会的样本。非 UTF-8 整文件跳过（decode-replace 会造成
    写回字节漂移，golden 就"修不回"原始文件了）。"""
    out = []
    for p in walk_java_files(ws_root):            # 已排除 /test/、/tests/、>200KB，sorted
        rel = p.relative_to(ws_root).as_posix()
        if is_test_path(rel):                     # 双保险：*Test.java 命名等
            continue
        if any(s in ("/" + rel) for s in _SKIP_SEGMENTS):
            continue
        try:
            src = p.read_bytes().decode("utf-8")  # strict
        except (OSError, UnicodeDecodeError):
            continue
        n = src.count("\n") + 1
        if n < min_lines or n > max_lines:
            continue
        out.append({"rel": rel, "lines": n})
    return out


def enumerate_candidates(rec: dict, max_lines: int, min_lines: int, seed: int) -> list:
    """单个 bug 的候选文件序列：tier1（modified_classes，relevant 测试必覆盖 →
    存活率高）在前，tier2（其余文件，靠传递依赖被 relevant 触达 → 存活率低但
    多样性好）在后；各自 seed 洗牌。文件数少于 quota 时按 pass 轮转复用
    （同文件不同 seed = 不同变异点，inject_bug 从全部可注入行随机选）。"""
    root = Path(CACHE_FIXED) / rec["bug_id"]
    if not root.exists():
        return []
    files = candidate_files(root, max_lines, min_lines)
    mod = set(rec.get("modified_classes", []))
    for f in files:
        f["tier"] = 1 if file_to_fqcn(f["rel"]) in mod else 2
    rng = random.Random(seed)
    t1 = [f for f in files if f["tier"] == 1]
    t2 = [f for f in files if f["tier"] == 2]
    rng.shuffle(t1)
    rng.shuffle(t2)
    return t1 + t2


def build_attempt_stream(recs: list, per_bug_files: dict, seed: int, max_passes: int) -> list:
    """(rec, file, seed_k) 尝试序列：pass 间轮转 bug（洗牌）→ quota 在 bug/项目间
    均匀铺开；seed_k 由全局尝试序号决定 → 同 seed 重跑得到同一条流（断点续跑基石）。"""
    active = [(rec, per_bug_files[rec["bug_id"]]) for rec in recs
              if per_bug_files.get(rec["bug_id"])]
    stream, k = [], 0
    for p in range(max_passes):
        if not active:
            break
        order = list(active)
        random.Random(seed + 7919 * p).shuffle(order)
        for rec, fs in order:
            f = fs[p % len(fs)]
            stream.append((rec, f, seed + 1000003 * k))
            k += 1
    return stream


# ══════════════════════════════════════════════════════════════
# 子进程 worker（模块级，ProcessPoolExecutor 可 pickle；Linux fork 继承全局配置）
# ══════════════════════════════════════════════════════════════

def _test_relevant(ex, ws) -> tuple:
    """一次 `test -r`（编译+relevant 套件）。返回 (status, failing)：
    status ∈ {green, failing, cannot_compile, run_error, timeout}；
    failing=None 表示拿不到可信结果（调用方必须保守丢弃）。"""
    rc, out = ex._run([str(ex.bin), "test", "-w", str(ws), "-r"])
    if rc == -99:
        return "timeout", None
    if "Cannot compile" in out:
        return "cannot_compile", None
    failing = parse_failing_tests(out)
    if failing is None:
        return "run_error", None
    return ("failing" if failing else "green"), failing


def warm_fixed(rec: dict) -> dict:
    """Phase 0：构建/校验 fixed 缓存（checkout f → EOL 归一 → compile → 回填）。"""
    bid = rec["bug_id"]
    if (Path(CACHE_FIXED) / bid).exists():
        return {"bug_id": bid, "ok": True, "cached": True}
    ws = None
    try:
        ex = FixedExecutor(BIN, CACHE_FIXED, RUN, timeout_s=2400)
        ws = ex.prepare_workspace(BugMeta.from_split_record(rec))
        return {"bug_id": bid, "ok": (Path(CACHE_FIXED) / bid).exists()}
    except Exception as e:  # noqa: BLE001
        return {"bug_id": bid, "ok": False, "error": str(e)[:200]}
    finally:
        if ws:
            shutil.rmtree(ws, ignore_errors=True)


def baseline_one(rec: dict) -> dict:
    """fixed 世界基线：理论上全绿；实测有失败也照实记录（敏感测试 = 注入后失败
    − 基线失败，做差后基线噪声自动消掉）。compile 都不过的 bug 直接排除出语料。"""
    ws = None
    try:
        ex = FixedExecutor(BIN, CACHE_FIXED, RUN, timeout_s=1800)
        ws = ex.prepare_workspace(BugMeta.from_split_record(rec))
        status, failing = _test_relevant(ex, ws)
        return {"bug_id": rec["bug_id"], "status": status, "baseline": failing or []}
    except Exception as e:  # noqa: BLE001
        return {"bug_id": rec["bug_id"], "status": "prepare_error", "baseline": [],
                "error": str(e)[:200]}
    finally:
        if ws:
            shutil.rmtree(ws, ignore_errors=True)


def attempt_one(a: dict) -> dict:
    """单次合成注入尝试（B/C 共用）。
    流程：fixed ws 副本 → 注入 → test -r → 敏感失败测试（做差基线）→
         [抽样] golden 审计 → build_sft_pair。任何一步不过 → dead_*（照实计数）。"""
    rec, t0 = a["rec"], time.time()
    res = {"src_bug": rec["bug_id"], "project": rec["project"], "file": a["file_rel"],
           "seed_k": a["seed_k"], "tier": a.get("tier", 0), "split": a["split_tag"]}
    ex = FixedExecutor(BIN, CACHE_FIXED, RUN, timeout_s=a.get("timeout_s", 900))
    ws = None
    try:
        ws = ex.prepare_workspace(BugMeta.from_split_record(rec))
        fp = ws / a["file_rel"]
        if not fp.exists():
            return {**res, "status": "file_missing"}
        try:
            src = fp.read_bytes().decode("utf-8")   # strict，与候选枚举同口径
        except UnicodeDecodeError:
            return {**res, "status": "decode_error"}
        inj = inject_bug(src, rel_path=a["file_rel"], rng=random.Random(a["seed_k"]))
        if inj is None:
            return {**res, "status": "no_inject_point"}
        res.update(mutation=inj.mutation, line_no=inj.line_no)
        fp.write_bytes(inj.buggy_src.encode("utf-8"))

        status, failing = _test_relevant(ex, ws)
        if status == "cannot_compile":
            return {**res, "status": "dead_compile"}
        if status in ("timeout", "run_error"):
            return {**res, "status": f"dead_{status}"}
        sensitive = [t for t in failing if t not in set(a["baseline"])]
        if not sensitive:
            return {**res, "status": "dead_behaviorless"}   # 无行为差异死样本（注入器文档 ⚠️ 必筛）

        # golden 审计（抽样）：阶梯 apply golden → 复跑 test -r → 敏感测试必须回绿。
        # 逻辑上必然成立（golden=精确回滚原始字节，基线已实测），审计防的是
        # flaky 测试与管线洞——审计不过宁可丢样本，不污染训练集。
        if a.get("audit"):
            pf = ws.parent / f"{ws.name}.golden.patch"
            ok, err = git_apply_ladder(ws, pf, inj.golden_patch, ex._run)
            pf.unlink(missing_ok=True)
            if not ok:
                return {**res, "status": "dead_golden_apply", "detail": err[-200:]}
            _, f2 = _test_relevant(ex, ws)
            if f2 is None or any(t in set(f2) for t in sensitive):
                return {**res, "status": "dead_golden_audit", "residual": (f2 or [])[:5]}
            res["audit"] = "pass"
        else:
            res["audit"] = "skipped"

        fqcn = file_to_fqcn(a["file_rel"])
        syn_rec = {"bug_id": f"SYN-{rec['bug_id']}-{a['seed_k']}", "project": rec["project"],
                   "vid": rec["vid"], "jira": "synthetic-injection",
                   "trigger_tests": sensitive, "modified_classes": [fqcn]}
        # prompt 的 code 段与 load_buggy_code 完全同格式（真实工作区路径标签，坑 #19）
        code = f"// File: {a['file_rel']}\n{inj.buggy_src}"
        res.update(
            status="kept", n_sensitive=len(sensitive),
            pair=build_sft_pair(syn_rec, code, inj.golden_patch),
            audit_rec={"source": "synthetic", "src_bug": rec["bug_id"],
                       "file": a["file_rel"], "tier": a.get("tier", 0),
                       "mutation": inj.mutation, "line_no": inj.line_no,
                       "original_line": inj.original_line[:160],
                       "buggy_line": inj.buggy_line[:160],
                       "n_sensitive": len(sensitive), "audit": res["audit"]},
            elapsed_s=round(time.time() - t0, 1))
        if a["split_tag"] == "val":
            # C. clean 评测集：带 buggy_src → 评测器可精确重建变异工作区
            res["eval_rec"] = {
                **syn_rec, "file_rel": a["file_rel"], "buggy_src": inj.buggy_src,
                "golden_patch": inj.golden_patch, "baseline_failing": sorted(set(a["baseline"])),
                "seed_k": a["seed_k"], "mutation": inj.mutation, "line_no": inj.line_no}
        return res
    except Exception as e:  # noqa: BLE001
        # 同 bug 并发 prepare 撞名（同毫秒）等极端情况落这里：丢样本不丢进程
        return {**res, "status": "error", "error": str(e)[:200],
                "elapsed_s": round(time.time() - t0, 1)}
    finally:
        if ws:
            shutil.rmtree(ws, ignore_errors=True)


def official_one(rec: dict) -> dict:
    """A. 官方对：golden diff 生成（复用 reward_calib_linux 实测逻辑：fresh
    checkout → EOL 归一 commit → 反向 apply src.patch → git diff）+ buggy 源码
    （load_buggy_code 与基线/SFT/评测同一口径）+ evaluate==1.0 质量闸。"""
    t0 = time.time()
    res = {"bug_id": rec["bug_id"], "project": rec["project"], "source": "official"}
    ws = None
    try:
        from reward_calib_linux import official_fix_diff
        from run_baseline_api import load_buggy_code
        diff = official_fix_diff(rec)
        ex = RewardExecutor(BIN, CACHE_BUGGY, RUN, timeout_s=900)
        bug = BugMeta.from_split_record(rec)
        ws = ex.prepare_workspace(bug)
        code, code_source = load_buggy_code(ws, rec, D4J, 300)
        gate = ex.evaluate(bug, fence_patch(diff))
        res.update(reward=gate.reward, stage=gate.stage, code_source=code_source,
                   elapsed_s=round(time.time() - t0, 1))
        if gate.reward == 1.0 and code_source == "workspace":
            res["status"] = "kept"
            res["pair"] = build_sft_pair(rec, code, diff)
            res["audit_rec"] = {"source": "official", "src_bug": rec["bug_id"],
                                "reward": gate.reward, "stage": gate.stage,
                                "code_source": code_source, "audit": "gate_1.0"}
        else:
            res["status"] = "rejected"
        return res
    except Exception as e:  # noqa: BLE001
        return {**res, "status": "error", "error": str(e)[:300],
                "elapsed_s": round(time.time() - t0, 1)}
    finally:
        if ws:
            shutil.rmtree(ws, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
# 编排
# ══════════════════════════════════════════════════════════════

def _detect_resources() -> tuple:
    """返回 (有效CPU核数, 内存GB)——必须读 cgroup，不能信 nproc/free！
    坑 #22 实测：AutoDL 无卡模式 cgroup 限死 0.5 核/2GB，但 nproc 显示宿主机
    128 核、free 显示 503G（容器看得见宿主规格）→ 16 并行 javac 把 2GB 顶满，
    Chart 全系被 OOM 误杀成"编译失败"，93% 调度周期被限流。"""
    cores = float(os.cpu_count() or 4)
    mem_gb = 64.0
    try:  # cgroup v1
        q = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read_text())
        p = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
        if q > 0:
            cores = min(cores, q / p)
    except (OSError, ValueError):
        pass
    try:
        m = int(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes").read_text())
        if 0 < m < (1 << 62):
            mem_gb = min(mem_gb, m / 2 ** 30)
    except (OSError, ValueError):
        pass
    try:  # cgroup v2
        cm = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if cm and cm[0] != "max":
            cores = min(cores, int(cm[0]) / int(cm[1]))
    except (OSError, ValueError, IndexError):
        pass
    try:
        mm = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if mm and mm != "max":
            mem_gb = min(mem_gb, int(mm) / 2 ** 30)
    except (OSError, ValueError):
        pass
    return cores, mem_gb


def _parallel_map(fn, items, workers, desc):
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(fn, it): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            out.append(r)
            tag = r.get("bug_id", "?")
            ok = r.get("ok", r.get("status", "?"))
            print(f"  [{desc} {i}/{len(futs)}] {tag:12s} -> {ok}", flush=True)
    return out


def _collect_staging() -> tuple:
    """从 staging jsonl 幂等收集三路 kept 样本（合并只认落盘事实，不认内存状态
    → 分 phase 跑 / 中途崩 / force 重跑后再 --phase merge 都得到同一结果）。
    官方对按 bug_id 去重（force 重跑时后写的行生效）。"""
    off_rows = _load_staging(OUTDIR / "official_staging.jsonl")
    off = {}
    for r in off_rows:
        if r.get("status") == "kept":
            off[r["bug_id"]] = r
    syn = [r for r in _load_staging(OUTDIR / "syn_staging_train.jsonl")
           if r.get("status") == "kept"]
    ev = [r for r in _load_staging(OUTDIR / "syn_staging_val.jsonl")
          if r.get("status") == "kept"]
    return sorted(off.values(), key=lambda r: r["bug_id"]), syn, ev


def merge_outputs(cfg) -> dict:
    """sft_train.jsonl（官方+合成，seed 洗牌）+ sft_audit.jsonl（行序对齐）+
    syn_eval.jsonl。行序 = shuffle 后的顺序，audit 与训练行一一对应。"""
    official_kept, syn_kept, eval_kept = _collect_staging()
    pairs = [r["pair"] for r in official_kept] + [r["pair"] for r in syn_kept]
    audits = ([{**r["audit_rec"], "bug_id": r["bug_id"]} for r in official_kept]
              + [{**r["audit_rec"], "bug_id": f"SYN-{r['src_bug']}-{r['seed_k']}"}
                 for r in syn_kept])
    order = list(range(len(pairs)))
    random.Random(cfg.seed).shuffle(order)
    with open(OUTDIR / "sft_train.jsonl", "w", encoding="utf-8") as fp, \
         open(OUTDIR / "sft_audit.jsonl", "w", encoding="utf-8") as fa:
        for i in order:
            fp.write(json.dumps(pairs[i], ensure_ascii=False) + "\n")
            fa.write(json.dumps({**audits[i], "idx": i}, ensure_ascii=False) + "\n")
    with open(OUTDIR / "syn_eval.jsonl", "w", encoding="utf-8") as fe:
        for r in eval_kept:
            fe.write(json.dumps(r["eval_rec"], ensure_ascii=False) + "\n")
    print(f"\n=== Task 2.1 SFT 数据汇总 ===\n"
          f"  sft_train.jsonl: {len(pairs)} 条（官方 {len(official_kept)} + 合成 {len(syn_kept)}）\n"
          f"  syn_eval.jsonl:  {len(eval_kept)} 条（val 语料，clean 评测集）\n"
          f"  目录: {OUTDIR}", flush=True)
    return {
        "official_kept": len(official_kept),
        "synthetic_train": {
            "kept": len(syn_kept), "quota": cfg.n_syn,
            "mutations": dict(Counter(r.get("mutation") for r in syn_kept)),
            "tiers": dict(Counter(r.get("tier") for r in syn_kept)),
            "projects": dict(Counter(r["project"] for r in syn_kept)),
            "audited_pass": sum(1 for r in syn_kept if r.get("audit") == "pass")},
        "synthetic_eval": {
            "kept": len(eval_kept), "quota": cfg.n_eval,
            "projects": dict(Counter(r["project"] for r in eval_kept))},
        "sft_train_total": len(pairs),
    }


def phase_fixed_cache(recs, cfg):
    print(f"[phase0] fixed 缓存（{len(recs)} bugs, workers={cfg.workers}）", flush=True)
    t0 = time.time()
    results = _parallel_map(warm_fixed, recs, cfg.workers, "fixed")
    failed = [r for r in results if not r.get("ok")]
    print(f"[phase0] 完成 ok={len(results) - len(failed)}/{len(results)} "
          f"failed={[r['bug_id'] for r in failed]} 耗时 {time.time() - t0:.0f}s", flush=True)
    return results


def phase_baseline(recs, cfg):
    bl_file = OUTDIR / "fixed_baseline.json"
    if bl_file.exists() and not cfg.force_baseline:
        print(f"[baseline] 复用 {bl_file}", flush=True)
        return json.loads(bl_file.read_text(encoding="utf-8"))
    print(f"[baseline] fixed 世界基线测量（{len(recs)} bugs）", flush=True)
    results = _parallel_map(baseline_one, recs, cfg.workers, "baseline")
    data = {r["bug_id"]: {"status": r["status"], "baseline": r["baseline"]} for r in results}
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bl_file.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    non_green = {b: v["status"] for b, v in data.items() if v["status"] != "green"}
    print(f"[baseline] 非全绿（做差后仍可用，噪声自动消）: {non_green or '无'}", flush=True)
    return data


def _load_staging(path: Path) -> list:
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def phase_synthetic(recs, baselines, per_bug_files, cfg, split_tag, quota, stage_file):
    """合成注入主循环：滚动提交 + quota 达标即停 + staging 断点续跑。"""
    done_rows = _load_staging(stage_file)
    kept_seeds = {(r["src_bug"], r["seed_k"]) for r in done_rows}
    seen_mutants = {(r["src_bug"], r["file"], r.get("line_no"), r.get("mutation"))
                    for r in done_rows if r.get("status") == "kept"}
    kept = [r for r in done_rows if r.get("status") == "kept"]
    if kept_seeds:
        print(f"[synthetic:{split_tag}] 续跑：staging 已有 {len(kept)} 条 kept", flush=True)

    stream = build_attempt_stream(recs, per_bug_files, cfg.seed, cfg.max_passes)
    it = iter(enumerate(stream))
    stats = Counter(r["status"] for r in done_rows)
    pairs_fh = open(stage_file, "a", encoding="utf-8")

    def submit_next(pool, futs):
        for idx, (rec, f, seed_k) in it:
            if (rec["bug_id"], seed_k) in kept_seeds:
                continue                       # 续跑跳过已收
            a = {"rec": rec, "file_rel": f["rel"], "tier": f["tier"], "seed_k": seed_k,
                 "baseline": baselines[rec["bug_id"]]["baseline"],
                 "audit": (idx % cfg.audit_every == 0),
                 "timeout_s": cfg.timeout_s, "split_tag": split_tag}
            futs.add(pool.submit(attempt_one, a))
            return True
        return False

    t0, n_att = time.time(), 0
    if len(kept) >= quota:
        print(f"[synthetic:{split_tag}] staging 已达 quota，跳过", flush=True)
        return kept, stats
    try:
        with ProcessPoolExecutor(max_workers=cfg.workers) as pool:
            futs = set()
            for _ in range(cfg.workers * 2):
                if not submit_next(pool, futs):
                    break
            while futs and len(kept) < quota:
                done, futs = wait(futs, return_when=FIRST_COMPLETED)
                for fut in done:
                    r = fut.result()
                    n_att += 1
                    # 去重：同文件同行同算子 = 同一条教学信号，不收第二遍
                    if r.get("status") == "kept":
                        key = (r["src_bug"], r["file"], r.get("line_no"), r.get("mutation"))
                        if key in seen_mutants:
                            r = {**r, "status": "dead_duplicate"}
                        else:
                            seen_mutants.add(key)
                            kept.append(r)
                            kept_seeds.add((r["src_bug"], r["seed_k"]))
                            pairs_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                            pairs_fh.flush()
                    stats[r["status"]] += 1
                    if n_att % 25 == 0:
                        rate = len(kept) / max(n_att, 1)
                        eta = (quota - len(kept)) / max(rate, 1e-6) * (time.time() - t0) / max(n_att, 1)
                        print(f"  [synthetic:{split_tag}] attempts={n_att} kept={len(kept)}/{quota} "
                              f"rate={rate:.0%} eta≈{eta / 60:.0f}min dead={dict(stats)}", flush=True)
                    if len(kept) < quota:
                        submit_next(pool, futs)
            for fut in futs:
                fut.cancel()
    finally:
        pairs_fh.close()
    print(f"[synthetic:{split_tag}] 结束 kept={len(kept)}/{quota} attempts={n_att} "
          f"分布={dict(stats)} 耗时 {(time.time() - t0) / 60:.1f}min", flush=True)
    return kept, stats


def phase_official(recs, cfg):
    stage = OUTDIR / "official_staging.jsonl"
    done = {} if cfg.force_official else {r["bug_id"]: r for r in _load_staging(stage)}
    todo = [r for r in recs if r["bug_id"] not in done]
    print(f"[official] {len(recs)} 条 train bug，已完成 {len(done)}，本次 {len(todo)}", flush=True)
    if todo:
        results = _parallel_map(official_one, todo, cfg.workers, "official")
        with open(stage, "a", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        done.update({r["bug_id"]: r for r in results})
    kept = [r for r in done.values() if r.get("status") == "kept"]
    bad = {r["bug_id"]: r.get("status") for r in done.values() if r.get("status") != "kept"}
    print(f"[official] kept={len(kept)} 非 kept={bad or '无'}", flush=True)
    return {"kept": len(kept), "not_kept": bad}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    global BIN, OUTDIR
    ap = argparse.ArgumentParser(description="Task 2.1 SFT 数据构建")
    ap.add_argument("--phase", default="all",
                    choices=["all", "fixed", "baseline", "official", "synthetic",
                             "syneval", "merge"],
                    help="all=全流程；单 phase 跑完即停（staging 落盘）；merge=只从 "
                         "staging 汇总最终 jsonl（幂等，分 phase 跑完后收尾用）")
    ap.add_argument("--n-syn", type=int, default=2000, help="合成 SFT 对目标条数")
    ap.add_argument("--n-eval", type=int, default=50, help="clean 合成评测集条数（val 语料）")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-code-lines", type=int, default=300,
                    help="注入文件行数上限（prompt 预算口径，与 load_buggy_code 一致）")
    ap.add_argument("--min-code-lines", type=int, default=20)
    ap.add_argument("--max-passes", type=int, default=150, help="候选流轮转上限（防死循环）")
    ap.add_argument("--audit-every", type=int, default=10, help="每 N 个尝试抽 1 个做 golden 审计")
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--d4j-bin", default=BIN)
    ap.add_argument("--split-json", default=str(Path(REPO) / "data" / "split.json"))
    ap.add_argument("--out-dir", default=str(OUTDIR))
    ap.add_argument("--force-baseline", action="store_true")
    ap.add_argument("--force-official", action="store_true")
    ap.add_argument("--force-lowcpu", action="store_true",
                    help="跳过资源闸门（无卡模式 0.5核/2GB 下强行跑，极慢且易 OOM）")
    cfg = ap.parse_args()

    # 资源闸门（坑 #22）：cgroup 实配 <2 核直接拒跑；<8 核自动降 workers
    cores, mem_gb = _detect_resources()
    print(f"[resources] 有效 CPU={cores:.1f} 核 / 内存={mem_gb:.1f}GB "
          f"(nproc 显示 {os.cpu_count()} 是宿主机规格，不可信)", flush=True)
    if not cfg.force_lowcpu:
        if cores < 2 or mem_gb < 3:
            sys.exit("资源不足：无卡模式（0.5核/2GB）跑不了本脚本——编译+测试是 CPU/内存密集活，"
                     "限流下会把 Chart 等重项目 OOM 误杀成'编译失败'。"
                     "请带卡开机，或控制台『克隆实例』到任意有空闲 GPU 的宿主机（勾选拷贝数据盘）再跑。")
        safe_w = max(1, min(cfg.workers, int(cores), int(mem_gb / 1.5)))
        if safe_w < cfg.workers:
            print(f"[resources] workers {cfg.workers} → {safe_w}（按实配资源封顶）", flush=True)
            cfg.workers = safe_w

    BIN = cfg.d4j_bin
    OUTDIR = Path(cfg.out_dir)
    OUTDIR.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(cfg.split_json).read_text(encoding="utf-8"))
    train_recs, val_recs = data["train"], data["val"]
    # 纪律自检：test split 永不进入任何数据生成路径（bug_id 级隔离）
    test_ids = {r["bug_id"] for r in data.get("test", [])}
    gen_ids = {r["bug_id"] for r in train_recs + val_recs}
    assert not (test_ids & gen_ids), f"test split 混入数据生成: {test_ids & gen_ids}"

    t_start = time.time()
    report = {"date": str(date.today()), "seed": cfg.seed, "phase": cfg.phase,
              "config": {k: v for k, v in vars(cfg).items()}}

    if cfg.phase == "merge":
        report["merge"] = merge_outputs(cfg)
        _write_report(report, t_start)
        print("SFT_DATA_DONE", flush=True)
        return

    if cfg.phase in ("all", "fixed"):
        warm = phase_fixed_cache(train_recs + val_recs, cfg)
        report["fixed_cache"] = {"ok": sum(1 for r in warm if r.get("ok")),
                                 "total": len(warm),
                                 "failed": [r["bug_id"] for r in warm if not r.get("ok")]}
        if cfg.phase == "fixed":
            _write_report(report, t_start)
            print("SFT_DATA_DONE (phase=fixed)", flush=True)
            return

    baselines = None
    if cfg.phase in ("all", "baseline", "synthetic", "syneval"):
        baselines = phase_baseline(train_recs + val_recs, cfg)
        report["baseline_non_green"] = {b: v["status"] for b, v in baselines.items()
                                        if v["status"] != "green"}
        if cfg.phase == "baseline":
            _write_report(report, t_start)
            print("SFT_DATA_DONE (phase=baseline)", flush=True)
            return

    if cfg.phase in ("all", "official"):
        report["official"] = phase_official(train_recs, cfg)
        if cfg.phase == "official":
            _write_report(report, t_start)
            print("SFT_DATA_DONE (phase=official)", flush=True)
            return

    if cfg.phase in ("all", "synthetic", "syneval"):
        usable_recs = lambda recs: [r for r in recs  # noqa: E731
                                    if baselines.get(r["bug_id"], {}).get("status")
                                    in ("green", "failing")]
        for tag, recs, quota, stage_name in (
                ("train", train_recs, cfg.n_syn, "syn_staging_train.jsonl"),
                ("val", val_recs, cfg.n_eval, "syn_staging_val.jsonl")):
            if cfg.phase == "synthetic" and tag != "train":
                continue
            if cfg.phase == "syneval" and tag != "val":
                continue
            ur = usable_recs(recs)
            per_bug = {}
            for i, rec in enumerate(ur):
                per_bug[rec["bug_id"]] = enumerate_candidates(
                    rec, cfg.max_code_lines, cfg.min_code_lines, cfg.seed + i)
            n_files = sum(len(v) for v in per_bug.values())
            print(f"[synthetic:{tag}] 可用 bug={len(ur)}/{len(recs)} 候选文件={n_files}", flush=True)
            _, stats = phase_synthetic(ur, baselines, per_bug, cfg, tag, quota,
                                       OUTDIR / stage_name)
            report[f"synthetic_{tag}_dead"] = dict(stats)

    report["merge"] = merge_outputs(cfg)
    _write_report(report, t_start)
    print("SFT_DATA_DONE", flush=True)


def _write_report(report, t_start):
    report["elapsed_min"] = round((time.time() - t_start) / 60, 1)
    (OUTDIR / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        print("SFT_DATA_FAIL", flush=True)
        sys.exit(1)
