#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 1.4 — reward Linux 校准与压测（AutoDL 实例上跑，不依赖 GPU）

三段式：
  pass1 正确性+冷启动：每 bug 生成官方修复 diff → evaluate 应得 1.0/full_pass；
        乱 diff → 0/patch_apply_failed；非 diff 文本 → -0.1/format_error；
        改测试文件 → 0/reward_hacking_test_file_modified
        （每 bug 首次评估会 checkout+compile 并回填预构建缓存）
  pass2 热缓存延迟：官方 patch 重评 → Gate 0 指标（目标单条 < 30s）
  pass3 并行吞吐：farm_evaluate N workers → Gate 0 指标（目标 ≥ 8 条/30s）

用法（init.sh 完成后）：
  /root/miniconda3/bin/python /root/rl_training/scripts/reward_calib_linux.py [n_bugs=8] [workers=8]
产物：
  /root/autodl-tmp/calib_results/calib_summary.json + <bug_id>.official.diff
结束标记：CALIB_DONE / CALIB_FAIL
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = "/root/rl_training"
D4J = Path("/root/defects4j")
BIN = str(D4J / "framework/bin/defects4j")
CACHE = "/root/autodl-tmp/d4j_cache"
RUN = "/root/autodl-tmp/d4j_run"
OUTDIR = Path("/root/autodl-tmp/calib_results")

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")
os.environ["PATH"] = (f"{D4J / 'framework/bin'}:{os.environ['JAVA_HOME']}/bin:"
                      + os.environ.get("PATH", ""))
sys.path.insert(0, REPO)

from reward.reward import (RewardExecutor, BugMeta, farm_evaluate,  # noqa: E402
                           is_test_path, normalize_java_eol)


def sh(cmd, cwd=None, timeout=2400):
    # ⚠️ 必须字节捕获再 decode：text=True 的通用换行翻译会把 git diff 输出里的
    # \r 悄悄剥掉（CRLF 文件的补丁落地变纯 LF → 与文件字节不匹配 → apply 失败）
    cp = subprocess.run(cmd, cwd=cwd, capture_output=True, timeout=timeout)
    return cp.returncode, (cp.stdout.decode("utf-8", "replace") + "\n"
                           + cp.stderr.decode("utf-8", "replace"))


def pick_bugs(n):
    """从 train split 按项目轮转取 n 条（保证覆盖编译最慢的 Chart）。"""
    split = json.loads(Path(REPO, "data/split.json").read_text(encoding="utf-8"))
    by = {}
    for r in split["train"]:
        by.setdefault(r["project"], []).append(r)
    order = ["Lang", "Time", "Chart"]
    out = []
    while len(out) < n and any(by.get(p) for p in order):
        for p in order:
            if by.get(p) and len(out) < n:
                out.append(by[p].pop(0))
    return out


def checkout_buggy(rec):
    ws = Path("/tmp") / f"calib_{rec['bug_id']}_{os.getpid()}"
    shutil.rmtree(ws, ignore_errors=True)
    rc, out = sh([BIN, "checkout", "-p", rec["project"],
                  "-v", f"{rec['vid']}b", "-w", str(ws)])
    if rc != 0:
        raise RuntimeError(f"checkout {rec['bug_id']}: {out[-400:]}")
    n = normalize_java_eol(ws)  # 与 reward.prepare_workspace 同一个世界（LF）
    if n:
        # 必须把归一化 commit 进 HEAD：否则后续 git diff 会把 EOL 变更（含测试
        # 文件）一并输出 → validate_patch_safety 把官方补丁误判 reward_hacking
        # （第三轮实测：Lang/Chart 全灭于 0.0s 文本级短路，纯 LF 的 Time 幸免）
        sh(["git", "add", "-A"], cwd=ws)
        sh(["git", "-c", "user.email=calib@local", "-c", "user.name=calib",
            "commit", "-q", "-m", "eol-normalize"], cwd=ws)
    return ws


def official_fix_diff(rec):
    """buggy 工作区 + 反向 apply 引 bug 补丁（src.patch 是 fixed→buggy 方向）→ git diff 即官方修复。"""
    ws = checkout_buggy(rec)
    try:
        patch = D4J / f"framework/projects/{rec['project']}/patches/{rec['vid']}.src.patch"
        # src.patch 内容行同样可能是 CRLF（随源文件），工作区已归一 LF → 补丁也要归一
        lf_patch = Path("/tmp") / f"src_lf_{rec['bug_id']}.patch"
        lf_patch.write_bytes(patch.read_bytes().replace(b"\r\n", b"\n"))
        rc, out = sh(["git", "apply", "-R", str(lf_patch)], cwd=ws)
        if rc != 0:
            raise RuntimeError(f"git apply -R {rec['bug_id']}: {out[-400:]}")
        _, diff = sh(["git", "diff"], cwd=ws)
        if not diff.strip():
            raise RuntimeError(f"empty official diff {rec['bug_id']}")
        return diff
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def hacked_diff(rec):
    """模拟典型 reward hacking：只改一个测试文件（追加注释），不修 bug。"""
    ws = checkout_buggy(rec)
    try:
        tf = next((p for p in sorted(ws.rglob("*.java"))
                   if is_test_path(p.relative_to(ws).as_posix())), None)
        if tf is None:
            raise RuntimeError(f"no test file found: {rec['bug_id']}")
        with tf.open("a", encoding="utf-8", errors="ignore") as f:
            f.write("\n// calibration: simulated reward hacking\n")
        _, diff = sh(["git", "diff"], cwd=ws)
        return diff, tf.relative_to(ws).as_posix()
    finally:
        shutil.rmtree(ws, ignore_errors=True)


GARBAGE_DIFF = ("```diff\n--- a/src/java/NoSuchFile.java\n+++ b/src/java/NoSuchFile.java\n"
                "@@ -1,2 +1,2 @@\n-line that does not exist xyz\n+whatever\n```")
NON_DIFF = "I looked at the code but I cannot figure out how to fix this bug."


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    OUTDIR.mkdir(parents=True, exist_ok=True)
    bugs = pick_bugs(n)
    print(f"calib bugs: {[b['bug_id'] for b in bugs]} workers={workers}", flush=True)

    ex_cold = RewardExecutor(BIN, CACHE, RUN, timeout_s=2400)
    pass1, diffs = [], {}
    for rec in bugs:
        bid = rec["bug_id"]
        t0 = time.time()
        row = {"bug_id": bid, "project": rec["project"]}
        try:
            off = official_fix_diff(rec)
            hack, hack_path = hacked_diff(rec)
            (OUTDIR / f"{bid}.official.diff").write_text(off, encoding="utf-8")
            diffs[bid] = off
            meta = BugMeta.from_split_record(rec)
            r1 = ex_cold.evaluate(meta, f"Here is my fix:\n```diff\n{off}```")
            r2 = ex_cold.evaluate(meta, GARBAGE_DIFF)
            r3 = ex_cold.evaluate(meta, NON_DIFF)
            r4 = ex_cold.evaluate(meta, f"Fix:\n```diff\n{hack}```")
            row.update({
                "official": {"reward": r1.reward, "stage": r1.stage,
                             "dur_s": r1.duration_s, "detail": r1.detail[:150]},
                "garbage": {"reward": r2.reward, "stage": r2.stage},
                "nondiff": {"reward": r3.reward, "stage": r3.stage},
                "hacked": {"reward": r4.reward, "stage": r4.stage,
                           "hack_path": hack_path, "detail": r4.detail[:150]},
                "cold_wall_s": round(time.time() - t0, 1),
            })
            row["correct"] = (r1.reward == 1.0 and r2.reward == 0.0
                              and r3.reward == -0.1 and r4.reward == 0.0
                              and r4.stage == "reward_hacking_test_file_modified")
            print(f"[pass1] {bid} correct={row['correct']} "
                  f"official={r1.reward}/{r1.stage} ({r1.duration_s}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            row["error"] = str(e)[:400]
            print(f"[pass1] {bid} ERROR {e}", flush=True)
        pass1.append(row)

    # ---- pass 2: 热缓存延迟（Gate 0 单条 <30s）----
    ex_eval = RewardExecutor(BIN, CACHE, RUN, timeout_s=900)
    pass2 = []
    for rec in bugs:
        bid = rec["bug_id"]
        if bid not in diffs:
            continue
        r = ex_eval.evaluate(BugMeta.from_split_record(rec), f"```diff\n{diffs[bid]}```")
        pass2.append({"bug_id": bid, "reward": r.reward, "stage": r.stage,
                      "dur_s": r.duration_s})
        print(f"[pass2] {bid} warm latency {r.duration_s}s "
              f"reward={r.reward} stage={r.stage}", flush=True)

    # ---- pass 3: 并行吞吐（Gate 0 ≥8 条/30s）----
    # 稳态吞吐：任务量 = 4×workers。只跑一轮 n=workers 时 wall≈最慢单条，
    # 量的是延迟不是吞吐（第一轮 5.24/30s 就是这么被拖垮的）
    base_tasks = [(rec, f"```diff\n{diffs[rec['bug_id']]}```")
                  for rec in bugs if rec["bug_id"] in diffs]
    tasks = base_tasks * 4
    t0 = time.time()
    farm = farm_evaluate({"defects4j_bin": BIN, "cache_dir": CACHE,
                          "run_dir": RUN, "timeout_s": 900},
                         tasks, max_workers=workers)
    wall = time.time() - t0
    pass3 = {"n": len(farm), "wall_s": round(wall, 1),
             "per_30s": round(len(farm) / wall * 30, 2) if wall > 0 else None,
             "rewards": [f["reward"] for f in farm]}
    print(f"[pass3] farm n={pass3['n']} wall={pass3['wall_s']}s "
          f"throughput={pass3['per_30s']} per 30s", flush=True)

    lat = [p["dur_s"] for p in pass2]
    n_ok = len([r for r in pass1 if r.get("correct")])
    gate0 = {
        "single_lt_30s": bool(lat) and max(lat) < 30,
        "median_latency_s": sorted(lat)[len(lat) // 2] if lat else None,
        "max_latency_s": max(lat) if lat else None,
        "farm_ge_8_per_30s": (pass3["per_30s"] or 0) >= 8,
        "all_correct": n_ok == len(bugs),
        "n_correct": f"{n_ok}/{len(bugs)}",
    }
    summary = {"n_bugs": n, "workers": workers, "pass1": pass1,
               "pass2_latency": pass2, "pass3_farm": pass3, "gate0": gate0}
    (OUTDIR / "calib_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("GATE0: " + json.dumps(gate0, ensure_ascii=False), flush=True)
    print("CALIB_DONE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        print(f"CALIB_FAIL: {e}", flush=True)
        raise
