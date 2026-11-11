#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task 1.3 — 预构建缓存：split.json 全部 bug checkout+compile → /root/autodl-tmp/d4j_cache/<bug_id>

RewardExecutor.prepare_workspace 本身会回填缓存；本脚本一次性预热全部，
之后 reward 判定/评测只做 copytree，不再现场 checkout+compile。
test split 也预热（缓存的是 buggy 工作区，不含官方修复，不违反锁死纪律，
只是让 Task 1.7 基线和 Task 6 终评跑得更快）。

用法: /root/miniconda3/bin/python /root/rl_training/scripts/build_cache.py [workers=4]
并行度默认保守 4：同项目并发构建可能共享本地依赖仓库，跑稳后再调大。
幂等：已有缓存目录的 bug 直接跳过；失败的打印 CACHE_FAIL 不中断，重跑即补。
标记: BUILD_CACHE_DONE
"""
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = "/root/rl_training"
D4J = Path("/root/defects4j")
BIN = str(D4J / "framework/bin/defects4j")
CACHE = "/root/autodl-tmp/d4j_cache"
RUN = "/root/autodl-tmp/d4j_run"

os.environ.setdefault("JAVA_HOME", "/usr/lib/jvm/java-11-openjdk-amd64")
os.environ["PATH"] = (f"{D4J / 'framework/bin'}:{os.environ['JAVA_HOME']}/bin:"
                      + os.environ.get("PATH", ""))
sys.path.insert(0, REPO)


def warm_one(rec):
    from reward.reward import RewardExecutor, BugMeta
    bid = rec["bug_id"]
    if (Path(CACHE) / bid).exists():
        return bid, 0.0, "cached"
    ex = RewardExecutor(BIN, CACHE, RUN, timeout_s=2400)
    t0 = time.time()
    try:
        ws = ex.prepare_workspace(BugMeta.from_split_record(rec))
        shutil.rmtree(ws, ignore_errors=True)   # 返回的是副本，缓存已回填，副本删掉
        return bid, round(time.time() - t0, 1), "ok"
    except Exception as e:  # noqa: BLE001
        return bid, round(time.time() - t0, 1), f"FAIL:{str(e)[:200]}"


def main():
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    split = json.loads(Path(REPO, "data/split.json").read_text(encoding="utf-8"))
    recs = split["train"] + split["val"] + split["test"]
    todo = [r for r in recs if not (Path(CACHE) / r["bug_id"]).exists()]
    print(f"total={len(recs)} todo={len(todo)} workers={workers}", flush=True)
    done = fail = 0
    with ProcessPoolExecutor(workers) as pool:
        futs = {pool.submit(warm_one, r): r["bug_id"] for r in todo}
        for fut in as_completed(futs):
            bid, dur, status = fut.result()
            if status.startswith("FAIL"):
                fail += 1
                print(f"CACHE_FAIL {bid} ({dur}s) {status}", flush=True)
            else:
                done += 1
                print(f"[{done + fail}/{len(todo)}] {bid} {status} ({dur}s)", flush=True)
    print(f"summary: ok={done} fail={fail}", flush=True)
    os.system("du -sh /root/autodl-tmp/d4j_cache 2>/dev/null; df -h /root/autodl-tmp | tail -1")
    print("BUILD_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
