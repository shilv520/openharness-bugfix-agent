#!/usr/bin/env python3
"""
Task 1/2 · 本地 base 模型基线跑分（Gate 1 的对照基准：base pass@1）

与 run_baseline_api.py 同一套流水线（prompt_builder 唯一模板 → RewardExecutor
真实判定 → 断点续跑 jsonl → 汇总报告），只是把 OpenAI 兼容 API 换成本地
HF transformers 生成（greedy，temperature=0，与 API 版基线口径一致）。

为什么不用 vLLM：base 环境 torch 2.1.2+cu118，vllm 要求 torch≥2.4，装它 = 动
整个环境；27 条样本 HF generate 足够快（4090 上 ~20-60s/条）。Task 2.4 的
正式批量评测（base vs SFT，pass@k）再上 vLLM/ms-swift 环境。

在 AutoDL 实例上跑（需要 defects4j 环境 + d4j_cache + 已下载的模型）：
  nohup /root/miniconda3/bin/python /root/rl_training/scripts/run_baseline_local.py \
      > /root/baseline_local.log 2>&1 &
标记: BASELINE_DONE / BASELINE_FAIL
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

# 碎片化治理：7B bf16 权重 15.2G 后 24G 卡余量 ~8G，reserved-unallocated 曾达 3G+
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")
from collections import Counter
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent            # rl_training/scripts
RL_ROOT = HERE.parent                             # rl_training
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RL_ROOT / "reward"))
sys.path.insert(0, str(RL_ROOT / "data"))

from prompt_builder import SYSTEM_PROMPT, build_prompt          # noqa: E402
from reward import BugMeta, RewardExecutor                      # noqa: E402
from run_baseline_api import load_buggy_code                    # noqa: E402

MODEL_DIR = "/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct"


def load_model(model_dir: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16).to("cuda").eval()
    return tok, model


def generate(tok, model, user_prompt: str, max_tokens: int) -> tuple[str, dict]:
    """chat template 必须与 SFT/评测完全一致（prompt_builder 唯一事实源）。"""
    import torch
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = out[0][inputs["input_ids"].shape[1]:]
    meta = {"prompt_tokens": int(inputs["input_ids"].shape[1]),
            "gen_tokens": int(gen.shape[0]),
            "gen_latency_s": round(time.time() - t0, 2)}
    return tok.decode(gen, skip_special_tokens=True), meta


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    home = Path.home()
    ap = argparse.ArgumentParser(description="本地 base 模型 Defects4J 基线")
    ap.add_argument("--model-dir", default=MODEL_DIR)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-json", default=str(RL_ROOT / "data" / "split.json"))
    ap.add_argument("--d4j-bin", default=str(home / "defects4j/framework/bin/defects4j"))
    ap.add_argument("--d4j-home", default=str(home / "defects4j"))
    ap.add_argument("--cache-dir", default="/root/autodl-tmp/d4j_cache")
    ap.add_argument("--run-dir", default="/root/autodl-tmp/baseline_run")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-code-lines", type=int, default=300,
                    help="buggy 源码总行数硬预算（SFT 序列 2048-4096 口径，全链路一致）")
    cfg = ap.parse_args()

    if not cfg.out:
        cfg.out = str(Path("/root/autodl-tmp/baseline_results")
                      / f"local_{Path(cfg.model_dir).name}_{cfg.split}.jsonl")
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(cfg.split_json).read_text(encoding="utf-8"))
    records = data[cfg.split]
    if cfg.limit:
        records = records[: cfg.limit]

    done: set = set()
    if Path(cfg.out).exists():
        for line in Path(cfg.out).read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["bug_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for r in records if r["bug_id"] not in done]
    print(f"[baseline-local] model={cfg.model_dir} split={cfg.split} "
          f"共 {len(records)} 条，已完成 {len(done)}，本次跑 {len(todo)} 条", flush=True)
    if not todo:
        print("BASELINE_DONE (nothing to do)", flush=True)
        return

    print("loading model ...", flush=True)
    tok, model = load_model(cfg.model_dir)
    executor = RewardExecutor(defects4j_bin=cfg.d4j_bin, cache_dir=cfg.cache_dir,
                              run_dir=cfg.run_dir, timeout_s=cfg.timeout_s)

    rows, failed = [], []
    for i, rec in enumerate(todo, 1):
        bid = rec["bug_id"]
        try:
            t0 = time.time()
            ws = executor.prepare_workspace(BugMeta.from_split_record(rec))
            try:
                code, code_source = load_buggy_code(
                    ws, rec, Path(cfg.d4j_home), cfg.max_code_lines)
            finally:
                import shutil
                shutil.rmtree(ws, ignore_errors=True)
            prompt = build_prompt(rec, code)
            output, meta = generate(tok, model, prompt, cfg.max_tokens)
            result = executor.evaluate(BugMeta.from_split_record(rec), output)
            row = {
                "bug_id": bid, "project": rec["project"], "vid": rec["vid"],
                "model": Path(cfg.model_dir).name, "split": cfg.split,
                "date": str(date.today()),
                "reward": result.reward, "stage": result.stage, "detail": result.detail,
                "reward_duration_s": result.duration_s, **meta,
                "code_source": code_source, "total_s": round(time.time() - t0, 2),
                "model_output": output,
            }
            rows.append(row)
            with open(cfg.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  [{i}/{len(todo)}] {bid:12s} reward={row['reward']:<5} "
                  f"stage={row['stage']:32s} gen={meta['gen_latency_s']}s "
                  f"({meta['gen_tokens']} tok) reward={result.duration_s}s "
                  f"code={code_source}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(bid)
            print(f"  [{i}/{len(todo)}] {bid:12s} ERROR: {e}", flush=True)
            traceback.print_exc()
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()   # 清碎片，别让一次 OOM 拖垮后面的样本

    # 汇总（含历史行）
    all_rows = list(rows)
    if done and Path(cfg.out).exists():
        for line in Path(cfg.out).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r["bug_id"] in done:
                    all_rows.append(r)
            except json.JSONDecodeError:
                pass
    n = len(all_rows)
    fixed = sum(1 for r in all_rows if r["reward"] == 1.0)
    partial = sum(1 for r in all_rows if 0 < r["reward"] < 1)
    report = {
        "baseline": "local_base_model",
        "model": Path(cfg.model_dir).name, "split": cfg.split, "n": n,
        "date": str(date.today()),
        "fix_rate": round(fixed / n, 4) if n else 0.0,
        "fixed": fixed, "partial": partial, "zero": n - fixed - partial,
        "mean_reward": round(sum(r["reward"] for r in all_rows) / n, 4) if n else 0.0,
        "stage_distribution": dict(Counter(r["stage"] for r in all_rows)),
        "code_source_distribution": dict(Counter(r["code_source"] for r in all_rows)),
        "per_bug": {r["bug_id"]: r["reward"] for r in sorted(all_rows, key=lambda x: x["bug_id"])},
        "failed_this_run": failed,
    }
    report_path = Path(cfg.out).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print("\n=== 本地 base 基线报告（Gate 1 对照基准） ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告: {report_path}\n明细: {cfg.out}")
    print("BASELINE_DONE" if not failed else f"BASELINE_DONE_WITH_FAILS:{failed}",
          flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("BASELINE_FAIL", flush=True)
        sys.exit(1)
