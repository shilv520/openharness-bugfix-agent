#!/usr/bin/env python3
"""
prescreen_grpo.py — GRPO 方差预筛（plan §4.7，策略 (a) 合成-only）

是什么：全量 GRPO 之前，对候选 prompt 池先用 sft_merged 采样 G 个 completion，用**与训练完全
相同的 reward 路径**（grpo_reward_core.grade_batch → reward.py 合成判定）判分，只保留组内 reward
有方差（0 < n_pass < G）的 prompt，落成 grpo_train_prescreened.jsonl。

为什么：GRPO 的梯度 = 组内 advantage = (r_i − mean)/std。一个 prompt 的 G 个采样若**全 0**
（模型完全不会）或**全 1**（已掌握），组内 std=0 → advantage=0 → 这一步零梯度、白烧 GPU。
预筛把这些「死 prompt」剔掉，让全量训练每一步都有学习信号。SFT 合成 pass@1≈52% → 单样本 p≈0.52，
G=4 时 P(有方差)=1−p^G−(1−p)^G≈87%，故大多数 prompt 会保留，预筛主要剔两端（恒失败/恒成功）。
关键：判分必须走 grade_batch（训练同一把尺子），否则预筛的「方差」与训练时的不一致就白筛了。

怎么做：transformers 加载 sft_merged（bf16/cuda/sdpa），左 padding 批量 generate（每 prompt 连续
重复 G 份），采样参数与 GRPO 训练对齐（temp 1.0 / top_p 0.95 / top_k 50 / max_new 512）；
每批 B*G 个 completion 一次性走 grade_batch（去重 + 8 线程并发判分）；按 prompt 聚合 n_pass，
留 0<n_pass<G。支持 --shuffle/--limit/--target-pool 控制预筛规模（目标池 150–300，~36min）。

产物：grpo_train_prescreened.jsonl（保留 prompt，含全部 reward 元数据列，无 golden）+ prescreen_report.json。
跑在 swift venv（需 torch/transformers + GPU）：
  /root/autodl-tmp/envs/swift/bin/python /root/rl_training/rl/prescreen_grpo.py --target-pool 300
纪律：不改 reward.py；无关机/cron/timer；JAVA_HOME/PATH 兜底同 plugin；判分故障率>5% 由 grade_batch 抛异常杀筛。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

# ---- 环境兜底（同 grpo_reward_plugin）：grade_batch 会 spawn defects4j 子进程，需 java/git/perl 在 PATH ----
_D4J_BIN = os.environ.get("D4J_BIN", os.path.expanduser("~/defects4j/framework/bin/defects4j"))
if os.environ.get("GRPO_JAVA_HOME"):
    os.environ["JAVA_HOME"] = os.environ["GRPO_JAVA_HOME"]
_JAVA_BIN = os.path.join(os.environ["JAVA_HOME"], "bin") if os.environ.get("JAVA_HOME") else ""
_PREPEND = os.pathsep.join(p for p in (os.path.dirname(_D4J_BIN), _JAVA_BIN) if p)
if _PREPEND:
    os.environ["PATH"] = _PREPEND + os.pathsep + os.environ.get("PATH", "")

# ---- sys.path：grpo_reward_core 在同目录 rl/ ----
_RL_DIR = os.path.dirname(os.path.abspath(__file__))
if _RL_DIR not in sys.path:
    sys.path.insert(0, _RL_DIR)

from grpo_reward_core import grade_batch  # noqa: E402  (训练同一把判分尺子)


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO 方差预筛（留 0<n_pass<G 的 prompt）")
    ap.add_argument("--train", default="/root/autodl-tmp/grpo_data/grpo_train.jsonl")
    ap.add_argument("--model-dir", default="/root/autodl-tmp/sft_merged")
    # R4 护栏④：默认输出不再指向活池（此前默认=grpo_train_prescreened.jsonl/prescreen_report.json，
    # 重跑即覆盖 168 池+预筛证据）；且已存在的输出文件默认拒绝覆盖，--force 才放行。
    ap.add_argument("--out", default="/root/autodl-tmp/grpo_data/grpo_train_prescreened_v2.jsonl")
    ap.add_argument("--report", default="/root/autodl-tmp/grpo_data/prescreen_report_v2.json")
    ap.add_argument("--force", action="store_true", help="允许覆盖已存在的 --out/--report（默认拒绝）")
    ap.add_argument("--g", type=int, default=4, help="每 prompt 采样数（预筛用 4，训练用 8）")
    ap.add_argument("--batch-prompts", type=int, default=4, help="每次 generate 的 prompt 数（×G = batch 序列数）")
    ap.add_argument("--max-new-tokens", type=int, default=512, help="与训练 max_completion_length 对齐（冒烟实测 1024 OOM→512）")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--shuffle", action="store_true", help="打乱候选顺序（配 --seed）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="最多预筛多少候选 prompt（0=全部）")
    ap.add_argument("--target-pool", type=int, default=0, help="保留够这么多就提前停（0=不提前停）")
    cfg = ap.parse_args()

    for p in (cfg.out, cfg.report):
        if Path(p).exists() and not cfg.force:
            raise SystemExit(f"[prescreen] ❌ 拒绝覆盖已存在的 {p}（R4 防误覆盖护栏；确要覆盖加 --force）")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = [json.loads(l) for l in Path(cfg.train).read_text(encoding="utf-8").splitlines() if l.strip()]
    if cfg.shuffle:
        random.Random(cfg.seed).shuffle(rows)
    if cfg.limit > 0:
        rows = rows[: cfg.limit]
    print(f"[prescreen] 候选 prompt: {len(rows)} | G={cfg.g} | batch_prompts={cfg.batch_prompts} "
          f"| max_new={cfg.max_new_tokens} | target_pool={cfg.target_pool or '∞'}", flush=True)

    print(f"[prescreen] 加载 tokenizer/model: {cfg.model_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg.model_dir, trust_remote_code=True)
    tok.padding_side = "left"                       # 生成必须左 padding
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda", trust_remote_code=True)
    model.eval()
    print("[prescreen] 模型就绪", flush=True)

    kept: list[dict] = []
    npass_hist = Counter()                          # n_pass 分桶（0..G）
    stage_hist = Counter()
    screened = 0
    t0 = time.time()
    env_aborts = 0

    def _grade(recs_batch, comps_batch):
        """走训练同一把尺子；>5% env 故障 grade_batch 会抛 RuntimeError。"""
        return grade_batch(recs_batch, comps_batch)

    try:
        for start in range(0, len(rows), cfg.batch_prompts):
            if cfg.target_pool and len(kept) >= cfg.target_pool:
                print(f"[prescreen] 达到 target_pool={cfg.target_pool}，提前停（已筛 {screened}）", flush=True)
                break
            batch = rows[start: start + cfg.batch_prompts]
            # 1) 构 prompt 字符串（与训练同模板：messages + add_generation_prompt）
            prompts = [tok.apply_chat_template(r["messages"], add_generation_prompt=True, tokenize=False)
                       for r in batch]
            # 2) 每 prompt 连续重复 G 份
            repeated = [p for p in prompts for _ in range(cfg.g)]
            recs_batch = [r for r in batch for _ in range(cfg.g)]
            enc = tok(repeated, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            in_len = enc["input_ids"].shape[1]
            # 3) 采样生成（参数与 GRPO 训练对齐）
            with torch.no_grad():
                out = model.generate(
                    **enc, do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
                    top_k=cfg.top_k, max_new_tokens=cfg.max_new_tokens,
                    pad_token_id=tok.pad_token_id)
            comps = tok.batch_decode(out[:, in_len:], skip_special_tokens=True)
            # 4) 判分（训练同一路径）
            try:
                rewards = _grade(recs_batch, comps)
            except RuntimeError as e:
                env_aborts += 1
                print(f"[prescreen] ❌ grade_batch 杀筛（harness 故障率>5%）：{e}", flush=True)
                break
            # 5) 按 prompt 聚合 n_pass，留 0<n_pass<G
            for j, r in enumerate(batch):
                grp = rewards[j * cfg.g: (j + 1) * cfg.g]
                n_pass = sum(1 for x in grp if x == 1.0)
                npass_hist[n_pass] += 1
                screened += 1
                if 0 < n_pass < cfg.g:
                    keep = {k: v for k, v in r.items()}     # 含 messages + 全部 reward 元数据列
                    keep["_prescreen_n_pass"] = n_pass
                    kept.append(keep)
            elapsed = time.time() - t0
            print(f"[prescreen] 已筛 {screened}/{len(rows)} | 保留 {len(kept)} | "
                  f"n_pass 分布 {dict(sorted(npass_hist.items()))} | {elapsed:.0f}s", flush=True)
    finally:
        # ---- 写产物（即使中途 target-pool/env 停也落盘已筛部分）----
        out_p = Path(cfg.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        elapsed = time.time() - t0
        report = {
            "train_in": len(rows), "screened": screened, "kept": len(kept),
            "dropped_all_fail": npass_hist.get(0, 0), "dropped_all_pass": npass_hist.get(cfg.g, 0),
            "npass_hist": {str(k): v for k, v in sorted(npass_hist.items())},
            "g": cfg.g, "batch_prompts": cfg.batch_prompts, "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature, "top_p": cfg.top_p, "top_k": cfg.top_k,
            "shuffle": cfg.shuffle, "seed": cfg.seed, "limit": cfg.limit, "target_pool": cfg.target_pool,
            "env_aborts": env_aborts, "elapsed_s": round(elapsed, 1),
            "per_prompt_s": round(elapsed / screened, 2) if screened else 0,
            "out": str(out_p),
        }
        Path(cfg.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n=== prescreen 报告 ===")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nkept → {out_p}（{len(kept)} prompts）")
        ok = (env_aborts == 0) and len(kept) > 0
        print("PRESCREEN_DONE" if ok else "PRESCREEN_DONE_WITH_WARNINGS")


if __name__ == "__main__":
    main()
