#!/bin/bash
# smoke_grpo.sh — GRPO 2 步 GPU 冒烟（plan §6 Phase 1）。在全量 2.5–7h 之前 fail-fast 验证整条管线。
#
# 验证什么（accept 判据，全过才放行全量）：
#   ① 峰值显存 ≤ 22 GiB（4090 24G，留余量）——后台 nvidia-smi 采样峰值；
#   ② loss 无 NaN/Inf；
#   ③ reward 非退化（不是全 0 也不是全同值——全 0=跑错世界/截断，见 plan §8 pitfall #1）；
#   ④ completion 里有且仅有一个 ```diff fence（格式没被 chat_template 搞坏）；
#   ⑤ logits_to_keep 生效（trl 0.24：logits 只占 B×(C+1)×V，显存不爆 = 间接证据）；
#   ⑥ batch/G 整除约束被 ms-swift 接受（否则启动即报错，立刻知道）。
#
# 两阶段（用 PHASE 切换；先 1 后 2）：
#   PHASE=1（默认，机制）：G=4 grad_accum=4 max_completion=512 max_steps=2 → 快（~3–5min），
#                          验 reward 接入/反向/无 NaN/diff 格式/整除约束。
#   PHASE=2（显存，全量配置复核）：G=4 grad_accum=4 max_completion=512 max_steps=1。
# ⚠️ 为什么不是 G=8（2026-11-10 实测两次 OOM 的根因）：OOM 在 **rollout 预填充激活**，不在训练 logits——
#   traceback: peft generate → transformers _sample → qwen2 MLP forward。ms-swift GRPO trainer 把整个
#   generation_batch（=G 条序列；smoke 集是特意挑的 ~4000 token 最长 prompt）塞进一次 generate
#   （pt_engine max_batch_size=0 硬编码不分批），MLP 中间激活 8×~4000×18944×bf16 ≈ 2.24GiB 分配失败。
#   它与 max_completion_length 无关（C 1024→512 后一字不差地同样 OOM）；trl 的 generation_batch_size
#   又不能小于 num_generations（sampler = gbs//G）→ 24G 卡上没有官方旋钮保住 G=8。
#   G=4/ga=4 为实证上限：phase1 峰值 21084MiB（≤22528 门槛，余量 1.4GiB），且与预筛 G=4 同口径。
# 用法：PHASE=2 bash smoke_grpo.sh   （或直接 export G/GRAD_ACCUM/... 覆盖）
set -euo pipefail

PHASE="${PHASE:-1}"
if [ "$PHASE" = "2" ]; then
  G="${G:-4}";  GRAD_ACCUM="${GRAD_ACCUM:-4}"; MAX_COMP="${MAX_COMP:-512}"; MAX_STEPS="${MAX_STEPS:-1}"
else
  G="${G:-4}";  GRAD_ACCUM="${GRAD_ACCUM:-4}"; MAX_COMP="${MAX_COMP:-512}";  MAX_STEPS="${MAX_STEPS:-2}"
fi

SWIFT=/root/autodl-tmp/envs/swift/bin/swift
MODEL=/root/autodl-tmp/sft_merged
DATA=/root/autodl-tmp/grpo_data/grpo_train_smoke.jsonl
PLUGIN=/root/rl_training/rl/grpo_reward_plugin.py
OUT=/root/grpo_run/smoke_out_phase${PHASE}

# ---- 环境（pre-flight 实测值）----
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export GRPO_JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64     # pre-flight #5b 实测
export D4J_BIN=/root/defects4j/framework/bin/defects4j
export GRPO_CACHE_DIR=/root/autodl-tmp/d4j_cache_fixed       # 86 warm
export GRPO_RUN_DIR=/root/grpo_run                           # overlay 27G（defects4j 工作区，可丢）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True       # 减碎片
mkdir -p "$GRPO_RUN_DIR" "$OUT"

echo "=== GRPO SMOKE phase=$PHASE G=$G grad_accum=$GRAD_ACCUM max_comp=$MAX_COMP max_steps=$MAX_STEPS ==="
echo "model=$MODEL data=$DATA out=$OUT"

# ---- 后台采样峰值显存（每 2s）----
VRAM_LOG=/root/grpo_run/smoke_vram_phase${PHASE}.log
: > "$VRAM_LOG"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$VRAM_LOG" 2>/dev/null || true
    sleep 2
  done ) &
VRAM_PID=$!
trap 'kill $VRAM_PID 2>/dev/null || true' EXIT

# ---- 冒烟训练（2 步即停）----
# ⚠️ set -e 会在 swift 失败时直接杀掉脚本 → FAILED 判定行永远打不出来（监控哨兵会死等）。
# 局部关 set -e，拿到真实 RC 后打出 OK/FAILED 判定再退。
set +e
"$SWIFT" rlhf --rlhf_type grpo \
  --model "$MODEL" \
  --train_type lora \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --target_modules all-linear \
  --dataset "$DATA" \
  --external_plugins "$PLUGIN" \
  --reward_funcs d4j_synthetic \
  --num_generations "$G" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_length 4096 --truncation_strategy delete \
  --max_completion_length "$MAX_COMP" \
  --temperature 1.0 --top_p 0.95 --top_k 50 \
  --beta 0.001 --epsilon 0.2 --num_iterations 1 \
  --loss_type grpo --scale_rewards group \
  --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --gradient_checkpointing true --torch_dtype bfloat16 --attn_impl sdpa \
  --use_vllm false --dynamic_sample false \
  --split_dataset_ratio 0 --eval_strategy no \
  --num_train_epochs 1 --max_steps "$MAX_STEPS" \
  --logging_steps 1 --save_steps "$MAX_STEPS" --save_total_limit 1 --log_completions true \
  --report_to tensorboard \
  --output_dir "$OUT" \
  2>&1 | tee /root/grpo_run/smoke_phase${PHASE}.log

RC=${PIPESTATUS[0]}
set -e
kill $VRAM_PID 2>/dev/null || true
PEAK=$(sort -n "$VRAM_LOG" | tail -1)
echo "=== SMOKE phase=$PHASE rc=$RC peak_vram=${PEAK:-?} MiB ==="
echo "判据：rc=0 且 peak_vram≤22528 MiB(22GiB) 且 日志无 nan 且 reward 非退化（查 smoke_phase${PHASE}.log 的 'reward'）"
[ "$RC" = "0" ] && echo "SMOKE_PHASE${PHASE}_OK" || echo "SMOKE_PHASE${PHASE}_FAILED"
