#!/bin/bash
# dapo_full_run.sh — Task 4 全量 DAPO（与 grpo_full_run.sh 单变量对照：只差 DAPO 四 trick + β=0）。
#
# ⚠️ 放行前置：① dapo_smoke.sh PHASE=1+2 全 OK；② flag 已在 `swift rlhf --help` 逐一验证；
#   ③ golden 20/20（reward 未动可沿用 killpg 版 2026-11-11 00:20 的复验记录）。
#
# 对照设计（Gate 2 问题「DAPO > GRPO？」要能回答，就必须除 trick 外全同）：
#   同起点 sft_merged（fresh LoRA，KL 锚=SFT 策略；DAPO 臂 β=0 无 KL 项，锚只在 GRPO 臂生效）、
#   同数据 grpo_train_prescreened.jsonl（168 条）、同步数 150、同采样温度/截断/优化器。
#   预期差异点：dynamic_sample 重采样 → 每步判分次数上升（步耗时 ×1.5–2.5 可能），全对组被过滤
#   → frac_zero_std 应显著低于 GRPO 臂同期；overlong_filter → clip_ratio 高时 loss 更干净。
# 监控：tail -f /root/dapo_run/full_run.log；reward-hacking 预警 tail -f /root/dapo_run/reward_log.jsonl；
#   β=0 无 KL 约束，漂移全指望监控：kl>0.01 或长度膨胀>2× 或 hack 计数上升 → 手动停、如实记。
# 断点续跑：RESUME_FROM=/root/dapo_out/v0-*/checkpoint-NN nohup bash dapo_full_run.sh ...
# 纪律：绝不自动关机/释放；test split 官方 patch 永不进训练；reward 改动必重跑单测+golden。
set -euo pipefail

SWIFT=/root/autodl-tmp/envs/swift/bin/swift
MODEL=/root/autodl-tmp/sft_merged
DATA=/root/autodl-tmp/grpo_data/grpo_train_prescreened.jsonl
PLUGIN=/root/rl_training/rl/grpo_reward_plugin.py
OUT=/root/dapo_out                                       # overlay（autodl-tmp 只剩 4.7G，ckpt 放数据盘会 ENOSPC）

export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export GRPO_JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export D4J_BIN=/root/defects4j/framework/bin/defects4j
export GRPO_CACHE_DIR=/root/autodl-tmp/d4j_cache_fixed
export GRPO_RUN_DIR=/root/dapo_run
export GRPO_REWARD_LOG=/root/dapo_run/reward_log.jsonl
export GRPO_REWARD_TIMEOUT_S=240
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$GRPO_RUN_DIR" "$OUT"

RESUME_FROM="${RESUME_FROM:-}"
EXTRA_ARGS=()
if [ -n "$RESUME_FROM" ]; then
  [ -d "$RESUME_FROM" ] || { echo "RESUME_FROM 不存在: $RESUME_FROM"; exit 2; }
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
  echo "resume from: $RESUME_FROM"
fi

echo "=== DAPO FULL RUN ==="
echo "model=$MODEL"
echo "data=$DATA  (行数: $(wc -l < "$DATA"))"
echo "out=$OUT"
echo "start: $(date -u +%FT%TZ)"

set +e
"$SWIFT" rlhf --rlhf_type grpo \
  --model "$MODEL" \
  --train_type lora \
  --lora_rank 16 --lora_alpha 32 --lora_dropout 0.05 --target_modules all-linear \
  --dataset "$DATA" \
  --external_plugins "$PLUGIN" \
  --reward_funcs d4j_synthetic \
  --num_generations 4 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_length 4096 --truncation_strategy delete \
  --max_completion_length 512 \
  --temperature 1.0 --top_p 0.95 --top_k 50 \
  --beta 0 --epsilon 0.2 --epsilon_high 0.28 --num_iterations 1 \
  --loss_type dapo --scale_rewards group \
  --dynamic_sample true --max_resample_times 3 --overlong_filter true \
  --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --gradient_checkpointing true --torch_dtype bfloat16 --attn_impl sdpa \
  --use_vllm false \
  --split_dataset_ratio 0 --eval_strategy no \
  --num_train_epochs 1 --max_steps 150 \
  --logging_steps 1 --save_steps 40 --save_total_limit 2 --log_completions true \
  --report_to tensorboard \
  --output_dir "$OUT" \
  "${EXTRA_ARGS[@]}"

RC=$?
set -e
echo "end: $(date -u +%FT%TZ) rc=$RC"
echo "产物：$OUT；遥测：$GRPO_REWARD_LOG；tensorboard：$OUT/runs"
[ "$RC" = "0" ] && echo "DAPO_FULL_RUN_DONE" || echo "DAPO_FULL_RUN_FAILED rc=$RC"
exit $RC
