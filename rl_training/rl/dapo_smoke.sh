#!/bin/bash
# dapo_smoke.sh — Task 4 DAPO 两步 GPU 冒烟（与 smoke_grpo.sh 同款两阶段判据）。
#
# ⚠️ 放行前置（缺一不可）：
#   ① flag 验证：swift rlhf --help 里确认本版本存在 loss_type=dapo / epsilon_high /
#      dynamic_sample / max_resample_times / overlong_filter（预研记的是 ms-swift 3.x 有，
#      实装版本号以 `swift --version` 为准；没有的旋钮必须降级并在日志注明，不许静默丢）；
#   ② reward 若再改动：单测 28/28 + golden 20/20（killpg 版已于 2026-11-11 00:20 复验过）；
#   ③ 冒烟 phase1+2 全 OK（rc=0、峰值显存 ≤22GiB、无 NaN、reward 非退化、diff 格式对）才许全量。
#
# 与 GRPO 臂的唯一差异 = DAPO 四 trick（单变量对照，其余逐参数一致）：
#   1. --loss_type dapo          token-level policy gradient loss（DAPO trick 3）
#   2. --epsilon_high 0.28       clip-higher：ε_low=0.2 / ε_high=0.28 解耦，防熵坍塌（trick 1）
#   3. --dynamic_sample true     全对/全错组重采样过滤（trick 2）——治 GRPO 首跑 step111 实证的全对组零信号
#      --max_resample_times 3    重采样上限（防死循环烧判分）
#   4. --overlong_filter true    超长截断样本掩掉 loss，不算惩罚噪声（trick 4 的 ms-swift 实现）
#   5. --beta 0                  DAPO 原文去掉 KL 项（忠实配置）；kl 仍会记录用于监控漂移
# ⚠️ dynamic_sample 会让每步生成量最多 ×(1+max_resample_times)：冒烟同时验证「重采样不爆显存/不卡死」。
#
# 用法：PHASE=1 bash dapo_smoke.sh → PHASE=2 bash dapo_smoke.sh
set -euo pipefail

PHASE="${PHASE:-1}"
if [ "$PHASE" = "2" ]; then
  G="${G:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-4}"; MAX_COMP="${MAX_COMP:-512}"; MAX_STEPS="${MAX_STEPS:-1}"
else
  G="${G:-4}"; GRAD_ACCUM="${GRAD_ACCUM:-4}"; MAX_COMP="${MAX_COMP:-512}"; MAX_STEPS="${MAX_STEPS:-2}"
fi

SWIFT=/root/autodl-tmp/envs/swift/bin/swift
MODEL=/root/autodl-tmp/sft_merged                       # 与 GRPO 臂同起点（公平对照）
DATA=/root/autodl-tmp/grpo_data/grpo_train_smoke.jsonl  # 与 GRPO 冒烟同数据
PLUGIN=/root/rl_training/rl/grpo_reward_plugin.py       # reward 不动（判分数学与 GRPO 完全一致）
OUT=/root/dapo_run/smoke_out_phase${PHASE}              # overlay（autodl-tmp 只剩 4.7G）

export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export GRPO_JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export D4J_BIN=/root/defects4j/framework/bin/defects4j
export GRPO_CACHE_DIR=/root/autodl-tmp/d4j_cache_fixed
export GRPO_RUN_DIR=/root/dapo_run
export GRPO_REWARD_LOG=/root/dapo_run/reward_log_smoke.jsonl
export GRPO_REWARD_TIMEOUT_S=240
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$GRPO_RUN_DIR" "$OUT"

echo "=== DAPO SMOKE phase=$PHASE G=$G grad_accum=$GRAD_ACCUM max_steps=$MAX_STEPS ==="
echo "model=$MODEL data=$DATA out=$OUT"

VRAM_LOG=/root/dapo_run/smoke_vram_phase${PHASE}.log
: > "$VRAM_LOG"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$VRAM_LOG" 2>/dev/null || true
    sleep 2
  done ) &
VRAM_PID=$!
trap 'kill $VRAM_PID 2>/dev/null || true' EXIT

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
  --beta 0 --epsilon 0.2 --epsilon_high 0.28 --num_iterations 1 \
  --loss_type dapo --scale_rewards group \
  --dynamic_sample true --max_resample_times 3 --overlong_filter true \
  --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --gradient_checkpointing true --torch_dtype bfloat16 --attn_impl sdpa \
  --use_vllm false \
  --split_dataset_ratio 0 --eval_strategy no \
  --num_train_epochs 1 --max_steps "$MAX_STEPS" \
  --logging_steps 1 --save_steps "$MAX_STEPS" --save_total_limit 1 --log_completions true \
  --report_to tensorboard \
  --output_dir "$OUT" \
  2>&1 | tee /root/dapo_run/smoke_phase${PHASE}.log

RC=${PIPESTATUS[0]}
set -e
kill $VRAM_PID 2>/dev/null || true
PEAK=$(sort -n "$VRAM_LOG" | tail -1)
echo "=== DAPO SMOKE phase=$PHASE rc=$RC peak_vram=${PEAK:-?} MiB ==="
[ "$RC" = "0" ] && echo "DAPO_SMOKE_PHASE${PHASE}_OK" || echo "DAPO_SMOKE_PHASE${PHASE}_FAILED"
