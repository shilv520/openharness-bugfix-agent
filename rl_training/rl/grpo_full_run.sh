#!/bin/bash
# grpo_full_run.sh — Task 3 全量 GRPO 训练（策略 (a) 合成-only）。plan §5/§7 定稿配置。
#
# ⚠️ 放行前置（缺一不可，全在冒烟+预筛绿了、且用户看过本命令点头后才跑）：
#   1. smoke phase1+phase2 全 OK（峰值显存≤22GiB、无 NaN、reward 非退化、diff 格式对、整除约束过）；
#   2. train golden 正对照全 1.0：grpo_reward_core.py 20 /root/autodl-tmp/grpo_data/grpo_golden.jsonl；
#   3. 方差预筛产出 grpo_train_prescreened.jsonl（目标池 150–300，剔掉恒 0/恒 1 的死 prompt）；
#   4. 用户已看本命令 + 冒烟结果并明确放行（承诺：全量前停下）。
#
# G=4（不是 8）+ max_completion_length=512（不是 1024）——冒烟实测定的：
#   ① G=8 两次 OOM，根因在 **rollout 预填充激活**（traceback: peft generate→qwen2 MLP forward）：
#      ms-swift 把整个 generation_batch（=G 条 ~4000 token 长 prompt）塞进一次 generate（PtEngine
#      max_batch_size=0 硬编码不分批），MLP 激活 8×4000×18944×bf16≈2.24GiB 分配失败；与 C 无关，
#      trl generation_batch_size 又不能<num_generations → 24G 卡保不住 G=8。G=4/ga=4 实证峰值
#      21084MiB（余量 1.4GiB），且与方差预筛 G=4 完全同口径。每优化步=1 prompt×4 采样，p≈0.52 时
#      P(组内有方差)≈87%，advantage 信号充足。G=8 留给将来更大显存的卡。
#   ② C=512：completion 实测均值 ~99 token、512 下 clipped_ratio=0.0 → 512 不截断且省显存。
#
# 实测 ~26.6s/步 × 150 步 ≈ 67min → 必须 nohup 后台跑（脱离 ssh 会话）：
#   cd /root/rl_training/rl && nohup bash grpo_full_run.sh > /root/grpo_run/full_run.log 2>&1 &
# 断点续跑（坑 #37 后加）：RESUME_FROM=/path/to/checkpoint-80 nohup bash grpo_full_run.sh ...
# 监控：tail -f /root/grpo_run/full_run.log ；reward-hacking 早预警 tail -f $GRPO_REWARD_LOG。
#
# ⚠️ 2026-11-10 首跑在 step 113/150 被旧版按批 5% 熔断杀掉（1 条 test_run_timeout/4=25%）：
#   ① 熔断已改窗口化（grpo_reward_core.py：最近 100 次判定 env 率>5% 才杀，单条只告警）；
#   ② timeout 族不再 2× 重试（确定性挂死重试必再挂，白烧一倍超时）；
#   ③ GRPO_REWARD_TIMEOUT_S 600→240（合法判分尾部 ~30s，240s 已是 8×；挂死成本 30min→4min）。
# 纪律：绝不自动关机/释放；test split 官方 patch 永不进训练；reward 改动必重跑 test_reward_unit.py。
set -euo pipefail

SWIFT=/root/autodl-tmp/envs/swift/bin/swift
MODEL=/root/autodl-tmp/sft_merged                          # SFT 合并全权重（GRPO 起点 + KL 锚点）
DATA=/root/autodl-tmp/grpo_data/grpo_train_prescreened.jsonl   # 方差预筛后的池（无 golden）
PLUGIN=/root/rl_training/rl/grpo_reward_plugin.py
OUT=/root/autodl-tmp/grpo_out                              # 数据盘（持久，6.6G）；LoRA ckpt 小 + limit 2 兜住

# ---- 环境（pre-flight 实测值）----
export CUDA_VISIBLE_DEVICES=0
export NPROC_PER_NODE=1
export GRPO_JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export D4J_BIN=/root/defects4j/framework/bin/defects4j
export GRPO_CACHE_DIR=/root/autodl-tmp/d4j_cache_fixed     # 86 warm fixed 世界
export GRPO_RUN_DIR=/root/grpo_run                         # overlay 27G（defects4j 工作区，可丢）
export GRPO_REWARD_LOG=/root/grpo_run/reward_log.jsonl     # 判分遥测（stage 直方图 + hack 计数）
export GRPO_REWARD_TIMEOUT_S=240                           # 单次判定超时（默认 600 太宽：挂死要烧 30min）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$GRPO_RUN_DIR" "$OUT"

# 断点续跑（可选）：RESUME_FROM=/root/autodl-tmp/grpo_out/v0-.../checkpoint-80
RESUME_FROM="${RESUME_FROM:-}"
EXTRA_ARGS=()
if [ -n "$RESUME_FROM" ]; then
  [ -d "$RESUME_FROM" ] || { echo "RESUME_FROM 不存在: $RESUME_FROM"; exit 2; }
  EXTRA_ARGS+=(--resume_from_checkpoint "$RESUME_FROM")
  echo "resume from: $RESUME_FROM"
fi

echo "=== GRPO FULL RUN ==="
echo "model=$MODEL"
echo "data=$DATA  (行数: $(wc -l < "$DATA"))"
echo "out=$OUT"
echo "start: $(date -u +%FT%TZ)"

# ⚠️ set -e 会在 swift 失败时直接杀掉脚本 → GRPO_FULL_RUN_FAILED 标记打不出来（监控死等）。局部关闭。
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
  --beta 0.001 --epsilon 0.2 --num_iterations 1 \
  --loss_type grpo --scale_rewards group \
  --learning_rate 1e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 \
  --gradient_checkpointing true --torch_dtype bfloat16 --attn_impl sdpa \
  --use_vllm false --dynamic_sample false \
  --split_dataset_ratio 0 --eval_strategy no \
  --num_train_epochs 1 --max_steps 150 \
  --logging_steps 1 --save_steps 40 --save_total_limit 2 --log_completions true \
  --report_to tensorboard \
  --output_dir "$OUT" \
  "${EXTRA_ARGS[@]}"

RC=$?
set -e
echo "end: $(date -u +%FT%TZ) rc=$RC"
echo "产物：$OUT（LoRA adapter checkpoint-*）；遥测：$GRPO_REWARD_LOG；tensorboard：$OUT/runs"
[ "$RC" = "0" ] && echo "GRPO_FULL_RUN_DONE" || echo "GRPO_FULL_RUN_FAILED rc=$RC"
exit $RC
