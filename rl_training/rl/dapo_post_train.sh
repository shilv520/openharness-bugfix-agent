#!/bin/bash
# dapo_post_train.sh — D1 夜班自动链（会话断了也照跑）：
#   等 DAPO 全量 DONE → 腾磁盘 → 合并 checkpoint-150 → 坑#32 chat_template 复查 → 双评测（A 合成 21 / B 真实 27）→ POST_CHAIN_DONE 标记。
#
# ⚠️ 磁盘账（overlay 30G）：grpo_merged 已占 15G（它的评测 A/B 已完成且报告已归档本地
#   evidence/07_grpo_full/，权重可随时用保留的 grpo_out/v1/checkpoint-150 重新合并）→
#   合并 dapo_merged（15G）前必须先删 grpo_merged，否则 ENOSPC。绝不动：sft_merged、
#   d4j_cache*、grpo_out、dapo_out、任何数据。
# 评测口径与 GRPO 臂完全一致：eval_synthetic 默认 syn_eval_clean 21 条 + golden control；
#   run_baseline_local --split test 27 条 + 默认 d4j_cache；python 用 conda 全路径（坑：非交互 ssh 无 conda PATH）。
set -u
LOG=/root/dapo_run/full_run.log
PY=/root/miniconda3/bin/python3

echo "[post] waiting for DAPO full run marker..."
while ! grep -qE "DAPO_FULL_RUN_(DONE|FAILED)" "$LOG" 2>/dev/null; do sleep 60; done
if ! grep -q "DAPO_FULL_RUN_DONE" "$LOG"; then
  echo "POST_CHAIN_ABORT_RUN_FAILED"; exit 1
fi
echo "[post] run DONE, proceeding $(date -u +%FT%TZ)"

CKPT=$(ls -d /root/dapo_out/v*/checkpoint-150 2>/dev/null | tail -1)
[ -n "$CKPT" ] || { echo "POST_CHAIN_NO_CKPT150"; exit 1; }

echo "[post] free disk: rm grpo_merged (评测已完成+已归档，可由 grpo_out/v1/checkpoint-150 重合并)"
rm -rf /root/grpo_merged /root/grpo_run/w*
df -h / | tail -1

echo "[post] merging $CKPT -> /root/dapo_merged"
/root/autodl-tmp/envs/swift/bin/swift export --adapters "$CKPT" --merge_lora true \
  --output_dir /root/dapo_merged > /root/dapo_merge.log 2>&1 \
  || { echo "POST_CHAIN_MERGE_FAILED"; exit 1; }

if ! grep -q '"chat_template"' /root/dapo_merged/tokenizer_config.json; then
  cp /root/autodl-tmp/sft_merged/tokenizer_config.json /root/dapo_merged/tokenizer_config.json
  echo "[post] TEMPLATE_COPIED (坑#32)"
else
  echo "[post] TEMPLATE_PRESENT"
fi

cd /root/rl_training/scripts || exit 1
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

echo "[post] eval A: synthetic 21 (tag=dapo)"
$PY eval_synthetic.py --model-dir /root/dapo_merged --tag dapo > /root/eval_syn_dapo.log 2>&1
rc1=$?
echo "[post] eval B: real test 27"
$PY run_baseline_local.py --model-dir /root/dapo_merged --split test > /root/eval_test_dapo.log 2>&1
rc2=$?

echo "POST_CHAIN_DONE rc1=$rc1 rc2=$rc2 ckpt=$CKPT end=$(date -u +%FT%TZ)"
exit 0
