#!/bin/bash
# blind100_chain.sh — R3 盲测夜链（预注册：docs/prereg_blind100_D1.md，判定标准已先于本脚本落盘）：
#   抽样+适配（硬断言零交集）→ 腾磁盘（删 dapo_merged，可由 dapo_out/v0/checkpoint-150 重合）
#   → 合并 GRPO checkpoint-150 → 坑#32 chat_template 复查
#   → SFT 臂（golden 正对照全开 100 条 = harness 效度门）
#   → GRPO 臂（golden 关：golden 与臂无关，SFT 臂已在同 100 世界上验过）
#   → BLIND_CHAIN_DONE rc1= rc2=
# DAPO 臂推迟 D2 白天（磁盘串行纪律 + 不改变扩池判定）。
# ⚠️ 磁盘账（overlay 30G）：21G used → rm dapo_merged(15G) → 合并 grpo_merged(15G) → ~9G free。
#   绝不删：sft_merged、d4j_cache*、grpo_out、dapo_out、任何数据文件。
# python/swift 全路径（坑：非交互 ssh 无 conda PATH）。
set -u
PY=/root/miniconda3/bin/python3
SWIFT=/root/autodl-tmp/envs/swift/bin/swift
OUT=/root/blind100
mkdir -p "$OUT"

echo "[blind] chain start $(date -u +%FT%TZ)"

echo "[blind] step 0: prepare sample + adapter"
$PY /root/rl_training/scripts/blind100_prepare.py > "$OUT/prepare.log" 2>&1
rc0=$?
cat "$OUT/prepare.log"
echo "BLIND100_SAMPLE rc=$rc0"
[ $rc0 -eq 0 ] || { echo "BLIND_CHAIN_FAILED prepare rc=$rc0 $(date -u +%FT%TZ)"; exit 1; }

echo "[blind] step 1: free overlay — rm dapo_merged（评测已完成+已归档，checkpoint-150 保留可重合）"
rm -rf /root/dapo_merged
df -h / | tail -1

CKPT=$(ls -d /root/autodl-tmp/grpo_out/v*/checkpoint-150 2>/dev/null | tail -1)
[ -n "$CKPT" ] || { echo "BLIND_CHAIN_NO_CKPT150"; exit 1; }
echo "[blind] step 2: merge $CKPT -> /root/grpo_merged"
$SWIFT export --adapters "$CKPT" --merge_lora true \
  --output_dir /root/grpo_merged > "$OUT/merge_grpo.log" 2>&1 \
  || { echo "BLIND_CHAIN_MERGE_FAILED $(date -u +%FT%TZ)"; exit 1; }
echo "BLIND100_MERGE_GRPO rc=0 $(date -u +%FT%TZ)"

if ! grep -q '"chat_template"' /root/grpo_merged/tokenizer_config.json; then
  cp /root/autodl-tmp/sft_merged/tokenizer_config.json /root/grpo_merged/tokenizer_config.json
  echo "[blind] TEMPLATE_COPIED (坑#32)"
else
  echo "[blind] TEMPLATE_PRESENT"
fi

cd /root/rl_training/scripts || exit 1
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

echo "[blind] step 3: arm SFT（golden 正对照全开，效度门：必须 100/100）$(date -u +%FT%TZ)"
$PY eval_synthetic.py --model-dir /root/autodl-tmp/sft_merged --tag blind100_sft \
  --eval-json "$OUT/blind_eval.jsonl" > "$OUT/eval_sft.log" 2>&1
rc1=$?
echo "BLIND100_ARM_SFT rc=$rc1 $(date -u +%FT%TZ)"
grep -c "GOLDEN_CONTROL_FAILED" "$OUT/eval_sft.log" 2>/dev/null | sed 's/^/[blind] golden_control_failed_lines=/'

echo "[blind] step 4: arm GRPO（golden 关）$(date -u +%FT%TZ)"
$PY eval_synthetic.py --model-dir /root/grpo_merged --tag blind100_grpo \
  --eval-json "$OUT/blind_eval.jsonl" --no-golden-control > "$OUT/eval_grpo.log" 2>&1
rc2=$?
echo "BLIND100_ARM_GRPO rc=$rc2 $(date -u +%FT%TZ)"

echo "BLIND_CHAIN_DONE rc1=$rc1 rc2=$rc2 end=$(date -u +%FT%TZ)"
exit 0
