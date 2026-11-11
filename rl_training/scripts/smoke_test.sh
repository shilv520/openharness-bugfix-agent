#!/usr/bin/env bash
# 无卡模式体检脚本：max_steps=2 验证整条训练配置（永远不要跳过这步直接上卡）。
# 用法: bash smoke_test.sh sft | bash smoke_test.sh grpo
#
# 注意:
# - 无卡模式跑不动真训练，本脚本主要验证: 依赖导入、数据集加载、reward 函数注册、
#   模型加载路径、输出目录可写。真正的显存/吞吐验证要在带卡后的前 30 步做。
# - <...> 占位符在 Task 2/3 落地时替换成实际路径。
set -u
MODE=${1:-grpo}
MODEL_DIR=${MODEL_DIR:-$HOME/models/qwen2.5-coder-7b}

case "$MODE" in
  sft)
    swift sft \
      --model "$MODEL_DIR" \
      --train_type lora \
      --dataset "<sft_data.jsonl>" \
      --max_steps 2 --save_steps 2 \
      --output_dir /tmp/smoke_sft
    ;;
  grpo)
    swift rlhf --rlhf_type grpo \
      --model "$MODEL_DIR" \
      --train_type lora --quant_bits 4 \
      --use_vllm true --vllm_mode colocate --gpu_memory_utilization 0.4 \
      --offload_model true --offload_optimizer true \
      --num_generations 8 --per_device_train_batch_size 4 \
      --dataset "<rl_prompts.jsonl>" \
      --external_plugins "<reward_plugin.py>" \
      --reward_funcs "<reward_fn_name>" \
      --max_steps 2 --save_steps 2 \
      --output_dir /tmp/smoke_grpo
    ;;
  *)
    echo "未知模式: $MODE (支持 sft | grpo)"; exit 1
    ;;
esac

CODE=$?
echo "smoke($MODE) exit=$CODE"
exit $CODE
