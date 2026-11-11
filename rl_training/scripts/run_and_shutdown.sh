#!/usr/bin/env bash
# 带卡模式保命脚本：命令跑完（无论成败）自动关机。
# 用法: bash run_and_shutdown.sh "swift rlhf --rlhf_type grpo ..."
set -u

if [ $# -lt 1 ]; then
    echo "用法: bash run_and_shutdown.sh \"<训练命令>\""
    exit 1
fi

LOG="run_$(date +%Y%m%d_%H%M%S).log"
echo "[boot] $(date) cmd: $*" | tee -a "$LOG"

bash -c "$*" 2>&1 | tee -a "$LOG"
CODE=${PIPESTATUS[0]}

echo "[done] $(date) exit=$CODE" | tee -a "$LOG"

# ==== 关机前的保命动作（按需取消注释）====
# 同步 checkpoint 到数据盘/网盘：
# cp -r output/ /root/autodl-tmp/ 2>/dev/null
# 或: python -c "from modelscope.hub.api import HubApi; ..."

echo "[shutdown] 10 秒后关机（反悔就 Ctrl-C）"
sleep 10
shutdown 2>/dev/null || /usr/bin/shutdown 2>/dev/null || echo "关机命令不可用，请手动到控制台关机！"
