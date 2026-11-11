#!/usr/bin/env bash
# Task 1 验收链：红绿闭环验证 → 语义断言（红必须 Failing tests: 1，绿必须 0）→ reward Linux 校准压测
# 日志: /root/redgreen_calib.log
# 标记: CHAIN START / SEMANTIC_OK / CHAIN_STOP_* / CALIB_START / CALIB_DONE / CALIB_FAIL / CHAIN END
LOG=/root/redgreen_calib.log
exec >"$LOG" 2>&1
echo "=== [$(date '+%F %T')] CHAIN START ==="

bash /root/rl_training/scripts/verify_redgreen.sh
rc=$?
[ $rc -eq 0 ] || { echo "CHAIN_STOP redgreen_exit=$rc"; exit $rc; }

grep -q "Failing tests: 1" /root/redgreen_samples/lang1_red.txt || { echo CHAIN_STOP_semantic_red; exit 41; }
grep -q "Failing tests: 0" /root/redgreen_samples/lang1_green.txt || { echo CHAIN_STOP_semantic_green; exit 42; }
echo "SEMANTIC_OK red=1 green=0"

echo "=== [$(date '+%F %T')] CALIB START ==="
/root/miniconda3/bin/python /root/rl_training/scripts/reward_calib_linux.py 8 8
rc=$?
echo "=== [$(date '+%F %T')] CHAIN END rc=$rc ==="
exit $rc
