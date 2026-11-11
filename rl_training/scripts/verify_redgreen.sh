#!/usr/bin/env bash
# Task 1.1 验收 — 红绿闭环验证（env_setup_autodl.md §3）+ 保存真实输出样本 + 生成官方修复 diff
# 用法: bash /root/rl_training/scripts/verify_redgreen.sh   （init.sh 完成后）
# 标记: REDGREEN_DONE / REDGREEN_FAIL
set -u
export PATH=/root/defects4j/framework/bin:$PATH
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
OUT=/root/redgreen_samples
mkdir -p "$OUT"

echo "== [1/6] RED: checkout Lang-1b =="
defects4j checkout -p Lang -v 1b -w /tmp/lang1 2>&1 | tail -3 || { echo REDGREEN_FAIL checkout_1b; exit 21; }
echo "== [2/6] RED: compile =="
defects4j compile -w /tmp/lang1 2>&1 | tail -3 || { echo REDGREEN_FAIL compile_1b; exit 22; }
echo "== [3/6] RED: trigger test（期望 Failing tests: 1）=="
defects4j test -w /tmp/lang1 -t org.apache.commons.lang3.math.NumberUtilsTest::TestLang747 2>&1 | tee "$OUT/lang1_red.txt"

echo "== [4/6] GREEN: checkout+compile Lang-1f =="
defects4j checkout -p Lang -v 1f -w /tmp/lang1f 2>&1 | tail -3 || { echo REDGREEN_FAIL checkout_1f; exit 24; }
defects4j compile -w /tmp/lang1f 2>&1 | tail -3 || { echo REDGREEN_FAIL compile_1f; exit 25; }
echo "== [5/6] GREEN: trigger test（期望 Failing tests: 0）=="
defects4j test -w /tmp/lang1f -t org.apache.commons.lang3.math.NumberUtilsTest::TestLang747 2>&1 | tee "$OUT/lang1_green.txt"

echo "== [6/6] 官方修复 diff（reward 校准素材：对 1b 工作区反向 apply 引 bug 补丁）=="
cd /tmp/lang1 || { echo REDGREEN_FAIL cd; exit 26; }
git apply -R /root/defects4j/framework/projects/Lang/patches/1.src.patch || { echo REDGREEN_FAIL apply_R; exit 27; }
git diff > "$OUT/lang1.official_fix.diff"
wc -l "$OUT/lang1.official_fix.diff"

echo "== 对账：框架元数据 Lang-1 trigger_tests（应与本地 split.json 一致）=="
cat /root/defects4j/framework/projects/Lang/trigger_tests/1

echo REDGREEN_DONE
