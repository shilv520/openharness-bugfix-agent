#!/bin/bash
# f56_eval_chain.sh — Task 6 终评基建：frozen 条件集三臂评测链（v2：9 代理对抗审查 5 条确认级全落地）
#
# 是什么：钉尺子（私有快照+provenance 对账）→ SFT 臂（golden 正对照全开 = 效度门）
#   → GRPO 臂（golden 关）→ 两臂证据闸全过 + 双 checkpoint 预验后才腾磁盘 rm grpo_merged
#   → 合并 DAPO checkpoint-150 → DAPO 臂 → 三臂 keyset 同一性终验 → 收尾腾磁盘。
# v2 修复（对应审查 confirmed #1-#5）：
#   F1 钉尺子：EVAL_JSON 在链 3h 生命周期里原被独立读 4 次，可被并发 --force run 原子改写
#      → 三臂在不同条数的尺子上静默打分。现 step 0 复制私有快照 + sha256 与 frozen56_report
#      的 final_sha256 对账（不等即拒跑），三臂一律评快照；链尾复查源文件是否被改（如实记录）。
#   F2 证据闸：rc==0 按设计包含「臂不完整/无效」（failed 行/env_error 行都缩 n 而 rc 仍 0）。
#      现每臂收工后从权威 report.json 硬闸：n==N_COND、harness_error==0、failed_this_run 空；
#      SFT 臂加 golden_checked==N_COND 且 golden_failed 空（效度门真的当门用）。
#      rm grpo_merged 挪到 GCKPT+DCKPT 双预验 + GRPO 证据闸之后（缺 DCKPT 不再白删）。
#   F3 终标不再谎报：DONE 行的 n_cond/golden_fail 改从三臂 report 取实数
#      （n_sft/n_grpo/n_dapo + pass_at_1 三连 + pinned_sha），日志 grep 只作人读摘要。
#   F4 三臂同一把尺子证明：三臂跑完后 python 核对 per_bug keyset 全等，不等 → 拒删
#      dapo_merged（保留补测）+ F56_CHAIN_FAILED；SFT 臂 n 不全即停（不烧后两臂 GPU）。
#   F5 GFAIL 从 report 取（续跑 nothing-to-do 早退不重印日志、report 不被覆盖且含全部续跑行）；
#      三个 eval log 与 merge log 改 >> 追加 + 时间戳分隔，保住首轮证据。
# 纪律：
#   ① 专属 --tag f56_*（base/sft/grpo/dapo 四个 tag 是 Task 2.2/3/5 冻结证据，禁止复用；
#      eval_synthetic 的 foreign-row fail-fast 会兜底拒绝，但别去撞它）；
#   ② 磁盘串行（overlay 30G，一次只容一个 15G 合并权重）；
#   ③ 绝不删：sft_merged、d4j_cache*、grpo_out、dapo_out、任何数据/证据文件；
#   ④ 补测配方：某臂个别条件 env_error/failed → 从 syneval_f56_<arm>.jsonl 删掉对应 bug_id 行
#      （f56_* 是本链新证据非冻结件）→ 同 tag 重跑该臂，续跑只补缺行（无需重新合并权重）。
# 前置：syn_eval_frozen56.jsonl 已定格（49 条，A2-A9 绿，独立复核 ALL PASS）且生成进程已收工
#      ——链开跑后冻结集绝不再动（尺子使用中禁止改刻度；快照隔离让这条纪律可被机器验证）。
# 用法：cd /root/rl_training/rl && nohup bash f56_eval_chain.sh > /root/f56_chain.log 2>&1 < /dev/null &
# 标记：F56_ARM_SFT rc= / F56_ARM_GRPO rc= / F56_MERGE_DAPO rc= / F56_ARM_DAPO rc=
#      / F56_EVAL_CHAIN_DONE rc1= rc2= rc3= n_cond= n_sft= n_grpo= n_dapo= pass_*= golden_fail=
#        eval_json_changed= pinned_sha=（任一环节硬失败 → F56_CHAIN_FAILED_* 并 exit 1）
set -u
PY=/root/miniconda3/bin/python3
SWIFT=/root/autodl-tmp/envs/swift/bin/swift
EVAL_JSON=/root/autodl-tmp/sft_data/syn_eval_frozen56.jsonl
REPORT_SRC=/root/autodl-tmp/sft_data/frozen56_report.json
RES=/root/autodl-tmp/baseline_results
OUT=/root/f56_eval
mkdir -p "$OUT"

echo "[f56] chain start $(date -u +%FT%TZ) (v2)"

# ---- step 0a: CPU/GPU 独占检查（放在快照之前——先确认无并发进程，再钉尺子，消除 wc/替换竞态）----
if ps aux | grep -q '[s]tage_frozen56'; then
  echo "F56_CHAIN_FAILED stage_frozen56 still running——串行纪律，等它收工再点火"
  exit 1
fi
if ps aux | grep -q '[e]val_synthetic'; then
  echo "F56_CHAIN_FAILED eval_synthetic already running——一链一实例，先收工再点火"
  exit 1
fi

# ---- step 0b: 钉尺子（F1）——私有快照 + provenance 对账 ----
[ -s "$EVAL_JSON" ] || { echo "F56_CHAIN_FAILED no_eval_json: $EVAL_JSON"; exit 1; }
[ -s "$REPORT_SRC" ] || { echo "F56_CHAIN_FAILED no_frozen56_report: $REPORT_SRC"; exit 1; }
SNAP="$OUT/frozen56_pinned.jsonl"
cp "$EVAL_JSON" "$SNAP"
SHA=$(sha256sum "$SNAP" | cut -d' ' -f1)
N_COND=$(wc -l < "$SNAP")
echo "[f56] pinned snapshot: $SNAP n=$N_COND sha256=$SHA"
grep -q '"failed": false' "$REPORT_SRC" \
  || { echo "F56_CHAIN_FAILED eval_json_provenance_mismatch: 生成报告 failed!=false（上次生成是失败收场）"; exit 1; }
REP_SHA=$($PY -c "import json;print(json.load(open('$REPORT_SRC'))['final_sha256'])" 2>/dev/null) \
  || { echo "F56_CHAIN_FAILED eval_json_provenance_mismatch: 报告不可解析"; exit 1; }
[ "$REP_SHA" = "$SHA" ] \
  || { echo "F56_CHAIN_FAILED eval_json_provenance_mismatch: 报告 final_sha256=$REP_SHA != 快照 $SHA（尺子被改写或报告与文件不同源）"; exit 1; }
REP_ROWS=$($PY -c "import json;print(json.load(open('$REPORT_SRC'))['final_rows'])" 2>/dev/null) || REP_ROWS=-1
[ "$N_COND" -eq "$REP_ROWS" ] \
  || { echo "F56_CHAIN_FAILED snapshot_rows=$N_COND != report final_rows=$REP_ROWS（快照半截？）"; exit 1; }
$PY -c "
import json
r = json.load(open('$REPORT_SRC'))
print('[f56] provenance: final_rows=%s kept=%s/%s A1_ok=%s under_quota=%s' % (
    r['final_rows'], r['kept_total'], r['target_total'],
    r['assertions']['A1_per_bug_exact']['ok'], r.get('under_quota_bugs')))
print('[f56] 注意：终评大表按上面 under_quota 如实报告覆盖度（当前 49 条 / 13 of 14 val bug，Lang-42=0）')
"

# ---- step 0c: 权重/checkpoint 预验（F2：全部在任何 rm 之前）----
[ -d /root/autodl-tmp/sft_merged ] || { echo "F56_CHAIN_FAILED no_sft_merged"; exit 1; }
[ -d /root/grpo_merged ] || { echo "F56_CHAIN_FAILED no_grpo_merged（可先从 grpo_out/v*/checkpoint-150 重合）"; exit 1; }
GCKPT=$(ls -d /root/autodl-tmp/grpo_out/v*/checkpoint-150 2>/dev/null | tail -1)
[ -n "$GCKPT" ] || { echo "F56_CHAIN_FAILED no_grpo_ckpt150——拒绝后续 rm grpo_merged（不可重合就不删）"; exit 1; }
DCKPT=$(ls -d /root/dapo_out/v*/checkpoint-150 2>/dev/null | tail -1)
[ -n "$DCKPT" ] || { echo "F56_CHAIN_FAILED no_dapo_ckpt150——拒绝进入腾磁盘段"; exit 1; }
echo "[f56] weights OK: sft_merged + grpo_merged 在位; GCKPT=$GCKPT DCKPT=$DCKPT"
df -h / | tail -1

cd /root/rl_training/scripts || { echo "F56_CHAIN_FAILED no_scripts_dir"; exit 1; }
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

# ---- 每臂证据闸（F2/F3/F4/F5）：权威是 report.json，不是 rc、不是日志 grep ----
arm_gate() {  # $1=arm(sft|grpo|dapo)  $2=是否要求 golden 全开(1/0)
  local arm=$1 want_golden=$2
  local REP="$RES/syneval_f56_${arm}.report.json"
  if [ ! -s "$REP" ]; then
    echo "F56_CHAIN_FAILED no_report:${arm}（nothing-to-do 早退不写 report——首跑必须产出；或臂根本没跑成）"
    return 1
  fi
  $PY - "$REP" "$arm" "$N_COND" "$want_golden" <<'PYGATE'
import json, sys
rep, arm, n_cond, want_golden = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
r = json.load(open(rep))
bad = []
if r["n"] != n_cond: bad.append(f"n={r['n']}!={n_cond}")
if r["harness_error_count"]: bad.append(f"harness_errors={r['harness_error_bugs']}")
if r["failed_this_run"]: bad.append(f"failed_this_run={r['failed_this_run']}")
if want_golden:
    if r["golden_control_checked"] != n_cond: bad.append(f"golden_checked={r['golden_control_checked']}!={n_cond}")
    if r["golden_control_failed"]: bad.append(f"golden_failed={r['golden_control_failed']}")
print(f"[f56] {arm} report: n={r['n']}/{n_cond} pass_at_1={r['pass_at_1']} fixed={r['fixed']} "
      f"harness_err={r['harness_error_count']} failed={len(r['failed_this_run'])} "
      f"golden_checked={r['golden_control_checked']} golden_failed={len(r['golden_control_failed'])}")
if bad:
    print("F56_CHAIN_FAILED %s_evidence_incomplete: %s" % (arm, "; ".join(bad)))
    sys.exit(1)
PYGATE
  return $?
}

# ---- step 1: SFT 臂（golden 全开 = 效度门；评的是钉住的快照）----
echo "[f56] arm SFT start $(date -u +%FT%TZ)"
{
  echo "===== chain run $(date -u +%FT%TZ) ====="
  $PY eval_synthetic.py --model-dir /root/autodl-tmp/sft_merged --tag f56_sft \
    --eval-json "$SNAP"
} >> "$OUT/eval_sft.log" 2>&1
rc1=$?
echo "F56_ARM_SFT rc=$rc1 $(date -u +%FT%TZ)"
grep -E "^SYNEVAL_DONE" "$OUT/eval_sft.log" | tail -1 | sed 's/^/[f56] sft marker: /'
[ $rc1 -eq 0 ] || { echo "F56_CHAIN_FAILED sft_arm rc=$rc1（看 $OUT/eval_sft.log 尾部）"; exit 1; }
arm_gate sft 1 || { echo "F56_CHAIN_FAILED sft_evidence——尺子不全/效度门未过，不烧 GRPO/DAPO 的 GPU；补测配方见脚本头注释④"; exit 1; }
# F5：GFAIL 从 report 取（日志会被重跑覆盖/早退不重印；report 不会且含全部续跑行）
GFAIL=$($PY -c "import json;print(len(json.load(open('$RES/syneval_f56_sft.report.json'))['golden_control_failed']))")
echo "[f56] golden_control_failed=$GFAIL/$N_COND（arm_gate 已硬闸=此处必 0；行留作人读摘要）"

# ---- step 2: GRPO 臂（golden 关——判 harness 不判模型，SFT 臂已在同 49 世界验过）----
echo "[f56] arm GRPO start $(date -u +%FT%TZ)"
{
  echo "===== chain run $(date -u +%FT%TZ) ====="
  $PY eval_synthetic.py --model-dir /root/grpo_merged --tag f56_grpo \
    --eval-json "$SNAP" --no-golden-control
} >> "$OUT/eval_grpo.log" 2>&1
rc2=$?
echo "F56_ARM_GRPO rc=$rc2 $(date -u +%FT%TZ)"
grep -E "^SYNEVAL_DONE" "$OUT/eval_grpo.log" | tail -1 | sed 's/^/[f56] grpo marker: /'
[ $rc2 -eq 0 ] || { echo "F56_CHAIN_FAILED grpo_arm rc=$rc2——grpo_merged 保留不删，供诊断/重跑"; exit 1; }
arm_gate grpo 0 || { echo "F56_CHAIN_FAILED grpo_evidence_incomplete——grpo_merged 保留不删；补测：删 out 文件对应行后同 tag 重跑本臂（续跑只补缺行）"; exit 1; }

# ---- step 3: 腾磁盘 → 合并 DAPO checkpoint-150（~35min）----
# F2 顺序：GCKPT/DCKPT 已在 step 0c 预验 + GRPO 臂证据闸已过 → 现在才允许破坏性 rm
echo "[f56] GRPO 臂证据闸通过；checkpoint-150 在 $GCKPT 可重合 → rm grpo_merged"
rm -rf /root/grpo_merged
df -h / | tail -1
echo "[f56] merge $DCKPT -> /root/dapo_merged $(date -u +%FT%TZ)"
{
  echo "===== chain run $(date -u +%FT%TZ) ====="
  $SWIFT export --adapters "$DCKPT" --merge_lora true \
    --output_dir /root/dapo_merged
} >> "$OUT/merge_dapo.log" 2>&1 \
  || { echo "F56_CHAIN_FAILED merge_dapo（看 $OUT/merge_dapo.log）——grpo_merged 已删可从 $GCKPT 重合；dapo_merged 残骸可 rm 重试"; exit 1; }
if ! grep -q '"chat_template"' /root/dapo_merged/tokenizer_config.json; then
  cp /root/autodl-tmp/sft_merged/tokenizer_config.json /root/dapo_merged/tokenizer_config.json
  echo "[f56] TEMPLATE_COPIED (坑#32)"
else
  echo "[f56] TEMPLATE_PRESENT"
fi
echo "F56_MERGE_DAPO rc=0 $(date -u +%FT%TZ)"

# ---- step 4: DAPO 臂 ----
echo "[f56] arm DAPO start $(date -u +%FT%TZ)"
{
  echo "===== chain run $(date -u +%FT%TZ) ====="
  $PY eval_synthetic.py --model-dir /root/dapo_merged --tag f56_dapo \
    --eval-json "$SNAP" --no-golden-control
} >> "$OUT/eval_dapo.log" 2>&1
rc3=$?
echo "F56_ARM_DAPO rc=$rc3 $(date -u +%FT%TZ)"
grep -E "^SYNEVAL_DONE" "$OUT/eval_dapo.log" | tail -1 | sed 's/^/[f56] dapo marker: /'
[ $rc3 -eq 0 ] || { echo "F56_CHAIN_FAILED dapo_arm rc=$rc3——dapo_merged 保留供诊断/补测（重合成本 35min）"; exit 1; }
arm_gate dapo 0 || { echo "F56_CHAIN_FAILED dapo_evidence_incomplete——dapo_merged 保留不删；补测：删 out 文件对应行后同 tag 重跑本臂"; exit 1; }

# ---- step 5: 三臂同一把尺子终验（F4：per_bug keyset 全等 + n 三连）----
$PY - <<'PYX'
import json
res = "/root/autodl-tmp/baseline_results"
reps = {a: json.load(open(f"{res}/syneval_f56_{a}.report.json")) for a in ("sft", "grpo", "dapo")}
keys = {a: set(r["per_bug"]) for a, r in reps.items()}
ns = {a: r["n"] for a, r in reps.items()}
print("[f56] cross-arm: n =", ns, "| per_bug keyset sizes =", {a: len(k) for a, k in keys.items()})
base = keys["sft"]
diffs = {a: {"sft_only": sorted(base - k), "arm_only": sorted(k - base)} for a, k in keys.items() if k != base}
if diffs or len(set(ns.values())) != 1:
    print("F56_VALIDITY_WARN 三臂条件集/分母不一致:", diffs, ns)
    raise SystemExit(1)
print("[f56] 三臂 per_bug keyset 全等 = 同一把尺子证明 OK; pass_at_1 =",
      {a: r["pass_at_1"] for a, r in reps.items()})
PYX
[ $? -eq 0 ] || { echo "F56_CHAIN_FAILED cross_arm_mismatch——dapo_merged 保留供补测；三臂 report 已在盘可逐条 diff"; exit 1; }

# ---- step 6: 收尾腾磁盘（三臂证据闸+keyset 全过才走到这里；checkpoint-150 保留，重合成本 ~35min）----
rm -rf /root/dapo_merged
df -h / | tail -1
# F1：链尾复查源冻结集是否在链期间被改写（快照已隔离影响；此行如实记录不裁决）
CHANGED=0
SHA_END=$(sha256sum "$EVAL_JSON" | cut -d' ' -f1)
[ "$SHA_END" = "$SHA" ] || CHANGED=1
N1=$($PY -c "import json;print(json.load(open('$RES/syneval_f56_sft.report.json'))['n'])")
N2=$($PY -c "import json;print(json.load(open('$RES/syneval_f56_grpo.report.json'))['n'])")
N3=$($PY -c "import json;print(json.load(open('$RES/syneval_f56_dapo.report.json'))['n'])")
P1=$($PY -c "import json;print(json.load(open('$RES/syneval_f56_sft.report.json'))['pass_at_1'])")
P2=$($PY -c "import json;print(json.load(open('$RES/syneval_f56_grpo.report.json'))['pass_at_1'])")
P3=$($PY -c "import json;print(json.load(open('$RES/syneval_f56_dapo.report.json'))['pass_at_1'])")
echo "F56_EVAL_CHAIN_DONE rc1=$rc1 rc2=$rc2 rc3=$rc3 n_cond=$N_COND n_sft=$N1 n_grpo=$N2 n_dapo=$N3 pass_sft=$P1 pass_grpo=$P2 pass_dapo=$P3 golden_fail=$GFAIL eval_json_changed=$CHANGED pinned_sha=$SHA $(date -u +%FT%TZ)"
