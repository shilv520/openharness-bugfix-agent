#!/usr/bin/env python3
"""
Task 2.2 gate · 合成评测器（syn_eval_clean.jsonl 上 base vs SFT，pass@1）

为什么需要它（坑 #31）：syn_eval 是「合成注入 bug」——把变异注入到 Defects4J 的
**fixed（全绿）版本**，trigger_tests = 被变异打挂的敏感测试，golden_patch = 精确回滚。
现有 run_baseline_local.py 用默认 RewardExecutor（VID_SUFFIX="b"，checkout **buggy**
版），那是真实 bug 的判定世界，**不能**用来判合成 bug：合成 bug 的「案发现场」是
fixed 世界 + 重放 buggy_src，不是 checkout 出来的 buggy 世界。build_sft_data.py 只
**写** syn_eval.jsonl（带 buggy_src/golden_patch，line 20「评测器可精确重建变异工作区」），
但从未实现这个评测器——本脚本补上。

判定机制（镜像 build_sft_data.attempt_one，但**重放存储的变异**而非现生成）：
  1. FixedExecutor.prepare_workspace → checkout {vid}f（全绿世界，缓存 CACHE_FIXED/<real_bug_id>）。
  2. 重放：把 ws/<file_rel> 覆写成记录里的 buggy_src → 精确还原模型被要求修的那个合成 buggy 态。
  3. git apply 模型补丁（复用 reward.py 的重试阶梯）+ `defects4j test -r`。
  4. 复用 reward.py 的分级表打分：trigger（=敏感测试）必须回绿；reward hacking / apply /
     compile 守卫全部继承。

模型加载：load_model = 纯 AutoModelForCausalLM.from_pretrained（无 peft/adapter 分支），
所以 SFT 侧必须喂**合并后的全权重 HF 目录**（/root/autodl-tmp/sft_merged，由 run_merge.sh
用 `swift export --merge_lora` 产出）。base env 无 peft、且是冻结时间胶囊（torch 2.1.2，
不能装包），故走「合并 best checkpoint」而非 adapter-grading；实测磁盘 ~20G 空闲 > 15.2G
合并产物，放得下（合并→打分→如需给 GRPO 腾地方再 rm）。

两类「不该算模型账」的行被显式剔除出 pass@1 总体（pass@1 严格只数 reward==1.0，绝不把
0.3 当过，否则真回归被误判为修复=假 PASS）：
  (a) HARNESS/环境故障：reward.py 把 evaluate 内的 RuntimeError/OSError 洗成
      detail="env_error:*" + grade 0.0/patch_apply_failed 并**正常返回**（坑：这正好落在
      base 模型最常合法产出的 patch_apply_failed 桶里，会误导操作者怪模型）。本脚本按
      detail 前缀把它们拆出来单独上报（harness_error_*），并从 n/pass_at_1 剔除。
  (b) baseline_failing 非空：fixed 世界非全绿，正确修复会被 pre-existing 失败误降为 0.3
      （trigger_pass_regression_fail），永远拿不到 1.0。加载时即剔除（base/SFT 同口径）。
      注：当前 syn_eval_clean.jsonl 21 条 baseline_failing 全空（85/86 绿，Time-10 已按
      run_error 排除），故 (b) 的确定性半边今天是休眠的；但坑 #30 计划重生成更大评测集，
      build_sft_data 会把 fixed 世界 baseline='failing' 的 bug 也纳入，届时本剔除即生效。
      活的风险是 eval 时相关测试套件 flaky 变红（环境已实证有噪声：Time-10 run_error、
      Lang-29/30 因噪声被拒），对正确补丁误降 0.3——故对 trigger_pass_regression_fail
      行做**一次重试**：flaky 恢复 1.0，持续 0.3 则是真回归（保守正确）。

GOLDEN 正对照（默认开，--no-golden-control 关）：每条记录额外用 golden_patch 跑一次判定，
**必须** reward==1.0；否则说明重建/判定管线坏了（不是模型问题），本轮 pass@1 不可信，marker
变脏。这是给载荷门（SFT→GRPO 决策）的廉价保险——曾被管线 bug 伪装成模型失败坑过（坑 #19
JFreeChart source/ 前缀）。golden 判定与模型无关，base 臂跑过即可证管线正确。

模型加载 + 生成与 run_baseline_local.py **完全一致**（HF transformers greedy temp=0，
prompt_builder 唯一模板，code 段 "// File: {rel}\\n{buggy_src}" 与 SFT 训练同格式），
所以 base vs SFT 苹果对苹果，且与 SFT 训练分布对齐。跑在 base env（torch 2.1.2，需 defects4j）。

用法（在实例上，base env，串行——GPU 装不下两个模型）：
  # base 臂（带 golden 正对照，证管线正确）
  nohup /root/miniconda3/bin/python /root/rl_training/scripts/eval_synthetic.py \
      --model-dir /root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct --tag base \
      > /root/syneval_base.log 2>&1 &
  # SFT 臂（合并模型；golden 与模型无关，base 臂已验，可 --no-golden-control 省时）
  nohup /root/miniconda3/bin/python /root/rl_training/scripts/eval_synthetic.py \
      --model-dir /root/autodl-tmp/sft_merged --tag sft --no-golden-control \
      > /root/syneval_sft.log 2>&1 &
标记: SYNEVAL_DONE（干净）/ SYNEVAL_DONE_WITH_FAILS:[...]（生成异常 / harness 故障 / golden 失败）
纪律：无关机/cron/timer 逻辑；不改 reward.py；test split 官方 patch 永不进训练。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import date
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:256")

HERE = Path(__file__).resolve().parent            # rl_training/scripts
RL_ROOT = HERE.parent                             # rl_training
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RL_ROOT / "reward"))
sys.path.insert(0, str(RL_ROOT / "data"))

from prompt_builder import build_prompt                       # noqa: E402
from reward import BugMeta, RewardExecutor                    # noqa: E402
from run_baseline_local import load_model, generate           # noqa: E402

# 与 build_sft_data.py 同值（不 import 它，避免拖入 injector 依赖）
HOME = Path.home()
BIN = str(HOME / "defects4j/framework/bin/defects4j")
CACHE_FIXED = "/root/autodl-tmp/d4j_cache_fixed"   # 全绿世界缓存（build_sft_data Phase 0 构建）
RUN = "/root/autodl-tmp/syn_eval_run"
BASE_MODEL = "/root/autodl-tmp/models/Qwen2.5-Coder-7B-Instruct"


class FixedExecutor(RewardExecutor):
    """checkout {vid}f（全绿世界）。仅借 checkout→EOL归一→compile→缓存回填流水线。"""
    VID_SUFFIX = "f"


class SyntheticEvalExecutor(FixedExecutor):
    """FixedExecutor + 在 prepare_workspace 里**重放存储的变异**（覆写 file_rel=buggy_src），
    使工作区精确处于模型被要求修复的合成 buggy 态。其余判定全继承 RewardExecutor.evaluate。
    缓存命中的是 real_bug_id（project-vid）的全绿世界；重放只作用于 copytree 出的独立 ws 副本，
    绝不污染 CACHE_FIXED。"""

    def __init__(self, *args, syn_rec: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._syn_rec = syn_rec

    def prepare_workspace(self, bug: BugMeta) -> Path:
        ws = super().prepare_workspace(bug)        # 全绿 fixed 世界（缓存副本）
        rec = self._syn_rec
        if rec is None:
            return ws
        fp = ws / rec["file_rel"]
        if not fp.exists():
            raise RuntimeError(f"file_rel not in fixed ws: {rec['file_rel']}")
        # 重放变异：strict utf-8 写回（与 build_sft_data line 233 同口径）
        fp.write_bytes(rec["buggy_src"].encode("utf-8"))
        return ws


def real_bug_id(rec: dict) -> str:
    """syn_eval 的 bug_id 是 'SYN-<project>-<vid>-<seed_k>'；缓存键是真实 'project-vid'。
    Defects4J 约定 bug_id == f'{project}-{vid}'，直接用 project+vid 重建（比字符串切割稳）。"""
    return f"{rec['project']}-{rec['vid']}"


def _is_env(r: dict) -> bool:
    """该行是否由 HARNESS/环境故障产生：reward.py 把 evaluate 内的 RuntimeError/OSError 洗成
    detail='env_error:*' + grade 0.0/patch_apply_failed 并正常返回。这类行不能算模型账。"""
    return str(r.get("detail", "")).startswith("env_error:")


def grade_once(cfg, bug: BugMeta, rec: dict, patch_text: str):
    """每次判定用全新 executor（工作区每次被消耗/清理），并在 finally 里用「前后快照差」
    清掉本次判定在 run_dir 里留下的任何目录——堵住 reward.py 在 env-error 路径上 ws 仍为
    None、自身 cleanup 被跳过导致的工作区泄漏（否则每条故障记录泄漏一个编译好的 Defects4J
    工作区，能撑爆磁盘，把可恢复故障变成整轮失败）。快照差与命名无关，比 glob 稳。"""
    run_dir = Path(cfg.run_dir)
    before = set(run_dir.iterdir()) if run_dir.exists() else set()
    ex = SyntheticEvalExecutor(cfg.d4j_bin, cfg.cache_fixed, cfg.run_dir,
                               timeout_s=cfg.timeout_s, syn_rec=rec)
    try:
        return ex.evaluate(bug, patch_text)
    finally:
        after = set(run_dir.iterdir()) if run_dir.exists() else set()
        for stale in (after - before):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="合成评测器：syn_eval_clean 上 base vs SFT")
    ap.add_argument("--model-dir", default=BASE_MODEL)
    ap.add_argument("--tag", default="", help="结果文件名标签（base / sft / ...）")
    ap.add_argument("--eval-json", default="/root/autodl-tmp/sft_data/syn_eval_clean.jsonl")
    ap.add_argument("--d4j-bin", default=BIN)
    ap.add_argument("--cache-fixed", default=CACHE_FIXED)
    ap.add_argument("--run-dir", default=RUN)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--golden-control", dest="golden_control", action="store_true", default=True,
                    help="每条额外用 golden_patch 判定一次，必须 reward==1.0，否则管线坏（默认开）")
    ap.add_argument("--no-golden-control", dest="golden_control", action="store_false",
                    help="跳过 golden 正对照（golden 与模型无关，base 臂验过即可在 SFT 臂省时）")
    cfg = ap.parse_args()

    tag = cfg.tag or Path(cfg.model_dir).name
    if not cfg.out:
        cfg.out = str(Path("/root/autodl-tmp/baseline_results") / f"syneval_{tag}.jsonl")
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.run_dir).mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in
               Path(cfg.eval_json).read_text(encoding="utf-8").splitlines() if l.strip()]
    # 本评测集全量 id（剔除/limit 之前）——「out 文件里的历史行属不属于本评测集」以此为准。
    # 没有它，沿用 --tag（如 syneval_sft.jsonl 已装 syn_eval_clean 的 21 行）跑冻结集时，
    # 历史行会静默混进汇总（n=77 的混合 pass@1 + golden 效度门被旧行「代打」通过），
    # 还会向冻结证据文件追加行/覆盖 report —— 对抗审查实证复现，见 frozen56 审查记录。
    eval_set_ids = {r["bug_id"] for r in records}
    # BUG#3 fix1：加载即剔除 baseline_failing 非空记录（fixed 世界非全绿，正确修复必被误降 0.3，
    # 永远拿不到 1.0）。base/SFT 同口径、确定性；今天 21 条全空故为 no-op，重生成后即生效。
    excluded_baseline = [r["bug_id"] for r in records if r.get("baseline_failing")]
    records = [r for r in records if not r.get("baseline_failing")]
    if excluded_baseline:
        print(f"[syn-eval] 剔除 {len(excluded_baseline)} 条 baseline_failing 非空记录"
              f"（fixed 世界非全绿，正确修复会被误判 0.3）：{excluded_baseline}", flush=True)
    if cfg.limit:
        records = records[: cfg.limit]

    # 断点续跑：跳过已判过的 synthetic bug_id。done 只认本评测集的行；异集历史行 fail-fast
    # （在加载模型之前、在第一次 append 之前——这正是保护冻结证据文件的关键时点）。
    done: set = set()
    foreign: set = set()
    if Path(cfg.out).exists():
        for line in Path(cfg.out).read_text(encoding="utf-8").splitlines():
            try:
                bid = json.loads(line)["bug_id"]
            except (json.JSONDecodeError, KeyError):
                continue
            (done if bid in eval_set_ids else foreign).add(bid)
    if foreign:
        msg = (f"{cfg.out} 已含 {len(foreign)} 条不属于本评测集({cfg.eval_json})的历史行"
               f"（例 {sorted(foreign)[:3]}）——沿用此 --tag/--out 会把两个评测集混成一个池，"
               f"并向已冻结证据追加行/覆盖其 .report.json")
        if cfg.limit:
            print(f"[syn-eval] [WARN] {msg}；已排除出汇总（--limit 冒烟模式，不阻断）", flush=True)
        else:
            print(f"SYNEVAL_FAIL {msg}；请为本评测集指定专属 --tag（如 f56_{tag}）"
                  f"或显式 --out 新路径", flush=True)
            sys.exit(1)
    todo = [r for r in records if r["bug_id"] not in done]
    print(f"[syn-eval] model={cfg.model_dir} tag={tag} golden_control={cfg.golden_control} "
          f"共 {len(records)} 条，已完成 {len(done)}，本次跑 {len(todo)} 条", flush=True)
    if not todo:
        print("SYNEVAL_DONE (nothing to do)", flush=True)
        return

    print("loading model ...", flush=True)
    tok, model = load_model(cfg.model_dir)

    rows, failed = [], []
    for i, rec in enumerate(todo, 1):
        bid = rec["bug_id"]
        try:
            t0 = time.time()
            # prompt：与 SFT 训练 / run_baseline_local 完全同格式
            code = f"// File: {rec['file_rel']}\n{rec['buggy_src']}"
            prompt = build_prompt(rec, code)
            output, meta = generate(tok, model, prompt, cfg.max_tokens)

            # 判定：real_bug_id 命中全绿缓存 → 重放 buggy_src → apply 模型补丁 → test -r → 打分
            bug = BugMeta(bug_id=real_bug_id(rec), project=rec["project"],
                          vid=int(rec["vid"]), trigger_tests=list(rec.get("trigger_tests", [])),
                          allowed_globs=rec.get("allowed_globs"))
            result = grade_once(cfg, bug, rec, output)

            # BUG#1 fix4：harness（非模型）故障的逐条吵闹信号（行仍落盘，证据留 detail）
            if str(result.detail).startswith("env_error:"):
                print(f"  [{i}/{len(todo)}] {bid:24s} HARNESS_ERROR {result.detail[:160]}",
                      flush=True)

            # BUG#3 fix2：trigger_pass_regression_fail 一次重试——flaky 环境噪声（实证 ~3%）
            # 恢复 1.0；持续 0.3 则是真回归（保守正确）。仅对 0.3 行触发，成本有界。
            tprf_recovered = False
            if result.stage == "trigger_pass_regression_fail":
                retry = grade_once(cfg, bug, rec, output)
                if retry.reward > result.reward:
                    result = retry
                    tprf_recovered = True

            # GOLDEN 正对照：golden_patch 必须 1.0，否则重建/判定管线坏（不是模型问题）。
            golden_reward = golden_stage = None
            if cfg.golden_control and rec.get("golden_patch"):
                gres = grade_once(cfg, bug, rec, rec["golden_patch"])
                golden_reward, golden_stage = gres.reward, gres.stage
                if gres.reward < 0.999:   # golden 必须 ==1.0；阈值避浮点等值
                    print(f"  [{i}/{len(todo)}] {bid:24s} ⚠️ GOLDEN_CONTROL_FAILED "
                          f"reward={gres.reward} stage={gres.stage} detail={str(gres.detail)[:120]}"
                          f" —— 重建/判定管线有问题，不是模型问题！", flush=True)

            row = {
                "bug_id": bid, "real_bug_id": real_bug_id(rec), "project": rec["project"],
                "vid": rec["vid"], "model": Path(cfg.model_dir).name, "tag": tag,
                "date": str(date.today()),
                "reward": result.reward, "stage": result.stage, "detail": result.detail,
                "n_trigger": len(rec.get("trigger_tests", [])),
                "n_baseline_failing": len(rec.get("baseline_failing", [])),
                "mutation": rec.get("mutation"), "line_no": rec.get("line_no"),
                "reward_duration_s": result.duration_s, **meta,
                "total_s": round(time.time() - t0, 2),
                "model_output": output,
                "tprf_recovered_on_retry": tprf_recovered,
                "golden_reward": golden_reward, "golden_stage": golden_stage,
            }
            rows.append(row)
            with open(cfg.out, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            gc = "" if golden_reward is None else f" golden={golden_reward}"
            rec_flag = " RECOVERED" if tprf_recovered else ""
            print(f"  [{i}/{len(todo)}] {bid:24s} reward={row['reward']:<5} "
                  f"stage={row['stage']:32s} trig={row['n_trigger']} "
                  f"base_fail={row['n_baseline_failing']} gen={meta['gen_latency_s']}s "
                  f"reward={result.duration_s}s{gc}{rec_flag}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed.append(bid)
            print(f"  [{i}/{len(todo)}] {bid:24s} ERROR: {e}", flush=True)
            traceback.print_exc()
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()

    # 汇总（只并本评测集的历史行；按 bug_id 去重、本次新行优先——防同一 id 双计）
    merged = {r["bug_id"]: r for r in rows}
    if done and Path(cfg.out).exists():
        for line in Path(cfg.out).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except (json.JSONDecodeError, KeyError):
                continue
            if r.get("bug_id") in done and r["bug_id"] not in merged:
                merged[r["bug_id"]] = r
    all_rows = list(merged.values())

    # BUG#1 fix1 + BUG#3 fix1：把两类「不该算模型账」的行拆出打分总体。
    # pass@1 严格只数 reward==1.0（绝不把 0.3 当过——那会把真回归误判为修复=假 PASS）。
    env_rows = [r for r in all_rows if _is_env(r)]
    baseline_excl = [r for r in all_rows
                     if not _is_env(r) and r.get("n_baseline_failing", 0) > 0]
    graded = [r for r in all_rows
              if not _is_env(r) and r.get("n_baseline_failing", 0) == 0]

    n = len(graded)
    # 混池不变量（必须坐在 report 写盘之前——违例时绝不能覆盖既有证据的 report）：
    # 打分总体永远不可能大于本评测集条数。
    assert n <= len(records), f"汇总总体 n={n} > 本评测集 {len(records)} 条——混入异集历史行"
    fixed = sum(1 for r in graded if r["reward"] > 0.999)   # ==1.0（grade 表上限），阈值避浮点等值
    partial = sum(1 for r in graded if 0 < r["reward"] < 1)
    tprf = sum(1 for r in graded if r.get("stage") == "trigger_pass_regression_fail")
    tprf_recovered = sum(1 for r in graded if r.get("tprf_recovered_on_retry"))
    golden_checked = [r for r in graded if r.get("golden_reward") is not None]
    golden_failed = [r for r in golden_checked if r.get("golden_reward") < 0.999]

    report = {
        "baseline": "synthetic_eval", "model": Path(cfg.model_dir).name, "tag": tag,
        "eval_json": cfg.eval_json, "n": n, "date": str(date.today()),
        "pass_at_1": round(fixed / n, 4) if n else 0.0,
        "fixed": fixed, "partial": partial, "zero": n - fixed - partial,
        "mean_reward": round(sum(r["reward"] for r in graded) / n, 4) if n else 0.0,
        "stage_distribution": dict(Counter(r["stage"] for r in graded)),
        # BUG#1 fix2：harness 故障显式上报（之前只聚合 stage，env_error 证据永远进不了报告）
        "harness_error_count": len(env_rows),
        "harness_error_bugs": [r["bug_id"] for r in env_rows],
        "harness_error_details": {r["bug_id"]: str(r.get("detail", ""))[:200] for r in env_rows},
        # BUG#3：baseline 非空剔除 + tprf 统计
        "excluded_baseline_nonempty": [r["bug_id"] for r in baseline_excl],
        "excluded_baseline_nonempty_at_load": excluded_baseline,
        "trigger_pass_regression_fail_count": tprf,
        "tprf_recovered_on_retry_count": tprf_recovered,
        # golden 正对照：证明重建+判定管线正确（golden 必须 1.0）
        "golden_control_checked": len(golden_checked),
        "golden_control_failed": [r["bug_id"] for r in golden_failed],
        "per_bug": {r["bug_id"]: r["reward"] for r in sorted(graded, key=lambda x: x["bug_id"])},
        "failed_this_run": failed,
    }
    report_path = Path(cfg.out).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 合成评测报告（Task 2.2 gate：base vs SFT，pass@1）===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # BUG#1 fix3：harness 故障必须吵出来，且 marker 不能是干净的 SYNEVAL_DONE
    if env_rows:
        print(f"⚠️⚠️ {len(env_rows)} 条是 HARNESS/环境故障（detail=env_error:*），已从 n/pass_at_1 "
              f"剔除，不是模型问题：{[r['bug_id'] for r in env_rows]}", flush=True)
        for r in env_rows:
            print(f"    {r['bug_id']}: {str(r.get('detail', ''))[:200]}", flush=True)
    # BUG#3 fix3：warn 改 keyed on tprf（活的风险是 eval 时 flaky，不是记录的 baseline），
    # 去掉「见 detail」（0.3 行 detail 恒空——reward.py 从不把回归测试名写进 detail），直接列 bug_id。
    if tprf:
        tprf_ids = [r["bug_id"] for r in graded
                    if r.get("stage") == "trigger_pass_regression_fail"]
        print(f"⚠️ {tprf} 条 trigger_pass_regression_fail（修复但相关测试套件有红）；其中 "
              f"{tprf_recovered} 条经一次重试恢复（=flaky 环境噪声）。仍为 0.3 的：{tprf_ids}"
              f" —— 0.3 不计入 pass@1（避免把真回归当修复）。", flush=True)
    if golden_failed:
        print(f"⚠️⚠️ GOLDEN 正对照失败 {len(golden_failed)} 条："
              f"{[r['bug_id'] for r in golden_failed]} —— golden_patch 都拿不到 1.0，"
              f"说明重建/判定管线坏了，本轮 pass@1 不可信！", flush=True)
    if baseline_excl or excluded_baseline:
        all_excl = sorted(set([r["bug_id"] for r in baseline_excl] + excluded_baseline))
        print(f"ℹ️ 剔除 baseline_failing 非空记录 {len(all_excl)} 条（fixed 世界非全绿）："
              f"{all_excl}", flush=True)

    print(f"报告: {report_path}\n明细: {cfg.out}")
    # marker：生成/判定异常 + harness 故障 + golden 失败，任一非空 → 不干净
    bad = list(failed) + [r["bug_id"] for r in env_rows] + [r["bug_id"] for r in golden_failed]
    print("SYNEVAL_DONE" if not bad else f"SYNEVAL_DONE_WITH_FAILS:{bad}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("SYNEVAL_FAIL", flush=True)
        sys.exit(1)
