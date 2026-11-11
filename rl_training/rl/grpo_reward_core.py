#!/usr/bin/env python3
"""
GRPO 合成奖励核心（framework-agnostic）— Task 3 GRPO reward 接入

为什么单独一个 core，而不是直接复用 reward.py / farm_evaluate（plan §3 + §8 pitfall #1）：

  1. **错世界 = 静默全 0**：farm_evaluate 的 _farm_worker 硬编码 `RewardExecutor`
     （VID_SUFFIX='b'，checkout **buggy** 世界）。合成 bug 的「案发现场」是 **fixed 全绿
     世界 + 重放存储的 mutation**，不是 checkout 出来的 buggy 世界。用 buggy 世界判合成
     补丁 → trigger 测试本来就绿 → reward 恒 0、且**不报错**。本 core 用 VID_SUFFIX='f'
     的 GRPOSyntheticExecutor（镜像 scripts/eval_synthetic.py 的 SyntheticEvalExecutor，
     那套已在 Task 2.2 gate 上被 golden 21/21=1.0 验证过）。

  2. **并发工作区撞车 = 静默错判**：GRPO 每个 generation round 并发判同一 bug_id 的 G 个
     completion。reward.py 的工作区命名是 `{bug_id}_{int(time.time()*1000)}`（毫秒级），
     同 bug_id 同毫秒 → 两个 completion 抢同一个 ws + 同一个 `{ws}.patch` → completion A
     被 B 的补丁打分。本 core 用 **uuid 命名 + 每线程独立 run_dir** 彻底隔离。

  3. **env-error 洗白 = 基础设施故障算到模型头上**：reward.py 把 evaluate 内的
     RuntimeError/OSError 洗成 `detail='env_error:*'` + 0.0 分**正常返回**（这正是
     eval_synthetic.py 修过的 BUG1）。本 core 按 detail 前缀分类 → 2× 超时重试一次 →
     仍故障则计 0.0 并计数；**一轮内 harness 故障率 >5% → 抛异常杀掉训练**（奖励信号
     已系统性不可信，继续跑只是烧 GPU）。

纪律：**零改 reward.py**（改它必须重跑 test_reward_unit.py；本 core 只子类化 + 组合）。
只 import reward.py（纯 stdlib），**不** import eval_synthetic（避免把 torch/模型加载
拖进奖励进程——判分只需 java/git/defects4j 子进程，不碰 GPU）。

环境变量（由 `swift rlhf` 启动命令导出；缺省值见下）：
  D4J_BIN               defects4j CLI（默认 ~/defects4j/framework/bin/defects4j）
  GRPO_CACHE_DIR        全绿 fixed 世界缓存（默认 /root/autodl-tmp/d4j_cache_fixed）
  GRPO_RUN_DIR          工作区根（默认 /root/grpo_run，放**系统盘**——autodl-tmp 只剩 ~6.6G）
  GRPO_REWARD_WORKERS   并发判定线程数（默认 8；16 核 / 每 JVM ≤2G / 64G RAM，对齐 build_sft_data --workers 14 先例）
  GRPO_REWARD_TIMEOUT_S 单次判定超时秒（默认 600）
  GRPO_MIN_FREE_GB      run_dir 所在盘最低空闲 GB（默认 2；低于则不判，防 ENOSPC 写坏缓存）
  GRPO_REWARD_LOG       逐条判定 sidecar（默认 GRPO_RUN_DIR/reward_log.jsonl）—— reward-hacking 遥测

自检（== pre-flight #5 的 live golden 校验，可在 swift venv 下直跑）：
  /root/autodl-tmp/envs/swift/bin/python /root/rl_training/rl/grpo_reward_core.py
  → 读 syn_eval_clean.jsonl 前 3 条，用 golden_patch 判定，必须全 ==1.0，否则管线/环境坏。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import uuid
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---- import reward.py（stdlib-only）。兼容 'reward.reward' 与顶层 'reward' 两种导入约定 ----
_RL_ROOT = Path(__file__).resolve().parent.parent          # rl_training
for _p in (str(_RL_ROOT / "reward"), str(_RL_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
try:
    from reward.reward import BugMeta, RewardExecutor, RewardResult   # rl_training 根在 sys.path
except ImportError:                                                   # reward/ 直接在 sys.path
    from reward import BugMeta, RewardExecutor, RewardResult

# ---- 配置（env 覆盖）----
HOME = Path.home()
D4J_BIN = os.environ.get("D4J_BIN", str(HOME / "defects4j/framework/bin/defects4j"))
CACHE_FIXED = os.environ.get("GRPO_CACHE_DIR", "/root/autodl-tmp/d4j_cache_fixed")
RUN_ROOT = os.environ.get("GRPO_RUN_DIR", "/root/grpo_run")
WORKERS = int(os.environ.get("GRPO_REWARD_WORKERS", "8"))
TIMEOUT_S = int(os.environ.get("GRPO_REWARD_TIMEOUT_S", "600"))
MIN_FREE_GB = float(os.environ.get("GRPO_MIN_FREE_GB", "2"))
REWARD_LOG = os.environ.get("GRPO_REWARD_LOG", str(Path(RUN_ROOT) / "reward_log.jsonl"))
# 熔断窗口：进程级滚动窗口，记最近 N 次判定的 is_env（坑 #37——G=4 小批下单条 env error
# 就是 25%，按批熔断 = 零容忍杀训练，1/4 没有统计意义；系统性烂要靠窗口率抓）。
_ENV_WINDOW: deque = deque(maxlen=int(os.environ.get("GRPO_REWARD_ENV_WINDOW", "100")))

# harness/环境故障的 detail 前缀（reward.py 把 RuntimeError/OSError + 超时/崩溃洗成这些）。
# 这些**不是模型的错**，不能算进 policy 的负优势；rate 超阈值要杀训练。
ENV_DETAIL_PREFIXES = (
    "env_error:", "test_run_timeout", "trigger_timeout", "regression_timeout",
    "test_run_error:", "trigger_run_error:", "regression_run_error:", "checkout failed",
    "disk_precheck_low_free:", "cache_miss:",
)
# reward-hacking / 越界 的 detail 前缀（遥测硬停信号；stage 可能是 patch_apply_failed，靠 detail 区分）
HACK_DETAIL_PREFIXES = ("test_file_touched", "path_outside_allowlist")

# build_recs 需要的列（每行都得带全；ms-swift rows_to_batched 对缺列补 None）
REQUIRED_COLS = ("real_bug_id", "project", "vid", "file_rel", "buggy_src", "trigger_tests")
OPTIONAL_COLS = ("bug_id", "allowed_globs", "mutation", "line_no", "seed_k")


# ══════════════════════════════════════════════════════════════════════
# 合成世界判定器（VID_SUFFIX='f' + uuid 工作区 + 重放 mutation）
# ══════════════════════════════════════════════════════════════════════
class GRPOSyntheticExecutor(RewardExecutor):
    """fixed 全绿世界（缓存副本）+ 重放存储的 mutation（覆写 file_rel=buggy_src）。

    与 scripts/eval_synthetic.py::SyntheticEvalExecutor 同语义（那套已被 golden 21/21 验证），
    但**改了工作区命名**：reward.py 用 {bug_id}_{毫秒}，GRPO 并发判同一 bug_id 会撞车；
    这里换成 {bug_id}_{uuid12}。其余判定全继承 RewardExecutor.evaluate（git apply 重试阶梯、
    EOL 归一、快/慢路径、分级表、reward-hacking 守卫一律不动）。
    """
    VID_SUFFIX = "f"                                        # f = fixed（全绿世界）

    def __init__(self, *args, syn_rec: dict, **kwargs):
        super().__init__(*args, **kwargs)
        self._syn_rec = syn_rec

    def prepare_workspace(self, bug: BugMeta) -> Path:
        cache_ws = self.cache_dir / bug.bug_id              # bug_id == real_bug_id == '{project}-{vid}'
        if not cache_ws.exists():
            # 绝不在训练 step 里冷 checkout（1-3 min，爆超时）；缓存必须预热（pre-flight #8）
            raise RuntimeError(f"cache_miss:{bug.bug_id} (never cold-checkout inside a GRPO step)")
        ws = self.run_dir / f"{bug.bug_id}_{uuid.uuid4().hex[:12]}"   # uuid 防撞车
        shutil.copytree(cache_ws, ws)
        rec = self._syn_rec
        fp = ws / rec["file_rel"]
        if not fp.exists():
            shutil.rmtree(ws, ignore_errors=True)           # 别泄漏半成品工作区
            raise RuntimeError(f"env_error:file_rel not in fixed ws: {rec['file_rel']}")
        # 重放变异：strict utf-8 写回（与 build_sft_data / eval_synthetic 同口径）
        fp.write_bytes(rec["buggy_src"].encode("utf-8"))
        return ws


# ══════════════════════════════════════════════════════════════════════
# 每线程独立 run_dir（叠加 uuid 命名，双保险隔离并发判定）
# ══════════════════════════════════════════════════════════════════════
_tl = threading.local()


def _thread_run_dir() -> Path:
    rd = getattr(_tl, "run_dir", None)
    if rd is None:
        rd = Path(RUN_ROOT) / f"w{threading.get_ident()}"
        rd.mkdir(parents=True, exist_ok=True)
        _tl.run_dir = rd
    return rd


def _is_env_detail(detail: str) -> bool:
    d = str(detail or "")
    return any(d.startswith(p) for p in ENV_DETAIL_PREFIXES)


def _is_hack_detail(detail: str) -> bool:
    d = str(detail or "")
    return any(d.startswith(p) for p in HACK_DETAIL_PREFIXES)


def _bug_meta(rec: dict) -> BugMeta:
    # allowed_globs 缺省 = [file_rel]：inject_bug 的 golden 只动 file_rel，锁死「改坏无关源文件
    # 让敏感测试空过」这个 reward-hacking 面（plan §3.4），零召回损失。数据里若显式给了就用给的。
    globs = rec.get("allowed_globs") or [rec["file_rel"]]
    return BugMeta(
        bug_id=rec["real_bug_id"], project=rec["project"], vid=int(rec["vid"]),
        trigger_tests=list(rec.get("trigger_tests") or []), allowed_globs=list(globs),
    )


@dataclass
class _GradeOutcome:
    reward: float
    stage: str
    detail: str
    is_env: bool
    is_hack: bool
    duration_s: float


def _log(rec: dict, completion: str, outcome: _GradeOutcome) -> None:
    """逐条判定写 sidecar（reward-hacking / env-error 遥测；plan §3.5）。best-effort，不抛。"""
    try:
        Path(REWARD_LOG).parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": round(time.time(), 3),
            "real_bug_id": rec.get("real_bug_id"),
            "reward": outcome.reward, "stage": outcome.stage,
            "detail": outcome.detail[:200], "is_env": outcome.is_env,
            "is_hack": outcome.is_hack, "duration_s": outcome.duration_s,
            "completion_head": (completion or "")[:120],
        }
        with open(REWARD_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _cleanup_stale(run_dir: Path, before: set) -> None:
    """前后快照差清理：删掉本次判定在 run_dir 里新增的**目录和工作区**（含 reward.py 写的
    {ws}.patch 兄弟文件——它的 finally 只 rmtree ws 目录、不删 .patch）。快照差与命名无关，
    比 glob 稳；堵住 env-error 路径上 ws=None、evaluate 自身 cleanup 被跳过导致的泄漏。"""
    try:
        after = set(run_dir.iterdir()) if run_dir.exists() else set()
    except OSError:
        return
    for stale in (after - before):
        try:
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            else:
                stale.unlink(missing_ok=True)
        except OSError:
            pass


def _sweep_old_workspaces(max_age_s: float) -> None:
    """惰性 janitor：清掉 run_dir 下超过 max_age_s 的残留（SIGKILL 泄漏的工作区 / .patch）。
    每个 generation round 调一次（~60-170s），足够勤；不开后台线程（省生命周期麻烦）。"""
    root = Path(RUN_ROOT)
    if not root.exists():
        return
    now = time.time()
    for w in root.glob("w*"):
        for child in w.iterdir() if w.is_dir() else []:
            try:
                if now - child.stat().st_mtime > max_age_s:
                    if child.is_dir():
                        shutil.rmtree(child, ignore_errors=True)
                    else:
                        child.unlink(missing_ok=True)
            except OSError:
                pass


def _grade_one(rec: dict, completion: str) -> _GradeOutcome:
    """判一个 completion。永不返回 None；永不抛（除非要杀训练的 >5% 由 grade_batch 统一判）。"""
    run_dir = _thread_run_dir()
    bug = _bug_meta(rec)

    # 磁盘预检：低于阈值不判（防 ENOSPC 把可恢复故障变成写坏缓存）
    try:
        free_gb = shutil.disk_usage(run_dir).free / 1e9
    except OSError:
        free_gb = 0.0
    if free_gb < MIN_FREE_GB:
        out = _GradeOutcome(0.0, "patch_apply_failed",
                            f"disk_precheck_low_free:{free_gb:.2f}GB<{MIN_FREE_GB}GB",
                            True, False, 0.0)
        _log(rec, completion, out)
        return out

    before = set(run_dir.iterdir()) if run_dir.exists() else set()
    t0 = time.time()
    try:
        ex = GRPOSyntheticExecutor(D4J_BIN, CACHE_FIXED, run_dir,
                                   timeout_s=TIMEOUT_S, syn_rec=rec)
        result = ex.evaluate(bug, completion)
        # env-error → 2× 超时重试一次（flaky/慢基础设施）；仍故障则用原结果（计 0.0 + 计数）。
        # ⚠️ 例外（坑 #38）：timeout 族是**确定性挂死**（模型错 patch 让测试死循环，Lang-6/Lang-30
        #   的 text.translate 家族实测两次），重试必然再挂死一次、白烧 2×TIMEOUT——不重试。
        if _is_env_detail(result.detail) and "timeout" not in str(result.detail):
            try:
                ex2 = GRPOSyntheticExecutor(D4J_BIN, CACHE_FIXED, run_dir,
                                            timeout_s=2 * TIMEOUT_S, syn_rec=rec)
                retry = ex2.evaluate(bug, completion)
                if not _is_env_detail(retry.detail):
                    result = retry
            except Exception:                              # noqa: BLE001 重试本身崩了就用原结果
                pass
    except Exception as e:                                 # noqa: BLE001 判定器构造/缓存缺失等
        result = RewardResult(bug_id=bug.bug_id, reward=0.0, stage="patch_apply_failed",
                              detail=f"env_error:{type(e).__name__}:{e}"[:500], duration_s=0.0)
    finally:
        _cleanup_stale(run_dir, before)

    out = _GradeOutcome(
        reward=float(result.reward), stage=result.stage, detail=str(result.detail or ""),
        is_env=_is_env_detail(result.detail), is_hack=_is_hack_detail(result.detail),
        duration_s=round(time.time() - t0, 2),
    )
    _log(rec, completion, out)
    return out


# ══════════════════════════════════════════════════════════════════════
# 批判定（GRPO 每个 generation round 调一次）
# ══════════════════════════════════════════════════════════════════════
def build_recs(n: int, kwargs: dict) -> list[dict]:
    """把 ms-swift 透传的数据集列（每列是一个与 completions 下标对齐的 LIST）zip 成逐条 rec。

    硬断言无 None：rows_to_batched 对缺失的键补 None，静默 None 会让重放 mutation 写空文件
    → 全 0 reward 且难查。列名陷阱见 plan §3.4（别用 response/answer/output/... 会被当 assistant
    turn 重映射丢弃）。"""
    recs: list[dict] = []
    for i in range(n):
        rec: dict = {}
        for k in REQUIRED_COLS + OPTIONAL_COLS:
            col = kwargs.get(k)
            if col is None:
                if k in REQUIRED_COLS:
                    raise ValueError(
                        f"[grpo-reward] 必需列 '{k}' 整个缺失（reward kwargs 只有 {sorted(kwargs)}）。"
                        f"数据集透传断了？检查 JSONL 列名是否撞了 ms-swift 保留字（plan §3.4）。")
                continue
            val = col[i] if isinstance(col, (list, tuple)) else col
            if val is None and k in REQUIRED_COLS:
                raise ValueError(f"[grpo-reward] 必需列 '{k}' 在下标 {i} 为 None（rows_to_batched 补位？）")
            rec[k] = val
        recs.append(rec)
    return recs


def grade_batch(recs: list[dict], completions: list[str]) -> list[float]:
    """判一批 completion，返回与输入**等长、同序**的 float 列表（GRPO 奖励函数契约）。

    - 去重相同 (real_bug_id, completion)：temperature 1.0 下模型常重复产出，省一次 ~8s 判定。
    - ThreadPoolExecutor 并发（subprocess 释放 GIL，真活是外部 java/ant/git；进程池会在
      CUDA 已初始化的 trainer 里 fork，脆弱——plan §3.5）。
    - 按 key 回填，天然同序（不依赖 future 完成顺序）。
    - 熔断（窗口化，坑 #37）：**最近 100 次判定**的 env 率 >5%（样本 ≥20）→ 抛 RuntimeError
      杀训练（奖励信号系统性不可信）。单批小样本（G=4）下单条 env error 只大声告警不杀。
    """
    n = len(completions)
    if n == 0:
        return []
    if len(recs) != n:
        raise ValueError(f"[grpo-reward] recs({len(recs)}) 与 completions({n}) 不等长")

    _sweep_old_workspaces(max_age_s=3 * TIMEOUT_S)

    # 去重：key=(real_bug_id, completion) -> 出现的所有下标
    key_to_idx: dict[tuple, list[int]] = {}
    for i, (rec, comp) in enumerate(zip(recs, completions)):
        key_to_idx.setdefault((rec["real_bug_id"], comp), []).append(i)
    unique = [(k, recs[v[0]], k[1]) for k, v in key_to_idx.items()]

    outcomes: dict[tuple, _GradeOutcome] = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        fut_to_key = {pool.submit(_grade_one, rec, comp): k for k, rec, comp in unique}
        # 按提交顺序收（其实按 key 回填已与顺序无关，但保持提交序便于读日志/复现）
        for fut in list(fut_to_key.keys()):
            k = fut_to_key[fut]
            outcomes[k] = fut.result()

    # 回填成同序 float 列表
    rewards = [0.0] * n
    for k, idxs in key_to_idx.items():
        r = outcomes[k].reward
        for i in idxs:
            rewards[i] = r

    # 遥测：stage 直方图 + env/hack 计数
    graded = [outcomes[k] for k in key_to_idx]
    env_n = sum(1 for o in graded if o.is_env)
    hack_n = sum(1 for o in graded if o.is_hack)
    hist = Counter(o.stage for o in graded)
    _log_round_summary(n, len(graded), env_n, hack_n, hist, rewards)

    # 熔断（窗口化，坑 #37）：按批的 5% 线在 G=4 下 = 单条 env error 零容忍杀训练——
    # 1/4=25% 没有统计意义（实测：一次 30min 测试挂死就杀掉 50 分钟的全量训练）。
    # 改成滚动窗口率：最近 100 次判定 env 率 >5%（样本 ≥20 才判）→ 系统性烂，loud fail 杀训练。
    # 单批的 env error 仍然大声告警 + 逐条落 sidecar 日志，绝不静默。
    for o in graded:
        _ENV_WINDOW.append(o.is_env)
    win_n = len(_ENV_WINDOW)
    win_env = sum(_ENV_WINDOW)
    if env_n:
        print(f"[grpo-reward] ⚠️ WARNING 本批 env error {env_n}/{len(graded)}"
              f"（detail 见 {REWARD_LOG}）；窗口故障率 {win_env}/{win_n} = {win_env/win_n:.1%}",
              flush=True)
    if win_n >= 20 and win_env / win_n > 0.05:
        raise RuntimeError(
            f"[grpo-reward] 窗口 harness 故障率 {win_env}/{win_n} = {win_env/win_n:.1%} > 5%"
            f"（最近 {win_n} 次判定）。奖励信号系统性不可信（java/defects4j/磁盘/缓存出问题），"
            f"杀训练以免烧 GPU 学噪声。查 {REWARD_LOG} 的 detail 前缀定位。")
    return rewards


def _log_round_summary(n: int, uniq: int, env_n: int, hack_n: int,
                       hist: Counter, rewards: list[float]) -> None:
    """每轮汇总写 sidecar（reward-hacking 早预警：hack_n>0、或 0.3/部分档占比异常上升即停）。"""
    try:
        Path(REWARD_LOG).parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": round(time.time(), 3), "type": "round_summary",
            "n_completions": n, "n_unique_graded": uniq,
            "env_error_count": env_n, "hack_count": hack_n,
            "stage_histogram": dict(hist),
            "reward_mean": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
            "reward_nonzero": sum(1 for r in rewards if r > 0),
            "frac_reward_zero_std_hint": round(
                sum(1 for r in rewards if r == rewards[0]) / len(rewards), 3) if rewards else 0.0,
        }
        with open(REWARD_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass
    # 同时打到 stdout（进 swift 训练日志，便于 tail 监控）
    print(f"[grpo-reward] round n={n} uniq={uniq} env={env_n} hack={hack_n} "
          f"stages={dict(hist)} mean={row['reward_mean']}", flush=True)
    if hack_n:
        print(f"[grpo-reward] ⚠️ {hack_n} 条命中 reward-hacking/越界（test_file_touched / "
              f"path_outside_allowlist）—— 查 {REWARD_LOG}", flush=True)


# ══════════════════════════════════════════════════════════════════════
# 自检（== pre-flight #5 live golden 校验；在 swift venv 下直跑）
# ══════════════════════════════════════════════════════════════════════
def _self_test(eval_json: str = "/root/autodl-tmp/sft_data/syn_eval_clean.jsonl", k: int = 3) -> bool:
    rows = [json.loads(l) for l in
            Path(eval_json).read_text(encoding="utf-8").splitlines() if l.strip()][:k]
    print(f"[self-test] env: D4J_BIN={D4J_BIN} CACHE_FIXED={CACHE_FIXED} RUN_ROOT={RUN_ROOT} "
          f"WORKERS={WORKERS} TIMEOUT_S={TIMEOUT_S}", flush=True)
    print(f"[self-test] 用 golden_patch 判 {len(rows)} 条预热 bug（必须全 ==1.0）...", flush=True)
    ok = 0
    for r in rows:
        rec = {
            "real_bug_id": f"{r['project']}-{r['vid']}", "project": r["project"], "vid": r["vid"],
            "file_rel": r["file_rel"], "buggy_src": r["buggy_src"],
            "trigger_tests": r.get("trigger_tests", []), "allowed_globs": r.get("allowed_globs"),
        }
        o = _grade_one(rec, r["golden_patch"])
        flag = "OK" if o.reward > 0.999 else "❌FAIL"
        print(f"  {rec['real_bug_id']:14s} golden reward={o.reward} stage={o.stage} "
              f"detail={o.detail[:80]} {flag}", flush=True)
        ok += (o.reward > 0.999)
    passed = (ok == len(rows))
    print(f"[self-test] golden {ok}/{len(rows)} == 1.0 → {'SELF_TEST_OK' if passed else 'SELF_TEST_FAILED'}",
          flush=True)
    return passed


if __name__ == "__main__":
    _k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    _json = sys.argv[2] if len(sys.argv) > 2 else "/root/autodl-tmp/sft_data/syn_eval_clean.jsonl"
    sys.exit(0 if _self_test(_json, _k) else 1)
