#!/usr/bin/env python3
"""
Task 1 · 节点 E — 平台强模型 API 基线跑分（终版大表的"参考上限"行）

流程：split.json 的 test split → 缓存工作区取 buggy 源码 → prompt_builder 统一模板
     → OpenAI 兼容 API → 模型输出喂 RewardExecutor 真实判定 → 汇总报告。
产出：baseline_report.json（Task 1 交付物"基线报告"）+ 逐条 jsonl（可审计/断点续跑）。

在 AutoDL 实例上跑（需要 defects4j 环境 + ~/d4j_cache）：
  export OPENAI_BASE_URL="https://api.deepseek.com/v1"     # 任意 OpenAI 兼容端点
  export OPENAI_API_KEY="sk-..."
  python rl_training/scripts/run_baseline_api.py --model deepseek-chat --limit 2   # 冒烟
  python rl_training/scripts/run_baseline_api.py --model deepseek-chat             # 正式 28 条

常用端点（选一个即可，成本都在几块钱以内）：
  DeepSeek   https://api.deepseek.com/v1                        deepseek-chat
  DashScope  https://dashscope.aliyuncs.com/compatible-mode/v1  qwen-max / qwen3-coder-plus
  火山方舟   https://ark.cn-beijing.volces.com/api/v3           doubao-seed-1-6-250615 等

说明：
  - test split 首次跑会现场 checkout+compile 回填 ~/d4j_cache（每条 1-3 min，一次性成本；
    之后 Task 2.4 评 base/SFT 复用同一份缓存）。建议 tmux/nohup 里跑。
  - 断点续跑：结果按行写 jsonl，重跑自动跳过已完成的 bug_id。
  - temperature 默认 0（基线要确定性）；prompt 与 SFT/评测共用 prompt_builder 唯一事实源。
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent            # rl_training/scripts
RL_ROOT = HERE.parent                             # rl_training
sys.path.insert(0, str(RL_ROOT / "reward"))
sys.path.insert(0, str(RL_ROOT / "data"))

import requests  # noqa: E402

from prompt_builder import SYSTEM_PROMPT, build_prompt, patch_to_bug_context  # noqa: E402
from reward import BugMeta, RewardExecutor  # noqa: E402

_WRITE_LOCK = threading.Lock()


# ---------------------------------------------------------------- prompt 侧

def load_buggy_code(ws: Path, rec: dict, d4j_home: Path, max_lines: int) -> tuple[str, str]:
    """返回 (code, source)。source ∈ {workspace, patch_context}。
    优先从工作区读 modified_classes 的完整源码（正式基线用这个）；
    找不到才降级用官方 patch 的 buggy 侧片段（仅冒烟，会标记出来）。

    ⚠️ max_lines 是硬预算（逐文件切片，不是文件间检查）：D4J 有数千行的巨类，
    旧实现只在"文件之间"检查预算 → 第一个大文件就冲破上限（实测 prompt 飙到
    16k tokens，4090 上 prefill 直接 OOM；SFT 序列预算也只有 2048-4096）。
    SFT / RL / 评测三处共用本函数 → 上下文口径全链路一致。"""
    files: list[str] = []
    n_lines = 0
    for cls in rec.get("modified_classes", []):
        if max_lines and n_lines >= max_lines:
            files.append("// ... (truncated, more modified classes omitted)")
            break
        rel = cls.replace(".", "/") + ".java"
        fname = rel.rsplit("/", 1)[-1]
        for cand in ws.rglob(fname):
            if cand.as_posix().endswith(rel):
                try:
                    text = cand.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                lines = text.splitlines()
                budget = (max_lines - n_lines) if max_lines else 0
                if budget and len(lines) > budget:
                    text = ("\n".join(lines[:budget])
                            + "\n// ... (file truncated to fit context budget)")
                # ⚠️ 标签必须是工作区真实相对路径，不能用类名推导路径（坑 #19）：
                # JFreeChart 实际在 source/org/jfree/...（非标准 Maven 布局），
                # 模型忠实按标签路径打补丁 → git apply 找不到文件 → Chart 全系假 0。
                actual_rel = cand.relative_to(ws).as_posix()
                files.append(f"// File: {actual_rel}\n{text}")
                n_lines += text.count("\n") + 2
                break
    if files:
        return "\n\n".join(files), "workspace"
    # 降级：官方 src.patch 的 buggy 侧（含 hunk 级定位，比 modified_classes 更强，
    # 正式基线数字必须用 workspace 来源，这里只为把链路跑通）
    patch = d4j_home / rec.get("src_patch_rel", "")
    if patch.exists():
        return patch_to_bug_context(patch.read_text(encoding="utf-8", errors="replace")), "patch_context"
    return "(no code available)", "missing"


def call_api(base_url: str, api_key: str, model: str, user_prompt: str,
             temperature: float, max_tokens: int, timeout: int = 180,
             max_retries: int = 3) -> tuple[str, float]:
    last_err: Exception | None = None
    for i in range(max_retries):
        try:
            t0 = time.time()
            r = requests.post(
                f"{base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"] or ""
            return content, round(time.time() - t0, 2)
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(3 * (2 ** i))
    raise RuntimeError(f"API 调用失败（重试 {max_retries} 次）: {last_err}")


# ---------------------------------------------------------------- 单条流水线

def run_one(rec: dict, cfg: argparse.Namespace) -> dict:
    ex_kwargs = dict(defects4j_bin=cfg.d4j_bin, cache_dir=cfg.cache_dir,
                     run_dir=cfg.run_dir, timeout_s=cfg.timeout_s)
    executor = RewardExecutor(**ex_kwargs)
    bug = BugMeta.from_split_record(rec)

    # 取 buggy 源码（缓存缺失时 prepare_workspace 会现场 checkout+compile 并回填）
    ws = executor.prepare_workspace(bug)
    try:
        code, code_source = load_buggy_code(ws, rec, Path(cfg.d4j_home), cfg.max_code_lines)
    finally:
        shutil.rmtree(ws, ignore_errors=True)

    prompt = build_prompt(rec, code)
    t_api, api_latency = call_api(cfg.base_url, cfg.api_key, cfg.model, prompt,
                                  cfg.temperature, cfg.max_tokens)
    result = executor.evaluate(bug, t_api)   # 真实判定：apply → compile → trigger → 回归

    row = {
        "bug_id": rec["bug_id"], "project": rec["project"], "vid": rec["vid"],
        "model": cfg.model, "split": cfg.split, "date": str(date.today()),
        "reward": result.reward, "stage": result.stage, "detail": result.detail,
        "reward_duration_s": result.duration_s, "api_latency_s": api_latency,
        "code_source": code_source, "n_trigger": len(rec.get("trigger_tests", [])),
        "model_output": t_api,
    }
    with _WRITE_LOCK:
        with open(cfg.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


# ---------------------------------------------------------------- 汇总报告

def summarize(rows: list[dict], cfg: argparse.Namespace) -> dict:
    n = len(rows)
    fixed = sum(1 for r in rows if r["reward"] == 1.0)
    partial = sum(1 for r in rows if 0 < r["reward"] < 1)
    mean_reward = round(sum(r["reward"] for r in rows) / n, 4) if n else 0.0
    report = {
        "baseline": "platform_strong_model",
        "model": cfg.model, "split": cfg.split, "n": n, "date": str(date.today()),
        "fix_rate": round(fixed / n, 4) if n else 0.0,
        "fixed": fixed, "partial": partial, "zero": n - fixed - partial,
        "mean_reward": mean_reward,
        "stage_distribution": dict(Counter(r["stage"] for r in rows)),
        "code_source_distribution": dict(Counter(r["code_source"] for r in rows)),
        "avg_reward_duration_s": round(sum(r["reward_duration_s"] for r in rows) / n, 2) if n else 0,
        "avg_api_latency_s": round(sum(r["api_latency_s"] for r in rows) / n, 2) if n else 0,
        "per_bug": {r["bug_id"]: r["reward"] for r in sorted(rows, key=lambda x: x["bug_id"])},
    }
    return report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    home = Path.home()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", required=True, help="API 模型名，如 deepseek-chat / qwen-max")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""),
                    help="OpenAI 兼容端点（默认读环境变量 OPENAI_BASE_URL）")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                    help="默认读环境变量 OPENAI_API_KEY")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--split-json", default=str(RL_ROOT / "data" / "split.json"))
    ap.add_argument("--d4j-bin", default=str(home / "defects4j/framework/bin/defects4j"))
    ap.add_argument("--d4j-home", default=str(home / "defects4j"))
    ap.add_argument("--cache-dir", default=str(home / "d4j_cache"))
    ap.add_argument("--run-dir", default="/tmp/baseline_run")
    ap.add_argument("--out", default="", help="结果 jsonl（默认 ./baseline_results/<model>.jsonl）")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-code-lines", type=int, default=600,
                    help="prompt 中 buggy 源码总行数上限（0=不限）")
    cfg = ap.parse_args()

    if not cfg.base_url or not cfg.api_key:
        sys.exit("缺少 API 配置：export OPENAI_BASE_URL / OPENAI_API_KEY（或 --base-url/--api-key）")
    model_slug = cfg.model.replace("/", "_").replace(":", "_")
    if not cfg.out:
        cfg.out = str(Path("baseline_results") / f"{model_slug}.jsonl")
    Path(cfg.out).parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(Path(cfg.split_json).read_text(encoding="utf-8"))
    records = data[cfg.split]
    if cfg.limit:
        records = records[: cfg.limit]

    # 断点续跑：跳过 jsonl 里已完成的 bug_id
    done: set[str] = set()
    if Path(cfg.out).exists():
        for line in Path(cfg.out).read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["bug_id"])
            except (json.JSONDecodeError, KeyError):
                pass
    todo = [r for r in records if r["bug_id"] not in done]
    print(f"[baseline] model={cfg.model} split={cfg.split} 共 {len(records)} 条，"
          f"已完成 {len(done)}，本次跑 {len(todo)} 条，workers={cfg.workers}")

    rows: list[dict] = []
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futs = {pool.submit(run_one, rec, cfg): rec["bug_id"] for rec in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            bid = futs[fut]
            try:
                row = fut.result()
                rows.append(row)
                print(f"  [{i}/{len(todo)}] {row['bug_id']:12s} reward={row['reward']:<5} "
                      f"stage={row['stage']:32s} reward={row['reward_duration_s']}s "
                      f"api={row['api_latency_s']}s code={row['code_source']}")
            except Exception as e:  # noqa: BLE001
                failed.append(bid)
                print(f"  [{i}/{len(todo)}] {bid:12s} ERROR: {e}")

    # 汇总（含历史已完成行）
    all_rows = rows
    if done:
        for line in Path(cfg.out).read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                if r["bug_id"] in done:
                    all_rows.append(r)
            except json.JSONDecodeError:
                pass
    report = summarize(all_rows, cfg)
    report["failed_this_run"] = failed
    report_path = Path(cfg.out).with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 基线报告（Task 1 交付物） ===")
    print(f"  模型: {report['model']}   split: {report['split']}   n={report['n']}")
    print(f"  fix rate (reward=1.0): {report['fix_rate']:.1%}  "
          f"(fixed={report['fixed']} partial={report['partial']} zero={report['zero']})")
    print(f"  mean reward: {report['mean_reward']}")
    print(f"  stage 分布: {report['stage_distribution']}")
    print(f"  code 来源: {report['code_source_distribution']}"
          + ("   ⚠️ 含 patch_context 降级条目，正式数字需全 workspace"
             if report["code_source_distribution"].get("patch_context") else ""))
    print(f"  reward 平均延迟: {report['avg_reward_duration_s']}s   API 平均延迟: {report['avg_api_latency_s']}s")
    print(f"  报告: {report_path}\n  明细: {cfg.out}")
    print("\n终版大表行（Markdown）:")
    print(f"  | 平台强模型 ({report['model']}) | {report['split']} | {report['n']} "
          f"| {report['fix_rate']:.1%} | {report['mean_reward']} |")
    if failed:
        print(f"\n⚠️ {len(failed)} 条失败: {failed}（重跑本命令自动续）")
        sys.exit(1)


if __name__ == "__main__":
    main()
