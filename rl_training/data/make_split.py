"""
Task 1.2 — 生成 Defects4J 子集 split（train/val/test）

只读元数据（active-bugs.csv + trigger_tests + patches/*.src.patch + modified_classes），
不需要 project_repos（git 仓库本体），Windows 本地即可跑。

收录标准（三件套齐全才收）：
  ① active-bugs.csv 中有 buggy/fixed revision
  ② trigger_tests/<vid> 存在且能解析出至少 1 个触发测试
  ③ patches/<vid>.src.patch 存在且非空

划分原则：
  - 按项目分层（每个项目都出现在 train/val/test 中）
  - 随机种子固定（可复现）
  - 产出后 test split 锁死：其官方 src.patch 永不进入任何训练数据

用法：
  python rl_training/data/make_split.py [--projects Lang,Time,Chart] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent                # rl_training/data
REPO_ROOT = HERE.parents[1]                           # 项目根目录
DEFECTS4J_HOME_DEFAULT = REPO_ROOT / "data" / "defects4j"

DEFAULT_PROJECTS = ["Lang", "Time", "Chart"]          # 构建最快的子集；顺利后可加 Mockito/Gson/Codec
TEST_RATIO = 0.25
VAL_RATIO = 0.13


def parse_trigger_tests(trigger_file: Path) -> list[str]:
    """trigger_tests/<vid> 格式：'--- class::method' 行 + 堆栈行。只取测试名。"""
    tests = []
    if not trigger_file.exists():
        return tests
    for line in trigger_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("--- "):
            name = line[4:].strip()
            if "::" in name:
                tests.append(name)
    return tests


def load_project_bugs(d4j_home: Path, project: str) -> list[dict]:
    proj_dir = d4j_home / "framework" / "projects" / project
    csv_path = proj_dir / "active-bugs.csv"
    if not csv_path.exists():
        print(f"[WARN] {project}: active-bugs.csv 不存在，跳过")
        return []

    records = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            vid_raw = (row.get("bug.id") or "").strip()
            if not vid_raw.isdigit():
                continue
            vid = int(vid_raw)

            src_patch = proj_dir / "patches" / f"{vid}.src.patch"
            if not src_patch.exists() or src_patch.stat().st_size == 0:
                continue  # 无官方补丁，不收

            triggers = parse_trigger_tests(proj_dir / "trigger_tests" / str(vid))
            if not triggers:
                continue  # 无触发测试 = 无法构造可验证 reward，不收

            # 注意：三个项目的 modified_classes 文件名都带 .src 后缀（如 10.src），
            # 保留无后缀查找作为兜底
            mod_classes_file = proj_dir / "modified_classes" / f"{vid}.src"
            if not mod_classes_file.exists():
                mod_classes_file = proj_dir / "modified_classes" / str(vid)
            mod_classes = (
                mod_classes_file.read_text(encoding="utf-8", errors="replace").split()
                if mod_classes_file.exists() else []
            )

            records.append({
                "bug_id": f"{project}-{vid}",
                "project": project,
                "vid": vid,
                "jira": row.get("report.id", ""),
                "buggy_rev": row.get("revision.id.buggy", ""),
                "fixed_rev": row.get("revision.id.fixed", ""),
                "trigger_tests": triggers,
                "n_trigger": len(triggers),
                "modified_classes": mod_classes,
                "src_patch_rel": str(src_patch.relative_to(d4j_home)).replace("\\", "/"),
            })
    return records


def stratified_split(records: list[dict], seed: int) -> dict[str, list[dict]]:
    """按项目分层，固定种子。返回 {'train': [...], 'val': [...], 'test': [...]}"""
    rng = random.Random(seed)
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    by_project: dict[str, list[dict]] = {}
    for r in records:
        by_project.setdefault(r["project"], []).append(r)

    for project in sorted(by_project):
        items = sorted(by_project[project], key=lambda x: x["vid"])
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * TEST_RATIO))
        n_val = max(1, round(n * VAL_RATIO))
        n_train = n - n_test - n_val
        if n_train < 1:  # 项目太小则全给 train，牺牲 val
            n_train, n_val, n_test = n - 1, 1, 0
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]

    for v in splits.values():
        v.sort(key=lambda x: x["bug_id"])
    return splits


def main():
    # Windows 控制台（GBK）中文输出乱码修正
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--d4j-home", default=str(DEFECTS4J_HOME_DEFAULT))
    ap.add_argument("--projects", default=",".join(DEFAULT_PROJECTS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d4j_home = Path(args.d4j_home)
    projects = [p.strip() for p in args.projects.split(",") if p.strip()]

    all_records = []
    for p in projects:
        recs = load_project_bugs(d4j_home, p)
        print(f"  {p}: 收录 {len(recs)} 条（三件套齐全）")
        all_records += recs

    if not all_records:
        print("[ERROR] 一条都没收到。检查 data/defects4j/framework/projects 是否存在。")
        return

    splits = stratified_split(all_records, args.seed)

    meta = {
        "created": str(date.today()),
        "seed": args.seed,
        "projects": projects,
        "d4j_home": str(d4j_home),
        "counts": {k: len(v) for k, v in splits.items()},
        "lock_notice": "test split 的官方 src.patch 永不进入任何训练数据；split 一经下游使用不再变更",
    }

    out_json = HERE / "split.json"
    out_json.write_text(json.dumps({"meta": meta, **splits}, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    out_csv = HERE / "manifest.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "bug_id", "project", "vid", "jira", "n_trigger"])
        for name in ("train", "val", "test"):
            for r in splits[name]:
                w.writerow([name, r["bug_id"], r["project"], r["vid"], r["jira"], r["n_trigger"]])

    print("\n=== split 汇总 ===")
    for name in ("train", "val", "test"):
        per_proj = {}
        for r in splits[name]:
            per_proj[r["project"]] = per_proj.get(r["project"], 0) + 1
        print(f"  {name:5s}: {len(splits[name]):3d}  {per_proj}")
    print(f"\n产出: {out_json}\n      {out_csv}")


if __name__ == "__main__":
    main()
