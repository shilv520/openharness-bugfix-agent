# -*- coding: utf-8 -*-
"""reward 包公共 API 重导出。

历史坑（Task 2.1 official 阶段首跑暴露）：reward/ 曾缺 __init__.py，是个命名空间包。
当「rl_training 根在 sys.path」时，`from reward import BugMeta` 会把这个目录当命名空间
包，而 BugMeta 实际在 reward/reward.py 里 → ImportError: cannot import name 'BugMeta'
from 'reward' (unknown location)。official_one 导入 run_baseline_api（其顶层用了短路径
`from reward import BugMeta, RewardExecutor`）时触发，导致 72/72 官方对全 error。

本文件把 reward.reward 的公共符号提升为包级 API：短路径 `from reward import X` 与
全路径 `from reward.reward import X` 两种写法都可用；对「在 reward/ 目录内直跑单测、
reward 解析成 reward.py 文件」的旧上下文也无害（文件级模块优先级更高）。
"""
from .reward import (  # noqa: F401
    BugMeta,
    RewardExecutor,
    RewardResult,
    TestOutcome,
    apply_variants,
    check_untouched,
    extract_patch,
    farm_evaluate,
    git_apply_ladder,
    grade_reward,
    is_test_path,
    match_failed_triggers,
    normalize_java_eol,
    parse_failing_tests,
    snapshot_test_hashes,
    touched_paths,
    validate_patch_safety,
    validate_paths_safety,
    # 下划线辅助：prompt_builder 单测直跑回退路径会用到
    _clean_diff_text,
    _find_raw_diff,
    _looks_like_diff,
)

__all__ = [
    "BugMeta", "RewardExecutor", "RewardResult", "TestOutcome",
    "apply_variants", "check_untouched", "extract_patch", "farm_evaluate",
    "git_apply_ladder", "grade_reward", "is_test_path", "match_failed_triggers",
    "normalize_java_eol", "parse_failing_tests", "snapshot_test_hashes",
    "touched_paths", "validate_patch_safety", "validate_paths_safety",
    "_clean_diff_text", "_find_raw_diff", "_looks_like_diff",
]
