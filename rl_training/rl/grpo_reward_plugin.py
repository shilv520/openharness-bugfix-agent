#!/usr/bin/env python3
"""
ms-swift GRPO 外部奖励插件（ORM 适配器）— Task 3

注册机制（ms-swift 3.12.6，plan §3.3 已对照 v3.12.6 源码核实）：
  subclass `swift.plugin.ORM` + `orms['d4j_synthetic'] = Cls`。
加载 / 选用：
  swift rlhf --rlhf_type grpo ... \
      --external_plugins /root/rl_training/rl/grpo_reward_plugin.py \
      --reward_funcs d4j_synthetic
调用契约（_compute_rewards_per_func, L327-361）：
  - `completions` = 已解码字符串 list（messages[-1]['content']）；
  - 数据集每个额外列作为 keyword arg 传入，值是与 completions **下标对齐的 LIST**
    （GRPO 强制 remove_unused_columns=False + identity_data_collator）；
  - `__init__` 必须**零参**（_prepare_rewards 会内省并注入匹配的 training args）；
  - **绝不返回 None**（None→torch.nan 且不被 sanitize）——core 已保证返回等长同序 float list。

薄适配：真正逻辑在 grpo_reward_core（framework-agnostic，可被裸 trl 适配器复用）。本文件只做
两件事：(1) ms-swift ORM 接线；(2) 环境兜底（JAVA_HOME/PATH/sys.path）——`swift rlhf` 子进程
未必继承 defects4j 所需的 java/git PATH（plan §8 pitfall #1c：env 没导好 → 每次判定 subprocess
层失败 → 被洗成 0.0，看着像「模型不会修」）。fallback 到裸 trl 时另写 grpo_reward_trl.py。
"""
import os
import sys
from typing import List

# ---- 环境兜底：defects4j 子进程需要 java + git/perl 在 PATH ----
_D4J_BIN = os.environ.get("D4J_BIN", os.path.expanduser("~/defects4j/framework/bin/defects4j"))
_D4J_DIR = os.path.dirname(_D4J_BIN)
# JAVA_HOME：优先 GRPO_JAVA_HOME（启动命令显式给，pre-flight #5b 确认实际路径），否则用继承值，
# 否则不硬设（避免写死一个错路径盖掉系统正确的 java）。
if os.environ.get("GRPO_JAVA_HOME"):
    os.environ["JAVA_HOME"] = os.environ["GRPO_JAVA_HOME"]
_JAVA_BIN = os.path.join(os.environ["JAVA_HOME"], "bin") if os.environ.get("JAVA_HOME") else ""
_PREPEND = os.pathsep.join(p for p in (_D4J_DIR, _JAVA_BIN) if p)
if _PREPEND:
    os.environ["PATH"] = _PREPEND + os.pathsep + os.environ.get("PATH", "")

# ---- sys.path：ms-swift import_external_file 只把插件**自己所在目录**加进来。
# core 在 rl/、reward.py 在 reward/、（如需）eval_synthetic 在 scripts/ —— 都显式补上。
_RL_DIR = os.path.dirname(os.path.abspath(__file__))            # rl_training/rl
_RL_ROOT = os.path.dirname(_RL_DIR)                             # rl_training
for _p in (_RL_DIR, _RL_ROOT, os.path.join(_RL_ROOT, "reward"), os.path.join(_RL_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from swift.plugin import ORM, orms                               # noqa: E402
from grpo_reward_core import build_recs, grade_batch             # noqa: E402


class D4JSyntheticReward(ORM):
    """合成 Defects4J 执行奖励：completion → 提取 diff → git apply → compile → `test -r` → 分级表。

    返回与 completions 等长同序的 float（GRPO 奖励契约）。所有故障在 core 内被分类/重试/计数，
    harness 故障率 >5% 时 core 抛异常杀训练（绝不静默把基础设施故障当模型差）。
    """

    def __init__(self):                                         # 零参（ms-swift _prepare_rewards 内省）
        pass

    def __call__(self, completions, **kwargs) -> List[float]:
        comps: List[str] = []
        for c in completions:
            # ms-swift 正常传纯字符串；防御性兼容对话格式 [{'role':..,'content':..}]（裸 trl 会这样）
            if isinstance(c, str):
                comps.append(c)
            elif isinstance(c, list) and c and isinstance(c[-1], dict):
                comps.append(c[-1].get("content", "") or "")
            elif isinstance(c, dict):
                comps.append(c.get("content", "") or "")
            else:
                comps.append("" if c is None else str(c))
        recs = build_recs(len(comps), kwargs)                   # 下标对齐 zip + 硬断言无 None
        return grade_batch(recs, comps)


orms["d4j_synthetic"] = D4JSyntheticReward
