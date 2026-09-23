# -*- coding: utf-8 -*-
"""Process 层：过程指标 —— 本项目最核心的原创部分。★★

为什么它是核心：终局收益任何比赛脚本都能给。
只有过程指标能告诉你「Agent 是不是把算力花在了等待上」，
而这直接定位到要改哪一行。

本文件只给了 1 个示例指标（action_distribution），其余 3 个是留给你写的。
写不出来的话，这个项目就退化成"一个能跑评测的脚本"，而不是"能归因的评测平台"。
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from ..types import EpisodeResult, StepResult


# ===========================================================================
# 示例指标（已实现，作为你的写法参照）
# ===========================================================================
def action_distribution(steps: Iterable[StepResult]) -> dict[str, float]:
    """各类动作的占比。

    它是"每单调用次数"的姊妹指标：一个告诉你钱花在哪类动作上，
    一个告诉你每做成一件业务要付多少次模型调用。
    """
    step_list = list(steps)
    if not step_list:
        return {}
    counter = Counter(s.action.name for s in step_list)
    n = len(step_list)
    # 降序返回，方便直接打印
    return {k: round(v / n, 4) for k, v in counter.most_common()}


def calls_per_success(steps: Iterable[StepResult]) -> float:
    """每成功业务动作消耗的模型调用次数 —— 成本效率的核心口径。

    项目一真实数据：A 组 13.38 vs B 组 1.86（差 7.2 倍）。
    这个数字比"总 token"有用得多，因为总 token 受步数影响，
    而它回答的是"效率"。
    """
    from .outcome import BUSINESS_ACTIONS

    step_list = list(steps)
    success = sum(1 for s in step_list if s.action.name in BUSINESS_ACTIONS)
    return round(len(step_list) / success, 3) if success else float("inf")


# ===========================================================================
# TODO(D4)：以下三个是你要自己写的指标。函数签名和验收口径都给了。
# ===========================================================================
def loop_signature_rate(steps: Iterable[StepResult], window: int = 8) -> float:
    """震荡率 = 滑窗内重复出现过的「(动作, 关键参数)」签名占比。

    TODO(D4) 你要实现它。规格：
      - 输入：episode 的 steps，按顺序
      - 输出：0.0~1.0 的浮点
      - 判定：第 i 步的 signature 只要出现在 [i-window, i) 区间内，就算一次重复
      - 用 StepResult.info["signature"]（已在 Runner 里写好，形如 "wait(duration=60)"）

    为什么不只看"动作重复"：Agent 连续 wait 是正常的。
    真正的问题是【同一参数】反复出现 —— 项目一里 A 组有 146 次 duration=60，
    那代表它没有从上次尝试中获得任何新信息。这才是"迷路"的指纹。

    实测口径（项目一）：A 组 428 步里出现 146 次完全相同的 wait(60)。

    提示：20 行以内能写完。不要引库。
    """
    raise NotImplementedError("TODO(D4)：实现震荡率，见函数 docstring")


def constraint_precision(
    steps: Iterable[StepResult],
    *,
    feasible_actions_key: str = "feasible_actions",
) -> float:
    """约束精度 = Agent 主动放弃的可行动作占比。

    TODO(D4) 你要实现它，但**先想清楚一个前提**：
      这个指标需要环境在每一步告诉你"当时有哪些动作是合法的"。
      项目一的赛方环境不提供这个接口。
      所以你有两个选择，二选一，并在 docs/metrics.md 里写明你选了哪个和为什么：
        A) 在 Adapter 的 env_step 里自己算一份可行集，塞进 StepResult.info
        B) 放弃这个指标，改用 loop_signature_rate + 罚分分层信噪比

    这个"先想清楚前提"的过程本身就是面试答案：
    「我原本想算 constraint_precision，但它需要环境暴露可行集，
      我的环境不提供，硬算出来会用假数据污染结论，所以我换成了 X。」

    Returns:
        (可行但未选的动作数) / (可行动作总数)，无可用信息时返回 float("nan")
    """
    raise NotImplementedError("TODO(D4)：先决定 A/B 方案，再实现（见 docstring）")


def penalty_signal_ratio(
    penalty_by_cause: dict[str, float],
    *,
    optimizable: Iterable[str] = ("decision_quality",),
) -> float:
    """罚分分层信噪比 = 真正由决策质量决定的罚分 / 总罚分。

    TODO(D4) 你要实现它。这个指标最重要的是它带来的那个结论：
      项目一里同一次失败实验，1 天基准上可优化信号只占 2.6%，
      30 天基准上是 67.2%，相差 25.5 倍。
      => 我用几周时间在一个信噪比只有 2.6% 的基准上做实验。

    这条结论比指标本身值钱：它说明「指标本身也有信噪比问题」。
    """
    raise NotImplementedError("TODO(D4)：实现罚分分层信噪比")


# ===========================================================================
def run_process(episodes: list[EpisodeResult], *, window: int = 8) -> dict[str, Any]:
    """Process 层的汇总入口。已实现的两个指标在这里，另外三个等你补齐后加进来。"""
    all_steps = [s for e in episodes for s in e.steps]
    out: dict[str, Any] = {
        "action_distribution": action_distribution(all_steps),
        "calls_per_success": calls_per_success(all_steps),
        "steps_total": len(all_steps),
        "steps_per_driver_median": _median([e.n_steps for e in episodes]),
        # 下面三个指标补齐后取消注释：
        # "loop_signature_rate": loop_signature_rate(all_steps, window),
        # "penalty_signal_ratio": penalty_signal_ratio(...),
        # "constraint_precision": constraint_precision(all_steps),
    }
    return out


def _median(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2
