# -*- coding: utf-8 -*-
"""Process 层：过程指标 —— 本项目最核心的原创部分。★★

为什么它是核心：终局收益任何比赛脚本都能给。
只有过程指标能告诉你「Agent 是不是把算力花在了等待上」，
而这直接定位到要改哪一行。

本文件只给了 1 个示例指标（action_distribution），其余 3 个是留给你写的。
写不出来的话，这个项目就退化成"一个能跑评测的脚本"，而不是"能归因的评测平台"。
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from ..types import EpisodeResult, StepResult

# 罚分归因的三类"成因"。它是 penalty_signal_ratio 的键空间 ——
# 键必须有穷且固定，否则两条跑批的罚分构成无法逐项相减。
#   impossible        : 基准长度内数学上不可能达成（如"整月至少 N 天"跑 1 天基准）
#   structural        : 跨日窗口承诺，需要提前数日规划，短基准内结构性冲突
#   decision_quality  : 当日动作即可满足/违反 —— 唯一真正由决策质量决定的那部分
PENALTY_CAUSES: tuple[str, ...] = ("impossible", "structural", "decision_quality")


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
def signature_counts(steps: Iterable[StepResult]) -> dict[str, int]:
    """各「(动作, 关键参数)」签名的出现次数，降序。

    它是 action_distribution 的细化版，也是 loop_signature_rate 的**对账工具**：
    震荡率只给一个 0~1 的比率，而"到底是哪个签名在重复"要看这张表。
    项目一 A 组的对账结果：`wait(duration=60)` 出现 146 次（428 步中）。
    """
    counter = Counter(_signature_of(s) for s in steps)
    return dict(counter.most_common())


def loop_signature_rate(steps: Iterable[StepResult], window: int = 8) -> float:
    """震荡率 = 滑窗内重复出现过的「(动作, 关键参数)」签名占比。

    规格：
      - 输入：episode 的 steps，按顺序
      - 输出：0.0~1.0 的浮点
      - 判定：第 i 步的 signature 只要出现在 [i-window, i) 区间内，就算一次重复
      - 用 StepResult.info["signature"]（已在 Runner 里写好，形如 "wait(duration=60)"）

    为什么不只看"动作重复"：Agent 连续 wait 是正常的。
    真正的问题是【同一参数】反复出现 —— 项目一里 A 组有 146 次 duration=60，
    那代表它没有从上次尝试中获得任何新信息。这才是"迷路"的指纹。

    实测口径（项目一）：A 组 428 步里出现 146 次完全相同的 wait(60)。

    实现要点（面试会问）：只记"每个签名最后一次出现的步号"就够了 ——
    若最近一次都落在窗外，更早的必然也在窗外，所以不需要保留整个窗口的队列。
    """
    sigs = [_signature_of(s) for s in steps]
    if not sigs:
        return 0.0
    last_seen: dict[str, int] = {}
    repeats = 0
    for i, sig in enumerate(sigs):
        prev = last_seen.get(sig)
        # prev 是"最近一次"，它在窗口内 == 该签名在 [i-window, i) 内出现过
        if prev is not None and i - prev <= window:
            repeats += 1
        last_seen[sig] = i
    return round(repeats / len(sigs), 4)


def constraint_precision(
    steps: Iterable[StepResult],
    *,
    feasible_actions_key: str = "feasible_actions",
) -> float:
    """约束精度 = Agent 主动放弃的可行动作占比。

    ★ 本仓库选 **B：放弃实现**（不是"没写完"）。理由见 docs/metrics.md 第 5 节，
      一句话版本：这个指标需要环境在每一步给出"当时哪些动作合法"，
      赛方环境不提供该接口；选 A（在 Adapter 里自己推算一份可行集）意味着
      用**我们自己猜的规则**去评价 Agent 有没有违反约束 —— 结论会被假数据污染，
      而且污染是静默的（数字很好看，没人会发现它是编的）。

    所以这里保留的是**契约，不是算法**：
      - 环境若真的在 StepResult.info[feasible_actions_key] 里给了可行集 -> 正常计算；
      - 没给 -> 返回 nan（调用方应落成 None），**绝不推断、绝不构造**。

    Returns:
        (可行但未选的动作数) / (可行动作总数)，无可用信息时返回 float("nan")
    """
    total = 0
    abandoned = 0
    for s in steps:
        feasible = s.info.get(feasible_actions_key)
        if not feasible:
            continue
        if s.action.name not in feasible:
            # 选了集合外的动作：那是"违规"不是"放弃"，不计入本指标口径
            continue
        total += len(feasible)
        abandoned += len(feasible) - 1
    if not total:
        return float("nan")
    return round(abandoned / total, 4)


def penalty_signal_ratio(
    penalty_by_cause: dict[str, float],
    *,
    optimizable: Iterable[str] = ("decision_quality",),
) -> float:
    """罚分分层信噪比 = 真正由决策质量决定的罚分 / 总罚分。

    这个指标最重要的是它带来的那个结论：
      项目一里同一次失败实验，1 天基准上可优化信号只占 2.6%，
      30 天基准上是 67.2%，相差 25.5 倍。
      => 我用几周时间在一个信噪比只有 2.6% 的基准上做实验。

    这条结论比指标本身值钱：它说明「指标本身也有信噪比问题」。

    Args:
        penalty_by_cause: {成因 -> 罚分金额}，键取自 PENALTY_CAUSES。
        optimizable: 哪些成因算"可优化"。默认只有 decision_quality ——
            impossible 与 structural 两类是基准长度造成的，改 Agent 也救不回来。

    Returns:
        0.0~1.0；无罚分或无可优化项时返回 nan（不是 0.0 ——
        「没有罚分」和「罚分全是噪声」是两件事，混成 0 会让人误以为是后者）。
    """
    wanted = set(optimizable)
    total = sum(v for v in penalty_by_cause.values() if v > 0)
    if not total:
        return float("nan")
    optimizable_sum = sum(
        v for k, v in penalty_by_cause.items() if k in wanted and v > 0
    )
    return round(optimizable_sum / total, 4)


def _signature_of(step: StepResult) -> str:
    """取一步的签名。info 里没有就现算 —— 历史跑批回放时 info 是空的。"""
    sig = step.info.get("signature")
    return sig if isinstance(sig, str) and sig else step.action.signature()


# ===========================================================================
def run_process(
    episodes: list[EpisodeResult],
    *,
    window: int = 8,
    penalty_by_cause: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Process 层的汇总入口。

    Args:
        penalty_by_cause: 罚分分层。**它不是从 steps 算得出来的** ——
            罚分成因来自环境的约束检查器，不是动作序列。
            所以它由调用方（Runner / 历史跑批插件）单独喂进来，
            不传就是 None（而不是 0.0）。
    """
    all_steps = [s for e in episodes for s in e.steps]
    out: dict[str, Any] = {
        "action_distribution": action_distribution(all_steps),
        "calls_per_success": calls_per_success(all_steps),
        "steps_total": len(all_steps),
        "steps_per_driver_median": _median([e.n_steps for e in episodes]),
        "loop_signature_rate": loop_signature_rate(all_steps, window),
        "penalty_signal_ratio": penalty_signal_ratio(penalty_by_cause) if penalty_by_cause else None,
        "constraint_precision": _none_if_nan(constraint_precision(all_steps)),
        # 震荡指纹：给"到底哪个签名在重复"留证据，否则比率无从对账
        "top_signatures": dict(list(signature_counts(all_steps).items())[:5]),
    }
    # penalty_signal_ratio 也可能是 nan（有罚分但一分都不可优化）
    if out["penalty_signal_ratio"] is not None:
        out["penalty_signal_ratio"] = _none_if_nan(out["penalty_signal_ratio"])
    return out


def _none_if_nan(x: float) -> float | None:
    """nan -> None。

    为什么不能直接留 nan：json.dumps 默认会写出 `NaN`，那是**非法 JSON**，
    下游（pandas / jq / 前端）解析会直接炸；而"没有这个数据"的正确表示是 null。
    """
    return None if isinstance(x, float) and math.isnan(x) else x


def _median(xs: list[int]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return float(s[mid]) if n % 2 else (s[mid - 1] + s[mid]) / 2
