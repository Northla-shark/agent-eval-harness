# -*- coding: utf-8 -*-
"""Cost 层：token 与延迟归因。

口径要点
--------
1. 不要只看总延迟。要拆 TTFT（首 token 延迟）/ 生成延迟 / 工具执行延迟。
   项目一只有 simulate_time_seconds（3,166s vs 9,281s）—— 没有拆分，
   所以"慢在哪"答不上来。
2. 预估器的精度本身就是一项可观测指标：
   偏差持续为正 => 上下文在膨胀；持续为负 => 预留过度、浪费配额。
"""
from __future__ import annotations

import statistics
from typing import Any

from ..budget.estimator import PROFILES, TokenProfile, estimate_step_tokens
from ..types import EpisodeResult


def token_total(episodes: list[EpisodeResult]) -> int:
    return sum(e.usage_total.total for e in episodes)


def token_per_driver(episodes: list[EpisodeResult]) -> dict[str, int]:
    return {e.driver_id: e.usage_total.total for e in episodes}


def estimate_deviation(
    episodes: list[EpisodeResult], profile_name: str
) -> dict[str, float]:
    """「预估 vs 实际」的偏差分布。

    这是预估器的体检报告：一个预估器如果从不校准，它就是一个装饰品。

    注意：这里的"实际"是逐步累加的真实 usage，
    "预估"用的是同一个 profile 的逐步预估 —— 所以偏差反映的是
    profile 与当前场景的失配程度，而不是预估器算错了。
    """
    profile: TokenProfile = PROFILES[profile_name]
    rel_errors: list[float] = []
    for e in episodes:
        est = sum(estimate_step_tokens(s.step, profile) for s in e.steps)
        act = e.usage_total.total
        if act > 0:
            rel_errors.append((est - act) / act)
    if not rel_errors:
        return {"n": 0}
    return {
        "n": len(rel_errors),
        "mean_rel_error": round(statistics.fmean(rel_errors), 4),
        "stdev_rel_error": round(statistics.pstdev(rel_errors), 4) if len(rel_errors) > 1 else 0.0,
        "max_abs_rel_error": round(max(abs(x) for x in rel_errors), 4),
    }


def run_cost(episodes: list[EpisodeResult], *, profile_name: str = "no_history") -> dict[str, Any]:
    n = len(episodes)
    total = token_total(episodes)
    return {
        "tokens_total": total,
        "tokens_per_driver_mean": round(total / n, 1) if n else 0.0,
        "tokens_per_driver_median": (
            statistics.median([e.usage_total.total for e in episodes]) if n else 0.0
        ),
        "estimate_deviation": estimate_deviation(episodes, profile_name),
    }
