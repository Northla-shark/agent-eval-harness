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

    注意两件事：
     1. 这里的"实际"是逐步累加的真实 usage，"预估"用的是同一个 profile 的
        **逐步**预估 —— 所以偏差反映的是 profile 与当前场景的失配程度，
        不是预估器算错了。
     2. 它是**规划口径**：默认"一步一次调用"，不知道一步到底打了几次模型。
        项目管理粒度真正的 per-call 偏差看 `metrics.json` 里的 `llm.est_mean_rel_error`
        （由 LLMGate 逐次记账得出）—— 那一个数才含调用粒度与 payload 大小两个信号。
        两个数并存是有意的：规划值用来定预算，per-call 值用来评价预估器。
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


def run_cost(
    episodes: list[EpisodeResult],
    *,
    profile_name: str = "no_history",
    saved_tokens: int = 0,
) -> dict[str, Any]:
    """Cost 层汇总。

    saved_tokens 必须进来：**省下的和花掉的要在同一张表里**。
    只报 tokens_total 的话，开缓存 = 成本下降 = 看起来像模型变便宜了；
    而这两种改进的工程量差一个数量级，混在一起就没法归因。
    """
    n = len(episodes)
    total = token_total(episodes)
    return {
        "tokens_total": total,
        "tokens_per_driver_mean": round(total / n, 1) if n else 0.0,
        "tokens_per_driver_median": (
            statistics.median([e.usage_total.total for e in episodes]) if n else 0.0
        ),
        # 缓存命中而没发出的 token。它**不计入** tokens_total（确实没花），
        # 但必须单独一行 —— 否则"省了多少"这件事在报告里消失。
        "tokens_saved_by_cache": int(saved_tokens),
        "cache_saving_rate": (
            round(saved_tokens / (total + saved_tokens), 4)
            if (total + saved_tokens) else 0.0
        ),
        "estimate_deviation": estimate_deviation(episodes, profile_name),
    }
