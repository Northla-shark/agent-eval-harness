# -*- coding: utf-8 -*-
"""Outcome 层：终局指标。

这一层最容易做，也最容易骗自己 —— 因为它只有一个标量。
它的作用是提供"锚"，不是提供"解释"。
"""
from __future__ import annotations

from typing import Any, Iterable

from ..types import EpisodeResult, StepResult

# 哪些动作算"成功业务动作"。这个集合必须来自领域定义，
# 而不是"看起来像成功的"。
BUSINESS_ACTIONS: frozenset[str] = frozenset({"take_order"})


def reward_total(steps: Iterable[StepResult]) -> float:
    return sum(s.reward for s in steps)


def business_action_count(steps: Iterable[StepResult]) -> int:
    return sum(1 for s in steps if s.action.name in BUSINESS_ACTIONS)


def episode_outcome(episode: EpisodeResult) -> dict[str, Any]:
    """一个 driver 的终局结果。"""
    n_orders = business_action_count(episode.steps)
    return {
        "driver_id": episode.driver_id,
        "reward_total": round(episode.reward_total, 2),
        "business_actions": n_orders,
        "reward_per_action": round(episode.reward_total / n_orders, 2) if n_orders else 0.0,
        "stop_reason": episode.stop_reason,
    }


def run_outcome(episodes: list[EpisodeResult]) -> dict[str, Any]:
    """聚合到 run 级。

    注意这里同时给了 mean 和 median：项目一里"每司机步数中位 88、范围 59~104"
    这种分布信息，比均值更能说明问题 —— 均值会把长尾司机藏起来。
    """
    if not episodes:
        return {"n_drivers": 0}
    rewards = sorted(e.reward_total for e in episodes)
    n = len(rewards)
    mid = n // 2
    median = rewards[mid] if n % 2 else (rewards[mid - 1] + rewards[mid]) / 2
    return {
        "n_drivers": n,
        "reward_mean": round(sum(rewards) / n, 2),
        "reward_median": round(median, 2),
        "reward_min": round(rewards[0], 2),
        "reward_max": round(rewards[-1], 2),
        "business_actions": sum(business_action_count(e.steps) for e in episodes),
    }
