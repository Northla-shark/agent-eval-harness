# -*- coding: utf-8 -*-
"""预算相关：预估 + 三重熔断。

拆成两个文件而不是一个，因为它们回答的是两个不同的问题：
  estimator.py —— 「这次大概要花多少」（事前，用于限流预扣与预算预留）
  guard.py     —— 「已经花超了，停」（事中，用于防止失控循环烧钱）
"""
from .estimator import (
    PROFILES,
    TokenProfile,
    estimate_step_tokens,
    estimate_task_tokens,
    is_superlinear,
)
from .guard import Budget, BudgetBreach, BudgetGuard

__all__ = [
    "PROFILES",
    "TokenProfile",
    "estimate_step_tokens",
    "estimate_task_tokens",
    "is_superlinear",
    "Budget",
    "BudgetBreach",
    "BudgetGuard",
]
