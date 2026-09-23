# -*- coding: utf-8 -*-
"""AgentEval Harness —— 评测、可观测与 A/B 实验平台。

包名统一用 evalharness。早期设计文档里出现过 harness/，已作废。
"""
from __future__ import annotations

__version__ = "0.1.0"

from .types import (
    Action,
    AgentMeta,
    EnvStep,
    EpisodeResult,
    Observation,
    StepResult,
    Usage,
    ZERO_USAGE,
)

__all__ = [
    "__version__",
    "Action",
    "AgentMeta",
    "EnvStep",
    "EpisodeResult",
    "Observation",
    "StepResult",
    "Usage",
    "ZERO_USAGE",
]
