# -*- coding: utf-8 -*-
"""Token 预估器：为限流器的预扣提供 estimated_tokens。

这个模块的特殊之处：它不是设计出来的，是从历史数据里拟合出来的。
系数来自项目一 14 组历史跑批（tools/calibrate_token_model.py，
已实测复现：S=3,499 / K=-0.4 / 误差 +0.9% 等）。

核心公式：总 token ≈ n·S + K·n(n−1)/2 + n·C

关键洞察：K 是否为 0 决定成本对步数是线性还是超线性 ——
把"不累积历史"的配置代进累积模型的系数，同样 88 步成本从 30.4 万涨到 64.9 万，差 2.1 倍。
这是"为什么要控制循环轮数"最硬的量化理由（对应 Q17）。

对应面试题：Q19（预算是什么、如何预估 token 和费用）、Q23（TPM 下如何提前估算）。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenProfile:
    """一组校准好的系数。切换上下文策略 / 模型 / 架构时换 profile。"""

    name: str
    fixed_per_step: int        # S：每步固定开销（system + 工具描述 + 状态注入）
    growth_per_step: float     # K：每步累积增量，0 表示上下文不累积
    completion_per_step: int   # C：单次输出
    p90_factor: float = 1.3    # 保守预留倍数

    def describe(self) -> str:
        kind = "超线性（含 n² 项）" if self.growth_per_step > 5 else "线性"
        return f"{self.name}: S={self.fixed_per_step}, K={self.growth_per_step}, C={self.completion_per_step} → {kind}"


# 系数来自项目一历史跑批的重新拟合，不是拍脑袋。
PROFILES: dict[str, TokenProfile] = {
    "no_history": TokenProfile("no_history", 3499, 0.0, 18),
    "history_5": TokenProfile("history_5", 5458, 57.6, 23),
    "history_reasoning": TokenProfile("history_reasoning", 5376, 88.9, 1949),
}


def estimate_task_tokens(
    steps: int, profile: TokenProfile, *, safe: bool = True
) -> int:
    """估算一次任务的 token 总消耗。

    safe=True 按 P90 预留 —— 按期望值预留的话，一半任务会超预算。
    这是"预算"和"预测"的区别：预算要保证不漏，所以必须取上分位。
    """
    total = steps * profile.fixed_per_step
    total += profile.growth_per_step * steps * (steps - 1) / 2
    total += steps * profile.completion_per_step
    return int(total * (profile.p90_factor if safe else 1.0))


def estimate_step_tokens(step_index: int, profile: TokenProfile) -> int:
    """单步预估：供限流器逐次预扣使用（比整任务预估更细粒度）。

    注意 K 项在单步里是 growth_per_step * step_index：
    第 i 步的输入里已经带了前面 i 步的历史，所以步索引越大、单步越贵。
    """
    return int(
        profile.fixed_per_step
        + profile.growth_per_step * step_index
        + profile.completion_per_step
    )


def is_superlinear(profile: TokenProfile) -> bool:
    """K>0 时成本含 n² 项 —— 这是「控制循环轮数」最硬的量化理由。"""
    return profile.growth_per_step > 5
