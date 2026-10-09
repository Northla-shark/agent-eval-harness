# -*- coding: utf-8 -*-
"""Token 预估器：为限流器的预扣提供 estimated_tokens。

这个模块的特殊之处：它不是设计出来的，是从历史数据里拟合出来的。
系数来自项目一 10 组历史跑批（tools/calibrate_token_model.py，
已实测复现：S=3,499 / K=-0.4 / 误差 +0.9% 等）。

核心公式：总 token ≈ n·S + K·n(n−1)/2 + n·C

关键洞察：K 是否为 0 决定成本对步数是线性还是超线性 ——
把"不累积历史"的配置代进累积模型的系数，同样 88 步成本从 30.4 万涨到 64.9 万，差 2.1 倍。
这是"为什么要控制循环轮数"最硬的量化理由（对应 Q17）。

D2 的合并：S/K/C（按步拟合）与 payload 字符粗估，之前是两套互不相干的口径
--------------------------------------------------------------------------
    预估器用途        旧公式                       缺什么
    BudgetGuard 预算   n·S + K·n(n−1)/2 + n·C     不知道这次 payload 多大，只能给均值
    LLMGate 预扣       max(S, chars//2) + C        不知道第几步，丢了 K（上下文膨胀项）

两笔账算的是同一个东西（这次请求要花多少 token），却给出差好几倍的答案 ——
`estimate_deviation` 在 truck 跑批上到过 -83.87%，根因在这里，不是"系数不准"。

合并后的单次预估 `estimate_call_tokens()`：
    prompt = max(S + K·step_index, chars // CHARS_PER_TOKEN)
    total  = prompt + C

**为什么是 max 而不是相加**：payload 里通常已经含了累积的历史，
再叠一次 K·i 会把同一段历史算两遍。max 的含义是"两个独立估计里取更保守的那个" ——
预算宁可高估（压低吞吐），也不能低估（击穿 TPM 上限）。

对应面试题：Q19（预算是什么、如何预估 token 和费用）、Q23（TPM 下如何提前估算）。
"""
from __future__ import annotations

from dataclasses import dataclass

# 字符 -> token 的粗估系数（中英混排约 2 字符 1 token）。
# 放在这里而不是散落在调用处：这是**口径常量的单一出处**，
# LLMGate、stub、成本归因三处必须用它，否则又会变成两套口径。
CHARS_PER_TOKEN = 2


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


def step_input_tokens(step_index: int, profile: TokenProfile) -> float:
    """第 i 步的**输入侧**规模，由 S/K 曲线给出 —— 不需要知道 payload。

    K 项在单步里是 growth_per_step * step_index：
    第 i 步的输入里已经带了前面 i 步的历史，所以步索引越大、单步越贵。
    """
    return profile.fixed_per_step + profile.growth_per_step * step_index


def estimate_step_tokens(
    step_index: int,
    profile: TokenProfile,
    *,
    payload_chars: int | None = None,
    calls_per_step: int = 1,
) -> int:
    """单步预估：供限流器逐次预扣 / BudgetGuard 预算使用。

    payload_chars 给了 -> 走**合并口径**（max(S+K·i, chars//2)），知道这次 payload 多大；
    没给        -> 只有步索引信息，退化为 S+K·i 的均值预估（预算场景下这是唯一能做的）。

    calls_per_step 是 D2 补的一个真问题：的默认实现假设"一步一次调用"，
    但项目一一步里打 3 次（agent1 / agent3 / agent4），
    所以按步预估天然少算 3 倍 —— 这也是旧 truck 跑批偏差 -83.87% 的一半来源。
    """
    per_call = estimate_call_tokens(payload_chars, step_index, profile)
    return int(per_call * max(1, calls_per_step))


def estimate_call_tokens(
    payload_chars: int | None,
    step_index: int,
    profile: TokenProfile,
) -> int:
    """★ D2 合并后的单次预估：同时知道「第几步」和「这次 payload 多大」。

    这是限流器 TPM 预扣的唯一入口 —— 预估偏小会把桶打穿，偏大会压低吞吐，
    所以两个信号都给时用 max（保守），两个都没有时才退化为纯 S/K 曲线。
    """
    floor = step_input_tokens(step_index, profile)
    if payload_chars is None:
        prompt = floor
    else:
        prompt = max(floor, payload_chars / CHARS_PER_TOKEN)
    return int(prompt + profile.completion_per_step)


def is_superlinear(profile: TokenProfile) -> bool:
    """K>0 时成本含 n² 项 —— 这是「控制循环轮数」最硬的量化理由。"""
    return profile.growth_per_step > 5
