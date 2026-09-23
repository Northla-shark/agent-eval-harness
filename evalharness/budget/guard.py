# -*- coding: utf-8 -*-
"""三维预算熔断：token / 步数 / 墙钟。

为什么是三维而不是一维
--------------------
借鉴 Inspect AI 的 Limits 分层，但多了一层最关键的：

  token  = 成本维度        —— 超了只是多花钱
  steps  = 上下文维度      —— 步数是 Agent 独有的失控信号，传统服务不需要它
  wall   = 租约维度        —— 运行环境有固定节流时，失控的循环不是多花钱，是多花时间

三者的修复优先级完全不同。把"步数超支"和"单步开销超支"混成一个数字，
你会得出「token 花超了，所以把预算调大一点」这种错误结论。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Dimension = Literal["tokens", "steps", "wall_clock"]


@dataclass(frozen=True)
class Budget:
    """一次 episode（一个司机的一整月）的预算上限。None 表示不限制。"""

    max_tokens: int | None = None
    max_steps: int | None = None
    max_wall_seconds: float | None = None


@dataclass(frozen=True)
class BudgetBreach:
    """一次熔断事件。必须落盘 —— "被熔断了"和"跑完了"是两种完全不同的结果。"""

    dimension: Dimension
    limit: float
    used: float
    hint: str

    def describe(self) -> str:
        return (
            f"熔断[{self.dimension}] used={self.used:.0f} limit={self.limit:.0f} "
            f"-> {self.hint}"
        )


class BudgetGuard:
    """无状态检查器。Runner 每步调一次，返回 None 表示还活着。

    刻意做成无状态：状态在 Runner 手里，Guard 只负责判断。
    这样 Guard 能被单测，也就能被信任。
    """

    def __init__(self, budget: Budget, *, profile_avg_step_tokens: int | None = None) -> None:
        self.budget = budget
        # 用来区分"步数超支"和"单步开销超支"的参照线
        self.profile_avg_step_tokens = profile_avg_step_tokens

    def check(
        self,
        *,
        tokens: int,
        steps: int,
        wall_seconds: float,
    ) -> BudgetBreach | None:
        b = self.budget

        # 先查步数：步数是最刚性的信号，它一超说明循环压根没收敛
        if b.max_steps is not None and steps >= b.max_steps:
            return BudgetBreach(
                "steps",
                float(b.max_steps),
                float(steps),
                "循环没收敛：检查终止条件、状态回填是否被正确读取、"
                "以及是否存在『无货可接』时缺少显式策略导致的轮询退化",
            )

        if b.max_tokens is not None and tokens >= b.max_tokens:
            # 同样都是 token 超支，病因可能是两种，修复方式完全不同：
            avg = tokens / max(1, steps)
            if (
                self.profile_avg_step_tokens
                and avg > self.profile_avg_step_tokens * 1.5
            ):
                hint = (
                    f"单步开销超支（平均 {avg:.0f} token/步，"
                    f"参照线 {self.profile_avg_step_tokens}）：上下文在膨胀，"
                    "该查历史累积策略与 prompt 分块"
                )
            else:
                hint = (
                    f"步数过多导致总量超支（平均 {avg:.0f} token/步）："
                    "该查循环轮数控制，而不是砍上下文"
                )
            return BudgetBreach("tokens", float(b.max_tokens), float(tokens), hint)

        if b.max_wall_seconds is not None and wall_seconds >= b.max_wall_seconds:
            return BudgetBreach(
                "wall_clock",
                float(b.max_wall_seconds),
                float(wall_seconds),
                "墙钟超支：通常是被限流拖住了。看 limiter.waited_seconds 确认，"
                "而不是直接调大超时",
            )

        return None
