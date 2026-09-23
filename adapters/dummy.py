# -*- coding: utf-8 -*-
"""假 Agent：不调用任何模型。

它的存在不是为了"凑一个能跑的东西"，而是三件事：
  1. 让整条链路（Runner / 限流 / 预算 / Span / 指标）可以在**零 API 成本**下开发与自检。
     你后面 6 天里会重跑上百次，全都打真模型是不可接受的。
  2. 它证明了 AgentAdapter 是可替换的 —— 换掉这个文件，就是一个真 Agent。
  3. variant="degenerate" 复现了项目一那个真实的退化形态：
     固定 wait 60 分钟轮询，步数 3 倍、wait 率 ~89%、成本 3 倍，而订单数并没有更多。
     这是整个平台的"价值演示样本"。
"""
from __future__ import annotations

import random

from evalharness.adapter import register
from evalharness.budget.estimator import PROFILES
from evalharness.types import Action, AgentMeta, EnvStep, Observation, Usage

MINUTES_PER_DAY = 1440


@register("dummy")
class DummyAdapter:
    """一个确定性的玩具环境 + 两种策略。

    Args:
        variant: "baseline"（会重定位、等 30 分钟）/ "degenerate"（只等 60 分钟）
        seed: 固定种子。**同一个 seed 的两个 variant 看到完全相同的货流** ——
              这是"受控对照"的物质基础：环境不变，只变策略。
        days: 仿真天数
        profile_name: 用哪组 token 系数（决定每步开销）
    """

    def __init__(
        self,
        variant: str = "baseline",
        seed: int = 0,
        days: int = 5,
        profile_name: str = "no_history",
    ) -> None:
        self.variant = variant
        self.seed = seed
        self.days = days
        self.profile = PROFILES[profile_name]
        self.profile_name = profile_name
        self._rng = random.Random(seed)
        self._sim_minute = 0
        self._idle_streak = 0
        self._cargo_available = False
        self._closed = False

    # ---------------- AgentAdapter 协议 ----------------
    def metadata(self) -> AgentMeta:
        return AgentMeta(
            name="dummy-sim",
            variant=self.variant,
            model="none",
            prompt_version="v0",
            extra=(("profile", self.profile_name),),
        )

    def reset(self, driver_id: str, seed: int) -> Observation:
        # 用 seed 而不是 driver_id 初始化随机流：
        # 同一个 seed 的两个 variant 看到的是同一条货流，
        # 否则"差异来自策略"这个结论不成立。
        self._rng = random.Random(seed)
        self._sim_minute = 0
        self._idle_streak = 0
        self._cargo_available = self._rng.random() < 0.55
        return self._obs(0)

    def act(self, obs: Observation) -> tuple[Action, Usage]:
        if self.variant == "degenerate":
            # 退化策略：绝大部分时间在等 —— 项目一 A 组的形态
            if self._cargo_available and self._rng.random() < 0.25:
                action = Action("take_order", {"cargo_id": f"c{obs.step}"})
            else:
                action = Action("wait", {"duration": 60})
                self._idle_streak += 1
        else:
            if self._cargo_available:
                action = Action("take_order", {"cargo_id": f"c{obs.step}"})
                self._idle_streak = 0
            elif self._idle_streak >= 2:
                action = Action("reposition", {"target": "hub-A"})
                self._idle_streak = 0
            else:
                action = Action("wait", {"duration": 30})
                self._idle_streak += 1

        # 制造一个"看起来像真的"token 账：
        # 输入按校准系数给（S + K*step），再加 ±8% 抖动 ——
        # 抖动是刻意的，否则"预估 vs 实际"的偏差指标会恒为 0，等于没测。
        jitter = 1.0 + self._rng.uniform(-0.08, 0.08)
        base_in = self.profile.fixed_per_step + self.profile.growth_per_step * obs.step
        usage = Usage(
            input_tokens=int(base_in * jitter),
            output_tokens=self.profile.completion_per_step,
            reasoning_tokens=0,
        )
        return action, usage

    def env_step(self, action: Action) -> EnvStep:
        if self._closed:
            raise RuntimeError("adapter 已 close()，不能再推进一步")

        if action.name == "take_order":
            elapsed = 180
            reward = self._rng.uniform(600.0, 1200.0)
            next_cargo = self._rng.random() < 0.55
        elif action.name == "reposition":
            elapsed = 90
            reward = -30.0
            next_cargo = self._rng.random() < 0.7
        else:  # wait
            elapsed = int(action.params.get("duration", 60))
            reward = 0.0
            next_cargo = self._rng.random() < 0.55

        self._sim_minute += elapsed
        self._cargo_available = next_cargo
        total_minutes = self.days * MINUTES_PER_DAY
        terminated = self._sim_minute >= total_minutes
        # 步号由 Runner 统一回填（Runner 用 dataclasses.replace 覆盖 step 字段）。
        # 环境侧不该自己数步数 —— 那会让"步数"这一个量有两个来源，早晚不一致。
        obs = Observation(
            step=-1,
            sim_minute=self._sim_minute,
            payload={"cargo_available": self._cargo_available},
        )
        return EnvStep(
            observation=obs,
            reward=round(reward, 2),
            terminated=terminated,
            truncated=False,
        )

    def close(self) -> None:
        self._closed = True

    # ---------------- 内部 ----------------
    def _obs(self, step: int) -> Observation:
        return Observation(
            step=step,
            sim_minute=self._sim_minute,
            payload={
                "cargo_available": self._cargo_available,
                "days": self.days,
            },
        )
