# -*- coding: utf-8 -*-
"""Layer 1 · Runner —— asyncio 并发 + Semaphore + 三维 BudgetGuard + 限流器。

这是全项目唯一一处真正的并发代码。三件事在这里合流：
  预估器 给出 estimated_tokens  ->  限流器 预扣  ->  LLM 调用  ->  限流器 结算

面试里三个模块的耦合点就是这几行。说不清这几行，就等于三个模块是分开的 demo。
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable

from .adapter import AgentAdapter
from .budget.estimator import PROFILES, TokenProfile, estimate_step_tokens
from .budget.guard import Budget, BudgetGuard
from .config import ConfigSnapshot
from .limiter import DualTokenBucketLimiter
from .llm_gate import LLMGate
from .observability import ConsoleTracer, Tracer
from .scenario import Scenario
from .types import EpisodeResult, StepResult, Usage, ZERO_USAGE


@dataclass
class RunResult:
    """一次 run 的全部产物。run_id 是它的主键。"""

    run_id: str
    scenario_name: str
    variant: str
    snapshot: ConfigSnapshot
    episodes: list[EpisodeResult]
    wall_seconds: float
    limiter_counters: dict[str, float] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_name": self.scenario_name,
            "variant": self.variant,
            "wall_seconds": round(self.wall_seconds, 3),
            "snapshot": self.snapshot.to_dict(),
            "limiter": self.limiter_counters,
            "llm": self.extra.get("llm", {}),
            "episodes": [
                {
                    "driver_id": e.driver_id,
                    "steps": e.n_steps,
                    "reward_total": e.reward_total,
                    "tokens": e.usage_total.total,
                    "stop_reason": e.stop_reason,
                }
                for e in self.episodes
            ],
        }


class Runner:
    """并发执行一个 Scenario。

    并发数、预算、限流参数**全部来自 Scenario**，不允许硬编码 ——
    硬编码 = 无法消融 = 无法回答"这个数字是怎么来的"。
    """

    def __init__(
        self,
        scenario: Scenario,
        adapter_factory: Callable[..., AgentAdapter],
        *,
        tracer: Tracer | None = None,
        repo_dir: Path | str = ".",
        impl_paths: list[Path | str] | None = None,
        llm_transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.scenario = scenario
        self.adapter_factory = adapter_factory
        self.tracer: Tracer = tracer or ConsoleTracer()
        self.sem = asyncio.Semaphore(max(1, scenario.concurrency))

        # 限流器：rpm/tpm 来自场景配置
        self.limiter = DualTokenBucketLimiter(scenario.rpm, scenario.tpm)

        # 预估器：profile 名字来自场景配置
        self.profile: TokenProfile = PROFILES[scenario.token_profile]
        avg_step = estimate_step_tokens(5, self.profile)

        # LLM 网关：**与 limiter 共享同一个实例**，并且在这里（Runner 里）建而不是在
        # adapter 里建。理由：限流额度是跨司机共享的，adapter 自己 new 一个就等于
        # 每个司机各限各的，总数会超。网关随后以依赖注入的形式交给 adapter。
        #
        # llm_transport=None -> dry-run。骨架默认不开真模型，这样 smoke 零成本可跑、
        # 也能在没有 API key 的机器上验证接线。要真跑，见 docs/d1-wiring.md 第 5 节。
        self.gate = LLMGate(self.limiter, self.tracer, self.profile, transport=llm_transport)

        # 三维预算熔断
        self.guard = BudgetGuard(
            Budget(
                max_tokens=scenario.max_tokens_per_driver,
                max_steps=scenario.max_steps,
                max_wall_seconds=(
                    scenario.max_wall_time_minutes * 60.0
                    if scenario.max_wall_time_minutes
                    else None
                ),
            ),
            profile_avg_step_tokens=avg_step,
        )

        # 配置快照：run_id 在 Runner 构造时就定下来
        self.snapshot = ConfigSnapshot.capture(
            scenario.to_config_dict(),
            impl_paths=impl_paths or [],
            repo_dir=repo_dir,
        )

    # ------------------------------------------------------------------
    async def run(self) -> RunResult:
        t0 = time.perf_counter()
        tasks = [
            self._guarded(driver_id, self.scenario.seed + i)
            for i, driver_id in enumerate(self.scenario.driver_ids)
        ]
        episodes = list(await asyncio.gather(*tasks))
        return RunResult(
            run_id=self.snapshot.run_id,
            scenario_name=self.scenario.name,
            variant=self.scenario.variant,
            snapshot=self.snapshot,
            episodes=episodes,
            wall_seconds=time.perf_counter() - t0,
            limiter_counters=self.limiter.counters(),
            extra={"llm": self.gate.counters()},
        )

    async def _guarded(self, driver_id: str, seed: int) -> EpisodeResult:
        # Semaphore 在这里限的是"同时在跑的司机数"，
        # 与限流器限的"每分钟发多少次请求"是两件事，不要混。
        async with self.sem:
            return await self._run_driver(driver_id, seed)

    # ------------------------------------------------------------------
    async def _run_driver(self, driver_id: str, seed: int) -> EpisodeResult:
        # gate 以关键字注入。adapter 不接受的参数（dummy 是显式声明并忽略）不会炸 ——
        # 所有 adapter 的工厂签名保持一致，是为了让"换一个被测对象"永远是一次配置改动。
        adapter = self.adapter_factory(gate=self.gate)
        meta = adapter.metadata()

        # 可选能力：adapter 声明"我的每一次模型调用都自己走 LLMGate 计过账了"。
        # 这时 Runner 就**不再做外层按步预扣** —— 否则同一笔 token 会被记两次，
        # TPM 桶被扣双份、RPM 桶按 (1 + N节点) 计数，配额会被无声地砍半。
        # 这不是第 6 个协议方法，只是一个可选属性：只有需要它的 adapter 才提供。
        self_limited = bool(getattr(adapter, "owns_llm_limits", False))
        steps: list[StepResult] = []
        usage_total: Usage = ZERO_USAGE
        reward_total = 0.0
        stop_reason = "completed"
        t0 = time.perf_counter()
        stall = 0  # 连续被限流的次数 —— 必须有上限，无界等待等于死锁

        obs = adapter.reset(driver_id, seed)
        try:
            while True:
                # --- span 3：Prompt 组装（骨架版只记 token 预估，不真拼 prompt）---
                estimated = estimate_step_tokens(obs.step, self.profile)
                with self.tracer.span(
                    "prompt_assemble", driver_id=driver_id, step=obs.step,
                    estimated_tokens=estimated,
                ):
                    pass

                # --- 限流：拿不到就等（自适应退避）---
                # 这就是对项目一"固定 sleep 2 秒"的替代：等待只发生在真正被限流时。
                # 粒度是 **step**：给那些"一步只打一次模型"的 adapter 用（dummy 就是）。
                # 一步打多次模型的（truck: 3 个节点）走 self_limited 分支，
                # 由 LLMGate 按**每次调用**计账 —— 粒度必须跟被测对象的真实调用次数对齐。
                if self.scenario.enforce_limits and not self_limited:
                    ok, wait = self.limiter.acquire(estimated)
                    if not ok:
                        stall += 1
                        if stall > 500:
                            # 连续 500 次拿不到令牌：不是"忙"，是配额配错了
                            stop_reason = "limiter_starved"
                            break
                        # 有上限的等待 —— 无界等待等于另一种形式的拒绝
                        nap = min(max(wait, 0.001), 0.05)
                        self.limiter.waited_seconds += nap
                        await asyncio.sleep(nap)
                        continue
                    stall = 0

                # --- span 4：LLM 推理 ---
                # 被测 Agent 是同步的 -> 用 to_thread 隔离，
                # 绝不允许在协程里直接做阻塞 IO。
                # scope="step"：这个名字会与 LLMGate 内部 scope="call" 的 span 同时出现。
                # **不是重名 bug，是层级关系的体现** —— 9 段 Span 是"阶段"划分，
                # Trace/Span/Generation 是"层级"划分；一次 step 下挂 N 次会议调用。
                # 指标层按 scope 过滤，人要能同时把两种划分讲出来。
                with self.tracer.span(
                    "llm_inference", driver_id=driver_id, step=obs.step,
                    model=meta.model, variant=meta.variant, scope="step",
                ) as box:
                    action, usage = await asyncio.to_thread(adapter.act, obs)
                    box["model_usage_input"] = usage.input_tokens
                    box["model_usage_output"] = usage.output_tokens

                # --- 结算：多退少补 ---
                # self_limited 时跳过：这笔 token 已在 LLMGate 里逐次结算过。
                if not self_limited:
                    self.limiter.settle(estimated, usage.total)

                # --- span 6：工具执行（env 侧推进）---
                with self.tracer.span(
                    "tool_execute", driver_id=driver_id, step=obs.step,
                    action=action.name,
                ):
                    env_step = await asyncio.to_thread(adapter.env_step, action)

                steps.append(
                    StepResult(
                        step=obs.step,
                        action=action,
                        reward=env_step.reward,
                        usage=usage,
                        terminated=env_step.terminated,
                        truncated=env_step.truncated,
                        info={"signature": action.signature()},
                    )
                )
                usage_total = usage_total + usage
                reward_total += env_step.reward

                if env_step.terminated:
                    stop_reason = "terminated"
                    break
                if env_step.truncated:
                    stop_reason = "truncated"
                    break

                breach = self.guard.check(
                    tokens=usage_total.total,
                    steps=len(steps),
                    wall_seconds=time.perf_counter() - t0,
                )
                if breach is not None:
                    # 落盘熔断原因 ——「被熔断了」和「跑完了」是两种完全不同的结果
                    stop_reason = f"budget:{breach.dimension}"
                    break

                # 步号由 Runner 统一回填：只有一处维护"第几步"，
                # 否则环境侧和平台侧会各数一套，早晚对不上。
                obs = replace(env_step.observation, step=len(steps))
        finally:
            adapter.close()

        return EpisodeResult(
            driver_id=driver_id,
            run_id=self.snapshot.run_id,
            variant=self.scenario.variant,
            steps=steps,
            usage_total=usage_total,
            reward_total=reward_total,
            terminated=(stop_reason == "terminated"),
            truncated=stop_reason != "terminated",
            stop_reason=stop_reason,
            wall_seconds=time.perf_counter() - t0,
        )


def run_scenario(
    scenario: Scenario,
    adapter_factory: Callable[..., AgentAdapter],
    *,
    tracer: Tracer | None = None,
    repo_dir: Path | str = ".",
    impl_paths: list[Path | str] | None = None,
    llm_transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> RunResult:
    """同步入口。命令行和脚本都用它，免得每处都写 asyncio.run。"""
    runner = Runner(
        scenario, adapter_factory, tracer=tracer, repo_dir=repo_dir,
        impl_paths=impl_paths, llm_transport=llm_transport,
    )
    return asyncio.run(runner.run())
