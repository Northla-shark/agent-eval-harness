# -*- coding: utf-8 -*-
"""跨层通用的值对象。

设计约束（写在这里，因为它们决定了为什么这几行代码长这样）：
  1. 全部 frozen。值对象一旦构造就不可变 —— 否则它会跨协程被意外改写，
     而并发下的这种改写是不可复现 bug 的头号来源。
  2. 不依赖任何第三方库。这些类型是整个平台的地基，地基引库 = 到处引库。
  3. 字段名对齐 OpenTelemetry GenAI 语义约定（input_tokens / output_tokens）。
     换来一个能力：换掉 Langfuse 不需要改字段名。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Usage:
    """一次 LLM 调用的 token 账。

    reasoning_tokens 单独记：推理 token 的单价与普通输出不同，
    混在 output 里会让成本归因失真（项目一里这一项最高占到输出的 95.2%）。
    """

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.reasoning_tokens + other.reasoning_tokens,
        )


# 可复用的零值：避免每步都构造一个新对象
ZERO_USAGE = Usage()


@dataclass(frozen=True)
class AgentMeta:
    """被测对象的身份。写进 Trace 的 metadata，事后按 variant 分组对比。

    variant 是消融实验的分组键：full / -geo-agent / +prefilter / 1-shot。
    """

    name: str
    variant: str
    model: str = "none"
    prompt_version: str = "v0"
    # 用 tuple 而不是 dict：frozen dataclass 里放 dict 会让实例不可哈希
    extra: tuple[tuple[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "variant": self.variant,
            "model": self.model,
            "prompt_version": self.prompt_version,
            **dict(self.extra),
        }


@dataclass(frozen=True)
class Observation:
    """环境交给 Agent 的当前状态。环境侧产生，Agent 只读。"""

    step: int
    sim_minute: int
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Action:
    """Agent 产出的一次动作。"""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def signature(self) -> str:
        """(动作, 关键参数) 签名 —— 震荡检测的输入。

        为什么不只用动作名：连续 wait 是正常的。真正的问题是【同一参数】反复出现
        （项目一里 A 组有 146 次 duration=60），那代表 Agent 没有从上次尝试里
        获得任何新信息 —— 这才是"迷路"的指纹。
        """
        if not self.params:
            return self.name
        kv = ",".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{self.name}({kv})"


@dataclass(frozen=True)
class EnvStep:
    """环境对一次动作的反馈。

    与 Action 分开的原因：决策质量与环境反馈必须能分开归因。
    如果二者混在一个返回值里，"这步亏了是决策错还是环境随机"就永远说不清。
    """

    observation: Observation
    reward: float
    terminated: bool = False   # 任务自然结束（到达月末）
    truncated: bool = False    # 因外部限制被截断（步数/预算/时间上限）


@dataclass(frozen=True)
class StepResult:
    """一步的完整产物。Runner 只认这个类型，所以任何 Adapter 都必须产出它。"""

    step: int
    action: Action
    reward: float
    usage: Usage
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeResult:
    """一个 driver 的一次整月仿真 = 一条 Trace。

    Trace 是"最小可比单元"：A/B 对比时，先比 driver 级，再聚合到 run 级，
    这样能看出"两组差在均值上，还是差在尾部司机上"。
    """

    driver_id: str
    run_id: str
    variant: str
    steps: list[StepResult]
    usage_total: Usage
    reward_total: float
    terminated: bool
    truncated: bool
    stop_reason: str = "completed"
    wall_seconds: float = 0.0

    @property
    def n_steps(self) -> int:
        return len(self.steps)
