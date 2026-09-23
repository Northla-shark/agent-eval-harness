# -*- coding: utf-8 -*-
"""AgentAdapter 协议 —— 平台与脚本的分界线就在这一个接口上。

一句话：任何满足这个协议的东西都能被评测，不管它是 LangGraph 图、一次 LLM 调用，
还是一个 if-else。这一条就是"评测平台"和"一堆脚本"的区别。
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from .types import Action, AgentMeta, EnvStep, Observation, Usage


@runtime_checkable
class AgentAdapter(Protocol):
    """被测对象的统一接入协议。

    接口刻意只有 5 个方法。多一个方法，接入方就多一份心智负担，
    "任何 Agent 都能接"这个承诺就少一分可信度。
    """

    def metadata(self) -> AgentMeta:
        """返回身份信息。会被写进每一步的 span metadata，用于事后分组。"""
        ...

    def reset(self, driver_id: str, seed: int) -> Observation:
        """开始一个新 episode（一个司机的一个整月），返回首个观测。

        每次都要能 reset：不能 reset 的 Agent 无法做重复实验，
        也就无法区分"改动有效"和"随机波动"。
        """
        ...

    def act(self, obs: Observation) -> tuple[Action, Usage]:
        """被测 Agent 的唯一决策入口。返回动作 + 这次决策花掉的 token。

        这里是**同步**签名。如果 Agent 内部是同步的（项目一就是），
        Runner 会用 asyncio.to_thread() 把它隔离到线程里，
        绝不允许在协程里直接做阻塞 IO。
        """
        ...

    def env_step(self, action: Action) -> EnvStep:
        """环境侧推进。这一层**不属于**被测 Agent。

        分开的理由：动作质量与环境反馈必须能分开归因。
        把 env 和 agent 合成一个方法，以后就永远说不清
        "这一步亏了"是决策错、还是环境本身随机。
        """
        ...

    def close(self) -> None:
        """释放资源。协议要求它存在，因为评测会开几百个实例。"""
        ...


# ---------------------------------------------------------------------------
# 注册表：用名字而不是 import 来挑选被测对象
# ---------------------------------------------------------------------------
AdapterFactory = Callable[..., AgentAdapter]

_REGISTRY: dict[str, AdapterFactory] = {}


def register(name: str) -> Callable[[AdapterFactory], AdapterFactory]:
    """装饰器：把一个 Adapter 工厂注册成可被名字找出来的东西。

    为什么要注册表：Scenario 里写的是 adapter_name: "dummy" 这样的字符串。
    这样"换一个被测对象"是一次配置改动，不是一次代码改动 ——
    消融实验的全部前提就在这里。
    """

    def deco(factory: AdapterFactory) -> AdapterFactory:
        _REGISTRY[name] = factory
        return factory

    return deco


def build(name: str, **kwargs: Any) -> AgentAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"未注册的 adapter: {name!r}，可选：{sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
