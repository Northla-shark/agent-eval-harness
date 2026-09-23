# -*- coding: utf-8 -*-
"""项目一（卡车司机 Agent）-> Adapter。D2 任务。

这个文件是"平台 vs 脚本"分界线的证明：项目一的 LangGraph 四个节点
在这里被包成一个满足 AgentAdapter 协议的对象，平台完全不知道它是 LangGraph。

做法（三步，别改顺序）：
  1) 把项目一 demo/agent/ 下的 model_decision_service.py / time_agent_langchain.py /
     tools/ 复制一份到本仓库（**只复制你自己写的部分**，
     不要复制 simkit/、server/bench/、server/data/、calc_monthly_income.py —— 那是赛方资产），
     去掉所有赛方 import。
  2) 在下面 TruckAdapter 里实现 5 个协议方法。
     关键点：LangGraph 的图在 __init__ 里 compile 一次，不要每步重编 ——
     compile 是"启动期完成装配"，运行期只跑图。
  3) 注册成 "truck"，然后在 scenarios/truck-driver-monthly-v1.yaml 里
     把 adapter_name 改成 truck。

验收（D2 结束标准）：
  1 个司机 × 5 天能跑通，且在 Langfuse / ConsoleTracer 里能看到
  某司机某一步下挂的 4 个节点 span 各自的 token 与耗时。
"""
from __future__ import annotations

from typing import Any

from evalharness.adapter import register
from evalharness.types import Action, AgentMeta, EnvStep, Observation, Usage


@register("truck")
class TruckAdapter:
    """TODO(D2)：把项目一的四节点 LangGraph 接进来。

    建议的字段：
        self._graph        —— 在 __init__ 里 compile 一次，别每步重编
        self._driver_id    —— reset() 时写入，贯穿整月
        self._state        —— LangGraph 的 State，reset() 时初始化
    """

    def __init__(self, variant: str = "full", seed: int = 0, **kwargs: Any) -> None:
        self.variant = variant
        self.seed = seed
        raise NotImplementedError(
            "TODO(D2)：见本文件 docstring。先把 adapters/dummy.py 跑通再动它。"
        )

    def metadata(self) -> AgentMeta:  # pragma: no cover - 尚未实现
        raise NotImplementedError

    def reset(self, driver_id: str, seed: int) -> Observation:  # pragma: no cover
        raise NotImplementedError

    def act(self, obs: Observation) -> tuple[Action, Usage]:  # pragma: no cover
        raise NotImplementedError

    def env_step(self, action: Action) -> EnvStep:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError
