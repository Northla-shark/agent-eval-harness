#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""零熟练度最小知识包 —— 5 个概念，每个都是「项目里非它不可」的地方。

    python tools/learn_basics.py

不用读教材。这 5 个概念就是项目二全部的后端门槛，
剩下的（限流器、LRU、队列、预估器）都已经在仓库里写好了。

每段都按同一个套路讲：
    它是干什么的 -> 没有它会怎样（用你自己项目一的痛点说） -> 跑一遍看数字
"""
from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Protocol, runtime_checkable


def title(n: int, text: str) -> None:
    print("\n" + "=" * 70)
    print(f"[{n}] {text}")
    print("=" * 70)


# ===========================================================================
title(1, "async def / await / asyncio.gather —— 并发的单位")
print("""
它是什么：async def 定义一个"可以被暂停的函数"。
          await 表示"这里要等外部结果，等的期间把控制权交出去，让别人先跑"。
          asyncio.gather(...) = 同时启动多个，全部等完。

没有它会怎样（用你项目一的真实数字说）：
  你在项目一里用的是固定 sleep(2) 秒。这确实保证了不超限，但代价是
  等待被写进了串行流程里 —— 一次 30 天评测 800 多步，每步几次调用，
  光等待就是几十分钟。
  await 不是"等得更快"，而是"等待期间不占着位置"。这就是唯一的区别，
  也是唯一的收益。
""")


async def one_driver(driver_id: str, minutes: float) -> str:
    await asyncio.sleep(minutes)  # 模拟"等模型返回"
    return driver_id


async def demo_serial_vs_concurrent() -> None:
    drivers = ["d001", "d002", "d003", "d004"]

    t0 = time.perf_counter()
    for d in drivers:
        await one_driver(d, 0.05)          # 串行：一个等完再等下一个
    serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    await asyncio.gather(*[one_driver(d, 0.05) for d in drivers])  # 并发
    concurrent = time.perf_counter() - t0

    print(f"    串行 4 个司机：{serial * 1000:7.1f} ms")
    print(f"    并发 4 个司机：{concurrent * 1000:7.1f} ms")
    print(f"    加速比：{serial / concurrent:.2f}x   （理论 4x，损耗来自调度）")
    assert concurrent < serial


# ===========================================================================
title(2, "asyncio.Semaphore —— 限并发，不是限速度")
print("""
它是什么：一个计数器。acquire 时 -1，release 时 +1，减到 0 就阻塞。
          用于"同时最多 N 个在跑"。

没有它会怎样：
  项目一为了不超限用固定 sleep。如果反过来一把全放开，
  800 步 × 4 节点全部同时发请求，会直接把厂商限流打爆（429），
  而且报错的位置是随机的 —— 你事后完全无法复现。

Semaphore 限的是"同时在跑的司机数"，限流器限的是"每分钟发多少次请求"。
这两件事经常被混为一谈，但它们是两个不同维度的旋钮。
""")

_peak = 0
_now = 0


async def demo_semaphore() -> None:
    global _peak, _now
    sem = asyncio.Semaphore(4)

    async def job(i: int) -> None:
        global _peak, _now
        async with sem:
            _now += 1
            _peak = max(_peak, _now)
            await asyncio.sleep(0.01)
            _now -= 1

    await asyncio.gather(*[job(i) for i in range(40)])
    print(f"    40 个任务，Semaphore(4) -> 观测到的峰值并发 = {_peak}")
    assert _peak <= 4


# ===========================================================================
title(3, "dataclass(frozen=True) —— 值对象")
print("""
它是什么：自动生成 __init__ / __repr__ / __eq__ 的类装饰器。
          frozen=True 表示构造完就不能改。

没有它会怎样：
  并发下，同一个对象会被多个 driver 的协程共享。
  任何一个地方随手 obj.step = 5，都会让另一个协程读到错的值 ——
  而且这种 bug 只在特定调度顺序下复现，属于最难查的一类。
  项目一没有并发所以没碰到；项目二一上 asyncio 就会碰到。
""")


@dataclass(frozen=True)
class Obs:
    step: int
    payload: dict


def demo_frozen() -> None:
    o = Obs(step=1, payload={"cargo": 3})
    print(f"    {o}")
    try:
        o.step = 2  # type: ignore[misc]
        raise SystemExit("不应该走到这里")
    except Exception as exc:
        print(f"    改字段被拒绝：{type(exc).__name__}: {exc}")

    # 派生新对象用 dataclasses.replace，不原地改
    from dataclasses import replace

    o2 = replace(o, step=2)
    print(f"    replace 派生：{o2}  （原对象不变：{o}）")
    assert o.step == 1 and o2.step == 2


# ===========================================================================
title(4, "typing.Protocol —— AgentAdapter 为什么能被'任何 Agent'实现")
print("""
它是什么：结构化子类型。"只要你的类有这几个方法，你就满足这个协议"，
          不需要继承，不需要 import 我的基类。

没有它会怎样：
  如果 AgentAdapter 写成一个抽象基类，那么每个被测 Agent 都必须
  import 并继承它 —— 平台框架就渗进了被测对象的代码里。
  评测平台一旦污染被测对象，A/B 就不再是纯净对照。
  Protocol 让"接入"变成零侵入。
""")


@runtime_checkable
class AgentAdapter(Protocol):
    def act(self, obs: int) -> str: ...


class MyAgent:
    """注意：它没有继承任何东西。"""

    def act(self, obs: int) -> str:
        return f"action-for-{obs}"


class NotAnAgent:
    pass


def demo_protocol() -> None:
    a = MyAgent()
    print(f"    MyAgent 满足协议？   {isinstance(a, AgentAdapter)}")
    print(f"    NotAnAgent 满足协议？{isinstance(NotAnAgent(), AgentAdapter)}")
    assert isinstance(a, AgentAdapter) and not isinstance(NotAnAgent(), AgentAdapter)


# ===========================================================================
title(5, "@contextmanager —— 9 段 Span 是怎么做到「一行包一段」的")
print("""
它是什么：把一个 with 语句拆成"进入前"和"退出时"两半。
          9 段 Span 就靠它：with tracer.span("llm_inference", step=3):。

没有它会怎样：
  要么每个节点手写 try/finally 记开始时间和结束时间（重复 9 遍，
  漏一个就少一段 span）；要么用装饰器包函数，但那样就没法在函数
  中间插一段 span —— 而一次单步决策里的 9 段是**顺序**发生的，
  不是 9 个函数。
""")


@contextmanager
def timed(name: str) -> Iterator[dict]:
    box: dict = {}
    t0 = time.perf_counter()
    try:
        yield box               # 业务代码在这里跑，可以往 box 里塞指标
    finally:
        ms = (time.perf_counter() - t0) * 1000
        print(f"    span={name:<14} {ms:6.2f}ms  收集到的指标={box}")


def demo_contextmanager() -> None:
    with timed("prompt_assemble") as box:
        time.sleep(0.01)
        box["tokens"] = 3499
    with timed("llm_inference") as box:
        time.sleep(0.02)
        box["model"] = "none"


# ===========================================================================
async def main() -> None:
    await demo_serial_vs_concurrent()
    await demo_semaphore()
    demo_frozen()
    demo_protocol()
    demo_contextmanager()
    print("\n" + "=" * 70)
    print("5 个概念全部跑通。项目二的后端门槛就这些 —— ")
    print("限流器 / LRU / 队列 / 预估器都已在 evalharness/ 里写好，不需要你重造。")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
