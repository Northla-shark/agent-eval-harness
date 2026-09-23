# -*- coding: utf-8 -*-
"""有界任务队列：背压 + 三种过载策略。

为什么必须有界
--------------
无界队列在限流场景下等于把内存打穿 ——
上游一直被限流、下游一直被塞，内存成了唯一的缓冲区。

对应面试题：Q21（限流时怎么办：排队而不是拒绝）、Q22（队列怎么选型）。
"""
from __future__ import annotations

import asyncio
import enum
from typing import Any


class Overload(enum.Enum):
    """过载时三种策略。没有"最好"的一种，只有和场景匹配的一种。"""

    QUEUE = "queue"      # 保成功率、牺牲延迟 —— 离线批处理任务的最优解
    DEGRADE = "degrade"  # 切小模型 / 砍上下文 / 少走一轮反思 —— 交互式场景
    REJECT = "reject"    # 明确返回忙碌 —— 保系统本身


class BoundedTaskQueue:
    def __init__(self, maxsize: int, policy: Overload = Overload.QUEUE) -> None:
        self.q: asyncio.Queue[Any] = asyncio.Queue(maxsize=maxsize)
        self.policy = policy

    async def submit(self, item: Any, deadline: float | None = None) -> str:
        if not self.q.full():
            await self.q.put(item)
            return "accepted"

        if self.policy is Overload.REJECT:
            return "rejected"

        if self.policy is Overload.DEGRADE:
            return "degraded"  # 交给编排层降级执行

        # QUEUE：阻塞等待，但带超时 ——「无限等待」等于另一种形式的拒绝
        try:
            await asyncio.wait_for(self.q.put(item), timeout=deadline)
            return "accepted_after_wait"
        except (asyncio.TimeoutError, TimeoutError):
            return "timeout"

    async def get(self) -> Any:
        return await self.q.get()

    def depth(self) -> int:
        """队列深度要做成指标暴露出去。

        它是系统过载最早、最灵敏的信号 —— 比报错早得多。
        """
        return self.q.qsize()
