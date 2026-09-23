# -*- coding: utf-8 -*-
"""双令牌桶限流器：RPM 桶 + TPM 桶，预扣-结算语义。

为什么必须做两个桶
------------------
只限 RPM：一次请求塞进十万 token 的上下文，照样能把额度和钱包打穿。
只限 TPM：几千个小请求照样能把并发连接打满。
两个桶都拿到令牌才能发请求。

LLM 限流与传统限流的本质差异
---------------------------
请求发出前你不知道它要花多少 token。所以 token 桶必须做成
「预扣 -> 结算（多退少补）」，而不是「拿一个减一」。

对应面试题：Q18（限流限次数还是限 token）、Q20（RPM/TPM）。
"""
from __future__ import annotations

import time
from typing import Callable


class TokenBucket:
    """标准令牌桶：以恒定速率补充，允许突发到 capacity。"""

    __slots__ = ("capacity", "rate", "tokens", "updated", "_clock")

    def __init__(
        self,
        capacity: float,
        refill_per_sec: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = float(capacity)
        self.rate = float(refill_per_sec)
        self.tokens = float(capacity)  # 初始满桶：允许冷启动突发
        self.updated = clock()
        self._clock = clock

    def _refill(self) -> None:
        # 惰性补充：不做后台定时器，避免线程/协程竞争
        now = self._clock()
        self.tokens = min(
            self.capacity, self.tokens + (now - self.updated) * self.rate
        )
        self.updated = now

    def try_acquire(self, n: float = 1.0) -> bool:
        self._refill()
        if self.tokens + 1e-9 >= n:
            self.tokens -= n
            return True
        return False

    def refund(self, n: float) -> None:
        self._refill()
        self.tokens = min(self.capacity, self.tokens + n)

    def wait_time(self, n: float = 1.0) -> float:
        """还需要等多少秒才能拿到 n 个令牌（0 表示现在就可以）。"""
        self._refill()
        if self.tokens >= n:
            return 0.0
        return (n - self.tokens) / self.rate


class DualTokenBucketLimiter:
    """RPM 桶（限请求次数）+ TPM 桶（限 token 量）。

    acquire: 预扣    settle: 按实际 usage 多退少补
    """

    def __init__(
        self,
        rpm: int,
        tpm: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rpm = TokenBucket(rpm, rpm / 60.0, clock)
        self.tpm = TokenBucket(tpm, tpm / 60.0, clock)
        self.rejected = 0        # 指标：被限流次数
        self.waited_seconds = 0.0  # 指标：累计等待时长

    def acquire(self, estimated_tokens: int) -> tuple[bool, float]:
        """返回 (是否拿到, 建议等待秒数)。"""
        # 1) 先拿请求票
        if not self.rpm.try_acquire(1):
            self.rejected += 1
            return False, self.rpm.wait_time(1)

        # 2) 再按预估值拿 token 额度
        # 拿不到必须回滚请求票 —— 否则 token 桶长期紧张时请求票会被逐步耗空，
        # RPM 桶明明有余量却发不出请求。这是一个很容易漏、且很难复现的饥饿问题。
        if not self.tpm.try_acquire(estimated_tokens):
            self.rpm.refund(1)
            self.rejected += 1
            return False, self.tpm.wait_time(estimated_tokens)

        return True, 0.0

    def settle(self, estimated_tokens: int, actual_tokens: int) -> None:
        """按实际用量结算。

        已知的简化（面试里主动讲，比写出"完美"代码更能证明你懂限流）：
          实际消耗远超预估、桶里又不够补扣时，try_acquire 会静默失败，欠账丢失。
          更严格的做法是把欠账记成负余额，在下次 acquire 时优先偿还。
          —— 这个升级留给 D2 由你自己实现，见 docs/architecture.md。
        """
        delta = actual_tokens - estimated_tokens
        if delta > 0:
            self.tpm.try_acquire(delta)
        elif delta < 0:
            self.tpm.refund(-delta)

    def counters(self) -> dict[str, float]:
        """暴露给 Tracing 层。

        这不是装饰：它是"限流时系统怎么表现"的可观测证据（对应 Q21）。
        面试里说不出"我被限流了多少次、累计等了多久"，就等于没做过限流。
        """
        return {
            "limiter.rejected": float(self.rejected),
            "limiter.waited_seconds": round(self.waited_seconds, 3),
            "limiter.tpm_tokens": round(self.tpm.tokens, 1),
            "limiter.rpm_tokens": round(self.rpm.tokens, 1),
        }
