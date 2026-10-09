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

    def consume(self, n: float) -> None:
        """无条件扣 n，**允许余额变成负数** —— 负值就是欠账。

        与 try_acquire 的分工：
          try_acquire —— 请求发出前的准入判断，拿不到就别发（保准入）
          consume     —— 请求已经发生了，费用必须记账，不能因为桶空就不记（保账实相符）
        用同一个 balance 表示「余额」和「欠账」，是为了让"还欠账"自然发生：
        负数余额会被 refill 慢慢填平，不需要第二条状态。
        """
        self._refill()
        self.tokens -= n

    def debt(self) -> float:
        """当前欠账（负余额的绝对值，没欠就是 0.0）。"""
        self._refill()
        return max(0.0, -self.tokens)

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

    acquire: 预扣（额度 = 本次预估 + 上次欠账）    settle: 按实际 usage 多退少补，不够则记欠账

    欠账（D2 补）是这个限流器最后一块拼图：没有它，一次严重低估的请求
    会把 TPM 上限静默突破，而 `rejected` 计数仍然显示 0 ——
    **最危险的不是被限流，是限流器以为自己没被击穿。**
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
        self.debt_events = 0     # 指标：产生过新欠账的结算次数
        self.debt_ever = 0.0     # 指标：累计产生过的欠账总额（不是当前余额）

    def acquire(self, estimated_tokens: int) -> tuple[bool, float]:
        """返回 (是否拿到, 建议等待秒数)。"""
        # 1) 先拿请求票
        if not self.rpm.try_acquire(1):
            self.rejected += 1
            return False, self.rpm.wait_time(1)

        # 2) 再按预估值拿 token 额度 —— **额度里必须先包含旧欠账**
        # 这是 D2 补的核心：请求额度 = 本次预估 + 上次结算欠下的。
        # 不加欠账的话，欠账永远不会被还，限流器会持续超发，TPM 上限形同虚设。
        need = estimated_tokens + self.tpm.debt()
        if not self.tpm.try_acquire(need):
            # 拿不到必须回滚请求票 —— 否则 token 桶长期紧张时请求票会被逐步耗空，
            # RPM 桶明明有余量却发不出请求。这是一个很容易漏、且很难复现的饥饿问题。
            self.rpm.refund(1)
            self.rejected += 1
            return False, self.tpm.wait_time(need)

        return True, 0.0

    def settle(self, estimated_tokens: int, actual_tokens: int) -> None:
        """按实际用量结算：多退少补，桶不够补扣就记欠账（负余额）。

        为什么欠账必须记下来而不是丢掉：
          一次请求实际花了 10 万 token、预估只有 1 万时，差额 9 万是**已经发生**的成本。
          旧实现用 try_acquire 去补扣，桶不够就静默返回 False，这 9 万从此消失 ——
          限流器会以为自己还有额度，继续放行，TPM 上限被静默突破。
          欠账 = 负余额：下一次 acquire 必须先把负余额填平才能再发请求，
          所以超发的那部分会在后续请求里被自动"还"回来。
        """
        delta = actual_tokens - estimated_tokens
        if delta > 0:
            before = self.tpm.debt()
            self.tpm.consume(delta)          # 允许转负：记账优先于准入
            grown = self.tpm.debt() - before
            if grown > 0:
                self.debt_ever += grown
                self.debt_events += 1
        elif delta < 0:
            # 退钱时 min(capacity, ...) 兜住上限；若此时有欠账，退的钱先抵欠账，
            # 因为欠账就是负余额 —— 不需要额外的分支。
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
            # 欠账必须可观测：它是"限流器被击穿过"的唯一证据。
            # 只看 rejected 会漏掉这一类 —— 欠账时请求是成功放行的。
            "limiter.tpm_debt": round(self.tpm.debt(), 1),
            "limiter.debt_events": float(self.debt_events),
            # 已偿还 = 累计欠过的 - 现在还欠的。
            # 不能在 acquire 里做减法：令牌补充会先把负余额填平（min(capacity,…)），
            # 那一刻债务已经消失了，事后再取差值恒为 0。
            "limiter.debt_repaid": round(max(0.0, self.debt_ever - self.tpm.debt()), 1),
        }
