# -*- coding: utf-8 -*-
"""LRU + TTL 缓存层 —— 一份代码答三道面试题（Q36 / Q37 / Q49）。

题目原文（真实面试题单 Q49）：
  请设计并实现一个带过期时间的 LRU KV Cache：
    - 所有 KV 的过期时间相同；
    - 最多存储 N 对 KV；
    - 容量已满时，优先淘汰一个已经过期的 KV；
    - 如果没有过期的 KV，则按照 LRU 规则淘汰最久未使用的数据；
    - 查询到已过期的 KV 时返回空；
    - 说明采用的数据结构，并实现查询与写入操作。

设计说明（面试时要能讲出来的部分）
----------------------------------
1) 数据结构
   - 主字典：key -> (value, expire_at)，配合「访问序」的 OrderedDict
     保证 get/put 均摊 O(1)，并天然维护 LRU 顺序。
   - 容量的判定用「存储条数」，不是「未过期条数」（题设是"最多存储 N 对"）。

2) 题设"所有 KV 的过期时间相同"是一条可被利用的强约束
   - 若理解为「统一 TTL、从各自写入时刻起算」：
     expire_at(key) = write_time(key) + TTL，写入时刻单调递增
     => 过期时刻的先后顺序 == 写入顺序
     => 「已过期的项」永远是**写入队列的前缀**
     => 找过期项不必 O(n) 扫描全表，只需看写入队列的队首。
   - 若理解为「所有项在同一个绝对时刻过期」（batch expiry）：
     则连每项的时间戳都不需要存，一个全局 deadline 即可。
   两种理解都实现了（v1 朴素 / v2 优化）。面试时主动指出这个歧义是加分项。

3) 惰性删除（lazy deletion）
   - get 命中但已过期 -> 删除并返回 None。
   - 不做后台清理线程：单次操作的复杂度可控，且避免了并发下的锁竞争。

4) 与 Agent 场景的对应（这才是它在本仓库里的真实用途）
   - 模型响应缓存：消融实验要反复重跑同一个 baseline，
     缓存能把"重复 baseline"的成本压到近零，直接提升可做的消融组数。
   - 司机级特征缓存：key 必须是「可见偏好集合的内容签名」，不能只用 driver_id。
     因为偏好有时间可见窗 —— key 少了时变输入，缓存会返回过期约束，
     而**约束会静默失效**：不报错、不崩溃，但校验形同虚设。
     「缓存键必须包含所有会影响结果的时变输入」—— 能主动讲这条比会写 LRU 更值钱。
   - 评测结果缓存：key = run_id（配置哈希 + 代码哈希），代码一变结果就不可复用。

复杂度
------
  v1 朴素版：get O(1)，put O(1) / 最坏 O(n)（容量满时扫描找过期项）
  v2 优化版：get O(1)，put 均摊 O(1)（过期项必在写入队列队首，附惰性清理）
"""

from collections import OrderedDict
import time


class LRUKVCache:
    """v2 优化版：利用"统一 TTL"把"找过期项"从 O(n) 扫描降为 O(1) 看队首。

    内部维护两个顺序：
      _store : OrderedDict[key -> (value, expire_at)]，按**访问序**（LRU 顺序，队首最旧）
      _writes: OrderedDict[key -> None]，按**写入序**（过期顺序，队首最先过期）
    """

    __slots__ = ("capacity", "ttl", "_clock", "_store", "_writes")

    def __init__(self, capacity: int, ttl: float, clock=time.monotonic):
        if capacity <= 0:
            raise ValueError("capacity 必须为正整数")
        if ttl <= 0:
            raise ValueError("ttl 必须为正数")
        self.capacity = capacity
        self.ttl = ttl
        self._clock = clock
        self._store: "OrderedDict[object, tuple]" = OrderedDict()
        self._writes: "OrderedDict[object, None]" = OrderedDict()

    # ---------- 读 ----------
    def get(self, key, default=None):
        item = self._store.get(key)
        if item is None:
            return default
        value, expire_at = item
        if self._clock() >= expire_at:      # 惰性删除：查到已过期就顺手清掉
            self._drop(key)
            return default
        self._store.move_to_end(key, last=True)   # 命中即刷新为最近使用
        return value

    def __contains__(self, key) -> bool:
        return self.get(key, _MISS) is not _MISS

    # ---------- 写 ----------
    def put(self, key, value):
        now = self._clock()
        expire_at = now + self.ttl        # 统一 TTL，从本次写入时刻起算

        if key in self._store:            # 已存在 -> 覆盖值、刷新 TTL、刷新 LRU
            self._store[key] = (value, expire_at)
            self._store.move_to_end(key, last=True)
            self._writes[key] = None      # 写入序也更新（过期时刻随之后移）
            self._writes.move_to_end(key, last=True)
            return

        if len(self._store) >= self.capacity:
            self._evict(now)
        self._store[key] = (value, expire_at)
        self._writes[key] = None

    # ---------- 淘汰 ----------
    def _evict(self, now):
        """容量已满时的淘汰：优先淘汰一个已过期项，否则按 LRU 淘汰。"""
        # 过期项必在写入序队首（统一 TTL + 单调时钟）-> O(1) 取候选
        while self._writes:
            oldest = next(iter(self._writes))
            item = self._store.get(oldest)
            if item is None:              # 已被惰性删除，对齐后继续
                self._writes.popitem(last=False)
                continue
            _, expire_at = item
            if now >= expire_at:          # 找到过期项 -> 淘汰它
                self._drop(oldest)
                return
            break                         # 队首都没过期，说明全表无过期项
        # 无过期项 -> LRU 淘汰（访问序队首 = 最久未使用）
        while self._store:
            lru_key = next(iter(self._store))
            self._drop(lru_key)
            return

    def _drop(self, key):
        self._store.pop(key, None)
        self._writes.pop(key, None)

    # ---------- 观测 ----------
    def __len__(self):
        """存储条数（含已过期未清理的项，因为容量是按存储条数算的）。"""
        return len(self._store)

    def live_size(self):
        """未过期条数（需要遍历，仅用于观测/断言）。"""
        now = self._clock()
        return sum(1 for _, exp in self._store.values() if now < exp)

    def keys_by_lru(self):
        """按 LRU 顺序返回 key（队首最久未使用）。用于测试与调试。"""
        return list(self._store.keys())

    def keys_by_expiry(self):
        """按过期先后返回 key（队首最先过期）。用于验证"前缀性质"。"""
        return list(self._writes.keys())


class LRUKVCacheNaive:
    """v1 朴素版：容量满时全表扫描找过期项。逻辑最直白，但要能说出它的 O(n)。"""

    def __init__(self, capacity: int, ttl: float, clock=time.monotonic):
        self.capacity = capacity
        self.ttl = ttl
        self._clock = clock
        self._store: "OrderedDict[object, tuple]" = OrderedDict()

    def get(self, key, default=None):
        item = self._store.get(key)
        if item is None:
            return default
        value, expire_at = item
        if self._clock() >= expire_at:
            del self._store[key]
            return default
        self._store.move_to_end(key, last=True)
        return value

    def put(self, key, value):
        now = self._clock()
        expire_at = now + self.ttl
        if key in self._store:
            self._store[key] = (value, expire_at)
            self._store.move_to_end(key, last=True)
            return
        if len(self._store) >= self.capacity:
            for k in list(self._store.keys()):        # O(n) 扫描
                if now >= self._store[k][1]:
                    del self._store[k]
                    break
            else:
                self._store.popitem(last=False)
        self._store[key] = (value, expire_at)

    def __len__(self):
        return len(self._store)


_MISS = object()


# ======================================================================
# 自测：可脱离项目独立运行，作为面试前的"手感练习"答案
#   python -m evalharness.cache
# ======================================================================
def self_test(verbose: bool = True) -> bool:
    # 可控时钟
    class Clock:
        def __init__(self): self.t = 1000.0
        def __call__(self): return self.t
        def advance(self, dt): self.t += dt

    def make(cls, cap, ttl):
        c = Clock()
        return cls(cap, ttl, clock=c), c

    passed = failed = 0

    def check(name, cond, extra=""):
        nonlocal passed, failed
        if cond:
            passed += 1
            if verbose:
                print(f"  [PASS] {name}")
        else:
            failed += 1
            print(f"  [FAIL] {name}  {extra}")

    for cls in (LRUKVCacheNaive, LRUKVCache):
        tag = cls.__name__
        if verbose:
            print(f"\n=== {tag} ===")

        c, clk = make(cls, 3, 10)
        c.put("a", 1); c.put("b", 2)
        check(f"{tag}/T1 基本读写", c.get("a") == 1 and c.get("b") == 2 and c.get("z") is None)

        c, clk = make(cls, 3, 10)
        c.put("a", 1); clk.advance(10)
        check(f"{tag}/T2 过期返回空", c.get("a") is None, f"got {c.get('a')}")

        c, clk = make(cls, 3, 100)
        c.put("a", 1); c.put("b", 2); c.put("c", 3)
        c.get("a")                      # a 变最近使用，b 成为最久未使用
        c.put("d", 4)
        check(f"{tag}/T3 LRU淘汰最久未使用",
              c.get("b") is None and c.get("a") == 1 and c.get("c") == 3 and c.get("d") == 4,
              f"lru order={c.keys_by_lru() if hasattr(c, 'keys_by_lru') else 'n/a'}")

        c, clk = make(cls, 3, 10)
        c.put("a", 1)
        clk.advance(10)                 # a 已过期
        c.put("b", 2); c.put("c", 3)
        c.get("b"); c.get("c")          # b、c 都是最近使用，a 也是最久未使用
        c.put("d", 4)                   # 容量满：应淘汰已过期的 a，而不是 LRU
        check(f"{tag}/T4 优先淘汰过期项",
              c.get("a") is None and c.get("b") == 2 and c.get("c") == 3 and c.get("d") == 4,
              f"store={getattr(c, '_store', None)}")

        c, clk = make(cls, 3, 10)
        c.put("x", 1)                   # 先写入，会先过期
        clk.advance(9)
        c.put("y", 2); c.put("z", 3)
        c.get("x")                      # x 成为"最近使用"（但即将过期）
        clk.advance(1)                  # 此刻 x 过期；LRU 顺序是 y -> z -> x
        c.put("w", 4)                   # 必须淘汰 x（过期），不能淘汰 y（LRU）
        check(f"{tag}/T5 过期优先于LRU",
              c.get("y") == 2 and c.get("z") == 3 and c.get("w") == 4,
              f"y={c.get('y')} z={c.get('z')} w={c.get('w')}")

        c, clk = make(cls, 2, 10)
        c.put("a", 1); clk.advance(6); c.put("a", 99); clk.advance(6)
        check(f"{tag}/T6 覆盖写刷新TTL", c.get("a") == 99, f"got {c.get('a')}")

        c, clk = make(cls, 5, 10_000)
        for i in range(50):
            c.put(f"k{i}", i)
        check(f"{tag}/T7 容量上界", len(c) <= 5, f"len={len(c)}")

        c, clk = make(cls, 2, 5)
        c.put("a", 1); c.put("b", 2); clk.advance(5)
        c.put("c", 3); c.put("d", 4); c.put("e", 5)
        check(f"{tag}/T8 全过期后写入", len(c) <= 2 and c.get("e") == 5, f"len={len(c)}")

    class Clock2:
        def __init__(self): self.t = 0.0
        def __call__(self): return self.t
        def advance(self, dt): self.t += dt
    clk = Clock2()
    c = LRUKVCache(10, 100, clock=clk)
    for k in "abcdef":
        c.put(k, k); clk.advance(1)
    c.get("a"); c.get("b")             # 访问序被打乱，写入序不应受影响
    check("LRUKVCache/T9 写入序不被访问打乱", c.keys_by_expiry() == list("abcdef"),
          f"{c.keys_by_expiry()}")
    check("LRUKVCache/T10 访问序被 get 打乱", c.keys_by_lru()[:2] == ["c", "d"],
          f"{c.keys_by_lru()}")

    if verbose:
        print(f"\n结果：{passed} 通过 / {failed} 失败")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if self_test() else 1)
