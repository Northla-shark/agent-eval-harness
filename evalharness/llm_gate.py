# -*- coding: utf-8 -*-
"""Layer 2.5 · LLM Gate —— 把「一次模型调用」收归平台，再以依赖注入的形式交给 Adapter。

为什么必须有这一层（这是 D1 最容易做错的一步）
================================================
Runner 外层是**按 step** 限流：一步 `acquire` 一次。
但项目一的一步里有 **3 个会调模型的节点**（agent1 偏好分类 / agent3 时间分析 / agent4 决策），
一次 step 实际打 3 次模型。如果限流只包在最外层：

  · RPM 桶按 step 计数 → 实际请求率被低估 3 倍 → 上游 429，且**报错位置随机、事后不可复现**
  · `llm_inference` span 只能记到「整步耗时」，拆不出是哪个节点慢、哪个节点贵
  · 缓存没有落点 —— prompt 只在节点内部才存在，外层看不到
  · token 账只能事后拿一个总数，无法按节点归因

所以把「一次 LLM 调用」做成平台的原子操作：**被测 Agent 不自己发 HTTP，它交出 payload，平台来发。**
这一条就是「平台」和「一堆脚本」在代码上的分界点。

对应面试题
==========
  Q5 / Q6   平台化与一次性脚本的区别（分界线就在这个注入点）
  Q18 / Q20 限流限度数还是限 token、RPM / TPM 双桶
  Q21       限流发生时系统怎么表现（计数器要能拿出来）
  Q36 / Q37 缓存键必须含所有时变输入，否则约束静默失效

已知的简化（面试主动讲，比写"完美"代码更能证明你懂）
====================================================
  1. token 预估是**按 payload 字符数**的粗估（中英混排 ≈ 2 char/token）。
     `budget/estimator.py` 里的 S/K/C 系数是按 **step 索引**拟合的，两者在 D2 合并 ——
     合并后预估器才同时知道「第几步」和「这次 payload 多大」。
  2. 令牌桶里的 `try_acquire` 是无锁的。D1 用 `concurrency: 1`（见 docs/d1-wiring.md 的解释），
     所以没有真实竞争。多线程要加锁 —— 这一条写在 docs 里，不假装它不存在。
"""
from __future__ import annotations

import inspect
import json
import time
from typing import Any, Callable

from .budget.estimator import TokenProfile
from .cache import LRUKVCache
from .limiter import DualTokenBucketLimiter
from .observability import Tracer
from .types import Usage


class LLMGateRejected(RuntimeError):
    """限流窗口内始终拿不到令牌。**必须是个异常，不能是静默降级** ——
    静默降级会把"被限流"伪装成"决策质量差"，这正是项目一当初用固定 sleep 掩盖掉的问题。"""


def _canonical(payload: dict[str, Any]) -> str:
    """稳定序列化：缓存键必须与 dict 的插入顺序无关，否则同一个 prompt 会算出两个 key。"""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def extract_usage(resp: dict[str, Any]) -> Usage:
    """OpenAI 兼容响应 -> Usage。

    reasoning_tokens 单独取：项目一里 reasoning 最高占输出的 95.2%，
    混进 output 会让成本归因失真。
    """
    usage = resp.get("usage") or {}
    if not isinstance(usage, dict):
        return Usage()
    details = usage.get("completion_tokens_details") or {}
    reasoning = 0
    if isinstance(details, dict):
        reasoning = int(details.get("reasoning_tokens") or 0)
    return Usage(
        input_tokens=max(0, int(usage.get("prompt_tokens") or 0)),
        output_tokens=max(0, int(usage.get("completion_tokens") or 0)),
        reasoning_tokens=max(0, reasoning),
    )


def _accepts_context(fn: Any) -> bool:
    """transport 是否愿意接收 `(node=, driver_id=, step=)` 上下文。

    用 `inspect` 判断，而不是 try/except TypeError 兜底 ——
    后者会把 transport **内部**真正的 TypeError 也一并吞掉，那种 bug 会退化成
    "偶尔返回奇怪结果"，是极难排查的一类故障。

    为什么要传上下文：网关知道"这次调用来自哪个节点"，stub 才能按节点作答。
    HTTP 层打桩做不到这件事 —— 那里只看得见一坨 payload。
    """
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return "node" in params


class LLMGate:
    """一次模型调用的唯一入口。

    Args:
        limiter:  与 Runner 共享的**同一个**双令牌桶 —— 共享是关键，
                  adapter 自己 new 一个 limiter 就等于每个司机各限各的，总数会超。
        tracer:   9 段 Span 的落点。这里只开 `llm_inference`，attrs 带 `node` 与 `scope="call"`。
        profile:  token 系数（决定 completion 侧预估）。
        transport: 真正发 HTTP 的函数。**为 None 时进入 dry-run** ——
                  返回一个合法的 OpenAI 兼容响应、usage 按 payload 长度估。
                  这不是"临时方案"：它让整条链路可以在零成本下开发与自检，
                  也是 `tools/check_truck.py` 能不需要 API key 就验证接线的物质基础。
    """

    def __init__(
        self,
        limiter: DualTokenBucketLimiter,
        tracer: Tracer,
        profile: TokenProfile,
        *,
        transport: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache_capacity: int = 512,
        cache_ttl_seconds: float = 900.0,
        max_wait_seconds: float = 5.0,
    ) -> None:
        self.limiter = limiter
        self.tracer = tracer
        self.profile = profile
        self.transport = transport or self._dry_transport
        self._takes_ctx = _accepts_context(self.transport)
        # "没花钱"有两种：压根没发（transport 为 None），
        # 或发给了替身（ScriptedTransport 声明了 is_fake）。两者都必须被标记出来，
        # 否则你会把一组估算出来的 token 当成真实账单写进报告。
        self.dry_run = transport is None or bool(getattr(transport, "is_fake", False))
        self.cache = LRUKVCache(cache_capacity, cache_ttl_seconds)
        self.max_wait_seconds = max_wait_seconds

        # 观测口径 —— 与 limiter.counters() 一起进 metrics.json
        self.calls = 0
        self.cache_hits = 0
        self.rejected = 0
        self.usage_total = Usage()
        self.saved_tokens = 0
        # 调用来源计数。port = 走平台端口；agent_own_client = 被测对象自建客户端被运行期接管。
        self.calls_by_source: dict[str, int] = {}

    # ------------------------------------------------------------------ 公共
    def call(
        self,
        payload: dict[str, Any],
        *,
        node: str = "llm",
        driver_id: str = "",
        step: int = -1,
        source: str = "port",
    ) -> dict[str, Any]:
        """发一次模型调用。返回原始响应 dict（保持 OpenAI 兼容形状，交给调用方自己解析）。

        `source` 标这次调用是怎么进网关的：
          · `"port"`            —— 被测对象走平台注入的端口进来（正规路径）
          · `"agent_own_client"` —— 被测对象自建客户端、被适配层**运行期接管**进来
        两个数必须分开记，见 `counters()` 里的 `llm.calls_foreign`：
        **被接管的调用数就是你原本会漏记的调用数**，它应该是 0，不是"忽略不计"。
        """
        self.calls_by_source[source] = self.calls_by_source.get(source, 0) + 1
        key = self._cache_key(payload)

        hit = self.cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            self.calls += 1
            usage = extract_usage(hit)
            self.saved_tokens += usage.total
            # 命中也要开 span：否则"某次决策为什么这么快"在瀑布图里看不出来。
            # estimated_tokens=0 是刻意的 —— 这次调用确实没花钱。
            with self.tracer.span(
                "llm_inference", driver_id=driver_id, step=step, node=node,
                scope="call", cache="hit", estimated_tokens=0, source=source,
            ) as box:
                box["cache_hit"] = True
                box["tokens_saved"] = usage.total
            # ★ 命中时把 usage 清零再返回。
            # 理由：下游（Runner / 成本指标 / 赛方口径的 usage 累计）读的是 resp["usage"]，
            # 那是"这次调用花掉的 token"。缓存命中没有产生上游调用，因此增量为 0。
            # 本来要花的那部分记在 `saved_tokens` 里 —— **省下的和花掉的必须分别可见**，
            # 否则"开了缓存"和"模型变便宜了"在成本表上看起来一模一样。
            # 用浅拷贝而不是改缓存对象：缓存里的原始响应必须保持原样，
            # 否则第一次命中就把 usage 清零了，第二次命中连 saved_tokens 都算不出来。
            return {**hit, "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                     "total_tokens": 0, "completion_tokens_details": {}}}

        estimated = self.estimate(payload)

        with self.tracer.span(
            "llm_inference", driver_id=driver_id, step=step, node=node,
            scope="call", cache="miss", estimated_tokens=estimated,
            dry_run=self.dry_run, source=source,
        ) as box:
            self._acquire_bounded(estimated, node=node)

            t0 = time.perf_counter()
            resp = self._invoke(payload, node=node, driver_id=driver_id, step=step)
            box["latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            usage = extract_usage(resp)
            # 预扣 -> 结算（多退少补）
            self.limiter.settle(estimated, usage.total)

            self.usage_total = self.usage_total + usage
            box["model_usage_input"] = usage.input_tokens
            box["model_usage_output"] = usage.output_tokens
            box["model_usage_reasoning"] = usage.reasoning_tokens
            box["token_estimation_error"] = usage.total - estimated

        self.calls += 1
        self.cache.put(key, resp)
        return resp

    # ------------------------------------------------------------------ 预估
    def estimate(self, payload: dict[str, Any]) -> int:
        """按字符数的粗估。**这是 D1 的已知简化**，D2 与 estimator.py 的 S/K/C 系数合并。

        为什么先要有个粗估：token 桶必须在**请求发出前**扣，否则限流形同虚设。
        粗估偏小 -> 桶会被打穿；偏大 -> 吞吐被人为压低。所以上限取两者较大值。
        """
        chars = len(_canonical(payload))
        prompt = max(self.profile.fixed_per_step, chars // 2)
        return prompt + self.profile.completion_per_step

    # ------------------------------------------------------------------ 内部
    def _invoke(self, payload: dict[str, Any], **ctx: Any) -> dict[str, Any]:
        """把上下文交给愿意接收它的 transport。真实 HTTP transport 只要 payload，
        所以这里按签名分流，而不是强行要求所有 transport 都改签名。"""
        if self._takes_ctx:
            return self.transport(payload, **ctx)
        return self.transport(payload)

    def _acquire_bounded(self, estimated: int, *, node: str) -> None:
        """有上限的等待。无界等待等于另一种形式的死锁。"""
        deadline = time.monotonic() + self.max_wait_seconds
        while True:
            ok, wait = self.limiter.acquire(estimated)
            if ok:
                if wait:
                    self.limiter.waited_seconds += wait
                return
            self.rejected += 1
            if time.monotonic() >= deadline:
                raise LLMGateRejected(
                    f"node={node} 在 {self.max_wait_seconds}s 内未取到令牌"
                    f"（estimated={estimated}）；不是'忙'，是配额配错了"
                )
            nap = min(max(wait, 0.001), 0.25)
            self.limiter.waited_seconds += nap
            time.sleep(nap)

    def _cache_key(self, payload: dict[str, Any]) -> str:
        """缓存键 = 整个 payload 的规范序列化。

        ★ 必须含**所有时变输入**。项目一里踩过的坑：
          只按 (prompt 前 200 字) 做键，导致"司机状态不同但前缀相同"的两步共用了一次响应 ——
          约束静默失效，而且不报错。教训：键里少一个字段，缓存就从优化变成 bug 源。
        """
        return _canonical(payload)

    def _dry_transport(self, payload: dict[str, Any]) -> dict[str, Any]:
        """不打真模型：返回一个合法的 OpenAI 兼容响应。

        content 是 "{}" —— 意味着解析层会走它的容错分支。这是刻意的：
        接线验证要证明的是"链路通"，不是"模型答得对"。
        """
        prompt = max(self.profile.fixed_per_step, len(_canonical(payload)) // 2)
        completion = self.profile.completion_per_step
        return {
            "id": "dry-run",
            "object": "chat.completion",
            "model": "dry-run",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "{}"},
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "_dry_run": True,
        }

    # ------------------------------------------------------------------ 观测
    def counters(self) -> dict[str, float]:
        hit_rate = (self.cache_hits / self.calls) if self.calls else 0.0
        return {
            "llm.calls": float(self.calls),
            "llm.calls_port": float(self.calls_by_source.get("port", 0)),
            # ★ 这个数应该尽量是 0。非 0 说明被测对象里有调用绕过了平台 ——
            #   不接管的话，这些 token 在任何成本表里都是"凭空消失"。
            "llm.calls_foreign": float(self.calls_by_source.get("agent_own_client", 0)),
            "llm.cache_hits": float(self.cache_hits),
            "llm.cache_hit_rate": round(hit_rate, 4),
            "llm.rejected": float(self.rejected),
            "llm.tokens_saved": float(self.saved_tokens),
            "llm.tokens_in": float(self.usage_total.input_tokens),
            "llm.tokens_out": float(self.usage_total.output_tokens),
            "llm.tokens_reasoning": float(self.usage_total.reasoning_tokens),
            "llm.tokens_total": float(self.usage_total.total),
            "llm.dry_run": 1.0 if self.dry_run else 0.0,
        }
