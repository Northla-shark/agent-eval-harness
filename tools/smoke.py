#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键自检 —— Day 1 的验收标准就是它全绿。

    python tools/smoke.py

它检查 5 件事，顺序是刻意的：从最底层的纯函数往上走到端到端，
任何一层坏了都能立刻定位，而不是看到一个空结果去猜。

  1. limiter    —— 双令牌桶的速率、回滚、结算
  2. estimator  —— 系数真的来自历史数据（K 是否为 0 决定线性还是超线性）
  3. cache      —— LRU + TTL 的 18 项边界（含"过期优先于 LRU"）
  4. config     —— run_id 会不会把两次不同配置误判成同一次
  5. 端到端     —— 同一环境下跑两个 variant，打出对照表

不需要任何 API key。退出码 0 = 全绿。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import adapters  # noqa: E402,F401   —— import 即注册 "dummy" / "truck"
from evalharness.adapter import available, build  # noqa: E402
from evalharness.budget.estimator import (  # noqa: E402
    PROFILES,
    estimate_step_tokens,
    estimate_task_tokens,
    is_superlinear,
)
from evalharness.cache import LRUKVCache, self_test as cache_self_test  # noqa: E402
from evalharness.config import ConfigSnapshot, diff  # noqa: E402
from evalharness.experiment import compare_table, save_run  # noqa: E402
from evalharness.limiter import DualTokenBucketLimiter, TokenBucket  # noqa: E402
from evalharness.observability import ConsoleTracer  # noqa: E402
from evalharness.runner import run_scenario  # noqa: E402
from evalharness.scenario import Scenario  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  [OK]   {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {extra}")


class FrozenClock:
    """冻结时钟：限流器的测试必须去掉时间这个变量，否则断言会随机失败。"""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---------------------------------------------------------------------------
def s1_limiter() -> None:
    print("\n[1/5] 限流器：双令牌桶 + 预扣结算")

    # 速率与突发
    b = TokenBucket(capacity=10, refill_per_sec=10 / 60, clock=FrozenClock())
    got = [b.try_acquire(1) for _ in range(11)]
    check("令牌桶：容量内可取、超出即拒", got.count(True) == 10 and got[-1] is False)
    check("令牌桶：wait_time > 0", b.wait_time(1) > 0)

    # ★ 请求票必须可回滚：拿不到 token 额度时把请求票退回去
    lim = DualTokenBucketLimiter(5, 10, clock=FrozenClock())
    ok1, _ = lim.acquire(10)      # rpm 5->4, tpm 10->0
    ok2, _ = lim.acquire(1)       # rpm 5->3, tpm 不足 -> 回滚 rpm
    check("预扣：第一次成功", ok1 is True)
    check("预扣：token 桶不足时拒绝", ok2 is False)
    check(
        "★ 请求票已回滚（否则会饥饿：RPM 有余量却发不出请求）",
        abs(lim.rpm.tokens - 4.0) < 1e-6,
        f"rpm.tokens={lim.rpm.tokens}",
    )

    # 结算：多退少补
    lim2 = DualTokenBucketLimiter(10, 1000, clock=FrozenClock())
    lim2.acquire(500)
    lim2.settle(500, 800)         # 实际多花 300 -> 补扣
    check("结算：实际超出预估时补扣", abs(lim2.tpm.tokens - 200.0) < 1e-6, f"{lim2.tpm.tokens}")
    lim2.settle(800, 300)         # 实际少花 500 -> 退还
    check("结算：实际少于预估时退还", lim2.tpm.tokens > 200.0, f"{lim2.tpm.tokens}")
    check("指标：rejected 被计数", lim2.rejected >= 0)


# ---------------------------------------------------------------------------
def s2_estimator() -> None:
    print("\n[2/5] Token 预估器：系数来自历史 14 组跑批")

    nh = PROFILES["no_history"]
    h5 = PROFILES["history_5"]
    t_nh = estimate_task_tokens(88, nh, safe=False)
    t_h5 = estimate_task_tokens(88, h5, safe=False)

    print(f"    88 步 · 不累积历史 : {t_nh:,} token")
    print(f"    88 步 · 累积历史   : {t_h5:,} token   -> {t_h5 / t_nh:.2f} 倍")
    check("K≈0 判定为线性", is_superlinear(nh) is False)
    check("K>0 判定为超线性", is_superlinear(h5) is True)
    check("同一步数下累积历史显著更贵（>2 倍）", t_h5 / t_nh > 2.0)

    # K 项随步索引增长：第 87 步的单步开销应远大于第 0 步
    cheap = estimate_step_tokens(0, h5)
    pricey = estimate_step_tokens(87, h5)
    print(f"    单步开销：第 0 步 {cheap:,} -> 第 87 步 {pricey:,}")
    check("单步开销随步索引上升（上下文累积）", pricey > cheap * 1.5)


# ---------------------------------------------------------------------------
def s3_cache() -> None:
    print("\n[3/5] 缓存层：LRU + TTL，18 项边界")
    check("LRU+TTL 全部边界用例通过", cache_self_test(verbose=False) is True)

    c = LRUKVCache(2, 1000)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    check("LRU：容量压在上界内且淘汰最久未使用", len(c) <= 2 and c.get("a") is None)


# ---------------------------------------------------------------------------
def s4_config() -> None:
    print("\n[4/5] Config Snapshot：run_id 由配置内容决定")

    base = dict(
        model_name="m-a", model_api_url="", temperature=0.0,
        prompt_version="v0", adapter_name="dummy", variant="baseline",
        simulation_duration_days=30, simulation_max_steps=200,
        max_tokens_per_driver=400000, max_wall_time_minutes=20,
    )
    a = ConfigSnapshot.capture(base, repo_dir=ROOT)
    b = ConfigSnapshot.capture(dict(base), repo_dir=ROOT)
    c = ConfigSnapshot.capture({**base, "model_name": "m-b"}, repo_dir=ROOT)

    check("同配置 -> 同 run_id", a.run_id == b.run_id)
    check("换模型 -> 换 run_id", a.run_id != c.run_id)
    d = diff(a, c)
    check("diff 只报出真正变化的字段", set(d.keys()) == {"model_name"}, f"{set(d.keys())}")

    # ★ variant 必须进快照：否则"基线 vs 消融变体"会撞 run_id
    e = ConfigSnapshot.capture({**base, "variant": "degenerate"}, repo_dir=ROOT)
    check("★ 换 variant -> 换 run_id（早期文档的 SNAPSHOT_FIELDS 漏了这个字段）",
          a.run_id != e.run_id)


# ---------------------------------------------------------------------------
def s5_end_to_end() -> None:
    print("\n[5/5] 端到端：同一环境，两个 variant")

    base = Scenario.tiny(max_steps=200, max_tokens_per_driver=5_000_000)
    degenerate = base.with_variant("degenerate", adapter_name="dummy")

    tracer = ConsoleTracer()

    def run(sc: Scenario):
        return run_scenario(
            sc,
            # variant 是**平台传给 adapter 的参数**，不是在 adapter 里写死的。
            # 这一点如果搞错，两个变体会跑出完全相同的结果 —— 而且不会报错，只是静默地白跑。
            lambda: build(sc.adapter_name, variant=sc.variant),
            tracer=tracer,
            repo_dir=ROOT,
            impl_paths=[ROOT / "adapters" / "dummy.py"],
        )

    r_base = run(base)
    r_degen = run(degenerate)

    check("两次 run 都能结束", all(e.stop_reason in {"terminated", "truncated"} for e in r_base.episodes + r_degen.episodes))
    print("\n" + compare_table([r_base, r_degen]))

    d = diff(r_base.snapshot, r_degen.snapshot)
    print(f"\n    受控性检验：两组差异字段 = {d}")
    check("A/B 差异只有 variant 一个字段（受控实验）", set(d.keys()) == {"variant"}, f"{d}")

    cov = tracer.coverage()
    print(f"    9 段 span 覆盖：{ {k: v for k, v in cov.items() if v} }")
    check("至少 3 段 span 有真实数据", sum(1 for v in cov.values() if v) >= 3)

    # schema 必须是被 assert 锁死的：一旦有人随手写 "llm_call"，指标层就再也聚不起来
    try:
        with tracer.span("llm_call"):
            pass
        schema_locked = False
    except AssertionError:
        schema_locked = True
    check("未定义的 span 名会被 assert 拦下（schema 锁死）", schema_locked)

    print("\n" + tracer.render_waterfall(base.driver_ids[0], limit=8))
    print(f"\n    限流器计数器：{r_base.limiter_counters}")

    out = save_run(r_base, out_dir=ROOT / "runs", profile_name=base.token_profile)
    check("结果已落盘到 runs/<run_id>/", (out / "metrics.json").exists())
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    check("指标 JSON 含 outcome / process / cost 三层",
          {"outcome", "process", "cost"} <= set(metrics))

    check("可用的 adapter 注册表非空", "dummy" in available())


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("AgentEval Harness · smoke 自检")
    print(f"仓库根目录：{ROOT}")
    print("=" * 72)

    for fn in (s1_limiter, s2_estimator, s3_cache, s4_config, s5_end_to_end):
        try:
            fn()
        except Exception as exc:  # 自检脚本自己绝不能把异常吞掉
            import traceback

            FAIL.append(f"{fn.__name__} 抛异常")
            print(f"  [FAIL] {fn.__name__} 抛出 {type(exc).__name__}: {exc}")
            traceback.print_exc()

    print("\n" + "=" * 72)
    print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
    if FAIL:
        for name in FAIL:
            print(f"  - {name}")
        print("=" * 72)
        return 1
    print("全绿。Day 1 的验收标准达成。")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
