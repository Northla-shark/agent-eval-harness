#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键自检 —— Day 1 的验收标准就是它全绿。

    python tools/smoke.py

它检查 7 件事，顺序是刻意的：从最底层的纯函数往上走到端到端，
任何一层坏了都能立刻定位，而不是看到一个空结果去猜。

  1. limiter    —— 双令牌桶的速率、回滚、结算
  2. estimator  —— 系数真的来自历史数据（K 是否为 0 决定线性还是超线性）
  3. cache      —— LRU + TTL 的 18 项边界（含"过期优先于 LRU"）
  4. config     —— run_id 会不会把两次不同配置误判成同一次
  5. 端到端     —— 同一环境下跑两个 variant，打出对照表
  6. 接线       —— LLMGate 的缓存与计数、TruckAdapter 的协议完整性
                  （**不需要项目一的代码、不需要 API key**）
  7. 指标层     —— 用历史跑批验证指标口径（**本机没有项目一则整节跳过**）

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
from evalharness.llm_gate import LLMGate  # noqa: E402
from evalharness.metrics import history as hist  # noqa: E402
from evalharness.metrics import process as proc  # noqa: E402
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
    print("\n[1/7] 限流器：双令牌桶 + 预扣结算")

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
    print("\n[2/7] Token 预估器：系数来自历史 14 组跑批")

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
    print("\n[3/7] 缓存层：LRU + TTL，18 项边界")
    check("LRU+TTL 全部边界用例通过", cache_self_test(verbose=False) is True)

    c = LRUKVCache(2, 1000)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    check("LRU：容量压在上界内且淘汰最久未使用", len(c) <= 2 and c.get("a") is None)


# ---------------------------------------------------------------------------
def s4_config() -> None:
    print("\n[4/7] Config Snapshot：run_id 由配置内容决定")

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
    print("\n[5/7] 端到端：同一环境，两个 variant")

    base = Scenario.tiny(max_steps=200, max_tokens_per_driver=5_000_000)
    degenerate = base.with_variant("degenerate", adapter_name="dummy")

    tracer = ConsoleTracer()

    def run(sc: Scenario):
        return run_scenario(
            sc,
            # variant 是**平台传给 adapter 的参数**，不是在 adapter 里写死的。
            # 这一点如果搞错，两个变体会跑出完全相同的结果 —— 而且不会报错，只是静默地白跑。
            # **kw 用来接 Runner 注入的 gate=：所有 adapter 的工厂签名保持一致。
            lambda **kw: build(sc.adapter_name, variant=sc.variant, **kw),
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
def s6_wiring() -> None:
    """接线自检 —— 这一段全绿，说明 D1 的"通路"完成，而且还没花一分钱。

    它刻意**不 import 项目一**：`truck` adapter 在项目一不存在时也必须能被构造
    （抛可读错误）并通过协议检查。这是"平台与被测对象解耦"的可执行证明，
    不是文档里的一句承诺。
    """
    print("\n[6/7] 接线：LLMGate + TruckAdapter 协议完整性")

    tracer = ConsoleTracer()
    limiter = DualTokenBucketLimiter(rpm=600, tpm=2_000_000)
    gate = LLMGate(limiter, tracer, PROFILES["no_history"])
    payload = {"model": "qwen3-max", "messages": [{"role": "user", "content": "在吗"}]}

    r1 = gate.call(payload, node="agent1", driver_id="d001", step=0)
    check("LLMGate dry-run 返回 OpenAI 兼容形状", isinstance(r1, dict) and "usage" in r1)
    check("dry-run 标记可见（防止误以为打了真模型）", r1.get("_dry_run") is True)

    gate.call(payload, node="agent1", driver_id="d001", step=0)
    check("★ 相同 payload 第二次命中缓存（缓存键含全部输入）", gate.cache_hits == 1,
          f"cache_hits={gate.cache_hits}")

    gate.call({**payload, "temperature": 0.7}, node="agent2", driver_id="d001", step=0)
    check("不同 payload 不误命中", gate.cache_hits == 1, f"cache_hits={gate.cache_hits}")
    check("usage 按次累计（token 账的唯一来源）", gate.usage_total.total > 0)

    cov = tracer.coverage()
    check("LLMGate 自发开 llm_inference span", cov["llm_inference"] >= 3, f"{cov['llm_inference']}")
    check("★ 限流/缓存计数器可读出（讲 Q21 时的现场证据）",
          {"llm.calls", "llm.cache_hit_rate", "llm.tokens_total"} <= set(gate.counters()))

    # --- TruckAdapter：只查「协议完整性 + 报错是否可读」，不碰项目一 ---
    from adapters.truck import TruckAdapter

    required = ("metadata", "reset", "act", "env_step", "close")
    missing = [m for m in required if not callable(getattr(TruckAdapter, m, None))]
    check("TruckAdapter 的 5 个协议方法全部存在", not missing, f"缺: {missing}")
    check("TruckAdapter 声明 owns_llm_limits=True（一步 3 次调用 -> 按次计账）",
          getattr(TruckAdapter, "owns_llm_limits", False) is True)

    # 项目一不在本机时必须是**可读的报错**，不是静默降级成 dummy。
    # 静默降级会让你以为在测项目一，其实在测一个假 Agent —— 而且它不会报错。
    try:
        TruckAdapter(gate=gate, project_root=r"Z:\__definitely_not_here__")
        loud = False
        note = ""
    except FileNotFoundError:
        loud, note = True, ""
    except Exception as exc:
        loud, note = True, f"（注意：抛的是 {type(exc).__name__}，期望 FileNotFoundError）"
    check("项目一不存在时抛可读错误，而不是静默降级", loud, note)


# ---------------------------------------------------------------------------
def s7_metrics_on_history() -> None:
    """指标层 × 历史跑批：口径一致性验证。

    ★ 为什么是"读历史数据"而不是"新跑一次仿真"：
        **用历史数据验证指标层的一致性，而不是新跑一次 —— 新跑一次只能验证一次。**
        新跑一次只提供一个样本点，且与指标实现同源（口径错了它会一起错，看不出来）。
        历史跑批是已经发生过的独立数据，零成本，且能同时验证多组。

    项目一不在本机时**整节跳过且不记失败**：平台不能依赖被测对象的私有目录才能自检。
    """
    print("\n[7/7] 指标层：用历史跑批验证口径（不是新跑一次）")

    if not hist.available():
        print("    [跳过] 本机无历史跑批（clone 本仓库的人默认没有）-> 不计失败")
        print("           要跑这一段：`python tools/check_metrics_history.py --root <路径>`")
        return

    runs = hist.list_runs()
    print(f"    发现 {len(runs)} 组跑批；抽验 A/B 两组 + 两种长度的基准")

    # --- 1) 震荡指纹：A 组的已知真值 ---
    a_name = "上传偏好+历史5步+休息时间统计 补充三种prompt+要求输出秒数"
    a = hist.load_run(a_name)
    if a is None:
        check("历史跑批可加载", False, f"找不到 {a_name}")
        return
    a_steps = [s for e in hist.load_episodes(a_name) for s in e.steps]
    counts = proc.signature_counts(a_steps)
    check(
        "★ 震荡指纹复现：wait(duration=60) = 146 次",
        counts.get("wait(duration=60)") == 146,
        f"实测 {counts.get('wait(duration=60)')}",
    )
    print(f"    A 组 {len(a_steps)} 步 · top3 签名：{list(counts.items())[:3]}")

    lsr_a = proc.loop_signature_rate(a_steps)
    print(f"    A 组震荡率 = {lsr_a:.4f}（B 组对照见 check_metrics_history.py）")
    check("震荡率落在 [0,1]", 0.0 <= lsr_a <= 1.0, f"{lsr_a}")

    # --- 2) 罚分信噪比的两个已知基准 ---
    for name, label, expected in [
        ("4个agent_0801_紧迫度", "1 天基准", 0.026),
        ("只接了llm 测30天", "30 天基准", 0.672),
    ]:
        r = hist.load_run(name)
        if r is None:
            check(f"{label} 可加载", False, name)
            continue
        psr = proc.penalty_signal_ratio(hist.penalty_by_cause(r))
        check(
            f"★ {label} 罚分信噪比 = {psr:.1%}（已知基准 {expected:.1%}）",
            abs(psr - expected) <= 0.005,
            f"实测 {psr:.4f}",
        )

    # --- 3) 三个指标都能在历史数据上跑出结果（不是 NotImplementedError）---
    episodes_a = hist.load_episodes(a_name)
    res = proc.run_process(episodes_a, penalty_by_cause=hist.penalty_by_cause(a))
    check("run_process 三个补齐指标全部有值",
          {"loop_signature_rate", "penalty_signal_ratio", "constraint_precision"} <= set(res))
    check(
        "★ constraint_precision 无可行集时是 None 而不是编造的数字",
        res["constraint_precision"] is None,
        f"实测 {res['constraint_precision']}",
    )
    check(
        "penalty_signal_ratio 有真实罚分数据时非 None",
        res["penalty_signal_ratio"] is not None,
    )


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 72)
    print("AgentEval Harness · smoke 自检")
    print(f"仓库根目录：{ROOT}")
    print("=" * 72)

    for fn in (
        s1_limiter, s2_estimator, s3_cache, s4_config,
        s5_end_to_end, s6_wiring, s7_metrics_on_history,
    ):
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
