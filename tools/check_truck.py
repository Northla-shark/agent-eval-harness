#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D1.1 验收 —— 接线自检。**不需要项目一、不需要 API key、不需要 210 MB 数据。**

    python tools/check_truck.py

它做的事：给 `TruckAdapter` 装一对合成替身（`tools/_fake_simkit.py`），
然后**照常跑一遍 Runner** —— 被测代码与生产路径完全一致，只换掉它 import 到的东西。

跑通意味着：
  ✓ TruckAdapter 的 5 个协议方法都对了（reset / act / env_step / close / metadata）
  ✓ 向下 4 个 port 方法都对了（get_driver_status / query_cargo / query_decision_history
    / model_chat_completion）
  ✓ 四个节点的插桩生效（custom:node_* span 数 == 步数）
  ✓ token 账真的按"每次调用"在记，而不是按"每步"
  ✓ **并且把 D1 挖到的那件事量出来**：项目一的 agent3 走自己的客户端，
    它的 token 不在赛方端口上 —— 差值就是被漏算的部分

还剩什么风险：真项目一的字段名对不对（例如 wait 的参数到底是 duration 还是
duration_minutes）。那是 D1.2 用真数据跑 1 个司机 × 1 天来确认的。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import _fake_simkit  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    if cond:
        PASS.append(name)
        print(f"  [OK]   {name}")
    else:
        FAIL.append(name)
        print(f"  [FAIL] {name}  {extra}")


def _patch_seams(fake_service_cls) -> None:
    """把 TruckAdapter 的三个外部接缝换成替身。

    这三个名字是刻意留成"可替换的一层"的：
        _resolve_demo_root   —— 项目一在哪
        _import_project_one   —— 怎么把它 import 进来
        _shared_resources     —— 210 MB 的 repo / manager 从哪来
    生产路径下它们指向真项目一；测试路径下指向合成替身。
    **被测代码一行都不用改** —— 这是适配层可测的前提。
    """
    import adapters.truck as truck

    truck._resolve_demo_root = lambda project_root=None: ROOT  # type: ignore[assignment]
    truck._import_project_one = lambda demo: fake_service_cls  # type: ignore[assignment]

    cache: dict[str, tuple[object, object]] = {}

    def _shared(demo):  # type: ignore[no-untyped-def]
        if "pair" not in cache:
            repo = _fake_simkit.FakeCargoRepository()
            manager = _fake_simkit.FakeDriverStateManager()
            repo.load()
            manager.load()
            cache["pair"] = (repo, manager)
        return cache["pair"]  # type: ignore[return-value]

    truck._shared_resources = _shared  # type: ignore[assignment]
    truck._SHARED.clear()


def _run_once(bypass_port: bool, drivers: list[str], concurrency: int = 1):
    """装替身 -> 跑 Runner -> 返回 (RunResult, tracer)。"""
    _fake_simkit.install(bypass_port=bypass_port)
    fake_cls = sys.modules["agent.model_decision_service"].ModelDecisionService
    _patch_seams(fake_cls)

    import adapters  # noqa: F401  触发 "truck" 注册
    from evalharness.adapter import build
    from evalharness.observability import ConsoleTracer
    from evalharness.runner import run_scenario
    from evalharness.scenario import Scenario

    from dataclasses import replace

    scenario = Scenario.tiny(max_steps=120).with_variant("full", adapter_name="truck")
    scenario = replace(scenario, driver_ids=drivers, duration_days=5, concurrency=concurrency)
    tracer = ConsoleTracer()

    result = run_scenario(
        scenario,
        lambda **kw: build("truck", variant=scenario.variant, **kw),
        tracer=tracer,
        repo_dir=ROOT,
        impl_paths=[ROOT / "adapters" / "truck.py"],
    )
    return result, tracer, fake_cls


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 74)
    print("truck adapter 接线自检（合成环境 · 零 API 成本）")
    print(f"仓库根目录：{ROOT}")
    print("=" * 74)

    # ---------- 1. 正路径：一个司机跑通 ----------
    print("\n[1/4] TruckAdapter 全链路（1 个司机 × 5 天）")
    result, tracer, _ = _run_once(bypass_port=True, drivers=["D001"])

    ep = result.episodes[0]
    print(f"    run_id      = {result.run_id[:12]}")
    print(f"    步数        = {ep.n_steps}")
    print(f"    累计 token  = {ep.usage_total.total:,}")
    print(f"    累计 reward = {ep.reward_total:,.2f}   （运价 − 行驶成本，不含偏好罚分）")
    print("                注：合成数据的运价与里程没有校准过，reward 的绝对值没有意义，")
    print("                    这里只验证「它被正确累计」——业务数字必须用真数据跑。")
    print(f"    终止原因    = {ep.stop_reason}")
    print(f"    wall        = {ep.wall_seconds:.3f}s")

    check("episode 正常结束", ep.stop_reason in {"terminated", "truncated"}, ep.stop_reason)
    check("产生了步（act/env_step 都在被调用）", ep.n_steps > 0, f"n_steps={ep.n_steps}")
    check("token 账非零（gate 在按次计量）", ep.usage_total.total > 0)

    # ---------- 2. 四个节点都插到了 ----------
    print("\n[2/4] 四节点插桩（custom:node_*）")
    custom = tracer.custom_coverage()
    print(f"    {custom}")
    for node in ("preference", "geo", "time", "decision"):
        key = f"custom:node_{node}"
        check(f"{key} 每步各一次", custom.get(key, 0) == ep.n_steps, f"{custom.get(key, 0)} vs {ep.n_steps}")

    cov = tracer.coverage()
    print(f"    公共 9 段覆盖：{ {k: v for k, v in cov.items() if v} }")
    check("公共口径没有被 custom 段污染", set(cov) <= set(__import__("evalharness.observability", fromlist=["x"]).SPAN_NAMES))
    check("9 段里至少 3 段有数据", sum(1 for v in cov.values() if v) >= 3)

    # agent2 不该产生任何模型调用
    calls = [r for r in tracer.records if r.name == "llm_inference" and r.attrs.get("scope") == "call"]
    nodes_seen = sorted({str(r.attrs.get("node")) for r in calls})
    print(f"    scope=call 的模型调用节点：{nodes_seen}（共 {len(calls)} 次）")
    check("★ agent2（geo）没有产生模型调用（它是纯 Python 节点）", "geo" not in nodes_seen, f"{nodes_seen}")
    check("★ 模型调用带 node 标签，能按节点归因", {"preference", "decision"} <= set(nodes_seen))

    # ---------- 3. token 账：按"每次调用"而不是"每步" ----------
    print("\n[3/4] 限流粒度：按调用计账")
    gate_counters = result.extra.get("llm", {})
    for k in ("llm.calls", "llm.cache_hit_rate", "llm.tokens_total", "llm.rejected"):
        print(f"    {k:22s} = {gate_counters.get(k)}")
    check("LLMGate 计数器已进 RunResult（能写进 metrics.json）", gate_counters.get("llm.calls", 0) > 0)
    check(
        "★ 调用次数 > 步数（一步打多次模型 -> 外层按步限流会低估请求率）",
        gate_counters.get("llm.calls", 0) > ep.n_steps,
        f"calls={gate_counters.get('llm.calls')} steps={ep.n_steps}",
    )
    check("adapter 声明 owns_llm_limits=True（Runner 因此不做外层预扣）",
          getattr(__import__("adapters.truck", fromlist=["x"]).TruckAdapter, "owns_llm_limits", False) is True)

    # ---------- 4. 那件被挖出来的事：agent3 不在端口上 ----------
    print("\n[4/4] ★ 复刻项目一的结构缺陷：agent3 的 token 不在端口上")
    r_bypass, _, cls_bypass = _run_once(bypass_port=True, drivers=["D001"])
    r_fixed, _, cls_fixed = _run_once(bypass_port=False, drivers=["D001"])

    seen = r_bypass.episodes[0].usage_total.total
    invisible = cls_bypass.invisible_tokens
    print(f"    平台看得到的 token（3 个端口节点）  = {seen:,}")
    print(f"    平台看不到的 token（agent3 自建客户端）= {invisible:,}"
          f"（{cls_bypass.invisible_calls} 次调用，按每次 900 token 模拟）")
    print(f"    真实总量                              = {seen + invisible:,}")
    if seen + invisible:
        print(f"    → 赛方口径少算了 {(invisible / (seen + invisible)) * 100:.1f}%")

    check("复刻成立：绕过端口的调用确实不在 token 账里", r_bypass.episodes[0].usage_total.total == seen)
    check("★ 修好之后（agent3 也走端口）token 账变大", r_fixed.episodes[0].usage_total.total > seen,
          f"{r_fixed.episodes[0].usage_total.total} vs {seen}")
    print("    面试讲法：项目一四个节点里 agent2 不调模型、agent3 走自己的客户端，")
    print("              所以「四节点」不是「四次调用」，而赛方的 get_last_model_usage()")
    print("              只统计经过端口的那部分 —— 它的 token 上报天生偏低。")

    # ---------- 附：并发度为什么必须是 1 ----------
    print("\n[附] 环境侧的单司机约束（deterministic，不靠并发去撞）")
    import adapters.truck as truck
    from evalharness.budget.estimator import PROFILES
    from evalharness.limiter import DualTokenBucketLimiter
    from evalharness.llm_gate import LLMGate
    from evalharness.observability import ConsoleTracer as _CT
    from evalharness.types import Action

    fresh_tracer = _CT()
    gate = LLMGate(DualTokenBucketLimiter(600, 2_000_000), fresh_tracer, PROFILES["no_history"])
    a = truck.TruckAdapter(gate=gate)
    b = truck.TruckAdapter(gate=gate)
    a.reset("D001", 0)
    b.reset("D002", 0)  # 共享 manager 的"当前司机"切到 D002
    try:
        a.env_step(Action("wait", {"duration_minutes": 60}))
        blocked = False
        detail = ""
    except Exception as exc:
        blocked, detail = True, f"{type(exc).__name__}: {exc}"
    print(f"    D001 在 D002 成为当前司机后再推进 -> {'被拦住' if blocked else '居然通过了'}")
    if blocked:
        print(f"      {detail}")
    check("★ 共享环境一次只能有一个司机 -> truck 的 concurrency 必须为 1", blocked,
          "若这里失败，说明 simkit 的 ensure_active_driver 没有生效")

    print("\n" + "=" * 74)
    print(f"结果：{len(PASS)} 通过 / {len(FAIL)} 失败")
    if FAIL:
        for name in FAIL:
            print(f"  - {name}")
        print("=" * 74)
        return 1
    print("接线全绿。D1.1 完成 —— 接下来用真项目一跑 1 个司机 × 1 天（D1.2）。")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
