#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D1.2 —— 用**真项目一**跑 1 个司机 × 1 天，把四个节点各自的耗时与 token 打出来。

    rem 先看能不能打通（不打模型，零成本）：
    .venv\\Scripts\\python.exe tools\\run_d1.py

    rem 真打模型（需要 DASHSCOPE_API_KEY，会产生费用）：
    .venv\\Scripts\\python.exe tools\\run_d1.py --live

    rem 换司机 / 换天数：
    .venv\\Scripts\\python.exe tools\\run_d1.py --driver D003 --days 2 --live

D1 的验收标准就是最后那张「按节点」的表：
**某司机某一步下挂的四个节点，各自的耗时与 token 都能看见。**
在那之前，项目一的两组跑批只有目录名能证明差异 —— 这张表是"可归因"的第一步。

为什么默认不打真模型：dry-run 会把整条链路（Adapter 5 方法 + port 4 方法 + 节点插桩 +
限流 + 缓存）全部走一遍，只有"模型答得对不对"没验。先花 0 元排掉所有接线错误，
再花一次钱验模型 —— 而不是反过来。
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import adapters  # noqa: E402,F401  触发 "truck" 注册
from evalharness.adapter import build  # noqa: E402
from evalharness.budget.estimator import PROFILES  # noqa: E402
from evalharness.experiment import save_run  # noqa: E402
from evalharness.observability import ConsoleTracer  # noqa: E402
from evalharness.runner import run_scenario  # noqa: E402
from evalharness.scenario import Scenario  # noqa: E402
from evalharness.stub import build_scripted_transport  # noqa: E402
from evalharness.transport import OpenAICompatibleTransport  # noqa: E402

DEFAULT_SCENARIO = ROOT / "scenarios" / "truck-driver-d1-smoke.yaml"


def node_table(tracer: ConsoleTracer) -> str:
    """按节点聚合：wall 耗时（来自 custom:node_*）+ token（来自 llm_inference 的 node 标签）。

    为什么要这张表：`llm_inference` 只有一个总耗时的时候，你只能说"这一步慢"；
    拆到节点才能说"慢在时间分析，而且它占了 63% 的 token"。
    **可归因是评测平台和被测量脚本的分界线。**
    """
    wall_ms: dict[str, float] = defaultdict(float)
    calls: dict[str, int] = defaultdict(int)
    tokens: dict[str, int] = defaultdict(int)
    for r in tracer.records:
        if r.name.startswith("custom:node_"):
            wall_ms[r.name[len("custom:node_"):]] += r.ms
        elif r.name == "llm_inference" and r.attrs.get("scope") == "call":
            node = str(r.attrs.get("node", "unknown"))
            calls[node] += 1
            tokens[node] += int(r.attrs.get("model_usage_input", 0)) + int(
                r.attrs.get("model_usage_output", 0)
            ) + int(r.attrs.get("model_usage_reasoning", 0))

    nodes = sorted(set(wall_ms) | set(calls))
    if not nodes:
        return "（没有节点记录 —— 插桩没生效，检查 _install_node_probes）"

    total_tokens = sum(tokens.values()) or 1
    total_ms = sum(wall_ms.values()) or 1.0
    head = f"{'节点':<12}{'耗时(ms)':>12}{'占比':>8}{'模型调用':>10}{'token':>12}{'token占比':>10}"
    lines = [head, "-" * len(head)]
    for n in nodes:
        lines.append(
            f"{n:<12}{wall_ms.get(n, 0.0):>12.1f}{wall_ms.get(n, 0.0) / total_ms * 100:>7.1f}%"
            f"{calls.get(n, 0):>10}{tokens.get(n, 0):>12,}{tokens.get(n, 0) / total_tokens * 100:>9.1f}%"
        )
    lines.append("-" * len(head))
    lines.append(
        f"{'合计':<12}{total_ms:>12.1f}{100.0:>7.1f}%{sum(calls.values()):>10}{sum(tokens.values()):>12,}{100.0:>9.1f}%"
    )
    return "\n".join(lines)


def action_table(history: list[dict[str, Any]]) -> str:
    """环境实际执行过的动作分布 —— 这是"链路真的被走通"的证据。

    为什么要有这张表：只跑 wait 的一次运行也能让上面的节点表全绿，
    但 `take_order` / `reposition` 两条路径一次都没进过。
    节点表证明"我看见了"，动作表证明"它真的做了"。
    """
    if not history:
        return "（没有历史记录）"
    counts: dict[str, int] = {}
    accepted = 0
    for h in history:
        act = h.get("action")
        name = str(act.get("action", "?")) if isinstance(act, dict) else "?"
        counts[name] = counts.get(name, 0) + 1
        res = h.get("result")
        if isinstance(res, dict) and res.get("accepted"):
            accepted += 1
    total = sum(counts.values()) or 1
    lines = [f"{'动作':<14}{'次数':>8}{'占比':>9}", "-" * 31]
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"{name:<14}{n:>8}{n / total * 100:>8.1f}%")
    lines.append("-" * 31)
    lines.append(f"{'合计':<14}{total:>8}{100.0:>8.1f}%   环境接受 {accepted} 次")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="D1.2：真项目一 × 1 个司机 × 1 天")
    ap.add_argument("--scenario", default=str(DEFAULT_SCENARIO))
    ap.add_argument("--driver", default=None, help="覆盖 driver_ids，例如 D003")
    ap.add_argument("--days", type=int, default=None, help="覆盖 duration_days")
    ap.add_argument("--project-root", default=None, help="项目一根目录（默认取 TRUCK_REPO 或 D:\\TJU\\4个agent_0708）")
    ap.add_argument("--live", action="store_true", help="真打模型（会花钱）。默认 dry-run。")
    ap.add_argument("--api-key", default=None, help="不传则读环境变量 DASHSCOPE_API_KEY")
    ap.add_argument("--model", default=None, help="覆盖 model_name")
    ap.add_argument(
        "--stub-debug",
        default=None,
        help="替身构造不出参数时，把当时的 prompt 落盘到这个路径（排查用）",
    )
    ap.add_argument(
        "--no-intercept",
        action="store_true",
        help="不接管 agent3 的自建客户端（复现赛方口径）。默认接管，见 docs/d1-wiring.md 第 6 节。",
    )
    args = ap.parse_args()

    scenario = Scenario.from_file(args.scenario)
    if args.driver:
        from dataclasses import replace

        scenario = replace(scenario, driver_ids=[args.driver])
    if args.days:
        from dataclasses import replace

        scenario = replace(scenario, duration_days=args.days)
    if args.model:
        from dataclasses import replace

        scenario = replace(scenario, model_name=args.model)

    print("=" * 78)
    print("D1.2 · 真项目一接入验证")
    print(f"  场景      : {scenario.name}  ({scenario.duration_days} 天 × {len(scenario.driver_ids)} 司机)")
    print(f"  司机      : {scenario.driver_ids}")
    print(f"  adapter   : {scenario.adapter_name}   concurrency={scenario.concurrency}")
    print(f"  模式      : {'LIVE（会花钱）' if args.live else 'DRY-RUN（不花钱）'}")
    print(f"  自建客户端 : {'原样放行（赛方口径，time 节点 token 不可见）' if args.no_intercept else '接管进网关（默认）'}")
    print("=" * 78)

    transport = None
    stub = None
    if args.live:
        key = args.api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        transport = OpenAICompatibleTransport(
            scenario.model_api_url or "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            key,
            model=scenario.model_name,
        )
    else:
        # 默认：脚本化替身。不打 API，但会给出**解析得出来的合法决策**，
        # 于是 take_order / reposition / wait 三条分支都会被真正走一遍 ——
        # 而不像返回 "{}" 那样每步都退化成 wait、三分之二的代码路径从没被执行。
        stub = build_scripted_transport(debug_dump=args.stub_debug)
        transport = stub
        print("\n模式说明：不打真模型。替身按节点返回合法 JSON，动作在")
        print("          take_order / reposition / wait 之间轮转。")
        print("          验的是「链路 + 环境对三种动作的处理」，不是「模型聪不聪明」。")
        print("          token 为按字符数估算，绝对值无意义，相对分布有结构意义。\n")

    tracer = ConsoleTracer()

    # 留一份 adapter 引用：跑完之后要读它的 history 统计"环境真正执行了哪些动作"。
    # 用 history 而不是 stub 的脚本，是因为**脚本说的是我想让它做什么，history 说的是它真做了什么**。
    made: list[Any] = []

    def factory(**kw):
        adapter = build(
            scenario.adapter_name,
            variant=scenario.variant,
            project_root=args.project_root,
            duration_days=scenario.duration_days,
            intercept_foreign_clients=not args.no_intercept,
            **kw,
        )
        made.append(adapter)
        return adapter

    try:
        result = run_scenario(
            scenario,
            factory,
            tracer=tracer,
            repo_dir=ROOT,
            impl_paths=[ROOT / "adapters" / "truck.py"],
            llm_transport=transport,
        )
    except FileNotFoundError as exc:
        print(f"\n[中断] {exc}")
        return 2

    ep = result.episodes[0]
    print(f"\nrun_id = {result.run_id[:12]}   wall = {result.wall_seconds:.2f}s")
    print(f"步数 = {ep.n_steps}   终止原因 = {ep.stop_reason}")
    print(f"累计 token = {ep.usage_total.total:,}   （in {ep.usage_total.input_tokens:,}"
          f" / out {ep.usage_total.output_tokens:,} / reasoning {ep.usage_total.reasoning_tokens:,}）")
    print(f"累计 reward = {ep.reward_total:,.2f}")

    print("\n【按节点归因】")
    print(node_table(tracer))

    if made:
        print("\n【环境实际执行的动作】")
        print(action_table(made[0].history))
    if stub is not None:
        print(f"\n替身自述：{stub.counters()}")
        if stub.fallbacks:
            print(f"  ⚠ 降级 {len(stub.fallbacks)} 次：{stub.fallbacks[:3]}")

    cov = tracer.coverage()
    print(f"\n公共 9 段覆盖：{ {k: v for k, v in cov.items() if v} }")
    print(f"被测对象自报阶段：{tracer.custom_coverage()}")
    print(f"限流器：{result.limiter_counters}")
    print(f"LLM 网关：{result.extra.get('llm', {})}")

    out = save_run(result, out_dir=ROOT / "runs", profile_name=scenario.token_profile)
    print(f"\n已落盘：{out}")

    print("\n" + "=" * 78)
    if args.no_intercept:
        print("D1 验收检查（--no-intercept：复现赛方口径）：")
        print("  注意：这个模式必须配 --live —— 不接管时 agent3 会自己发 HTTP，")
        print("        dry-run 下它会因没有 key 而抛错。这是它「不受平台保护」的直接证据。")
        print("  [ ] time 节点 token 为 0 -> 这就是赛方统计漏记的那部分")
        print("  [ ] 与接管模式的 token 合计之差 = 漏记量")
    else:
        print("D1 验收检查：")
        print("  [ ] 【按节点归因】表里 4 个节点都有耗时")
        print("  [ ] 【动作分布】表里 take_order / reposition / wait 三种都出现过")
        print("      —— 只有 wait 说明替身没生效，env_step 有两条分支从未被执行")
        print("  [ ] preference / time / decision 三个节点 token 都非零")
        print("  [ ] geo 的 token 为 0 -> **符合预期**：agent2 是纯 Python 节点，不调模型")
        print("  [ ] llm.calls_foreign > 0 -> 说明 agent3 的自建客户端被接管了；")
        print("      这个数越大，赛方口径漏记得越多（见 docs/d1-wiring.md 第 6 节）")
        print("  [ ] stop_reason 是 terminated 或 budget:xxx（不是 limiter_starved）")
        if not args.live:
            print("\n  数字来源（面试被追问时必须说得清）：")
            print("    · 耗时        —— **实测**（perf_counter 包住 transport 调用）")
            print("    · 模型调用数  —— **实测**（网关计数）")
            print("    · token       —— **估算**（字符数 / 2），绝对值不是账单，")
            print("                     相对占比有意义，因为它量的是各节点 prompt 的真实大小")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
