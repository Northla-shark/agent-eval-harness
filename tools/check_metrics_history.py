#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D4 验收工具：用项目一历史跑批验证指标层的一致性。

    python tools/check_metrics_history.py                 # 跑默认 4 组
    python tools/check_metrics_history.py --all           # 跑全部
    python tools/check_metrics_history.py "只接了llm 测30天"

它不跑任何仿真，只读历史产物（只读，不修改被测 Agent 目录）。
项目一不在本机时打印跳过说明并以 0 退出 —— **平台不能依赖被测对象的私有目录才能自检**。

为什么要有这个工具（一句话）：
    用历史数据验证指标层的一致性，而不是新跑一次 —— 新跑一次只能验证一次。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evalharness.metrics import history as hist  # noqa: E402
from evalharness.metrics import process as proc  # noqa: E402

# 默认跑的四组：A/B 对照 + 两个不同长度的基准（信噪比对比用）
DEFAULT_RUNS = [
    "上传偏好+历史5步+休息时间统计 补充三种prompt+要求输出秒数",
    "上传偏好+历史5步+休息时间统计 补充三种prompt+要求输出秒数-qwen3.7-max",
    "4个agent_0801_紧迫度",
    "只接了llm 测30天",
]

# 信噪比的两个已知基准（docs/metrics.md 第 3 节的来源）
SNR_BASELINES = [
    ("4个agent_0801_紧迫度", "1 天基准", 0.026),
    ("只接了llm 测30天", "30 天基准", 0.672),
]

OK = 0
BAD = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global OK, BAD
    if cond:
        OK += 1
        print(f"  [OK]   {name}")
    else:
        BAD += 1
        print(f"  [FAIL] {name}  {extra}")


def ratio(x: float, y: float) -> str:
    return f"{x / y:.2f}" if y else "—"


def main() -> int:
    ap = argparse.ArgumentParser(description="用历史跑批验证指标层")
    ap.add_argument("runs", nargs="*", help="跑批目录名；不传则用默认 4 组")
    ap.add_argument("--all", action="store_true", help="跑全部跑批")
    ap.add_argument("--root", default=hist.DEFAULT_RESULTS_ROOT, help="demo/results 路径")
    ap.add_argument("--window", type=int, default=8, help="震荡率滑窗")
    args = ap.parse_args()

    print("=" * 78)
    print("D4 验收 · 指标层 × 历史跑批一致性")
    print(f"数据源：{args.root}（只读）")
    print("=" * 78)

    if not hist.available(args.root):
        print("\n[跳过] 本机找不到历史跑批（或目录下没有 monthly_income_202603.json）。")
        print("       这不是失败：clone 本仓库的人没有项目一，指标层仍应可用。")
        print("       要跑这一段，把 --root 指向含跑批产物的目录。")
        return 0

    names = hist.list_runs(args.root) if args.all else (args.runs or DEFAULT_RUNS)

    print(f"\n【按跑批归因】（震荡率滑窗 window={args.window}）")
    print(
        "| 跑批 | 天数 | 步数 | wait率 | 每单调用 | 震荡率 | 罚分信噪比 | 步数对账 |"
        "\n|---|---|---|---|---|---|---|---|"
    )
    loaded: dict[str, dict] = {}
    for name in names:
        r = hist.load_run(name, args.root)
        if r is None:
            print(f"| {name} | — | — | — | — | — | — | 找不到产物 |")
            continue
        loaded[name] = r
        episodes = hist.load_episodes(name, args.root)
        all_steps = [s for e in episodes for s in e.steps]
        dist = proc.action_distribution(all_steps)
        wait_rate = dist.get("wait", 0.0)
        psr = proc.penalty_signal_ratio(hist.penalty_by_cause(r))
        lsr = proc.loop_signature_rate(all_steps, window=args.window)
        per_order = ratio(r["llm_calls"], r["accepted_orders"])
        cons = r["steps_vs_summary_consistent"]
        print(
            f"| {name[:34]} | {r['days']} | {r['steps']} | {wait_rate:.1%} "
            f"| {per_order} | {lsr:.4f} "
            f"| {'—' if psr != psr else f'{psr:.1%}'} "
            f"| {'一致' if cons else ('不一致' if cons is False else '无 summary')} |"
        )

    # --- 震荡指纹：到底哪个签名在重复（否则震荡率无从对账）---
    print("\n【震荡指纹 top3】（项目一 A 组的已知真值：wait(duration=60) 出现 146 次）")
    for name in names:
        if name not in loaded:
            continue
        episodes = hist.load_episodes(name, args.root)
        all_steps = [s for e in episodes for s in e.steps]
        top = list(proc.signature_counts(all_steps).items())[:3]
        total = max(1, len(all_steps))
        pretty = ", ".join(f"{k}={v} ({v / total:.1%})" for k, v in top)
        print(f"  {name[:40]:42} {pretty}")

    # --- 罚分分层 ---
    print("\n【罚分分层】（A=基准内不可能 / B=结构性冲突 / C=决策质量）")
    for name in names:
        if name not in loaded:
            continue
        r = loaded[name]
        causes = hist.penalty_by_cause(r)
        total = sum(causes.values())
        if not total:
            print(f"  {name[:40]:42} 无罚分")
            continue
        parts = "  ".join(
            f"{k}={v:,.0f} ({v / total:.1%})" for k, v in sorted(causes.items())
        )
        print(f"  {name[:40]:42} {parts}")

    # --- 信噪比基准对账 ---
    print("\n【信噪比基准对账】")
    snrs: dict[str, float] = {}
    for name, label, expected in SNR_BASELINES:
        r = loaded.get(name) or hist.load_run(name, args.root)
        if r is None:
            print(f"  [跳过] {label}：找不到跑批 {name}")
            continue
        causes = hist.penalty_by_cause(r)
        psr = proc.penalty_signal_ratio(causes)
        # 倍数要用未舍入的比值：舍到 4 位再相除会把 25.5 倍算成 25.6 倍
        snrs[label] = causes.get("decision_quality", 0.0) / sum(causes.values())
        ok = psr == psr and abs(psr - expected) <= 0.005
        check(
            f"{label} 信噪比 = {psr:.1%}（已知基准 {expected:.1%}）",
            ok,
            f"实测 {psr:.4f}",
        )

    if len(snrs) == 2:
        lo, hi = min(snrs.values()), max(snrs.values())
        print(f"  => 30 天基准的可优化信号占比是 1 天基准的 {hi / lo:.1f} 倍")

    print("\n" + "=" * 78)
    print(f"结果：{OK} 通过 / {BAD} 失败")
    print("=" * 78)
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
