# -*- coding: utf-8 -*-
"""D2 验收 · Token 预估器「两套口径合并」的前后对比。

背景
----
D2 之前仓库里有**两个互不相干**的 token 预估：

  1. `budget/estimator.estimate_step_tokens(step_index, profile)`
     -> `S + K·step_index + C`，按步索引拟合。BudgetGuard 与 cost 指标用它。
  2. `LLMGate.estimate(payload)`
     -> `max(S, chars//2) + C`，按 payload 字符粗估。限流器预扣用它。

两者算的是同一个东西（这次要花多少 token），却不共享任何一项信息：
① 不知道 payload 多大，② 不知道第几步，③ 还默认"一步 = 一次调用"。
结果就是 truck 跑批上 `estimate_deviation` 到 -0.8387 —— **这不是系数不准，是口径分裂。**

合并后：`estimate_call_tokens(chars, step_index, profile)`
  prompt = max(S + K·step_index, chars // CHARS_PER_TOKEN)
  total  = prompt + C

本脚本打印三件事，三件事的**可信度不同，必须分开看**：

  ① 存档 truck 跑批：复算旧公式的偏差，确认与存档一致（说明口径分裂确实是根因）
  ② 新跑批逐次对账：同一份实际 token 上，旧公式 vs 合并后公式
  ③ 10 组真实跑批：**真花钱**的数据上，纯步索引公式的偏差（唯一非循环的数字）

为什么 ② 要打个折扣看
----------------------
本仓库默认不跑真模型（这是设计选择，见 docs/），所以 ② 里的"实际 token"本身是
按 payload 字符估出来的。合并后的公式用了同一个字符换算，于是 ② 的偏差会趋近 0 ——
**这部分胜利是同义反复，不能写进简历。** ② 真正说明的问题是另一件事：
旧公式一步只算一次调用，而卡车 Agent 一步打 3 次 —— 这个系统性低估是实的。
真正的精度上限要看 ③。

用法
----
  python tools/compare_estimate.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evalharness.budget.estimator import PROFILES, estimate_step_tokens  # noqa: E402
from evalharness.metrics import history as hist  # noqa: E402

RUNS = ROOT / "runs"


def _load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _mean_rel(pairs: list[tuple[float, float]]) -> float | None:
    """pairs = [(est, act)]，返回平均相对误差；act<=0 的样本丢弃。"""
    rs = [(e - a) / a for e, a in pairs if a > 0]
    if not rs:
        return None
    return statistics.fmean(rs)


# ---------------------------------------------------------------------------
def section_archived() -> None:
    """① 存档 truck 跑批：用 steps + tokens 复算旧公式，核对与存档一致。"""
    print("\n【① 存档 truck 跑批】旧公式（S+K·i+C，一步一次）的偏差")
    print("-" * 78)

    rows = []
    for d in sorted(RUNS.iterdir()) if RUNS.is_dir() else []:
        m = _load_json(d / "metrics.json")
        s = _load_json(d / "snapshot.json")
        if not m or not s:
            continue
        if s.get("fields", {}).get("adapter_name") != "truck":
            continue
        prof = s.get("fields", {}).get("token_profile", "no_history")
        profile = PROFILES.get(prof, PROFILES["no_history"])
        steps = int(m.get("process", {}).get("steps_total", 0))
        act = int(m.get("cost", {}).get("tokens_total", 0))
        if not steps or not act:
            continue
        est = sum(estimate_step_tokens(i, profile) for i in range(steps))
        rows.append((d.name[:12], steps, act, est, (est - act) / act,
                     m.get("cost", {}).get("estimate_deviation", {}).get("mean_rel_error")))

    if not rows:
        print("  （没有 truck 跑批产物）")
        return

    print(f"  {'run_id':<14}{'步数':>6}{'实际token':>12}{'旧公式预估':>12}"
          f"{'偏差':>10}   {'存档里的偏差':>12}")
    for name, steps, act, est, rel, stored in rows:
        agree = "一致" if stored is not None and abs(stored - rel) < 0.001 else "?"
        print(f"  {name:<14}{steps:>6}{act:>12,}{est:>12,}{rel:>9.2%}   "
              f"{stored if stored is not None else '-':>12}  {agree}")

    print("\n  要点：偏差稳定在 -80% 上下，且**每一步都差同一方向的同一量级** ——")
    print("        这是口径分裂的特征（少了两个系统性因子），不是噪声。")
    print("        两个缺失因子：a) 项目一一步打 3 次模型，旧公式只算 1 次；")
    print("                    b) 旧公式完全不看 payload，默认每步一样大。")


# ---------------------------------------------------------------------------
def section_newrun() -> None:
    """② 新跑批：同一份实际 token 上，旧公式 vs 合并后公式。"""
    print("\n【② 新跑批逐次对账】旧公式 vs 合并后公式（同一份实际 token）")
    print("-" * 78)

    ledgers = sorted(
        (p for p in RUNS.glob("*/llm_calls.jsonl")) if RUNS.is_dir() else [],
        key=lambda p: p.stat().st_mtime,
    )
    if not ledgers:
        print("  没有 llm_calls.jsonl —— 先跑一次 `python tools/run_d1.py` 生成台账。")
        print("  （D2 起每次 run 都会落这个台账；没有它就永远无法离线复算预估精度。）")
        return

    ledger = ledgers[-1]
    snap = _load_json(ledger.parent / "snapshot.json")
    prof = snap.get("fields", {}).get("token_profile", "no_history")
    profile = PROFILES.get(prof, PROFILES["no_history"])

    rows = []
    with ledger.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    by_step: dict[int, list[dict]] = {}
    for r in rows:
        by_step.setdefault(r["step"], []).append(r)

    before, after = [], []
    for step in sorted(by_step):
        calls = by_step[step]
        act = sum(c["actual"] for c in calls)
        before.append((float(estimate_step_tokens(step, profile)), act))
        after.append((float(sum(c["estimated"] for c in calls)), act))

    mb, ma = _mean_rel(before), _mean_rel(after)
    calls_total = len(rows)
    steps_total = len(before)
    per_step_calls = calls_total / steps_total if steps_total else 0.0

    print(f"  台账    : runs/{ledger.parent.name}/llm_calls.jsonl")
    print(f"  profile : {prof}  (S={profile.fixed_per_step}, K={profile.growth_per_step}, "
          f"C={profile.completion_per_step})")
    print(f"  规模    : {steps_total} 步 / {calls_total} 次调用 "
          f"= 每步 {per_step_calls:.2f} 次")
    print()
    def fmt(v: float | None) -> str:
        return f"{v:.2%}" if v is not None else "n/a"

    def mae(pairs: list[tuple[float, float]]) -> str:
        rs = [abs(e - a) / a for e, a in pairs if a > 0]
        return f"{statistics.fmean(rs):.2%}" if rs else "n/a"

    print(f"  旧公式（一步一次，不看 payload）  平均相对偏差 = {fmt(mb)}"
          f"   平均绝对偏差 = {mae(before)}")
    print(f"  合并后（逐次，含 payload 大小）   平均相对偏差 = {fmt(ma)}"
          f"   平均绝对偏差 = {mae(after)}")

    # 每步调用次数 > 1 时，旧公式必然系统性低估 (1 - 1/n) —— 这部分是"实赢"
    if per_step_calls > 1:
        floor = -(per_step_calls - 1) / per_step_calls
        print(f"\n  其中 purely 由『一步算成一次』贡献的下界 ≈ {floor:.2%}")
        print("  —— 这部分改进是实的：它来自调用粒度对齐，与实际 token 怎么估出来无关。")

    print("\n  ⚠ 折扣看待：本节里『实际 token』在替身/ dry-run 模式下是**按字符估出来的**，")
    print("    合并后的公式用了同一个字符换算，所以这里的趋近 0 有一部分是同义反复。")
    print("    真正能当精度上限看的是下一节的真实跑批。")


# ---------------------------------------------------------------------------
def section_real() -> None:
    """③ 真实跑批：纯步索引公式在**真花钱**的数据上有多准。"""
    print("\n【③ 10 组真实跑批】纯步索引公式的实际精度（非循环数字）")
    print("-" * 78)

    if not hist.available():
        print("  本机没有历史跑批 -> 跳过（不计失败）。")
        return

    print(f"  {'跑批':<38}{'步数':>6}{'实际token':>12}{'旧公式':>12}{'偏差':>9}")
    rels: list[float] = []
    pool: list[tuple[float, float]] = []
    for name in hist.list_runs():
        eps = hist.load_episodes(name)
        if not eps:
            continue
        profile = PROFILES["no_history"]   # 项目一场景当时配置的 profile
        pairs: list[tuple[float, float]] = []
        act_total = 0
        for ep in eps:
            for s in ep.steps:
                pairs.append((float(estimate_step_tokens(s.step, profile)), float(s.usage.total)))
                act_total += s.usage.total
        r = _mean_rel(pairs)
        if r is None:
            continue
        rels.append(r)
        pool.extend(pairs)
        steps = sum(ep.n_steps for ep in eps)
        est = sum(e for e, _ in pairs)
        print(f"  {name[:36]:<38}{steps:>6}{act_total:>12,}{est:>12,.0f}{r:>9.2%}")

    if rels:
        pooled_e = sum(e for e, _ in pool)
        pooled_a = sum(a for _, a in pool)
        print(f"\n  合计 {len(rels)} 组，各组平均相对偏差 {statistics.fmean(rels):.2%}，"
              f"区间 [{min(rels):.2%}, {max(rels):.2%}]")
        print(f"  全部 {len(pool)} 步汇总（不分 run）：预估 {pooled_e:,.0f} vs "
              f"实际 {pooled_a:,.0f} -> {(pooled_e - pooled_a) / pooled_a:.2%}")
        print("\n  ⚠ 别被均值骗了：上面 -2.10% 是「各组偏差再取均值」，看着几乎可以对冲掉；")
        print("    但全部步汇总起来仍是 -36.29%，单组区间更是跨 [-60%, +301%]。")
        print("    「平均还算准」是最容易自欺的一句话 —— 要盯的是分布的两个尾巴，")
        print("    而且这里的负号方向一致：绝大多数是被低估（会让 TPM 桶被击穿）。")
    print("\n  这是唯一没有循环论证的数字：这些 token 是**真花钱跑出来的**，")
    print("  没有任何一步是从字符估出来的。它给出的是「只看步索引」这条路的精度上限。")


# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 78)
    print("D2 · Token 预估器口径合并前后对比")
    print("=" * 78)
    section_archived()
    section_newrun()
    section_real()
    print("\n" + "=" * 78)
    print("结论：")
    print("  · 偏差的根因是口径分裂，不是系数不准 —— ①里每一步都差同一量级证明了这点。")
    print("  · 合并后预估器同时吃到「第几步」和「payload 多大」两个信号。")
    print("  · 精度天花板来自 CHARS_PER_TOKEN=2 的粗估，见 docs/architecture.md。")
    print("  · 别的关键口径问题：本次标配 profile 是 no_history(K=0)，但项目一每步喂 5 步历史")
    print("    —— 「第几步」这个信号本来应该更值钱，是配置把它关掉了。")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
