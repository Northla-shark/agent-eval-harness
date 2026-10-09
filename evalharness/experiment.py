# -*- coding: utf-8 -*-
"""Layer 4 · Experiment —— A/B 对照、消融编排、结果落盘。

这一层本身不产生新指标，它的价值在于"让两次 run 可比且可复述"：
  同一 Scenario、不同 variant -> 差异只允许出现在 variant 上
  不同 Scenario 也能比 -> 但必须先打出 ConfigSnapshot.diff，证明差异字段只有哪些
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .metrics import cost as cost_m
from .metrics import outcome as outcome_m
from .metrics import process as process_m
from .runner import RunResult


def run_metrics(result: RunResult, *, profile_name: str = "no_history") -> dict[str, Any]:
    """把一次 run 的三层指标汇总成一个 JSON-able dict。"""
    llm = dict(result.extra.get("llm", {}))
    return {
        "run_id": result.run_id,
        "variant": result.variant,
        "scenario": result.scenario_name,
        "wall_seconds": round(result.wall_seconds, 3),
        "outcome": outcome_m.run_outcome(result.episodes),
        "process": process_m.run_process(result.episodes),
        "cost": cost_m.run_cost(
            result.episodes,
            profile_name=profile_name,
            # 缓存省下的 token 进成本表：省下的与花掉的必须在同一张表上分开可见。
            saved_tokens=int(llm.get("llm.tokens_saved", 0)),
        ),
        "limiter": result.limiter_counters,
        # LLMGate 的逐次台账（含预估器自身的偏差分布）。D2 起它才进 metrics.json ——
        # 之前 RunResult.to_dict() 里带了但落盘版漏了，等于这一层观测不存在。
        "llm": llm,
    }


def save_run(result: RunResult, out_dir: Path | str = "runs", *, profile_name: str = "no_history") -> Path:
    """落盘。目录名用 run_id 而不是时间戳 —— 时间戳等于没有标识。

    runs/<run_id>/
      ├── snapshot.json   # 配置快照，可被 diff
      ├── metrics.json    # 三层指标
      └── episodes.jsonl  # 每行一个司机，便于 jq / pandas 直接读
    """
    d = Path(out_dir) / result.run_id
    d.mkdir(parents=True, exist_ok=True)

    (d / "snapshot.json").write_text(result.snapshot.to_json(), encoding="utf-8")
    (d / "metrics.json").write_text(
        json.dumps(run_metrics(result, profile_name=profile_name), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (d / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for e in result.episodes:
            f.write(json.dumps({
                "driver_id": e.driver_id,
                "steps": e.n_steps,
                "reward_total": e.reward_total,
                "tokens": e.usage_total.total,
                "stop_reason": e.stop_reason,
                "wall_seconds": round(e.wall_seconds, 3),
            }, ensure_ascii=False) + "\n")
    return d


def compare_table(
    results: Sequence[RunResult],
    *,
    profiles: dict[str, str] | None = None,
) -> str:
    """把多次 run 排成一张 markdown 对照表。

    这张表就是你 README 里「Results」章节的内容 ——
    面试官不读你的动机，他要先看数字。有数字他才回头读动机。
    """
    profiles = profiles or {}
    header = (
        "| 变体 | run_id | 司机数 | 步数(中位) | 总token | 净收益(中位) "
        "| wait率 | 每单调用 | 截断 |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for r in results:
        m = run_metrics(r, profile_name=profiles.get(r.variant, "no_history"))
        proc = m["process"]
        out = m["outcome"]
        ct = m["cost"]
        wait_rate = proc["action_distribution"].get("wait", 0.0)
        truncated = sum(1 for e in r.episodes if e.truncated)
        lines.append(
            f"| {r.variant} | `{r.run_id}` | {out['n_drivers']} "
            f"| {proc['steps_per_driver_median']:.0f} | {ct['tokens_total']:,} "
            f"| {out['reward_median']:.0f} | {wait_rate:.1%} "
            f"| {proc['calls_per_success']} | {truncated} |"
        )
    return "\n".join(lines)


def diff_runs(a: RunResult, b: RunResult) -> dict[str, tuple[Any, Any]]:
    """两次 run 的受控性检验。空 dict = 配置完全相同。

    面试用法：
      「我对比的两组，差异字段只有 X 一个，而且我额外比了决策逻辑的文件哈希。」
    """
    from .config import diff

    return diff(a.snapshot, b.snapshot)
