# -*- coding: utf-8 -*-
"""历史跑批接入插件 —— 由项目一的 `repro_metrics.py` 重构而来。★ D4

为什么是"接历史数据"而不是"新跑一次仿真"（这是本模块的立项理由，别改）：
    **用历史数据验证指标层的一致性，而不是新跑一次 —— 新跑一次只能验证一次。**
    指标层要证明的不是"这次没崩"，而是"同一套口径在不同来源的数据上给出同样的结论"。
    新跑一次只提供一个样本点，且与实现同源 —— 它对口径错误是免疫的（错也一起错）。
    历史跑批是**已经发生过**的独立数据，且零成本。

读什么（全部只读，不修改被测 Agent 目录下的任何文件）：
    <root>/<跑批名>/monthly_income_202603.json      官方评分输出（净收益 / 罚分 / 逐司机明细）
    <root>/<跑批名>/run_summary_202603.json         仿真元信息（天数 / 步数）
    <root>/<跑批名>/actions_202603_D0xx_*.jsonl     逐步动作日志（动作 / 参数 / token）

不读什么（合规红线，写死在这里）：
    `cargo_dataset.jsonl` / `drivers.json` / 仿真原始日志。
    本模块只从**动作序列**和**罚分分层**取数，两者都不含赛题数据，
    且本模块**没有任何写文件的代码路径** —— 读到的东西不会落盘、不会进仓库。

设计约束：
    1. 项目一不在本机时（clone 仓库的人），`available()` 返回 False，
       上层必须跳过而不是报错 —— 平台不能依赖被测对象的私有目录才能自检。
    2. 只喂 **Process 层**。历史日志里逐步 reward 与 token 拆分不可得，
       所以不伪造它们去填 Outcome / Cost 层；那两层的历史数字直接读官方 summary 字段。
"""
from __future__ import annotations

import glob
import json
import os
from typing import Any, Iterable

from ..types import Action, EpisodeResult, StepResult, Usage

# 默认位置。它是**可覆盖的路径参数**，不是硬编码依赖：
#   - 环境变量 EVALHARNESS_HISTORY_ROOT 优先（便于在没有项目一的机器上强制走跳过分支）
#   - 传不存在的路径 -> available() 为 False -> 上层跳过，不算失败
DEFAULT_RESULTS_ROOT = os.environ.get(
    "EVALHARNESS_HISTORY_ROOT", r"D:\TJU\4个agent_0708\demo\results"
)

# 项目一罚分规则的三层 -> 本仓库 PENALTY_CAUSES 的 canonical 键。
# 这层映射存在的意义：规则文案是中文的、会变，而指标键必须稳定，
# 否则两条跑批的罚分构成没法逐项相减。
LAYER_TO_CAUSE: dict[str, str] = {
    "A 天数累计类": "impossible",
    "B 跨日窗口承诺类": "structural",
    "C 单日可决策类": "decision_quality",
}

# 项目一动作日志里的参数名 -> 本仓库 Action 的参数名。
# 只有一处差异（duration_minutes / duration），但它会让同一个动作在两个来源下
# 得到两个不同的签名 —— 震荡率就失去了跨来源可比性，所以必须归一。
_PARAM_ALIASES: dict[str, str] = {"duration_minutes": "duration"}


# ---------------------------------------------------------------------------
def available(results_root: str = DEFAULT_RESULTS_ROOT) -> bool:
    """历史跑批在当前机器上是否可读。不可读 -> 上层跳过，不算失败。"""
    return os.path.isdir(results_root) and bool(list_runs(results_root))


def list_runs(results_root: str = DEFAULT_RESULTS_ROOT) -> list[str]:
    """列出所有含官方评分产物的跑批目录名。"""
    if not os.path.isdir(results_root):
        return []
    return sorted(
        name
        for name in os.listdir(results_root)
        if os.path.isfile(os.path.join(results_root, name, "monthly_income_202603.json"))
    )


# ---------------------------------------------------------------------------
def classify_penalty_rule(rule: str) -> str:
    """按「决策可控性」给一条罚分规则分层（重构自 repro_metrics.classify_penalty_rule）。

    A 天数累计类     —— 规则要求"整月至少 N 天"，在不足 N 天的基准里数学上不可能达成。
    B 跨日窗口承诺类 —— 带明确时间窗的长期承诺，需要提前数日规划，短基准内结构性冲突。
    C 单日可决策类   —— 当日动作即可满足/违反，是真正的决策质量信号。
    """
    if "临时约定" in rule:
        return "B 跨日窗口承诺类"
    if (
        ("自然月" in rule and ("整天" in rule or "天完全" in rule))
        or rule.startswith("每月至少")
        or rule.startswith("月度至少")
    ):
        return "A 天数累计类"
    return "C 单日可决策类"


def penalty_by_cause(loaded: dict[str, Any]) -> dict[str, float]:
    """把一个已加载跑批的罚分按成因汇总成 canonical 键。

    只统计 penalty > 0 的规则：没触发的规则不进分母，
    否则"没违反的约束"会被算成"可优化空间"，结论是反的。
    """
    out: dict[str, float] = {}
    for layer, amount in loaded["penalty_layers"].items():
        if amount > 0:
            cause = LAYER_TO_CAUSE.get(layer, "decision_quality")
            out[cause] = out.get(cause, 0.0) + amount
    return out


# ---------------------------------------------------------------------------
def load_run(name: str, results_root: str = DEFAULT_RESULTS_ROOT) -> dict[str, Any] | None:
    """读一个跑批目录，返回汇总 dict。缺产物 -> None（不抛，方便批量遍历）。"""
    run_dir = os.path.join(results_root, name)
    mi_path = os.path.join(run_dir, "monthly_income_202603.json")
    if not os.path.isfile(mi_path):
        return None

    with open(mi_path, encoding="utf-8") as f:
        mi = json.load(f)
    rs_path = os.path.join(run_dir, "run_summary_202603.json")
    rs: dict[str, Any] = {}
    if os.path.isfile(rs_path):
        with open(rs_path, encoding="utf-8") as f:
            rs = json.load(f)

    summary = mi["summary"]
    steps_by_driver: dict[str, list[StepResult]] = {}
    action_counter: dict[str, int] = {}
    accepted_orders = 0
    attempted_orders = 0
    llm_calls = 0

    for path in sorted(glob.glob(os.path.join(run_dir, "actions_202603_*.jsonl"))):
        driver_id = os.path.basename(path).split("_")[2]
        steps = _load_action_log(path)
        steps_by_driver[driver_id] = steps
        for s in steps:
            action_counter[s.action.name] = action_counter.get(s.action.name, 0) + 1
            if (s.info.get("token_usage_total") or 0) > 0:
                llm_calls += 1
            if s.action.name == "take_order":
                attempted_orders += 1
                if s.info.get("accepted"):
                    accepted_orders += 1

    step_files_steps = sum(len(v) for v in steps_by_driver.values())
    if step_files_steps and rs.get("completed_steps"):
        # 双源对账：动作日志行数 vs 官方 completed_steps。
        # 两者不一致说明这一组的产物不完整，它的数字不该被引用。
        consistent = step_files_steps == rs["completed_steps"]
    else:
        consistent = None

    layers, total_penalty = _penalty_layers(mi.get("drivers", []))
    net = summary.get("total_net_income_all_drivers", 0.0)
    penalty = summary.get("total_preference_penalty", 0.0)

    return {
        "name": name,
        "days": rs.get("simulation_duration_days"),
        "completed_steps": rs.get("completed_steps"),
        "wall_seconds": rs.get("simulate_time_seconds"),
        "net": net,
        "penalty": penalty,
        "op_profit": round(net + penalty, 1),   # 自定义口径，官方没有这个字段
        "tokens": (summary.get("total_token_usage") or {}).get("total_tokens", 0),
        "steps": step_files_steps,
        "steps_by_driver": {k: len(v) for k, v in steps_by_driver.items()},
        "_steps": steps_by_driver,
        "action_counter": action_counter,
        "accepted_orders": accepted_orders,
        "attempted_orders": attempted_orders,
        "llm_calls": llm_calls,
        "penalty_layers": layers,
        "penalty_total_from_rules": total_penalty,
        "steps_vs_summary_consistent": consistent,
    }


def load_episodes(
    name: str, results_root: str = DEFAULT_RESULTS_ROOT
) -> list[EpisodeResult]:
    """把一个跑批还原成 EpisodeResult 列表 —— 这样**指标层代码原封不动**就能吃历史数据。

    这是"插件"二字的落点：平台不认识项目一的日志格式，
    格式转换发生在这里，metrics/process.py 一行都不用改。
    """
    loaded = load_run(name, results_root)
    if loaded is None:
        return []
    out: list[EpisodeResult] = []
    for driver_id, steps in sorted(loaded["_steps"].items()):
        usage_total = Usage()
        for s in steps:
            usage_total = usage_total + s.usage
        out.append(
            EpisodeResult(
                driver_id=driver_id,
                run_id=f"history:{name}",
                variant=name,
                steps=steps,
                usage_total=usage_total,
                # 历史日志没有逐步 reward，这里填 0 而不是"看起来像的假值"。
                # Outcome 层的历史数字请直接读 loaded["net"] / ["op_profit"]。
                reward_total=0.0,
                terminated=True,
                truncated=False,
                stop_reason="history_replay",
            )
        )
    return out


# ---------------------------------------------------------------------------
def _load_action_log(path: str) -> list[StepResult]:
    """读一个司机的动作日志 -> StepResult 列表。"""
    steps: list[StepResult] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            act = rec.get("action") or {}
            name = act.get("action") or "unknown"
            params = _canon_params(act.get("params") or {})
            usage_total = (rec.get("token_usage") or {}).get("total_tokens", 0) or 0
            steps.append(
                StepResult(
                    step=i,
                    action=Action(name=name, params=params),
                    reward=0.0,
                    # 历史日志只有 token 合计，拆分不可得 —— 全部记在 input 侧并注明。
                    # 因此历史回放只喂 Process 层，不喂 Cost 层（拆分失真会误导单价归因）。
                    usage=Usage(input_tokens=int(usage_total)),
                    terminated=False,
                    truncated=False,
                    info={
                        "signature": Action(name=name, params=params).signature(),
                        "accepted": bool((rec.get("result") or {}).get("accepted")),
                        "token_usage_total": int(usage_total),
                    },
                )
            )
    return steps


def _canon_params(params: dict[str, Any]) -> dict[str, Any]:
    return {_PARAM_ALIASES.get(k, k): v for k, v in params.items()}


def _penalty_layers(drivers: Iterable[dict[str, Any]]) -> tuple[dict[str, float], float]:
    """逐司机汇总罚分分层。返回 ({层名: 金额}, 合计)。"""
    layers: dict[str, float] = {}
    total = 0.0
    for dr in drivers:
        for r in (dr.get("preference_check") or {}).get("rules", []):
            p = r.get("penalty", 0) or 0
            if p > 0:
                layer = classify_penalty_rule(r.get("rule", ""))
                layers[layer] = layers.get(layer, 0.0) + p
                total += p
    return layers, total
