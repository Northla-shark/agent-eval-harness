# AgentEval Harness

面向长时程决策型 Agent 的 **评测、可观测与 A/B 实验平台**。
不做新的 Agent，做「让 Agent 可被衡量、可被迭代」的那套基础设施。

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

## Why

我把一个决策 Agent 跑起来了，但我不知道它**到底是变好了还是变坏了**：
只有一个终局标量、无法归因到模块、无法区分「改动有效」和「随机波动」。
这个项目补的就是这块地基。

## Results

下面两组数字都是**实测**，来源与复现命令一并标注。仓库里不允许出现占位数字。

### 对照一 · dummy 变体（`tools/smoke.py` 第 5 节，同环境、唯一差异字段 = variant）

| 变体 | 步数(中位) | 总 token | 净收益(中位) | wait 率 | 每单调用次数 |
|---|---|---|---|---|---|
| baseline | 59 | 622,643 | 30,485 | 36.5% | 1.679 |
| degenerate | 91 | 964,692 | 13,328 | 84.4% | 6.395 |

复现：`.venv\Scripts\python.exe tools\smoke.py`（不需要任何 API key）。

### 对照二 · 历史跑批 A/B（`tools/check_metrics_history.py --all`）

被测对象：项目一四节点货车司机决策 Agent，7 天基准，两组**同 prompt、仅模型不同**
（B 组目录名后缀即 `-qwen3.7-max`）。

| 组 | 步数 | token | 官方净收益 | 罚分 | wait 率 | 每单调用 | 震荡率 | 罚分信噪比 |
|---|---|---|---|---|---|---|---|---|
| A 组 | 428 | 2,870,662 | −22,400.5 | 29,490.0 | 91.4% | 13.38 | 0.7173 | 25.4% |
| B 组 | 193 | 1,526,899 | 12,321.18 | 33,894.7 | 36.8% | 1.86 | 0.1192 | 33.6% |

「官方净收益」= `monthly_income_202603.json` 的 `summary.total_net_income_all_drivers`（已扣罚分）。
10 组跑批的「步数对账」全部一致（动作日志行数 == 官方 `completed_steps`）——
这一列如果哪天不一致，说明该组产物不完整，它的数字不该被引用。

### 指标本身也有信噪比

| 基准 | impossible（基准内不可能达成） | structural（跨日窗口冲突） | **decision_quality（真正由决策质量决定）** |
|---|---|---|---|
| 1 天 | 41.8% | 55.6% | **2.6%** |
| 30 天 | 8.2% | 24.6% | **67.2%** |

相差 **25.5 倍**：同一个失败实验，换一个基准长度，可优化信号占比从 2.6% 变成 67.2%。

### 结论

**先说误判。** A/B 两组同 prompt、只差模型，终局收益相差很大（官方净收益
−22,400.5 vs 12,321.18）。最直觉的解释是**模型选型问题**——B 组换了个模型，所以 B 组更好。

**再说工具。** 过程指标给出的答案不是这样。把动作按「(动作, 关键参数)」做签名计数后，
A 组的 `wait(duration=60)` 出现了 **146 次**（占其 428 步的 34.1%），B 组只有 **9 次**。
也就是说两组真正的差别不是"模型聪不聪明"，而是 **A 组退化成了每小时一次的等待轮询**：
它每步都在等 60 分钟，没有从上一轮获得任何新信息，于是把 91.4% 的步数和 287 万 token
全花在了等待上。对应的震荡率 0.7173 vs 0.1192、每单调用 13.38 vs 1.86，指的都是同一件事。

**而这一点终局指标完全看不出来。** 净收益只告诉你"少了多少钱"，不告诉你钱花在哪一行。
这也是本仓库把 Process 层放在 Outcome 层之上的原因。

## Quickstart

```bash
uv venv
uv pip install -r requirements.txt
python tools/smoke.py
```

三行命令跑完，你会看到一个 A/B 对照表 + 每个 span 的出现次数。
**不需要任何 API key** —— 仓库自带 `adapters/dummy.py`，它是一个不调模型的完整 adapter，
clone 下来就能跑通「Scenario → Runner → Observability → Metrics → Experiment」全链路。

```bash
python tools/check_metrics_history.py   # 可选：用历史跑批验证指标层口径（本机无历史数据则跳过）
```

> **`truck` adapter 不一样。** 它依赖**仓库之外**的私有 Agent 实现（项目一，不随本仓库分发）。
> 项目一不存在时它会**抛出可读的错误，而不是静默降级成 dummy** ——
> 静默降级会让你以为在测真实 Agent，实际在测一个假 Agent，而且不报错。
> 要跑 dummy 请显式把 `adapter_name` 改成 `dummy`。

## 架构

```
Layer 0  Scenario      ── 场景 / 固定种子 / 样本集 / variant 列表
Layer 0.5 ConfigSnapshot── 配置快照 / run_id / 差异比对     ★ 没有它就不是受控实验
Layer 1  Runner        ── asyncio 并发 + Semaphore + 三维 BudgetGuard
Layer 2  Observability ── 9 段 Span 的 schema + 后端上报（Langfuse 只做存储）
Layer 3  Metrics       ── Outcome / Process(★) / Cost
Layer 4  Experiment    ── A/B + 消融 + 报告
```

**为什么 Process 层是核心**：终局收益任何比赛脚本都能给。
`wait 率` / `每单调用次数` 才能告诉你「Agent 是不是把算力花在了等待上」——
这直接定位到要改哪一行。

## 当前进度

| 级别 | 内容 | 状态 |
|---|---|---|
| P0 | ConfigSnapshot（run_id = 配置哈希 + 代码哈希） | 已实现（smoke 第 4 节断言不同配置不撞 run_id） |
| P0 | Adapter 协议 + asyncio 并发 + 三维 BudgetGuard | 已实现并接进 Runner |
| P0 | 9 段 Span schema + ConsoleTracer | ConsoleTracer 已实现；**LangfuseTracer 未实现**（D3，`span()` 仍抛 `NotImplementedError`） |
| P0 | 指标四件套 | 已实现（动作分布 / 每单调用 / 震荡率 / 罚分信噪比）；约束精度**主动放弃**，理由见 `docs/metrics.md` 第 5 节 |
| P1 | 双令牌桶限流器 | 已实现并接进 Runner；**`settle()` 的欠账偿还未实现**（D2） |
| P1 | 有界队列 | 模块已实现（`queue.py`），**未接进 Runner** |
| P1 | Token 预估器 | 已实现（系数来自历史 10 组跑批校准） |
| P1 | LRU + TTL 缓存层 | 已实现，18/18 自测通过 |
| P1 | 项目一接入（TruckAdapter + LLMGate + 节点级归因） | 已实现（D1）；`tools/check_truck.py` 17 项通过 |
| P1 | 模型替身（`stub.py`，零 API 成本跑全链路） | 已实现 |
| P2 | 历史跑批回放 | 已实现（`metrics/history.py`）；本机无被测对象时自动跳过 |
| P2 | 三组消融 | **未做（D5）** —— 现有 Results 是 dummy 变体对照 + 历史 A/B 对照，**不是消融** |
| P2 | 文档收口与发布 | **未做（D6）** |

自检：`.venv\Scripts\python.exe tools\smoke.py` → 42 通过 / 0 失败
（本机无项目一历史数据时第 7 节跳过，为 35 / 0）。

## 设计决策

见 `docs/architecture.md`。**这是面试官唯一会认真读的部分。**

指标定义、动机与边界见 `docs/metrics.md`。

## Licence / Acknowledgements

MIT。观测后端使用 [Langfuse](https://github.com/langfuse/langfuse)（MIT），
仅作为 trace 存储与可视化，评测逻辑与 Span schema 均为自研。
