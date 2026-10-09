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

> 待填 —— D4 跑完 3 组消融后用真实数字替换。
> 仓库里不允许出现 `XX%` 形式的占位数字。

| 变体 | 净收益 | 步数 | Token | wait 率 | 每单调用次数 |
|---|---|---|---|---|---|
| baseline(4-agent) | 待填 | 待填 | 待填 | 待填 | 待填 |
| -geo-agent | 待填 | 待填 | 待填 | 待填 | 待填 |
| +prefilter | 待填 | 待填 | 待填 | 待填 | 待填 |

**结论（待填）**：

## Quickstart

```bash
uv venv
uv pip install -r requirements.txt
python tools/smoke.py
```

三行命令跑完，你会看到一个 A/B 对照表 + 每个 span 的出现次数。
不需要任何 API key —— `adapters/dummy.py` 不调用模型。

```bash
python tools/check_metrics_history.py   # 可选：用历史跑批验证指标层口径（本机无历史数据则跳过）
```

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
| P0 | ConfigSnapshot（run_id = 配置哈希 + 代码哈希） | 骨架已就位 |
| P0 | Adapter 协议 + asyncio 并发 + 三维 BudgetGuard | 骨架已就位 |
| P0 | 9 段 Span schema + ConsoleTracer | 骨架已就位，Langfuse 待接 |
| P0 | 指标四件套 | 已实现（动作分布 / 每单调用 / 震荡率 / 罚分信噪比）；约束精度**主动放弃**，理由见 `docs/metrics.md` 第 5 节 |
| P1 | 双令牌桶限流器 + 有界队列 | 已实现，待接进 Runner |
| P1 | Token 预估器 | 已实现（系数来自历史 14 组跑批校准） |
| P1 | LRU + TTL 缓存层 | 已实现，18/18 自测通过 |
| P2 | replay（历史跑批回放） | 已实现（`metrics/history.py`）；本机无被测对象时自动跳过 |
| P2 | 消融 | 待你写（D5） |

## 设计决策

见 `docs/architecture.md`。**这是面试官唯一会认真读的部分。**

## Licence / Acknowledgements

MIT。观测后端使用 [Langfuse](https://github.com/langfuse/langfuse)（MIT），
仅作为 trace 存储与可视化，评测逻辑与 Span schema 均为自研。
