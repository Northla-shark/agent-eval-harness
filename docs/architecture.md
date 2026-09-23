# 架构与核心设计决策

> 这个文件比代码更常被面试官读。写的时候记住：每个决策后面都要跟一句「不这么做会怎样」。

## 1. 为什么要有这一层（五层 + 一层）

| 层 | 职责 | 删掉它会怎样 |
|---|---|---|
| L0 Scenario | 场景 / 种子 / driver 列表 / variant | 阈值散落在代码里，无法消融 |
| L0.5 ConfigSnapshot | run_id = 配置哈希 + 代码哈希 | 两组跑批只有目录名能证明差异，不构成受控实验 |
| L1 Runner | asyncio + Semaphore + 三维熔断 + 限流 | 无法重复实验，也压不住失控循环 |
| L2 Observability | 9 段 Span schema + 后端上报 | 只有日志没有 schema，答不上"钱花在哪一步" |
| L3 Metrics | Outcome / Process / Cost | 只有一个终局标量，涨跌无法归因 |
| L4 Experiment | A/B + 消融 + 落盘 | 结论无法复述，也无法被别人检查 |

## 2. 已知的原始缺陷（这条是面试里最好的答案）

项目一 `run_summary_*.json` 只有 7 个字段：month、simulate_time_seconds、
simulation_duration_days、completed_steps、remaining_cargo_count、
driver_completed_steps、driver_result_files。

**没有 model_name，没有配置指纹，没有代码版本号。**
所以「428 步 vs 193 步」这两组跑批只有目录名能证明差异 —— 这不是受控实验。

修法：`run_id = hash(配置字段) + hash(决策逻辑文件内容)`。

## 3. 一处对早期设计文档的修正（已实测验证）

早期版本的 `SNAPSHOT_FIELDS` 漏了 `variant`。后果是真实的：
消融实验里「基线」与「消融变体」如果只有 variant 不同、其他字段全一样，
两者会算出**相同的 run_id**，于是两次实验被错误地归为同一次。
已在 `evalharness/config.py` 补上，并在 `tools/smoke.py` 第 4 节加了断言。

## 4. 三个模块的耦合点

```
estimator.estimate_step_tokens(step, profile)
        │  estimated_tokens
        ▼
limiter.acquire(estimated_tokens)  ──拿不到就等（自适应退避），不固定 sleep
        │  拿到
        ▼
adapter.act(obs)  ──►  (Action, Usage)
        │  actual_tokens
        ▼
limiter.settle(estimated, actual)  ── 多退少补
```

说不清这几行，就等于限流器、预估器、Runner 是三个分开的 demo。

## 5. 已知的简化（主动讲，不要说"完美"）

1. `limiter.settle()` 在「实际远超预估且桶内不够」时，欠账会静默丢失。
   严格做法：记负余额，下次 acquire 时优先偿还。
2. `ConsoleTracer` 只做内存记录，没有跨进程持久化。
3. 震荡检测只覆盖「动作 + 关键参数」重复，没有覆盖「状态回退后重复」。

## 6. 待办

- [ ] D2：把 `adapters/truck.py` 接上项目一（LangGraph 四节点）
- [ ] D3：接 Langfuse（本机 Docker 未安装，走云端免费额度）
- [ ] D4：补齐 `metrics/process.py` 的 3 个 TODO 指标
