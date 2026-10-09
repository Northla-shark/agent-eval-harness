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

**D2 已修掉的一条**（留着说明曾经的问题形态）：
`limiter.settle()` 原先在「实际远超预估且桶内不够补扣」时用 `try_acquire` 补扣，
拿不到就返回 False —— **欠账静默丢失**。后果不是报错，而是更坏的：
限流器以为自己还有额度，继续放行，TPM 上限被无声突破，`limiter.rejected` 仍然显示 0。
现在改成负余额（`TokenBucket.consume` 允许转负），`acquire` 的额度 =
`本次预估 + 当前欠账`，必须先还清才能发下一次。
配套可观测项：`limiter.tpm_debt` / `limiter.debt_events` / `limiter.debt_repaid`。
> 一句话记法：**最危险的不是被限流，是限流器以为自己没被击穿。**

仍然存在的简化：

1. **token 仍是按字符粗估，没有真实 tokenizer。**
   `CHARS_PER_TOKEN = 2`（中英混排的经验值），真实 tokenizer 对纯中文约 1.5 char/token、
   对 JSON/代码可以低到 3。所以预估值在 ±30% 量级是**结构性**的，不是 bug。
   实测：合并口径后，同一份新跑批上平均绝对偏差从 74.47% 降到 27.17%。
   升级路径：`tiktoken` / 模型厂商的 count_tokens 端点 —— 但那要多一次网络往返，
   所以生产上更常见的做法是**保留粗估 + 用 settle 的差额做滚动校准**
   （把 `limiter.debt_ever` 与单次最大的几次偏差喂回去调 S/K/C）。

2. **预估系数 S/K/C 与被测场景是手工绑定的。**
   `scenarios/truck-driver-d1-smoke.yaml` 配的是 `no_history`（K=0），
   但项目一每步会喂 5 步历史 —— 也就是说「第几步」这个信号本来更值钱，
   是**配置把它关掉了**。这是真实存在的一处配置错误，暂未改（改它会让所有 run_id 变化）。
   升级路径：让 Scenario 校验 `token_profile` 与 adapter 声明的上下文策略是否匹配，不匹配就报错。

3. **纯步索引公式在真实数据上的方差极大。**
   10 组真实跑批（token 是真花出来的钱）上，汇总偏差看着只有 -2.10%，
   但**单组区间跨 [-60.26%, +301.19%]** —— 单看均值是最容易自欺的一句话。
   所以 `tools/compare_estimate.py` 同时打汇总值和分布的两个尾巴。
   升级路径：按「上下文策略 × 模型」分桶拟合 profile，而不是全场景共用一套系数。

4. **`ConsoleTracer` 只做内存记录，没有跨进程持久化**（Langfuse 接入属 D3，未做）。

5. **震荡检测只覆盖「动作 + 关键参数」重复**，没有覆盖「状态回退后重复」。

6. **令牌桶是无锁的。** 当前 truck 场景 `concurrency: 1`（受项目一环境限制），
   所以没有真实竞争；多线程/多进程要按 shared counter 重做，不能假装它存在。

## 6. 待办

- [x] D1：把 `adapters/truck.py` 接上项目一（LangGraph 四节点）—— **已完成**
- [x] D2：限流器欠账偿还 + 模型响应缓存 + 预估器口径合并 —— **已完成**（2026-10-09）
- [x] D4：补齐 `metrics/process.py` 的 3 个 TODO 指标 —— **已完成**（2026-10-09）
- [ ] D3：接 Langfuse（本机 Docker 未安装，走云端免费额度）
- [ ] D5：三组消融，把真数字填进 README 的 Results
- [ ] D6：文档收口与发布

D2 的验收数字与命令：

```
.venv\Scripts\python.exe tools\smoke.py            # 55 通过 / 0 失败
.venv\Scripts\python.exe tools\compare_estimate.py # 预估器口径合并前后对比
```
