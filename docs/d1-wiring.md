# D1 · 把项目一接进来（Wiring）

> 结论先行：**没有复制项目一的任何一个文件，也没有改动项目一的任何一行代码。**
> 「接入」这件事的全部工作量，是写一个双向 Adapter + 一个模型调用网关。
> 之所以能这么轻，是因为项目一自己就把边界画好了 —— 见第 1 节。

配套可运行物：

| 文件 | 作用 | 怎么跑 |
|---|---|---|
| `adapters/truck.py` | 双向 Adapter（AgentAdapter 5 方法 + SimulationApiPort 4 方法） | 由 Runner 注入 |
| `evalharness/llm_gate.py` | 模型调用网关（限流 / 缓存 / 计账 / span） | 由 Runner 注入 |
| `tools/check_truck.py` | 合成环境下的接线自检（不需要项目一、不需要 API key） | `python tools/check_truck.py` |
| `tools/run_d1.py` | 真项目一 × 1 司机 × 1 天 | `python tools/run_d1.py`（默认 dry-run） |

---

## 1. 为什么不用复制代码：`SimulationApiPort` 就是现成的缝

项目一的 `demo/simkit/ports.py` 里定义了两个 Protocol：

```python
class AgentDecisionPort(Protocol):
    def decide(self, driver_id: str) -> dict[str, Any]: ...

class SimulationApiPort(Protocol):
    def get_driver_status(self, driver_id: str) -> dict[str, Any]: ...
    def query_cargo(self, driver_id, latitude, longitude) -> dict[str, Any]: ...
    def query_decision_history(self, driver_id: str, step: int) -> dict[str, Any]: ...
    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]: ...
```

而 `ModelDecisionService.__init__` 的第一句是：

```python
def __init__(self, api: SimulationApiPort, api_key: str = "", ...)
```

**环境对项目一来说是注入进来的。** 所以「把项目一接进评测平台」的准确说法是：

> **扮演那个环境。**

这一句同时解决了合规问题：`SimulationApiPort` 的 4 个方法里**没有「读数据集」这条路**
—— 因为赛题《数据说明》§1.1 明令禁止决策代码直接读 `cargo_dataset.jsonl` / `drivers.json`。
所以这个 Adapter 在**物理上**就拿不到数据文件路径。本仓库因此不含一个字节的赛题数据，
也不含一个字节的项目一代码。

---

## 2. 双向 Adapter：中间那一层才是「适配器」三个字的意思

```
我的 Runner ──AgentAdapter(5 方法)──▶ TruckAdapter ──SimulationApiPort(4 方法)──▶ 项目一四节点图
            ▲ 向上：跟 dummy 没有任何区别        ▲ 向下：跟赛方环境一模一样
```

向上 5 个方法（`evalharness/adapter.py` 的 Protocol，runtime_checkable）：

| 方法 | 项目一侧的实现 |
|---|---|
| `metadata()` | 返回 `AgentMeta`，把「四节点 / 串行 / agent2 不调模型」写进 extra |
| `reset(driver_id, seed)` | 建一次 `ModelDecisionService`（图在这里 compile 一次，整月复用）；`_rewind_repo` 把时间拨回 0 |
| `act(obs)` | `service.decide(driver_id)` —— 跑一次四节点图 |
| `env_step(action)` | 用赛方的 `simkit.simulation_actions` 推进环境 |
| `close()` | 只解除对 service 的引用，**不释放共享的 repo/manager** |

向下 4 个方法（`_TruckPort`）：`get_driver_status` / `query_cargo` / `query_decision_history`
直接转调；第 4 个 `model_chat_completion` 是平台价值最集中的一处，见第 3 节。

两个刻意的设计：

- **`act()` 与 `env_step()` 必须分开。** 动作质量与环境反馈要能分开归因。
  `env_step` 不属于被测 Agent，所以它放在 Adapter 而不是 service 里。
- **`query_cargo` 之后必须照抄扫描成本。** `apply_cargo_query_scan_cost(...)` 会推进仿真时间；
  漏掉它，你的评测会比真实环境「更快」，步数不可比。

---

## 3. LLMGate：为什么计账粒度必须是「一次调用」而不是「一步」

Runner 外层是按 step 限流的：一步 `acquire` 一次。
但项目一的一步里有 **3 个会调模型的节点**（agent1 偏好分类 / agent3 时间分析 / agent4 决策），
一次 step 实际打 3 次模型。如果限流只包在最外层：

- RPM 桶按 step 计数 → 实际请求率被低估 3 倍 → 上游 429，且**报错位置随机、事后不可复现**
- `llm_inference` span 只能记到「整步耗时」，拆不出是哪个节点慢、哪个节点贵
- 缓存没有落点 —— prompt 只在节点内部才存在，外层看不到
- token 账只能事后拿一个总数，无法按节点归因

所以把「一次 LLM 调用」做成平台的原子操作：

> **被测 Agent 不自己发 HTTP，它交出 payload，平台来发。**

配套的两个约定：

- `TruckAdapter.owns_llm_limits = True`：告诉 Runner「我的每一步调用已经在网关里计过账了」，
  于是 Runner **不再在外层按 step 预扣** —— 否则同一笔 token 记两次，配额被无声砍到三分之一。
  它是可选类属性，不是第 6 个协议方法：**协议方法一旦加，所有 adapter 都要实现；能力声明不用。**
- 缓存命中时返回体的 `usage` 一律清零。下游读 `resp["usage"]` 得到的是「这次调用花掉的 token」，
  命中就是 0。省下的那部分记在 `saved_tokens` —— **省下的和花掉的必须分别可见**，
  否则「开了缓存」和「模型变便宜了」在成本表上长得一模一样。

---

## 4. `custom:` 前缀：被测对象可以自报阶段，但污染不到公共口径

平台保留 9 段公共 Span（`evalharness/observability.py` 的 `SPAN_NAMES`）。
项目一的四个节点是它自己的内部结构，平台不该知道，但又必须能看见 ——
于是 `TruckAdapter._install_node_probes()` 在**运行期**替换四个节点方法的绑定，
各开一段 `custom:node_{preference|geo|time|decision}`。

- 前缀由 `observability.CUSTOM_PREFIX` 锁死；不带前缀的非法名字会被 `assert` 拦下；
- `tracer.coverage()` 只算公共 9 段，`custom_coverage()` 单独算 —— **公共聚合口径不会被污染**；
- 探针用 `threading.local()` 而不是实例属性记「当前节点名」：Runner 用 `to_thread` 每司机一个线程，
  实例属性会串。

---

## 5. 真跑一次（live）

dry-run 会把整条链路（Adapter 5 方法 + port 4 方法 + 节点插桩 + 限流 + 缓存）全部走一遍，
只有「模型答得对不对」没验。**先花 0 元排掉所有接线错误，再花一次钱验模型。**

```bash
# 1) 先零成本自检（不需要项目一）
python tools/check_truck.py          # 期望 17 通过 / 0 失败

# 2) dry-run 打通真项目一（仍然是 0 元）
python tools/run_d1.py               # 1 司机 × 1 天

# 3) 真打模型（会花钱）
$env:DASHSCOPE_API_KEY = "sk-..."
python tools/run_d1.py --live

# 4) 复现赛方口径（不接管 agent3，time 节点 token 不可见）
python tools/run_d1.py --live --no-intercept
```

**前提**：项目一不在本仓库里。位置由环境变量 `TRUCK_REPO` 或 `--project-root` 给出，
默认 `D:\TJU\4个agent_0708`。找不到就抛 `FileNotFoundError` ——
**绝不静默降级成 dummy**：静默降级会让你以为在测项目一，其实在测一个假 Agent，而且它不报错。

D1 的验收标准就是最后那张「按节点」的表：

```
节点            耗时(ms)     占比   模型调用       token   token占比
decision          ...        ...       ...         ...        ...
geo               ...        ...         0           0        0.0%
preference        ...        ...       ...         ...        ...
time              ...        ...       ...         ...        ...
```

> **某司机某一步下挂的四个节点，各自的耗时与 token 都能看见。**
> 在那之前，项目一的两组跑批只有目录名能证明差异。这张表是「可归因」的第一步。

---

## 6. 实测发现：agent3 绕过了端口，赛方口径少记 token

`demo/agent/time_agent_langchain.py:82`（`TimeAgent.__init__`）：

```python
self._llm = ChatOpenAI(model=model_name,
                       api_key=api_key or os.environ.get("DASHSCOPE_API_KEY", ""),
                       base_url=api_base, callbacks=[self._callback])
```

它**自己 new 了客户端**，不经过注入进来的 `api`。加上 `agent2`（地点分析）是纯 Python 几何计算、
根本不调模型，于是：

> **「四节点」不等于「四次调用」。一步实际是 3 次调用（agent1 + agent3 + agent4）。**

后果有三条：

1. agent3 的 token 不在 `EmbeddedDecisionEnvironment.get_last_model_usage()` 里
   → **赛方口径的 token 统计天然偏低**（低多少：见 `tools/check_truck.py` 第 4 节的实测数字）；
2. 这部分请求不受限流、不进缓存、没有 span → **它既不慢也不贵，因为它压根没被测量**；
3. dry-run 下它会真的去打网络并抛异常 → 接线验证跑不完。

**平台侧的处置**：`TruckAdapter._install_foreign_client_probe()` 在运行期替换
`llm._generate`（LangChain `BaseChatModel` 的同步唯一出口；HTTP 层 patch 不到，
因为 langchain_openai 走的是 `client.with_raw_response.create(...)`），把这部分调用也接进网关，
打上 `source="agent_own_client"`。

### 实测数字（dry-run，1 司机 × 1 天 D001，23 步）

```
【按节点归因】
节点            耗时(ms)     占比   模型调用     token   token占比
decision          24.0     8.2%        23     235,926      47.0%
geo               14.1     4.8%         0           0       0.0%
preference         7.8     2.7%        23       3,517       0.7%
time             247.0    84.3%        23     262,165      52.3%

llm.calls = 69  =  calls_port 46  +  calls_foreign 23
```

两点：

1. **每一步 = 3 次模型调用**（preference + time + decision），不是 1 次。
   按 step 限流会把请求率低估 3 倍。
2. **`time` 一个节点占了 84.3% 的墙钟耗时和 52.3% 的 token，而它在赛方口径里完全不可见。**
   也就是说：项目一最贵、最慢的那个节点，恰恰是它自己统计不到的那个。

（这些 token 是 dry-run 按 payload 字符数估的，绝对值不代表真实成本；
但节点间的相对占比有结构意义 —— 它反映各节点 prompt 的大小分布。）

- **不改项目一的一行源码**；
- 计数看得见：`llm.calls_foreign`。它应该是 0 —— 不是「忽略不计」，是 0。
  非 0 就是一个待修的结构缺陷，而且现在有数字可以证明；
- `ChatResult.llm_output["token_usage"]` 必须填：项目一的 `TokenStatsCallback` 从这里读 token，
  留空等于接管了一个客户端却把它的账记没了。

---

## 7. `concurrency` 必须是 1（硬约束，不是配置偏好）

两个原因：

1. `DriverStateManager` 只有一个全局 `_simulation_progress_minutes` 和一个 `_current_driver_id`，
   `ensure_active_driver()` 会在 driver_id ≠ 当前司机时直接抛
   `当前正在模拟 D002，不允许操作 D001`（`tools/check_truck.py` 附录里 deterministic 复现）；
2. `cargo_dataset.jsonl` 是 210 MB，`CargoRepository.load()` 全量解析成 Python 对象，
   一份常驻内存就以 GB 计。多份 = 直接把机器打爆。

所以 `adapters/truck.py` 里写死了 `MAX_SUPPORTED_CONCURRENCY = 1`，`Scenario` 里配 `concurrency: 1`。

> **平台的并发能力 ≠ 环境的并发可行性** —— 这正是 `concurrency` 写在 Scenario 里
> 而不是写死在 Runner 里的理由。

顺带一个适配层的活：`sync_time_minutes()` 是单调向前的
（`if current_time_minutes < self._current_time_minutes: return`），
而 Runner 每个司机都从 sim_minute=0 开始，第二个司机的时间就回不去了。
赛方自己的做法是每换一个司机重新 `repo.load()`，也就是**每换一个司机重读 210 MB**。
`_rewind_repo()` 用 5 行私有字段重置替代它 —— 重复解析 210 MB × N 司机换不来任何正确性收益。

---

## 8. 已知简化（面试主动讲，比写「完美」代码更能证明你懂）

1. token 预估是按 payload 字符数的粗估（中英混排 ≈ 2 char/token）。
   `budget/estimator.py` 里的 S/K/C 系数是按 **step 索引**拟合的，两者在 D2 合并 ——
   合并后预估器才同时知道「第几步」和「这次 payload 多大」。
2. 令牌桶的 `try_acquire` 是无锁的。D1 用 `concurrency: 1`，所以没有真实竞争。
   多线程要加锁 —— 这一条写在这里，不假装它不存在。
3. `env_step` 的步级 reward = 运价 − 行驶成本，**不含偏好罚分**。
   罚分属于「评测口径」不属于「环境回报」；混进来会导致「环境替你被判了分」，
   A/B 时说不清差异来自决策还是来自罚则。罚分单独在 D4 的 `penalty_signal_ratio` 里归因。
4. `_rewind_repo` 碰的是项目一的私有字段。项目一升级后有失效风险 ——
   `tools/check_truck.py` 的覆盖面就是用来在它失效时立刻发现的。

---

## 9. 只有真跑才会暴露的两个 bug（都是「不报错」的那种）

合成替身全绿（17/17）之后，接真项目一跑第一遍就连撞两个。
**这两个 bug 的共同点是：它们不会抛异常，只会让你拿到一张错的或空的表。**

### bug 1 · 历史记录的字段形状和赛方不一致

赛方 `simulation_orchestrator.py:119-141` 存进历史的是
`"action": action` —— **整个动作 dict** `{"action": name, "params": {...}}`。
我第一版存的是动作名字符串，于是项目一在 `model_decision_service.py:670` 的
`last_action.get("action")` 上 `AttributeError: 'str' object has no attribute 'get'`。

> 适配层不是「给个差不多的 dict」就行 —— **被适配方会按字段名去 get**。
> 少一个字段是 KeyError，字段类型错了是 AttributeError，
> 而最糟的是第三种：字段名碰巧对上了，值语义不同，跑完也不报错。

### bug 2 · 实例级插桩对已经 compile 的图无效

`ModelDecisionService.__init__` 里 `workflow.add_node("agent1", self._agent1_node)`
—— 这一行求值时 **绑定方法**就已经被 LangGraph 存进编译结果了。
之后 `service._agent1_node = wrapped` 对图毫无作用。
后果：69 次模型调用全部 `node=unknown`，节点归因表只剩一行，**而且不报错，看起来跑通了**。

> 修法是改成**类级替换且必须在实例化之前**。类级替换的代价是包装函数拿不到 adapter 实例，
> 只能靠模块级单槽 `_ACTIVE_ADAPTER` 传递 —— 而这又撞上第三个坑：`reset()` 在主线程、
> `act()` 被 `to_thread` 丢进工作线程，用 thread-local 会静默退化成「不开 span」。
> 三层坑叠在一起，任何一层单独看都「没问题」。

这两个 bug 是 `check_truck.py` **抓不到**的（合成替身是我按自己的理解写的，
理解错了它就跟着错）。所以 D1 必须有两道关：**合成自检（快）+ 真跑一次（准）**。
