# -*- coding: utf-8 -*-
"""项目一（卡车司机 Agent）-> Adapter。D1 主任务。

先说结论：这件事比预想的小得多
==============================
最初的设想是「把项目一的代码复制进来，外面包一层」。读完项目一的接口定义之后发现
**不需要复制任何代码，也不需要队友同意** —— 因为项目一自己就把边界画好了。
它的 `demo/simkit/ports.py` 定义了：

    class AgentDecisionPort(Protocol):     decide(driver_id) -> dict[str, Any]
    class SimulationApiPort(Protocol):     get_driver_status / query_cargo
                                           / query_decision_history
                                           / model_chat_completion

`ModelDecisionService` 的构造函数第一句就是 `def __init__(self, api: SimulationApiPort, ...)`
—— 也就是说，**赛方环境对项目一来说是「注入进来的」**。

所以「把项目一接进评测平台」的准确说法是：**扮演那个环境**。
于是这个 Adapter 是双向的，中间那一层才是「适配器」三个字的真正含义：

        我的 Runner ──AgentAdapter(5 方法)──▶ TruckAdapter ──SimulationApiPort(4 方法)──▶ 项目一四节点图
                     ▲ 向上：看起来跟 dummy 没有任何区别      ▲ 向下：看起来跟赛方环境一模一样

它顺手解决了一个合规问题
========================
`SimulationApiPort` 的 4 个方法里**没有「读数据集」这条路** —— 因为赛题
《数据说明》§1.1 明令禁止决策代码直接读 `cargo_dataset.jsonl` / `drivers.json`。
所以这个 Adapter 在物理上就拿不到数据文件路径，观测只能从它自己的方法里出去。
本仓库因此不含**一个字节**的赛题数据，也不含**一个字节**的项目一代码。

运行前提（README 里要写明的那句）
=================================
项目一的代码**不在这个仓库里**。位置由 `TRUCK_REPO` 环境变量或 `project_root=` 给出，
默认 `D:\\TJU\\4个agent_0708`。找不到就抛 `FileNotFoundError` ——
**绝不静默降级成 dummy**：静默降级会让你以为在测项目一，其实在测一个假 Agent，而且它不报错。
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from evalharness.adapter import register
from evalharness.llm_gate import extract_usage
from evalharness.types import Action, AgentMeta, EnvStep, Observation, Usage

DEFAULT_PROJECT_ROOT = r"D:\TJU\4个agent_0708"
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.7-max"
MINUTES_PER_DAY = 1440
# 仿真时钟起点，与赛方 `simulation_orchestrator._SIMULATION_EPOCH` 一致
# （2026-03-01 00:00，见 `simkit/cargo_repository.py` 的注释）。
# 必须一致：项目一拿 `simulation_end_time` 去算休息时长与月度进度。
_SIMULATION_EPOCH = datetime(2026, 3, 1, 0, 0, 0)


def _format_sim_clock(simulation_minutes: int) -> str:
    """仿真分钟 -> 墙上时间。与赛方 `_format_sim_clock` 同格式。"""
    return (_SIMULATION_EPOCH + timedelta(minutes=int(simulation_minutes))).strftime("%Y-%m-%d %H:%M")

# 项目一 LangGraph 的四个节点：节点名 -> 实例方法名。
# agent2（地点分析）**不调用模型**，是纯 Python 几何计算 —— 这一点很重要，
# 「四节点」不等于「四次 LLM 调用」。见 docs/d1-wiring.md 第 6 节的实测。
_NODE_METHODS: tuple[tuple[str, str], ...] = (
    ("preference", "_agent1_node"),
    ("geo", "_agent2_node"),
    ("time", "_agent3_node"),
    ("decision", "_agent4_node"),
)

# 当前节点名。用 thread-local 而不是实例属性：Runner 把 act() 放进 to_thread，
# 每个司机一个线程；节点内部若再有并发，实例属性会串。
_NODE = threading.local()

# 当前 adapter。类级插桩的包装函数拿不到 adapter 实例，只能从这里取。
#
# ★ 为什么是**模块级单槽**而不是 thread-local（`_NODE` 是 thread-local，这个不是）：
#   `reset()` 在主线程调用，`act()` 被 Runner 用 `asyncio.to_thread` 丢进工作线程 ——
#    thread-local 在工作线程里读不到，插桩会静默退化成"不开 span"（我踩过一次：
#    llm_inference 的 node 标签是对的，custom:node_* 却一条都没有，且不报错）。
#   能用单槽的前提是 `MAX_SUPPORTED_CONCURRENCY == 1`（同一个进程同一时刻只有一个
#   活跃 driver）。**将来放开并发时必须改回 thread-local 并在 act() 里重设。**
_ACTIVE_ADAPTER: Any = None


def _current_node() -> str:
    return getattr(_NODE, "name", "unknown")


# ---------------------------------------------------------------------------
# 项目一代码的定位与 import（不复制文件）
# ---------------------------------------------------------------------------
def _resolve_demo_root(project_root: str | Path | None) -> Path:
    """定位项目一的 `demo/` 目录。找不到必须**响**。"""
    root = Path(project_root or os.environ.get("TRUCK_REPO") or DEFAULT_PROJECT_ROOT)
    demo = root / "demo"
    marker = demo / "agent" / "model_decision_service.py"
    if not marker.is_file():
        raise FileNotFoundError(
            "找不到项目一，本 adapter 无法工作。\n"
            f"  期望文件：{marker}\n"
            f"  当前 TRUCK_REPO = {root}\n"
            "  修法二选一：\n"
            "    1) set TRUCK_REPO=D:\\TJU\\4个agent_0708\n"
            "    2) TruckAdapter(project_root=r'D:\\TJU\\4个agent_0708')\n"
            "  注意：本仓库**刻意不含**项目一代码（它是团队成果、且赛题数据有使用限制）。\n"
            "  项目一不在时请改用 dummy adapter —— 但要显式改，不要指望它静默降级。"
        )
    return demo


def _import_project_one(demo: Path) -> Any:
    """把项目一的 `demo/` 挂到 sys.path 再 import。

    挂 `demo/` 而不是挂 `demo/agent/`，因为 `model_decision_service.py` 里写的是
    `from simkit.ports import SimulationApiPort` 和 `from .time_agent_langchain import TimeAgent`
    —— 前者要 `demo/` 在路径上，后者要 `agent` 是个包。
    """
    if str(demo) not in sys.path:
        sys.path.insert(0, str(demo))
    from agent.model_decision_service import ModelDecisionService  # noqa: E402

    return ModelDecisionService


# ---------------------------------------------------------------------------
# 共享的仿真资源
# ---------------------------------------------------------------------------
_SHARED_LOCK = threading.Lock()
_SHARED: dict[str, dict[str, Any]] = {}

# 并发上限（写在这里是为了让它可被审阅，而不是散在文档里）
#   truck adapter 的并发度必须 = 1，两个原因，都是硬约束：
#     1. DriverStateManager 只有一个全局 `_simulation_progress_minutes` 和一个
#        `_current_driver_id`，`ensure_active_driver()` 会在 driver_id 不等于当前司机时直接抛错。
#     2. cargo_dataset.jsonl 是 210 MB，`CargoRepository.load()` 全量解析成 Python 对象
#        （每个货源一个 dict），一份常驻内存就以 GB 计。多份 = 直接把机器打爆。
#   结论：**平台的并发能力不等于环境的并发可行性** —— 这正是 concurrency 写在
#   Scenario 里而不是写死在 Runner 里的理由。truck 场景配 concurrency: 1。
MAX_SUPPORTED_CONCURRENCY = 1


def _shared_resources(demo: Path) -> tuple[Any, Any]:
    """进程级单例的 (CargoRepository, DriverStateManager)。**210 MB 只解析一次。**"""
    key = str(demo)
    with _SHARED_LOCK:
        if key not in _SHARED:
            from simkit.cargo_repository import CargoRepository
            from simkit.driver_state_manager import DriverStateManager

            data = demo / "server" / "data"
            repo = CargoRepository(data / "cargo_dataset.jsonl")
            manager = DriverStateManager(data / "drivers.json")
            repo.load()  # 210 MB。慢一次，但只慢这一次。
            _SHARED[key] = {"repo": repo, "manager": manager}
        return _SHARED[key]["repo"], _SHARED[key]["manager"]


def _rewind_repo(repo: Any) -> None:
    """把 CargoRepository 的仿真时间回退到 0。

    为什么必须碰它的私有字段：`sync_time_minutes()` 是**单调向前**的 ——
        `if current_time_minutes < self._current_time_minutes: return`
    而我的 Runner 每个司机都从 sim_minute=0 开始，第二个司机的时间就回不去了。
    赛方自己的做法是每换一个司机重新 `repo.load()`，也就是**每换一个司机重读 210 MB**。

    取舍：重复解析 210 MB × N 司机换不来任何正确性收益，而这段 shim 只有 5 行、
    有注释、有测试。**这就是适配层该干的活** —— 把「环境的实现细节」和「平台的调用约定」隔开。
    """
    repo._pending_cursor = 0
    repo._online = {}
    repo._online_expire_heap = []
    repo._online_dirty = True
    repo._current_time_minutes = 0


def _slice_history(records: list[dict[str, Any]], step: int) -> list[dict[str, Any]]:
    """与赛方 `embedded_agent._slice_decision_history_records` 同一语义：
    -1 全部 / 0 空 / 正整数取末尾 N 条。"""
    if step == -1:
        return list(records)
    if step <= 0:
        return []
    return list(records) if len(records) <= step else records[-step:]


# ---------------------------------------------------------------------------
# 向下：SimulationApiPort 的实现 —— 「赛方环境的替身」
# ---------------------------------------------------------------------------
class _TruckPort:
    """`SimulationApiPort` 的实现。这就是「扮演赛方环境」的那一层。

    第四项 `model_chat_completion` 是重点。赛方在那里的实现是：

        def model_chat_completion(self, payload):
            import time
            time.sleep(2)                      # ← 就是它：把等待写进了串行流程
            resp = self._model_gateway.chat_completion(payload)
            ...

    我把它换成 `LLMGate` —— **等待只发生在真正被限流时**，而且每一次调用都带
    span、token 账和缓存键。项目一里那 91.4% 的 wait 步，根因就在这里。
    """

    def __init__(self, adapter: "TruckAdapter") -> None:
        self._a = adapter

    # 1/4
    def get_driver_status(self, driver_id: str) -> dict[str, Any]:
        self._a.query_calls += 1
        return self._a.manager().get_driver_status(driver_id)

    # 2/4
    def query_cargo(self, driver_id: str, latitude: float, longitude: float) -> dict[str, Any]:
        from simkit import simulation_actions

        manager = self._a.manager()
        before = manager.get_simulation_progress_minutes()
        raw = simulation_actions.query_cargo(
            self._a.repo(), manager, driver_id, latitude, longitude,
            k=self._a.nearest_cargo_limit,
        )
        items = raw.get("items", [])
        if not isinstance(items, list):
            raise TypeError("query_cargo 返回的 items 必须是列表")
        # 扫描成本必须照抄：它推进仿真时间。漏掉它 = 你的评测比真实环境"更快"，步数不可比。
        simulation_actions.apply_cargo_query_scan_cost(
            self._a.repo(), manager, driver_id, len(items),
            cargo_view_batch_size=self._a.cargo_view_batch_size,
        )
        scan = manager.get_simulation_progress_minutes() - before
        self._a.scan_minutes += scan
        self._a.query_calls += 1
        return raw

    # 3/4
    def query_decision_history(self, driver_id: str, step: int) -> dict[str, Any]:
        if step < -1:
            raise ValueError("step 须为 >= -1；-1 表示全部，0 表示返回 0 条，正整数表示末尾若干条")
        self._a.query_calls += 1
        records = list(self._a.history)
        out = _slice_history(records, step)
        return {
            "driver_id": driver_id.strip(),
            "total_steps": len(records),
            "step_param": step,
            "returned_count": len(out),
            "records": out,
        }

    # 4/4 —— 平台价值最集中的一处
    def model_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        resp = self._a.gate.call(
            payload,
            node=_current_node(),
            driver_id=self._a.driver_id,
            step=self._a.step,
        )
        # 逐次累计成本步 usage。形状与赛方 reset_last_model_usage / get_last_model_usage 一致，
        # 这样"本步花了多少 token"只有一个来源。
        usage = extract_usage(resp)
        self._a.step_usage = self._a.step_usage + usage
        self._a.llm_calls_this_step += 1
        return resp


# ---------------------------------------------------------------------------
# 向上：AgentAdapter 的实现
# ---------------------------------------------------------------------------
@register("truck")
class TruckAdapter:
    """项目一四节点 LangGraph 的 Adapter。

    关键实现约定：
      · **图在项目一的 `ModelDecisionService.__init__` 里 compile 一次**，
        这里 `reset()` 建一次 service，整月复用 —— 不要每步重建。
      · `env_step()` 用赛方的 `simkit.simulation_actions` 推进环境。
        这一层**不属于被测 Agent**，所以它放在 Adapter 而不是 service 里。
      · 内部有状态、且 `CargoRepository` 210 MB → `concurrency` 必须是 1。
    """

    # 可选能力：声明"我的每一次模型调用都已经走 LLMGate 计过账了"。
    # Runner 看到它就不再在外层按 step 预扣/结算 —— 否则同一笔 token 记两次，
    # TPM 桶被扣双份、RPM 桶按 (1 + 节点数) 计数，配额会被无声地砍到三分之一。
    owns_llm_limits = True

    def __init__(
        self,
        variant: str = "full",
        seed: int = 0,
        *,
        gate: Any = None,
        project_root: str | Path | None = None,
        duration_days: int = 5,
        nearest_cargo_limit: int = 100,
        cargo_view_batch_size: int = 10,
        reposition_speed_km_per_hour: float = 60.0,
        api_key: str | None = None,
        api_base: str | None = None,
        model_name: str | None = None,
        intercept_foreign_clients: bool = True,
    ) -> None:
        if gate is None:
            raise ValueError(
                "TruckAdapter 需要 Runner 注入 gate=<LLMGate>。"
                "直接 build('truck') 是拿不到网关的 —— 网关必须由 Runner 建，"
                "因为限流额度是跨司机共享的（见 evalharness/llm_gate.py 头部说明）。"
            )
        self.gate = gate
        self.variant = variant
        self.seed = seed
        self.duration_days = int(duration_days)
        self.nearest_cargo_limit = int(nearest_cargo_limit)
        self.cargo_view_batch_size = int(cargo_view_batch_size)
        self.reposition_speed_km_per_hour = float(reposition_speed_km_per_hour)

        self.api_key = api_key if api_key is not None else os.environ.get("DASHSCOPE_API_KEY", "")
        self.api_base = api_base or DEFAULT_API_BASE
        self.model_name = model_name or DEFAULT_MODEL
        # 是否接管项目一里绕过端口的自建客户端（agent3 / TimeAgent）。默认开，理由见
        # `_install_foreign_client_probe` 的文档串。关掉它你就能复现赛方口径，
        # 代价是那部分 token 在任何成本表里都看不见。
        self.intercept_foreign_clients = bool(intercept_foreign_clients)

        # 先把「项目一在哪」这条硬前提查掉，再谈 import。
        # 顺序是刻意的：报错信息要说人话，不能是 ImportError 里一串路径。
        self.demo = _resolve_demo_root(project_root)
        self._service_cls = _import_project_one(self.demo)

        # 运行期状态（reset() 里初始化）
        self.driver_id: str = ""
        self.step: int = 0
        self.history: list[dict[str, Any]] = []
        self.scan_minutes: int = 0
        self.step_usage: Usage = Usage()
        self.llm_calls_this_step: int = 0
        self.query_calls: int = 0
        self._closed = False
        self._service: Any = None

    # ------------------------------------------------------------ 协议 1/5
    def metadata(self) -> AgentMeta:
        return AgentMeta(
            name="truck-multiagent",
            variant=self.variant,
            model=self.model_name,
            prompt_version="v1",
            extra=(
                ("nodes", "agent1/agent2/agent3/agent4"),
                ("structure", "LangGraph StateGraph 串行：偏好分类→地点分析→时间分析→决策"),
                ("llm_nodes", "agent1/agent3/agent4（agent2 不调模型，纯 Python）"),
            ),
        )

    # ------------------------------------------------------------ 协议 2/5
    def reset(self, driver_id: str, seed: int) -> Observation:
        # 并发度必须为 1，两个硬原因见 MAX_SUPPORTED_CONCURRENCY 的注释。
        # 这里**刻意不抛异常**：并发配错应该由 Scenario 修，而不是让整个 run 当场失败；
        # 但这条约束必须写在代码里 —— 否则下一个人一定会试，而且会在跑到第二个司机时才炸。
        repo, manager = _shared_resources(self.demo)
        manager.load()  # 12 KB：顺带清空 _taken_cargo_ids 与订单计数
        manager.start_simulation(driver_id=driver_id, progress_minutes=0)
        _rewind_repo(repo)
        repo.sync_time_minutes(0)

        self._repo_obj = repo
        self._manager_obj = manager
        self.driver_id = driver_id
        self.step = 0
        self.history = []
        self.scan_minutes = 0
        self.step_usage = Usage()
        self.llm_calls_this_step = 0
        self.query_calls = 0
        self._closed = False

        self._port = _TruckPort(self)
        # ★ 必须先插桩再建 service。原因见 `_install_node_probes` 的文档串：
        #   图在 `ModelDecisionService.__init__` 里 compile，`add_node("agent1", self._agent1_node)`
        #   那一刻就把**绑定方法**抓走了，事后替换实例属性对已经编译好的图没有任何作用。
        #   （我第一版就是事后替换，结果是 69 次模型调用全部 node=unknown，
        #    节点归因表只剩一行 —— 而且不报错，看起来"跑通了"。）
        global _ACTIVE_ADAPTER
        _ACTIVE_ADAPTER = self
        self._install_node_probes(self._service_cls)
        self._service = self._service_cls(
            self._port,
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            reposition_speed=self.reposition_speed_km_per_hour,
        )
        self.foreign_intercepted = (
            self._install_foreign_client_probe(self._service)
            if self.intercept_foreign_clients
            else False
        )
        return self._observe()

    # ------------------------------------------------------------ 协议 3/5
    def act(self, obs: Observation) -> tuple[Action, Usage]:
        """跑一次四节点图，返回 (动作, 本步真实 token)。

        注意签名是**同步**的：项目一是同步代码，Runner 会用 `asyncio.to_thread` 隔离。
        我在这里什么都不用做 —— 适配器不需要知道平台是异步的。
        """
        if self._closed:
            raise RuntimeError("adapter 已 close()，不能再决策")
        if obs.step >= 0:
            self.step = int(obs.step)

        self.scan_minutes = 0
        self.step_usage = Usage()
        self.llm_calls_this_step = 0
        self.query_calls = 0

        t0 = time.perf_counter()
        action_dict = self._service.decide(self.driver_id)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if not isinstance(action_dict, dict):
            raise ValueError(f"项目一 decide() 必须返回 dict，实际是 {type(action_dict).__name__}")

        name = str(action_dict.get("action", "")).strip().lower() or "wait"
        raw_params = action_dict.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}

        # 把本步的内部计量挂到 span 上 —— 这一步的"决策质量"和"工程开销"必须能分开看。
        with self.gate.tracer.span(
            "custom:step_summary", driver_id=self.driver_id, step=self.step,
            variant=self.variant, action=name,
        ) as box:
            box["llm_calls"] = self.llm_calls_this_step
            box["env_queries"] = self.query_calls
            box["scan_minutes"] = self.scan_minutes
            box["nodes_ms"] = round(elapsed_ms, 2)
            box["tokens_this_step"] = self.step_usage.total

        return Action(name=name, params=params), self.step_usage

    # ------------------------------------------------------------ 协议 4/5
    def env_step(self, action: Action) -> EnvStep:
        """环境侧推进。**这一层不是被测 Agent** —— 它用赛方的 simulation_actions。

        与 `act()` 分开的理由：动作质量与环境反馈必须能分开归因。
        """
        from simkit import simulation_actions

        repo, manager = self._repo_obj, self._manager_obj
        driver_id = self.driver_id
        status_before = manager.get_driver_status(driver_id)
        sim_before = manager.get_simulation_progress_minutes()

        reward = 0.0
        result: dict[str, Any]

        if action.name == "wait":
            duration = int(action.params.get("duration_minutes", action.params.get("duration", 60)) or 60)
            result = simulation_actions.wait(repo, manager, driver_id, max(1, duration))

        elif action.name == "reposition":
            lat, lng = _pick_lat_lng(action.params)
            result = simulation_actions.reposition(
                repo, manager, driver_id, lat, lng,
                speed_km_per_hour=self.reposition_speed_km_per_hour,
            )

        elif action.name == "take_order":
            cargo_id = str(action.params.get("cargo_id", "")).strip()
            # 必须**在** take_order 之前读价：take_order 第一件事就是把它从在线池摘掉。
            price_yuan = 0.0
            peeked = repo.get_by_id(cargo_id) if cargo_id else None
            if isinstance(peeked, dict):
                try:
                    price_yuan = float(peeked.get("price", 0.0)) / 100.0  # 数据集里是分
                except (TypeError, ValueError):
                    price_yuan = 0.0
            try:
                result = simulation_actions.take_order(
                    repo, manager, driver_id, cargo_id,
                    reposition_speed_km_per_hour=self.reposition_speed_km_per_hour,
                    simulation_horizon_minutes=self.duration_days * MINUTES_PER_DAY,
                )
            except ValueError as exc:
                # 货源已被别人摘走 / 不在池里。**不算异常，算一步无效动作** ——
                # 这与赛方 orchestrator 的处理一致（它返回 accepted=False 并推进 1 分钟）。
                manager.advance_progress(driver_id, 1)
                repo.sync_time_minutes(manager.get_simulation_progress_minutes())
                result = {"accepted": False, "detail": f"cargo_unavailable: {exc}"}

            if result.get("accepted") and result.get("income_eligible", True):
                cost_per_km = float(status_before.get("cost_per_km") or 0.0)
                distance = float(result.get("pickup_deadhead_km", 0.0) or 0.0) + float(
                    result.get("haul_distance_km", 0.0) or 0.0
                )
                # 步级 reward = 运价 − 行驶成本。
                # 不含**偏好罚分**：罚分属于「评测口径」，不属于「环境回报」。
                # 混进来的后果是"环境替你被判了分"，A/B 时就说不清差异来自决策还是来自罚则。
                # 罚分单独在 D4 的 penalty_signal_ratio 里用 10 组历史跑批归因。
                reward = price_yuan - cost_per_km * distance
        else:
            # 未知动作不杀死评测：推进 1 分钟并记下来。
            # 「让它跑完并记账」比「当场抛异常」更有用 —— 后者的代价是丢掉整轮评测。
            manager.advance_progress(driver_id, 1)
            repo.sync_time_minutes(manager.get_simulation_progress_minutes())
            result = {"accepted": False, "detail": f"unknown_action: {action.name}"}

        sim_after = manager.get_simulation_progress_minutes()
        status_after = manager.get_driver_status(driver_id)
        elapsed = sim_after - sim_before

        # ★ 历史记录的字段形状必须**照抄赛方**
        #   （`server/bench/simulation_orchestrator.py:119-141` 的 `_normalize_for_output` 输入）。
        #   我第一版把 `"action"` 写成了动作名字符串，结果项目一在
        #   `model_decision_service.py:670` 的 `last_action.get("action")` 上直接
        #   `AttributeError: 'str' object has no attribute 'get'` ——
        #   赛方存的是**整个动作 dict** `{"action": name, "params": {...}}`。
        #   教训：适配层不是"给个差不多的 dict"就行，**被适配方会按字段名去 get**。
        #   少一个字段是 KeyError，字段类型错了是 AttributeError，两者都不报错地跑歪才最糟。
        action_exec_cost = max(0, elapsed - self.scan_minutes)
        self.history.append(
            {
                "step": self.step,
                "driver_id": driver_id,
                "step_elapsed_minutes": elapsed,
                "query_scan_cost_minutes": self.scan_minutes,
                "action_exec_cost_minutes": action_exec_cost,
                "position_before": {
                    "lat": float(status_before["current_lat"]),
                    "lng": float(status_before["current_lng"]),
                },
                "position_after": {
                    "lat": float(status_after["current_lat"]),
                    "lng": float(status_after["current_lng"]),
                },
                "simulation_end_time": _format_sim_clock(sim_after),
                "action": {
                    "action": action.name,
                    "params": dict(action.params),
                    # 平台口径的 usage（赛方这里是 get_last_model_usage() 的返回值）
                    "model_usage": {
                        "input_tokens": self.step_usage.input_tokens,
                        "output_tokens": self.step_usage.output_tokens,
                        "reasoning_tokens": self.step_usage.reasoning_tokens,
                        "total_tokens": self.step_usage.total,
                    },
                },
                "token_usage": {
                    "input_tokens": self.step_usage.input_tokens,
                    "output_tokens": self.step_usage.output_tokens,
                    "reasoning_tokens": self.step_usage.reasoning_tokens,
                    "total_tokens": self.step_usage.total,
                },
                "result": result if isinstance(result, dict) else str(result),
                # ---- 以下三个是平台自己加的，赛方没有 ----
                "reward": round(reward, 2),
                "llm_calls": self.llm_calls_this_step,
                "env_queries": self.query_calls,
            }
        )

        horizon = self.duration_days * MINUTES_PER_DAY
        # 与赛方主循环同一套终止条件：时间到、或货源池空（`while ... and self._repo.size > 0`）。
        terminated = sim_after >= horizon or repo.size == 0
        obs = Observation(
            step=-1,  # 步号由 Runner 统一回填，环境侧不自己数
            sim_minute=sim_after,
            payload={
                "driver_id": driver_id,
                "sim_wall_time": manager.get_simulation_wall_time(),
                "elapsed_minutes": elapsed,
                "remaining_cargo": repo.size,
                "position": {
                    "lat": float(status_after["current_lat"]),
                    "lng": float(status_after["current_lng"]),
                },
                "completed_orders": status_after.get("completed_order_count", 0),
                "status": status_after,
            },
        )
        return EnvStep(observation=obs, reward=round(reward, 2), terminated=terminated, truncated=False)

    # ------------------------------------------------------------ 协议 5/5
    def close(self) -> None:
        global _ACTIVE_ADAPTER
        self._closed = True
        self._service = None
        self._port = None
        if _ACTIVE_ADAPTER is self:
            _ACTIVE_ADAPTER = None
        # **不要**释放共享的 repo / manager：它们跨 adapter 实例共享，
        # 释放一次会让后面的司机丢掉数据源。共享资源的生命周期归模块，不归实例。
        # 这是"把 210 MB 只解析一次"的代价，也是它的前提。

    # ------------------------------------------------------------ 内部
    def repo(self) -> Any:
        return self._repo_obj

    def manager(self) -> Any:
        return self._manager_obj

    def _observe(self) -> Observation:
        manager = self._manager_obj
        return Observation(
            step=0,
            sim_minute=manager.get_simulation_progress_minutes(),
            payload={
                "driver_id": self.driver_id,
                "sim_wall_time": manager.get_simulation_wall_time(),
                "remaining_cargo": self._repo_obj.size,
                "status": manager.get_driver_status(self.driver_id),
            },
        )

    def _install_node_probes(self, service_cls: Any) -> None:
        """给四个节点各开一段 `custom:node_*` span，并在调用期间标记"当前节点名"。

        ★ 为什么必须是**类级**替换（`service_cls._agent1_node = ...`）而不是实例级
        ==========================================================================
        `ModelDecisionService.__init__` 的最后一步是 `self._graph = self._build_graph()`，
        而 `_build_graph()` 里写的是：

            workflow.add_node("agent1", self._agent1_node)   # 偏好分类
            ...
            return workflow.compile()

        `self._agent1_node` 在**这一行求值时就已经是绑定方法对象**了，LangGraph 把它存进了
        编译后的图。之后你再 `service._agent1_node = wrapped`，图里跑的仍然是旧的那一个。
        所以插桩必须发生在**类**上，且必须在**实例化之前**。

        为什么用运行期替换而不是改项目一源码
        ====================================
          1. 本仓库不含一个字节的项目一代码，所有插桩必须发生在运行期；
          2. 这是 adapter 该干的事 —— **被测对象的内部结构由适配层探测，平台不需要知道**。
        平台侧的配套设计是 `observability.CUSTOM_PREFIX`：9 段是公共词汇表，
        被测对象可以自报阶段，但必须带 `custom:` 前缀，所以公共聚合口径不会被污染。

        代价（写在明处）：类级替换是**进程级**的，所以包装函数不能闭包捕获 adapter 实例
        （同一个类可能被多个 adapter 用）。当前 adapter 走 `_ACTIVE` thread-local 传递。
        `concurrency: 1` 下这是安全的；若将来放开并发，每个驱动线程各自 `reset()` 时会
        各自覆盖 `_ACTIVE.adapter`，依然安全 —— 但**同一线程内不能并发两个 adapter**。
        """
        if getattr(service_cls, "_evalharness_node_probe", False):
            return
        for node, method_name in _NODE_METHODS:
            original = getattr(service_cls, method_name, None)
            if not callable(original):
                continue
            setattr(service_cls, method_name, _make_node_wrapper(node, original))
        service_cls._evalharness_node_probe = True  # type: ignore[attr-defined]

    def _wrap_node(self, node: str, fn: Any) -> Any:
        """保留给测试/单节点场景的实例级包装（类级插桩之外的一层保险）。"""
        def wrapped(state: Any) -> Any:
            previous = getattr(_NODE, "name", None)
            _NODE.name = node
            try:
                with self.gate.tracer.span(
                    f"custom:node_{node}",
                    driver_id=self.driver_id,
                    step=self.step,
                    variant=self.variant,
                ):
                    return fn(state)
            finally:
                _NODE.name = previous or "unknown"

        return wrapped

    def _install_foreign_client_probe(self, service: Any) -> bool:
        """接管项目一里**绕过 `SimulationApiPort` 的自建模型客户端**（agent3 / TimeAgent）。

        为什么必须有这一段
        ==================
        `TimeAgent.__init__` 里有一句（`demo/agent/time_agent_langchain.py:82`）：

            self._llm = ChatOpenAI(model=..., api_key=api_key or os.environ["DASHSCOPE_API_KEY"],
                                   base_url=api_base, callbacks=[self._callback])

        它**自己 new 了一个客户端**，不经过注入进来的 `api`。后果有三条，每条都很硬：

          1. 这部分 token 不在 `EmbeddedDecisionEnvironment.get_last_model_usage()` 里，
             → **赛方口径的 token 统计天然少记**。少记的比例见 docs/d1-wiring.md 第 6 节。
          2. 这部分请求不受限流、不进缓存、没有 span
             → **它既不慢也不贵，因为它压根没被测量**。
          3. dry-run 下它会**真的去打网络**并抛异常 → 接线验证根本跑不完。

        平台侧的处置原则（这一段代码就是那个原则的落点）
        ================================================
          · **不改项目一的一行源码** —— 运行期替换 `llm._generate`，属于适配层的探测行为；
          · 接过来的调用打上 `source="agent_own_client"`，与正规 `source="port"` 分开计数；
          · 计数要**看得见**：`llm.calls_foreign`。它应该是 0 —— 不是"忽略不计"，是 0。
            非 0 就是一个待修的结构缺陷，而且现在有数字可以证明。

        为什么替换 `_generate` 而不是别的钩子
        ====================================
        `_generate` 是 LangChain `BaseChatModel` 的**同步唯一出口**：不管上面是
        `create_agent` / `bind_tools` / `invoke`，最终都汇到这里。HTTP 层反而不行 ——
        langchain_openai 走的是 `client.with_raw_response.create(...)`，patch 不到。
        """
        if getattr(service, "_time_agent", None) is None:
            return False
        llm = getattr(service._time_agent, "_llm", None)
        if llm is None or getattr(llm, "_evalharness_probe", False):
            return False
        if not callable(getattr(llm, "_generate", None)):
            return False

        adapter = self
        model_label = str(getattr(llm, "model_name", "") or getattr(llm, "model", "") or "")

        def patched_generate(messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
            payload = _payload_from_messages(messages, model_label, kwargs)
            resp = adapter.gate.call(
                payload,
                node=_current_node(),
                driver_id=adapter.driver_id,
                step=adapter.step,
                source="agent_own_client",
            )
            # 与端口路径同一套记账口径：本步 usage / 调用数都 +1，不留第二套账。
            adapter.step_usage = adapter.step_usage + extract_usage(resp)
            adapter.llm_calls_this_step += 1
            return _chat_result_from_response(resp)

        llm._generate = patched_generate  # type: ignore[method-assign]
        llm._evalharness_probe = True  # type: ignore[attr-defined]
        return True


# ---------------------------------------------------------------------------
# 第三方客户端接管所需的两个转换器（lazy import，平台本身不强依赖 langchain）
# ---------------------------------------------------------------------------
def _payload_from_messages(messages: Any, model: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    """LangChain 消息列表 -> OpenAI 兼容 payload。

    先做一次 `json.dumps(default=str)` 往返：langchain 的 `tools` / `response_format`
    可能是 pydantic 对象，直接进缓存键会炸，也会让同一次调用算出两个 key。
    """
    out: list[dict[str, Any]] = []
    for m in messages or []:
        role = str(getattr(m, "type", "") or "")
        role = {"human": "user", "ai": "assistant"}.get(role, role or "user")
        content = getattr(m, "content", "")
        item: dict[str, Any] = {"role": role, "content": content if isinstance(content, str) else str(content)}
        if role == "tool":
            item["tool_call_id"] = getattr(m, "tool_call_id", "")
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "type": tc.get("type", "function"),
                    "function": {"name": tc.get("name", ""), "arguments": json.dumps(tc.get("args", {}), ensure_ascii=False, sort_keys=True)},
                }
                for tc in tool_calls
                if isinstance(tc, dict)
            ]
        out.append(item)

    payload: dict[str, Any] = {"model": model, "messages": out}
    for key in ("tools", "tool_choice", "temperature", "response_format"):
        if key in kwargs:
            payload[key] = kwargs[key]
    return json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def _chat_result_from_response(resp: dict[str, Any]) -> Any:
    """OpenAI 兼容响应 -> LangChain `ChatResult`。

    `llm_output["token_usage"]` 必须填：项目一的 `TokenStatsCallback` 就是从这里读 token 的。
    留空的话项目一自己的计数器会一直是 0 —— **接管了一个客户端却把它的账记没了**，
    那还不如不接管。
    """
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    choices = resp.get("choices") or [{}]
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content", "") or ""
    usage = extract_usage(resp)
    return ChatResult(
        generations=[
            ChatGeneration(
                message=AIMessage(content=content),
                generation_info={"finish_reason": first.get("finish_reason", "stop")},
            )
        ],
        llm_output={
            "token_usage": {
                "prompt_tokens": usage.input_tokens,
                "completion_tokens": usage.output_tokens,
                "total_tokens": usage.total,
                "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens},
            },
            "model_name": resp.get("model", ""),
        },
    )


def _make_node_wrapper(node: str, original: Any) -> Any:
    """生成类级节点包装函数。

    签名必须是 `(self, state)`：类级替换拿到的是**函数**，调用时 self 会被自动传入。
    """
    def wrapped(self: Any, state: Any) -> Any:
        previous = getattr(_NODE, "name", None)
        _NODE.name = node
        adapter = _ACTIVE_ADAPTER
        try:
            if adapter is None:  # 没有活跃 adapter（例如项目一自己的单测），原样执行
                return original(self, state)
            with adapter.gate.tracer.span(
                f"custom:node_{node}",
                driver_id=getattr(adapter, "driver_id", ""),
                step=getattr(adapter, "step", -1),
                variant=getattr(adapter, "variant", ""),
            ):
                return original(self, state)
        finally:
            _NODE.name = previous or "unknown"

    wrapped.__name__ = getattr(original, "__name__", f"node_{node}")
    wrapped._evalharness_node = node  # type: ignore[attr-defined]
    return wrapped


# ---------------------------------------------------------------------------
def _pick_lat_lng(params: dict[str, Any]) -> tuple[float, float]:
    """从 params 里取坐标。宽容一点，但取不到就报清楚的错。

    为什么宽容：`reposition` 的参数名在赛方 orchestrator 里是 `latitude`/`longitude`，
    而模型有时候会吐 `target_lat`/`target_lng`。**取不到时宁可报错也不要用 (0,0) 兜底** ——
    用 (0,0) 兜底会让司机"成功"空驶到一个不存在的位置，而且日志里看起来一切正常。
    """
    for lat_key, lng_key in (
        ("latitude", "longitude"),
        ("lat", "lng"),
        ("target_lat", "target_lng"),
        ("target_latitude", "target_longitude"),
    ):
        if lat_key in params and lng_key in params:
            try:
                return float(params[lat_key]), float(params[lng_key])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"reposition 坐标不是数字: {params[lat_key]!r}, {params[lng_key]!r}") from exc
    target = params.get("target")
    if isinstance(target, dict) and "lat" in target and "lng" in target:
        return float(target["lat"]), float(target["lng"])
    raise ValueError(
        f"reposition 缺少坐标，收到 params={params!r}。"
        "期望 latitude/longitude（赛方 orchestrator 的字段名）"
    )
