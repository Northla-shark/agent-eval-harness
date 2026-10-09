# -*- coding: utf-8 -*-
"""合成环境替身 —— 让仓库「clone 下来就能跑一次 truck 形状的评测」。

它同时服务两个目的，这也是它敢放在仓库里的原因：

  1. **接线验证**（`tools/check_truck.py` 用它）：在没有项目一、没有 API key、
     没有 210 MB 赛题数据的机器上，把 `TruckAdapter` 的 5 个协议方法 + 4 个 port 方法
     全部走一遍。跑通说明 **adapter 自己的逻辑是对的**，
     剩下的风险只剩"真项目一的字段名对不对"——那是 D1.2 的事。

  2. **合成数据**：合规裁定要求仓库里只能放合成数据（赛题《特别说明》六、5
     「不得用于任何商业用途」+ 复赛数据集不对外发布）。所以 20 条货源、2 个司机
     **全是手写的假数据**，字段名对齐官方文档，数值全是我编的。

  它不是仿真器：几何、时序、成本都是最简实现，只保证"行为形状"与赛方一致
  （时间单调向前、货源上架/下架、空驶耗时、装货窗）。**不要用它算任何业务指标。**
"""
from __future__ import annotations

import math
import sys
import types
from datetime import datetime, timedelta
from typing import Any

SIM_EPOCH = datetime(2026, 3, 1, 0, 0, 0)
WALL_FMT = "%Y-%m-%d %H:%M:%S"


def _wall(minutes: int) -> str:
    return (SIM_EPOCH + timedelta(minutes=int(minutes))).strftime(WALL_FMT)


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371.0
    p1, l1, p2, l2 = map(math.radians, (lat1, lng1, lat2, lng2))
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin((l2 - l1) / 2) ** 2
    return 2.0 * r * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def _minutes_for(km: float, speed_kmh: float) -> int:
    if km <= 0:
        return 1
    return max(1, math.ceil(km / max(1e-6, speed_kmh) * 60))


# ---------------------------------------------------------------------------
# 合成的货源与司机（数量很少、数值全假）
# ---------------------------------------------------------------------------
def build_cargos() -> list[dict[str, Any]]:
    """20 条假货源。create_time 分散在头 3 天，保证任何一天都有货可接。"""
    hubs = [(31.23, 121.47), (32.06, 118.80), (30.27, 120.15), (34.75, 113.62)]
    out: list[dict[str, Any]] = []
    for i in range(20):
        a = hubs[i % len(hubs)]
        b = hubs[(i + 1 + i % 2) % len(hubs)]
        create = (i % 3) * 1440 + 360 + i * 37
        out.append(
            {
                "cargo_id": f"C{i:04d}",
                "create_time": _wall(create),
                "remove_time": _wall(create + 5 * 1440),
                "cost_time_minutes": 180 + (i % 5) * 60,
                "price": 120000 + (i % 7) * 45000,  # 单位：分
                "start": {"lat": a[0] + i * 0.002, "lng": a[1] - i * 0.002},
                "end": {"lat": b[0] - i * 0.001, "lng": b[1] + i * 0.001},
                "load_time": [_wall(create + 120), _wall(create + 24 * 60)],
            }
        )
    return out


def build_drivers() -> list[dict[str, Any]]:
    return [
        {
            "driver_id": "D001",
            "name": "合成司机甲",
            "vehicle_no": "假A00001",
            "truck_length": "13m",
            "cost_per_km": 6.5,
            "current_lat": 31.23,
            "current_lng": 121.47,
            "preferences": [
                {"content": "每晚需连续休息满 8 小时", "penalty_amount": 300, "penalty_cap": 6000},
                {"content": "自然月内至少有 4 个整天不接单", "penalty_amount": 3000, "penalty_cap": 6000},
            ],
        },
        {
            "driver_id": "D002",
            "name": "合成司机乙",
            "vehicle_no": "假A00002",
            "truck_length": "9.6m",
            "cost_per_km": 5.2,
            "current_lat": 32.06,
            "current_lng": 118.80,
            "preferences": [
                {"content": "每天 23:00 前回家且到次日 08:00 不再接单", "penalty_amount": 600, "penalty_cap": 6000},
            ],
        },
    ]


# ---------------------------------------------------------------------------
# CargoRepository 替身
# ---------------------------------------------------------------------------
class FakeCargoRepository:
    """真实实现用 numpy 向量化最近邻；这里用纯 Python —— 只验证"行为形状"。"""

    def __init__(self, path: Any = None, earth_radius_km: float = 6371.0) -> None:
        self._pending: list[tuple[int, int, str, dict[str, Any]]] = []
        self._pending_cursor = 0
        self._online: dict[str, tuple[int, dict[str, Any]]] = {}
        self._current_time_minutes = 0
        # 与真实实现同名，因为 truck.py 的 _rewind_repo 会重置这几个字段。
        # 这本身就是那条 shim 的必要性证明：接口一致才能互换。
        self._online_expire_heap: list[tuple[int, str]] = []
        self._online_dirty = True

    @property
    def size(self) -> int:
        return (len(self._pending) - self._pending_cursor) + len(self._online)

    def load(self) -> None:
        items = build_cargos()
        self._pending = sorted(
            (
                self._to_minutes(c["create_time"]),
                self._to_minutes(c["remove_time"]),
                c["cargo_id"],
                c,
            )
            for c in items
        )
        self._pending_cursor = 0
        self._online = {}
        self._current_time_minutes = 0
        self._online_dirty = True

    def get_by_id(self, cargo_id: str) -> dict[str, Any] | None:
        item = self._online.get(cargo_id)
        return None if item is None else item[1]

    def remove_by_id(self, cargo_id: str) -> dict[str, Any] | None:
        item = self._online.pop(cargo_id, None)
        return None if item is None else item[1]

    def nearest_pickup_km(self, latitude: float, longitude: float, current_time_minutes: int, k: int = 100):
        self.sync_time_minutes(current_time_minutes)
        pairs = [
            (_haversine_km(latitude, longitude, c["start"]["lat"], c["start"]["lng"]), c)
            for _, c in self._online.values()
        ]
        pairs.sort(key=lambda p: p[0])
        return pairs[: max(0, k)]

    def sync_time_minutes(self, current_time_minutes: int) -> None:
        if current_time_minutes < 0:
            current_time_minutes = 0
        # ★ 与真实实现一致：**单调向前**。第二条断言就是 truck.py 里 _rewind_repo 的存在理由。
        if current_time_minutes < self._current_time_minutes:
            return
        self._current_time_minutes = current_time_minutes
        while self._pending_cursor < len(self._pending):
            create, remove, cid, rec = self._pending[self._pending_cursor]
            if create > current_time_minutes:
                break
            self._pending_cursor += 1
            if remove >= current_time_minutes:
                self._online[cid] = (remove, rec)
        for cid, (remove, _) in list(self._online.items()):
            if remove < current_time_minutes:
                del self._online[cid]

    def wall_time_to_simulation_minutes(self, text: str) -> int:
        return self._to_minutes(text)

    @staticmethod
    def _to_minutes(text: str) -> int:
        return int((datetime.strptime(text.strip(), WALL_FMT) - SIM_EPOCH).total_seconds() // 60)


# ---------------------------------------------------------------------------
# DriverStateManager 替身
# ---------------------------------------------------------------------------
class FakeDriverStateManager:
    def __init__(self, drivers_path: Any = None) -> None:
        self._drivers: dict[str, dict[str, Any]] = {}
        self._current_driver_id: str | None = None
        self._progress = 0
        self._started = False
        self._taken: set[str] = set()
        self._orders: dict[str, int] = {}

    def load(self) -> None:
        self._drivers = {d["driver_id"]: dict(d) for d in build_drivers()}
        self._current_driver_id = next(iter(self._drivers))
        self._progress = 0
        self._started = False
        self._taken = set()
        self._orders = {k: 0 for k in self._drivers}

    def list_driver_ids(self) -> list[str]:
        return list(self._drivers)

    def start_simulation(self, driver_id: str | None, progress_minutes: int = 0) -> dict[str, Any]:
        chosen = driver_id or self._current_driver_id
        if chosen not in self._drivers:
            raise ValueError(f"driver_id 不存在: {chosen}")
        self._current_driver_id = chosen
        self._started = True
        self._progress = int(progress_minutes)
        return self.get_driver_status(chosen)

    def get_driver_status(self, driver_id: str) -> dict[str, Any]:
        p = self._drivers[driver_id]
        return {
            "driver_id": driver_id,
            "name": p.get("name", ""),
            "vehicle_no": p.get("vehicle_no", ""),
            "truck_length": p.get("truck_length", ""),
            "cost_per_km": p.get("cost_per_km"),
            "preferences": list(p.get("preferences", [])),
            "current_lat": float(p["current_lat"]),
            "current_lng": float(p["current_lng"]),
            "simulation_started": self._started,
            "simulation_progress_minutes": self._progress,
            "simulation_wall_time": self.get_simulation_wall_time(),
            "completed_order_count": self._orders.get(driver_id, 0),
        }

    def ensure_active_driver(self, driver_id: str) -> None:
        if driver_id not in self._drivers:
            raise ValueError(f"driver_id 不存在: {driver_id}")
        if not self._started:
            raise ValueError("模拟尚未开始")
        # 与真实实现一致：一次只能有一个"当前司机"。这就是 truck 并发度必须为 1 的原因。
        if driver_id != self._current_driver_id:
            raise ValueError(f"当前正在模拟 {self._current_driver_id}，不允许操作 {driver_id}")

    def update_driver_position(self, driver_id: str, latitude: float, longitude: float) -> None:
        self.ensure_active_driver(driver_id)
        self._drivers[driver_id]["current_lat"] = float(latitude)
        self._drivers[driver_id]["current_lng"] = float(longitude)

    def advance_progress(self, driver_id: str, duration_minutes: int) -> int:
        self.ensure_active_driver(driver_id)
        self._progress += int(duration_minutes)
        return self._progress

    def get_simulation_progress_minutes(self) -> int:
        return self._progress

    def get_simulation_wall_time(self) -> str:
        return _wall(self._progress)

    def take_order(self, driver_id, cargo_id, duration_minutes, end_latitude, end_longitude):
        self.ensure_active_driver(driver_id)
        if cargo_id in self._taken:
            raise ValueError(f"cargo_id 已被占用: {cargo_id}")
        self._taken.add(cargo_id)
        self._orders[driver_id] = self._orders.get(driver_id, 0) + 1
        self._progress += int(duration_minutes)
        self._drivers[driver_id]["current_lat"] = float(end_latitude)
        self._drivers[driver_id]["current_lng"] = float(end_longitude)
        return {
            "accepted": True,
            "detail": "合成环境：已完单",
            "driver_id": driver_id,
            "cargo_id": cargo_id,
            "simulation_progress_minutes": self._progress,
            "simulation_wall_time": self.get_simulation_wall_time(),
        }


# ---------------------------------------------------------------------------
# simulation_actions 替身（函数名与语义对齐赛方）
# ---------------------------------------------------------------------------
def _query_cargo(repo, manager, driver_id, latitude, longitude, k):
    manager.ensure_active_driver(driver_id)
    now = manager.get_simulation_progress_minutes()
    pairs = repo.nearest_pickup_km(latitude, longitude, current_time_minutes=now, k=k)
    return {
        "driver_id": driver_id,
        "items": [{"distance_km": float(d), "cargo": {**c, "price": round(c["price"] / 100.0, 2)}} for d, c in pairs],
    }


def _apply_scan_cost(repo, manager, driver_id, items_count, *, cargo_view_batch_size):
    cost = math.ceil(items_count / cargo_view_batch_size) if items_count else 0
    if cost > 0:
        manager.advance_progress(driver_id, cost)
        repo.sync_time_minutes(manager.get_simulation_progress_minutes())
    return manager.get_simulation_progress_minutes()


def _wait(repo, manager, driver_id, duration_minutes):
    manager.advance_progress(driver_id, duration_minutes)
    repo.sync_time_minutes(manager.get_simulation_progress_minutes())
    return {"simulation_progress_minutes": manager.get_simulation_progress_minutes(),
            "simulation_wall_time": manager.get_simulation_wall_time()}


def _reposition(repo, manager, driver_id, latitude, longitude, speed_km_per_hour):
    cur = manager.get_driver_status(driver_id)
    km = _haversine_km(cur["current_lat"], cur["current_lng"], latitude, longitude)
    manager.update_driver_position(driver_id, latitude, longitude)
    manager.advance_progress(driver_id, _minutes_for(km, speed_km_per_hour))
    repo.sync_time_minutes(manager.get_simulation_progress_minutes())
    return {"current_lat": latitude, "current_lng": longitude,
            "simulation_progress_minutes": manager.get_simulation_progress_minutes(),
            "simulation_wall_time": manager.get_simulation_wall_time(),
            "distance_km": round(km, 2)}


def _take_order(repo, manager, driver_id, cargo_id, reposition_speed_km_per_hour, simulation_horizon_minutes=None):
    cargo = repo.remove_by_id(cargo_id)
    if cargo is None:
        raise ValueError(f"cargo_id 不存在: {cargo_id}")
    cur = manager.get_driver_status(driver_id)
    dead_km = _haversine_km(cur["current_lat"], cur["current_lng"], cargo["start"]["lat"], cargo["start"]["lng"])
    if dead_km > 1e-6:
        manager.update_driver_position(driver_id, cargo["start"]["lat"], cargo["start"]["lng"])
        manager.advance_progress(driver_id, _minutes_for(dead_km, reposition_speed_km_per_hour))
        repo.sync_time_minutes(manager.get_simulation_progress_minutes())
    res = manager.take_order(
        driver_id, cargo_id, duration_minutes=int(cargo["cost_time_minutes"]),
        end_latitude=cargo["end"]["lat"], end_longitude=cargo["end"]["lng"],
    )
    repo.sync_time_minutes(int(res["simulation_progress_minutes"]))
    res["pickup_deadhead_km"] = round(dead_km, 2)
    res["haul_distance_km"] = round(_haversine_km(cargo["start"]["lat"], cargo["start"]["lng"],
                                                  cargo["end"]["lat"], cargo["end"]["lng"]), 2)
    res["income_eligible"] = True
    return res


# ---------------------------------------------------------------------------
# 假的项目一决策服务
# ---------------------------------------------------------------------------
def build_fake_service_class(bypass_port: bool = True) -> type:
    """返回一个 `ModelDecisionService` 的替身。

    它**刻意复刻项目一的两个结构特征**，因为这两个特征正是 D1 要量出来的东西：

      1. 四个节点串行：agent1 偏好分类 → agent2 地点分析 → agent3 时间分析 → agent4 决策
      2. **agent2 不调模型**（纯 Python）；**agent3 走自己的客户端**，
         不经过 `self._api.model_chat_completion`（真实代码见 time_agent_langchain.py
         第 82–87 行：它自己 new 了一个 ChatOpenAI）。

    参数 `bypass_port=True` 复刻真实结构；改成 False 可以看"修好之后"的 token 账。
    `check_truck.py` 两种都跑，差值就是被漏算的那部分。
    """

    class FakeModelDecisionService:
        #: 平台看不到的那部分调用记在这里（模拟 TimeAgent 自带的 TokenStatsCallback）
        invisible_tokens = 0
        invisible_calls = 0

        def __init__(self, api: Any, api_key: str = "", api_base: str = "", model_name: str = "",
                     reposition_speed: float = 50.0) -> None:
            self._api = api
            self._model_name = model_name or "dry-run"
            self._reposition_speed = reposition_speed
            self._bypass = bypass_port
            self.graph_compiles = 1  # 真实代码在 __init__ 里 compile 一次，这里只留个标记

        # --- 四个节点 ---
        def _agent1_node(self, state: dict[str, Any]) -> dict[str, Any]:
            resp = self._api.model_chat_completion(
                {"model": self._model_name, "messages": [
                    {"role": "system", "content": "你是偏好分类器。只输出 JSON。"},
                    {"role": "user", "content": "把司机偏好分成时间/地点两类。"}]}
            )
            return {"agent1_result": resp}

        def _agent2_node(self, state: dict[str, Any]) -> dict[str, Any]:
            # ★ 纯 Python：真实代码里 agent2 一段模型调用都没有。
            st = self._api.get_driver_status(state.get("driver_id", "D001"))
            return {"agent2_result": {"lat": st["current_lat"], "lng": st["current_lng"]}}

        def _agent3_node(self, state: dict[str, Any]) -> dict[str, Any]:
            cargo = self._api.query_cargo(state.get("driver_id", "D001"), 31.23, 121.47)
            if self._bypass:
                # 自建客户端：平台看不见。这里用"本地记账"模拟它。
                type(self).invisible_calls += 1
                type(self).invisible_tokens += 900
                return {"agent3_result": {"feasible": len(cargo.get("items", []))}}
            resp = self._api.model_chat_completion(
                {"model": self._model_name, "messages": [
                    {"role": "system", "content": "你是时间约束分析专家。只输出 JSON。"},
                    {"role": "user", "content": "分析时间可行性。"}]}
            )
            return {"agent3_result": resp}

        def _agent4_node(self, state: dict[str, Any]) -> dict[str, Any]:
            hist = self._api.query_decision_history(state.get("driver_id", "D001"), 3)
            resp = self._api.model_chat_completion(
                {"model": self._model_name, "messages": [
                    {"role": "system", "content": "你是卡车司机的决策者。只输出 JSON。"},
                    {"role": "user", "content": f"已决策 {hist['total_steps']} 步，给出下一步动作。"}]}
            )
            return {"agent4_result": resp}

        def decide(self, driver_id: str) -> dict[str, Any]:
            state: dict[str, Any] = {"driver_id": driver_id, "step": 0}
            for fn in (self._agent1_node, self._agent2_node, self._agent3_node, self._agent4_node):
                state.update(fn(state))
            return self._parse(driver_id, state)

        def _parse(self, driver_id: str, state: dict[str, Any]) -> dict[str, Any]:
            """合成策略：有货就接最近的，否则等 60 分钟。

            形状严格对齐赛方 orchestrator `_apply_action()` 认的字段：
              wait        -> params.duration_minutes
              reposition  -> params.latitude / params.longitude
              take_order  -> params.cargo_id
            """
            items = []
            try:
                items = self._api.query_cargo(driver_id, 31.23, 121.47).get("items", [])
            except Exception:
                items = []
            if items:
                cid = items[0]["cargo"]["cargo_id"]
                return {"action": "take_order", "params": {"cargo_id": cid}}
            return {"action": "wait", "params": {"duration_minutes": 60}}

    return FakeModelDecisionService


# ---------------------------------------------------------------------------
# 装进 sys.modules
# ---------------------------------------------------------------------------
def install(bypass_port: bool = True) -> None:
    """把上述替身注册成 `simkit.*` 与 `agent.*`，让 TruckAdapter 原样 import 到它们。

    为什么不改 TruckAdapter 的 import：**适配器的代码必须与生产路径完全一致**，
    测试才有效。所以这里动的是 sys.modules，不是被测代码。
    """
    simkit = types.ModuleType("simkit")
    simkit.__path__ = []  # 声明为包

    m_repo = types.ModuleType("simkit.cargo_repository")
    m_repo.CargoRepository = FakeCargoRepository
    m_dsm = types.ModuleType("simkit.driver_state_manager")
    m_dsm.DriverStateManager = FakeDriverStateManager
    m_act = types.ModuleType("simkit.simulation_actions")
    m_act.query_cargo = _query_cargo
    m_act.apply_cargo_query_scan_cost = _apply_scan_cost
    m_act.wait = _wait
    m_act.reposition = _reposition
    m_act.take_order = _take_order
    m_ports = types.ModuleType("simkit.ports")

    simkit.cargo_repository = m_repo
    simkit.driver_state_manager = m_dsm
    simkit.simulation_actions = m_act
    simkit.ports = m_ports

    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = []
    m_svc = types.ModuleType("agent.model_decision_service")
    m_svc.ModelDecisionService = build_fake_service_class(bypass_port=bypass_port)
    agent_pkg.model_decision_service = m_svc

    for name, mod in (
        ("simkit", simkit),
        ("simkit.cargo_repository", m_repo),
        ("simkit.driver_state_manager", m_dsm),
        ("simkit.simulation_actions", m_act),
        ("simkit.ports", m_ports),
        ("agent", agent_pkg),
        ("agent.model_decision_service", m_svc),
    ):
        sys.modules[name] = mod
