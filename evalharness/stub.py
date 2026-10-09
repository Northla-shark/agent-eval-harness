# -*- coding: utf-8 -*-
"""Layer 2.5 配套 · ScriptedTransport —— 不打真模型，但让被测 Agent 做出**合法**的决策。

为什么需要它（而不只是返回一个 "{}"）
======================================
`LLMGate._dry_transport` 返回 `content="{}"`。它能证明「链路通」，
但项目一的 `_parse_action()` 会把 `{}` 判为非法动作（action 不在
{take_order, reposition, wait} 里），于是走容错分支 —— **每步都退化成 wait**。

后果：跑了 23 步，但 `take_order` 和 `reposition` 这两条代码路径一次都没进过。
一次只覆盖三分之一的分支的"跑通"，是不能写进简历的。

ScriptedTransport 的解决办法：**按节点返回该项目一解析层能接受的合法 JSON**，
并且让动作在三条分支之间轮转 —— 于是不花一分钱，也能证明
「adapter 的 env_step 对三种动作的处理都是对的」。

它不假装自己是什么
==================
· 它**不产生真实 token**，usage 是按字符数估的（和 `_dry_transport` 一样）；
· 它**不评判决策质量** —— 它给出的"决策"是脚本，不是推理结果；
· 它只负责一件事：**把链路里除"模型智能"以外的部分全部走到**。
  评测平台的价值本来就不在于"我调的模型更聪明"，而在于"我能测出这个 Agent 到底在干什么"。

名词
====
· **Stub**：测试替身的一种，指"用预设答案替代真实依赖"。这里替代的是大模型。
  与之相对的还有 Mock（会断言"你有没有用正确参数调我"）、Spy（只记录调用）。
  我们这里是 Stub + 一点点 Spy（顺带记录了调用次数与 payload 大小）。
· **Fixture / 夹具**：把一次真实交互录下来，之后反复回放。这是"不花钱也能回归测试"的
  标准做法。ScriptedTransport 是手写夹具；下一步可以升级成"录一次真跑，之后回放"。
· **确定性（determinism）**：同样的输入永远得到同样的输出。真模型做不到（temperature > 0），
  所以**真模型恰恰是回归测试的敌人** —— 这也是"尽量不跑真模型"的工程依据，不是偷懒。
"""
from __future__ import annotations

import json
import re
from typing import Any

# --------------------------------------------------------------------------- #
# 从 prompt 里"反向提取"真实参数
#
# 为什么必须提取而不是写死：stub 给出的 cargo_id 必须是**当前这步真实存在的货源**，
# 否则环境会拒绝接单，take_order 分支依然走不到。
# 这就是 stub 相对"写死一个假 ID"的关键差别 —— 假 ID 会让被测对象的容错分支
# 把失败吞掉，于是你以为验证了 take_order，其实只验证了"接单失败的处理"。
# --------------------------------------------------------------------------- #
# 注意键名：项目一的货源坐标是 `start: {"lat": ..., "lng": ...}`，
# **不是** latitude/longitude（见 simkit/cargo_repository.py:160-162）。
# 司机坐标是 `current_lat` / `current_lng` —— 因为正则以 `"lat"` 开头引号匹配，
# `current_lat` 不会被误匹配，所以拿到的就是货源位置，正好是 reposition 该去的地方。
_CARGO_ID = re.compile(r'"cargo_id"\s*:\s*"([A-Za-z0-9_\-]{1,64})"')
_LAT = re.compile(r'"(?:lat|latitude)"\s*:\s*(-?\d+(?:\.\d+)?)')
_LNG = re.compile(r'"(?:lng|longitude)"\s*:\s*(-?\d+(?:\.\d+)?)')

# 项目一 `_agent1_node` 期待的形状（见 model_decision_service.py:113-115 的 prompt 示例）
AGENT1_RESPONSE: dict[str, Any] = {
    "time_pref": [{"content": "每天 23 点之后不再接单，次日 6 点开工"}],
    "geo_pref": [{"content": "不去东经 117.5 以东、北纬 39.2 以北的区域"}],
    "other_pref": [],
}

# 项目一 `_agent3_node` / `_parse_action` 会读的形状（见 model_decision_service.py:268-285 的占位结果）
# `monthly_rest_plan.should_schedule_rest_day` 必须给 False，
# 否则 agent4 会走"强制 wait"分支（第 402-421 行），脚本就白写了。
AGENT3_RESPONSE: dict[str, Any] = {
    "estimated_arrival_load": {},
    "estimated_arrival_unload": {},
    "daily_rest_met": True,
    "daily_rest_remaining_minutes": 0,
    "monthly_rest_urgency": 0.0,
    "month_free_days": 0,
    "month_free_days_target": 5,
    "current_time_in_no_order_window": False,
    "penalty_if_violate_daily": 0,
    "penalty_if_violate_monthly": 0,
    "cargo_time_reports": {},
    "monthly_rest_plan": {
        "should_schedule_rest_day": False,
        "rest_date": None,
        "recommended_wait_minutes": 0,
        "reason": "stub：不触发强制休息，以便覆盖 take_order / reposition 分支",
    },
}


class ScriptedTransport:
    """按节点返回预设响应的假模型。

    Args:
        script:    agent4 的动作轮转表。默认三条分支各来一遍。
        wait_minutes: wait 分支的时长。
        chars_per_token: 字符 -> token 的粗估系数（中英混排约 2 字符 1 token）。
        fallback_wait: 当 take_order 拿不到 cargo_id、或 reposition 拿不到坐标时，
                       降级成 wait 而不是抛错 —— 降级要有记录，见 `fallbacks`。
    """

    # 向网关声明"我是不花钱的替身"。网关据此把 llm.dry_run 置 1，
    # 于是报告里的 token 数字永远带着"这是估算"的标记，不会被误当成真实账单。
    is_fake = True

    def __init__(
        self,
        script: tuple[str, ...] = ("take_order", "reposition", "wait"),
        *,
        wait_minutes: int = 60,
        chars_per_token: int = 2,
        fallback_wait: bool = True,
        debug_dump: str | None = None,
    ) -> None:
        self.script = tuple(script)
        self.wait_minutes = wait_minutes
        self.chars_per_token = chars_per_token
        self.fallback_wait = fallback_wait
        self.debug_dump = debug_dump

        # Spy 侧面：记录它到底干了什么，便于在验收表里核对
        self.calls = 0
        self.by_node: dict[str, int] = {}
        self.actions_emitted: list[str] = []
        self.fallbacks: list[str] = []
        # 轮转游标。**必须是"第几次被调用"，不能是"第几次成功"** ——
        # 用成功次数算的话，一旦某分支连续失败，游标就永远卡在那儿，
        # 于是后面每一步都在重试同一个注定失败的分支（这个 bug 真踩过）。
        self._seq = 0

    # ------------------------------------------------------------------ 调用
    def __call__(
        self,
        payload: dict[str, Any],
        *,
        node: str = "unknown",
        driver_id: str = "",
        step: int = -1,
    ) -> dict[str, Any]:
        """签名故意带上下文参数：网关知道"是谁在调"，stub 才能按节点作答。

        这是 stub 相对"在 HTTP 层打桩"的优势 —— HTTP 层只能看到一坨 payload，
        分不清这次是偏好分类还是最终决策。
        """
        self.calls += 1
        self.by_node[node] = self.by_node.get(node, 0) + 1

        content = self._content_for(node, payload)
        return {
            "id": "stub",
            "object": "chat.completion",
            "model": "stub",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": self._usage(payload, content),
            "_stub": True,
            "_stub_node": node,
        }

    # ------------------------------------------------------------------ 内容
    def _content_for(self, node: str, payload: dict[str, Any]) -> str:
        n = (node or "").lower()
        if n in {"preference", "agent1"}:
            return json.dumps(AGENT1_RESPONSE, ensure_ascii=False)
        if n in {"geo", "agent2"}:
            # agent2 是纯 Python 节点，正常不会走到这里。走到说明接线变了 —— 如实返回空对象。
            return "{}"
        if n in {"time", "agent3"}:
            return json.dumps(AGENT3_RESPONSE, ensure_ascii=False)
        if n in {"decision", "agent4"}:
            return json.dumps(self._next_action(payload), ensure_ascii=False)
        # 未知节点：宁可返回空对象让它走容错，也不要编一个形状出来。
        return "{}"

    def _next_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        """按轮转表产出下一个动作。**拿不到真实参数就降级，不伪造。**"""
        idx = self._seq
        self._seq += 1
        if not self.script:
            return {"action": "wait", "params": {"duration_minutes": self.wait_minutes}}
        wanted = self.script[idx % len(self.script)]
        blob = self._payload_text(payload)

        if wanted == "take_order":
            m = _CARGO_ID.search(blob)
            if not m:
                self._dump("take_order", blob)
                self.fallbacks.append("take_order: prompt 里找不到 cargo_id")
                if not self.fallback_wait:
                    raise RuntimeError("prompt 里没有 cargo_id，无法构造 take_order")
                self.actions_emitted.append("wait_fallback")
                return {"action": "wait", "params": {"duration_minutes": self.wait_minutes}}
            self.actions_emitted.append("take_order")
            return {"action": "take_order", "params": {"cargo_id": m.group(1)}}

        if wanted == "reposition":
            la, ln = _LAT.search(blob), _LNG.search(blob)
            if not (la and ln):
                self._dump("reposition", blob)
                self.fallbacks.append("reposition: prompt 里找不到经纬度")
                if not self.fallback_wait:
                    raise RuntimeError("prompt 里没有经纬度，无法构造 reposition")
                self.actions_emitted.append("wait_fallback")
                return {"action": "wait", "params": {"duration_minutes": self.wait_minutes}}
            self.actions_emitted.append("reposition")
            return {
                "action": "reposition",
                "params": {"latitude": float(la.group(1)), "longitude": float(ln.group(1))},
            }

        self.actions_emitted.append("wait")
        return {"action": "wait", "params": {"duration_minutes": self.wait_minutes}}

    def _dump(self, why: str, blob: str) -> None:
        """只在排查时开。把 prompt 落盘，而不是打印 —— 一次跑 22 步，打印会淹没终端。"""
        if not self.debug_dump:
            return
        from pathlib import Path

        Path(self.debug_dump).write_text(
            f"# {why}\n\n{blob[:8000]}", encoding="utf-8"
        )

    # ------------------------------------------------------------------ 内部
    @staticmethod
    def _payload_text(payload: dict[str, Any]) -> str:
        """把 messages 拼成一段可搜索的文本。prompt 的具体结构由被测对象决定，
        平台不去猜它的字段名 —— 只用正则在整段文本里找，这样换 prompt 也不容易失效。"""
        parts: list[str] = []
        for msg in payload.get("messages") or []:
            if not isinstance(msg, dict):
                continue
            c = msg.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):  # 多模态 content 是 list[dict]
                for item in c:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
        if not parts:
            parts.append(json.dumps(payload, ensure_ascii=False))
        return "\n".join(parts)

    def _usage(self, payload: dict[str, Any], content: str) -> dict[str, Any]:
        """粗估。**明确声明这是估算** —— 见 counters() 里的 llm.dry_run 标记，
        任何把这里的数字当真实账单用的行为都是错的。"""
        text = self._payload_text(payload)
        prompt_tokens = max(1, len(text) // self.chars_per_token)
        completion_tokens = max(1, len(content) // self.chars_per_token)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "completion_tokens_details": {"reasoning_tokens": 0},
        }

    # ------------------------------------------------------------------ 观测
    def counters(self) -> dict[str, Any]:
        return {
            "stub.calls": self.calls,
            "stub.by_node": dict(self.by_node),
            "stub.actions": list(self.actions_emitted),
            "stub.fallbacks": list(self.fallbacks),
        }


def build_scripted_transport(**kw: Any) -> ScriptedTransport:
    """工厂函数。放在这里是给 `tools/run_d1.py` 一个稳定的导入点，
    也方便以后换成"录制回放"实现时只改这一处。"""
    return ScriptedTransport(**kw)
