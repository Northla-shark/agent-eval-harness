# -*- coding: utf-8 -*-
"""Layer 2 · Observability —— 只做语义映射，不做存储。

一句话划清边界：**Langfuse 只做存储与可视化，Span 的命名与 schema 是我们自己定义的。**
这句话面试必须能讲 —— 否则你的设计贡献会被归给开源项目。

9 段 Span 是"阶段"划分（输入预处理 → ... → 记忆回写），
而 Trace/Span/Generation 是"层级"划分。两者不冲突，但要能同时讲出来。
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol

# 9 段 span 的权威清单。用 assert 锁死，防止命名漂移 ——
# 一旦有人随手写个 "llm_call"，指标层就再也聚不起来了。
SPAN_NAMES: tuple[str, ...] = (
    "input_preprocess",     # 1 解析 / 意图 / 敏感词
    "memory_retrieve",      # 2 短期 / 长期 / RAG
    "prompt_assemble",      # 3 各内容块的 token 占比
    "llm_inference",        # 4 含 TTFT 与生成耗时拆分
    "decision_parse",       # 5 记录命中第几级容错
    "tool_execute",         # 6 工具名 / 耗时 / 返回体大小
    "result_verify",        # 7 合法性校验（语义侧 Guardrail）
    "response_generate",    # 8 本 Agent 输出动作不输出回复 -> 标 skipped
    "memory_writeback",     # 9 记忆回写
)


# 扩展轴：上面 9 段是**平台保留的公共词汇表**，指标层按它们聚合。
# 被测对象内部的阶段（例如项目一的 agent1/agent2/agent3/agent4 四个节点）
# 属于"它的结构"，**不该被硬编码进平台** —— 否则换一个 Agent 就得改平台。
# 所以给一条通用的口子：`custom:` 前缀。
#
# 一句话讲法：「平台只规定 9 个公共阶段，被测对象可以自报阶段，
# 但必须带前缀 —— 这样我的聚合口径不会被任何单个 Agent 的结构污染。」
CUSTOM_PREFIX = "custom:"


def is_valid_span_name(name: str) -> bool:
    """合法的 span 名 = 9 段公共词汇 或 带 `custom:` 前缀的自报阶段。"""
    return name in SPAN_NAMES or name.startswith(CUSTOM_PREFIX)


class Tracer(Protocol):
    """Tracing 后端的最小协议。换 Langfuse / Phoenix / 自建 collector 都实现它。"""

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        """进入一段 span。yield 出来的 box 用于往当前 span 里塞业务指标。"""
        ...


@dataclass
class SpanRecord:
    name: str
    driver_id: str
    step: int
    ms: float
    attrs: dict[str, Any] = field(default_factory=dict)


class ConsoleTracer:
    """控制台 tracer：没有任何依赖，也不需要 API key。

    它存在的理由不是"临时方案"，而是：
      1. Day 1 就能看到端到端链路跑通，不必等 Langfuse 配好；
      2. 调试时它比任何 UI 都快；
      3. 它证明了 Tracer 是一个可替换的协议。
    """

    def __init__(self) -> None:
        self.records: list[SpanRecord] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        assert is_valid_span_name(name), (
            f"未定义的 span 名: {name!r} —— 要么用 SPAN_NAMES 里的公共阶段，"
            f"要么用 {CUSTOM_PREFIX!r} 前缀自报阶段"
        )
        box: dict[str, Any] = {}
        t0 = time.perf_counter()
        try:
            yield box
        finally:
            self.records.append(
                SpanRecord(
                    name=name,
                    driver_id=str(attrs.get("driver_id", "")),
                    step=int(attrs.get("step", -1)),
                    ms=(time.perf_counter() - t0) * 1000.0,
                    attrs={**attrs, **box},
                )
            )

    # ---- 观测口径 ----

    def coverage(self) -> dict[str, int]:
        """每段 span 各出现多少次。

        验收标准就写在这里：9 段中至少 6 段有数据。
        另外 2 段（memory_retrieve / memory_writeback）在你做出 Memory 模块之前
        应该是 0 —— 这是"我缺什么"的诚实答案，不要硬塞。
        """
        out = {name: 0 for name in SPAN_NAMES}
        for r in self.records:
            # custom: 前缀的不计入 —— 公共口径不能被单个被测对象的结构污染。
            if r.name in out:
                out[r.name] += 1
        return out

    def custom_coverage(self) -> dict[str, int]:
        """被测对象自报的阶段各出现多少次（例如 custom:node_preference）。"""
        out: dict[str, int] = {}
        for r in self.records:
            if r.name.startswith(CUSTOM_PREFIX):
                out[r.name] = out.get(r.name, 0) + 1
        return dict(sorted(out.items()))

    def render_waterfall(self, driver_id: str, limit: int = 30) -> str:
        """把某个司机的一小段 span 打成瀑布图（文本版）。

        面试演示用这个就够了：它证明"某司机某一步的完整链路能被单独审"，
        这句话本身比 UI 截图更有信息量。
        """
        rows = [r for r in self.records if r.driver_id == driver_id][:limit]
        if not rows:
            return f"（driver={driver_id} 没有 span 记录）"
        lines = [f"driver={driver_id}  共 {len(rows)} 段 span"]
        for r in rows:
            bar = "#" * max(1, int(r.ms / 2))
            lines.append(
                f"  step={r.step:<3} {r.name:<18} {r.ms:7.2f}ms {bar}"
            )
        return "\n".join(lines)


class LangfuseTracer(ConsoleTracer):
    """D3 任务：接 Langfuse。

    接入步骤（三步，别自己发明）：
      1) uv pip install -r requirements-langfuse.txt
      2) 复制 .env.example 为 .env，填 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
         （本机 Docker 未安装 —— 不要试自部署，直接用云端免费额度）
      3) 把下面 span() 里的 raise 换成：
             with self._lf.start_as_current_span(name=name, metadata=attrs) as sp:
                 box = {}
                 yield box
                 sp.update(metadata={**attrs, **box})

    映射表（这张表才是你的设计，要能画出来）：
        一个司机的一整月   -> Trace   (session_id = driver_id)
        一次单步决策       -> Span
        一次 LLM 调用      -> Generation（自动带 token 与延迟）
        消融分组           -> tags = [variant]
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._lf: Any = None

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        raise NotImplementedError(
            "LangfuseTracer 未接入：见 evalharness/observability.py 的 docstring，"
            "以及 docs/architecture.md 第 3 节"
        )
