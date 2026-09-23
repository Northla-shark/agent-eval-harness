# -*- coding: utf-8 -*-
"""Layer 0 · Scenario —— 一次实验的完整定义。

为什么要有这一层：任何阈值（并发数、预算、TTL、限流参数）都不允许硬编码在函数体里。
硬编码的阈值 = 无法做消融 = 无法回答"这个数字是怎么来的"。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Scenario:
    """场景定义。固定种子 + 固定 driver 列表 + 固定预算。

    注意 duration_days / max_steps 这两个名字与 SNAPSHOT_FIELDS 里的
    simulation_duration_days / simulation_max_steps 不同 —— 映射在 to_config_dict()
    里显式写出来，而不是靠名字碰巧一致。名字一旦碰巧一致，将来改一处就会静默失配。
    """

    name: str
    driver_ids: list[str]
    duration_days: int
    max_steps: int
    variant: str = "baseline"
    seed: int = 0

    # ---- 与 SNAPSHOT_FIELDS 对齐的字段 ----
    model_name: str = "none"
    model_api_url: str = ""
    temperature: float = 0.0
    prompt_version: str = "v0"
    adapter_name: str = "dummy"
    max_tokens_per_driver: int | None = None
    max_wall_time_minutes: float | None = None

    # ---- 运行侧参数（不进 run_id）----
    concurrency: int = 4
    rpm: int = 600
    tpm: int = 2_000_000
    token_profile: str = "no_history"
    enforce_limits: bool = True

    raw: dict[str, Any] = field(default_factory=dict)

    def to_config_dict(self) -> dict[str, Any]:
        """扁平化成 SNAPSHOT_FIELDS 认得的形状。

        这是唯一一处"配置 -> 快照"的映射点。多写几个字，换来的是
        "改了哪个字段都能被 diff 抓到"。
        """
        return {
            "model_name": self.model_name,
            "model_api_url": self.model_api_url,
            "temperature": self.temperature,
            "prompt_version": self.prompt_version,
            "adapter_name": self.adapter_name,
            "variant": self.variant,
            "simulation_duration_days": self.duration_days,
            "simulation_max_steps": self.max_steps,
            "max_tokens_per_driver": self.max_tokens_per_driver,
            "max_wall_time_minutes": self.max_wall_time_minutes,
        }

    def with_variant(self, variant: str, *, adapter_name: str | None = None) -> "Scenario":
        """派生一个消融变体。

        用 dataclasses.replace 而不是手改字段：frozen 类型上改字段只能通过派生，
        这保证了"变体"和"基线"共享其他所有参数 —— 消融实验最容易犯的错就是
        在改 A 的同时顺手改了 B。
        """
        from dataclasses import replace

        return replace(
            self,
            variant=variant,
            adapter_name=adapter_name or variant,
        )

    @classmethod
    def from_file(cls, path: Path | str) -> "Scenario":
        """从 YAML 读。需要 pyyaml。"""
        import yaml

        p = Path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        known = {f for f in cls.__dataclass_fields__ if f != "raw"}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["name"] = kwargs.get("name", p.stem)
        kwargs.setdefault("driver_ids", [])
        kwargs.setdefault("duration_days", 30)
        kwargs.setdefault("max_steps", 200)
        return cls(raw=data, **kwargs)

    @classmethod
    def tiny(cls, **overrides: Any) -> "Scenario":
        """一个 5 天 / 3 司机的小场景。开发和自检用，不用于出结论。

        为什么值得专门写一个 tiny()：项目一最贵的一课是
        「1 天基准的罚分信噪比只有 2.6%，30 天基准是 67.2%」——
        短基准只能用来验证链路通不通，绝对不能用来下结论。
        函数名里带 tiny 就是为了让这句话每次都被看见。
        """
        base: dict[str, Any] = dict(
            name="smoke-tiny",
            driver_ids=["d001", "d002", "d003"],
            duration_days=5,
            max_steps=40,
            seed=7,
            model_name="dummy",
            adapter_name="dummy",
            prompt_version="v0",
            max_tokens_per_driver=2_000_000,
            max_wall_time_minutes=10.0,
            concurrency=3,
        )
        base.update(overrides)
        return cls(**base)
