# -*- coding: utf-8 -*-
"""adapter 实现集合。

import 这一句就完成注册 —— 这样 Scenario 里写 adapter_name: "dummy"
就能把它找出来，换被测对象是配置改动而不是代码改动。
"""
from . import dummy, truck  # noqa: F401  （import 即注册）
from .dummy import DummyAdapter  # noqa: F401

__all__ = ["dummy", "truck", "DummyAdapter"]
