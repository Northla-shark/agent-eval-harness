# -*- coding: utf-8 -*-
"""Layer 3 · Metrics —— 三层指标：Outcome / Process / Cost。

为什么需要三层：终局收益任何比赛脚本都能给，它是"结果好"。
Process 层回答的是"决策好"——这两件事在长时程任务里经常不一致：
项目一里 A 组烧 2.87 倍 token 只换来 1/6.5 的利润，
只看终局指标会误判成模型选型问题，拆开才发现是 91.4% 的 wait 率。
"""
from . import cost, outcome, process

__all__ = ["cost", "outcome", "process"]
