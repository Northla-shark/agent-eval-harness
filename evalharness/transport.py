# -*- coding: utf-8 -*-
"""模型传输层：把 payload 发出去，把 JSON 拿回来。

为什么平台要自带这一层，而不是让被测 Agent 自己 requests.post
==============================================================
因为**限流、缓存、计量、Span 都要挂在"发出请求"这个动作上**。
只要请求是 Agent 自己发的，平台就只能在事后看到一个总数 ——
拆不出是哪个节点慢、哪一步贵、哪次被限流。

这只用一个标准库（urllib），不引 httpx / requests：
地基层的依赖越少越好，而且这条路径在评测里调用次数以千计，
少一层依赖就少一处版本冲突。
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

DEFAULT_TIMEOUT = 60.0


class TransportError(RuntimeError):
    """HTTP 层出错。**必须带上响应体** —— 429 的响应体里通常写着配额细节，
    只报状态码等于把最有用的信息丢掉。"""


class OpenAICompatibleTransport:
    """OpenAI 兼容的 /chat/completions 客户端。

    适配范围：DashScope 兼容模式、vLLM、Ollama、OpenAI 本体、任何实现了
    `POST {url} {"model":..., "messages":[...]}` 的服务。
    """

    def __init__(
        self,
        url: str,
        api_key: str,
        *,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        if not url:
            raise ValueError("模型 URL 不能为空（Scenario.model_api_url）")
        if not api_key:
            raise ValueError(
                "缺少 API key。设环境变量 DASHSCOPE_API_KEY，或用 --api-key 传。\n"
                "  注意：key 只放在环境变量或 .env 里，**绝不写进仓库**。"
            )
        self.url = url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.calls = 0

    def __call__(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        if self.model and "model" not in body:
            body["model"] = self.model
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        self.calls += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            raise TransportError(f"HTTP {exc.code} {self.url}\n{detail}") from exc
        except urllib.error.URLError as exc:
            raise TransportError(f"连接失败 {self.url}: {exc.reason}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise TransportError(f"响应不是 JSON（前 400 字）：{raw[:400]}") from exc
        if not isinstance(parsed, dict):
            raise TransportError(f"响应不是 JSON 对象：{type(parsed).__name__}")
        return parsed
