"""Any OpenAI-compatible server: vLLM, LM Studio, llama.cpp server... or OpenAI itself.

Sends the image as a data URL and asks for ``response_format: json_schema``.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from ..config import ProviderConfig
from .base import RETRYABLE_STATUS, Answer, ProviderError, api_key


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, config: ProviderConfig, client: httpx.Client | None = None) -> None:
        self._cfg = config
        self._url = (config.url or "https://api.openai.com/v1").rstrip("/")
        self._client = client or httpx.Client(timeout=config.timeout_s)

    def ask(self, image: bytes, mime: str, instruction: str, schema: dict[str, Any]) -> Answer:
        headers = {}
        key = api_key(self._cfg, required=False)
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data_url = f"data:{mime};base64,{base64.b64encode(image).decode()}"
        body = {
            "model": self._cfg.model,
            "temperature": self._cfg.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "inspection", "schema": schema, "strict": True},
            },
        }
        start = time.monotonic()
        try:
            response = self._client.post(f"{self._url}/chat/completions", json=body, headers=headers)
        except httpx.TransportError as exc:
            raise ProviderError(f"server not reachable at {self._url}: {exc}", retryable=True) from exc
        if response.status_code != 200:
            raise ProviderError(
                f"server answered {response.status_code}: {response.text[:200]}",
                retryable=response.status_code in RETRYABLE_STATUS,
            )
        data = response.json()
        return Answer(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self._cfg.model),
            latency_ms=int((time.monotonic() - start) * 1000),
        )
