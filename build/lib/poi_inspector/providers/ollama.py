"""Local models through Ollama: the image never leaves the plant network.

Uses the native ``/api/chat`` endpoint with ``format`` set to the JSON schema, so
the model is constrained to valid JSON while it generates.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from ..config import ProviderConfig
from .base import RETRYABLE_STATUS, Answer, ProviderError


class OllamaProvider:
    name = "ollama"

    def __init__(self, config: ProviderConfig, client: httpx.Client | None = None) -> None:
        self._cfg = config
        self._url = (config.url or "http://localhost:11434").rstrip("/")
        self._client = client or httpx.Client(timeout=config.timeout_s)

    def ask(self, image: bytes, mime: str, instruction: str, schema: dict[str, Any]) -> Answer:
        body: dict[str, Any] = {
            "model": self._cfg.model,
            "messages": [
                {"role": "user", "content": instruction, "images": [base64.b64encode(image).decode()]},
            ],
            "format": schema,
            "stream": False,
            "options": {"temperature": self._cfg.temperature},
        }
        if self._cfg.think is not None:
            body["think"] = self._cfg.think
        start = time.monotonic()
        try:
            response = self._client.post(f"{self._url}/api/chat", json=body)
        except httpx.TransportError as exc:
            raise ProviderError(f"ollama not reachable at {self._url}: {exc}", retryable=True) from exc
        if response.status_code != 200:
            raise ProviderError(
                f"ollama answered {response.status_code}: {response.text[:200]}",
                retryable=response.status_code in RETRYABLE_STATUS,
            )
        data = response.json()
        return Answer(
            text=data["message"]["content"],
            model=data.get("model", self._cfg.model),
            latency_ms=int((time.monotonic() - start) * 1000),
        )
