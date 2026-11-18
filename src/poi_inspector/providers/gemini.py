"""Google Gemini (cloud). Optional: ``pip install poi-vision-inspector[gemini]``."""

from __future__ import annotations

import time
from typing import Any

from ..config import ProviderConfig
from .base import RETRYABLE_STATUS, Answer, ProviderError, api_key


def _gemini_schema(node: Any) -> Any:
    """Gemini's schema dialect: nullable types use ``nullable``, no ``additionalProperties``."""
    if isinstance(node, list):
        return [_gemini_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "additionalProperties":
            continue
        if key == "type" and isinstance(value, list):
            out["type"] = next(t for t in value if t != "null")
            if "null" in value:
                out["nullable"] = True
        elif key == "properties":
            out[key] = {name: _gemini_schema(sub) for name, sub in value.items()}
        else:
            out[key] = _gemini_schema(value)
    return out


class GeminiProvider:
    name = "gemini"

    def __init__(self, config: ProviderConfig, client: Any = None) -> None:
        self._cfg = config
        if client is None:
            try:
                from google import genai
            except ImportError as exc:
                raise ProviderError("install the extra: pip install 'poi-vision-inspector[gemini]'", False) from exc
            client = genai.Client(api_key=api_key(config, required=True))
        self._client = client

    def ask(self, image: bytes, mime: str, instruction: str, schema: dict[str, Any]) -> Answer:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            temperature=self._cfg.temperature,
            response_mime_type="application/json",
            response_schema=_gemini_schema(schema),
        )
        contents = [types.Part.from_bytes(data=image, mime_type=mime), instruction]
        start = time.monotonic()
        try:
            response = self._client.models.generate_content(model=self._cfg.model, contents=contents, config=config)
        except errors.APIError as exc:
            raise ProviderError(f"gemini answered {exc.code}: {exc}", retryable=exc.code in RETRYABLE_STATUS) from exc
        except OSError as exc:
            raise ProviderError(f"gemini not reachable: {exc}", retryable=True) from exc
        return Answer(
            text=response.text or "",
            model=getattr(response, "model_version", None) or self._cfg.model,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
