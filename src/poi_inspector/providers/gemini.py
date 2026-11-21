"""Google Gemini (cloud). Optional: ``pip install poi-vision-inspector[gemini]``."""

from __future__ import annotations

import json
import time
from typing import Any

from ..config import ProviderConfig
from ..prompt import BOX_ORDER
from .base import RETRYABLE_STATUS, Answer, ProviderError, api_key

# Gemini is trained to give boxes as box_2d [ymin, xmin, ymax, xmax]. Asked for another
# order it mixes both from one answer to the next, so it is asked in its own and the
# answer is turned back into bbox_2d [left, top, right, bottom].
NATIVE_BOX_ORDER = """Return box_2d as [ymin, xmin, ymax, xmax],
coordinates normalized to 0..1000 relative to the WHOLE input image (y first, x second)."""


def _native_question(instruction: str, schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if BOX_ORDER not in instruction or "bbox_2d" not in schema.get("properties", {}):
        return instruction, schema
    text = instruction.replace(BOX_ORDER, NATIVE_BOX_ORDER).replace("bbox_2d", "box_2d")
    properties = {("box_2d" if k == "bbox_2d" else k): v for k, v in schema["properties"].items()}
    required = ["box_2d" if k == "bbox_2d" else k for k in schema.get("required", [])]
    return text, {**schema, "properties": properties, "required": required}


def _box_back(text: str) -> str:
    """box_2d [ymin, xmin, ymax, xmax] -> bbox_2d [left, top, right, bottom]."""
    try:
        data = json.loads(text)
    except ValueError:
        return text  # the inspector says it is not valid JSON
    if not isinstance(data, dict) or "box_2d" not in data:
        return text
    box = data.pop("box_2d")
    if isinstance(box, list) and len(box) == 4:
        box = [box[1], box[0], box[3], box[2]]
    data["bbox_2d"] = box
    return json.dumps(data, ensure_ascii=False)


def _gemini_schema(node: Any) -> Any:
    """Gemini's schema dialect: there is no ``null`` type, a nullable field says
    ``nullable``; no ``additionalProperties``."""
    if isinstance(node, list):
        return [_gemini_schema(item) for item in node]
    if not isinstance(node, dict):
        return node
    # {"anyOf": [{"type": "null"}, X]} is X, nullable.
    options = node.get("anyOf")
    if isinstance(options, list) and {"type": "null"} in options:
        rest = [o for o in options if o != {"type": "null"}]
        if len(rest) == 1 and isinstance(rest[0], dict):
            merged = {**{k: v for k, v in node.items() if k != "anyOf"}, **rest[0]}
            return {**_gemini_schema(merged), "nullable": True}
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "additionalProperties":
            continue
        if key == "type" and isinstance(value, list):
            out["type"] = next(t for t in value if t != "null")
            if "null" in value:
                out["nullable"] = True
        elif key == "enum" and isinstance(value, list):
            out[key] = [v for v in value if v is not None]  # null is said by nullable
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

        instruction, schema = _native_question(instruction, schema)
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
            text=_box_back(response.text or ""),
            model=getattr(response, "model_version", None) or self._cfg.model,
            latency_ms=int((time.monotonic() - start) * 1000),
        )
