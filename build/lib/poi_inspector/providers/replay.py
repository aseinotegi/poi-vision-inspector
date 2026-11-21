"""Recorded answers, keyed by image hash: tests, CI and demos without a model.

The answers file is JSON: ``{"<sha256 of the image>": {...answer...}, "*": {...}}``.
``*`` is the fallback. Nothing is invented: what was recorded is what comes back.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config import ProviderConfig
from .base import Answer, ProviderError


class ReplayProvider:
    name = "replay"

    def __init__(self, config: ProviderConfig, answers: dict[str, Any] | None = None) -> None:
        if answers is None:
            answers = json.loads(Path(config.answers or "").read_text(encoding="utf-8"))
        self._answers = answers

    def ask(self, image: bytes, mime: str, instruction: str, schema: dict[str, Any]) -> Answer:
        digest = hashlib.sha256(image).hexdigest()
        answer = self._answers.get(digest, self._answers.get("*"))
        if answer is None:
            raise ProviderError(f"no recorded answer for image {digest[:12]}", retryable=False)
        text = answer if isinstance(answer, str) else json.dumps(answer)
        return Answer(text=text, model="replay", latency_ms=0)
