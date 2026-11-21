"""The contract every provider fulfils."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import ProviderConfig

# HTTP statuses worth retrying: rate limits, timeouts, server errors.
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class ProviderError(Exception):
    """The model gave no answer (network, quota, server). Not the same as "could not read"."""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass(frozen=True)
class Answer:
    text: str  # the JSON as returned by the model
    model: str  # model actually used, as reported by the provider when it says
    latency_ms: int


class Provider(Protocol):
    name: str

    def ask(self, image: bytes, mime: str, instruction: str, schema: dict[str, Any]) -> Answer: ...


def api_key(config: ProviderConfig, required: bool) -> str | None:
    if not config.api_key_env:
        if required:
            raise ProviderError(f"{config.kind}: set 'api_key_env' to the variable holding the key", False)
        return None
    key = os.environ.get(config.api_key_env)
    if required and not key:
        raise ProviderError(f"{config.kind}: environment variable {config.api_key_env} is empty", False)
    return key
