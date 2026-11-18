"""Multimodal models behind one interface: image + instruction + schema in, JSON out."""

from __future__ import annotations

from ..config import ProviderConfig
from .base import Answer, Provider, ProviderError


def make_provider(config: ProviderConfig) -> Provider:
    if config.kind == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(config)
    if config.kind == "openai":
        from .openai_compat import OpenAICompatibleProvider

        return OpenAICompatibleProvider(config)
    if config.kind == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(config)
    from .replay import ReplayProvider

    return ReplayProvider(config)


__all__ = ["Answer", "Provider", "ProviderError", "make_provider"]
