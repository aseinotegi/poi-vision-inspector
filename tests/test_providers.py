"""Each provider builds the request its server expects and maps errors the same way."""

import base64
import json

import httpx
import pytest

from poi_inspector.config import ProviderConfig
from poi_inspector.providers import ProviderError, make_provider
from poi_inspector.providers.gemini import _gemini_schema
from poi_inspector.providers.ollama import OllamaProvider
from poi_inspector.providers.openai_compat import OpenAICompatibleProvider
from poi_inspector.providers.replay import ReplayProvider

IMG = b"\x89PNG fake"
SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": ["number", "null"]}, "status_reason": {"type": "string", "enum": ["a"]}},
    "required": ["value", "status_reason"],
    "additionalProperties": False,
}


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_ollama_sends_image_and_schema_and_reads_the_answer():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content), url=str(request.url))
        return httpx.Response(200, json={"model": "qwen3-vl:8b", "message": {"content": '{"value": 6.8}'}})

    cfg = ProviderConfig(kind="ollama", model="qwen3-vl", url="http://gpu-box:11434", think=False)
    answer = OllamaProvider(cfg, client(handler)).ask(IMG, "image/png", "read it", SCHEMA)
    assert seen["url"] == "http://gpu-box:11434/api/chat"
    assert seen["format"] == SCHEMA and seen["stream"] is False and seen["think"] is False
    assert base64.b64decode(seen["messages"][0]["images"][0]) == IMG
    assert answer.text == '{"value": 6.8}' and answer.model == "qwen3-vl:8b"


def test_ollama_without_think_does_not_send_it():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    OllamaProvider(ProviderConfig(kind="ollama", model="m"), client(handler)).ask(IMG, "image/png", "x", SCHEMA)
    assert "think" not in seen


@pytest.mark.parametrize("status, retryable", [(503, True), (429, True), (404, False), (400, False)])
def test_http_errors_say_whether_to_retry(status, retryable):
    provider = OllamaProvider(ProviderConfig(kind="ollama", model="m"), client(lambda r: httpx.Response(status)))
    with pytest.raises(ProviderError) as err:
        provider.ask(IMG, "image/png", "x", SCHEMA)
    assert err.value.retryable is retryable


def test_unreachable_server_is_retryable():
    def handler(request):
        raise httpx.ConnectError("refused")

    provider = OllamaProvider(ProviderConfig(kind="ollama", model="m"), client(handler))
    with pytest.raises(ProviderError) as err:
        provider.ask(IMG, "image/png", "x", SCHEMA)
    assert err.value.retryable


def test_openai_compatible_sends_a_data_url_and_json_schema(monkeypatch):
    monkeypatch.setenv("VLLM_KEY", "secret")
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content), auth=request.headers.get("authorization"))
        return httpx.Response(200, json={"model": "qwen-vl", "choices": [{"message": {"content": "{}"}}]})

    cfg = ProviderConfig(kind="openai", model="qwen-vl", url="http://vllm:8000/v1", api_key_env="VLLM_KEY")
    OpenAICompatibleProvider(cfg, client(handler)).ask(IMG, "image/png", "read it", SCHEMA)
    content = seen["messages"][0]["content"]
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["response_format"]["json_schema"]["schema"] == SCHEMA
    assert seen["auth"] == "Bearer secret"


def test_gemini_schema_dialect():
    converted = _gemini_schema(SCHEMA)
    assert converted["properties"]["value"] == {"type": "number", "nullable": True}
    assert "additionalProperties" not in converted
    assert converted["properties"]["status_reason"]["enum"] == ["a"]


def test_gemini_needs_its_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        make_provider(ProviderConfig(kind="gemini", model="gemini-2.5-flash", api_key_env="GEMINI_API_KEY"))


def test_replay_answers_by_image_hash_with_fallback():
    import hashlib

    digest = hashlib.sha256(IMG).hexdigest()
    provider = ReplayProvider(
        ProviderConfig(kind="replay", answers="x.json"), answers={digest: {"value": 1}, "*": {"value": 0}}
    )
    assert json.loads(provider.ask(IMG, "image/png", "", {}).text) == {"value": 1}
    assert json.loads(provider.ask(b"other", "image/png", "", {}).text) == {"value": 0}
