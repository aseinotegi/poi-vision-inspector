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


def test_gemini_schema_has_no_null_type_for_the_real_questions():
    state = {"type": ["string", "null"], "enum": ["OK", "NOK", None]}
    assert _gemini_schema(state) == {"type": "string", "nullable": True, "enum": ["OK", "NOK"]}
    box = {"anyOf": [{"type": "null"}, {"type": "array", "items": {"type": "number"}, "minItems": 4}]}
    assert _gemini_schema(box) == {"type": "array", "items": {"type": "number"}, "minItems": 4, "nullable": True}
    # Nothing in the converted question schemas still says "null".
    from poi_inspector.config import Expect, Poi, Profile
    from poi_inspector.prompt import build_question

    numeric = Profile(kind="numeric", description="d", instruction="i", reasons=["lectura_valida"])
    poi = Poi(tag="PI-1", profile="m", uns_topic="a/b", expect=Expect(unit="bar", min=0, max=10))
    assert '"null"' not in json.dumps(_gemini_schema(build_question(poi, numeric).schema))


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


class FakeGemini:
    """Stands in for google.genai.Client: keeps the request, gives a fixed answer."""

    def __init__(self, answer: str):
        self.answer, self.seen = answer, {}
        self.models = self

    def generate_content(self, model, contents, config):
        self.seen = {"instruction": contents[1], "schema": config.response_schema}
        return type("Response", (), {"text": self.answer, "model_version": model})()


def test_gemini_is_asked_for_its_own_box_order_and_answers_in_ours():
    from poi_inspector.config import Expect, Poi, Profile
    from poi_inspector.prompt import build_question
    from poi_inspector.providers.gemini import GeminiProvider

    profile = Profile(kind="numeric", description="a gauge", instruction="Read it.", reasons=["lectura_valida"])
    poi = Poi(profile="m", tag="PI-1", uns_topic="a/b", expect=Expect(unit="bar", min=0, max=10))
    question = build_question(poi, profile)
    fake = FakeGemini('{"value": 5.4, "box_2d": [304, 438, 917, 877]}')
    provider = GeminiProvider(ProviderConfig(kind="gemini", model="gemini-x"), client=fake)

    answer = provider.ask(IMG, "image/jpeg", question.instruction, question.schema)

    assert "[ymin, xmin, ymax, xmax]" in fake.seen["instruction"]
    assert "bbox_2d" not in fake.seen["instruction"]
    assert "box_2d" in fake.seen["schema"]["properties"] and "bbox_2d" not in fake.seen["schema"]["properties"]
    assert json.loads(answer.text) == {"value": 5.4, "bbox_2d": [438, 304, 877, 917]}


def test_gemini_answer_without_a_box_is_left_alone():
    from poi_inspector.providers.gemini import _box_back

    assert json.loads(_box_back('{"value": null, "box_2d": null}')) == {"value": None, "bbox_2d": None}
    assert _box_back("not json") == "not json"
