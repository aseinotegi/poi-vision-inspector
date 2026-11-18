from pathlib import Path

import pytest

from poi_inspector.config import load_config

CONFIG = Path(__file__).resolve().parents[1] / "config" / "inspector.yaml"


@pytest.fixture
def config():
    return load_config(CONFIG)


class ScriptedProvider:
    """Answers with whatever the test scripts, and records what it was asked."""

    name = "scripted"

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def ask(self, image, mime, instruction, schema):
        from poi_inspector.providers import Answer, ProviderError

        self.calls.append({"image": image, "instruction": instruction, "schema": schema})
        answer = self.answers.pop(0)
        if isinstance(answer, ProviderError):
            raise answer
        import json

        return Answer(text=answer if isinstance(answer, str) else json.dumps(answer), model="test-vlm", latency_ms=42)


@pytest.fixture
def scripted():
    return ScriptedProvider
