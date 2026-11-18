"""Configuration: which model to ask, what to look for, and where.

Three blocks in one YAML file (see ``config/inspector.yaml``):

* ``provider``  which multimodal model answers: a local one (Ollama, any
  OpenAI-compatible server) or a cloud one (Gemini). Swapped without touching code.
* ``profiles``  what to look for and how to answer, per kind of object: a gauge
  returns a number, an extinguisher returns OK/NOK with a reason.
* ``pois``      where the robot stops. Each point of interest says which profile
  applies there, what it should read (unit, range) and where the result goes in
  the Unified Namespace.

Adding a new kind of inspection is a new profile and a new POI. No training.
Everything is validated at start-up, so a typo fails loudly before the first round.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_TOPIC_RE = re.compile(r"^[^/+#\s]+(/[^/+#\s]+)*$")


class ConfigError(ValueError):
    pass


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ProviderConfig(Strict):
    kind: Literal["ollama", "openai", "gemini", "replay"]
    model: str = ""
    url: str | None = None  # Ollama / OpenAI-compatible server
    api_key_env: str | None = None  # name of the variable holding the key, never the key
    timeout_s: float = Field(60.0, gt=0)
    temperature: float = Field(0.0, ge=0, le=2)
    retries: int = Field(2, ge=0, le=10)
    think: bool | None = None  # Ollama: turn reasoning off for models that have it
    answers: str | None = None  # replay: file with recorded answers

    @model_validator(mode="after")
    def _needs(self) -> ProviderConfig:
        if self.kind != "replay" and not self.model:
            raise ValueError(f"provider '{self.kind}' needs a 'model'")
        if self.kind == "replay" and not self.answers:
            raise ValueError("provider 'replay' needs an 'answers' file")
        return self


class Profile(Strict):
    """What to look for and how to answer, for one kind of object."""

    kind: Literal["numeric", "state"]
    description: str  # what the object is, in plant words
    instruction: str  # how to inspect it
    reasons: list[str]  # closed list of status reasons the model may give
    values: list[str] = ["OK", "NOK"]  # state profiles only
    min_confidence: float = Field(0.8, ge=0, le=1)

    @field_validator("reasons", "values")
    @classmethod
    def _keys(cls, items: list[str]) -> list[str]:
        if not items:
            raise ValueError("must not be empty")
        for item in items:
            if not _KEY_RE.match(item.lower()):
                raise ValueError(f"'{item}' must be a short key (letters, digits, '_' and '-')")
        return items

    @model_validator(mode="after")
    def _numeric_reasons(self) -> Profile:
        if self.kind == "numeric" and "lectura_valida" not in self.reasons:
            raise ValueError("numeric profiles need the reason 'lectura_valida'")
        return self


class Expect(Strict):
    """What a numeric POI should read: anything outside the range is not trusted."""

    unit: str
    min: float
    max: float
    max_step: float | None = Field(None, gt=0)  # largest believable change between two rounds

    @model_validator(mode="after")
    def _range(self) -> Expect:
        if self.min >= self.max:
            raise ValueError("expect.min must be below expect.max")
        return self


class Poi(Strict):
    profile: str
    tag: str  # name on the plant (P&ID tag), given to the model as context
    uns_topic: str
    expect: Expect | None = None

    @field_validator("uns_topic")
    @classmethod
    def _topic(cls, topic: str) -> str:
        if not _TOPIC_RE.match(topic):
            raise ValueError(f"'{topic}' is not a valid topic (no wildcards, no empty levels)")
        return topic


class Config(Strict):
    provider: ProviderConfig
    profiles: dict[str, Profile]
    pois: dict[str, Poi]

    @model_validator(mode="after")
    def _references(self) -> Config:
        for poi_id, poi in self.pois.items():
            if poi.profile not in self.profiles:
                raise ValueError(f"POI '{poi_id}' uses unknown profile '{poi.profile}'")
            kind = self.profiles[poi.profile].kind
            if kind == "numeric" and poi.expect is None:
                raise ValueError(f"POI '{poi_id}' reads a number: it needs 'expect' (unit, min, max)")
            if kind == "state" and poi.expect is not None:
                raise ValueError(f"POI '{poi_id}' checks a state: 'expect' does not apply")
        return self


def parse_config(raw: object) -> Config:
    try:
        return Config.model_validate(raw)
    except ValidationError as exc:
        problems = [f"{'.'.join(str(p) for p in e['loc']) or 'config'}: {e['msg']}" for e in exc.errors()]
        raise ConfigError("invalid configuration\n  " + "\n  ".join(problems)) from None


# Deployment details that change per machine, not per plant: where the model runs.
ENV_OVERRIDES = {"POI_PROVIDER_URL": "url", "POI_PROVIDER_MODEL": "model"}


def load_config(path: str | Path, environ: dict[str, str] | None = None) -> Config:
    import os

    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    env = os.environ if environ is None else environ
    if isinstance(raw, dict) and isinstance(raw.get("provider"), dict):
        for variable, key in ENV_OVERRIDES.items():
            if env.get(variable):
                raw["provider"][key] = env[variable]
    return parse_config(raw)
