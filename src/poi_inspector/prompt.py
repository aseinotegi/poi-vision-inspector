"""From a POI to a question: the instruction and the JSON schema the model must fill.

The schema is the contract. Providers enforce it where they can (constrained
decoding in Ollama, structured output in Gemini and OpenAI-compatible servers),
and the inspector validates the answer again anyway.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from .config import Config, Poi, Profile

NUMERIC_TEMPLATE = """{instruction}

Point of interest: {tag}. You should find: {description}.
Expected unit: {unit}. Scale of this instrument: {min} to {max} {unit}.

Answer only with the JSON requested:
- value: the reading as a number, or null if you cannot read it reliably.
- unit: the unit you read on the dial.
- status_reason: one of {reasons}.
- confidence: from 0 to 1, how sure you are of the value.
- evidence: one short sentence on what you saw (where the needle points, what hides it).
Never guess a value you cannot see. A null with the right reason is a good answer."""

STATE_TEMPLATE = """{instruction}

Point of interest: {tag}. You should find: {description}.

Answer only with the JSON requested:
- value: one of {values}.
- status_reason: one of {reasons}.
- confidence: from 0 to 1, how sure you are.
- evidence: one short sentence on what you saw.
If the object is not there, that is a finding: answer with the matching reason.
If you cannot assess the state, return value null and status_reason no_legible.
OK must mean correcto; an adverse finding must be NOK."""

GROUNDING = """
Locate the object you inspected. Return bbox_2d as [left, top, right, bottom],
coordinates normalized to 0..1000 relative to the WHOLE input image (x first, y second).
If the target is absent or cannot be localized, return bbox_2d null. Never invent a box.
The point-of-interest tag is supplied context, not evidence that you read a physical label.
"""


def box_schema() -> dict[str, Any]:
    return {
        "anyOf": [
            {"type": "null"},
            {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000}, "minItems": 4, "maxItems": 4},
        ]
    }


@dataclass(frozen=True)
class Question:
    instruction: str
    schema: dict[str, Any]


def numeric_schema(profile: Profile) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "value": {"type": ["number", "null"]},
            "unit": {"type": ["string", "null"]},
            "status_reason": {"type": "string", "enum": list(profile.reasons)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
            "bbox_2d": box_schema(),
        },
        "required": ["value", "unit", "status_reason", "confidence", "evidence", "bbox_2d"],
        "additionalProperties": False,
    }


def state_schema(profile: Profile) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "value": {"type": ["string", "null"], "enum": [*profile.values, None]},
            "status_reason": {"type": "string", "enum": list(profile.reasons)},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "evidence": {"type": "string"},
            "bbox_2d": box_schema(),
        },
        "required": ["value", "status_reason", "confidence", "evidence", "bbox_2d"],
        "additionalProperties": False,
    }


def build_question(poi: Poi, profile: Profile) -> Question:
    reasons = ", ".join(profile.reasons)
    if profile.kind == "numeric":
        assert poi.expect is not None  # guaranteed by config validation
        text = NUMERIC_TEMPLATE.format(
            instruction=profile.instruction.strip(),
            tag=poi.tag,
            description=profile.description,
            unit=poi.expect.unit,
            min=f"{poi.expect.min:g}",
            max=f"{poi.expect.max:g}",
            reasons=reasons,
        )
        return Question(text + GROUNDING, numeric_schema(profile))
    text = STATE_TEMPLATE.format(
        instruction=profile.instruction.strip(),
        tag=poi.tag,
        description=profile.description,
        values=", ".join(profile.values),
        reasons=reasons,
    )
    return Question(text + GROUNDING, state_schema(profile))


def fingerprint(config: Config) -> str:
    """Short hash of everything that shapes the answers: profiles, POIs, templates.

    Stored with every result, so a change of prompt is visible in the history.
    """
    payload = json.dumps(
        {
            "profiles": {k: v.model_dump() for k, v in sorted(config.profiles.items())},
            "pois": {k: v.model_dump() for k, v in sorted(config.pois.items())},
            "templates": [NUMERIC_TEMPLATE, STATE_TEMPLATE, GROUNDING],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]
