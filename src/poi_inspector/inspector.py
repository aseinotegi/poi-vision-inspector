"""The inspection itself: ask the model about one POI and decide how much to trust the answer.

The model's own ``confidence`` is a self-assessment, not a calibrated probability.
Trust comes from checks that do not depend on the model:

* the answer must follow the schema and the profile's closed lists;
* a number must be in the instrument's range and in the expected unit;
* a number must not jump more than ``max_step`` from the last trusted reading;
* low self-reported confidence is one more reason to doubt, never a reason to trust.

Every result says why it was trusted or not, and carries the trace needed to audit
it later: provider, model, prompt fingerprint, image hash, latency.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from jsonschema import Draft202012Validator

from .config import Config, Poi, Profile
from .prompt import build_question, fingerprint
from .providers import Provider, ProviderError

log = logging.getLogger(__name__)

GOOD, UNCERTAIN, REJECTED = "good", "uncertain", "rejected"
UNIT_ALIASES = {"bar": {"bar", "bars", "bar(g)", "barg"}, "°c": {"°c", "ºc", "c", "degc"}}


@dataclass
class Inspection:
    poi: str
    tag: str
    kind: str  # numeric | state
    value: Any  # number, state, or None
    unit: str | None
    status_reason: str
    confidence: float | None
    evidence: str
    quality: str  # good | uncertain | rejected
    checks: list[str] = field(default_factory=list)  # why it is not "good"
    trace: dict[str, Any] = field(default_factory=dict)
    bbox_2d: list[float] | None = None
    bbox_source: str | None = None  # model | replay | annotated; never inferred from coordinates alone
    other_boxes: list[list[float]] = field(default_factory=list)  # same kind, seen but not inspected

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InspectionFailed(Exception):
    """No answer at all from the model, after retries. Nothing was read."""


class Inspector:
    def __init__(
        self,
        config: Config,
        provider: Provider,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.provider = provider
        self.prompt_fingerprint = fingerprint(config)
        self._sleep = sleep
        self._last_good: dict[str, float] = {}

    def forget(self) -> None:
        """Drops the reading history (used when photos are unrelated, as in an evaluation)."""
        self._last_good.clear()

    def inspect(self, poi_id: str, image: bytes, mime: str = "image/jpeg") -> Inspection:
        poi = self.config.pois.get(poi_id)
        if poi is None:
            raise KeyError(f"unknown POI '{poi_id}'")
        profile = self.config.profiles[poi.profile]
        question = build_question(poi, profile)
        answer = self._ask_with_retry(image, mime, question.instruction, question.schema)

        trace = {
            "provider": self.provider.name,
            "model": answer.model,
            "prompt": self.prompt_fingerprint,
            "image_sha256": hashlib.sha256(image).hexdigest(),
            "latency_ms": answer.latency_ms,
            "raw": answer.text,
            "instruction": question.instruction,
            "schema": question.schema,
            "temperature": self.config.provider.temperature,
        }
        inspection = self._judge(poi_id, poi, profile, answer.text)
        inspection.trace = trace
        inspection.other_boxes = _other_boxes(answer.text, inspection.bbox_2d)
        if inspection.bbox_2d is not None:
            inspection.bbox_source = "replay" if self.provider.name == "replay" else "model"
        if inspection.quality == GOOD and profile.kind == "numeric":
            self._last_good[poi_id] = float(inspection.value)
        return inspection

    # -- asking ---------------------------------------------------------------

    def _ask_with_retry(self, image: bytes, mime: str, instruction: str, schema: dict):
        attempts = self.config.provider.retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self.provider.ask(image, mime, instruction, schema)
            except ProviderError as exc:
                if not exc.retryable or attempt == attempts:
                    raise InspectionFailed(f"{exc} (attempt {attempt}/{attempts})") from exc
                delay = 2 ** (attempt - 1)
                log.warning("model call failed (%s), retrying in %ss", exc, delay)
                self._sleep(delay)
        raise InspectionFailed("retry loop exhausted")  # pragma: no cover

    # -- judging --------------------------------------------------------------

    def _judge(self, poi_id: str, poi: Poi, profile: Profile, text: str) -> Inspection:
        base: dict[str, Any] = {"poi": poi_id, "tag": poi.tag, "kind": profile.kind}
        try:
            data = json.loads(text, parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
            if not isinstance(data, dict):
                raise ValueError("not an object")
            # Old recorded answers have no localization. Null never implies a predicted box.
            data.setdefault("bbox_2d", None)
            data.setdefault("other_boxes", [])
            errors = list(Draft202012Validator(build_question(poi, profile).schema).iter_errors(data))
            if errors:
                return Inspection(
                    **base,
                    value=None,
                    unit=None,
                    status_reason="respuesta_invalida",
                    confidence=None,
                    evidence="",
                    quality=REJECTED,
                    checks=[f"schema: {e.message}" for e in errors[:5]],
                )
        except ValueError:
            return Inspection(
                **base,
                value=None,
                unit=None,
                status_reason="respuesta_invalida",
                confidence=None,
                evidence=text[:200],
                quality=REJECTED,
                checks=["the answer is not valid JSON"],
            )

        reason = str(data.get("status_reason", "")).strip().lower()
        confidence = _as_float(data.get("confidence"))
        evidence = str(data.get("evidence") or "")[:300]
        checks: list[str] = []
        box = data.get("bbox_2d")
        if box and not (box[0] < box[2] and box[1] < box[3]):
            checks.append("invalid bounding box: inverted or empty coordinates")
            box = None
        if not evidence.strip():
            checks.append("missing visual evidence")

        if reason not in profile.reasons:
            checks.append(f"reason '{reason}' is not in the profile")
        if confidence is None or not 0 <= confidence <= 1:
            checks.append("missing or invalid confidence")
            confidence = None
        elif confidence < profile.min_confidence:
            checks.append(f"model confidence {confidence:.2f} below {profile.min_confidence:.2f}")

        if profile.kind == "state":
            if data.get("value") is None or reason == "no_legible":
                return Inspection(
                    **base,
                    value=None,
                    unit=None,
                    status_reason=reason,
                    confidence=confidence,
                    evidence=evidence,
                    quality=UNCERTAIN,
                    checks=checks + ([] if data.get("value") is None else ["a state with reason 'no_legible'"]),
                    bbox_2d=box,
                )
            value = str(data.get("value", "")).strip().upper()
            if (value == "OK" and reason != "correcto") or (value == "NOK" and reason == "correcto"):
                checks.append("state and reason contradict each other")
            if value not in profile.values:
                return Inspection(
                    **base,
                    value=None,
                    unit=None,
                    status_reason=reason or "respuesta_invalida",
                    confidence=confidence,
                    evidence=evidence,
                    quality=REJECTED,
                    checks=checks + [f"state '{value}' is not one of {profile.values}"],
                )
            return Inspection(
                **base,
                value=value,
                unit=None,
                status_reason=reason,
                confidence=confidence,
                evidence=evidence,
                quality=UNCERTAIN if checks else GOOD,
                checks=checks,
                bbox_2d=box,
            )

        result = self._judge_number(base, poi, reason, confidence, evidence, data, checks)
        result.bbox_2d = box
        return result

    def _judge_number(
        self,
        base: dict[str, Any],
        poi: Poi,
        reason: str,
        confidence: float | None,
        evidence: str,
        data: dict[str, Any],
        checks: list[str],
    ) -> Inspection:
        expect = poi.expect
        assert expect is not None  # numeric POIs always have one (config validation)
        value = _as_float(data.get("value"))
        unit = data.get("unit")

        if value is None:
            # "could not read" is a legitimate answer, as long as it says why
            if reason == "lectura_valida":
                checks.append("'lectura_valida' without a value")
            return Inspection(
                **base,
                value=None,
                unit=expect.unit,
                status_reason=reason,
                confidence=confidence,
                evidence=evidence,
                quality=UNCERTAIN,
                checks=checks,
            )

        if reason != "lectura_valida":
            checks.append(f"a value with reason '{reason}'")
        if not unit:
            checks.append("missing unit")
        elif not _same_unit(str(unit), expect.unit):
            checks.append(f"unit '{unit}' instead of '{expect.unit}'")
        margin = 0.05 * (expect.max - expect.min)
        if not expect.min - margin <= value <= expect.max + margin:
            return Inspection(
                **base,
                value=value,
                unit=expect.unit,
                status_reason="fuera_de_rango",
                confidence=confidence,
                evidence=evidence,
                quality=REJECTED,
                checks=checks + [f"{value:g} outside the scale {expect.min:g}-{expect.max:g}"],
            )
        last = self._last_good.get(base["poi"])
        if expect.max_step is not None and last is not None and abs(value - last) > expect.max_step:
            checks.append(f"jump of {value - last:+.2f} {expect.unit} since the last trusted reading")

        return Inspection(
            **base,
            value=round(value, 3),
            unit=expect.unit,
            status_reason=reason,
            confidence=confidence,
            evidence=evidence,
            quality=UNCERTAIN if checks else GOOD,
            checks=checks,
        )


def _other_boxes(text: str, box: list[float] | None) -> list[list[float]]:
    """The other objects of the same kind the model saw, when it also located the inspected one."""
    if box is None:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    others = data.get("other_boxes") if isinstance(data, dict) else None
    if not isinstance(others, list):
        return []
    valid = [
        o
        for o in others
        if isinstance(o, list)
        and len(o) == 4
        and all(isinstance(v, int | float) and 0 <= v <= 1000 for v in o)
        and o[0] < o[2]
        and o[1] < o[3]
        and o != box
    ]
    return valid[:8]


def _as_float(raw: Any) -> float | None:
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        value = float(raw)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def _same_unit(read: str, expected: str) -> bool:
    read, expected = read.strip().lower(), expected.strip().lower()
    return read == expected or read in UNIT_ALIASES.get(expected, set())
