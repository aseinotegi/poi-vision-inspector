"""MQTT service: the robot publishes a photo of a POI, the inspector publishes what it saw.

Listens on ``PHOTO_TOPIC`` (default ``planta1/robots/+/foto``) for messages like::

    {"value": "IMG_0412", "poi": "PI-104", "image_url": "http://robot-01/photos/IMG_0412.jpg"}

(``image_b64`` instead of ``image_url`` also works). For each photo it publishes:

* ``<uns_topic>`` of the POI, retained: the value, if there is one worth publishing
  (``good`` or ``uncertain``; a rejected reading never overwrites the last value).
* ``<uns_topic>/inspeccion``, retained: the whole inspection, with checks and trace.

Photos are inspected one at a time, in order: a local model is a shared resource,
and a queue keeps the latency predictable instead of piling up requests.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import queue
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import Poi, load_config
from .inspector import REJECTED, Inspection, InspectionFailed, Inspector
from .prompt import build_question
from .providers import make_provider

log = logging.getLogger("inspector")


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now_iso() -> str:
    return _iso(datetime.now(timezone.utc))


def parse_time(raw: Any) -> datetime:
    if not isinstance(raw, str):
        raise ValueError("capture timestamp must be an ISO-8601 string")
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"invalid capture timestamp '{raw}'") from None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _execution_id(poi_id: str, image_id: str, captured_at: str, received_at: str) -> str:
    identity = "\0".join((poi_id, image_id, captured_at, received_at))
    return hashlib.sha256(identity.encode()).hexdigest()[:24]


def value_message(result: Inspection, timing: dict[str, Any]) -> dict[str, Any] | None:
    """The UNS message for the POI's value, or None if nothing should be published."""
    if result.value is None or result.quality == REJECTED:
        return None
    message = {
        "value": result.value,
        "ts": timing["captured_at"],
        "source": "inspector",
        "quality": result.quality,
        "status_reason": result.status_reason,
        "confidence": result.confidence,
        "model": result.trace.get("model"),
        **timing,
    }
    if result.unit:
        message["unit"] = result.unit
    return message


def inspection_message(result: Inspection, image_id: str, timing: dict[str, Any]) -> dict[str, Any]:
    body = result.to_dict()
    body.update(
        {
            "value": result.value,
            "ts": timing["processed_at"],
            "source": "inspector",
            "image": image_id,
            **timing,
        }
    )
    return body


def load_image(photo: dict[str, Any], client: httpx.Client) -> tuple[bytes, str]:
    if "image_b64" in photo:
        return base64.b64decode(photo["image_b64"]), photo.get("mime", "image/jpeg")
    if "image_url" in photo:
        response = client.get(photo["image_url"])
        response.raise_for_status()
        return response.content, response.headers.get("content-type", "image/jpeg").split(";")[0]
    raise ValueError("photo message without 'image_url' or 'image_b64'")


class InspectionService:
    def __init__(
        self,
        inspector: Inspector,
        publish,
        client: httpx.Client | None = None,
        max_photo_age_s: float = 45.0,
        clock: Callable[[], datetime] | None = None,
        trial_prefix: str = "planta1/ensayos/vision",
    ) -> None:
        self._inspector = inspector
        self._publish = publish  # (topic, payload dict, retain) -> None
        self._client = client or httpx.Client(timeout=15)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=100)
        self._max_photo_age_s = max_photo_age_s
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._trial_prefix = trial_prefix.rstrip("/")

    def _enqueue(self, payload: bytes | bytearray, trial: bool) -> None:
        try:
            photo = json.loads(payload)
        except ValueError:
            log.warning("ignoring a photo message that is not JSON")
            return
        if not isinstance(photo, dict):
            log.warning("ignoring a photo message that is not a JSON object")
            return
        poi = photo.get("poi") or photo.get("point")
        if poi not in self._inspector.config.pois:
            log.debug("photo of %r: not a configured POI", poi)
            return
        request = str(photo.get("request") or "")
        if trial and not request:
            log.warning("ignoring a vision trial without a request id")
            return
        received_at = _iso(self._clock())
        item = {**photo, "poi": poi, "received_at": received_at, "trial": trial}
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            log.warning("inspection queue full, dropping photo %s", photo.get("value"))
            if trial:
                self._publish(
                    f"{self._trial_prefix}/resultado",
                    {
                        "value": None,
                        "state": "error",
                        "quality": "error",
                        "status_reason": "cola_llena",
                        "error": "inspection queue full",
                        "request": request,
                        "mode": "ensayo",
                        "poi": poi,
                        "image": str(photo.get("value", "")),
                        "ts": received_at,
                        "source": "inspector",
                    },
                    False,
                )
                return
        if trial:
            self._publish(
                f"{self._trial_prefix}/estado",
                {
                    "value": "en-cola",
                    "state": "en-cola",
                    "request": request,
                    "mode": "ensayo",
                    "poi": poi,
                    "image": str(photo.get("value", "")),
                    "ts": received_at,
                    "source": "inspector",
                },
                False,
            )

    def on_photo(self, payload: bytes | bytearray) -> None:
        self._enqueue(payload, trial=False)

    def on_trial(self, payload: bytes | bytearray) -> None:
        self._enqueue(payload, trial=True)

    def _failure(self, poi_id: str, poi: Poi, context: dict[str, Any], error: str, quality: str = "error") -> None:
        reason = "captura_caducada" if quality == "stale" else "error_inspeccion"
        trial = context.get("mode") == "ensayo"
        topic = f"{self._trial_prefix}/resultado" if trial else f"{poi.uns_topic}/inspeccion"
        body = {
            "poi": poi_id,
            "tag": poi.tag,
            "kind": self._inspector.config.profiles[poi.profile].kind,
            "value": None,
            "ts": context["processed_at"],
            "source": "inspector",
            "quality": quality,
            "status_reason": reason,
            "checks": [error],
            "error": error[:300],
            **context,
        }
        if trial:
            body["state"] = "error"
        self._publish(topic, body, not trial)

    def handle(self, photo: dict[str, Any]) -> None:
        poi_id = photo["poi"]
        poi = self._inspector.config.pois[poi_id]
        image_id = str(photo.get("value", ""))
        trial = photo.get("trial") is True
        processed_at = _iso(self._clock())
        received_at = str(photo.get("received_at") or processed_at)
        captured_at = str(photo.get("captured_at") or photo.get("ts") or received_at)
        context: dict[str, Any] = {
            "inspection_id": _execution_id(poi_id, image_id, captured_at, received_at),
            "image": image_id,
            "captured_at": captured_at,
            "received_at": received_at,
            "processed_at": processed_at,
        }
        if photo.get("image_file"):
            context["image_file"] = str(photo["image_file"])
        if trial:
            context.update({"request": str(photo["request"]), "mode": "ensayo"})
        try:
            captured = parse_time(captured_at)
            received = parse_time(received_at)
            processed = parse_time(processed_at)
            age_s = (processed - captured).total_seconds()
            if age_s < -5:
                raise ValueError("capture timestamp is more than 5 seconds in the future")
            context.update(
                {
                    "captured_at": _iso(captured),
                    "received_at": _iso(received),
                    "queue_wait_ms": max(0, round((processed - received).total_seconds() * 1000)),
                    "capture_age_ms": max(0, round(age_s * 1000)),
                }
            )
            if self._max_photo_age_s > 0 and age_s > self._max_photo_age_s:
                self._failure(
                    poi_id, poi, context, f"stale capture: {age_s:.1f}s old, limit {self._max_photo_age_s:g}s", "stale"
                )
                log.warning("%s %s: discarded stale capture (%.1fs old)", poi_id, image_id, age_s)
                return
            if trial:
                self._publish(
                    f"{self._trial_prefix}/estado",
                    {
                        "value": "analizando",
                        "state": "analizando",
                        "request": context["request"],
                        "mode": "ensayo",
                        "poi": poi_id,
                        "image": image_id,
                        "inspection_id": context["inspection_id"],
                        "ts": processed_at,
                        "source": "inspector",
                    },
                    False,
                )
            image, mime = load_image(photo, self._client)
            result = self._inspector.inspect(poi_id, image, mime)
        except (InspectionFailed, httpx.HTTPError, ValueError) as exc:
            log.error("%s %s: no inspection (%s)", poi_id, image_id, exc)
            self._failure(poi_id, poi, context, str(exc))
            return
        result.trace.update(context)
        body = inspection_message(result, image_id, context)
        if trial:
            body["state"] = "completada"
            self._publish(f"{self._trial_prefix}/resultado", body, False)
        else:
            self._publish(f"{poi.uns_topic}/inspeccion", body, True)
            message = value_message(result, context)
            if message:
                self._publish(poi.uns_topic, message, True)
        log.info(
            "%s %s → %s %s (%s, %s ms%s)",
            poi_id,
            image_id,
            result.value,
            result.unit or "",
            result.quality,
            result.trace.get("latency_ms"),
            ", trial" if trial else "",
        )

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                photo = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.handle(photo)


def describe(config, inspector: Inspector) -> dict[str, Any]:
    """The inspector's configuration as a UNS message: model, and for every point of
    interest the exact prompt and JSON schema the model gets. No secrets: no keys, no URLs."""
    pois = {}
    for poi_id, poi in sorted(config.pois.items()):
        profile = config.profiles[poi.profile]
        question = build_question(poi, profile)
        pois[poi_id] = {
            "profile": poi.profile,
            "kind": profile.kind,
            "tag": poi.tag,
            "uns_topic": poi.uns_topic,
            "expect": poi.expect.model_dump() if poi.expect else None,
            "min_confidence": profile.min_confidence,
            "prompt": question.instruction,
            "schema": question.schema,
        }
    return {
        "value": inspector.prompt_fingerprint,
        "source": "inspector",
        "provider": config.provider.kind,
        "model": config.provider.model or config.provider.kind,
        "temperature": config.provider.temperature,
        "retries": config.provider.retries,
        "timeout_s": config.provider.timeout_s,
        "think": config.provider.think,
        "profiles": {k: {"kind": v.kind, "description": v.description} for k, v in sorted(config.profiles.items())},
        "pois": pois,
    }


def main(config_path: str) -> int:
    import signal

    import paho.mqtt.client as mqtt

    config = load_config(config_path)
    inspector = Inspector(config, make_provider(config.provider))
    photo_topic = os.environ.get("PHOTO_TOPIC", "planta1/robots/+/foto")
    status_topic = os.environ.get("STATUS_TOPIC", "planta1/servicios/inspector")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="poi-inspector")
    offline = json.dumps({"value": "offline", "source": "inspector"})
    client.will_set(status_topic, offline, qos=1, retain=True)

    def publish(topic: str, payload: dict[str, Any], retain: bool) -> None:
        client.publish(topic, json.dumps(payload, ensure_ascii=False, separators=(",", ":")), qos=1, retain=retain)

    trial_prefix = os.environ.get("TRIAL_TOPIC", "planta1/ensayos/vision").rstrip("/")
    service = InspectionService(
        inspector,
        publish,
        max_photo_age_s=float(os.environ.get("MAX_PHOTO_AGE_S", "45")),
        trial_prefix=trial_prefix,
    )

    def on_connect(c, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            log.error("broker refused the connection: %s", reason_code)
            return
        c.subscribe([(photo_topic, 1), (f"{trial_prefix}/foto", 1)])
        status = {"value": "online", "source": "inspector", "model": f"{config.provider.kind}/{config.provider.model}"}
        c.publish(status_topic, json.dumps(status), qos=1, retain=True)
        # What it asks and how, retained: anyone on the UNS can see the exact prompts.
        described = json.dumps(describe(config, inspector), ensure_ascii=False)
        c.publish(f"{status_topic}/config", described, qos=1, retain=True)
        log.info("listening on %s, %d POIs, provider %s", photo_topic, len(config.pois), config.provider.kind)

    client.on_connect = on_connect

    def on_message(c, u, msg) -> None:
        if msg.retain:
            return
        if msg.topic == f"{trial_prefix}/foto":
            service.on_trial(msg.payload)
        else:
            service.on_photo(msg.payload)

    client.on_message = on_message
    if os.environ.get("MQTT_USERNAME"):  # a broker with users and ACLs
        client.username_pw_set(os.environ["MQTT_USERNAME"], os.environ.get("MQTT_PASSWORD"))
    client.connect_async(os.environ.get("MQTT_HOST", "localhost"), int(os.environ.get("MQTT_PORT", "1883")))
    client.loop_start()

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    service.run(stop)
    client.publish(status_topic, offline, qos=1, retain=True).wait_for_publish(timeout=2)
    client.loop_stop()
    return 0
