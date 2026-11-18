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
import json
import logging
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Any

import httpx

from .config import load_config
from .inspector import REJECTED, Inspection, InspectionFailed, Inspector
from .providers import make_provider

log = logging.getLogger("inspector")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def value_message(result: Inspection) -> dict[str, Any] | None:
    """The UNS message for the POI's value, or None if nothing should be published."""
    if result.value is None or result.quality == REJECTED:
        return None
    message = {
        "value": result.value,
        "ts": now_iso(),
        "source": "inspector",
        "quality": result.quality,
        "status_reason": result.status_reason,
        "confidence": result.confidence,
        "model": result.trace.get("model"),
    }
    if result.unit:
        message["unit"] = result.unit
    return message


def inspection_message(result: Inspection, image_id: str) -> dict[str, Any]:
    body = result.to_dict()
    body.update({"value": result.value, "ts": now_iso(), "source": "inspector", "image": image_id})
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
    def __init__(self, inspector: Inspector, publish, client: httpx.Client | None = None) -> None:
        self._inspector = inspector
        self._publish = publish  # (topic, payload dict, retain) -> None
        self._client = client or httpx.Client(timeout=15)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=100)

    def on_photo(self, payload: bytes | bytearray) -> None:
        try:
            photo = json.loads(payload)
        except ValueError:
            log.warning("ignoring a photo message that is not JSON")
            return
        poi = photo.get("poi") or photo.get("point")
        if poi not in self._inspector.config.pois:
            log.debug("photo of %r: not a configured POI", poi)
            return
        try:
            self._queue.put_nowait({**photo, "poi": poi})
        except queue.Full:
            log.warning("inspection queue full, dropping photo %s", photo.get("value"))

    def handle(self, photo: dict[str, Any]) -> None:
        poi_id = photo["poi"]
        poi = self._inspector.config.pois[poi_id]
        image_id = str(photo.get("value", ""))
        try:
            image, mime = load_image(photo, self._client)
            result = self._inspector.inspect(poi_id, image, mime)
        except (InspectionFailed, httpx.HTTPError, ValueError) as exc:
            log.error("%s %s: no inspection (%s)", poi_id, image_id, exc)
            self._publish(
                f"{poi.uns_topic}/inspeccion",
                {
                    "value": None,
                    "ts": now_iso(),
                    "source": "inspector",
                    "quality": "error",
                    "image": image_id,
                    "error": str(exc)[:300],
                },
                True,
            )
            return
        self._publish(f"{poi.uns_topic}/inspeccion", inspection_message(result, image_id), True)
        message = value_message(result)
        if message:
            self._publish(poi.uns_topic, message, True)
        log.info(
            "%s %s → %s %s (%s, %s ms)",
            poi_id,
            image_id,
            result.value,
            result.unit or "",
            result.quality,
            result.trace.get("latency_ms"),
        )

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                photo = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self.handle(photo)


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

    service = InspectionService(inspector, publish)

    def on_connect(c, userdata, flags, reason_code, properties) -> None:
        c.subscribe(photo_topic, qos=1)
        status = {"value": "online", "source": "inspector", "model": f"{config.provider.kind}/{config.provider.model}"}
        c.publish(status_topic, json.dumps(status), qos=1, retain=True)
        log.info("listening on %s, %d POIs, provider %s", photo_topic, len(config.pois), config.provider.kind)

    client.on_connect = on_connect
    client.on_message = lambda c, u, msg: None if msg.retain else service.on_photo(msg.payload)
    client.connect_async(os.environ.get("MQTT_HOST", "localhost"), int(os.environ.get("MQTT_PORT", "1883")))
    client.loop_start()

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    service.run(stop)
    client.publish(status_topic, offline, qos=1, retain=True).wait_for_publish(timeout=2)
    client.loop_stop()
    return 0
