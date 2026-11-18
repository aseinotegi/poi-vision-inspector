"""A stand-in robot for the docker compose demo: every few seconds it "stops" at a POI
and publishes a synthetic photo of it (gauge at a random pressure, extinguisher in a
random state), inline as base64, the way the inspector expects photos to arrive.
"""

from __future__ import annotations

import base64
import json
import os
import random
import threading
import time

from .synth import EXTINGUISHER_STATES, Trouble, extinguisher_photo, gauge_photo


def next_photo(rng: random.Random, count: int) -> tuple[dict, str]:
    if count % 2 == 0:
        value = round(rng.uniform(3.5, 7.5), 1)
        image = gauge_photo(value, trouble=Trouble.random(rng), seed=count)
        truth = f"PI-104 at {value} bar"
        poi = "PI-104"
    else:
        state = rng.choice(EXTINGUISHER_STATES)
        image = extinguisher_photo(state, trouble=Trouble.random(rng), seed=count)
        truth = f"EXT-12 {state}"
        poi = "EXT-12"
    message = {
        "value": f"IMG_{count:04d}",
        "poi": poi,
        "image_b64": base64.b64encode(image).decode(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "demo-robot",
    }
    return message, truth


def main() -> int:
    import paho.mqtt.client as mqtt

    topic = os.environ.get("PHOTO_TOPIC", "planta1/robots/robot-01/foto")
    period = float(os.environ.get("ROUND_S", "20"))
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="demo-robot")
    client.connect_async(os.environ.get("MQTT_HOST", "localhost"), int(os.environ.get("MQTT_PORT", "1883")))
    client.loop_start()
    rng, count, stop = random.Random(), 0, threading.Event()
    while not stop.wait(period):
        message, truth = next_photo(rng, count)
        client.publish(topic, json.dumps(message), qos=1)
        print(f"photo {message['value']}: {truth} (the truth, to compare with the inspector)", flush=True)
        count += 1
    return 0
