import base64
import json

import httpx

from poi_inspector.inspector import Inspector
from poi_inspector.providers import ProviderError
from poi_inspector.service import InspectionService

IMG = b"\xff\xd8 photo"


def make(config, scripted, *answers):
    sent = []
    inspector = Inspector(config, scripted(*answers), sleep=lambda s: None)
    images = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=IMG, headers={"content-type": "image/jpeg"})
        )
    )
    service = InspectionService(inspector, lambda t, p, r: sent.append((t, p, r)), images)
    return service, sent


def photo(poi="PI-104", **extra):
    return json.dumps({"value": "IMG_0412", "poi": poi, "image_url": "http://robot/IMG_0412.jpg", **extra}).encode()


def drain(service):
    while not service._queue.empty():
        service.handle(service._queue.get())


def test_a_good_reading_goes_to_the_poi_path_and_the_detail_next_to_it(config, scripted):
    service, sent = make(
        config,
        scripted,
        {
            "value": 6.8,
            "unit": "bar",
            "status_reason": "lectura_valida",
            "confidence": 0.97,
            "evidence": "needle at 6.8",
        },
    )
    service.on_photo(photo())
    drain(service)
    topics = {t: (p, r) for t, p, r in sent}
    value, retained = topics["planta1/exterior/pi-104/presion"]
    assert value["value"] == 6.8 and value["unit"] == "bar" and value["quality"] == "good" and retained
    detail, _ = topics["planta1/exterior/pi-104/presion/inspeccion"]
    assert detail["image"] == "IMG_0412" and detail["trace"]["model"] == "test-vlm"


def test_rejected_readings_never_overwrite_the_value(config, scripted):
    service, sent = make(
        config,
        scripted,
        {"value": 42, "unit": "bar", "status_reason": "lectura_valida", "confidence": 0.99, "evidence": ""},
    )
    service.on_photo(photo())
    drain(service)
    assert [t for t, _, _ in sent] == ["planta1/exterior/pi-104/presion/inspeccion"]


def test_states_and_inline_images(config, scripted):
    service, sent = make(
        config, scripted, {"value": "NOK", "status_reason": "ausente", "confidence": 0.9, "evidence": "empty bracket"}
    )
    service.on_photo(
        json.dumps({"value": "IMG_7", "point": "EXT-12", "image_b64": base64.b64encode(IMG).decode()}).encode()
    )
    drain(service)
    assert ("planta1/exterior/ext-12/estado", "NOK") in [(t, p["value"]) for t, p, _ in sent]


def test_model_down_is_reported_as_an_error(config, scripted):
    service, sent = make(config, scripted, ProviderError("refused", retryable=False))
    service.on_photo(photo())
    drain(service)
    assert len(sent) == 1 and sent[0][1]["quality"] == "error"


def test_photos_of_other_points_are_ignored(config, scripted):
    service, sent = make(config, scripted)
    service.on_photo(photo(poi="PI-999"))
    service.on_photo(b"not json")
    assert service._queue.empty()


def test_demo_robot_alternates_both_pois_with_inline_photos():
    import random

    from poi_inspector.demo_robot import next_photo

    rng = random.Random(1)
    first, truth = next_photo(rng, 0)
    second, _ = next_photo(rng, 1)
    assert (first["poi"], second["poi"]) == ("PI-104", "EXT-12")
    assert base64.b64decode(first["image_b64"])[:2] == b"\xff\xd8" and "bar" in truth
