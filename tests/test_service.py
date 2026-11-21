import base64
import json
from datetime import datetime, timedelta, timezone

import httpx

from poi_inspector.inspector import Inspector
from poi_inspector.providers import ProviderError
from poi_inspector.service import InspectionService, describe

IMG = b"\xff\xd8 photo"
NOW = datetime(2026, 1, 2, 12, tzinfo=timezone.utc)


def make(config, scripted, *answers, clock=None):
    sent = []
    inspector = Inspector(config, scripted(*answers), sleep=lambda s: None)
    images = httpx.Client(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, content=IMG, headers={"content-type": "image/jpeg"})
        )
    )
    service = InspectionService(inspector, lambda t, p, r: sent.append((t, p, r)), images, clock=clock or (lambda: NOW))
    return service, sent


def photo(poi="PI-104", **extra):
    return json.dumps({"value": "IMG_0412", "poi": poi, "image_url": "http://robot/IMG_0412.jpg", **extra}).encode()


def drain(service):
    while not service._queue.empty():
        service.handle(service._queue.get())


def answer(value=6.8):
    return {
        "value": value,
        "unit": "bar",
        "status_reason": "lectura_valida",
        "confidence": 0.97,
        "evidence": "needle at 6.8",
    }


def test_a_good_reading_goes_to_the_poi_path_and_the_detail_next_to_it(config, scripted):
    service, sent = make(config, scripted, answer())
    service.on_photo(photo())
    drain(service)
    topics = {t: (p, r) for t, p, r in sent}
    value, retained = topics["planta1/exterior/pi-104/presion"]
    assert value["value"] == 6.8 and value["unit"] == "bar" and value["quality"] == "good" and retained
    detail, _ = topics["planta1/exterior/pi-104/presion/inspeccion"]
    assert detail["image"] == "IMG_0412" and detail["trace"]["model"] == "test-vlm"
    assert value["inspection_id"] == detail["inspection_id"] == detail["trace"]["inspection_id"]
    assert value["ts"] == detail["captured_at"] and detail["processed_at"] == detail["ts"]


def test_capture_processing_and_queue_wait_are_kept_separate(config, scripted):
    captured = NOW - timedelta(seconds=12)
    received = NOW - timedelta(seconds=7)
    times = iter([received, NOW])
    service, sent = make(config, scripted, answer(), clock=lambda: next(times))
    service.on_photo(photo(ts=captured.isoformat()))
    drain(service)
    topics = {t: p for t, p, _ in sent}
    value = topics["planta1/exterior/pi-104/presion"]
    detail = topics["planta1/exterior/pi-104/presion/inspeccion"]
    assert value["ts"] == captured.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    assert detail["captured_at"] == value["ts"] and detail["processed_at"] != value["ts"]
    assert detail["queue_wait_ms"] == 7000 and detail["capture_age_ms"] == 12000


def test_a_stale_capture_is_discarded_before_inference_and_never_updates_the_value(config, scripted):
    service, sent = make(config, scripted, answer())
    service.on_photo(photo(ts="2000-01-01T00:00:00Z"))
    drain(service)
    assert [t for t, _, _ in sent] == ["planta1/exterior/pi-104/presion/inspeccion"]
    detail = sent[0][1]
    assert detail["quality"] == "stale" and detail["status_reason"] == "captura_caducada"
    assert detail["captured_at"] == "2000-01-01T00:00:00.000Z" and detail["poi"] == "PI-104"
    assert service._inspector.provider.calls == []


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


def test_model_down_is_reported_as_an_error_with_the_photo_context(config, scripted):
    service, sent = make(config, scripted, ProviderError("refused", retryable=False))
    service.on_photo(photo())
    drain(service)
    assert len(sent) == 1 and sent[0][1]["quality"] == "error"
    detail = sent[0][1]
    assert detail["poi"] == "PI-104" and detail["tag"] == "PI-104" and detail["kind"] == "numeric"
    assert detail["image"] == "IMG_0412" and detail["inspection_id"]
    assert detail["captured_at"] and detail["processed_at"]


def test_an_invalid_capture_timestamp_is_an_error_not_a_fresh_reading(config, scripted):
    service, sent = make(config, scripted, answer())
    service.on_photo(photo(ts="not-a-date"))
    drain(service)
    assert len(sent) == 1 and sent[0][1]["quality"] == "error"
    assert sent[0][1]["poi"] == "PI-104" and service._inspector.provider.calls == []


def trial_photo(request="trial-1", **extra):
    return json.dumps(
        {
            "value": "pi-104_6.75.jpg",
            "image_file": "pi-104_6.75.jpg",
            "poi": "PI-104",
            "request": request,
            "image_b64": base64.b64encode(IMG).decode(),
            **extra,
        }
    ).encode()


def test_a_trial_reports_states_but_never_publishes_a_plant_value(config, scripted):
    service, sent = make(config, scripted, answer())
    service.on_trial(trial_photo())
    assert sent[0][0] == "planta1/ensayos/vision/estado" and sent[0][1]["state"] == "en-cola"
    drain(service)
    assert [t for t, _, _ in sent] == [
        "planta1/ensayos/vision/estado",
        "planta1/ensayos/vision/estado",
        "planta1/ensayos/vision/resultado",
    ]
    assert sent[1][1]["state"] == "analizando"
    result = sent[2][1]
    assert result["state"] == "completada" and result["mode"] == "ensayo" and result["request"] == "trial-1"
    assert result["image_file"] == "pi-104_6.75.jpg" and result["inspection_id"]
    assert not any(t.startswith("planta1/exterior/") for t, _, _ in sent)


def test_a_trial_error_keeps_its_context_and_stays_outside_the_plant_uns(config, scripted):
    service, sent = make(config, scripted, ProviderError("refused", retryable=False))
    service.on_trial(trial_photo(request="failed-trial"))
    drain(service)
    result = sent[-1][1]
    assert sent[-1][0] == "planta1/ensayos/vision/resultado" and result["state"] == "error"
    assert result["poi"] == "PI-104" and result["request"] == "failed-trial" and result["mode"] == "ensayo"
    assert not any(t.startswith("planta1/exterior/") for t, _, _ in sent)


def test_robot_photo_cannot_turn_itself_into_a_trial(config, scripted):
    service, sent = make(config, scripted, answer())
    service.on_photo(photo(trial=True, request="forged"))
    drain(service)
    assert sent[0][0] == "planta1/exterior/pi-104/presion/inspeccion"
    assert sent[0][1].get("mode") is None and sent[0][1].get("request") is None


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


def test_the_configuration_is_published_with_the_exact_prompts_and_no_secrets(config, scripted):
    inspector = Inspector(config, scripted)
    body = describe(config, inspector)
    assert body["value"] == inspector.prompt_fingerprint
    poi = next(iter(body["pois"].values()))
    assert "Answer only with the JSON requested" in poi["prompt"]
    assert poi["schema"]["type"] == "object"
    text = json.dumps(body)
    assert "api_key" not in text and "http" not in text


def test_trials_can_be_answered_by_another_model(config, scripted):
    sent = []
    rounds = Inspector(config, scripted(), sleep=lambda s: None)  # the robot's model: never asked
    trials = Inspector(config, scripted(answer(7.1)), sleep=lambda s: None)
    service = InspectionService(
        rounds, lambda t, p, r: sent.append((t, p, r)), clock=lambda: NOW, trial_inspector=trials
    )
    service.on_trial(trial_photo())
    drain(service)
    assert sent[-1][1]["state"] == "completada" and sent[-1][1]["value"] == 7.1
