import pytest

from poi_inspector.inspector import GOOD, REJECTED, UNCERTAIN, InspectionFailed, Inspector
from poi_inspector.providers import ProviderError

IMG = b"\xff\xd8 fake jpeg"


def reading(value, reason="lectura_valida", confidence=0.95, unit="bar"):
    return {"value": value, "unit": unit, "status_reason": reason, "confidence": confidence, "evidence": "needle"}


def state(value, reason, confidence=0.93):
    return {"value": value, "status_reason": reason, "confidence": confidence, "evidence": "seen"}


def make(config, scripted, *answers):
    provider = scripted(*answers)
    return Inspector(config, provider, sleep=lambda s: None), provider


def test_a_clean_reading_is_trusted_and_traced(config, scripted):
    inspector, provider = make(config, scripted, reading(6.8))
    result = inspector.inspect("PI-104", IMG)
    assert (result.value, result.unit, result.quality, result.checks) == (6.8, "bar", GOOD, [])
    assert result.trace["model"] == "test-vlm" and result.trace["prompt"] == inspector.prompt_fingerprint
    assert len(result.trace["image_sha256"]) == 64
    assert "PI-104" in provider.calls[0]["instruction"]


def test_could_not_read_is_a_legitimate_answer(config, scripted):
    inspector, _ = make(config, scripted, reading(None, "no_legible", 0.9))
    result = inspector.inspect("PI-104", IMG)
    assert result.value is None and result.quality == UNCERTAIN and result.status_reason == "no_legible"
    assert result.checks == []  # nothing wrong with saying "I cannot read it"


def test_valid_reason_without_value_is_flagged(config, scripted):
    inspector, _ = make(config, scripted, reading(None))
    assert "'lectura_valida' without a value" in inspector.inspect("PI-104", IMG).checks


def test_out_of_scale_readings_are_rejected(config, scripted):
    inspector, _ = make(config, scripted, reading(14.0))
    result = inspector.inspect("PI-104", IMG)
    assert result.quality == REJECTED and result.status_reason == "fuera_de_rango"


@pytest.mark.parametrize(
    "answer, check",
    [
        (reading(6.8, confidence=0.5), "confidence 0.50 below 0.80"),
        (reading(6.8, unit="psi"), "unit 'psi'"),
        (reading(6.8, unit=None), "missing unit"),
        ({**reading(6.8), "evidence": ""}, "missing visual evidence"),
        (reading(6.8, reason="no_legible"), "a value with reason 'no_legible'"),
    ],
)
def test_doubtful_readings_are_kept_but_marked_uncertain(config, scripted, answer, check):
    inspector, _ = make(config, scripted, answer)
    result = inspector.inspect("PI-104", IMG)
    assert result.quality == UNCERTAIN and any(check in c for c in result.checks)


@pytest.mark.parametrize(
    "answer",
    [
        {**reading(6.8), "confidence": "high"},
        reading(6.8, reason="esta_bien"),
        {**reading(6.8), "unexpected": True},
        {**reading(6.8), "bbox_2d": [1, 2, 3, 4, 5]},
    ],
)
def test_schema_violations_are_rejected(config, scripted, answer):
    inspector, _ = make(config, scripted, answer)
    result = inspector.inspect("PI-104", IMG)
    assert result.quality == REJECTED and result.status_reason == "respuesta_invalida"
    assert any(check.startswith("schema:") for check in result.checks)


def test_unit_aliases_are_accepted(config, scripted):
    inspector, _ = make(config, scripted, reading(6.8, unit="BAR"))
    assert inspector.inspect("PI-104", IMG).quality == GOOD


def test_an_impossible_jump_since_the_last_trusted_reading_is_doubted(config, scripted):
    inspector, _ = make(config, scripted, reading(4.2), reading(9.5), reading(5.0))
    assert inspector.inspect("PI-104", IMG).quality == GOOD
    jump = inspector.inspect("PI-104", IMG)
    assert jump.quality == UNCERTAIN and "jump of +5.30" in jump.checks[0]
    assert inspector.inspect("PI-104", IMG).quality == GOOD  # compared with 4.2, the last trusted one


def test_garbage_is_rejected_not_crashed(config, scripted):
    inspector, _ = make(config, scripted, "Sure! The gauge reads about 6.8 bar.")
    result = inspector.inspect("PI-104", IMG)
    assert result.quality == REJECTED and result.status_reason == "respuesta_invalida"


def test_extinguisher_states(config, scripted):
    inspector, _ = make(config, scripted, state("OK", "correcto"), state("NOK", "ausente"), state("MAYBE", "ausente"))
    ok = inspector.inspect("EXT-12", IMG)
    missing = inspector.inspect("EXT-12", IMG)
    nonsense = inspector.inspect("EXT-12", IMG)
    assert (ok.value, ok.quality) == ("OK", GOOD)
    assert (missing.value, missing.status_reason, missing.quality) == ("NOK", "ausente", GOOD)
    assert nonsense.quality == REJECTED


def test_an_ok_state_with_no_legible_is_never_good(config, scripted):
    inspector, _ = make(config, scripted, state("OK", "no_legible", confidence=0.99))
    result = inspector.inspect("EXT-12", IMG)
    assert result.value is None and result.quality == UNCERTAIN and result.status_reason == "no_legible"


def test_a_contradictory_state_and_reason_are_not_good(config, scripted):
    inspector, _ = make(config, scripted, state("OK", "ausente"))
    result = inspector.inspect("EXT-12", IMG)
    assert result.value == "OK" and result.quality == UNCERTAIN
    assert "state and reason contradict each other" in result.checks


def test_a_model_box_has_coordinates_and_explicit_provenance(config, scripted):
    inspector, _ = make(config, scripted, {**reading(6.8), "bbox_2d": [100, 150, 700, 850]})
    result = inspector.inspect("PI-104", IMG)
    assert result.bbox_2d == [100, 150, 700, 850] and result.bbox_source == "model"


def test_a_replayed_box_is_not_presented_as_a_live_model_prediction(config, scripted):
    inspector, provider = make(config, scripted, {**reading(6.8), "bbox_2d": [100, 150, 700, 850]})
    provider.name = "replay"
    assert inspector.inspect("PI-104", IMG).bbox_source == "replay"


def test_an_inverted_box_is_not_good_or_drawable(config, scripted):
    inspector, _ = make(config, scripted, {**reading(6.8), "bbox_2d": [700, 850, 100, 150]})
    result = inspector.inspect("PI-104", IMG)
    assert result.bbox_2d is None and result.bbox_source is None and result.quality == UNCERTAIN
    assert "invalid bounding box" in result.checks[0]


def test_transient_errors_are_retried(config, scripted):
    busy = ProviderError("429", retryable=True)
    inspector, provider = make(config, scripted, busy, busy, reading(6.8))
    assert inspector.inspect("PI-104", IMG).value == 6.8
    assert len(provider.calls) == 3


def test_no_answer_is_a_failure_not_an_empty_reading(config, scripted):
    inspector, provider = make(config, scripted, ProviderError("401 bad key", retryable=False))
    with pytest.raises(InspectionFailed, match="401"):
        inspector.inspect("PI-104", IMG)
    assert len(provider.calls) == 1  # not retried


def test_retries_give_up(config, scripted):
    busy = ProviderError("503", retryable=True)
    inspector, provider = make(config, scripted, busy, busy, busy)
    with pytest.raises(InspectionFailed, match="attempt 3/3"):
        inspector.inspect("PI-104", IMG)


def test_unknown_poi(config, scripted):
    inspector, _ = make(config, scripted)
    with pytest.raises(KeyError):
        inspector.inspect("PI-999", IMG)
