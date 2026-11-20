import copy

from poi_inspector.prompt import build_question, fingerprint


def test_numeric_question_carries_the_poi_context(config):
    q = build_question(config.pois["PI-104"], config.profiles["manometro"])
    assert "PI-104" in q.instruction and "0 to 10 bar" in q.instruction
    assert q.schema["properties"]["status_reason"]["enum"] == ["lectura_valida", "no_legible", "no_encontrado"]
    assert q.schema["properties"]["value"]["type"] == ["number", "null"]
    assert set(q.schema["required"]) == {"value", "unit", "status_reason", "confidence", "evidence", "bbox_2d"}
    assert q.schema["properties"]["bbox_2d"]["anyOf"][1]["maxItems"] == 4


def test_state_question_uses_the_profile_lists(config):
    q = build_question(config.pois["EXT-12"], config.profiles["extintor"])
    assert q.schema["properties"]["value"]["enum"] == ["OK", "NOK", None]
    assert "presion_baja" in q.schema["properties"]["status_reason"]["enum"]
    assert "EXT-12" in q.instruction


def test_fingerprint_is_stable_and_changes_with_any_prompt_input(config):
    assert fingerprint(config) == fingerprint(copy.deepcopy(config))
    changed = config.model_copy(
        update={
            "profiles": {
                **config.profiles,
                "manometro": config.profiles["manometro"].model_copy(update={"instruction": "Read it."}),
            }
        }
    )
    assert fingerprint(changed) != fingerprint(config)
