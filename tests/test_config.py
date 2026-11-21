import copy

import pytest
import yaml

from poi_inspector.config import ConfigError, parse_config

from .conftest import CONFIG

RAW = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def broken(**changes):
    raw = copy.deepcopy(RAW)
    for path, value in changes.items():
        node = raw
        keys = path.split("__")
        for key in keys[:-1]:
            node = node[key]
        if value is None:
            node.pop(keys[-1])
        else:
            node[keys[-1]] = value
    return raw


def test_the_shipped_configuration_is_valid(config):
    assert set(config.profiles) == {"manometro", "extintor"}
    assert config.pois["PI-104"].expect.unit == "bar"
    assert config.provider.kind == "ollama"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"pois__PI-104__profile": "termometro"}, "unknown profile"),
        ({"pois__PI-104__expect": None}, "needs 'expect'"),
        ({"pois__EXT-12__expect": {"unit": "bar", "min": 0, "max": 1}}, "does not apply"),
        ({"pois__PI-104__expect__min": 20}, "below expect.max"),
        ({"pois__PI-104__uns_topic": "planta1/+/presion"}, "not a valid topic"),
        ({"profiles__manometro__reasons": ["no_legible"]}, "lectura_valida"),
        ({"profiles__extintor__reasons": []}, "must not be empty"),
        ({"profiles__extintor__colour": "red"}, "Extra inputs"),
        ({"provider__model": ""}, "needs a 'model'"),
        ({"provider__kind": "watson"}, "provider.kind"),
    ],
)
def test_mistakes_are_caught_at_start_with_a_readable_message(changes, message):
    with pytest.raises(ConfigError, match=message):
        parse_config(broken(**changes))


def test_replay_needs_its_answers_file():
    with pytest.raises(ConfigError, match="answers"):
        parse_config(broken(provider={"kind": "replay"}))


def test_where_the_model_runs_can_come_from_the_environment():
    from poi_inspector.config import load_config

    config = load_config(CONFIG, environ={"POI_PROVIDER_URL": "http://ollama:11434", "POI_PROVIDER_MODEL": "gemma3"})
    assert (config.provider.url, config.provider.model) == ("http://ollama:11434", "gemma3")


def test_a_trial_provider_from_the_environment():
    from poi_inspector.config import load_config

    config = load_config(CONFIG, environ={"POI_TRIAL_PROVIDER": "gemini", "POI_TRIAL_MODEL": "gemini-x"})
    assert config.trial_provider.kind == "gemini" and config.trial_provider.model == "gemini-x"
    assert config.trial_provider.api_key_env == "GEMINI_API_KEY" and config.trial_provider.retries == 1
    assert load_config(CONFIG, environ={}).trial_provider is None
