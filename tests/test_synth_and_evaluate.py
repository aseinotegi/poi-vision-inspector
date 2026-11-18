import csv
import io
import json

import pytest
from PIL import Image

from poi_inspector.evaluate import evaluate, make_synthetic_dataset
from poi_inspector.inspector import Inspector
from poi_inspector.synth import Trouble, extinguisher_photo, gauge_photo


def test_synthetic_photos_are_valid_jpegs_and_reproducible():
    a = gauge_photo(6.8, seed=3)
    assert a == gauge_photo(6.8, seed=3)
    assert a != gauge_photo(4.2, seed=3)
    assert Image.open(io.BytesIO(a)).size == (640, 640)
    hard = extinguisher_photo("ausente", trouble=Trouble(tilt_deg=10, blur=1.5, glare=0.5, darkness=0.3), seed=1)
    assert Image.open(io.BytesIO(hard)).format == "JPEG"


def test_unknown_extinguisher_state():
    with pytest.raises(ValueError):
        extinguisher_photo("rojo")


def test_dataset_has_labels_for_every_photo(tmp_path):
    assert make_synthetic_dataset(tmp_path, gauges=6, extinguishers=6) == 12
    rows = list(csv.DictReader((tmp_path / "labels.csv").open()))
    assert all((tmp_path / r["file"]).exists() for r in rows)
    assert {r["expected"] for r in rows if r["poi"] == "EXT-12"} == {"OK", "NOK:ausente", "NOK:presion_baja"}


class Oracle:
    """A provider that knows the labels: perfect on gauges except one confident mistake."""

    name = "oracle"

    def __init__(self, labels, wrong_file):
        self.by_hash = labels
        self.wrong = wrong_file

    def ask(self, image, mime, instruction, schema):
        import hashlib

        from poi_inspector.providers import Answer

        file, expected = self.by_hash[hashlib.sha256(image).hexdigest()]
        if expected.replace(".", "").isdigit():
            value = float(expected) + (2.0 if file == self.wrong else 0)
            body = {
                "value": value,
                "unit": "bar",
                "status_reason": "lectura_valida",
                "confidence": 0.95,
                "evidence": "",
            }
        else:
            state, _, reason = expected.partition(":")
            body = {"value": state, "status_reason": reason or "correcto", "confidence": 0.95, "evidence": ""}
        return Answer(json.dumps(body), "oracle", 10)


def test_evaluation_counts_confident_mistakes(tmp_path, config):
    import hashlib

    make_synthetic_dataset(tmp_path, gauges=5, extinguishers=3)
    rows = list(csv.DictReader((tmp_path / "labels.csv").open()))
    labels = {hashlib.sha256((tmp_path / r["file"]).read_bytes()).hexdigest(): (r["file"], r["expected"]) for r in rows}
    inspector = Inspector(config, Oracle(labels, wrong_file="gauge_001.jpg"))
    report = evaluate(inspector, tmp_path, tmp_path / "results.csv")
    summary = report.summary()
    assert summary["gauges"] == 5 and summary["gauges_wrong_but_trusted"] == 1
    assert summary["state_accuracy_pct"] == 100.0 and summary["states_wrong_but_trusted"] == 0
    assert "Wrong but trusted" in report.markdown()
    assert (tmp_path / "results.csv").read_text().count("\n") == 9
