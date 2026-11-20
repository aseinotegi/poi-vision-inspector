"""Measures how well a provider inspects, on photos whose answer is known.

A dataset is a folder with images and ``labels.csv`` (``file,poi,expected``).
``expected`` is a number for numeric POIs and ``OK`` / ``NOK:<reason>`` for states.

The headline number is not accuracy: it is **wrong but trusted**, readings marked
``good`` that are actually wrong. Those are the ones that would reach a rule and move
a robot. An honest ``uncertain`` costs a second photo; a confident mistake costs more.
"""

from __future__ import annotations

import csv
import hashlib
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .inspector import GOOD, InspectionFailed, Inspector
from .synth import EXTINGUISHER_STATES, Trouble, extinguisher_photo, gauge_photo

TOLERANCE = 0.02  # a reading within 2 % of full scale counts as right


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class Report:
    numeric: list[dict] = field(default_factory=list)
    state: list[dict] = field(default_factory=list)
    failures: int = 0
    latencies: list[int] = field(default_factory=list)
    detail: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""

    def summary(self) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = {
            "images": len(self.detail),
            "unique_images": len({r["image_sha256"] for r in self.detail}),
            "abstentions": sum(r["quality"] not in (GOOD, "failed") for r in self.detail),
            "failed_calls": self.failures,
        }
        if self.numeric:
            trusted = [r for r in self.numeric if r["quality"] == GOOD]
            errors = [abs(r["error_fs"]) for r in trusted]
            out |= {
                "gauges": len(self.numeric),
                "gauges_trusted_pct": 100 * len(trusted) / len(self.numeric),
                "gauge_mae_pct_fs": 100 * statistics.mean(errors) if errors else None,
                "gauge_p95_pct_fs": 100 * _p95(errors) if errors else None,
                "gauges_wrong_but_trusted": sum(e > TOLERANCE for e in errors),
            }
        if self.state:
            trusted = [r for r in self.state if r["quality"] == GOOD]
            right = [r for r in trusted if r["got"] == r["expected"]]
            out |= {
                "states": len(self.state),
                "states_trusted_pct": 100 * len(trusted) / len(self.state),
                "state_accuracy_pct": 100 * len(right) / len(trusted) if trusted else None,
                "states_wrong_but_trusted": len(trusted) - len(right),
            }
        if self.latencies:
            out["latency_p50_ms"] = int(statistics.median(self.latencies))
            out["latency_p95_ms"] = int(_p95(self.latencies))
        return out

    def document(self, inspector: Inspector, dataset: Path) -> dict[str, Any]:
        provider = inspector.config.provider
        return {
            "schema_version": 1,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "provider": {
                "kind": provider.kind,
                "model": provider.model,
                "url": provider.url,
                "timeout_s": provider.timeout_s,
                "temperature": provider.temperature,
                "retries": provider.retries,
                "think": provider.think,
                "reported_models": sorted({r.get("model") for r in self.detail if r.get("model")}),
            },
            "prompt_fingerprint": inspector.prompt_fingerprint,
            "dataset": {
                "path": str(dataset),
                "labels_sha256": hashlib.sha256((dataset / "labels.csv").read_bytes()).hexdigest(),
            },
            "summary": self.summary(),
            "images": self.detail,
        }

    def markdown(self) -> str:
        labels = {
            "images": "Evaluated images",
            "unique_images": "Unique image contents",
            "abstentions": "Readings not accepted as good",
            "gauges": "Gauge photos",
            "gauges_trusted_pct": "Trusted readings (%)",
            "gauge_mae_pct_fs": "Mean error, trusted (% of full scale)",
            "gauge_p95_pct_fs": "P95 error, trusted (% of full scale)",
            "gauges_wrong_but_trusted": f"Wrong but trusted (> {TOLERANCE:.0%} FS)",
            "states": "Extinguisher photos",
            "states_trusted_pct": "Trusted answers (%)",
            "state_accuracy_pct": "Accuracy, trusted (%)",
            "states_wrong_but_trusted": "Wrong but trusted",
            "failed_calls": "Calls with no answer",
            "latency_p50_ms": "Latency p50 (ms)",
            "latency_p95_ms": "Latency p95 (ms)",
        }
        rows = ["| Metric | Value |", "|---|---|"]
        for key, value in self.summary().items():
            shown = "—" if value is None else f"{value:.1f}" if isinstance(value, float) else str(value)
            rows.append(f"| {labels.get(key, key)} | {shown} |")
        return "\n".join(rows)


def _p95(values: list[float] | list[int]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def make_synthetic_dataset(out: Path, gauges: int = 40, extinguishers: int = 30, seed: int = 7) -> int:
    """Writes a reproducible labelled set: easy and hard photos of both POIs."""
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(gauges):
        value = round(rng.uniform(0.3, 9.7), 1)
        trouble = Trouble.random(rng, hard=i % 4 == 3)
        name = f"gauge_{i:03d}.jpg"
        (out / name).write_bytes(gauge_photo(value, trouble=trouble, seed=seed + i))
        rows.append((name, "PI-104", f"{value:g}"))
    for i in range(extinguishers):
        state = EXTINGUISHER_STATES[i % len(EXTINGUISHER_STATES)]
        trouble = Trouble.random(rng, hard=i % 5 == 4)
        name = f"extinguisher_{i:03d}.jpg"
        (out / name).write_bytes(extinguisher_photo(state, trouble=trouble, seed=seed + 100 + i))
        rows.append((name, "EXT-12", "OK" if state == "ok" else f"NOK:{state}"))
    with (out / "labels.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "poi", "expected"])
        writer.writerows(rows)
    return len(rows)


def evaluate(inspector: Inspector, dataset: Path, results_csv: Path | None = None) -> Report:
    report = Report(started_at=_now())
    with (dataset / "labels.csv").open(encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    for row in labels:
        image = (dataset / row["file"]).read_bytes()
        image_sha256 = hashlib.sha256(image).hexdigest()
        inspector.forget()  # independent photos: no "jump since last reading" check
        try:
            result = inspector.inspect(row["poi"], image)
        except InspectionFailed as exc:
            report.failures += 1
            report.detail.append(
                {
                    **row,
                    "image_sha256": image_sha256,
                    "got": "",
                    "quality": "failed",
                    "checks": "no answer",
                    "error": str(exc),
                }
            )
            continue
        report.latencies.append(result.trace.get("latency_ms", 0))
        if result.kind == "numeric":
            expect = inspector.config.pois[row["poi"]].expect
            assert expect is not None
            span = expect.max - expect.min
            error = 0.0 if result.value is None else (result.value - float(row["expected"])) / span
            report.numeric.append({"quality": result.quality, "error_fs": error})
            got = "" if result.value is None else f"{result.value:g}"
        else:
            got = result.value if result.value != "NOK" else f"NOK:{result.status_reason}"
            report.state.append({"quality": result.quality, "got": got, "expected": row["expected"]})
        report.detail.append(
            {
                **row,
                "image_sha256": image_sha256,
                "got": got,
                "quality": result.quality,
                "confidence": result.confidence,
                "status_reason": result.status_reason,
                "checks": "; ".join(result.checks),
                "model": result.trace.get("model"),
                "provider": result.trace.get("provider"),
                "latency_ms": result.trace.get("latency_ms"),
            }
        )
    report.finished_at = _now()
    if results_csv:
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "poi", "expected", "got", "quality", "checks"])
            writer.writeheader()
            writer.writerows({k: r.get(k, "") for k in writer.fieldnames} for r in report.detail)
    return report
