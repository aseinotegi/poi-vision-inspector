"""Measures how well a provider inspects, on photos whose answer is known.

A dataset is a folder with images and ``labels.csv`` (``file,poi,expected``).
``expected`` is a number for numeric POIs and ``OK`` / ``NOK:<reason>`` for states.

The headline number is not accuracy: it is **wrong but trusted**, readings marked
``good`` that are actually wrong. Those are the ones that would reach a rule and move
a robot. An honest ``uncertain`` costs a second photo; a confident mistake costs more.
"""

from __future__ import annotations

import csv
import random
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .inspector import GOOD, InspectionFailed, Inspector
from .synth import EXTINGUISHER_STATES, Trouble, extinguisher_photo, gauge_photo

TOLERANCE = 0.02  # a reading within 2 % of full scale counts as right


@dataclass
class Report:
    numeric: list[dict] = field(default_factory=list)
    state: list[dict] = field(default_factory=list)
    failures: int = 0
    latencies: list[int] = field(default_factory=list)

    def summary(self) -> dict[str, float | int | None]:
        out: dict[str, float | int | None] = {"failed_calls": self.failures}
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

    def markdown(self) -> str:
        labels = {
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
    report = Report()
    detail = []
    with (dataset / "labels.csv").open(encoding="utf-8") as f:
        labels = list(csv.DictReader(f))
    for row in labels:
        image = (dataset / row["file"]).read_bytes()
        inspector.forget()  # independent photos: no "jump since last reading" check
        try:
            result = inspector.inspect(row["poi"], image)
        except InspectionFailed:
            report.failures += 1
            detail.append({**row, "got": "", "quality": "failed", "checks": "no answer"})
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
        detail.append({**row, "got": got, "quality": result.quality, "checks": "; ".join(result.checks)})
    if results_csv:
        with results_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "poi", "expected", "got", "quality", "checks"])
            writer.writeheader()
            writer.writerows(detail)
    return report
