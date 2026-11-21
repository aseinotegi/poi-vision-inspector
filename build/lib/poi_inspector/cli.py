"""Command line.

poi-inspect check  --config config/inspector.yaml          validate the configuration
poi-inspect read   photo.jpg --poi PI-104                   inspect one photo, print JSON
poi-inspect synth  data/synthetic                           write a labelled synthetic set
poi-inspect evaluate data/synthetic                         metrics for the configured provider
poi-inspect serve                                           MQTT service (see service.py)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .prompt import fingerprint


def _inspector(config_path: str):
    from .inspector import Inspector
    from .providers import make_provider

    config = load_config(config_path)
    return Inspector(config, make_provider(config.provider))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="poi-inspect", description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="config/inspector.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="validate the configuration")
    read = sub.add_parser("read", help="inspect one photo")
    read.add_argument("image", type=Path)
    read.add_argument("--poi", required=True)
    synth = sub.add_parser("synth", help="write a labelled synthetic dataset")
    synth.add_argument("out", type=Path)
    synth.add_argument("--gauges", type=int, default=40)
    synth.add_argument("--extinguishers", type=int, default=30)
    ev = sub.add_parser("evaluate", help="measure the configured provider on a dataset")
    ev.add_argument("dataset", type=Path)
    ev.add_argument("--results", type=Path, help="per-photo CSV")
    ev.add_argument("--report", type=Path, help="versioned JSON evaluation evidence")
    sub.add_parser("serve", help="run the MQTT inspection service")
    sub.add_parser("demo-robot", help="publish synthetic POI photos to MQTT (demo)")
    args = parser.parse_args(argv)
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    try:
        if args.command == "check":
            config = load_config(args.config)
            print(
                f"ok: {len(config.profiles)} profiles, {len(config.pois)} POIs, "
                f"provider {config.provider.kind}/{config.provider.model or '-'}, prompt {fingerprint(config)}"
            )
        elif args.command == "read":
            from .inspector import InspectionFailed

            mime = "image/png" if args.image.suffix.lower() == ".png" else "image/jpeg"
            try:
                result = _inspector(args.config).inspect(args.poi, args.image.read_bytes(), mime)
            except InspectionFailed as exc:
                print(f"no answer from the model: {exc}", file=sys.stderr)
                return 2
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        elif args.command == "synth":
            from .evaluate import make_synthetic_dataset

            count = make_synthetic_dataset(args.out, args.gauges, args.extinguishers)
            print(f"{count} photos and labels.csv written to {args.out}")
        elif args.command == "evaluate":
            from .evaluate import evaluate

            inspector = _inspector(args.config)
            report = evaluate(inspector, args.dataset, args.results)
            cfg = inspector.config.provider
            print(f"\nProvider {cfg.kind} · model {cfg.model or '-'} · prompt {inspector.prompt_fingerprint}\n")
            print(report.markdown())
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                document = report.document(inspector, args.dataset)
                args.report.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(f"\nEvaluation evidence saved to {args.report}")
        elif args.command == "demo-robot":
            from .demo_robot import main as demo_robot

            return demo_robot()
        else:
            from .service import main as serve

            return serve(args.config)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
