"""Command-line prediction for one farm record.

    python predict.py --json samples/valid.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from src.config import PROJECT_ROOT
from src.inference import ArtifactsMissingError, load_bundle, predict_record
from src.schemas import describe_errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict yield, risk and cluster for one farm.")
    parser.add_argument("--json", required=True, type=Path, help="path to one JSON farm record")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        record = json.loads(args.json.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.json}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"ERROR: {args.json} is not valid JSON ({exc})", file=sys.stderr)
        return 2

    try:
        bundle = load_bundle(args.root)
        result = predict_record(bundle, record)
    except ArtifactsMissingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    except ValidationError as exc:
        print(json.dumps({"error": "invalid_input", "problems": describe_errors(exc)}, indent=2),
              file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
