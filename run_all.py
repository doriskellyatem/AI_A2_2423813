"""One command to recreate data, train every model and regenerate every artifact.

    python run_all.py --student-id ug2423813@ines.ac.rw --seed 7389

Student registration: 25/27389  |  Verification code: AI-A2-7389
"""
from __future__ import annotations

import argparse
import sys

from src.identity import IdentityError
from src.pipeline import run_pipeline
from src.validation import DataValidationError

# Defaults = my own details, so a bare `python run_all.py` also works.
# The assessor can still pass --student-id and --seed explicitly.
DEFAULT_STUDENT_ID = "ug2423813@ines.ac.rw"
DEFAULT_SEED = 3813  # last four digits of 24/23813


def main() -> int:
    parser = argparse.ArgumentParser(description="Musanze Harvest Intelligence Service")
    parser.add_argument("--student-id", default=DEFAULT_STUDENT_ID,
                        help=f"institutional email (default: {DEFAULT_STUDENT_ID})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"last four digits of 24/23813 (default: {DEFAULT_SEED})")
    args = parser.parse_args()
    try:
        result = run_pipeline(args.student_id, args.seed)
    except (IdentityError, DataValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    paths = result["paths"]
    print("\nGenerated artifacts:")
    for folder in (paths.data_dir, paths.artifacts_dir, paths.plots_dir, paths.models_dir):
        for item in sorted(folder.glob("*")):
            if item.is_file() and item.name != "generate_data.py" and item.suffix != ".pyc" and item.name != "__init__.py":
                print("  ", item.relative_to(paths.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
