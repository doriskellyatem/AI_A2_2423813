"""Synthetic Musanze potato-farm data generator.

Run directly:
    python data/generate_data.py --student-id <username@domain> --seed <LLLL>

or import ``generate_dataset`` (run_all.py does this). The same seed always
produces byte-identical CSV output, which is proven by the SHA-256 fingerprint.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def sha256_of_file(path: Path) -> str:
    """Hash the exact bytes of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_dataset(seed: int, csv_path: Path, fingerprint_path: Path | None = None) -> dict:
    """Create the individual dataset and return a small evidence dictionary."""
    rng = np.random.default_rng(seed)
    n = 240 + (seed % 41)                      # base rows: 240..280

    # --- features, drawn in a fixed order so the random stream is stable -------
    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fert = rng.uniform(0, 250, n)
    field = rng.uniform(0.2, 4.0, n)
    season = rng.choice(np.array(["A", "B"]), size=n)
    support = rng.integers(0, 2, size=n)
    noise = rng.normal(0.0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil
        + 0.02 * fert
        + 1.30 * field
        - 0.25 * np.abs(temp - 18)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)

    # high_risk: yield PER HECTARE below its median (computed on true values,
    # before any value is blanked out).
    yield_per_ha = yield_tons / field
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)

    frame = pd.DataFrame(
        {
            "farm_id": [f"F{i + 1:03d}" for i in range(n)],
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "soil_moisture_pct": soil,
            "fertilizer_kg": fert,
            "field_size_ha": field,
            "season": season,
            "cooperative_support": support,
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        }
    )

    # --- deliberate data problems ---------------------------------------------
    m = round(0.05 * n)
    soil_rows = rng.choice(n, size=m, replace=False)
    remaining = np.setdiff1d(np.arange(n), soil_rows)        # keep the two sets distinct
    fert_rows = rng.choice(remaining, size=m, replace=False)
    frame.loc[soil_rows, "soil_moisture_pct"] = np.nan
    frame.loc[fert_rows, "fertilizer_kg"] = np.nan

    dup_rows = rng.choice(n, size=3, replace=False)
    frame = pd.concat([frame, frame.iloc[dup_rows]], ignore_index=True)

    # --- write + fingerprint ---------------------------------------------------
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False, float_format="%.4f", lineterminator="\n")
    digest = sha256_of_file(csv_path)
    if fingerprint_path is not None:
        fingerprint_path = Path(fingerprint_path)
        fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint_path.write_text(f"{digest}  {csv_path.name}\n", encoding="utf-8")

    return {
        "seed": seed,
        "base_rows": int(n),
        "total_rows_with_duplicates": int(len(frame)),
        "missing_per_column_injected": int(m),
        "duplicate_rows_appended": 3,
        "sha256": digest,
        "csv_path": str(csv_path),
    }


def _cli() -> int:
    sys.path.insert(0, str(ROOT))
    from src.config import ProjectPaths
    from src.identity import IdentityError, build_identity

    parser = argparse.ArgumentParser(description="Generate the individual dataset.")
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    try:
        identity = build_identity(args.student_id, args.seed)
    except IdentityError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    paths = ProjectPaths(ROOT)
    info = generate_dataset(identity.seed, paths.csv_path(identity.reg_digits), paths.fingerprint_path)
    print(f"Verification code : {identity.verification_code}")
    print(f"Rows (with dups)  : {info['total_rows_with_duplicates']}")
    print(f"SHA-256           : {info['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
