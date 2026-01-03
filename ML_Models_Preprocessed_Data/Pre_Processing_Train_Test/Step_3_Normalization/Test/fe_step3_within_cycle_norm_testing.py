# fe_step3_within_cycle_norm.py
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import sys

# Columns expected from Step 2
REQ_COLS = [
    "group_id","spice","target",
    "sensor_index","heater_profile_step_index",
    "n_samples",
    "log_mean","log_std","log_median","log_min","log_max","log_p10","log_p90",
    "log_delta","log_slope_per_s"
]

# We will create relative features from these baseline columns
REL_BASE_COLS = ["log_mean","log_median","log_p10","log_p90"]

def safe_outpath(base: Path) -> Path:
    if not base.exists():
        return base
    i = 1
    while True:
        cand = base.with_name(f"{base.stem}_{i}{base.suffix}")
        if not cand.exists():
            return cand
        i += 1

def main(src: Path, out_dir: Path):
    # Make sure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)

    # Check if required columns are present
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Step-2 file: {missing}")

    # Make sure sensor_index and heater_profile_step_index are integers
    df["sensor_index"] = df["sensor_index"].astype(int)
    df["heater_profile_step_index"] = df["heater_profile_step_index"].astype(int)

    # Extract baseline values at heater step 0 for each group_id and sensor_index
    base = (
        df.loc[df["heater_profile_step_index"] == 0,
               ["group_id","sensor_index"] + REL_BASE_COLS]
        .rename(columns={c: f"base_{c}" for c in REL_BASE_COLS})
        .copy()
    )

    # Count how many pairs have missing step 0
    expected_pairs = df.groupby(["group_id","sensor_index"]).size().shape[0]
    have_base_pairs = base.groupby(["group_id","sensor_index"]).size().shape[0]
    if have_base_pairs < expected_pairs:
        missing_pairs = expected_pairs - have_base_pairs
        print(f"[WARN] {missing_pairs} (group_id,sensor) pairs lack step-0 baseline. "
              f"Relative features will be NaN for those pairs.", file=sys.stderr)

    # Merge baseline values back to the main dataframe
    merged = df.merge(base, on=["group_id","sensor_index"], how="left")

    # Create relative features by subtracting step 0 baseline in log space
    for c in REL_BASE_COLS:
        merged[f"{c}_rel"] = merged[c] - merged[f"base_{c}"]

    # Save the output
    out_path = safe_outpath(out_dir / f"{Path(src).stem}_step3_norm.csv")
    merged.to_csv(out_path, index=False)

    # Print a quick summary
    total_rows = len(merged)
    na_rel = merged[[f"{c}_rel" for c in REL_BASE_COLS]].isna().any(axis=1).sum()
    print(f"[OK] Wrote: {out_path}")
    print(f"[INFO] Rows: {total_rows}, rows with any *_rel = NaN (likely missing step-0): {na_rel}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 3: within-cycle baseline normalization by heater step 0")
    p.add_argument("--src", required=True, type=str, help="Path to Step-2 CSV")
    p.add_argument("--out_dir", required=True, type=str, help="Output directory for Step-3 CSV")
    args = p.parse_args()
    main(Path(args.src), Path(args.out_dir))
