# fe_step5_make_wide_table.py
# Purpose: Convert Step-3 normalized stepwise summaries into a wide per-cycle feature table,
# then merge Step-4 context features (temp_mean, rh_mean, pressure_mean).

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import sys

# Stats to extract from Step-3 for each (sensor_index, heater_profile_step_index)
STAT_COLS_ABS = [
    "log_mean","log_std","log_median","log_min","log_max","log_p10","log_p90",
    "log_delta","log_slope_per_s"
]
STAT_COLS_REL = [
    "log_mean_rel","log_median_rel","log_p10_rel","log_p90_rel"
]
COUNT_COL = "n_samples"

ID_COLS = ["group_id","spice","target"]
KEY_COLS = ID_COLS + ["sensor_index","heater_profile_step_index"]

def safe_outpath(base: Path) -> Path:
    if not base.exists():
        return base
    i = 1
    while True:
        cand = base.with_name(f"{base.stem}_{i}{base.suffix}")
        if not cand.exists():
            return cand
        i += 1

def make_colname(sensor_idx: int, step_idx: int, stat_name: str) -> str:
    return f"S{sensor_idx}_H{step_idx}_{stat_name}"

def main(src_step3: Path, src_ctx: Path, out_dir: Path):
    # Create output directory if needed
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load inputs
    df = pd.read_csv(src_step3)
    ctx = pd.read_csv(src_ctx)

    # Basic checks
    needed_cols = set(KEY_COLS + STAT_COLS_ABS + STAT_COLS_REL + [COUNT_COL])
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in Step-3 file: {missing}")

    missing_ctx = [c for c in ID_COLS if c not in ctx.columns]
    for c in ["temp_mean","rh_mean","pressure_mean"]:
        if c not in ctx.columns:
            missing_ctx.append(c)
    if missing_ctx:
        raise ValueError(f"Missing required columns in Step-4 context file: {missing_ctx}")

    # Ensure integer sensor and step indices
    df["sensor_index"] = df["sensor_index"].astype(int)
    df["heater_profile_step_index"] = df["heater_profile_step_index"].astype(int)

    # Sort is not required but makes iteration deterministic
    df = df.sort_values(KEY_COLS)

    # Build wide rows, one per (group_id, spice, target)
    rows = []
    expected_cells_per_cycle = None  # will compute once we see max sensor/step coverage

    for (gid, sp, tgt), g in df.groupby(ID_COLS, sort=False):
        entry = {"group_id": gid, "spice": sp, "target": tgt}

        # Fill absolute stats
        for _, r in g.iterrows():
            s = int(r["sensor_index"])
            h = int(r["heater_profile_step_index"])

            for stat in STAT_COLS_ABS:
                entry[make_colname(s, h, stat)] = r[stat]

            for stat in STAT_COLS_REL:
                entry[make_colname(s, h, stat)] = r[stat]

            entry[make_colname(s, h, "n")] = int(r[COUNT_COL])

        rows.append(entry)

        # Calculate expected cells the first time (sensors * steps)
        if expected_cells_per_cycle is None:
            # Count how many unique (sensor, step) pairs exist in this group
            expected_cells_per_cycle = g[["sensor_index","heater_profile_step_index"]].drop_duplicates().shape[0]

    wide = pd.DataFrame(rows)

    # Merge context features on (group_id, spice, target)
    final = wide.merge(ctx[ID_COLS + ["temp_mean","rh_mean","pressure_mean"]],
                       on=ID_COLS, how="left")

    # Quick validation and messages
    num_cycles = final.shape[0]
    num_feature_cols = final.shape[1] - len(ID_COLS)
    print(f"[INFO] Cycles (rows): {num_cycles}")
    print(f"[INFO] Feature columns (including context): {num_feature_cols}")

    # Warn if any cycle is missing some (sensor, step) cells
    # Heuristic: check that all S*_H*_n columns exist for all cycles
    n_cols = [c for c in final.columns if c.endswith("_n")]
    if len(n_cols) == 0:
        print("[WARN] No *_n columns found. Cannot verify cell counts.", file=sys.stderr)
    else:
        # Count NaNs in *_n columns
        nan_counts = final[n_cols].isna().sum(axis=1)
        bad = (nan_counts > 0).sum()
        if bad > 0:
            print(f"[WARN] {bad} cycle rows have missing sensor/step cells (NaN in *_n).", file=sys.stderr)

    # Build output path and write
    # If the Step-3 file ends with *_step3_norm.csv we can shorten the name; otherwise just append _features
    stem = Path(src_step3).stem
    if stem.endswith("_step3_norm"):
        stem = stem[:-len("_step3_norm")]
    out_path = safe_outpath(out_dir / f"{stem}_features.csv")
    final.to_csv(out_path, index=False)

    print(f"[OK] Wrote features: {out_path}")
    print(f"[INFO] Columns total: {final.shape[1]}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step 5: Make wide per-cycle features and merge context")
    p.add_argument("--summary", required=True, type=str, help="Path to Step-3 CSV (normalized stepwise)")
    p.add_argument("--context", required=True, type=str, help="Path to Step-4 context CSV")
    p.add_argument("--out_dir", required=True, type=str, help="Output directory for final features CSV")
    args = p.parse_args()
    main(Path(args.summary), Path(args.context), Path(args.out_dir))
