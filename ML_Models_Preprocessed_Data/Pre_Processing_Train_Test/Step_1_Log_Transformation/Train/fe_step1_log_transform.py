# fe_step1_log_transform.py
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

REQ_COLS = [
    "group_id","spice","target",
    "sensor_index","heater_profile_step_index","scanning_cycle_index",
    "timestamp_since_poweron","resistance_gassensor"
]

def safe_outpath(base: Path) -> Path:
    if not base.exists(): return base
    i = 1
    while True:
        cand = base.with_name(f"{base.stem}_{i}{base.suffix}")
        if not cand.exists(): return cand
        i += 1

def main(src: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Add log1p(resistance)
    df["log_resistance"] = np.log1p(df["resistance_gassensor"].astype(float))

    # Sort for deterministic per-step slope calculation later
    df = df.sort_values(["group_id","sensor_index","heater_profile_step_index","timestamp_since_poweron"])

    out_path = safe_outpath(out_dir / f"{src.stem}_step1_log.csv")
    df.to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path}  (rows={len(df)})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step1: add log_resistance and sort")
    p.add_argument("--src", required=True, type=str, help="Path to master labeled CSV")
    p.add_argument("--out_dir", required=True, type=str, help="Output directory for step1 CSV")
    args = p.parse_args()
    main(Path(args.src), Path(args.out_dir))
