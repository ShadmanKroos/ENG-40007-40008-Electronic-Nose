# fe_step2_stepwise_summaries.py
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

REQ_COLS = [
    "group_id","spice","target",
    "sensor_index","heater_profile_step_index",
    "timestamp_since_poweron","log_resistance"
]

def safe_outpath(base: Path) -> Path:
    if not base.exists(): return base
    i = 1
    while True:
        cand = base.with_name(f"{base.stem}_{i}{base.suffix}")
        if not cand.exists(): return cand
        i += 1

def per_group_stats(g: pd.DataFrame) -> pd.Series:
    # NEW: enforce sort by timestamp inside the group for safety
    g = g.sort_values("timestamp_since_poweron", kind="mergesort")

    lr = g["log_resistance"].astype(float).to_numpy()
    ts = g["timestamp_since_poweron"].astype(float).to_numpy()

    n = lr.size
    q10 = np.nanpercentile(lr, 10)
    q90 = np.nanpercentile(lr, 90)

    first, last = lr[0], lr[-1]
    dt_ms = ts[-1] - ts[0]
    slope_per_s = (last - first) / (dt_ms/1000.0) if dt_ms != 0 else np.nan
    delta = last - first

    return pd.Series({
        "n_samples": n,
        "log_mean":   np.nanmean(lr),
        "log_std":    np.nanstd(lr, ddof=1) if n>1 else 0.0,
        "log_median": np.nanmedian(lr),
        "log_min":    np.nanmin(lr),
        "log_max":    np.nanmax(lr),
        "log_p10":    q10,
        "log_p90":    q90,
        "log_delta":  delta,
        "log_slope_per_s": slope_per_s
    })

def main(src: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)

    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keys = ["group_id","spice","target","sensor_index","heater_profile_step_index"]
    agg = df.groupby(keys, sort=False).apply(per_group_stats).reset_index()

    out_path = safe_outpath(out_dir / f"{src.stem}_step2_stepwise.csv")
    agg.to_csv(out_path, index=False)
    print(f"[OK] Wrote: {out_path}  (rows={len(agg)})")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Step2: per-step summaries of log_resistance")
    p.add_argument("--src", required=True, type=str, help="Path to Step1 CSV")
    p.add_argument("--out_dir", required=True, type=str, help="Output directory for step2 CSV")
    args = p.parse_args()
    main(Path(args.src), Path(args.out_dir))
