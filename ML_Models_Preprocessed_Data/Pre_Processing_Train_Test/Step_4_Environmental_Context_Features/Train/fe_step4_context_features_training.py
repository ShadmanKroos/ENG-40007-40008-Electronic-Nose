# fe_step4_context_features.py
# Purpose: Compute per-cycle context features from the master labeled file.
# Context features are the mean temperature, mean relative humidity, and mean pressure
# for each cycle identified by group_id. We keep spice and target for alignment.

from pathlib import Path
import argparse
import pandas as pd

# Required columns in the master labeled file
REQ_COLS = [
    "group_id", "spice", "target",
    "temperature", "relative_humidity", "pressure"
]

def safe_outpath(base: Path) -> Path:
    # If base does not exist, use it
    if not base.exists():
        return base
    # Otherwise, append a numeric suffix to avoid overwrite
    i = 1
    while True:
        cand = base.with_name(f"{base.stem}_{i}{base.suffix}")
        if not cand.exists():
            return cand
        i += 1

def main(src: Path, out_dir: Path):
    # Create output directory if needed
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the master labeled file (training or testing)
    df = pd.read_csv(src)

    # Check for required columns
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in master labeled file: {missing}")

    # Group by cycle using group_id, and keep spice and target for alignment
    # Compute per-cycle means of temperature, relative_humidity, and pressure
    ctx = (
        df.groupby(["group_id", "spice", "target"], as_index=False)
          .agg(
              temp_mean=("temperature", "mean"),
              rh_mean=("relative_humidity", "mean"),
              pressure_mean=("pressure", "mean")
          )
    )

    # Write the context features file next to the master input
    out_path = safe_outpath(out_dir / f"{Path(src).stem}_step4_context.csv")
    ctx.to_csv(out_path, index=False)

    # Print a small summary
    print(f"[OK] Wrote: {out_path}")
    print(f"[INFO] Rows (cycles): {len(ctx)}")
    # Optional: show a couple of lines to confirm structure
    print(ctx.head(3).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: per-cycle context features (temperature, RH, pressure means)")
    parser.add_argument("--src", required=True, type=str, help="Path to master labeled CSV (training or testing)")
    parser.add_argument("--out_dir", required=True, type=str, help="Output directory for Step-4 CSV")
    args = parser.parse_args()
    main(Path(args.src), Path(args.out_dir))
