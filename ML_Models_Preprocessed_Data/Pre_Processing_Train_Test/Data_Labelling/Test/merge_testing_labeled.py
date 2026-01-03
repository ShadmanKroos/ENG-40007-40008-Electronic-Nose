from pathlib import Path
import pandas as pd

def merge_labeled_files(src_dir: Path, out_path: Path):
    # Find all labeled CSVs in the source directory
    files = list(src_dir.glob("*_labeled.csv"))
    if not files:
        raise FileNotFoundError(f"No labeled CSV files found in {src_dir}")

    print(f"Found {len(files)} labeled files:")
    for f in files:
        print(" -", f.name)

    # Load and concatenate
    dfs = [pd.read_csv(f) for f in files]
    master = pd.concat(dfs, ignore_index=True)

    # Save merged file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)

    print(f"\n[OK] Merged dataset written to: {out_path}")
    print(f"[INFO] Shape: {master.shape[0]} rows × {master.shape[1]} columns")

if __name__ == "__main__":
    # Adjust paths as needed
    src_dir = Path("../labeled")         # folder with your 4 test-labeled CSVs
    out_path = Path("../labeled/master_testing_labeled.csv")

    merge_labeled_files(src_dir, out_path)
