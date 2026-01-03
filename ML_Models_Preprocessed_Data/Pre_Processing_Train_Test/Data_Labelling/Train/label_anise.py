from pathlib import Path
import pandas as pd
import json
import argparse

SPICE = "Anise"
LABEL_MAP = {"Anise": 0, "Chilli": 1, "Cinnamon": 2, "Nutmeg": 3}
DEFAULT_OUT_DIR = Path("../labeled")

def safe_outpath(base: Path) -> Path:
    if not base.exists():
        return base
    i = 1
    while True:
        cand = base.with_name(f"{base.stem}_{i}{base.suffix}")
        if not cand.exists():
            return cand
        i += 1

def main(src: Path, out_dir: Path = DEFAULT_OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)

    df["spice"]  = SPICE
    df["target"] = LABEL_MAP[SPICE]
    if "scanning_cycle_index" in df.columns:
        df["group_id"] = df["scanning_cycle_index"].apply(lambda c: f"{SPICE}_cycle_{int(c)}")
    else:
        df["group_id"] = f"{SPICE}_file"

    out_path = safe_outpath(out_dir / f"{src.stem}_labeled.csv")
    df.to_csv(out_path, index=False)

    (out_dir / "label_mapping.json").write_text(json.dumps(LABEL_MAP, indent=2))
    print(f"[OK] Labeled file: {out_path}")
    print(f"[OK] Label mapping: {(out_dir / 'label_mapping.json')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Anise dataset")
    parser.add_argument("--src", type=str, required=True, help="Path to Anise CSV (raw)")
    args = parser.parse_args()
    main(Path(args.src))
