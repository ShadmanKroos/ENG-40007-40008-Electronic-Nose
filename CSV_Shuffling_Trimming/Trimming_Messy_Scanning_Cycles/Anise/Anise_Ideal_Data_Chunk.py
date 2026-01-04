import pandas as pd
from pathlib import Path

# === Configuration ===
# Change this path if needed:
INPUT_PATH = Path("Anise_Raw_Data_Semester2_reordered.csv")
OUTPUT_PATH = INPUT_PATH.with_name(INPUT_PATH.stem + "_perfect_only.csv")

CHUNK_SIZE = 400
EXPECTED_SENSOR = set(range(0, 8))   # 0..7
EXPECTED_HEATER = set(range(0, 10))  # 0..9
EXPECTED_CYCLE  = set(range(1, 6))   # 1..5

# Expected equal counts within a perfect 400-row chunk
# Each scanning_cycle_index should appear 80 times (8 sensors * 10 heater steps)
# Each heater_profile_step_index should appear 40 times (8 sensors * 5 cycles)
# Each sensor_index should appear 50 times (10 heater steps * 5 cycles)
EXPECTED_COUNTS = {
    "scanning_cycle_index": 80,
    "heater_profile_step_index": 40,
    "sensor_index": 50,
}

def is_perfect_chunk(chunk: pd.DataFrame) -> bool:
    """Validate a 400-row chunk as 'perfect' per the given loop criteria."""
    if len(chunk) != CHUNK_SIZE:
        return False
    
    # Check exact value sets
    if set(chunk["sensor_index"].unique()) != EXPECTED_SENSOR:
        return False
    if set(chunk["heater_profile_step_index"].unique()) != EXPECTED_HEATER:
        return False
    if set(chunk["scanning_cycle_index"].unique()) != EXPECTED_CYCLE:
        return False

    # Check equal counts per expected combination sizes
    vc_sensor = chunk["sensor_index"].value_counts(dropna=False)
    vc_heater = chunk["heater_profile_step_index"].value_counts(dropna=False)
    vc_cycle  = chunk["scanning_cycle_index"].value_counts(dropna=False)

    if not all(vc_sensor.get(v, 0) == EXPECTED_COUNTS["sensor_index"] for v in EXPECTED_SENSOR):
        return False
    if not all(vc_heater.get(v, 0) == EXPECTED_COUNTS["heater_profile_step_index"] for v in EXPECTED_HEATER):
        return False
    if not all(vc_cycle.get(v, 0) == EXPECTED_COUNTS["scanning_cycle_index"] for v in EXPECTED_CYCLE):
        return False

    return True

def main():
    df = pd.read_csv(INPUT_PATH)

    perfect_rows = 0
    n = len(df)

    # Walk forward chunk-by-chunk until the first imperfect chunk
    for start in range(0, n, CHUNK_SIZE):
        end = start + CHUNK_SIZE
        if end > n:
            # Partial chunk at the end is imperfect by definition
            break
        if is_perfect_chunk(df.iloc[start:end]):
            perfect_rows = end
        else:
            break

    # Everything after 'perfect_rows' is imperfect (as per your data organization)
    imperfect_start_idx = perfect_rows  # zero-based index into df
    imperfect_rows = n - perfect_rows

    # Write out only the perfect prefix
    df.iloc[:perfect_rows].to_csv(OUTPUT_PATH, index=False)

    # Report (CSV row numbers: header is row 1, first data row is row 2)
    if imperfect_rows > 0:
        csv_row_number_start = imperfect_start_idx + 2  # +1 to convert 0-based to 1-based data row, +1 for header
        print(f"Imperfect data begins at original CSV row: {csv_row_number_start} (header is row 1).")
        print(f"Total imperfect rows dropped: {imperfect_rows}.")
    else:
        print("No imperfect data detected. Entire file consists of perfect chunks.")

    print(f"Created new file without imperfect chunks: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
