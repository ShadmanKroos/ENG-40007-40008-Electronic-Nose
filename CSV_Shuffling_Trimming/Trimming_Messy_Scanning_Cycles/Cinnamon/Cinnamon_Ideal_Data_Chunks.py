import pandas as pd
from pathlib import Path

# === Change ONLY this path per spice ===
INPUT_PATH = Path("Cinnamon_Sem_Two_Recorded_reordered.csv")
# ======================================

OUTPUT_PATH = INPUT_PATH.with_name(INPUT_PATH.stem + "_perfect_only.csv")

CHUNK_SIZE = 400
EXPECTED_SENSOR = set(range(0, 8))   # 0..7
EXPECTED_HEATER = set(range(0, 10))  # 0..9
EXPECTED_CYCLE  = set(range(1, 6))   # 1..5

EXPECTED_COUNTS = {
    "scanning_cycle_index": 80,   # 8 sensors * 10 heaters
    "heater_profile_step_index": 40,  # 8 sensors * 5 cycles
    "sensor_index": 50,           # 10 heaters * 5 cycles
}

def is_perfect_chunk(chunk: pd.DataFrame) -> bool:
    if len(chunk) != CHUNK_SIZE:
        return False
    if set(chunk["sensor_index"].unique()) != EXPECTED_SENSOR:
        return False
    if set(chunk["heater_profile_step_index"].unique()) != EXPECTED_HEATER:
        return False
    if set(chunk["scanning_cycle_index"].unique()) != EXPECTED_CYCLE:
        return False

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

    for start in range(0, n, CHUNK_SIZE):
        end = start + CHUNK_SIZE
        if end > n:
            break
        if is_perfect_chunk(df.iloc[start:end]):
            perfect_rows = end
        else:
            break

    imperfect_start_idx = perfect_rows
    imperfect_rows = n - perfect_rows

    df.iloc[:perfect_rows].to_csv(OUTPUT_PATH, index=False)

    if imperfect_rows > 0:
        csv_row_number_start = imperfect_start_idx + 2  # header is row 1
        print(f"Imperfect data begins at original CSV row: {csv_row_number_start} (header is row 1).")
        print(f"Total imperfect rows dropped: {imperfect_rows}.")
    else:
        print("No imperfect data detected. Entire file consists of perfect chunks.")

    print(f"Created new file without imperfect chunks: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
