import pandas as pd
from collections import defaultdict, deque

# === Configuration ===
INPUT_CSV  = "Anise_Raw_Data_Semester2.csv"          # your original file (won't be overwritten)
OUTPUT_CSV = "Anise_Raw_Data_Semester2_reordered.csv"  # new file with corrected ordering

# === Load data ===
df = pd.read_csv(INPUT_CSV)

# Basic sanity checks
required_cols = {"sensor_index", "heater_profile_step_index", "scanning_cycle_index", "timestamp_since_poweron"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# We'll never modify cell values; only reorder rows.
# Keep a stable integer index to preserve original ordering when needed.
df = df.reset_index(drop=False).rename(columns={"index": "_orig_row"})
n = len(df)

# === Discover the canonical ascending sets used by the loop ===
# We assume the correct canonical order is ascending for each dimension, as in your sample dataset.
sensors = sorted(pd.unique(df["sensor_index"]))
heaters = sorted(pd.unique(df["heater_profile_step_index"]))
cycles  = sorted(pd.unique(df["scanning_cycle_index"]))

num_sensors = len(sensors)
num_heaters = len(heaters)
num_cycles_unique = len(cycles)

if num_sensors == 0 or num_heaters == 0 or num_cycles_unique == 0:
    raise ValueError("One of the loop dimensions has zero unique values; cannot proceed.")

rows_per_cycle = num_sensors * num_heaters

# === Generate the expected triple sequence for the entire file length ===
# Expected nested order (outer → inner): cycle -> heater -> sensor
expected = []
exp_count = 0
cycle_pos = 0
while exp_count < n:
    c = cycles[cycle_pos % num_cycles_unique]
    for h in heaters:
        for s in sensors:
            if exp_count >= n:
                break
            expected.append((c, h, s))
            exp_count += 1
    cycle_pos += 1

# === Find first deviation row in the original file ===
orig_triples = list(
    zip(df["scanning_cycle_index"], df["heater_profile_step_index"], df["sensor_index"])
)

first_mismatch_pos = None
for i, (obs, exp) in enumerate(zip(orig_triples, expected)):
    if obs != exp:
        first_mismatch_pos = i  # 0-based position
        break

# === Stable reconstruction of the "in-pattern" block ===
# Group original row indices by their (cycle, heater, sensor) triple, preserving original order.
buckets = defaultdict(deque)
for idx, triple in enumerate(orig_triples):
    buckets[triple].append(idx)

# Build the ordered "good" indices by walking the expected sequence
used = set()
good_indices = []
for triple in expected:
    if buckets[triple]:
        idx = buckets[triple].popleft()  # earliest unused row for this triple
        good_indices.append(idx)
        used.add(idx)
    else:
        # No row available for this expected triple (dataset may be incomplete here); skip
        # This simply shortens the good block a bit and leaves unmatched rows as outliers.
        continue

# Any indices not used are considered "out-of-pattern" and appended at the end (stable order)
outlier_indices = [i for i in range(n) if i not in used]

# === Assemble the final DataFrame: in-pattern first, then outliers ===
reordered = pd.concat([df.iloc[good_indices], df.iloc[outlier_indices]], axis=0)
# Drop helper column and reset index
reordered = reordered.drop(columns=["_orig_row"]).reset_index(drop=True)

# === Write new file (non-destructive) ===
reordered.to_csv(OUTPUT_CSV, index=False)

# === Console report ===
print("=== Loop Reconstruction Report ===")
print(f"Input rows: {n}")
print(f"Unique sensors: {num_sensors} -> {sensors}")
print(f"Unique heater steps: {num_heaters} -> {heaters}")
print(f"Unique scan cycles: {num_cycles_unique} -> {cycles}")
print(f"Rows per full cycle (sensors × heaters): {rows_per_cycle}")
print(f"In-pattern rows placed first: {len(good_indices)}")
print(f"Out-of-pattern rows moved to end: {len(outlier_indices)}")
if first_mismatch_pos is None:
    print("The original file was already in the expected nested order for its length.")
else:
    # Convert to 1-based row number to match spreadsheet conventions
    print(f"First deviation from the pattern starts at original row: {first_mismatch_pos + 1}")

# Optional: quick timestamp sanity check on the "good" block
good_ts = reordered.loc[:len(good_indices)-1, "timestamp_since_poweron"].values
nonmonotonic = (good_ts[1:] < good_ts[:-1]).any() if len(good_ts) > 1 else False
if nonmonotonic:
    print("WARNING: Within the in-pattern block, timestamp_since_poweron is not strictly nondecreasing.")
else:
    print("Timestamp alignment check: in-pattern block appears time-consistent (nondecreasing).")

print(f"\nWrote reordered file to: {OUTPUT_CSV}")
