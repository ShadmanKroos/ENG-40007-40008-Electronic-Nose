import json
import pandas as pd

# Update these filenames if your paths differ
input_file = 'Anise_Raw_Data.txt'
output_file = 'Anise_Raw_Data_Semester2.csv'

# 1. Load the raw JSON‐formatted text file
with open(input_file, 'r') as f:
    payload = json.load(f)

# 2. Extract column names from the metadata
columns = [col['key'] for col in payload['rawDataBody']['dataColumns']]

# 3. Extract the data block (list of rows)
data_rows = payload['rawDataBody']['dataBlock']

# 4. Build a DataFrame and export to CSV
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(output_file, index=False)

# 5. Print a quick summary
print(f"Wrote {df.shape[0]} rows and {df.shape[1]} columns to '{output_file}'")
