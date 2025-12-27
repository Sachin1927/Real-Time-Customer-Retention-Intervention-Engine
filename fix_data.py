import pandas as pd
from pathlib import Path

# 1. Point to your CSV file
file_path = Path('data/raw/telco_churn_simulated.csv')

# 2. Load the data
df = pd.read_csv(file_path)

# 3. Add the missing Timestamp columns
# We set the time to "Now" so Feast knows this data is current
print("Adding timestamps...")
df['event_timestamp'] = pd.to_datetime('today')
df['created_timestamp'] = pd.to_datetime('today')

# 4. Overwrite the file
df.to_csv(file_path, index=False)

print(f"Success! Saved {len(df)} rows with timestamps to {file_path}")
print("Columns are now:", df.columns.tolist())