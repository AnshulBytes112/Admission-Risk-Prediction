from pathlib import Path
import pandas as pd

path = Path("data/processed/features.parquet")

if not path.exists():
    raise SystemExit(f"ERROR: Parquet path not found: {path}")

df = pd.read_parquet(path)

print("\n=== PHASE 4 PARQUET VALIDATION ===")
print(f"Row count: {len(df)}")
print(f"Column count: {len(df.columns)}")

print("\n=== SCHEMA ===")
print(df.dtypes.to_string())

print("\n=== DATA QUALITY ===")
if "event_id" in df.columns:
    print(f"Missing event_id: {df['event_id'].isna().sum()}")
    print(f"Duplicate event_id values: {df['event_id'].duplicated().sum()}")
else:
    print("ERROR: event_id column is missing")

if "admission_within_24h" in df.columns:
    print("Target values:")
    print(df["admission_within_24h"].value_counts(dropna=False).sort_index().to_string())
    print(f"Missing target: {df['admission_within_24h'].isna().sum()}")
    print(f"Non-binary target values: {(~df['admission_within_24h'].isin([0, 1])).sum()}")
else:
    print("ERROR: admission_within_24h column is missing")

print("\n=== NULL COUNTS ===")
print(df.isna().sum().to_string())

print("\n=== SYNTHETIC DATA CHECK ===")
if "is_synthetic" in df.columns:
    print(df["is_synthetic"].value_counts(dropna=False).to_string())
elif "synthetic_data" in df.columns:
    print(df["synthetic_data"].value_counts(dropna=False).to_string())
elif "data_origin" in df.columns:
    print(df["data_origin"].value_counts(dropna=False).to_string())
else:
    print("WARNING: No recognizable synthetic-data marker found")

print("\n=== SAMPLE ROWS ===")
print(df.head(5).to_string(index=False))
