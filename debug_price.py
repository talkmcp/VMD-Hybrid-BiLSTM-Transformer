"""Debug: ตรวจสอบว่า rss3_FOBm1 มีค่าจริงหรือไม่"""
import pandas as pd
import numpy as np

BASE_DIR = r'C:\Users\montc\Dropbox\PythonProj\Forecasting'
RAW_FILE = BASE_DIR + r'\rubber_data_combined_Update.xlsx'

df = pd.read_excel(RAW_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

print("Columns in raw file:")
print(df.columns.tolist())
print()

# Stage 3 test period
STAGE3_START = '2018-05-07'
TRAIN_RATIO  = 0.8
VAL_RATIO    = 0.1

s3 = df[(df['date'] >= STAGE3_START)].copy().reset_index(drop=True)
n  = len(s3)
n_val = int(n * (TRAIN_RATIO + VAL_RATIO))
test_s3 = s3.iloc[n_val:].copy().reset_index(drop=True)

print(f"Stage 3 rows: {n}, test start index: {n_val}")
print(f"Test rows: {len(test_s3)}")
print(f"Test date range: {test_s3['date'].iloc[0]} → {test_s3['date'].iloc[-1]}")
print()

col = 'rss3_FOBm1'
if col in test_s3.columns:
    print(f"'{col}' in test_s3: YES")
    print(f"  NaN count: {test_s3[col].isna().sum()} / {len(test_s3)}")
    print(f"  Sample values: {test_s3[col].dropna().values[:5]}")
    print(f"  Range: [{test_s3[col].min():.2f}, {test_s3[col].max():.2f}]")
else:
    print(f"'{col}' NOT in test_s3!")
    print("Available columns:", [c for c in test_s3.columns if 'rss3' in c.lower()])

print()
print("--- Direct from raw df ---")
mask = (df['date'] >= test_s3['date'].iloc[0]) & (df['date'] <= test_s3['date'].iloc[-1])
sub  = df[mask]
print(f"Rows matching: {len(sub)}")
if col in sub.columns:
    print(f"'{col}' NaN: {sub[col].isna().sum()} / {len(sub)}")
    print(f"Sample: {sub[col].dropna().values[:5]}")
