import pandas as pd
import numpy as np

# Load the raw parquet data
print("=== BASIC PRESSURE DATA CHECK ===")
df = pd.read_parquet('mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet')
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"Total records in parquet: {len(df)}")
print(f"Pressure column exists: {'Pressure' in df.columns}")

# Check all pressure data (not filtered)
pressure_all = df['Pressure']
pressure_valid_all = pressure_all.dropna()

print(f"\nALL PRESSURE DATA:")
print(f"Total pressure measurements: {len(pressure_valid_all)}")
print(f"Pressure range: {pressure_valid_all.min():.2f} - {pressure_valid_all.max():.2f} hPa")
print(f"Pressure std dev: {pressure_valid_all.std():.6f}")
print(f"Unique pressure values: {len(pressure_valid_all.unique())}")

print(f"\nFirst 20 pressure values:")
print(pressure_valid_all.head(20).tolist())

print(f"\nLast 20 pressure values:")
print(pressure_valid_all.tail(20).tolist())

# Check the specific Oct 1-2 period
print(f"\n=== OCTOBER 1-2, 2023 PERIOD ===")
start_time = pd.to_datetime('2023-10-01')
end_time = pd.to_datetime('2023-10-02')
oct_data = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)].copy()

print(f"Records in Oct 1-2 period: {len(oct_data)}")

pressure_oct = oct_data['Pressure']
pressure_valid_oct = pressure_oct.dropna()

print(f"Valid pressure points in Oct 1-2: {len(pressure_valid_oct)}")
if len(pressure_valid_oct) > 0:
    print(f"Pressure range: {pressure_valid_oct.min():.2f} - {pressure_valid_oct.max():.2f} hPa")
    print(f"Pressure std dev: {pressure_valid_oct.std():.6f}")
    print(f"Unique pressure values: {len(pressure_valid_oct.unique())}")
    
    print(f"\nAll pressure values in Oct 1-2:")
    print(pressure_valid_oct.tolist())
    
    print(f"\nPressure data with timestamps:")
    pressure_with_time = oct_data[oct_data['Pressure'].notna()][['datetime', 'Pressure']]
    print(pressure_with_time)
    
    # Check if all values are identical
    if len(pressure_valid_oct.unique()) == 1:
        print(f"\n🚨 PROBLEM FOUND: All pressure values are identical: {pressure_valid_oct.iloc[0]} hPa")
    else:
        print(f"\n✅ Pressure varies: {len(pressure_valid_oct.unique())} unique values")
else:
    print("🚨 NO PRESSURE DATA FOUND in Oct 1-2 period!")

# Check other periods to see if pressure varies elsewhere
print(f"\n=== CHECKING OTHER PERIODS FOR PRESSURE VARIATION ===")
for month in ['2023-09', '2023-11', '2024-01']:
    month_data = df[df['datetime'].dt.to_period('M') == month]
    if len(month_data) > 0:
        month_pressure = month_data['Pressure'].dropna()
        if len(month_pressure) > 0:
            print(f"{month}: {len(month_pressure)} points, range: {month_pressure.min():.2f} - {month_pressure.max():.2f} hPa, unique: {len(month_pressure.unique())}")
        else:
            print(f"{month}: No pressure data")
    else:
        print(f"{month}: No data for this period")