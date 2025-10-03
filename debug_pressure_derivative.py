import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the pressure data
df = pd.read_parquet('mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet')
df['datetime'] = pd.to_datetime(df['datetime'])

# Filter to Oct 1-5, 2023
start_time = pd.to_datetime('2023-10-01')
end_time = pd.to_datetime('2023-10-05')
oct_data = df[(df['datetime'] >= start_time) & (df['datetime'] <= end_time)].copy()

print(f"Total Oct 1-5 records: {len(oct_data)}")
pressure_data = oct_data.set_index('datetime')['Pressure'].sort_index()
pressure_valid = pressure_data.dropna()

print(f"Valid pressure points: {len(pressure_valid)}")
print(f"Pressure range: {pressure_valid.min():.2f} - {pressure_valid.max():.2f} hPa")

# Show the time spacing of valid pressure data
if len(pressure_valid) > 1:
    time_diffs = np.diff(pressure_valid.index)
    time_diffs_minutes = [pd.Timedelta(td).total_seconds()/60 for td in time_diffs]
    print(f"Time differences between pressure measurements:")
    print(f"  Min: {min(time_diffs_minutes):.1f} min")
    print(f"  Max: {max(time_diffs_minutes):.1f} min")  
    print(f"  Mean: {np.mean(time_diffs_minutes):.1f} min")
    print(f"  Median: {np.median(time_diffs_minutes):.1f} min")

print(f"\nFirst 10 pressure measurements:")
print(pressure_valid.head(10))

# Test the problematic interpolation approach
print(f"\n=== TESTING CURRENT APPROACH ===")
# This is what the PMF code does
n_concentration_points = 1123  # From PMF analysis
idx_regular = pd.date_range(start=start_time, end=end_time, periods=n_concentration_points)
time_step_minutes = (idx_regular[1] - idx_regular[0]).total_seconds() / 60
print(f"Regular grid: {len(idx_regular)} points, {time_step_minutes:.2f} min intervals")

# Interpolate pressure to regular grid
p_interpolated = pressure_data.reindex(idx_regular).interpolate(method='time', limit_direction='both')
p_interp_valid = p_interpolated.dropna()
print(f"After interpolation: {len(p_interp_valid)} valid points")

# Calculate derivatives on regular grid
dpdt_regular = p_interpolated.diff() / (time_step_minutes / 60)  # hPa/hr
dpdt_valid = dpdt_regular.dropna()
unique_values = len(dpdt_valid.unique())
print(f"Unique derivative values: {unique_values}")
print(f"Derivative range: {dpdt_valid.min():.6f} - {dpdt_valid.max():.6f} hPa/hr")

# Show why derivatives are so limited
print(f"\nFirst 10 interpolated pressures:")
print(p_interpolated.head(10))
print(f"\nFirst 10 derivatives:")
print(dpdt_regular.head(10))

# Better approach: calculate derivatives on actual pressure measurements
print(f"\n=== BETTER APPROACH ===")
dpdt_actual = pressure_valid.diff()  # Pressure differences
time_diffs_hr = [pd.Timedelta(td).total_seconds()/3600 for td in np.diff(pressure_valid.index)]
dpdt_actual_rates = dpdt_actual.iloc[1:] / time_diffs_hr  # hPa/hr

print(f"Derivatives on actual measurements:")
print(f"  Count: {len(dpdt_actual_rates)}")
print(f"  Range: {dpdt_actual_rates.min():.6f} - {dpdt_actual_rates.max():.6f} hPa/hr")
print(f"  Unique values: {len(dpdt_actual_rates.unique())}")
print(f"  Std dev: {dpdt_actual_rates.std():.6f} hPa/hr")