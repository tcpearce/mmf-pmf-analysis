import pandas as pd
import numpy as np

print("=== DEMONSTRATING TIME INDEXING ISSUE ===")

# Load and filter data
df_raw = pd.read_parquet('mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet')
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

start_date = pd.to_datetime('2023-10-01')
end_date = pd.to_datetime('2023-10-05 23:59:59')
df_filtered = df_raw[(df_raw['datetime'] >= start_date) & (df_raw['datetime'] <= end_date)].copy()

# PMF species (corrected)
particle_species = ['PM1 FIDAS', 'PM2.5 FIDAS', 'PM4 FIDAS', 'PM10 FIDAS', 'TSP FIDAS']
voc_species = ['Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene']
inorganic_species = ['NO', 'NO2', 'SO2']
pmf_species = particle_species + voc_species + inorganic_species

# Get actual timestamps where concentration data exists
concentration_with_time = df_filtered[pmf_species + ['datetime']].dropna()
print(f"Concentration data: {len(concentration_with_time)} rows")
print(f"Time range: {concentration_with_time['datetime'].min()} to {concentration_with_time['datetime'].max()}")

# Get pressure data for this dataset
pressure_df = df_filtered.copy()
pressure_df['datetime'] = pd.to_datetime(pressure_df['datetime'])
pressure_indexed = pressure_df.set_index('datetime')['Pressure'].sort_index()

print(f"\nPressure data points: {pressure_indexed.notna().sum()}")
print(f"Pressure range: {pressure_indexed.min():.2f} - {pressure_indexed.max():.2f} hPa")

# PROBLEM: Current PMF logic (creates sparse time index)
print(f"\n=== PROBLEM: CURRENT PMF LOGIC ===")
start_time = pressure_df['datetime'].min()
end_time = pressure_df['datetime'].max()
print(f"Full time range: {start_time} to {end_time}")
print(f"Duration: {end_time - start_time}")

# This creates only 129 time points across 5 days!
idx_sparse = pd.date_range(start=start_time, end=end_time, periods=len(concentration_with_time))
print(f"Created {len(idx_sparse)} time points")
print(f"Time step between points: {idx_sparse[1] - idx_sparse[0]}")

# Interpolate pressure to this sparse grid
p_on_sparse = pressure_indexed.reindex(idx_sparse).interpolate(method='time', limit_direction='both')
print(f"Interpolated pressure valid points: {p_on_sparse.notna().sum()}")
print(f"Interpolated pressure range: {p_on_sparse.min():.2f} - {p_on_sparse.max():.2f} hPa")
print(f"Unique interpolated values: {len(p_on_sparse.dropna().unique())}")

if len(p_on_sparse.dropna().unique()) == 1:
    print(f"🚨 RESULT: Constant pressure = {p_on_sparse.dropna().iloc[0]:.2f} hPa")

# SOLUTION: Use actual concentration timestamps
print(f"\n=== SOLUTION: USE ACTUAL CONCENTRATION TIMESTAMPS ===")
concentration_timestamps = concentration_with_time['datetime'].values
print(f"Using {len(concentration_timestamps)} actual timestamps")

# Interpolate pressure to the actual concentration timestamps
p_on_actual = pressure_indexed.reindex(pd.to_datetime(concentration_timestamps)).interpolate(method='time', limit_direction='both')
print(f"Interpolated pressure valid points: {p_on_actual.notna().sum()}")
if p_on_actual.notna().sum() > 0:
    print(f"Interpolated pressure range: {p_on_actual.min():.2f} - {p_on_actual.max():.2f} hPa")
    print(f"Unique interpolated values: {len(p_on_actual.dropna().unique())}")
    
    if len(p_on_actual.dropna().unique()) == 1:
        print(f"🚨 STILL CONSTANT: {p_on_actual.dropna().iloc[0]:.2f} hPa")
    else:
        print(f"✅ PRESSURE VARIES: {len(p_on_actual.dropna().unique())} unique values")

# Check actual time gaps in concentration data
time_diffs = pd.Series(concentration_timestamps).diff().dropna()
print(f"\nTime gaps in concentration data:")
print(f"  Min gap: {time_diffs.min()}")
print(f"  Max gap: {time_diffs.max()}")
print(f"  Mean gap: {time_diffs.mean()}")

# Check time gaps in pressure data
pressure_timestamps = pressure_indexed.dropna().index
pressure_time_diffs = pd.Series(pressure_timestamps).diff().dropna()
print(f"\nTime gaps in pressure data:")
print(f"  Min gap: {pressure_time_diffs.min()}")
print(f"  Max gap: {pressure_time_diffs.max()}")
print(f"  Mean gap: {pressure_time_diffs.mean()}")

# Alternative approach: Use denser time grid
print(f"\n=== ALTERNATIVE: USE DENSER TIME GRID ===")
# Create 15-minute resolution grid for the concentration time range
start_conc = concentration_with_time['datetime'].min()
end_conc = concentration_with_time['datetime'].max()
idx_dense = pd.date_range(start=start_conc, end=end_conc, freq='15T')
print(f"Dense grid: {len(idx_dense)} points at 15-minute resolution")

p_on_dense = pressure_indexed.reindex(idx_dense).interpolate(method='time', limit_direction='both')
print(f"Dense grid pressure valid points: {p_on_dense.notna().sum()}")
if p_on_dense.notna().sum() > 0:
    print(f"Dense grid pressure range: {p_on_dense.min():.2f} - {p_on_dense.max():.2f} hPa")
    print(f"Dense grid unique values: {len(p_on_dense.dropna().unique())}")
    
    if len(p_on_dense.dropna().unique()) > 1:
        print(f"✅ DENSE GRID WORKS: Pressure varies")
        
        # Now downsample to concentration timestamps
        p_final = p_on_dense.reindex(pd.to_datetime(concentration_timestamps)).interpolate(method='time')
        print(f"Final downsampled pressure range: {p_final.min():.2f} - {p_final.max():.2f} hPa")
        print(f"Final unique values: {len(p_final.dropna().unique())}")
    else:
        print(f"🚨 DENSE GRID STILL CONSTANT")