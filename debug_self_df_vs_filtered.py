import pandas as pd
import numpy as np

# Load raw data
print("=== CHECKING WHAT SELF.DF CONTAINS VS FILTERED DATA ===")

# This mimics what PMF loads initially
df_raw = pd.read_parquet('mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet')
df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])

print(f"Raw data loaded: {len(df_raw)} records")

# This mimics the date filtering done during PMF analysis
start_date = pd.to_datetime('2023-10-01')
end_date = pd.to_datetime('2023-10-05 23:59:59')
df_filtered = df_raw[(df_raw['datetime'] >= start_date) & (df_raw['datetime'] <= end_date)].copy()

print(f"After date filtering: {len(df_filtered)} records")

# Check pressure in raw data
pressure_raw = df_raw['Pressure'].dropna()
print(f"\nPRESSURE IN RAW DATA:")
print(f"  Points: {len(pressure_raw)}")
print(f"  Range: {pressure_raw.min():.2f} - {pressure_raw.max():.2f} hPa")
print(f"  Unique values: {len(pressure_raw.unique())}")

# Check pressure in filtered data
pressure_filtered = df_filtered['Pressure'].dropna()
print(f"\nPRESSURE IN FILTERED DATA:")
print(f"  Points: {len(pressure_filtered)}")
print(f"  Range: {pressure_filtered.min():.2f} - {pressure_filtered.max():.2f} hPa")
print(f"  Unique values: {len(pressure_filtered.unique())}")

# Now check what happens with concentration_data filtering
# (this is what's likely happening to create the discrepancy)
particle_species = ['PM1 FIDAS', 'PM2.5 FIDAS', 'PM4 FIDAS', 'PM10 FIDAS', 'TSP FIDAS']
voc_species = ['Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene']
inorganic_species = ['NO', 'NO2', 'SO2']
pmf_species = particle_species + voc_species + inorganic_species

# This mimics what creates concentration_data (PMF input matrix)
concentration_data = df_filtered[pmf_species].dropna()
print(f"\nCONCENTRATION DATA (what goes to PMF):")
print(f"  Rows: {len(concentration_data)}")

# The issue: PMF pressure analysis uses df (all data) but interpolates to match concentration_data length
# Let's see what the discrepancy might be

# When PMF does pressure analysis, it uses:
# 1. self.df (which could be raw data or filtered data) 
# 2. Creates time index with length of concentration_data
# 3. But the time range comes from self.df, not concentration_data

# Let's simulate this exact logic:
print(f"\n=== SIMULATING PMF PRESSURE ANALYSIS LOGIC ===")

# Case 1: If self.df is the raw data (not filtered)
pressure_df_raw = df_raw.copy()
pressure_df_raw['datetime'] = pd.to_datetime(pressure_df_raw['datetime'])
pressure_indexed_raw = pressure_df_raw.set_index('datetime')['Pressure'].sort_index()

start_time_raw = pressure_df_raw['datetime'].min()
end_time_raw = pressure_df_raw['datetime'].max()

print(f"If using RAW data for pressure:")
print(f"  Time range: {start_time_raw} to {end_time_raw}")
print(f"  Creating index with {len(concentration_data)} periods")

idx_raw = pd.date_range(start=start_time_raw, end=end_time_raw, periods=len(concentration_data))
p_on_idx_raw = pressure_indexed_raw.reindex(idx_raw).interpolate(method='time', limit_direction='both')

print(f"  Result - Pressure range: {p_on_idx_raw.min():.2f} - {p_on_idx_raw.max():.2f} hPa")
print(f"  Result - Unique values: {len(p_on_idx_raw.dropna().unique())}")

# Case 2: If self.df is filtered data  
pressure_df_filtered = df_filtered.copy()
pressure_df_filtered['datetime'] = pd.to_datetime(pressure_df_filtered['datetime'])
pressure_indexed_filtered = pressure_df_filtered.set_index('datetime')['Pressure'].sort_index()

start_time_filtered = pressure_df_filtered['datetime'].min()
end_time_filtered = pressure_df_filtered['datetime'].max()

print(f"\nIf using FILTERED data for pressure:")
print(f"  Time range: {start_time_filtered} to {end_time_filtered}")
print(f"  Creating index with {len(concentration_data)} periods")

idx_filtered = pd.date_range(start=start_time_filtered, end=end_time_filtered, periods=len(concentration_data))
p_on_idx_filtered = pressure_indexed_filtered.reindex(idx_filtered).interpolate(method='time', limit_direction='both')

print(f"  Result - Pressure range: {p_on_idx_filtered.min():.2f} - {p_on_idx_filtered.max():.2f} hPa")
print(f"  Result - Unique values: {len(p_on_idx_filtered.dropna().unique())}")

# Check if either case results in constant pressure
print(f"\n=== DIAGNOSIS ===")
if len(p_on_idx_raw.dropna().unique()) == 1:
    print(f"🚨 RAW case produces CONSTANT pressure: {p_on_idx_raw.dropna().iloc[0]:.2f} hPa")
if len(p_on_idx_filtered.dropna().unique()) == 1:
    print(f"🚨 FILTERED case produces CONSTANT pressure: {p_on_idx_filtered.dropna().iloc[0]:.2f} hPa")

# Let's also check the actual datetime range of concentration_data indices
# The issue might be that concentration_data corresponds to specific timestamps, 
# not the full range of self.df
concentration_with_time = df_filtered[pmf_species + ['datetime']].dropna()
concentration_time_range = (concentration_with_time['datetime'].min(), concentration_with_time['datetime'].max())

print(f"\nCONCENTRATION DATA actual time range:")
print(f"  {concentration_time_range[0]} to {concentration_time_range[1]}")

# Check pressure in this specific time window
pressure_in_conc_window = df_filtered[
    (df_filtered['datetime'] >= concentration_time_range[0]) & 
    (df_filtered['datetime'] <= concentration_time_range[1])
]['Pressure'].dropna()

print(f"  Pressure in concentration time window:")
print(f"    Points: {len(pressure_in_conc_window)}")
if len(pressure_in_conc_window) > 0:
    print(f"    Range: {pressure_in_conc_window.min():.2f} - {pressure_in_conc_window.max():.2f} hPa")
    print(f"    Unique values: {len(pressure_in_conc_window.unique())}")
    if len(pressure_in_conc_window.unique()) == 1:
        print(f"    🚨 CONSTANT VALUE: {pressure_in_conc_window.iloc[0]:.2f} hPa")
else:
    print(f"    🚨 NO PRESSURE DATA IN CONCENTRATION TIME WINDOW!")