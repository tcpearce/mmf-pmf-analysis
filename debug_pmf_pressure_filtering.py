import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Reproduce the exact same filtering logic as PMF analysis
print("=== REPRODUCING PMF FILTERING LOGIC ===")

# Load data exactly like PMF does
df = pd.read_parquet('mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet')
df['datetime'] = pd.to_datetime(df['datetime'])

# Apply the same date filter as PMF
start_date = pd.to_datetime('2023-10-01')
end_date = pd.to_datetime('2023-10-05 23:59:59')
df_filtered = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)].copy()

print(f"After date filtering: {len(df_filtered)} records")

# Check pressure in the filtered data
pressure_filtered = df_filtered['Pressure'].dropna()
print(f"Valid pressure points after filtering: {len(pressure_filtered)}")
print(f"Pressure range after filtering: {pressure_filtered.min():.2f} - {pressure_filtered.max():.2f} hPa")
print(f"Unique pressure values after filtering: {len(pressure_filtered.unique())}")
print(f"Pressure std dev after filtering: {pressure_filtered.std():.6f}")

# Check what columns are available first
print(f"\n=== CHECKING AVAILABLE COLUMNS ===")
print(f"Total columns: {len(df_filtered.columns)}")
print(f"Sample columns: {df_filtered.columns[:10].tolist()}...")

# Check if the issue happens during PMF species filtering
print(f"\n=== CHECKING SPECIES FILTERING IMPACT ===")
# These are the PMF species as used in the analysis (adjusted for what exists)
particle_species = ['PM1 FIDAS', 'PM2.5 FIDAS', 'PM4 FIDAS', 'PM10 FIDAS', 'TSP FIDAS']
voc_species = ['Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene']
inorganic_species = ['NO', 'NO2', 'SO2']  # Removed NOx_ppb as it doesn't exist

pmf_species = particle_species + voc_species + inorganic_species
print(f"PMF species: {len(pmf_species)} total")
print(f"Checking if these columns exist in data...")

missing_species = []
existing_species = []
for species in pmf_species:
    if species not in df_filtered.columns:
        missing_species.append(species)
    else:
        existing_species.append(species)

if missing_species:
    print(f"Missing species: {missing_species}")
else:
    print("All PMF species exist in data")
    
print(f"Existing species ({len(existing_species)}): {existing_species}")

# Now apply the same data filtering as PMF (remove rows where any PMF species is NaN)
print(f"\n=== APPLYING PMF DATA COMPLETENESS FILTERING ===")
pmf_data_only = df_filtered[['datetime'] + existing_species + ['Pressure']].copy()
print(f"Data before removing NaN rows: {len(pmf_data_only)}")

# Remove rows where any existing PMF species has NaN
pmf_data_clean = pmf_data_only.dropna(subset=existing_species)
print(f"Data after removing NaN PMF species: {len(pmf_data_clean)}")

# Check pressure in the PMF-filtered data
pressure_pmf_clean = pmf_data_clean['Pressure']
pressure_pmf_valid = pressure_pmf_clean.dropna()

print(f"\n=== PRESSURE AFTER PMF FILTERING ===")
print(f"Pressure points after PMF filtering: {len(pressure_pmf_valid)}")
if len(pressure_pmf_valid) > 0:
    print(f"Pressure range: {pressure_pmf_valid.min():.2f} - {pressure_pmf_valid.max():.2f} hPa")
    print(f"Unique pressure values: {len(pressure_pmf_valid.unique())}")
    print(f"Pressure std dev: {pressure_pmf_valid.std():.6f}")
    
    if len(pressure_pmf_valid.unique()) == 1:
        print(f"🚨 ISSUE FOUND: After PMF filtering, all pressure values are identical: {pressure_pmf_valid.iloc[0]} hPa")
        print(f"This means PMF species filtering is eliminating pressure variation!")
    else:
        print(f"✅ Pressure still varies after PMF filtering: {len(pressure_pmf_valid.unique())} unique values")
        
    # Show some pressure values with timestamps
    print(f"\nSample pressure values after PMF filtering:")
    sample_data = pmf_data_clean[['datetime', 'Pressure']].head(20)
    print(sample_data)
else:
    print("🚨 NO PRESSURE DATA REMAINS after PMF filtering!")

# Create a diagnostic plot
print(f"\n=== CREATING DIAGNOSTIC PLOTS ===")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Original pressure data (before PMF filtering)
pressure_with_time = df_filtered[df_filtered['Pressure'].notna()][['datetime', 'Pressure']]
ax1.plot(pressure_with_time['datetime'], pressure_with_time['Pressure'], 'b.-', markersize=3)
ax1.set_ylabel('Pressure (hPa)')
ax1.set_title(f'Original Pressure Data (After Date Filtering)\n{len(pressure_with_time)} points, Range: {pressure_with_time["Pressure"].min():.1f}-{pressure_with_time["Pressure"].max():.1f} hPa')
ax1.grid(True, alpha=0.3)

# Plot 2: Pressure after PMF species filtering
if len(pressure_pmf_valid) > 0:
    pmf_pressure_with_time = pmf_data_clean[['datetime', 'Pressure']]
    ax2.plot(pmf_pressure_with_time['datetime'], pmf_pressure_with_time['Pressure'], 'r.-', markersize=3)
    ax2.set_title(f'Pressure After PMF Species Filtering\n{len(pmf_pressure_with_time)} points, Range: {pmf_pressure_with_time["Pressure"].min():.1f}-{pmf_pressure_with_time["Pressure"].max():.1f} hPa')
else:
    ax2.text(0.5, 0.5, 'NO PRESSURE DATA AFTER PMF FILTERING', ha='center', va='center', transform=ax2.transAxes)
    ax2.set_title('Pressure After PMF Species Filtering - NO DATA')

ax2.set_ylabel('Pressure (hPa)')
ax2.set_xlabel('Date')
ax2.grid(True, alpha=0.3)

# Format dates
for ax in [ax1, ax2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig('pressure_diagnostic.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\n=== SUMMARY ===")
print(f"Raw pressure data: VARIES ({pressure_filtered.min():.0f}-{pressure_filtered.max():.0f} hPa)")
if len(pressure_pmf_valid) > 0:
    print(f"After PMF filtering: {'VARIES' if len(pressure_pmf_valid.unique()) > 1 else 'CONSTANT'} ({pressure_pmf_valid.min():.0f}-{pressure_pmf_valid.max():.0f} hPa)")
else:
    print(f"After PMF filtering: NO DATA")