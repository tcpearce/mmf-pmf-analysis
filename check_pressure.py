import pandas as pd
import numpy as np

# Load the data
df = pd.read_parquet('mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet')
df['datetime'] = pd.to_datetime(df['datetime'])

# Check October 2023 data
oct_2023 = df[(df['datetime'] >= '2023-10-01') & (df['datetime'] <= '2023-10-05')]
print(f'Oct 2023 total records: {oct_2023.shape[0]}')
print(f'Non-null pressure: {oct_2023["Pressure"].notna().sum()}')
print('\nFirst few records:')
print(oct_2023[['datetime', 'Pressure']].head(10))

if oct_2023['Pressure'].notna().any():
    print(f'\nPressure range: {oct_2023["Pressure"].min():.2f} - {oct_2023["Pressure"].max():.2f} hPa')
else:
    print('\nNo pressure data available for October 2023!')
    
# Check when pressure data starts being available
pressure_available = df[df['Pressure'].notna()]
if len(pressure_available) > 0:
    print(f'\nPressure data availability:')
    print(f'First available: {pressure_available["datetime"].min()}')
    print(f'Last available: {pressure_available["datetime"].max()}')
    print(f'Total records with pressure: {len(pressure_available)}')
else:
    print('\nNo pressure data available in entire dataset!')