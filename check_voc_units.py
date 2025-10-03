import pandas as pd

# Load the data  
df = pd.read_parquet('mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet')

print('VOC columns and sample values:')
voc_cols = ['Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene']
for col in voc_cols:
    non_null_values = df[col].dropna()
    if len(non_null_values) > 0:
        print(f'{col}: {non_null_values.iloc[0:3].tolist()}')
    else:
        print(f'{col}: No non-null values found')

# Check if there's a metadata file that shows units
try:
    metadata_file = 'mmf_parquet_final/MMF9_Galingale_View_combined_data_metadata.txt'
    with open(metadata_file, 'r') as f:
        content = f.read()
        print('\n=== METADATA CONTENT ===')
        print(content)
except FileNotFoundError:
    print('\nNo metadata file found')