#!/usr/bin/env python3
"""
Script to regenerate parquet files with correct MMF IDs and station names.
"""

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Load station mapping
with open('station_lookup.json', 'r') as f:
    STATION_LOOKUP = json.load(f)

# Clean mapping (remove metadata)
station_mapping = {k: v for k, v in STATION_LOOKUP.items() if not k.startswith('_')}

# Reverse mapping to get MMF ID from station name
STATION_TO_MMF = {v: k for k, v in station_mapping.items() if v is not None}
STATION_TO_MMF['Maries Way'] = None  # Special case - no MMF number

def get_mmf_info(directory_name):
    """Extract MMF ID and station name from directory name."""
    if directory_name == 'MMF1_Cemetery_Road':
        return '1', 'Cemetery Road'
    elif directory_name == 'MMF2_Silverdale_Pumping_Station':
        return '2', 'Silverdale Pumping Station'
    elif directory_name == 'MMF6_Fire_Station':
        return '6', 'Fire Station'
    elif directory_name == 'MMF9_Galingale_View':
        return '9', 'Galingale View'
    elif directory_name == 'Maries_Way':
        return None, 'Maries Way'
    else:
        return None, None

def process_excel_to_parquet(excel_path, output_dir, mmf_id, station_name):
    """Convert Excel data to parquet with correct metadata."""
    print(f"Processing: {excel_path.name}")
    
    try:
        # Read both gas and particle data sheets
        gas_data = pd.read_excel(excel_path, sheet_name=1)  # Sheet 2: Gas data
        particle_data = pd.read_excel(excel_path, sheet_name=2)  # Sheet 3: Particle data
        
        print(f"  Gas data: {len(gas_data)} rows")
        print(f"  Particle data: {len(particle_data)} rows")
        
        # Process gas data (5-minute intervals)
        gas_data = gas_data.copy()
        gas_data['datetime'] = pd.to_datetime(gas_data['DATE'].astype(str) + ' ' + 
                                             gas_data['TIME'].astype(str), 
                                             errors='coerce')
        gas_data = gas_data.dropna(subset=['datetime'])
        gas_data = gas_data.drop(['DATE', 'TIME'], axis=1)
        gas_data['gas_data_available'] = True
        gas_data['particle_data_available'] = False
        
        # Process particle data (15-minute intervals)
        particle_data = particle_data.copy()
        particle_data['datetime'] = pd.to_datetime(particle_data['DATE'].astype(str) + ' ' + 
                                                  particle_data['TIME'].astype(str), 
                                                  errors='coerce')
        particle_data = particle_data.dropna(subset=['datetime'])
        particle_data = particle_data.drop(['DATE', 'TIME'], axis=1)
        particle_data['gas_data_available'] = False
        particle_data['particle_data_available'] = True
        
        # Create a complete 5-minute time grid
        start_time = min(gas_data['datetime'].min(), particle_data['datetime'].min())
        end_time = max(gas_data['datetime'].max(), particle_data['datetime'].max())
        
        time_grid = pd.date_range(start=start_time, end=end_time, freq='5min')
        combined_df = pd.DataFrame({'datetime': time_grid})
        
        # Merge gas data (exact match on 5-minute intervals)
        combined_df = combined_df.merge(gas_data, on='datetime', how='left')
        
        # Forward-fill particle data from 15-minute to 5-minute intervals
        particle_data_5min = particle_data.set_index('datetime').reindex(time_grid, method='ffill').reset_index()
        particle_data_5min.rename(columns={'index': 'datetime'}, inplace=True)
        
        # Merge particle data
        particle_cols = [col for col in particle_data_5min.columns if col not in combined_df.columns]
        particle_subset = particle_data_5min[['datetime'] + particle_cols]
        combined_df = combined_df.merge(particle_subset, on='datetime', how='left')
        
        # Update availability flags
        combined_df['gas_data_available'] = combined_df['gas_data_available'].fillna(False)
        combined_df['particle_data_available'] = combined_df['particle_data_available'].fillna(False)
        
        # Add MMF ID and station name columns
        combined_df['mmf_id'] = mmf_id
        combined_df['station_name'] = station_name
        
        # Reorder columns to put identifiers first
        identifier_cols = ['datetime', 'mmf_id', 'station_name']
        other_cols = [col for col in combined_df.columns if col not in identifier_cols]
        combined_df = combined_df[identifier_cols + other_cols]
        
        # Create output filename
        if mmf_id:
            output_filename = f"MMF{mmf_id}_{station_name.replace(' ', '_')}_combined_data.parquet"
        else:
            output_filename = f"{station_name.replace(' ', '_')}_combined_data.parquet"
        
        output_path = output_dir / output_filename
        
        # Create custom metadata
        metadata = {
            'mmf_id': str(mmf_id) if mmf_id else 'null',
            'station_name': station_name,
            'schema_version': 'v2',
            'processing_date': datetime.now().isoformat(),
            'source_file': excel_path.name,
            'record_count': str(len(combined_df)),
            'date_range_start': str(combined_df['datetime'].min()),
            'date_range_end': str(combined_df['datetime'].max()),
            'time_interval': '5_minutes',
            'gas_data_original_interval': '5_minutes',
            'particle_data_original_interval': '15_minutes_forward_filled'
        }
        
        # Convert to PyArrow table with metadata
        table = pa.Table.from_pandas(combined_df)
        
        # Add metadata to schema
        schema = table.schema
        metadata_dict = {k.encode(): v.encode() for k, v in metadata.items()}
        schema_with_metadata = schema.with_metadata(metadata_dict)
        table = table.cast(schema_with_metadata)
        
        # Write parquet file
        pq.write_table(table, output_path)
        
        print(f"  ✅ Created: {output_filename}")
        print(f"  Records: {len(combined_df):,}")
        print(f"  Date range: {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
        print()
        
        return {
            'station_name': station_name,
            'mmf_id': mmf_id,
            'output_file': str(output_path),
            'record_count': len(combined_df),
            'date_range_start': str(combined_df['datetime'].min()),
            'date_range_end': str(combined_df['datetime'].max())
        }
        
    except Exception as e:
        print(f"  ❌ ERROR processing {excel_path.name}: {str(e)}")
        return None

def create_empty_fire_station_parquet(output_dir):
    """Create empty parquet file for Fire Station with schema only."""
    print("Creating empty Fire Station parquet file...")
    
    # Create empty dataframe with expected schema
    empty_df = pd.DataFrame({
        'datetime': pd.Series([], dtype='datetime64[ns]'),
        'mmf_id': pd.Series([], dtype='string'),
        'station_name': pd.Series([], dtype='string'),
        'WD': pd.Series([], dtype='float64'),
        'WS': pd.Series([], dtype='float64'), 
        'CH4': pd.Series([], dtype='float64'),
        'H2S': pd.Series([], dtype='float64'),
        'SO2': pd.Series([], dtype='float64'),
        'PM1 FIDAS': pd.Series([], dtype='float64'),
        'PM2.5 FIDAS': pd.Series([], dtype='float64'),
        'PM4 FIDAS': pd.Series([], dtype='float64'),
        'PM10 FIDAS': pd.Series([], dtype='float64'),
        'TSP FIDAS': pd.Series([], dtype='float64'),
        'TEMP': pd.Series([], dtype='float64'),
        'Pressure': pd.Series([], dtype='float64'),
        'gas_data_available': pd.Series([], dtype='bool'),
        'particle_data_available': pd.Series([], dtype='bool')
    })
    
    # Add Fire Station metadata
    empty_df['mmf_id'] = '6'
    empty_df['station_name'] = 'Fire Station'
    
    # Create metadata
    metadata = {
        'mmf_id': '6',
        'station_name': 'Fire Station',
        'schema_version': 'v2',
        'processing_date': datetime.now().isoformat(),
        'record_count': '0',
        'status': 'empty_placeholder',
        'note': 'Raw data not available - placeholder file for schema consistency'
    }
    
    # Convert to PyArrow table with metadata
    table = pa.Table.from_pandas(empty_df)
    metadata_dict = {k.encode(): v.encode() for k, v in metadata.items()}
    schema_with_metadata = table.schema.with_metadata(metadata_dict)
    table = table.cast(schema_with_metadata)
    
    # Write empty parquet file
    output_path = output_dir / "MMF6_Fire_Station_combined_data.parquet"
    pq.write_table(table, output_path)
    
    print(f"  ✅ Created empty placeholder: MMF6_Fire_Station_combined_data.parquet")
    print()

def main():
    """Main processing function."""
    print("REGENERATING PARQUET FILES WITH CORRECT METADATA")
    print("=" * 60)
    
    corrected_data_dir = Path('./mmf_data_corrected')
    output_parquet_dir = Path('./mmf_parquet_corrected')
    output_parquet_dir.mkdir(exist_ok=True)
    
    processing_results = []
    
    # Process each station directory
    for station_dir in corrected_data_dir.iterdir():
        if station_dir.is_dir():
            mmf_id, station_name = get_mmf_info(station_dir.name)
            
            print(f"PROCESSING: {station_dir.name}")
            print(f"MMF ID: {mmf_id}, Station: {station_name}")
            
            # Find Excel files in raw directory
            raw_dir = station_dir / 'raw'
            if raw_dir.exists():
                excel_files = list(raw_dir.glob('*.xlsx'))
                
                if excel_files:
                    # Process the most recent/comprehensive Excel file
                    # Usually the raw files with long hash names are more complete
                    excel_file = max(excel_files, key=lambda x: x.stat().st_size)
                    
                    result = process_excel_to_parquet(
                        excel_file, output_parquet_dir, mmf_id, station_name
                    )
                    
                    if result:
                        processing_results.append(result)
                        
                elif station_dir.name == 'MMF6_Fire_Station':
                    # Create empty Fire Station file
                    create_empty_fire_station_parquet(output_parquet_dir)
                    processing_results.append({
                        'station_name': 'Fire Station',
                        'mmf_id': '6',
                        'output_file': str(output_parquet_dir / "MMF6_Fire_Station_combined_data.parquet"),
                        'record_count': 0,
                        'status': 'empty_placeholder'
                    })
                else:
                    print(f"  ⚠️  No Excel files found in {raw_dir}")
            else:
                print(f"  ⚠️  Raw directory not found: {raw_dir}")
            
            print()
    
    # Save processing results
    with open('parquet_regeneration_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': processing_results,
            'summary': {
                'total_files_created': len(processing_results),
                'stations_processed': [r['station_name'] for r in processing_results]
            }
        }, f, indent=2)
    
    print("=" * 60)
    print("PARQUET REGENERATION COMPLETED")
    print(f"Files created: {len(processing_results)}")
    print("Results saved to: parquet_regeneration_results.json")

if __name__ == "__main__":
    main()