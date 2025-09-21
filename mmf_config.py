#!/usr/bin/env python3
"""
Central configuration for MMF data paths after the station mapping corrections.
Use this file to ensure all scripts reference the corrected data locations.
"""

from pathlib import Path
import json

# Corrected data paths (post-migration)
MMF_PARQUET_DIR = Path('mmf_parquet_final')
MMF_RAW_DATA_DIR = Path('mmf_data_corrected')

# Legacy paths (for backup/comparison only - DO NOT USE for new analysis)
MMF_PARQUET_LEGACY = Path('mmf_parquet')
MMF_RAW_DATA_LEGACY = Path('mmf_data')

# Configuration files
STATION_LOOKUP_FILE = Path('station_lookup.json')

# Load station mapping
def get_station_mapping():
    """Get the corrected station mapping."""
    try:
        with open(STATION_LOOKUP_FILE, 'r') as f:
            config = json.load(f)
        return {k: v for k, v in config.items() if not k.startswith('_')}
    except FileNotFoundError:
        # Fallback mapping if file not found
        return {
            'MMF1': 'Cemetery Road',
            'MMF2': 'Silverdale Pumping Station',
            'MMF6': 'Fire Station',
            'MMF9': 'Galingale View',
            'Maries_Way': None
        }

# Station lookup functions
def get_mmf_parquet_file(station_or_mmf):
    """
    Get the correct parquet file path for a station or MMF number.
    
    Args:
        station_or_mmf: Either 'MMF1', 'MMF2', etc. or station name like 'Cemetery Road'
        
    Returns:
        Path to the corrected parquet file
    """
    station_mapping = get_station_mapping()
    
    if station_or_mmf.startswith('MMF'):
        # It's an MMF ID
        mmf_id = station_or_mmf
        station_name = station_mapping.get(mmf_id)
        if station_name:
            filename = f"{mmf_id}_{station_name.replace(' ', '_')}_combined_data.parquet"
        else:
            filename = f"{mmf_id}_combined_data.parquet"
    else:
        # It's a station name - find the MMF ID
        station_name = station_or_mmf
        mmf_id = None
        for mmf, station in station_mapping.items():
            if station == station_name:
                mmf_id = mmf
                break
        
        if mmf_id and mmf_id != 'Maries_Way':
            filename = f"{mmf_id}_{station_name.replace(' ', '_')}_combined_data.parquet"
        else:
            filename = f"{station_name.replace(' ', '_')}_combined_data.parquet"
    
    return MMF_PARQUET_DIR / filename

def get_mmf_raw_data_dir(station_or_mmf):
    """
    Get the correct raw data directory for a station or MMF number.
    
    Args:
        station_or_mmf: Either 'MMF1', 'MMF2', etc. or station name like 'Cemetery Road'
        
    Returns:
        Path to the corrected raw data directory
    """
    station_mapping = get_station_mapping()
    
    if station_or_mmf.startswith('MMF'):
        # It's an MMF ID
        mmf_id = station_or_mmf
        station_name = station_mapping.get(mmf_id)
        if station_name:
            if station_name == 'Fire Station':
                dirname = 'MMF6_Fire_Station'
            else:
                dirname = f"{mmf_id}_{station_name.replace(' ', '_')}"
        else:
            dirname = mmf_id
    else:
        # It's a station name
        station_name = station_or_mmf
        if station_name == 'Maries Way':
            dirname = 'Maries_Way'
        elif station_name == 'Cemetery Road':
            dirname = 'MMF1_Cemetery_Road'
        elif station_name == 'Silverdale Pumping Station':
            dirname = 'MMF2_Silverdale_Pumping_Station'
        elif station_name == 'Fire Station':
            dirname = 'MMF6_Fire_Station'
        elif station_name == 'Galingale View':
            dirname = 'MMF9_Galingale_View'
        else:
            dirname = station_name.replace(' ', '_')
    
    return MMF_RAW_DATA_DIR / dirname

def list_available_stations():
    """List all available stations in the corrected dataset."""
    station_mapping = get_station_mapping()
    stations = []
    
    for mmf_id, station_name in station_mapping.items():
        if station_name:
            stations.append(f"{mmf_id} ({station_name})")
        else:
            stations.append("Maries_Way (No MMF number)")
    
    return stations

def get_corrected_mmf_files():
    """Get dictionary of all corrected MMF parquet files."""
    files = {}
    station_mapping = get_station_mapping()
    
    for mmf_id, station_name in station_mapping.items():
        if station_name:
            if mmf_id != 'Maries_Way':
                files[mmf_id] = get_mmf_parquet_file(mmf_id)
        
    # Add Maries Way separately
    files['Maries_Way'] = get_mmf_parquet_file('Maries Way')
    
    return files

# Validation functions
def validate_paths():
    """Validate that all expected corrected data paths exist."""
    issues = []
    
    # Check main directories
    if not MMF_PARQUET_DIR.exists():
        issues.append(f"Missing corrected parquet directory: {MMF_PARQUET_DIR}")
    
    if not MMF_RAW_DATA_DIR.exists():
        issues.append(f"Missing corrected raw data directory: {MMF_RAW_DATA_DIR}")
    
    # Check parquet files
    try:
        corrected_files = get_corrected_mmf_files()
        for mmf_id, file_path in corrected_files.items():
            if not file_path.exists():
                issues.append(f"Missing parquet file for {mmf_id}: {file_path}")
    except Exception as e:
        issues.append(f"Error checking parquet files: {e}")
    
    return issues

if __name__ == "__main__":
    print("MMF Configuration - Corrected Data Paths")
    print("=" * 50)
    print(f"Parquet directory: {MMF_PARQUET_DIR}")
    print(f"Raw data directory: {MMF_RAW_DATA_DIR}")
    print()
    
    print("Available stations:")
    for station in list_available_stations():
        print(f"  - {station}")
    print()
    
    print("Corrected parquet files:")
    for mmf_id, file_path in get_corrected_mmf_files().items():
        exists = "✅" if file_path.exists() else "❌"
        print(f"  {exists} {mmf_id}: {file_path}")
    print()
    
    # Validate paths
    issues = validate_paths()
    if issues:
        print("⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ All corrected data paths validated successfully!")