#!/usr/bin/env python3
"""
Script to relocate MMF raw data files to correct directories based on station names.
"""

import os
import shutil
import csv
import json
from pathlib import Path
from datetime import datetime

# Mapping of actual station names to correct directory names  
STATION_TO_DIR = {
    'Cemetery Road': 'MMF1_Cemetery_Road',
    'Silverdale Pumping Station': 'MMF2_Silverdale_Pumping_Station', 
    'Fire Station': 'MMF6_Fire_Station',
    'Galingale View': 'MMF9_Galingale_View',
    'Maries Way': 'Maries_Way'
}

# Current incorrect MMF directory mapping
CURRENT_WRONG_MAPPING = {
    'MMF1': 'Silverdale Pumping Station',  # Should be Cemetery Road
    'MMF2': 'Cemetery Road',               # Should be Silverdale Pumping Station
    'MMF6': 'Galingale View',              # Should be Fire Station
    'MMF9': 'Maries Way'                   # Should be Galingale View
}

def get_station_from_filename(filename):
    """Extract station name from filename."""
    if 'Cemetery' in filename or 'Cemetery_Road' in filename:
        return 'Cemetery Road'
    elif 'Silverdale Pumping Station' in filename or 'Pumping_Station' in filename:
        return 'Silverdale Pumping Station'
    elif 'Fire Station' in filename or 'Fire_Station' in filename:
        return 'Fire Station'
    elif 'Galingale View' in filename or 'Galingale_View' in filename:
        return 'Galingale View'
    elif 'Maries Way' in filename or 'Maries_Way' in filename:
        return 'Maries Way'
    else:
        return None

def main():
    """Main relocation function."""
    print("MMF FILE RELOCATION SCRIPT")
    print("=" * 50)
    
    # Track moves
    moves_log = []
    
    # Create corrected parquet directory structure too
    os.makedirs('mmf_parquet_corrected', exist_ok=True)
    
    # Process each incorrectly named MMF directory
    source_dir = Path('./mmf_data')
    target_dir = Path('./mmf_data_corrected')
    
    if not source_dir.exists():
        print("ERROR: mmf_data directory not found!")
        return
    
    print(f"Moving files from {source_dir} to {target_dir}")
    print()
    
    for old_mmf_dir in source_dir.iterdir():
        if old_mmf_dir.is_dir() and old_mmf_dir.name.startswith('MMF'):
            mmf_id = old_mmf_dir.name
            actual_station = CURRENT_WRONG_MAPPING.get(mmf_id)
            
            if actual_station:
                correct_dir = STATION_TO_DIR[actual_station]
                
                print(f"Processing {mmf_id} ({actual_station}):")
                print(f"  Target directory: {correct_dir}")
                
                # Move raw files
                raw_source = old_mmf_dir / 'raw'
                raw_target = target_dir / correct_dir / 'raw'
                
                if raw_source.exists():
                    for file_path in raw_source.glob('*.xlsx'):
                        target_path = raw_target / file_path.name
                        print(f"  Moving: {file_path.name}")
                        shutil.copy2(file_path, target_path)
                        
                        moves_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'source': str(file_path),
                            'target': str(target_path),
                            'old_mmf_id': mmf_id,
                            'station_name': actual_station,
                            'correct_dir': correct_dir
                        })
                
                # Move processed files  
                processed_source = old_mmf_dir / 'processed'
                processed_target = target_dir / correct_dir / 'processed'
                
                if processed_source.exists():
                    for file_path in processed_source.glob('*.xlsx'):
                        target_path = processed_target / file_path.name
                        print(f"  Moving: {file_path.name}")
                        shutil.copy2(file_path, target_path)
                        
                        moves_log.append({
                            'timestamp': datetime.now().isoformat(),
                            'source': str(file_path),
                            'target': str(target_path),
                            'old_mmf_id': mmf_id,
                            'station_name': actual_station,
                            'correct_dir': correct_dir
                        })
                
                print()
    
    # Check for any Fire Station files that might exist elsewhere
    print("Checking for Fire Station files...")
    fire_station_found = False
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if 'Fire Station' in filename or 'Fire_Station' in filename:
                print(f"🚨 ALERT: Fire Station file found: {root}/{filename}")
                fire_station_found = True
    
    if not fire_station_found:
        print("ℹ️  No Fire Station files found (as expected)")
    
    # Save move log
    with open('mmf_relocation_log.csv', 'w', newline='') as f:
        if moves_log:
            fieldnames = moves_log[0].keys()
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(moves_log)
    
    print(f"\\nRelocation completed!")
    print(f"Moved {len(moves_log)} files")
    print(f"Move log saved to: mmf_relocation_log.csv")
    
    # Display summary
    print("\\nSUMMARY OF MOVES:")
    print("-" * 30)
    for station, count in {}.items():
        station_moves = [m for m in moves_log if m['station_name'] == station]
        if station_moves:
            print(f"{station}: {len(station_moves)} files")

if __name__ == "__main__":
    main()