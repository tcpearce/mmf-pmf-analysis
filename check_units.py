#!/usr/bin/env python3
"""
Check if original Excel units have been stored in parquet files.
Compare with the original Excel files to verify unit preservation.
"""

import pyarrow.parquet as pq
import pandas as pd
from pathlib import Path
import sys

def check_parquet_metadata(parquet_file):
    """Check if parquet file contains unit metadata."""
    print(f"\n=== CHECKING PARQUET METADATA: {parquet_file.name} ===")
    
    # Read parquet metadata
    pq_file = pq.ParquetFile(parquet_file)
    schema = pq_file.schema
    metadata = pq_file.metadata
    
    print("\nFile-level metadata:")
    if metadata.metadata:
        for key, value in metadata.metadata.items():
            print(f"  {key.decode()}: {value.decode()}")
    else:
        print("  No file-level metadata found")
    
    print("\nColumn-level metadata:")
    has_units = False
    for i in range(len(schema)):
        col = schema.column(i)
        col_name = col.name
        if col.metadata:
            print(f"  {col_name}: {col.metadata}")
            has_units = True
        else:
            print(f"  {col_name}: No metadata")
    
    if not has_units:
        print("\n❌ NO UNIT METADATA FOUND IN PARQUET FILE")
    
    return has_units

def check_excel_units(excel_file, sheet_name):
    """Extract units from Excel file (row 2)."""
    print(f"\n=== CHECKING EXCEL UNITS: {excel_file.name}, Sheet: {sheet_name} ===")
    
    try:
        # Read first 3 rows to get headers and units
        df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=3, header=None)
        
        if len(df) >= 2:
            headers = df.iloc[0].values  # Row 1: headers
            units = df.iloc[1].values    # Row 2: units
            
            print("\nColumn headers and units from Excel:")
            for i, (header, unit) in enumerate(zip(headers, units)):
                if pd.notna(header) and pd.notna(unit):
                    print(f"  {header}: {unit}")
                elif pd.notna(header):
                    print(f"  {header}: [no unit specified]")
            
            return headers, units
        else:
            print("❌ Not enough rows in Excel file to extract units")
            return None, None
            
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return None, None

def compare_units(station):
    """Compare units between Excel and parquet for a station."""
    print(f"\n{'='*80}")
    print(f"UNIT COMPARISON FOR {station}")
    print(f"{'='*80}")
    
    # Check parquet file
    parquet_file = Path('mmf_parquet_final') / f'{station}_combined_data.parquet'
    if not parquet_file.exists():
        print(f"❌ Parquet file not found: {parquet_file}")
        return
    
    has_parquet_units = check_parquet_metadata(parquet_file)
    
    # Check Excel file - look in raw data directory first, then current directory
    excel_paths = [
        Path(f'mmf_data_corrected/{station}/raw/').glob('*.xlsx'),
        [Path(f'{station}.xlsx')]
    ]
    
    excel_file = None
    for path_list in excel_paths:
        if hasattr(path_list, '__iter__') and not isinstance(path_list, (str, Path)):
            # This is a glob result
            for path in path_list:
                if path.exists():
                    excel_file = path
                    break
        else:
            # This is a single path
            for path in path_list:
                if path.exists():
                    excel_file = path
                    break
        if excel_file:
            break
    
    if not excel_file:
        print(f"❌ Excel file not found for {station}")
        return
    
    print(f"Found Excel file: {excel_file}")
    
    # Try different sheet names
    sheet_names = [f'{station}_5 All', f'{station} All', 'All', None]
    excel_headers, excel_units = None, None
    
    for sheet_name in sheet_names:
        try:
            if sheet_name is None:
                # Try to read without specifying sheet name
                excel_headers, excel_units = check_excel_units(excel_file, 0)
            else:
                excel_headers, excel_units = check_excel_units(excel_file, sheet_name)
            
            if excel_headers is not None:
                break
        except Exception as e:
            continue
    
    if excel_headers is None:
        print("❌ Could not read Excel file units")
        return
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY:")
    if has_parquet_units:
        print("✅ Parquet file contains unit metadata")
    else:
        print("❌ Parquet file MISSING unit metadata")
        print("   Original Excel units are NOT preserved in parquet!")
    
    print(f"✅ Excel file contains units in row 2")

def main():
    """Check units for all stations."""
    stations = ['MMF1', 'MMF2', 'MMF6', 'MMF9']
    
    print("CHECKING UNIT PRESERVATION IN PARQUET FILES")
    print("="*80)
    
    for station in stations:
        try:
            compare_units(station)
        except Exception as e:
            print(f"❌ Error checking {station}: {e}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL ASSESSMENT")
    print(f"{'='*80}")
    print("The data processing script likely needs to be updated to:")
    print("1. Extract units from Excel row 2 during processing")
    print("2. Store units as parquet column metadata")
    print("3. Preserve original unit information for analysis")

if __name__ == "__main__":
    main()
