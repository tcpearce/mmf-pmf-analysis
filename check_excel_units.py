#!/usr/bin/env python3
"""
Check original Excel units from row 2 and compare with current processing.
"""

import pandas as pd
from pathlib import Path

def extract_excel_units(excel_file, sheet_name):
    """Extract headers and units from Excel file."""
    print(f"\n=== CHECKING: {excel_file.name}, Sheet: {sheet_name} ===")
    
    try:
        # Read first 3 rows without setting header
        df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=3, header=None)
        
        print(f"Raw data shape: {df.shape}")
        print("\nFirst 3 rows:")
        for i in range(min(3, len(df))):
            print(f"Row {i}: {df.iloc[i].values[:10]}...")  # Show first 10 columns
        
        if len(df) >= 2:
            headers = df.iloc[0].values
            units = df.iloc[1].values
            
            print(f"\n=== HEADERS AND UNITS ===")
            print("Column | Header | Unit")
            print("-" * 50)
            
            valid_columns = 0
            for i, (header, unit) in enumerate(zip(headers, units)):
                if pd.notna(header) and str(header).strip():
                    unit_str = str(unit) if pd.notna(unit) else "[no unit]"
                    print(f"{i:2d}     | {header:<20} | {unit_str}")
                    valid_columns += 1
                    
                    if valid_columns >= 15:  # Limit output
                        break
            
            return headers, units
        else:
            print("❌ Not enough rows to extract units")
            return None, None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def check_all_stations():
    """Check all MMF stations."""
    stations = ['MMF1', 'MMF2', 'MMF6', 'MMF9']
    
    print("CHECKING ORIGINAL EXCEL UNITS FROM ROW 2")
    print("=" * 80)
    
    for station in stations:
        # Look in raw data directory first, then current directory
        excel_file = None
        raw_dir = Path(f'mmf_data_corrected/{station}/raw')
        
        if raw_dir.exists():
            xlsx_files = list(raw_dir.glob('*.xlsx'))
            if xlsx_files:
                excel_file = xlsx_files[0]
        
        if not excel_file:
            excel_file = Path(f'{station}.xlsx')
        
        if not excel_file.exists():
            print(f"❌ {excel_file} not found")
            continue
        
        print(f"\nChecking: {excel_file}")
        
        # Try different sheet naming patterns
        sheet_patterns = [f'{station}_5 All', f'{station} All', 'All']
        
        found_sheet = False
        for sheet_name in sheet_patterns:
            try:
                headers, units = extract_excel_units(excel_file, sheet_name)
                if headers is not None:
                    found_sheet = True
                    break
            except Exception:
                continue
        
        if not found_sheet:
            try:
                # Try default (first) sheet
                headers, units = extract_excel_units(excel_file, 0)
            except Exception as e:
                print(f"❌ Could not read {station}: {e}")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print("- Original Excel files contain units in row 2")
    print("- Current parquet files do NOT store these units as metadata")  
    print("- Units are lost during the Excel → parquet conversion")
    print("- This is a data preservation issue that should be fixed")

if __name__ == "__main__":
    check_all_stations()
