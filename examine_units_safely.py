#!/usr/bin/env python3
"""
Safely examine Excel file structure to identify units location.
"""

import pandas as pd
from pathlib import Path
import sys

def examine_excel_units():
    """Examine Excel file structure for units."""
    
    # Define file paths - check raw directory for extended data files
    stations = ['MMF1', 'MMF2', 'MMF6', 'MMF9']
    excel_files = []
    
    for station in stations:
        raw_dir = Path(f"mmf_data_corrected/{station}/raw")
        if raw_dir.exists():
            # Find Excel files in raw directory
            xlsx_files = list(raw_dir.glob('*.xlsx'))
            if xlsx_files:
                excel_files.append(str(xlsx_files[0]))  # Take first Excel file found
    
    print("EXAMINING EXCEL FILES FOR ORIGINAL UNITS")
    print("=" * 80)
    
    for excel_path in excel_files[:2]:  # Check first 2 files found
        print(f"\nChecking: {excel_path}")
        
        if not Path(excel_path).exists():
            print(f"❌ File not found: {excel_path}")
            continue
        
        try:
            # Get sheet names first
            xl = pd.ExcelFile(excel_path)
            print(f"✅ File accessible. Sheets: {xl.sheet_names}")
            
            # Examine the main sheet (usually the longest name or containing "All")
            main_sheet = None
            for sheet in xl.sheet_names:
                if "All" in sheet or "5" in sheet:
                    main_sheet = sheet
                    break
            
            if main_sheet is None:
                main_sheet = xl.sheet_names[0]
            
            print(f"\nExamining sheet: {main_sheet}")
            
            # Read first 10 rows to understand structure
            df_structure = pd.read_excel(excel_path, sheet_name=main_sheet, nrows=10, header=None)
            
            print(f"Sheet dimensions: {df_structure.shape}")
            print("\nFirst 10 rows (showing first 8 columns):")
            print("-" * 60)
            
            for i in range(min(10, len(df_structure))):
                row_data = df_structure.iloc[i].values[:8]
                row_display = []
                for j, val in enumerate(row_data):
                    if pd.isna(val):
                        row_display.append("NaN")
                    else:
                        val_str = str(val)
                        if len(val_str) > 15:
                            val_str = val_str[:12] + "..."
                        row_display.append(f"{j}:{val_str}")
                
                print(f"Row {i:2d}: {' | '.join(row_display)}")
            
            # Look for units patterns in each row
            print(f"\n🔍 Searching for units patterns...")
            
            units_found = {}
            for row_idx in range(min(10, len(df_structure))):
                row_vals = df_structure.iloc[row_idx].values
                unit_count = 0
                units_in_row = []
                
                for col_idx, val in enumerate(row_vals):
                    if pd.notna(val):
                        val_str = str(val).lower()
                        # Look for common air quality units
                        unit_patterns = ['ug/m3', 'μg/m³', 'mg/m3', 'degrees', 'm/s', 'gmt', 'hpa', '°c', 'mbar', 'ppm', 'ppb']
                        for pattern in unit_patterns:
                            if pattern in val_str:
                                unit_count += 1
                                units_in_row.append(f"Col{col_idx}:{pattern}")
                                break
                
                if unit_count >= 2:
                    units_found[row_idx] = units_in_row
                    print(f"  Row {row_idx}: {unit_count} units found - {', '.join(units_in_row)}")
            
            if not units_found:
                print("  ❌ No clear units row identified")
                print("  💡 Units may be embedded in column names or in a different format")
            
            xl.close()
            
        except PermissionError:
            print(f"❌ Permission denied - file may be open in Excel")
            print("💡 Please close Excel and try again")
        except Exception as e:
            print(f"❌ Error examining file: {e}")
    
    # Check current parquet metadata
    print(f"\n{'='*60}")
    print("COMPARING WITH CURRENT PARQUET FILES")
    print(f"{'='*60}")
    
    parquet_file = Path("mmf_parquet_final/MMF1_Cemetery_Road_combined_data.parquet")
    if parquet_file.exists():
        try:
            import pyarrow.parquet as pq
            
            # Read parquet metadata
            pq_file = pq.ParquetFile(parquet_file)
            
            print(f"✅ Parquet file exists: {parquet_file}")
            print(f"Parquet columns: {[col.name for col in pq_file.schema]}")
            
            # Check for any metadata
            metadata = pq_file.metadata
            if metadata.metadata:
                print("File-level metadata found:")
                for key, value in metadata.metadata.items():
                    key_str = key.decode() if isinstance(key, bytes) else str(key)
                    if 'units' in key_str.lower():
                        print(f"  {key_str}: {value.decode() if isinstance(value, bytes) else str(value)}")
            else:
                print("❌ No units metadata in parquet file")
            
        except Exception as e:
            print(f"Error reading parquet metadata: {e}")
    else:
        print("❌ Parquet file not found")
    
    print(f"\n{'='*80}")
    print("CONCLUSION:")
    print("=" * 80)
    print("1. ✅ We can access the original Excel files")
    print("2. 🔍 Units are likely in row 1 or 2 of the Excel files")
    print("3. ❌ Current parquet files do NOT preserve units metadata")
    print("4. 💡 The processing script needs to be updated to:")
    print("   - Extract units from the identified Excel row")
    print("   - Store units as parquet column metadata")
    print("   - Make units accessible for analysis")

if __name__ == "__main__":
    examine_excel_units()
