#!/usr/bin/env python3
"""
BTEX data integration script for MMF parquet files.
Integrates 30-minute BTEX (Benzene, Toluene, Ethylbenzene, m&p-Xylene) data 
into existing MMF2 and MMF9 parquet files using established best practices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import shutil
import json

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('btex_integration.log'),
        logging.StreamHandler()
    ]
)

class BTEXIntegrator:
    def __init__(self):
        self.btex_excel_path = Path('mmf_data_corrected/BTEX/BTEX data for UKHSA.xlsx')
        self.parquet_dir = Path('mmf_parquet_final')
        self.backup_dir = Path(f'mmf_parquet_backup_btex_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # VOC columns to extract and integrate
        self.voc_columns = ['Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene']
        
        # Columns to ignore/drop from BTEX sheets
        self.ignore_columns = ['wd 2', 'ws 2', 'WD 9', 'WS 9', 'Temp']
        
        logging.info("BTEX Integration initialized")
        
    def create_backup(self):
        """Create timestamped backup of existing parquet files and metadata."""
        try:
            self.backup_dir.mkdir(exist_ok=True)
            
            # Files to backup for MMF2 and MMF9
            files_to_backup = [
                'MMF2_Silverdale_Pumping_Station_combined_data.parquet',
                'MMF2_Silverdale_Pumping_Station_combined_data_metadata.txt',
                'MMF9_Galingale_View_combined_data.parquet',
                'MMF9_Galingale_View_combined_data_metadata.txt'
            ]
            
            for filename in files_to_backup:
                src = self.parquet_dir / filename
                dst = self.backup_dir / filename
                if src.exists():
                    shutil.copy2(src, dst)
                    logging.info(f"Backed up: {filename}")
                else:
                    logging.warning(f"File not found for backup: {filename}")
            
            logging.info(f"Backup created: {self.backup_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating backup: {e}")
            return False
    
    def find_header_row(self, df_sample):
        """Find the row containing the data headers (adapted from process_mmf_corrected.py)."""
        for i in range(len(df_sample)):
            row_values = df_sample.iloc[i].astype(str).str.upper()
            if any('DATE' in str(val) for val in row_values):
                logging.info(f"Found header row at index {i}")
                return i
        
        # If no DATE found, check if first row looks like headers
        first_row = df_sample.iloc[0].astype(str)
        if any(col in first_row.str.upper().values for col in ['DATE', 'TIME', 'BENZENE']):
            logging.info("Using row 0 as header")
            return 0
        
        logging.warning("No clear header row found, using row 0")
        return 0
    
    def extract_units_from_sheet(self, sheet_name):
        """Extract units from BTEX sheet (adapted from process_mmf_corrected.py)."""
        try:
            # Read first few rows to find units
            df_sample = pd.read_excel(self.btex_excel_path, sheet_name=sheet_name, 
                                    nrows=5, engine='openpyxl')
            
            # Units are typically in the first data row for VOC columns
            units_dict = {}
            for col in self.voc_columns:
                if col in df_sample.columns:
                    # Look for µg/m3 in first few rows
                    for idx in range(len(df_sample)):
                        cell_value = str(df_sample[col].iloc[idx])
                        if 'µg/m3' in cell_value or 'ug/m3' in cell_value:
                            units_dict[col] = 'ug/m3'  # Normalize to match existing convention
                            break
                    
                    # Default if not found
                    if col not in units_dict:
                        units_dict[col] = 'ug/m3'
            
            logging.info(f"Extracted units from {sheet_name}: {units_dict}")
            return units_dict
            
        except Exception as e:
            logging.error(f"Error extracting units from {sheet_name}: {e}")
            # Return default units
            return {col: 'ug/m3' for col in self.voc_columns}
    
    def fix_datetime_parsing(self, df):
        """Fix datetime parsing (adapted from process_mmf_corrected.py)."""
        try:
            datetime_series = []
            
            for idx, row in df.iterrows():
                try:
                    date_str = str(row['Date']).strip()
                    time_str = str(row['Time']).strip()
                    
                    # Skip if either is NaN or empty
                    if date_str in ['nan', 'NaT', ''] or time_str in ['nan', 'NaT', '']:
                        datetime_series.append(pd.NaT)
                        continue
                    
                    # Handle time as timedelta (Excel sometimes parses time this way)
                    if 'day' in time_str:
                        # Extract time from timedelta format like "1 day, 0:00:00"
                        parts = time_str.split(', ')
                        if len(parts) == 2:
                            days = int(parts[0].split()[0])
                            time_part = parts[1]
                            # Add days to date and use time part
                            base_date = pd.to_datetime(date_str)
                            base_date = base_date + pd.Timedelta(days=days)
                            combined_str = f"{base_date.strftime('%Y-%m-%d')} {time_part}"
                        else:
                            combined_str = f"{date_str} 00:00:00"
                    else:
                        # Normal case: combine date and time strings
                        combined_str = f"{date_str} {time_str}"
                    
                    # Parse datetime without timezone to match existing data
                    dt = pd.to_datetime(combined_str, errors='coerce')
                    datetime_series.append(dt)
                    
                except Exception as e:
                    datetime_series.append(pd.NaT)
                    continue
            
            df['datetime'] = datetime_series
            
            # Remove rows with invalid datetime
            initial_rows = len(df)
            df = df.dropna(subset=['datetime'])
            logging.info(f"Removed {initial_rows - len(df)} rows with invalid datetime")
            
            # Sort by datetime and remove duplicates (aggregate by mean if duplicates exist)
            if df['datetime'].duplicated().any():
                logging.info("Found duplicate datetimes, aggregating by mean")
                # Group by datetime and take mean for numeric columns
                numeric_cols = [col for col in self.voc_columns if col in df.columns]
                df = df.groupby('datetime')[numeric_cols].mean().reset_index()
            else:
                df = df.sort_values('datetime').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            logging.error(f"Error in datetime parsing: {e}")
            return None
    
    def read_btex_sheet(self, sheet_name):
        """Read and clean BTEX data from Excel sheet."""
        try:
            logging.info(f"Reading BTEX data from sheet: {sheet_name}")
            
            # Read full sheet
            df = pd.read_excel(self.btex_excel_path, sheet_name=sheet_name, engine='openpyxl')
            logging.info(f"Initial shape: {df.shape}")
            logging.info(f"Initial columns: {list(df.columns)}")
            
            # Drop unnamed columns and ignore columns
            cols_to_drop = []
            for col in df.columns:
                if str(col).startswith('Unnamed:') or col in self.ignore_columns:
                    cols_to_drop.append(col)
            
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                logging.info(f"Dropped columns: {cols_to_drop}")
            
            # Keep only required columns
            required_cols = ['Date', 'Time'] + [col for col in self.voc_columns if col in df.columns]
            df = df[required_cols]
            logging.info(f"Kept columns: {list(df.columns)}")
            
            # Remove rows where VOC columns contain unit strings (like 'µg/m3')
            for col in self.voc_columns:
                if col in df.columns:
                    # Remove rows where this column contains unit strings
                    unit_mask = df[col].astype(str).str.contains('µg/m3|ug/m3', na=False)
                    df = df[~unit_mask]
            
            logging.info(f"After removing unit rows: {df.shape}")
            
            # Convert VOC columns to numeric
            for col in self.voc_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Fix datetime parsing
            df = self.fix_datetime_parsing(df)
            if df is None:
                return None, {}
            
            # Extract units
            units_dict = self.extract_units_from_sheet(sheet_name)
            
            logging.info(f"Final BTEX data: {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
            logging.info(f"VOC columns summary:")
            for col in self.voc_columns:
                if col in df.columns:
                    non_na_count = df[col].notna().sum()
                    logging.info(f"  {col}: {non_na_count} non-NaN values")
            
            return df, units_dict
            
        except Exception as e:
            logging.error(f"Error reading BTEX sheet {sheet_name}: {e}")
            return None, {}
    
    def load_existing_parquet(self, station):
        """Load existing parquet file for the station."""
        try:
            if station == 'MMF2':
                parquet_file = self.parquet_dir / 'MMF2_Silverdale_Pumping_Station_combined_data.parquet'
                metadata_file = self.parquet_dir / 'MMF2_Silverdale_Pumping_Station_combined_data_metadata.txt'
            elif station == 'MMF9':
                parquet_file = self.parquet_dir / 'MMF9_Galingale_View_combined_data.parquet'
                metadata_file = self.parquet_dir / 'MMF9_Galingale_View_combined_data_metadata.txt'
            else:
                raise ValueError(f"Unknown station: {station}")
            
            df = pd.read_parquet(parquet_file)
            logging.info(f"Loaded {station} parquet: {df.shape}")
            logging.info(f"Columns: {list(df.columns)}")
            logging.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df, parquet_file, metadata_file
            
        except Exception as e:
            logging.error(f"Error loading parquet for {station}: {e}")
            return None, None, None
    
    def integrate_btex_data(self, parquet_df, btex_df, station):
        """Integrate BTEX data into existing parquet data using exact timestamp matching."""
        try:
            logging.info(f"Integrating BTEX data for {station}")
            
            # Ensure datetime is the merge key and both are timezone-naive
            if parquet_df['datetime'].dt.tz is not None:
                parquet_df['datetime'] = parquet_df['datetime'].dt.tz_convert(None)
            if btex_df['datetime'].dt.tz is not None:
                btex_df['datetime'] = btex_df['datetime'].dt.tz_convert(None)
            
            # Prepare BTEX data for merge (keep only datetime and VOC columns)
            btex_merge = btex_df[['datetime'] + [col for col in self.voc_columns if col in btex_df.columns]].copy()
            
            # Left join: parquet data is authoritative timeline, BTEX fills in where timestamps match exactly
            logging.info("Performing exact timestamp left join...")
            merged_df = parquet_df.merge(btex_merge, on='datetime', how='left', suffixes=('', '_btex'))
            
            # Count matches
            for col in self.voc_columns:
                if col in merged_df.columns:
                    non_na_count = merged_df[col].notna().sum()
                    total_rows = len(merged_df)
                    coverage = (non_na_count / total_rows) * 100
                    logging.info(f"  {col}: {non_na_count}/{total_rows} rows ({coverage:.1f}% coverage)")
            
            # Ensure column order: insert BTEX after gas species, before PM columns
            existing_cols = list(parquet_df.columns)
            
            # Find insertion point (after last gas species, before first PM)
            insert_idx = len(existing_cols)  # Default to end
            for i, col in enumerate(existing_cols):
                if 'PM' in col or 'FIDAS' in col:
                    insert_idx = i
                    break
            
            # Reorder columns
            new_columns = existing_cols[:insert_idx]
            for col in self.voc_columns:
                if col in merged_df.columns:
                    new_columns.append(col)
            new_columns.extend(existing_cols[insert_idx:])
            
            # Reorder the dataframe
            merged_df = merged_df[new_columns]
            
            logging.info(f"Integration complete. Final shape: {merged_df.shape}")
            logging.info(f"New column order: {list(merged_df.columns)}")
            
            return merged_df
            
        except Exception as e:
            logging.error(f"Error integrating BTEX data for {station}: {e}")
            return None
    
    def save_updated_parquet(self, df, station, btex_units, parquet_file, metadata_file):
        """Save updated parquet file with BTEX data and update metadata."""
        try:
            logging.info(f"Saving updated parquet for {station}")
            
            # Read existing parquet metadata
            existing_parquet = pq.ParquetFile(parquet_file)
            existing_metadata = existing_parquet.metadata.metadata if existing_parquet.metadata and existing_parquet.metadata.metadata else {}
            
            # Update metadata with BTEX information
            updated_metadata = existing_metadata.copy()
            
            # Add BTEX units to existing units metadata
            if 'units' in updated_metadata:
                try:
                    units_dict = json.loads(updated_metadata['units'])
                    units_dict.update(btex_units)
                    updated_metadata['units'] = json.dumps(units_dict)
                except:
                    # If existing units can't be parsed, create new
                    updated_metadata['units'] = json.dumps(btex_units)
            else:
                updated_metadata['units'] = json.dumps(btex_units)
            
            # Add integration provenance
            updated_metadata.update({
                'btex_integrated': 'true',
                'btex_source_excel': str(self.btex_excel_path),
                'btex_integration_method': '30min exact-match onto 5min grid; no interpolation',
                'btex_integration_date': datetime.now().isoformat(),
                'schema_version': 'v2'
            })
            
            # Convert to PyArrow table and add metadata
            table = pa.Table.from_pandas(df)
            table = table.replace_schema_metadata(updated_metadata)
            
            # Write parquet file
            pq.write_table(table, parquet_file)
            logging.info(f"Saved parquet file: {parquet_file}")
            
            # Update metadata text file
            self.update_metadata_text(metadata_file, station, btex_units, df)
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving parquet for {station}: {e}")
            return False
    
    def update_metadata_text(self, metadata_file, station, btex_units, df):
        """Update the metadata text file to include BTEX columns."""
        try:
            # Read existing metadata
            with open(metadata_file, 'r') as f:
                content = f.read()
            
            # Add BTEX information to columns and units section
            btex_info = "\n"
            for col, unit in btex_units.items():
                btex_info += f"  {col} ({unit})\n"
            
            # Insert BTEX columns before "Columns and units:" section ends
            if "gas_data_available (boolean flag)" in content:
                content = content.replace(
                    "  gas_data_available (boolean flag)",
                    btex_info.strip() + "\n  gas_data_available (boolean flag)"
                )
            
            # Add BTEX integration note to processing notes
            if "Processing notes:" in content:
                btex_note = f"- BTEX data: 30-minute intervals from {self.btex_excel_path.name}, exact-match alignment\n"
                content = content.replace(
                    "- Missing values preserved as NaN",
                    btex_note + "- Missing values preserved as NaN"
                )
            
            # Update record count and other stats if needed
            records_line = f"Records: {len(df)}"
            content = content.replace(
                f"Records: {len(df) - sum(df[col].notna() for col in btex_units.keys() if col in df.columns)}",
                records_line
            )
            
            # Write updated metadata
            with open(metadata_file, 'w') as f:
                f.write(content)
            
            logging.info(f"Updated metadata file: {metadata_file}")
            
        except Exception as e:
            logging.error(f"Error updating metadata text file: {e}")
    
    def validate_integration(self, original_df, integrated_df, btex_df, station):
        """Validate the BTEX integration."""
        try:
            logging.info(f"Validating BTEX integration for {station}")
            
            # Check row count unchanged
            if len(original_df) != len(integrated_df):
                logging.error(f"Row count changed: {len(original_df)} -> {len(integrated_df)}")
                return False
            
            # Check no unnamed columns
            unnamed_cols = [col for col in integrated_df.columns if str(col).startswith('Unnamed:')]
            if unnamed_cols:
                logging.error(f"Found unnamed columns in output: {unnamed_cols}")
                return False
            
            # Check ignore columns not present
            ignore_present = [col for col in integrated_df.columns if col in self.ignore_columns]
            if ignore_present:
                logging.error(f"Found ignored columns in output: {ignore_present}")
                return False
            
            # Check BTEX columns are present and numeric
            for col in self.voc_columns:
                if col in integrated_df.columns:
                    if not pd.api.types.is_numeric_dtype(integrated_df[col]):
                        logging.error(f"BTEX column {col} is not numeric")
                        return False
                else:
                    logging.warning(f"Expected BTEX column {col} not found in output")
            
            # Check datetime ordering preserved
            if not integrated_df['datetime'].is_monotonic_increasing:
                logging.error("Datetime ordering not preserved")
                return False
            
            # Coverage summary
            logging.info("BTEX Coverage Summary:")
            for col in self.voc_columns:
                if col in integrated_df.columns:
                    non_na_count = integrated_df[col].notna().sum()
                    total_rows = len(integrated_df)
                    coverage = (non_na_count / total_rows) * 100
                    logging.info(f"  {col}: {non_na_count}/{total_rows} ({coverage:.2f}%)")
            
            logging.info("BTEX integration validation passed")
            return True
            
        except Exception as e:
            logging.error(f"Error validating integration for {station}: {e}")
            return False
    
    def process_station(self, station, sheet_name):
        """Process BTEX integration for a single station."""
        try:
            logging.info(f"\n{'='*70}")
            logging.info(f"Processing BTEX integration for {station}")
            logging.info(f"{'='*70}")
            
            # Read BTEX data
            btex_df, btex_units = self.read_btex_sheet(sheet_name)
            if btex_df is None:
                logging.error(f"Failed to read BTEX data for {station}")
                return False
            
            # Load existing parquet
            parquet_df, parquet_file, metadata_file = self.load_existing_parquet(station)
            if parquet_df is None:
                logging.error(f"Failed to load existing parquet for {station}")
                return False
            
            # Integrate BTEX data
            integrated_df = self.integrate_btex_data(parquet_df, btex_df, station)
            if integrated_df is None:
                logging.error(f"Failed to integrate BTEX data for {station}")
                return False
            
            # Validate integration
            if not self.validate_integration(parquet_df, integrated_df, btex_df, station):
                logging.error(f"BTEX integration validation failed for {station}")
                return False
            
            # Save updated parquet
            if not self.save_updated_parquet(integrated_df, station, btex_units, parquet_file, metadata_file):
                logging.error(f"Failed to save updated parquet for {station}")
                return False
            
            logging.info(f"Successfully integrated BTEX data for {station}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {station}: {e}")
            return False

def main():
    """Main execution function."""
    integrator = BTEXIntegrator()
    
    logging.info("="*70)
    logging.info("BTEX DATA INTEGRATION STARTING")
    logging.info("="*70)
    
    # Create backup
    if not integrator.create_backup():
        logging.error("Failed to create backup. Aborting.")
        return
    
    # Define stations and their corresponding BTEX sheets
    stations = [
        ('MMF2', 'MMF2 data 30Min'),
        ('MMF9', 'MMF9 data 30Min')
    ]
    
    successful = []
    failed = []
    
    # Process each station
    for station, sheet_name in stations:
        if integrator.process_station(station, sheet_name):
            successful.append(station)
        else:
            failed.append(station)
    
    # Summary
    logging.info("\n" + "="*70)
    logging.info("BTEX INTEGRATION COMPLETE")
    logging.info("="*70)
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Backup created: {integrator.backup_dir}")
    
    if failed:
        logging.error("Some integrations failed. Check logs for details.")
        return 1
    else:
        logging.info("All BTEX integrations completed successfully!")
        return 0

if __name__ == "__main__":
    exit(main())