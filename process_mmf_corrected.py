#!/usr/bin/env python3
"""
Corrected MMF processing script with proper station name mappings.
Based on process_mmf_extended.py but uses correct directory structure and adds MMF IDs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
import json

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmf_processing_corrected.log'),
        logging.StreamHandler()
    ]
)

class CorrectedMMFProcessor:
    def __init__(self, output_dir="mmf_parquet_corrected"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load station mapping
        with open('station_lookup.json', 'r') as f:
            station_config = json.load(f)
        
        # Extract clean mapping (remove metadata)
        self.station_mapping = {k: v for k, v in station_config.items() if not k.startswith('_')}
        
        # Reverse mapping for lookups
        self.station_to_mmf = {v: k for k, v in self.station_mapping.items() if v is not None}
        self.station_to_mmf['Maries Way'] = None  # Special case
        
        logging.info(f"Station mapping loaded: {self.station_mapping}")

    def get_mmf_info(self, directory_name):
        """Extract MMF ID and station name from corrected directory structure."""
        mapping = {
            'MMF1_Cemetery_Road': ('1', 'Cemetery Road'),
            'MMF2_Silverdale_Pumping_Station': ('2', 'Silverdale Pumping Station'),
            'MMF6_Fire_Station': ('6', 'Fire Station'),
            'MMF9_Galingale_View': ('9', 'Galingale View'),
            'Maries_Way': (None, 'Maries Way')
        }
        return mapping.get(directory_name, (None, None))

    def find_header_row(self, filepath, sheet_name):
        """Find the row containing the data headers."""
        try:
            # Read first 10 rows to find header
            df_sample = pd.read_excel(filepath, sheet_name=sheet_name, nrows=10, header=None)
            
            for i in range(len(df_sample)):
                row_values = df_sample.iloc[i].astype(str).str.upper()
                if any('DATE' in str(val) for val in row_values):
                    logging.info(f"Found header row at index {i} in sheet {sheet_name}")
                    return i
            
            # If no DATE found, check if first row looks like headers
            first_row = df_sample.iloc[0].astype(str)
            if any(col in first_row.str.upper().values for col in ['DATE', 'TIME', 'H2S', 'PM']):
                logging.info(f"Using row 0 as header in sheet {sheet_name}")
                return 0
            
            logging.warning(f"No clear header row found in sheet {sheet_name}, using row 0")
            return 0
            
        except Exception as e:
            logging.error(f"Error finding header in {sheet_name}: {str(e)}")
            return 0

    def fix_datetime_parsing(self, df):
        """Fix datetime parsing issues with mixed timezones."""
        try:
            # Create datetime column with proper handling
            datetime_series = []
            
            for idx, row in df.iterrows():
                try:
                    date_str = str(row['DATE']).strip()
                    time_str = str(row['TIME']).strip()
                    
                    # Skip if either is NaN or empty
                    if date_str in ['nan', 'NaT', ''] or time_str in ['nan', 'NaT', '']:
                        datetime_series.append(pd.NaT)
                        continue
                    
                    # Combine date and time strings
                    combined_str = f"{date_str} {time_str}"
                    
                    # Parse datetime with UTC to avoid timezone issues
                    dt = pd.to_datetime(combined_str, utc=True, errors='coerce')
                    
                    # Convert to timezone-naive
                    if dt is not pd.NaT and dt.tz is not None:
                        dt = dt.tz_convert(None)
                    
                    datetime_series.append(dt)
                    
                except Exception as e:
                    datetime_series.append(pd.NaT)
                    continue
            
            df['datetime'] = datetime_series
            
            # Remove rows with invalid datetime
            initial_rows = len(df)
            df = df.dropna(subset=['datetime'])
            logging.info(f"Removed {initial_rows - len(df)} rows with invalid datetime")
            
            # Sort by datetime and remove duplicates
            df = df.sort_values('datetime').drop_duplicates(subset=['datetime'])
            
            return df
            
        except Exception as e:
            logging.error(f"Error in datetime parsing: {str(e)}")
            return None

    def extract_units_from_excel(self, filepath, sheet_name, header_row):
        """Extract units from Excel row following the header row."""
        try:
            # Read the units row (header_row + 1)
            units_df = pd.read_excel(filepath, sheet_name=sheet_name, 
                                   skiprows=header_row, nrows=2, header=None)
            
            if len(units_df) < 2:
                logging.warning(f"Not enough rows to extract units from {sheet_name}")
                return {}
            
            # Get headers and units
            headers = units_df.iloc[0].values
            units = units_df.iloc[1].values
            
            # Create units dictionary
            units_dict = {}
            for header, unit in zip(headers, units):
                if pd.notna(header) and pd.notna(unit):
                    header_str = str(header).strip()
                    unit_str = str(unit).strip()
                    
                    # Clean up unit format (remove parentheses, etc.)
                    unit_clean = unit_str.replace('(', '').replace(')', '')
                    
                    # Skip DATE/TIME as they don't need units stored
                    if header_str not in ['DATE', 'TIME']:
                        units_dict[header_str] = unit_clean
            
            logging.info(f"Extracted units from {sheet_name}: {units_dict}")
            return units_dict
            
        except Exception as e:
            logging.error(f"Error extracting units from {sheet_name}: {str(e)}")
            return {}

    def read_sheet_data(self, filepath, sheet_name):
        """Read data from a specific sheet with improved error handling and unit extraction."""
        try:
            # Find header row
            header_row = self.find_header_row(filepath, sheet_name)
            
            # Extract units from the row following headers
            units_dict = self.extract_units_from_excel(filepath, sheet_name, header_row)
            
            # Read with identified header
            logging.info(f"Reading sheet {sheet_name} with header at row {header_row}")
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=header_row)
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            logging.info(f"Sheet {sheet_name}: {len(df)} rows, columns: {list(df.columns)}")
            
            # Check if we have the required columns
            if 'DATE' not in df.columns or 'TIME' not in df.columns:
                logging.error(f"Required columns DATE/TIME not found in {sheet_name}")
                return None, {}
            
            # Fix datetime parsing
            df = self.fix_datetime_parsing(df)
            if df is None:
                return None, {}
            
            logging.info(f"Final data: {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df, units_dict
            
        except Exception as e:
            logging.error(f"Error reading sheet {sheet_name}: {str(e)}")
            return None, {}

    def align_and_combine_data(self, gas_df, particle_df):
        """Align gas and particle data on 5-minute intervals."""
        try:
            # Determine overall time range
            start_time = min(gas_df['datetime'].min(), particle_df['datetime'].min())
            end_time = max(gas_df['datetime'].max(), particle_df['datetime'].max())
            
            # Create 5-minute timebase
            timebase = pd.date_range(
                start=start_time.floor('5min'),
                end=end_time.ceil('5min'),
                freq='5min'
            )
            
            logging.info(f"Created timebase: {len(timebase)} timestamps from {timebase[0]} to {timebase[-1]}")
            
            # Create base DataFrame
            combined_df = pd.DataFrame({'datetime': timebase})
            
            # Prepare gas data (remove DATE/TIME, keep datetime)
            gas_clean = gas_df.drop(['DATE', 'TIME'], axis=1, errors='ignore')
            
            # Prepare particle data (remove DATE/TIME, keep datetime)
            particle_clean = particle_df.drop(['DATE', 'TIME'], axis=1, errors='ignore')
            
            # Merge gas data
            logging.info("Merging gas data...")
            combined_df = combined_df.merge(gas_clean, on='datetime', how='left')
            
            # Merge particle data
            logging.info("Merging particle data...")
            combined_df = combined_df.merge(particle_clean, on='datetime', how='left', suffixes=('', '_particle'))
            
            # Forward fill particle data to match 5-minute intervals (within reasonable limits)
            particle_cols = [col for col in combined_df.columns 
                           if any(pm in col for pm in ['PM', 'TSP', 'TEMP', 'AMB_PRES'])]
            
            logging.info(f"Forward filling particle columns: {particle_cols}")
            for col in particle_cols:
                if col in combined_df.columns:
                    # Forward fill with limit to avoid filling very long gaps
                    combined_df[col] = combined_df[col].fillna(method='ffill', limit=3)
            
            # Add data availability flags
            h2s_col = 'H2S' if 'H2S' in combined_df.columns else None
            pm25_col = 'PM2.5' if 'PM2.5' in combined_df.columns else 'PM2.5 FIDAS' if 'PM2.5 FIDAS' in combined_df.columns else None
            
            combined_df['gas_data_available'] = combined_df[h2s_col].notna() if h2s_col else False
            combined_df['particle_data_available'] = combined_df[pm25_col].notna() if pm25_col else False
            
            logging.info(f"Combined dataset: {len(combined_df)} records")
            logging.info(f"Gas data points: {combined_df['gas_data_available'].sum()}")
            logging.info(f"Particle data points: {combined_df['particle_data_available'].sum()}")
            
            return combined_df
            
        except Exception as e:
            logging.error(f"Error aligning data: {str(e)}")
            return None

    def save_to_parquet(self, df, mmf_id, station_name, all_units):
        """Save DataFrame to parquet with correct MMF ID, station name and units metadata."""
        try:
            # Add MMF ID and station name columns
            df = df.copy()
            df['mmf_id'] = mmf_id
            df['station_name'] = station_name
            
            # Reorder columns to put identifiers first
            identifier_cols = ['datetime', 'mmf_id', 'station_name']
            other_cols = [col for col in df.columns if col not in identifier_cols]
            df = df[identifier_cols + other_cols]
            
            # Create output filename
            if mmf_id:
                output_filename = f"MMF{mmf_id}_{station_name.replace(' ', '_')}_combined_data.parquet"
            else:
                output_filename = f"{station_name.replace(' ', '_')}_combined_data.parquet"
            
            output_path = self.output_dir / output_filename
            
            # Create PyArrow table
            table = pa.Table.from_pandas(df)
            
            # Create metadata with units
            metadata = {}
            for col in df.columns:
                if col in all_units:
                    metadata[f"{col}_unit"] = all_units[col]
                elif col == 'datetime':
                    metadata[f"{col}_unit"] = 'timestamp'
                elif col == 'mmf_id':
                    metadata[f"{col}_unit"] = 'identifier'
                elif col == 'station_name':
                    metadata[f"{col}_unit"] = 'text'
                elif 'available' in col:
                    metadata[f"{col}_unit"] = 'boolean'
            
            # Add processing metadata
            metadata['processing_date'] = datetime.now().isoformat()
            metadata['mmf_id'] = str(mmf_id) if mmf_id else 'null'
            metadata['station_name'] = station_name
            metadata['schema_version'] = 'v2'
            metadata['data_frequency'] = '5min'
            metadata['source'] = 'MMF Excel files (corrected mapping)'
            metadata['processing_type'] = 'corrected_extended'
            
            # Convert metadata to bytes for PyArrow
            arrow_metadata = {k.encode(): str(v).encode() for k, v in metadata.items()}
            
            # Update table metadata
            existing_metadata = table.schema.metadata or {}
            existing_metadata.update(arrow_metadata)
            table = table.replace_schema_metadata(existing_metadata)
            
            # Save with metadata
            pq.write_table(table, output_path)
            
            logging.info(f"Saved corrected parquet file: {output_filename} ({len(df)} records)")
            logging.info(f"MMF ID: {mmf_id}, Station: {station_name}")
            
            # Create metadata file
            metadata_path = self.output_dir / f"{output_filename.replace('.parquet', '_metadata.txt')}"
            with open(metadata_path, 'w') as f:
                f.write(f"Metadata for MMF{mmf_id} - {station_name} (Corrected Dataset)\\n")
                f.write("=" * 70 + "\\n\\n")
                f.write(f"File: {output_filename}\\n")
                f.write(f"MMF ID: {mmf_id}\\n")
                f.write(f"Station Name: {station_name}\\n")
                f.write(f"Records: {len(df)}\\n")
                f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\\n")
                f.write(f"Time span: {(df['datetime'].max() - df['datetime'].min()).days} days\\n")
                f.write(f"Processing: Corrected station mapping with MMF IDs\\n\\n")
                
                f.write("Corrections Applied:\\n")
                f.write("- Added correct MMF ID column\\n")
                f.write("- Added station name column\\n")
                f.write("- Used corrected station name mapping\\n")
                f.write("- Schema version upgraded to v2\\n\\n")
                
                f.write("Processing notes:\\n")
                f.write("- Gas data: 5-minute intervals (original frequency)\\n")
                f.write("- Particle data: 15-minute intervals forward-filled to 5-minute\\n")
                f.write("- Missing values preserved as NaN\\n")
                f.write("- Timezone information removed for consistency\\n\\n")
                
                f.write("Columns and units:\\n")
                for col in df.columns:
                    f.write(f"  {col}")
                    # Add units based on column names and stored units
                    if col in all_units:
                        f.write(f" ({all_units[col]})")
                    elif col == 'datetime':
                        f.write(" (timestamp)")
                    elif col == 'mmf_id':
                        f.write(" (identifier)")
                    elif col == 'station_name':
                        f.write(" (text)")
                    elif 'available' in col:
                        f.write(" (boolean flag)")
                    f.write("\\n")
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving parquet for MMF{mmf_id} - {station_name}: {str(e)}")
            return False

    def process_mmf_file(self, filepath, directory_name):
        """Process a single MMF Excel file with corrected metadata."""
        try:
            mmf_id, station_name = self.get_mmf_info(directory_name)
            
            logging.info(f"\\n{'='*70}")
            logging.info(f"Processing {directory_name}")
            logging.info(f"MMF ID: {mmf_id}, Station: {station_name}")
            logging.info(f"File: {filepath.name}")
            logging.info(f"{'='*70}")
            
            # Get sheet names
            excel_file = pd.ExcelFile(filepath)
            sheets = excel_file.sheet_names
            logging.info(f"Available sheets: {sheets}")
            
            if len(sheets) < 3:
                logging.error(f"Expected 3 sheets, found {len(sheets)}")
                return False
            
            # Read gas data (5-minute, typically sheet 1)
            gas_sheet = sheets[1]
            logging.info(f"Processing gas data from sheet: {gas_sheet}")
            gas_data, gas_units = self.read_sheet_data(filepath, gas_sheet)
            if gas_data is None:
                logging.error(f"Failed to read gas data from {gas_sheet}")
                return False
            
            # Read particle data (15-minute, typically sheet 2)
            particle_sheet = sheets[2]
            logging.info(f"Processing particle data from sheet: {particle_sheet}")
            particle_data, particle_units = self.read_sheet_data(filepath, particle_sheet)
            if particle_data is None:
                logging.error(f"Failed to read particle data from {particle_sheet}")
                return False
            
            # Combine units from both sheets
            all_units = {**gas_units, **particle_units}
            logging.info(f"Combined units: {all_units}")
            
            # Combine and align data
            logging.info("Aligning and combining datasets...")
            combined_data = self.align_and_combine_data(gas_data, particle_data)
            if combined_data is None:
                return False
            
            # Save to parquet with corrected metadata
            if not self.save_to_parquet(combined_data, mmf_id, station_name, all_units):
                return False
            
            logging.info(f"Successfully processed {directory_name} -> MMF{mmf_id} ({station_name})")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {directory_name}: {str(e)}")
            return False

def main():
    processor = CorrectedMMFProcessor()
    
    # Define corrected MMF files structure
    corrected_data_dir = Path('mmf_data_corrected')
    
    # Find all directories with raw data
    station_directories = []
    for station_dir in corrected_data_dir.iterdir():
        if station_dir.is_dir():
            raw_dir = station_dir / 'raw'
            if raw_dir.exists():
                excel_files = list(raw_dir.glob('*.xlsx'))
                if excel_files:
                    # Use the largest file (usually the comprehensive one)
                    excel_file = max(excel_files, key=lambda x: x.stat().st_size)
                    station_directories.append((station_dir.name, excel_file))
                elif station_dir.name == 'MMF6_Fire_Station':
                    # Handle empty Fire Station directory
                    logging.info(f"Fire Station directory empty as expected")
                    continue
                else:
                    logging.warning(f"No Excel files found in {raw_dir}")
    
    successful = []
    failed = []
    
    # Process each station
    for directory_name, filepath in station_directories:
        logging.info(f"\\nStarting processing of {directory_name}...")
        if processor.process_mmf_file(filepath, directory_name):
            successful.append(directory_name)
        else:
            failed.append(directory_name)
    
    # Handle Fire Station separately (create empty file)
    try:
        from regenerate_corrected_parquet import create_empty_fire_station_parquet
        create_empty_fire_station_parquet(processor.output_dir)
        successful.append('MMF6_Fire_Station')
        logging.info("Created empty Fire Station placeholder file")
    except Exception as e:
        logging.error(f"Error creating Fire Station placeholder: {e}")
        failed.append('MMF6_Fire_Station')
    
    # Create overall report
    report_path = processor.output_dir / "processing_report_corrected.md"
    with open(report_path, 'w') as f:
        f.write("# MMF Data Processing Report (Corrected Mapping)\\n\\n")
        f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("## Summary\\n")
        f.write(f"- Successfully processed: {len(successful)} stations\\n")
        f.write(f"- Failed: {len(failed)} stations\\n\\n")
        
        f.write("## Corrections Applied\\n")
        f.write("- **MMF1**: Now correctly points to Cemetery Road (was Silverdale Pumping Station)\\n")
        f.write("- **MMF2**: Now correctly points to Silverdale Pumping Station (was Cemetery Road)\\n")
        f.write("- **MMF6**: Fire Station (placeholder - no data available)\\n")
        f.write("- **MMF9**: Now correctly points to Galingale View (was Maries Way)\\n")
        f.write("- **Maries Way**: Now correctly has no MMF number\\n\\n")
        
        if successful:
            f.write("### Successfully Processed:\\n")
            for station in successful:
                f.write(f"- {station}\\n")
            f.write("\\n")
        
        if failed:
            f.write("### Failed:\\n")
            for station in failed:
                f.write(f"- {station}\\n")
            f.write("\\n")
        
        f.write("## Schema Changes\\n")
        f.write("- Added `mmf_id` column with correct MMF numbers\\n")
        f.write("- Added `station_name` column for clarity\\n")
        f.write("- Updated schema version to v2\\n")
        f.write("- Enhanced metadata with correction tracking\\n")
    
    logging.info(f"\\n{'='*70}")
    logging.info("CORRECTED PROCESSING COMPLETE!")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Results saved to: {processor.output_dir}")
    logging.info(f"{'='*70}")

if __name__ == "__main__":
    main()