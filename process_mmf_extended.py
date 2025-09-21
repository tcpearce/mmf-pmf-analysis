import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings
import pyarrow as pa
import pyarrow.parquet as pq

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmf_processing_extended.log'),
        logging.StreamHandler()
    ]
)

class ExtendedMMFProcessor:
    def __init__(self, output_dir="mmf_parquet"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Backup directory for existing files
        self.backup_dir = self.output_dir / "backup"
        self.backup_dir.mkdir(exist_ok=True)

    def backup_existing_files(self, station_name):
        """Backup existing parquet files before processing."""
        existing_files = [
            f"{station_name}_combined_data.parquet",
            f"{station_name}_metadata.txt", 
            f"{station_name}_summary.txt"
        ]
        
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for filename in existing_files:
            source_path = self.output_dir / filename
            if source_path.exists():
                backup_path = self.backup_dir / f"{filename}.backup_{backup_timestamp}"
                source_path.replace(backup_path)
                logging.info(f"Backed up {filename} to {backup_path}")

    def load_existing_parquet(self, station_name):
        """Load existing parquet file if it exists."""
        parquet_path = self.output_dir / f"{station_name}_combined_data.parquet"
        
        if parquet_path.exists():
            try:
                df = pd.read_parquet(parquet_path)
                logging.info(f"Loaded existing {station_name} data: {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
                return df
            except Exception as e:
                logging.error(f"Error loading existing {station_name} data: {e}")
                return None
        else:
            logging.info(f"No existing parquet file found for {station_name}")
            return None

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
            pm25_col = 'PM2.5' if 'PM2.5' in combined_df.columns else None
            
            combined_df['gas_data_available'] = combined_df[h2s_col].notna() if h2s_col else False
            combined_df['particle_data_available'] = combined_df[pm25_col].notna() if pm25_col else False
            
            logging.info(f"Combined dataset: {len(combined_df)} records")
            logging.info(f"Gas data points: {combined_df['gas_data_available'].sum()}")
            logging.info(f"Particle data points: {combined_df['particle_data_available'].sum()}")
            
            return combined_df
            
        except Exception as e:
            logging.error(f"Error aligning data: {str(e)}")
            return None

    def merge_with_existing_data(self, new_df, existing_df, station_name):
        """Merge new data with existing data, overwriting overlaps."""
        if existing_df is None:
            logging.info(f"{station_name}: No existing data to merge, using new data only")
            return new_df
        
        try:
            logging.info(f"{station_name}: Merging new data with existing data")
            logging.info(f"  Existing: {len(existing_df)} records, {existing_df['datetime'].min()} to {existing_df['datetime'].max()}")
            logging.info(f"  New: {len(new_df)} records, {new_df['datetime'].min()} to {new_df['datetime'].max()}")
            
            # Find overlap periods
            new_start = new_df['datetime'].min()
            new_end = new_df['datetime'].max()
            existing_start = existing_df['datetime'].min()
            existing_end = existing_df['datetime'].max()
            
            overlap_start = max(new_start, existing_start)
            overlap_end = min(new_end, existing_end)
            
            if overlap_start <= overlap_end:
                logging.info(f"  Overlap period: {overlap_start} to {overlap_end}")
                
                # Remove overlapping data from existing dataset
                existing_clean = existing_df[~((existing_df['datetime'] >= overlap_start) & 
                                             (existing_df['datetime'] <= overlap_end))]
                logging.info(f"  Removed {len(existing_df) - len(existing_clean)} overlapping records from existing data")
            else:
                existing_clean = existing_df
                logging.info("  No temporal overlap found")
            
            # Combine datasets
            combined_df = pd.concat([existing_clean, new_df], ignore_index=True)
            combined_df = combined_df.sort_values('datetime').drop_duplicates(subset=['datetime'])
            
            logging.info(f"  Final combined dataset: {len(combined_df)} records from {combined_df['datetime'].min()} to {combined_df['datetime'].max()}")
            
            return combined_df
            
        except Exception as e:
            logging.error(f"Error merging {station_name} data: {str(e)}")
            return new_df

    def save_to_parquet(self, df, station_name, all_units):
        """Save DataFrame to parquet with units metadata."""
        try:
            output_path = self.output_dir / f"{station_name}_combined_data.parquet"
            
            # Create PyArrow table
            table = pa.Table.from_pandas(df)
            
            # Create metadata with units
            metadata = {}
            for col in df.columns:
                if col in all_units:
                    metadata[f"{col}_unit"] = all_units[col]
                elif col == 'datetime':
                    metadata[f"{col}_unit"] = 'timestamp'
                elif 'available' in col:
                    metadata[f"{col}_unit"] = 'boolean'
            
            # Add processing metadata
            metadata['processing_date'] = datetime.now().isoformat()
            metadata['station'] = station_name
            metadata['data_frequency'] = '5min'
            metadata['source'] = 'MMF Excel files (extended dataset)'
            metadata['processing_type'] = 'merged_extended'
            
            # Convert metadata to bytes for PyArrow
            arrow_metadata = {k.encode(): str(v).encode() for k, v in metadata.items()}
            
            # Update table metadata
            existing_metadata = table.schema.metadata or {}
            existing_metadata.update(arrow_metadata)
            table = table.replace_schema_metadata(existing_metadata)
            
            # Save with metadata
            pq.write_table(table, output_path)
            
            logging.info(f"Saved parquet file with units metadata: {output_path} ({len(df)} records)")
            logging.info(f"Units stored: {len([k for k in metadata.keys() if k.endswith('_unit')])} columns")
            
            # Create metadata file
            metadata_path = self.output_dir / f"{station_name}_metadata.txt"
            with open(metadata_path, 'w') as f:
                f.write(f"Metadata for {station_name} (Extended Dataset)\\n")
                f.write("=" * 50 + "\\n\\n")
                f.write(f"File: {station_name}_combined_data.parquet\\n")
                f.write(f"Records: {len(df)}\\n")
                f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\\n")
                f.write(f"Time span: {(df['datetime'].max() - df['datetime'].min()).days} days\\n")
                f.write(f"Time interval: 5 minutes\\n")
                f.write(f"Processing: Extended dataset with merge/overwrite logic\\n\\n")
                
                f.write("Processing notes:\\n")
                f.write("- Gas data: 5-minute intervals (original frequency)\\n")
                f.write("- Particle data: 15-minute intervals forward-filled to 5-minute\\n")
                f.write("- Missing values preserved as NaN\\n")
                f.write("- Timezone information removed for consistency\\n")
                f.write("- New data overwrites existing data in overlap periods\\n\\n")
                
                f.write("Columns and units:\\n")
                for col in df.columns:
                    f.write(f"  {col}")
                    # Add units based on column names and stored units
                    if col in all_units:
                        f.write(f" ({all_units[col]})")
                    elif 'datetime' in col:
                        f.write(" (timestamp)")
                    elif 'available' in col:
                        f.write(" (boolean flag)")
                    f.write("\\n")
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving parquet for {station_name}: {str(e)}")
            return False

    def create_summary(self, df, station_name):
        """Create a data summary file."""
        try:
            summary_path = self.output_dir / f"{station_name}_summary.txt"
            
            with open(summary_path, 'w') as f:
                f.write(f"Data Summary for {station_name} (Extended Dataset)\\n")
                f.write("=" * 60 + "\\n\\n")
                f.write(f"Total records: {len(df)}\\n")
                f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\\n")
                f.write(f"Time span: {(df['datetime'].max() - df['datetime'].min()).days} days\\n\\n")
                
                f.write("Data Availability:\\n")
                for col in df.columns:
                    if col not in ['datetime', 'gas_data_available', 'particle_data_available']:
                        available = df[col].notna().sum()
                        percentage = (available / len(df)) * 100
                        f.write(f"  {col}: {available:,} / {len(df):,} ({percentage:.1f}%)\\n")
                
                f.write("\\nBasic Statistics (non-zero values only):\\n")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['gas_data_available', 'particle_data_available']:
                        # Filter out zeros for better statistics
                        data = df[col][df[col] > 0]
                        if len(data) > 0:
                            f.write(f"\\n{col}:\\n")
                            f.write(f"  Non-zero count: {len(data):,}\\n")
                            f.write(f"  Mean: {data.mean():.3f}\\n")
                            f.write(f"  Median: {data.median():.3f}\\n")
                            f.write(f"  Min: {data.min():.3f}\\n")
                            f.write(f"  Max: {data.max():.3f}\\n")
            
            logging.info(f"Created summary: {summary_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating summary for {station_name}: {str(e)}")
            return False

    def process_mmf_file(self, filepath, station_name):
        """Process a single MMF Excel file with merge logic."""
        try:
            logging.info(f"\\n{'='*70}")
            logging.info(f"Processing {station_name}: {filepath.name}")
            logging.info(f"{'='*70}")
            
            # Backup existing files
            self.backup_existing_files(station_name)
            
            # Load existing data
            existing_data = self.load_existing_parquet(station_name)
            
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
            
            # Merge with existing data (overwrite overlaps)
            final_data = self.merge_with_existing_data(combined_data, existing_data, station_name)
            
            # Save to parquet with units
            if not self.save_to_parquet(final_data, station_name, all_units):
                return False
            
            # Create summary
            if not self.create_summary(final_data, station_name):
                return False
            
            logging.info(f"Successfully processed {station_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {station_name}: {str(e)}")
            return False

def main():
    processor = ExtendedMMFProcessor()
    
    # Define new MMF files (raw data)
    mmf_files = {
        'MMF1': Path('mmf_data/MMF1/raw/c39163361bc4854cac6f969b148b4c64_Silverdale Ambient Air Monitoring Data - MMF Silverdale Pumping Station - Mar 2021 to July 2025.xlsx'),
        'MMF2': Path('mmf_data/MMF2/raw/7969ed6f77e41d4fd840a70cd840d42f_Silverdale_Ambient_Air_Monitoring_Data_-_Cemetery_Road_-_Mar_2021_-_Aug_2024.xlsx'),
        'MMF6': Path('mmf_data/MMF6/raw/61379dace1c94403959b18fbd97184b7_Silverdale Ambient Air Monitoring Data -MMF Galingale View - Mar 2021 to Jul 2025.xlsx'),
        'MMF9': Path('mmf_data/MMF9/raw/990db34b09e3de9643eb085dc78ce32d_Silverdale Ambient Air Monitoring Data - MMF Maries Way - Sep 2024 to July 2025.xlsx')
    }
    
    successful = []
    failed = []
    
    # Process one file at a time
    for station, filepath in mmf_files.items():
        if filepath.exists():
            logging.info(f"\\nStarting processing of {station}...")
            if processor.process_mmf_file(filepath, station):
                successful.append(station)
            else:
                failed.append(station)
        else:
            logging.error(f"File not found: {filepath}")
            failed.append(station)
    
    # Create overall report
    report_path = processor.output_dir / "processing_report_extended.md"
    with open(report_path, 'w') as f:
        f.write("# MMF Data Processing Report (Extended Dataset)\\n\\n")
        f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
        
        f.write("## Summary\\n")
        f.write(f"- Successfully processed: {len(successful)} stations\\n")
        f.write(f"- Failed: {len(failed)} stations\\n\\n")
        
        if successful:
            f.write("### Successful:\\n")
            for station in successful:
                f.write(f"- {station}\\n")
            f.write("\\n")
        
        if failed:
            f.write("### Failed:\\n")
            for station in failed:
                f.write(f"- {station}\\n")
            f.write("\\n")
        
        f.write("## Processing Details\\n")
        f.write("- **Extended Dataset Processing:**\\n")
        f.write("  - Loaded existing parquet files where available\\n")
        f.write("  - Processed new extended XLSX files\\n")
        f.write("  - Merged datasets with overlap overwrite logic\\n")
        f.write("  - Backed up original files before processing\\n\\n")
        
        f.write("- **Data Coverage (New Extended Files):**\\n")
        f.write("  - MMF1: 2021-03-05 to 2025-07-01 (extends backward and forward)\\n")
        f.write("  - MMF2: 2021-04-14 to 2024-08-31 (extends forward 1 year)\\n")
        f.write("  - MMF6: 2021-03-06 to 2025-07-31 (extends backward and forward)\\n")
        f.write("  - MMF9: 2024-09-02 to 2025-07-31 (new data post-2023)\\n\\n")
        
        f.write("- **Data Structure:**\\n")
        f.write("  - Gas data: 5-minute intervals (WD, WS, H2S, CH4, SO2, NOX, NO, NO2)\\n")
        f.write("  - Particle data: 15-minute intervals (PM1, PM2.5, PM4, PM10, TSP, TEMP, AMB_PRES)\\n")
        f.write("  - Combined on 5-minute timebase\\n")
        f.write("  - Particle data forward-filled (max 3 intervals)\\n")
        f.write("  - Data availability flags added\\n")
        f.write("  - Saved as parquet files with metadata\\n")
        f.write("  - Overlap periods: New data overwrites existing data\\n")
    
    logging.info(f"\\n{'='*70}")
    logging.info("EXTENDED PROCESSING COMPLETE!")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Results saved to: {processor.output_dir}")
    logging.info(f"Backups saved to: {processor.backup_dir}")
    logging.info(f"{'='*70}")

if __name__ == "__main__":
    main()