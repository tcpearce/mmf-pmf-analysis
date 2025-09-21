import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmf_processing.log'),
        logging.StreamHandler()
    ]
)

class MMFProcessor:
    def __init__(self, output_dir="mmf_parquet"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define column mappings with units
        self.gas_columns = {
            'DATE': {'unit': 'date', 'type': 'datetime'},
            'TIME': {'unit': 'time', 'type': 'time'},
            'WD': {'unit': 'degr', 'type': 'float'},
            'WS': {'unit': 'm/s', 'type': 'float'},
            'H2S': {'unit': 'ug/m3', 'type': 'float'},
            'CH4': {'unit': 'mg/m3', 'type': 'float'},
            'SO2': {'unit': 'ug/m3', 'type': 'float'}
        }
        
        self.particle_columns = {
            'DATE': {'unit': 'date', 'type': 'datetime'},
            'TIME': {'unit': 'time', 'type': 'time'},
            'PM1 FIDAS': {'unit': 'ug/m3', 'type': 'float'},
            'PM2.5': {'unit': 'ug/m3', 'type': 'float'},
            'PM4 FIDAS': {'unit': 'ug/m3', 'type': 'float'},
            'PM10': {'unit': 'ug/m3', 'type': 'float'},
            'TSP': {'unit': 'ug/m3', 'type': 'float'},
            'TEMP': {'unit': 'oC', 'type': 'float'},
            'AMB_PRES': {'unit': 'hPa', 'type': 'float'}
        }

    def read_and_clean_sheet(self, filepath, sheet_name, expected_columns):
        """Read a sheet and clean the data."""
        try:
            df = pd.read_excel(filepath, sheet_name=sheet_name)
            
            # Find the header row (look for DATE column)
            header_row = None
            for i, row in df.iterrows():
                if 'DATE' in str(row.iloc[0]).upper():
                    header_row = i
                    break
            
            if header_row is None:
                logging.error(f"Could not find header row in sheet {sheet_name}")
                return None
            
            # Re-read with proper header
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=header_row)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Create datetime column
            if 'DATE' in df.columns and 'TIME' in df.columns:
                # Combine DATE and TIME columns
                df['datetime'] = pd.to_datetime(
                    df['DATE'].astype(str) + ' ' + df['TIME'].astype(str),
                    errors='coerce'
                )
                
                # Remove rows with invalid datetime
                df = df.dropna(subset=['datetime'])
                
                # Sort by datetime
                df = df.sort_values('datetime')
                
                # Remove duplicate datetimes
                df = df.drop_duplicates(subset=['datetime'])
                
                logging.info(f"Sheet {sheet_name}: {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error reading sheet {sheet_name}: {str(e)}")
            return None

    def align_timeseries(self, gas_data, particle_data):
        """Align gas (5-min) and particle (15-min) data on a 5-minute timebase."""
        
        # Determine the overall time range
        start_time = min(gas_data['datetime'].min(), particle_data['datetime'].min())
        end_time = max(gas_data['datetime'].max(), particle_data['datetime'].max())
        
        # Create 5-minute timebase
        timebase = pd.date_range(
            start=start_time.floor('5min'),
            end=end_time.ceil('5min'),
            freq='5min'
        )
        
        logging.info(f"Created timebase: {len(timebase)} timestamps from {timebase[0]} to {timebase[-1]}")
        
        # Create base DataFrame
        aligned_df = pd.DataFrame({'datetime': timebase})
        
        # Merge gas data (5-minute intervals)
        gas_data_clean = gas_data.copy()
        gas_data_clean = gas_data_clean.drop(['DATE', 'TIME'], axis=1, errors='ignore')
        aligned_df = aligned_df.merge(gas_data_clean, on='datetime', how='left', suffixes=('', '_gas'))
        
        # For particle data (15-minute intervals), we need to interpolate or forward-fill
        particle_data_clean = particle_data.copy()
        particle_data_clean = particle_data_clean.drop(['DATE', 'TIME'], axis=1, errors='ignore')
        
        # Merge particle data
        aligned_df = aligned_df.merge(particle_data_clean, on='datetime', how='left', suffixes=('', '_particle'))
        
        # Forward fill particle data to match 5-minute intervals
        particle_cols = [col for col in aligned_df.columns if col not in ['datetime'] and col in self.particle_columns.keys()]
        for col in particle_cols:
            if col in aligned_df.columns:
                aligned_df[col] = aligned_df[col].fillna(method='ffill')
        
        return aligned_df

    def add_metadata_to_parquet(self, df, station_name):
        """Add metadata to the parquet file."""
        
        # Create metadata dictionary
        metadata = {
            'station': station_name,
            'processed_date': datetime.now().isoformat(),
            'description': f'Air quality data from {station_name}',
            'gas_measurement_frequency': '5 minutes',
            'particle_measurement_frequency': '15 minutes (forward-filled to 5 minutes)',
            'time_alignment': '5-minute timebase'
        }
        
        # Add column units as metadata
        for col in df.columns:
            if col == 'datetime':
                continue
            elif col in self.gas_columns:
                metadata[f'{col}_unit'] = self.gas_columns[col]['unit']
            elif col in self.particle_columns:
                metadata[f'{col}_unit'] = self.particle_columns[col]['unit']
        
        # Convert DataFrame to PyArrow table with metadata
        table = pa.Table.from_pandas(df)
        
        # Add custom metadata
        custom_metadata = {key: str(value) for key, value in metadata.items()}
        existing_metadata = table.schema.pandas_metadata
        if existing_metadata:
            existing_metadata.update(custom_metadata)
        else:
            existing_metadata = custom_metadata
            
        # Create new schema with metadata
        new_schema = table.schema.with_metadata(existing_metadata)
        table = table.cast(new_schema)
        
        return table

    def process_mmf_file(self, filepath, station_name):
        """Process a single MMF Excel file."""
        try:
            logging.info(f"\nProcessing {station_name}: {filepath}")
            
            # Read the Excel file to get sheet names
            excel_file = pd.ExcelFile(filepath)
            sheets = excel_file.sheet_names
            logging.info(f"Available sheets: {sheets}")
            
            # Assume sheet 1 (index 1) is gas data, sheet 2 (index 2) is particle data
            if len(sheets) < 3:
                logging.error(f"Expected at least 3 sheets, found {len(sheets)}")
                return False
            
            gas_sheet = sheets[1]  # Second sheet
            particle_sheet = sheets[2]  # Third sheet
            
            # Read gas data (5-minute intervals)
            gas_data = self.read_and_clean_sheet(filepath, gas_sheet, self.gas_columns)
            if gas_data is None:
                return False
            
            # Read particle data (15-minute intervals)
            particle_data = self.read_and_clean_sheet(filepath, particle_sheet, self.particle_columns)
            if particle_data is None:
                return False
            
            # Align timeseries on 5-minute timebase
            aligned_data = self.align_timeseries(gas_data, particle_data)
            
            # Add data quality flags
            aligned_data['gas_data_available'] = aligned_data['H2S'].notna()
            aligned_data['particle_data_available'] = aligned_data['PM2.5'].notna()
            
            logging.info(f"Final aligned dataset: {len(aligned_data)} records")
            logging.info(f"Gas data availability: {aligned_data['gas_data_available'].sum()} / {len(aligned_data)} records")
            logging.info(f"Particle data availability: {aligned_data['particle_data_available'].sum()} / {len(aligned_data)} records")
            
            # Save to parquet
            output_path = self.output_dir / f"{station_name}_combined_data.parquet"
            
            # Add metadata and save
            table = self.add_metadata_to_parquet(aligned_data, station_name)
            pq.write_table(table, output_path)
            
            logging.info(f"Saved parquet file: {output_path}")
            
            # Create summary file
            summary_path = self.output_dir / f"{station_name}_data_summary.txt"
            self.create_summary_file(aligned_data, station_name, summary_path)
            
            return True
            
        except Exception as e:
            logging.error(f"Error processing {station_name}: {str(e)}")
            return False

    def create_summary_file(self, df, station_name, output_path):
        """Create a summary file for the processed data."""
        with open(output_path, 'w') as f:
            f.write(f"Data Summary for {station_name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total records: {len(df)}\n")
            f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n")
            f.write(f"Time interval: 5 minutes\n\n")
            
            f.write("Data Availability:\n")
            for col in df.columns:
                if col not in ['datetime', 'gas_data_available', 'particle_data_available']:
                    available = df[col].notna().sum()
                    percentage = (available / len(df)) * 100
                    f.write(f"  {col}: {available} / {len(df)} ({percentage:.1f}%)\n")
            
            f.write("\nData Statistics:\n")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['gas_data_available', 'particle_data_available']:
                    stats = df[col].describe()
                    f.write(f"\n{col}:\n")
                    f.write(f"  Count: {int(stats['count'])}\n")
                    f.write(f"  Mean: {stats['mean']:.3f}\n")
                    f.write(f"  Min: {stats['min']:.3f}\n")
                    f.write(f"  Max: {stats['max']:.3f}\n")

    def verify_data_integrity(self, original_excel, parquet_path, station_name):
        """Verify that the parquet file contains all the original data."""
        try:
            # Read parquet file
            parquet_df = pd.read_parquet(parquet_path)
            
            # Read original Excel sheets
            gas_data = self.read_and_clean_sheet(original_excel, 1, self.gas_columns)
            particle_data = self.read_and_clean_sheet(original_excel, 2, self.particle_columns)
            
            verification_results = {}
            
            # Check gas data
            gas_count_original = len(gas_data) if gas_data is not None else 0
            gas_count_parquet = parquet_df['gas_data_available'].sum()
            verification_results['gas_records'] = {
                'original': gas_count_original,
                'parquet': gas_count_parquet,
                'match': gas_count_original == gas_count_parquet
            }
            
            # Check particle data  
            particle_count_original = len(particle_data) if particle_data is not None else 0
            # For particles, we need to count unique 15-minute intervals
            if particle_data is not None:
                particle_intervals = len(particle_data)
            else:
                particle_intervals = 0
                
            verification_results['particle_records'] = {
                'original': particle_intervals,
                'parquet_unique': parquet_df['particle_data_available'].sum(),
                'note': 'Particle data is forward-filled from 15-min to 5-min intervals'
            }
            
            # Save verification results
            verification_path = self.output_dir / f"{station_name}_verification.txt"
            with open(verification_path, 'w') as f:
                f.write(f"Data Verification Report for {station_name}\n")
                f.write("=" * 50 + "\n\n")
                for key, result in verification_results.items():
                    f.write(f"{key}:\n")
                    for subkey, value in result.items():
                        f.write(f"  {subkey}: {value}\n")
                    f.write("\n")
            
            logging.info(f"Verification complete for {station_name}")
            return verification_results
            
        except Exception as e:
            logging.error(f"Error verifying data for {station_name}: {str(e)}")
            return None

def main():
    # Install pyarrow if not available
    try:
        import pyarrow
    except ImportError:
        logging.error("PyArrow is required. Install with: pip install pyarrow")
        return
    
    processor = MMFProcessor()
    
    # Process each MMF station
    mmf_files = {
        'MMF1': Path('mmf_data/MMF1/processed/Silverdale Ambient Air Monitoring Data - MMF1 - Mar 21 to Aug 23.xlsx'),
        'MMF2': Path('mmf_data/MMF2/processed/Silverdale Ambient Air Monitoring Data - MMF2 - Mar 21 to Aug 23.xlsx'),
        'MMF6': Path('mmf_data/MMF6/processed/Silverdale Ambient Air Monitoring Data - MMF6 - Mar 21 to June 23.xlsx'),
        'MMF9': Path('mmf_data/MMF9/processed/Silverdale Ambient Air Monitoring Data - MMF9 - Mar 21 to Aug 23.xlsx')
    }
    
    successful_conversions = []
    failed_conversions = []
    
    for station, filepath in mmf_files.items():
        if filepath.exists():
            success = processor.process_mmf_file(filepath, station)
            if success:
                successful_conversions.append(station)
                # Verify data integrity
                parquet_path = processor.output_dir / f"{station}_combined_data.parquet"
                processor.verify_data_integrity(filepath, parquet_path, station)
            else:
                failed_conversions.append(station)
        else:
            logging.error(f"File not found: {filepath}")
            failed_conversions.append(station)
    
    # Create overall processing report
    report_path = processor.output_dir / "processing_report.md"
    with open(report_path, 'w') as f:
        f.write("# MMF Data Processing Report\n\n")
        f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Data Structure\n")
        f.write("- **Sheet 1**: Metadata (not processed)\n")
        f.write("- **Sheet 2**: Gas measurements (5-minute intervals)\n")
        f.write("  - Columns: DATE, TIME, WD (degr), WS (m/s), H2S (ug/m3), CH4 (mg/m3), SO2 (ug/m3)\n")
        f.write("- **Sheet 3**: Particle measurements (15-minute intervals)\n")
        f.write("  - Columns: DATE, TIME, PM1 FIDAS (ug/m3), PM2.5 (ug/m3), PM4 FIDAS (ug/m3), PM10 (ug/m3), TSP (ug/m3), TEMP (oC), AMB_PRES (hPa)\n\n")
        
        f.write("## Processing Method\n")
        f.write("1. Extract gas data (5-minute intervals) from sheet 2\n")
        f.write("2. Extract particle data (15-minute intervals) from sheet 3\n")
        f.write("3. Create unified 5-minute timebase covering full date range\n")
        f.write("4. Align gas data to 5-minute timebase\n")
        f.write("5. Forward-fill particle data to match 5-minute intervals\n")
        f.write("6. Add data availability flags\n")
        f.write("7. Save as parquet with metadata\n\n")
        
        f.write("## Results\n")
        f.write(f"Successfully processed: {', '.join(successful_conversions) if successful_conversions else 'None'}\n")
        f.write(f"Failed to process: {', '.join(failed_conversions) if failed_conversions else 'None'}\n\n")
        
        f.write("## Output Files\n")
        for station in successful_conversions:
            f.write(f"- {station}_combined_data.parquet\n")
            f.write(f"- {station}_data_summary.txt\n")
            f.write(f"- {station}_verification.txt\n")
    
    logging.info(f"\nProcessing complete. Results saved to {processor.output_dir}")
    logging.info(f"Successfully processed: {len(successful_conversions)} stations")
    logging.info(f"Failed: {len(failed_conversions)} stations")

if __name__ == "__main__":
    main()
