import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmf_processing_improved.log'),
        logging.StreamHandler()
    ]
)

class MMFProcessor:
    def __init__(self, output_dir="mmf_parquet"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def find_header_row(self, filepath, sheet_name):
        """Find the row containing the data headers."""
        try:
            # Read first 20 rows to find header
            df_sample = pd.read_excel(filepath, sheet_name=sheet_name, nrows=20)
            
            for i in range(len(df_sample)):
                row_values = df_sample.iloc[i].astype(str).str.upper()
                if any('DATE' in str(val) for val in row_values):
                    logging.info(f"Found header row at index {i} in sheet {sheet_name}")
                    return i
            
            logging.warning(f"No header row found in sheet {sheet_name}, using row 0")
            return 0
            
        except Exception as e:
            logging.error(f"Error finding header in {sheet_name}: {str(e)}")
            return 0

    def read_sheet_data(self, filepath, sheet_name):
        """Read data from a specific sheet."""
        try:
            # Find header row
            header_row = self.find_header_row(filepath, sheet_name)
            
            # Read with identified header
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=header_row)
            
            # Clean column names
            df.columns = df.columns.astype(str).str.strip()
            
            logging.info(f"Sheet {sheet_name}: {len(df)} rows, columns: {list(df.columns)}")
            
            # Create datetime column if DATE and TIME exist
            if 'DATE' in df.columns and 'TIME' in df.columns:
                try:
                    df['datetime'] = pd.to_datetime(
                        df['DATE'].astype(str) + ' ' + df['TIME'].astype(str),
                        errors='coerce'
                    )
                    
                    # Remove rows with invalid datetime
                    initial_rows = len(df)
                    df = df.dropna(subset=['datetime'])
                    logging.info(f"Removed {initial_rows - len(df)} rows with invalid datetime")
                    
                    # Sort by datetime and remove duplicates
                    df = df.sort_values('datetime').drop_duplicates(subset=['datetime'])
                    
                    logging.info(f"Final data: {len(df)} records from {df['datetime'].min()} to {df['datetime'].max()}")
                    
                except Exception as e:
                    logging.error(f"Error creating datetime column: {str(e)}")
                    return None
            
            return df
            
        except Exception as e:
            logging.error(f"Error reading sheet {sheet_name}: {str(e)}")
            return None

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
            
            logging.info(f"Created timebase: {len(timebase)} timestamps")
            
            # Create base DataFrame
            combined_df = pd.DataFrame({'datetime': timebase})
            
            # Prepare gas data (remove DATE/TIME, keep datetime)
            gas_clean = gas_df.drop(['DATE', 'TIME'], axis=1, errors='ignore')
            
            # Prepare particle data (remove DATE/TIME, keep datetime)
            particle_clean = particle_df.drop(['DATE', 'TIME'], axis=1, errors='ignore')
            
            # Merge gas data
            combined_df = combined_df.merge(gas_clean, on='datetime', how='left', suffixes=('', '_gas'))
            
            # Merge particle data
            combined_df = combined_df.merge(particle_clean, on='datetime', how='left', suffixes=('', '_particle'))
            
            # Forward fill particle data to match 5-minute intervals
            particle_cols = [col for col in combined_df.columns 
                           if col.startswith('PM') or col in ['TEMP', 'AMB_PRES']]
            
            for col in particle_cols:
                if col in combined_df.columns:
                    combined_df[col] = combined_df[col].fillna(method='ffill')
            
            # Add data availability flags
            combined_df['gas_data_available'] = combined_df['H2S'].notna() if 'H2S' in combined_df.columns else False
            combined_df['particle_data_available'] = (
                combined_df['PM2.5'].notna() if 'PM2.5' in combined_df.columns else False
            )
            
            logging.info(f"Combined dataset: {len(combined_df)} records")
            return combined_df
            
        except Exception as e:
            logging.error(f"Error aligning data: {str(e)}")
            return None

    def save_to_parquet(self, df, station_name):
        """Save DataFrame to parquet with metadata."""
        try:
            output_path = self.output_dir / f"{station_name}_combined_data.parquet"
            
            # Save as parquet
            df.to_parquet(output_path, index=False)
            
            logging.info(f"Saved parquet file: {output_path}")
            
            # Create metadata file
            metadata_path = self.output_dir / f"{station_name}_metadata.txt"
            with open(metadata_path, 'w') as f:
                f.write(f"Metadata for {station_name}\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"File: {station_name}_combined_data.parquet\n")
                f.write(f"Records: {len(df)}\n")
                f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n")
                f.write(f"Time interval: 5 minutes\n\n")
                
                f.write("Columns and suspected units:\n")
                for col in df.columns:
                    f.write(f"  {col}")
                    # Add suspected units based on column names
                    if 'H2S' in col or 'SO2' in col:
                        f.write(" (ug/m3)")
                    elif 'CH4' in col:
                        f.write(" (mg/m3)")
                    elif 'PM' in col or 'TSP' in col:
                        f.write(" (ug/m3)")
                    elif 'WD' in col:
                        f.write(" (degrees)")
                    elif 'WS' in col:
                        f.write(" (m/s)")
                    elif 'TEMP' in col:
                        f.write(" (°C)")
                    elif 'AMB_PRES' in col:
                        f.write(" (hPa)")
                    f.write("\n")
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving parquet for {station_name}: {str(e)}")
            return False

    def create_summary(self, df, station_name):
        """Create a data summary file."""
        try:
            summary_path = self.output_dir / f"{station_name}_summary.txt"
            
            with open(summary_path, 'w') as f:
                f.write(f"Data Summary for {station_name}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total records: {len(df)}\n")
                f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n\n")
                
                f.write("Data Availability:\n")
                for col in df.columns:
                    if col not in ['datetime', 'gas_data_available', 'particle_data_available']:
                        available = df[col].notna().sum()
                        percentage = (available / len(df)) * 100
                        f.write(f"  {col}: {available} / {len(df)} ({percentage:.1f}%)\n")
                
                f.write("\nBasic Statistics:\n")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['gas_data_available', 'particle_data_available']:
                        stats = df[col].describe()
                        f.write(f"\n{col}:\n")
                        f.write(f"  Count: {int(stats['count'])}\n")
                        if stats['count'] > 0:
                            f.write(f"  Mean: {stats['mean']:.3f}\n")
                            f.write(f"  Min: {stats['min']:.3f}\n")
                            f.write(f"  Max: {stats['max']:.3f}\n")
            
            logging.info(f"Created summary: {summary_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating summary for {station_name}: {str(e)}")
            return False

    def process_mmf_file(self, filepath, station_name):
        """Process a single MMF Excel file."""
        try:
            logging.info(f"\nProcessing {station_name}: {filepath.name}")
            
            # Get sheet names
            excel_file = pd.ExcelFile(filepath)
            sheets = excel_file.sheet_names
            logging.info(f"Available sheets: {sheets}")
            
            if len(sheets) < 3:
                logging.error(f"Expected 3 sheets, found {len(sheets)}")
                return False
            
            # Read gas data (5-minute, typically sheet 1)
            gas_sheet = sheets[1]
            gas_data = self.read_sheet_data(filepath, gas_sheet)
            if gas_data is None:
                logging.error(f"Failed to read gas data from {gas_sheet}")
                return False
            
            # Read particle data (15-minute, typically sheet 2)
            particle_sheet = sheets[2]
            particle_data = self.read_sheet_data(filepath, particle_sheet)
            if particle_data is None:
                logging.error(f"Failed to read particle data from {particle_sheet}")
                return False
            
            # Combine and align data
            combined_data = self.align_and_combine_data(gas_data, particle_data)
            if combined_data is None:
                return False
            
            # Save to parquet
            if not self.save_to_parquet(combined_data, station_name):
                return False
            
            # Create summary
            if not self.create_summary(combined_data, station_name):
                return False
            
            logging.info(f"Successfully processed {station_name}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing {station_name}: {str(e)}")
            return False

def main():
    processor = MMFProcessor()
    
    # Define MMF files
    mmf_files = {
        'MMF1': Path('mmf_data/MMF1/processed/Silverdale Ambient Air Monitoring Data - MMF1 - Mar 21 to Aug 23.xlsx'),
        'MMF2': Path('mmf_data/MMF2/processed/Silverdale Ambient Air Monitoring Data - MMF2 - Mar 21 to Aug 23.xlsx'),
        'MMF6': Path('mmf_data/MMF6/processed/Silverdale Ambient Air Monitoring Data - MMF6 - Mar 21 to June 23.xlsx'),
        'MMF9': Path('mmf_data/MMF9/processed/Silverdale Ambient Air Monitoring Data - MMF9 - Mar 21 to Aug 23.xlsx')
    }
    
    successful = []
    failed = []
    
    for station, filepath in mmf_files.items():
        if filepath.exists():
            if processor.process_mmf_file(filepath, station):
                successful.append(station)
            else:
                failed.append(station)
        else:
            logging.error(f"File not found: {filepath}")
            failed.append(station)
    
    # Create overall report
    report_path = processor.output_dir / "processing_report.md"
    with open(report_path, 'w') as f:
        f.write("# MMF Data Processing Report\n\n")
        f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n")
        f.write(f"- Successfully processed: {len(successful)} stations\n")
        f.write(f"- Failed: {len(failed)} stations\n\n")
        
        if successful:
            f.write("### Successful:\n")
            for station in successful:
                f.write(f"- {station}\n")
            f.write("\n")
        
        if failed:
            f.write("### Failed:\n")
            for station in failed:
                f.write(f"- {station}\n")
            f.write("\n")
        
        f.write("## Data Processing Details\n")
        f.write("- Gas data: 5-minute intervals (WD, WS, H2S, CH4, SO2)\n")
        f.write("- Particle data: 15-minute intervals (PM1, PM2.5, PM4, PM10, TSP, TEMP, AMB_PRES)\n")
        f.write("- Combined on 5-minute timebase\n")
        f.write("- Particle data forward-filled to match gas data frequency\n")
        f.write("- Data availability flags added\n")
        f.write("- Saved as parquet files with metadata\n")
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")

if __name__ == "__main__":
    main()
