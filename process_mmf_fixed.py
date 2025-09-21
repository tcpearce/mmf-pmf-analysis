import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import warnings

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmf_processing_fixed.log'),
        logging.StreamHandler()
    ]
)

class MMFProcessor:
    def __init__(self, output_dir="mmf_parquet", target_timebase: str = "15min", aggregate_method: str = "mean", min_valid_subsamples: int = 2, include_voc: bool = True,
                 filter_start: str = None, filter_end: str = None, gas_sheet: str = None, particle_sheet: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        # Aggregation configuration
        # Supported target_timebase values: '15min', '30min'
        if target_timebase not in ("15min", "30min"):
            logging.warning(f"Unsupported target_timebase '{target_timebase}', defaulting to '15min'")
            target_timebase = "15min"
        self.target_timebase = target_timebase
        if aggregate_method not in ("mean", "median"):
            logging.warning(f"Unsupported aggregate_method '{aggregate_method}', defaulting to 'mean'")
            aggregate_method = "mean"
        self.aggregate_method = aggregate_method
        # Minimum number of valid sub-samples required to compute an aggregate window
        self.min_valid_subsamples = max(1, int(min_valid_subsamples))
        # Whether VOC (30-min native) will be included downstream
        self.include_voc = include_voc
        # Optional global date filter applied after alignment
        self.filter_start = pd.to_datetime(filter_start) if filter_start else None
        self.filter_end = pd.to_datetime(filter_end) if filter_end else None
        # Optional explicit sheet names
        self.gas_sheet = gas_sheet
        self.particle_sheet = particle_sheet

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
            
            # Read with identified header, skipping the units row that follows the header
            logging.info(f"Reading sheet {sheet_name} with header at row {header_row}")
            df = pd.read_excel(filepath, sheet_name=sheet_name, header=header_row, skiprows=lambda x: x == header_row + 1)
            
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

    def _aggregate_dataframe(self, df: pd.DataFrame, native_freq: str, target_freq: str, method: str, min_valid: int):
        """Aggregate a datetime-indexed dataframe from native_freq to target_freq with counts and min coverage.
        Returns (aggregated_values_df, counts_df)."""
        try:
            dfi = df.set_index('datetime')
            # Identify numeric columns (exclude any non-numeric)
            numeric_cols = [c for c in dfi.columns if pd.api.types.is_numeric_dtype(dfi[c])]
            if method == 'median':
                agg_vals = dfi[numeric_cols].resample(target_freq).median()
            else:
                agg_vals = dfi[numeric_cols].resample(target_freq).mean()
            counts = dfi[numeric_cols].resample(target_freq).count()
            # Enforce minimum coverage per window per column
            mask_insufficient = counts < int(min_valid)
            agg_vals = agg_vals.mask(mask_insufficient, other=pd.NA)
            agg_vals = agg_vals.reset_index()
            counts = counts.reset_index()
            return agg_vals, counts
        except Exception as e:
            logging.error(f"Error aggregating dataframe: {e}")
            return None, None

    def align_and_combine_data(self, gas_df, particle_df):
        """Align gas (5-min) and particle (15-min) data to target_timebase without forward-fill."""
        try:
            tgt = self.target_timebase  # '15min' or '30min'
            method = self.aggregate_method
            min_valid = self.min_valid_subsamples
            logging.info(f"Aggregating to target_timebase={tgt}, method={method}, min_valid_subsamples={min_valid}")
            if tgt not in ("15min", "30min"):
                raise ValueError(f"Unsupported target_timebase: {tgt}")

            # Clean input frames
            gas_clean = gas_df.drop(['DATE', 'TIME'], axis=1, errors='ignore')
            particle_clean = particle_df.drop(['DATE', 'TIME'], axis=1, errors='ignore')

            # Aggregate gas (native 5min)
            gas_vals, gas_counts = self._aggregate_dataframe(gas_clean, native_freq="5min", target_freq=tgt, method=method, min_valid=min_valid)
            if gas_vals is None:
                return None
            # Prefix counts columns with n_
            if gas_counts is not None:
                gas_counts = gas_counts.rename(columns={c: f"n_{c}" for c in gas_counts.columns if c != 'datetime'})

            # Aggregate particle (native 15min)
            if tgt == "15min":
                # No aggregation needed, but compute counts as 1 for non-NaN values
                dfi = particle_clean.set_index('datetime')
                numeric_cols = [c for c in dfi.columns if pd.api.types.is_numeric_dtype(dfi[c])]
                part_vals = dfi[numeric_cols].resample("15min").mean().reset_index()
                part_counts = dfi[numeric_cols].resample("15min").count().reset_index()
                # Enforce minimum coverage (min_valid of 1 means at least 1 valid 15-min sample)
                mask_insufficient = part_counts[numeric_cols] < int(max(1, min_valid))
                for col in numeric_cols:
                    part_vals.loc[mask_insufficient[col], col] = pd.NA
                part_counts = part_counts.rename(columns={c: f"n_{c}" for c in numeric_cols})
            else:
                # 30-min target: aggregate particle from 15min → 30min
                part_vals, part_counts = self._aggregate_dataframe(particle_clean, native_freq="15min", target_freq="30min", method=method, min_valid=min_valid)
                if part_counts is not None:
                    part_counts = part_counts.rename(columns={c: f"n_{c}" for c in part_counts.columns if c != 'datetime'})

            # Merge on datetime
            combined = gas_vals.merge(part_vals, on='datetime', how='outer', suffixes=('', ''))

            # Apply optional global date filter
            if self.filter_start is not None:
                combined = combined[combined['datetime'] >= self.filter_start]
            if self.filter_end is not None:
                combined = combined[combined['datetime'] <= self.filter_end]
            if gas_counts is not None:
                combined = combined.merge(gas_counts, on='datetime', how='left')
            if part_counts is not None:
                combined = combined.merge(part_counts, on='datetime', how='left')

            # Sort and reindex
            combined = combined.sort_values('datetime').reset_index(drop=True)

            # Add availability flags (handle column names with suffixes)
            h2s_col = next((c for c in combined.columns if 'H2S' in c), None)
            pm25_col = next((c for c in combined.columns if 'PM2.5' in c), None)
            combined['gas_data_available'] = combined[h2s_col].notna() if h2s_col else False
            combined['particle_data_available'] = combined[pm25_col].notna() if pm25_col else False

            logging.info(f"Combined dataset (aligned to {tgt}): {len(combined)} records")
            logging.info(f"Gas data points: {combined['gas_data_available'].sum()}")
            logging.info(f"Particle data points: {combined['particle_data_available'].sum()}")

            # VOC considerations: VOC is 30-min native. Record compatibility for downstream.
            self.voc_expected_timebase = "30min"
            self.voc_compatible = (tgt == "30min")
            if self.include_voc and not self.voc_compatible:
                logging.warning("VOC data is 30-min native; target_timebase != 30min. Downstream integration should either switch to 30min or exclude VOC.")

            return combined
        
        except Exception as e:
            logging.error(f"Error aligning data: {str(e)}")
            return None

    def save_to_parquet(self, df, station_name, all_units):
        """Save DataFrame to parquet with units and aggregation metadata."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
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
                elif col.startswith('n_'):
                    metadata[f"{col}_unit"] = 'count'
            
            # Add processing metadata
            metadata['processing_date'] = datetime.now().isoformat()
            metadata['station'] = station_name
            metadata['native_timebase_gas'] = '5min'
            metadata['native_timebase_particle'] = '15min'
            metadata['voc_expected_timebase'] = '30min'
            metadata['include_voc'] = self.include_voc
            metadata['voc_compatible'] = self.voc_compatible if hasattr(self, 'voc_compatible') else False
            metadata['aggregation_timebase'] = self.target_timebase
            metadata['aggregation_method'] = self.aggregate_method
            metadata['min_valid_subsamples'] = self.min_valid_subsamples
            metadata['source'] = 'MMF Excel files'
            if self.filter_start is not None:
                metadata['filter_start'] = str(self.filter_start)
            if self.filter_end is not None:
                metadata['filter_end'] = str(self.filter_end)
            
            # Convert metadata to bytes for PyArrow
            arrow_metadata = {k.encode(): str(v).encode() for k, v in metadata.items()}
            
            # Update table metadata
            existing_metadata = table.schema.metadata or {}
            existing_metadata.update(arrow_metadata)
            table = table.replace_schema_metadata(existing_metadata)
            
            # Save with metadata
            pq.write_table(table, output_path)
            
            logging.info(f"Saved parquet file with units/aggregation metadata: {output_path} ({len(df)} records)")
            logging.info(f"Units/metadata stored: {len(metadata)} entries")
            
            # Create metadata file
            metadata_path = self.output_dir / f"{station_name}_metadata.txt"
            with open(metadata_path, 'w') as f:
                f.write(f"Metadata for {station_name}\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"File: {station_name}_combined_data.parquet\n")
                f.write(f"Records: {len(df)}\n")
                f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n")
                f.write(f"Aggregation timebase: {self.target_timebase}\n")
                if self.filter_start is not None or self.filter_end is not None:
                    f.write(f"Filter window: {self.filter_start or 'None'} to {self.filter_end or 'None'}\n")
                f.write(f"Aggregation method: {self.aggregate_method}\n")
                f.write(f"Min valid subsamples: {self.min_valid_subsamples}\n")
                f.write(f"VOC compatible with timebase: {self.voc_compatible if hasattr(self, 'voc_compatible') else False}\n\n")
                
                f.write("Processing notes:\n")
                f.write("- Gas data: 5-minute native, aggregated to target timebase\n")
                f.write("- Particle data: 15-minute native, aggregated to target timebase as needed\n")
                f.write("- No forward-fill applied; aggregation enforces minimum coverage and preserves NaNs\n")
                f.write("- Count columns prefixed with 'n_' provide the number of valid sub-samples per window per species\n")
                f.write("- Missing values preserved as NaN\n")
                f.write("- Timezone information removed for consistency\n\n")
                
                f.write("Columns and units:\n")
                for col in df.columns:
                    f.write(f"  {col}")
                    # Add units based on column names
                    if col.startswith('n_'):
                        f.write(" (count)")
                    elif 'H2S' in col or 'SO2' in col:
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
                    elif 'datetime' in col:
                        f.write(" (timestamp)")
                    elif 'available' in col:
                        f.write(" (boolean flag)")
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
                f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n")
                f.write(f"Time span: {(df['datetime'].max() - df['datetime'].min()).days} days\n\n")
                
                f.write("Data Availability:\n")
                for col in df.columns:
                    if col not in ['datetime', 'gas_data_available', 'particle_data_available']:
                        available = df[col].notna().sum()
                        percentage = (available / len(df)) * 100
                        f.write(f"  {col}: {available:,} / {len(df):,} ({percentage:.1f}%)\n")
                
                f.write("\nBasic Statistics (non-zero values only):\n")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if col not in ['gas_data_available', 'particle_data_available']:
                        # Filter out zeros for better statistics
                        data = df[col][df[col] > 0]
                        if len(data) > 0:
                            f.write(f"\n{col}:\n")
                            f.write(f"  Non-zero count: {len(data):,}\n")
                            f.write(f"  Mean: {data.mean():.3f}\n")
                            f.write(f"  Median: {data.median():.3f}\n")
                            f.write(f"  Min: {data.min():.3f}\n")
                            f.write(f"  Max: {data.max():.3f}\n")
            
            logging.info(f"Created summary: {summary_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error creating summary for {station_name}: {str(e)}")
            return False

    def process_mmf_file(self, filepath, station_name):
        """Process a single MMF Excel file."""
        try:
            logging.info(f"\n{'='*60}")
            logging.info(f"Processing {station_name}: {filepath.name}")
            logging.info(f"{'='*60}")
            
            # Get sheet names
            excel_file = pd.ExcelFile(filepath)
            sheets = excel_file.sheet_names
            logging.info(f"Available sheets: {sheets}")
            
            # Select gas/particle sheets
            if self.gas_sheet and self.gas_sheet in sheets:
                gas_sheet = self.gas_sheet
            else:
                gas_sheet = sheets[1] if len(sheets) > 1 else sheets[0]
            if self.particle_sheet and self.particle_sheet in sheets:
                particle_sheet = self.particle_sheet
            else:
                particle_sheet = sheets[2] if len(sheets) > 2 else sheets[-1]
            
            # Read gas data (5-minute)
            logging.info(f"Processing gas data from sheet: {gas_sheet}")
            gas_data, gas_units = self.read_sheet_data(filepath, gas_sheet)
            if gas_data is None:
                logging.error(f"Failed to read gas data from {gas_sheet}")
                return False
            
            # Read particle data (15-minute)
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
            
            # Save to parquet with units
            if not self.save_to_parquet(combined_data, station_name, all_units):
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
    import argparse
    parser = argparse.ArgumentParser(description='Process MMF Excel files into aligned parquet with aggregation')
    parser.add_argument('--output-dir', default='mmf_parquet', help='Output directory for parquet files')
    parser.add_argument('--station', choices=['MMF1','MMF2','MMF6','MMF9','Maries_Way'], help='Station code for explicit processing with --raw-excel')
    parser.add_argument('--raw-excel', help='Absolute path to the raw Excel file to process (no fallback used when provided)')
    parser.add_argument('--sheet-gas', help='Explicit gas sheet name (optional)')
    parser.add_argument('--sheet-particle', help='Explicit particle sheet name (optional)')
    parser.add_argument('--start-date', help='Global filter start date (YYYY-MM-DD) applied after alignment')
    parser.add_argument('--end-date', help='Global filter end date (YYYY-MM-DD) applied after alignment')
    parser.add_argument('--timebase', choices=['15min','30min'], default='15min', help='Target aggregation timebase (default: 15min)')
    parser.add_argument('--aggregate', choices=['mean','median'], default='mean', help='Aggregation method for resampling (default: mean)')
    parser.add_argument('--min-valid-subsamples', type=int, default=2, help='Minimum sub-samples required in a window to compute aggregate (default: 2)')
    parser.add_argument('--include-voc', action='store_true', help='Indicate VOC (30min) will be included downstream (records compatibility)')
    args = parser.parse_args()

    processor = MMFProcessor(output_dir=args.output_dir, target_timebase=args.timebase, aggregate_method=args.aggregate, min_valid_subsamples=args.min_valid_subsamples, include_voc=args.include_voc,
                             filter_start=args.start_date, filter_end=args.end_date, gas_sheet=args.sheet_gas, particle_sheet=args.sheet_particle)
    
    # If explicit raw Excel path is provided, process only that station without any fallback
    raw_excel_path = getattr(args, 'raw_excel', None)
    if raw_excel_path:
        if not args.station:
            logging.error("--station must be provided when using --raw-excel")
            return
        filepath = Path(raw_excel_path)
        if not filepath.exists():
            logging.error(f"Raw Excel file not found: {filepath}")
            return
        logging.info(f"\nStarting processing of {args.station} (explicit raw Excel)...")
        success = processor.process_mmf_file(filepath, args.station)
        logging.info("\n" + "="*60)
        logging.info("PROCESSING COMPLETE!")
        logging.info(f"Successful: {[args.station] if success else []}")
        logging.info(f"Failed: {[] if success else [args.station]}")
        logging.info(f"Results saved to: {processor.output_dir}")
        logging.info("="*60)
        return

    # Define MMF files using actual raw/ directory paths (hash-prefixed filenames)
    mmf_files = {
        'MMF1': Path('mmf_data_corrected/MMF1_Cemetery_Road/raw/7969ed6f77e41d4fd840a70cd840d42f_Silverdale_Ambient_Air_Monitoring_Data_-_Cemetery_Road_-_Mar_2021_-_Aug_2024.xlsx'),
        'MMF2': Path('mmf_data_corrected/MMF2_Silverdale_Pumping_Station/raw/c39163361bc4854cac6f969b148b4c64_Silverdale Ambient Air Monitoring Data - MMF Silverdale Pumping Station - Mar 2021 to July 2025.xlsx'),
        'MMF6': None,  # No raw Excel file available for MMF6
        'MMF9': Path('mmf_data_corrected/MMF9_Galingale_View/raw/61379dace1c94403959b18fbd97184b7_Silverdale Ambient Air Monitoring Data -MMF Galingale View - Mar 2021 to Jul 2025.xlsx')
    }
    
    successful = []
    failed = []
    
    # Process one file at a time
    for station, filepath in mmf_files.items():
        if filepath is None:
            logging.warning(f"No raw Excel file available for {station}, skipping")
            failed.append(station)
            continue
            
        if not filepath.exists():
            # Fallback: search corrected data tree for any Excel source under station-specific folder
            base_dir = Path('mmf_data_corrected')
            candidates = []
            station_dir = base_dir / station
            if station_dir.exists():
                candidates = list(station_dir.rglob('*.xlsx'))
            if not candidates:
                # Broad search for files containing station code in path
                candidates = [p for p in base_dir.rglob('*.xlsx') if station in str(p)]
            if candidates:
                logging.warning(f"File not found: {filepath}. Using fallback: {candidates[0]}")
                filepath = candidates[0]
        if filepath.exists():
            logging.info(f"\nStarting processing of {station}...")
            if processor.process_mmf_file(filepath, station):
                successful.append(station)
            else:
                failed.append(station)
        else:
            logging.error(f"File not found for {station}: {filepath}")
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
        
        f.write("## Processing Details\n")
        f.write("- **Fixed Issues:**\n")
        f.write("  - Timezone mixing in datetime parsing\n")
        f.write("  - Header row detection improved\n")
        f.write("  - Memory optimization for large files\n")
        f.write("  - Better error handling\n\n")
        
        f.write("- **Data Structure:**\n")
        f.write("  - Gas data: 5-minute intervals (WD, WS, H2S, CH4, SO2)\n")
        f.write("  - Particle data: 15-minute intervals (PM1, PM2.5, PM4, PM10, TSP, TEMP, AMB_PRES)\n")
        f.write("  - Combined on 5-minute timebase\n")
        f.write("  - Particle data forward-filled (max 3 intervals)\n")
        f.write("  - Data availability flags added\n")
        f.write("  - Saved as parquet files with metadata\n")
    
    logging.info(f"\n{'='*60}")
    logging.info("PROCESSING COMPLETE!")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    logging.info(f"Results saved to: {processor.output_dir}")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()
