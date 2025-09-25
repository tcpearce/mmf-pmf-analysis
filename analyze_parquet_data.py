import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import argparse
from mmf_config import MMF_PARQUET_DIR, get_mmf_parquet_file, get_corrected_mmf_files

class ParquetAnalyzer:
    def __init__(self, parquet_file):
        self.parquet_file = Path(parquet_file)
        self.df = None
        self.metadata = None
        
    def load_data(self):
        """Load parquet data and metadata."""
        try:
            # Read parquet file
            self.df = pd.read_parquet(self.parquet_file)
            
            # Read parquet metadata
            parquet_file = pq.ParquetFile(self.parquet_file)
            self.metadata = parquet_file.metadata
            self.schema = parquet_file.schema
            
            print(f"Successfully loaded: {self.parquet_file.name}")
            print(f"Data shape: {self.df.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading {self.parquet_file}: {str(e)}")
            return False
    
    def print_file_info(self):
        """Print comprehensive file information."""
        print("\n" + "="*80)
        print(f"PARQUET FILE ANALYSIS: {self.parquet_file.name}")
        print("="*80)
        
        # Basic file info
        print(f"\nFILE INFORMATION:")
        print(f"File size: {self.parquet_file.stat().st_size / (1024*1024):.2f} MB")
        print(f"Records: {len(self.df):,}")
        print(f"Columns: {len(self.df.columns)}")
        
        # Date range
        if 'datetime' in self.df.columns:
            print(f"Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
            print(f"Time span: {(self.df['datetime'].max() - self.df['datetime'].min()).days} days")
        
        # Parquet metadata
        print(f"\nPARQUET METADATA:")
        if self.metadata:
            print(f"Row groups: {self.metadata.num_row_groups}")
            print(f"Schema version: {self.metadata.format_version}")
        
        # Schema information
        print(f"\nCOLUMN SCHEMA:")
        for i in range(len(self.schema)):
            col_name = self.schema.column(i).name
            col_type = str(self.schema.column(i).physical_type)
            print(f"{i+1:2d}. {col_name:<20} | {col_type}")
    
    def extract_units_from_metadata(self):
        """Extract units from parquet metadata."""
        try:
            import pyarrow.parquet as pq
            
            pq_file = pq.ParquetFile(self.parquet_file)
            metadata = pq_file.metadata.metadata
            
            units = {}
            if metadata:
                for key, value in metadata.items():
                    key_str = key.decode() if isinstance(key, bytes) else str(key)
                    if key_str.endswith('_unit'):
                        col_name = key_str.replace('_unit', '')
                        unit_value = value.decode() if isinstance(value, bytes) else str(value)
                        units[col_name] = unit_value
            
            return units
        except Exception as e:
            print(f"Warning: Could not extract units from metadata: {e}")
            return {}

    def print_column_info(self):
        """Print detailed column information with stored and suspected units."""
        print(f"\nCOLUMN DETAILS:")
        print("-" * 80)
        
        # Extract stored units from metadata
        stored_units = self.extract_units_from_metadata()
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            non_null = self.df[col].notna().sum()
            null_count = self.df[col].isna().sum()
            null_pct = (null_count / len(self.df)) * 100
            
            # Get units - prioritize stored units over suspected
            if col in stored_units:
                units = stored_units[col] + " (from original Excel)"
                units_source = "✅ Stored"
            else:
                units = self.get_suspected_units(col) + " (inferred)"
                units_source = "🔍 Inferred"
            
            print(f"\n{col}:")
            print(f"  Data type: {dtype}")
            print(f"  Non-null values: {non_null:,} ({100-null_pct:.1f}%)")
            print(f"  Missing values: {null_count:,} ({null_pct:.1f}%)")
            print(f"  Units: {units}")
            print(f"  Units source: {units_source}")
            
            # Basic statistics for numeric columns
            if pd.api.types.is_numeric_dtype(self.df[col]):
                valid_data = self.df[col].dropna()
                if len(valid_data) > 0:
                    print(f"  Range: {valid_data.min():.3f} to {valid_data.max():.3f}")
                    print(f"  Mean: {valid_data.mean():.3f}")
                    print(f"  Median: {valid_data.median():.3f}")
    
    def get_suspected_units(self, column_name):
        """Get suspected units based on column name."""
        col_lower = column_name.lower()
        
        if 'datetime' in col_lower:
            return 'timestamp'
        elif 'h2s' in col_lower or 'so2' in col_lower:
            return 'μg/m³'
        elif 'ch4' in col_lower:
            return 'mg/m³'
        elif any(pm in col_lower for pm in ['pm1', 'pm2.5', 'pm4', 'pm10', 'tsp']):
            return 'μg/m³'
        elif any(voc in col_lower for voc in ['benzene', 'toluene', 'ethylbenzene', 'xylene']):
            return 'μg/m³'
        elif 'wd' in col_lower:
            return 'degrees'
        elif 'ws' in col_lower:
            return 'm/s'
        elif 'temp' in col_lower:
            return '°C'
        elif 'pres' in col_lower or 'pressure' in col_lower:
            return 'hPa'
        elif 'nox' in col_lower or 'no2' in col_lower or 'no' in col_lower:
            return 'μg/m³ (assumed)'
        elif 'available' in col_lower:
            return 'boolean flag'
        else:
            return 'unknown'
    
    def check_time_intervals(self):
        """Analyze time intervals to verify 5-minute and 15-minute alignment."""
        print(f"\nTIME INTERVAL ANALYSIS:")
        print("-" * 50)
        
        if 'datetime' not in self.df.columns:
            print("No datetime column found!")
            return
        
        # Calculate time differences
        time_diffs = self.df['datetime'].diff()
        
        # Count different intervals
        intervals = time_diffs.dt.total_seconds() / 60  # Convert to minutes
        interval_counts = intervals.value_counts().sort_index()
        
        print("Time interval distribution (minutes):")
        for interval, count in interval_counts.head(10).items():
            if pd.notna(interval):
                print(f"  {interval:6.1f} min: {count:,} occurrences ({count/len(intervals)*100:.1f}%)")
        
        # Check for expected 5-minute intervals
        five_min_intervals = (intervals == 5.0).sum()
        total_intervals = len(intervals) - 1  # Exclude first NaT
        
        print(f"\nExpected 5-minute intervals: {five_min_intervals:,} / {total_intervals:,} ({five_min_intervals/total_intervals*100:.1f}%)")
        
        # Check for gaps
        large_gaps = intervals[intervals > 15].dropna()
        if len(large_gaps) > 0:
            print(f"Time gaps > 15 minutes: {len(large_gaps)} occurrences")
            print(f"Largest gap: {large_gaps.max():.1f} minutes")
    
    def analyze_data_availability(self):
        """Analyze data availability patterns."""
        print(f"\nDATA AVAILABILITY ANALYSIS:")
        print("-" * 50)
        
        # Check gas vs particle data availability
        if 'gas_data_available' in self.df.columns and 'particle_data_available' in self.df.columns:
            gas_available = self.df['gas_data_available'].sum()
            particle_available = self.df['particle_data_available'].sum()
            total_records = len(self.df)
            
            print(f"Gas data available: {gas_available:,} / {total_records:,} ({gas_available/total_records*100:.1f}%)")
            print(f"Particle data available: {particle_available:,} / {total_records:,} ({particle_available/total_records*100:.1f}%)")
            
            # Check simultaneous availability
            both_available = (self.df['gas_data_available'] & self.df['particle_data_available']).sum()
            print(f"Both gas & particle available: {both_available:,} / {total_records:,} ({both_available/total_records*100:.1f}%)")
        
        # Identify key measurement columns
        gas_cols = [col for col in self.df.columns if any(gas in col.upper() for gas in ['H2S', 'CH4', 'SO2', 'NOX', 'NO2', 'NO', 'WD', 'WS'])]
        particle_cols = [col for col in self.df.columns if any(pm in col.upper() for pm in ['PM1', 'PM2.5', 'PM4', 'PM10', 'TSP', 'TEMP', 'PRES'])]
        
        print(f"\nGas measurement columns ({len(gas_cols)}):")
        for col in gas_cols:
            available = self.df[col].notna().sum()
            print(f"  {col:<20}: {available:,} / {len(self.df):,} ({available/len(self.df)*100:.1f}%)")
        
        print(f"\nParticle measurement columns ({len(particle_cols)}):")
        for col in particle_cols:
            available = self.df[col].notna().sum()
            print(f"  {col:<20}: {available:,} / {len(self.df):,} ({available/len(self.df)*100:.1f}%)")
    
    def check_forward_filling(self):
        """Check if forward filling was applied correctly for particle data."""
        print(f"\nFORWARD FILLING ANALYSIS:")
        print("-" * 50)
        
        if 'datetime' not in self.df.columns:
            return
        
        # Look for particle columns
        particle_cols = [col for col in self.df.columns if any(pm in col.upper() for pm in ['PM2.5', 'PM10', 'TEMP'])]
        
        for col in particle_cols[:3]:  # Check first 3 particle columns
            print(f"\nAnalyzing {col}:")
            
            # Find consecutive identical values (indicating forward filling)
            data = self.df[col].dropna()
            if len(data) < 2:
                continue
                
            consecutive_identical = 0
            current_streak = 1
            max_streak = 1
            
            for i in range(1, len(data)):
                if data.iloc[i] == data.iloc[i-1]:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    if current_streak > 1:
                        consecutive_identical += current_streak
                    current_streak = 1
            
            if current_streak > 1:
                consecutive_identical += current_streak
            
            print(f"  Values with consecutive identical neighbors: {consecutive_identical:,}")
            print(f"  Maximum consecutive identical values: {max_streak}")
            print(f"  This suggests forward-filling up to {max_streak * 5} minutes")
    
    def print_sample_data(self, n_samples=20):
        """Print sample data with units in column headers."""
        print(f"\nSAMPLE DATA ({n_samples} records):")
        print("=" * 120)
        
        # Get stored units from metadata
        stored_units = self.extract_units_from_metadata()
        
        # Display ALL columns
        sample_df = self.df.head(n_samples).copy()
        
        # Create new column names with units
        new_columns = []
        for col in sample_df.columns:
            if col in stored_units:
                unit = stored_units[col]
                new_col = f"{col} ({unit})"
            else:
                # Use inferred units
                unit = self.get_suspected_units(col)
                new_col = f"{col} ({unit})"
            new_columns.append(new_col)
        
        # Rename columns to include units
        sample_df.columns = new_columns
        
        # Format for display - show all columns
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 15)  # Slightly larger to accommodate units
        pd.set_option('display.expand_frame_repr', False)  # Don't wrap to multiple lines
        
        print(sample_df.to_string(index=True))
        
        # Reset display options
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
        pd.reset_option('display.max_colwidth')
        pd.reset_option('display.expand_frame_repr')
    
    def check_missing_data_patterns(self):
        """Analyze patterns in missing data."""
        print(f"\nMISSING DATA PATTERNS:")
        print("-" * 50)
        
        # Calculate missing data statistics
        missing_stats = []
        for col in self.df.columns:
            if col != 'datetime':
                missing_count = self.df[col].isna().sum()
                missing_pct = (missing_count / len(self.df)) * 100
                missing_stats.append({
                    'column': col,
                    'missing_count': missing_count,
                    'missing_pct': missing_pct
                })
        
        # Sort by missing percentage
        missing_stats.sort(key=lambda x: x['missing_pct'], reverse=True)
        
        print("Columns with most missing data:")
        for stat in missing_stats[:10]:
            print(f"  {stat['column']:<20}: {stat['missing_count']:,} ({stat['missing_pct']:.1f}%)")
        
        # Check for time periods with high missing data
        if 'datetime' in self.df.columns:
            self.df['date'] = self.df['datetime'].dt.date
            daily_missing = self.df.groupby('date').apply(lambda x: x.isna().sum().sum())
            
            if len(daily_missing) > 0:
                print(f"\nDays with highest missing data:")
                top_missing_days = daily_missing.nlargest(5)
                for date, missing_count in top_missing_days.items():
                    print(f"  {date}: {missing_count} missing values")

def main():
    parser = argparse.ArgumentParser(description='Analyze MMF parquet data files (corrected dataset)')
    parser.add_argument('station', choices=['MMF1', 'MMF2', 'MMF6', 'MMF9', 'Maries_Way'], 
                       help='MMF station to analyze')
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of sample records to display (default: 20)')
    
    args = parser.parse_args()
    
    # Use corrected file path
    try:
        parquet_file = get_mmf_parquet_file(args.station)
    except Exception as e:
        print(f"Error determining file path for {args.station}: {e}")
        return
    
    if not parquet_file.exists():
        print(f"Error: File {parquet_file} not found!")
        print("Available corrected files:")
        corrected_files = get_corrected_mmf_files()
        for mmf_id, file_path in corrected_files.items():
            exists = "✅" if file_path.exists() else "❌"
            print(f"  {exists} {mmf_id}: {file_path.name}")
        return
    
    # Create analyzer and run analysis
    analyzer = ParquetAnalyzer(parquet_file)
    
    if not analyzer.load_data():
        return
    
    # Run all analyses
    analyzer.print_file_info()
    analyzer.print_column_info()
    analyzer.check_time_intervals()
    analyzer.analyze_data_availability()
    analyzer.check_forward_filling()
    analyzer.check_missing_data_patterns()
    analyzer.print_sample_data(args.samples)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
