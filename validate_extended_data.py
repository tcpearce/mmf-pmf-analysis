import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
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
        logging.FileHandler('data_validation.log'),
        logging.StreamHandler()
    ]
)

class DataValidator:
    def __init__(self, parquet_dir="mmf_parquet", backup_dir="mmf_parquet/backup"):
        self.parquet_dir = Path(parquet_dir)
        self.backup_dir = Path(backup_dir)
        self.stations = ['MMF1', 'MMF2', 'MMF6', 'MMF9']
        self.validation_results = {}

    def load_data(self, station_name, use_backup=False):
        """Load parquet data and metadata."""
        try:
            if use_backup:
                # Find the backup file
                backup_files = list(self.backup_dir.glob(f"{station_name}_combined_data.parquet.backup_*"))
                if not backup_files:
                    logging.error(f"No backup file found for {station_name}")
                    return None, None
                parquet_path = backup_files[0]  # Use the most recent backup
            else:
                parquet_path = self.parquet_dir / f"{station_name}_combined_data.parquet"
            
            if not parquet_path.exists():
                logging.error(f"File not found: {parquet_path}")
                return None, None
            
            # Load data
            df = pd.read_parquet(parquet_path)
            
            # Load metadata
            parquet_file = pq.ParquetFile(parquet_path)
            metadata = {}
            if parquet_file.metadata.metadata:
                for key, value in parquet_file.metadata.metadata.items():
                    key_str = key.decode() if isinstance(key, bytes) else str(key)
                    value_str = value.decode() if isinstance(value, bytes) else str(value)
                    metadata[key_str] = value_str
            
            logging.info(f"Loaded {'backup' if use_backup else 'new'} {station_name}: {len(df)} records, "
                        f"{df['datetime'].min()} to {df['datetime'].max()}")
            
            return df, metadata
            
        except Exception as e:
            logging.error(f"Error loading {station_name} {'backup' if use_backup else 'new'}: {e}")
            return None, None

    def find_overlap_period(self, old_df, new_df):
        """Find the overlapping time period between old and new data."""
        old_start = old_df['datetime'].min()
        old_end = old_df['datetime'].max()
        new_start = new_df['datetime'].min()
        new_end = new_df['datetime'].max()
        
        overlap_start = max(old_start, new_start)
        overlap_end = min(old_end, new_end)
        
        if overlap_start <= overlap_end:
            return overlap_start, overlap_end
        else:
            return None, None

    def compare_overlapping_data(self, station_name, old_df, new_df):
        """Compare data in overlapping time periods."""
        results = {
            'station': station_name,
            'overlap_found': False,
            'overlap_period': None,
            'common_columns': [],
            'missing_columns_old': [],
            'missing_columns_new': [],
            'data_matches': {},
            'missing_data_handling': {},
            'statistical_comparison': {}
        }
        
        # Find overlap period
        overlap_start, overlap_end = self.find_overlap_period(old_df, new_df)
        
        if overlap_start is None:
            logging.warning(f"{station_name}: No overlapping time period found")
            return results
        
        results['overlap_found'] = True
        results['overlap_period'] = (overlap_start, overlap_end)
        
        logging.info(f"{station_name}: Overlap period: {overlap_start} to {overlap_end}")
        
        # Filter to overlap period
        old_overlap = old_df[(old_df['datetime'] >= overlap_start) & 
                           (old_df['datetime'] <= overlap_end)].copy()
        new_overlap = new_df[(new_df['datetime'] >= overlap_start) & 
                           (new_df['datetime'] <= overlap_end)].copy()
        
        logging.info(f"{station_name}: Overlap records - Old: {len(old_overlap)}, New: {len(new_overlap)}")
        
        # Sort both by datetime for comparison
        old_overlap = old_overlap.sort_values('datetime').reset_index(drop=True)
        new_overlap = new_overlap.sort_values('datetime').reset_index(drop=True)
        
        # Compare column sets
        old_cols = set(old_df.columns)
        new_cols = set(new_df.columns)
        common_cols = old_cols & new_cols
        missing_in_new = old_cols - new_cols
        missing_in_old = new_cols - old_cols
        
        results['common_columns'] = list(common_cols)
        results['missing_columns_old'] = list(missing_in_old)
        results['missing_columns_new'] = list(missing_in_new)
        
        logging.info(f"{station_name}: Common columns: {len(common_cols)}")
        if missing_in_new:
            logging.warning(f"{station_name}: Columns missing in new data: {missing_in_new}")
        if missing_in_old:
            logging.info(f"{station_name}: New columns added: {missing_in_old}")
        
        # Compare data for common columns
        for col in common_cols:
            if col == 'datetime':
                continue
                
            try:
                # Align data on datetime
                old_col = old_overlap.set_index('datetime')[col]
                new_col = new_overlap.set_index('datetime')[col]
                
                # Find common timestamps
                common_times = old_col.index.intersection(new_col.index)
                
                if len(common_times) == 0:
                    results['data_matches'][col] = 'no_common_timestamps'
                    continue
                
                old_common = old_col.loc[common_times]
                new_common = new_col.loc[common_times]
                
                # Handle different data types
                if pd.api.types.is_numeric_dtype(old_common) and pd.api.types.is_numeric_dtype(new_common):
                    # Numeric comparison
                    # Check for exact matches (including NaN)
                    exact_matches = 0
                    total_records = len(old_common)
                    
                    for i in range(len(old_common)):
                        old_val = old_common.iloc[i]
                        new_val = new_common.iloc[i]
                        
                        if pd.isna(old_val) and pd.isna(new_val):
                            exact_matches += 1
                        elif pd.isna(old_val) or pd.isna(new_val):
                            continue  # One is NaN, other isn't
                        elif abs(old_val - new_val) < 1e-10:  # Very small tolerance for floating point
                            exact_matches += 1
                    
                    match_rate = exact_matches / total_records if total_records > 0 else 0
                    results['data_matches'][col] = {
                        'type': 'numeric',
                        'total_records': total_records,
                        'exact_matches': exact_matches,
                        'match_rate': match_rate
                    }
                    
                    # Missing data comparison
                    old_missing = old_common.isna().sum()
                    new_missing = new_common.isna().sum()
                    results['missing_data_handling'][col] = {
                        'old_missing': int(old_missing),
                        'new_missing': int(new_missing),
                        'missing_match': old_missing == new_missing
                    }
                    
                    # Statistical comparison for non-missing values
                    old_valid = old_common.dropna()
                    new_valid = new_common.dropna()
                    
                    if len(old_valid) > 0 and len(new_valid) > 0:
                        results['statistical_comparison'][col] = {
                            'old_stats': {
                                'count': len(old_valid),
                                'mean': float(old_valid.mean()),
                                'std': float(old_valid.std()),
                                'min': float(old_valid.min()),
                                'max': float(old_valid.max())
                            },
                            'new_stats': {
                                'count': len(new_valid),
                                'mean': float(new_valid.mean()),
                                'std': float(new_valid.std()),
                                'min': float(new_valid.min()),
                                'max': float(new_valid.max())
                            }
                        }
                else:
                    # Non-numeric comparison (including boolean)
                    if pd.api.types.is_bool_dtype(old_common) and pd.api.types.is_bool_dtype(new_common):
                        # Boolean comparison
                        exact_matches = (old_common == new_common).sum()
                        nan_matches = (old_common.isna() & new_common.isna()).sum()
                        total_matches = exact_matches + nan_matches
                    else:
                        # Other non-numeric comparison
                        exact_matches = (old_common == new_common).sum()
                        nan_matches = (old_common.isna() & new_common.isna()).sum()
                        total_matches = exact_matches + nan_matches
                    
                    match_rate = total_matches / len(old_common) if len(old_common) > 0 else 0
                    
                    results['data_matches'][col] = {
                        'type': 'non_numeric',
                        'total_records': len(old_common),
                        'exact_matches': int(total_matches),
                        'match_rate': float(match_rate)
                    }
                    
            except Exception as e:
                logging.error(f"Error comparing column {col} for {station_name}: {e}")
                results['data_matches'][col] = f'error: {str(e)}'
        
        return results

    def compare_metadata(self, station_name, old_metadata, new_metadata):
        """Compare metadata between old and new files."""
        results = {
            'station': station_name,
            'units_match': {},
            'common_metadata': {},
            'metadata_differences': {}
        }
        
        # Extract and compare units
        old_units = {k.replace('_unit', ''): v for k, v in old_metadata.items() if k.endswith('_unit')}
        new_units = {k.replace('_unit', ''): v for k, v in new_metadata.items() if k.endswith('_unit')}
        
        # Compare units for common columns
        common_unit_cols = set(old_units.keys()) & set(new_units.keys())
        for col in common_unit_cols:
            matches = old_units[col] == new_units[col]
            results['units_match'][col] = {
                'match': matches,
                'old_unit': old_units[col],
                'new_unit': new_units[col]
            }
            
            if not matches:
                logging.warning(f"{station_name}: Unit mismatch for {col} - Old: '{old_units[col]}', New: '{new_units[col]}'")
        
        # Compare other metadata
        common_metadata_keys = set(old_metadata.keys()) & set(new_metadata.keys())
        for key in common_metadata_keys:
            if key.endswith('_unit'):
                continue  # Already handled above
            
            old_val = old_metadata[key]
            new_val = new_metadata[key]
            matches = old_val == new_val
            
            results['common_metadata'][key] = {
                'match': matches,
                'old_value': old_val,
                'new_value': new_val
            }
            
            if not matches and key not in ['processing_date', 'processing_type']:
                logging.warning(f"{station_name}: Metadata mismatch for {key} - Old: '{old_val}', New: '{new_val}'")
        
        # Find metadata differences
        old_only = set(old_metadata.keys()) - set(new_metadata.keys())
        new_only = set(new_metadata.keys()) - set(old_metadata.keys())
        
        results['metadata_differences'] = {
            'old_only': list(old_only),
            'new_only': list(new_only)
        }
        
        return results

    def validate_station(self, station_name):
        """Validate a single station's data."""
        logging.info(f"\n{'='*60}")
        logging.info(f"VALIDATING {station_name}")
        logging.info(f"{'='*60}")
        
        # Load old and new data
        old_df, old_metadata = self.load_data(station_name, use_backup=True)
        new_df, new_metadata = self.load_data(station_name, use_backup=False)
        
        if old_df is None or new_df is None:
            logging.error(f"Failed to load data for {station_name}")
            return None
        
        # Compare overlapping data
        data_comparison = self.compare_overlapping_data(station_name, old_df, new_df)
        
        # Compare metadata
        metadata_comparison = self.compare_metadata(station_name, old_metadata, new_metadata)
        
        # Combine results
        validation_result = {
            'station': station_name,
            'old_data_info': {
                'records': len(old_df),
                'date_range': (old_df['datetime'].min(), old_df['datetime'].max()),
                'columns': list(old_df.columns)
            },
            'new_data_info': {
                'records': len(new_df),
                'date_range': (new_df['datetime'].min(), new_df['datetime'].max()),
                'columns': list(new_df.columns)
            },
            'data_comparison': data_comparison,
            'metadata_comparison': metadata_comparison
        }
        
        return validation_result

    def generate_validation_report(self, validation_results):
        """Generate a comprehensive validation report."""
        report_path = self.parquet_dir / "validation_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# MMF Extended Data Validation Report\n\n")
            f.write(f"Validation completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall summary
            f.write("## Summary\n\n")
            total_stations = len(validation_results)
            successful_validations = sum(1 for r in validation_results.values() if r is not None)
            
            f.write(f"- Total stations validated: {total_stations}\n")
            f.write(f"- Successful validations: {successful_validations}\n")
            f.write(f"- Failed validations: {total_stations - successful_validations}\n\n")
            
            # Station-by-station results
            for station_name, result in validation_results.items():
                if result is None:
                    f.write(f"## {station_name}: VALIDATION FAILED\n\n")
                    continue
                
                f.write(f"## {station_name}\n\n")
                
                # Basic info comparison
                old_info = result['old_data_info']
                new_info = result['new_data_info']
                
                f.write("### Data Overview\n")
                f.write("| Aspect | Original | Extended | Change |\n")
                f.write("|--------|----------|----------|--------|\n")
                f.write(f"| Records | {old_info['records']:,} | {new_info['records']:,} | +{new_info['records'] - old_info['records']:,} |\n")
                f.write(f"| Start Date | {old_info['date_range'][0]} | {new_info['date_range'][0]} | - |\n")
                f.write(f"| End Date | {old_info['date_range'][1]} | {new_info['date_range'][1]} | - |\n")
                f.write(f"| Columns | {len(old_info['columns'])} | {len(new_info['columns'])} | +{len(new_info['columns']) - len(old_info['columns'])} |\n\n")
                
                # Overlap analysis
                data_comp = result['data_comparison']
                if data_comp['overlap_found']:
                    f.write("### Overlap Period Validation\n")
                    overlap_start, overlap_end = data_comp['overlap_period']
                    f.write(f"**Overlap Period:** {overlap_start} to {overlap_end}\n\n")
                    
                    f.write("#### Data Matching Results\n")
                    f.write("| Column | Type | Records | Matches | Match Rate | Status |\n")
                    f.write("|--------|------|---------|---------|------------|--------|\n")
                    
                    for col, match_info in data_comp['data_matches'].items():
                        if isinstance(match_info, dict):
                            if 'match_rate' in match_info:
                                status = "PASS" if match_info['match_rate'] > 0.99 else "CHECK" if match_info['match_rate'] > 0.95 else "FAIL"
                                f.write(f"| {col} | {match_info.get('type', 'unknown')} | {match_info.get('total_records', 0)} | {match_info.get('exact_matches', 0)} | {match_info['match_rate']:.4f} | {status} |\n")
                            else:
                                f.write(f"| {col} | - | - | - | - | Unknown |\n")
                        else:
                            f.write(f"| {col} | - | - | - | - | {match_info} |\n")
                    
                    f.write("\n#### Missing Data Handling\n")
                    f.write("| Column | Old Missing | New Missing | Match |\n")
                    f.write("|--------|-------------|-------------|-------|\n")
                    
                    for col, missing_info in data_comp['missing_data_handling'].items():
                        status = "MATCH" if missing_info['missing_match'] else "DIFFER"
                        f.write(f"| {col} | {missing_info['old_missing']} | {missing_info['new_missing']} | {status} |\n")
                    
                else:
                    f.write("### No Overlap Period Found\n")
                    f.write("The original and extended datasets do not have overlapping time periods.\n\n")
                
                # Metadata comparison
                meta_comp = result['metadata_comparison']
                f.write("\n### Metadata Validation\n")
                
                f.write("#### Units Comparison\n")
                f.write("| Column | Original Unit | Extended Unit | Match |\n")
                f.write("|--------|---------------|---------------|-------|\n")
                
                for col, unit_info in meta_comp['units_match'].items():
                    status = "MATCH" if unit_info['match'] else "DIFFER"
                    f.write(f"| {col} | {unit_info['old_unit']} | {unit_info['new_unit']} | {status} |\n")
                
                f.write("\n")
            
            f.write("## Validation Criteria\n")
            f.write("- **Data Matching:** PASS (>99%), CHECK (95-99%), FAIL (<95%)\n")
            f.write("- **Units:** Must match exactly between original and extended datasets\n")
            f.write("- **Missing Data:** Missing value counts should match in overlap periods\n")
            f.write("- **Metadata:** Processing metadata should be preserved\n\n")
            
        logging.info(f"Validation report saved to: {report_path}")

    def run_validation(self):
        """Run complete validation for all stations."""
        logging.info(f"\n{'='*80}")
        logging.info("STARTING MMF EXTENDED DATA VALIDATION")
        logging.info(f"{'='*80}")
        
        validation_results = {}
        
        for station in self.stations:
            try:
                result = self.validate_station(station)
                validation_results[station] = result
                
                if result is not None:
                    data_comp = result['data_comparison']
                    if data_comp['overlap_found']:
                        # Summary statistics
                        total_cols_checked = len(data_comp['data_matches'])
                        high_match_cols = sum(1 for match_info in data_comp['data_matches'].values() 
                                            if isinstance(match_info, dict) and 
                                            match_info.get('match_rate', 0) > 0.99)
                        
                        logging.info(f"{station} SUMMARY: {high_match_cols}/{total_cols_checked} columns have >99% match rate")
                    else:
                        logging.info(f"{station} SUMMARY: No overlap period - this is expected for some stations")
                
            except Exception as e:
                logging.error(f"Validation failed for {station}: {e}")
                validation_results[station] = None
        
        # Generate report
        self.generate_validation_report(validation_results)
        
        logging.info(f"\n{'='*80}")
        logging.info("VALIDATION COMPLETE!")
        logging.info(f"{'='*80}")
        
        return validation_results

def main():
    validator = DataValidator()
    results = validator.run_validation()
    
    # Print summary to console
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for station_name, result in results.items():
        if result is None:
            print(f"{station_name}: VALIDATION FAILED")
            continue
            
        data_comp = result['data_comparison']
        if data_comp['overlap_found']:
            total_cols = len(data_comp['data_matches'])
            high_match = sum(1 for m in data_comp['data_matches'].values() 
                           if isinstance(m, dict) and m.get('match_rate', 0) > 0.99)
            print(f"{station_name}: {high_match}/{total_cols} columns >99% match")
        else:
            print(f"{station_name}: No overlap (expected for some stations)")

if __name__ == "__main__":
    main()