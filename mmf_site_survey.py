#!/usr/bin/env python3
"""
MMF Site Data Survey Script
===========================

This script surveys all available MMF (Mobile Monitoring Facility) sites and generates
a comprehensive summary table showing:
- MMF site numbers and names
- Date ranges of available data
- All columns/parameters available at each site
- Data completeness statistics
- Export to CSV and formatted text report

Usage:
    python mmf_site_survey.py [--output-dir results] [--detailed]
    
Options:
    --output-dir DIR    Output directory (default: mmf_survey_results)
    --detailed          Include detailed column statistics
    --export-csv        Export results to CSV format
    --export-html       Export results to HTML format
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import MMF configuration
try:
    from mmf_config import get_station_mapping, get_corrected_mmf_files, get_mmf_parquet_file
    from analyze_parquet_data import ParquetAnalyzer
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Make sure you're running from the correct directory with mmf_config.py and analyze_parquet_data.py")
    sys.exit(1)

class MMFSiteSurvey:
    def __init__(self, output_dir="mmf_survey_results", detailed=False):
        """
        Initialize MMF Site Survey.
        
        Args:
            output_dir (str): Directory for output files
            detailed (bool): Include detailed statistics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.detailed = detailed
        self.survey_results = []
        
        print("🔍 MMF Site Data Survey")
        print("=" * 50)
        print(f"📂 Output directory: {self.output_dir}")
        print(f"📊 Detailed analysis: {detailed}")
        print()
    
    def survey_all_sites(self):
        """Survey all available MMF sites and collect comprehensive information."""
        print("🌐 Surveying all MMF sites...")
        
        # Get station mapping
        station_mapping = get_station_mapping()
        
        # Find all available parquet files
        parquet_dir = Path("mmf_parquet_final")
        if not parquet_dir.exists():
            # Try alternative locations
            for alt_dir in ["mmf_parquet", ".", "../mmf_parquet_final"]:
                if Path(alt_dir).exists():
                    parquet_dir = Path(alt_dir)
                    break
        
        # Search for parquet files
        parquet_files = list(parquet_dir.glob("*.parquet")) if parquet_dir.exists() else []
        
        if not parquet_files:
            print("❌ No parquet files found. Checking known stations...")
            # Try known stations
            for station_code in station_mapping.keys():
                try:
                    parquet_file = get_mmf_parquet_file(station_code)
                    if parquet_file.exists():
                        parquet_files.append(parquet_file)
                except:
                    continue
        
        if not parquet_files:
            print("❌ No MMF data files found!")
            return
        
        print(f"📁 Found {len(parquet_files)} data files")
        
        # Survey each file
        for i, parquet_file in enumerate(parquet_files, 1):
            print(f"   📊 Surveying {parquet_file.name} ({i}/{len(parquet_files)})")
            site_info = self._survey_site(parquet_file, station_mapping)
            if site_info:
                self.survey_results.append(site_info)
        
        print(f"\n✅ Survey complete: {len(self.survey_results)} sites analyzed")
    
    def _survey_site(self, parquet_file, station_mapping):
        """Survey a single MMF site and extract comprehensive information."""
        try:
            # Extract site code from filename
            filename = parquet_file.stem
            site_code = self._extract_site_code(filename)
            
            # Load data
            analyzer = ParquetAnalyzer(parquet_file)
            if not analyzer.load_data():
                print(f"      ⚠️ Failed to load {parquet_file.name}")
                return None
            
            df = analyzer.df
            
            # Get site information
            site_name = station_mapping.get(site_code, "Unknown")
            display_name = f"{site_code} - {site_name}" if site_name and site_name != "Unknown" else site_code
            
            # Analyze datetime information
            datetime_info = self._analyze_datetime(df)
            
            # Analyze columns
            columns_info = self._analyze_columns(df)
            
            # Analyze species-specific date ranges
            species_date_ranges = self._analyze_species_date_ranges(df)
            
            # Calculate data completeness
            completeness_stats = self._calculate_completeness(df)
            
            # Get file information
            file_stats = parquet_file.stat()
            file_size_mb = file_stats.st_size / (1024 * 1024)
            
            site_info = {
                'site_code': site_code,
                'site_name': site_name,
                'display_name': display_name,
                'file_path': str(parquet_file),
                'file_name': parquet_file.name,
                'file_size_mb': round(file_size_mb, 2),
                'total_records': len(df),
                'date_range_start': datetime_info['start'],
                'date_range_end': datetime_info['end'],
                'date_range_days': datetime_info['days'],
                'total_columns': len(df.columns),
                'datetime_column': datetime_info['datetime_column'],
                'gas_columns': columns_info['gas_columns'],
                'voc_columns': columns_info['voc_columns'],  # Added VOC columns
                'particle_columns': columns_info['particle_columns'],
                'meteorological_columns': columns_info['meteorological_columns'],
                'quality_columns': columns_info['quality_columns'],
                'other_columns': columns_info['other_columns'],
                'all_columns': list(df.columns),
                'species_date_ranges': species_date_ranges,  # Added species date ranges
                'completeness_stats': completeness_stats
            }
            
            if self.detailed:
                # Add detailed column statistics
                site_info['column_details'] = self._get_detailed_column_stats(df)
            
            return site_info
            
        except Exception as e:
            print(f"      ❌ Error surveying {parquet_file.name}: {e}")
            return None
    
    def _extract_site_code(self, filename):
        """Extract site code from filename."""
        # Handle different filename patterns
        if 'MMF1' in filename or 'Cemetery_Road' in filename:
            return 'MMF1'
        elif 'MMF2' in filename or 'Silverdale' in filename:
            return 'MMF2'
        elif 'MMF6' in filename or 'Fire_Station' in filename:
            return 'MMF6'
        elif 'MMF9' in filename or 'Galingale' in filename:
            return 'MMF9'
        elif 'Maries_Way' in filename:
            return 'Maries_Way'
        else:
            # Try to extract MMFx pattern
            import re
            match = re.search(r'MMF(\d+)', filename)
            if match:
                return f'MMF{match.group(1)}'
            else:
                return filename.split('_')[0]  # Use first part of filename
    
    def _analyze_datetime(self, df):
        """Analyze datetime information in the dataset."""
        datetime_info = {
            'start': None,
            'end': None,
            'days': 0,
            'datetime_column': None
        }
        
        # Look for datetime columns
        datetime_columns = ['datetime', 'Datetime', 'DATETIME', 'timestamp', 'time']
        datetime_col = None
        
        for col in datetime_columns:
            if col in df.columns:
                datetime_col = col
                break
        
        if datetime_col:
            try:
                datetime_series = pd.to_datetime(df[datetime_col])
                datetime_info['datetime_column'] = datetime_col
                datetime_info['start'] = datetime_series.min()
                datetime_info['end'] = datetime_series.max()
                datetime_info['days'] = (datetime_info['end'] - datetime_info['start']).days + 1
            except:
                pass
        
        return datetime_info
    
    def _analyze_columns(self, df):
        """Categorize columns by type."""
        columns_info = {
            'gas_columns': [],
            'voc_columns': [],  # Added VOC category
            'particle_columns': [],
            'meteorological_columns': [],
            'quality_columns': [],
            'other_columns': []
        }
        
        # Define column categories
        gas_species = ['H2S', 'CH4', 'SO2', 'NOX', 'NO', 'NO2', 'NH3', 'CO', 'O3']
        voc_species = ['Benzene', 'Toluene', 'Ethylbenzene', 'Xylene', 'BTEX']  # Added BTEX VOCs
        particle_species = ['PM1', 'PM2.5', 'PM4', 'PM10', 'TSP', 'FIDAS']
        meteorological_terms = ['TEMP', 'temp', 'temperature', 'wind', 'Wind', 'WIND', 
                               'pressure', 'Pressure', 'PRESSURE', 'humidity', 'Humidity',
                               'HUMID', 'precip', 'rain', 'solar', 'radiation']
        quality_terms = ['data_available', 'quality', 'flag', 'status', 'valid', 'QA', 'QC']
        
        for col in df.columns:
            if col.lower() in ['datetime', 'timestamp', 'time']:
                continue  # Skip datetime columns
            elif any(gas in col for gas in gas_species):
                columns_info['gas_columns'].append(col)
            elif any(voc in col for voc in voc_species):
                columns_info['voc_columns'].append(col)  # Added VOC detection
            elif any(particle in col for particle in particle_species):
                columns_info['particle_columns'].append(col)
            elif any(met in col for met in meteorological_terms):
                columns_info['meteorological_columns'].append(col)
            elif any(qual in col for qual in quality_terms):
                columns_info['quality_columns'].append(col)
            else:
                columns_info['other_columns'].append(col)
        
        return columns_info
    
    def _analyze_species_date_ranges(self, df):
        """Analyze date ranges for different species to show when each type of data is available."""
        species_ranges = {}
        
        # Find datetime column
        datetime_col = None
        for col in ['datetime', 'Datetime', 'DATETIME', 'timestamp', 'time']:
            if col in df.columns:
                datetime_col = col
                break
        
        if not datetime_col:
            return species_ranges
        
        try:
            datetime_series = pd.to_datetime(df[datetime_col])
            
            # Define all species categories
            all_species_categories = {
                'Gas Species': ['H2S', 'CH4', 'SO2', 'NOX', 'NO', 'NO2', 'NH3', 'CO', 'O3'],
                'VOC Species': ['Benzene', 'Toluene', 'Ethylbenzene', 'Xylene', 'BTEX', 'm&p-Xylene'],
                'Particle Species': ['PM1', 'PM2.5', 'PM4', 'PM10', 'TSP', 'FIDAS'],
                'Meteorological': ['TEMP', 'temp', 'temperature', 'WIND', 'wind', 'Wind', 'WD', 'WS',
                                  'pressure', 'Pressure', 'PRESSURE']
            }
            
            # Analyze each column to find species-specific date ranges
            for col in df.columns:
                if col == datetime_col or col.lower() in ['mmf_id', 'station_name']:
                    continue
                
                # Skip boolean/flag columns
                if df[col].dtype == bool or 'available' in col.lower():
                    continue
                
                # Get non-null data for this column
                valid_data_mask = df[col].notna()
                
                if valid_data_mask.sum() == 0:
                    continue  # No data for this column
                
                valid_datetimes = datetime_series[valid_data_mask]
                
                if len(valid_datetimes) > 0:
                    date_range_start = valid_datetimes.min()
                    date_range_end = valid_datetimes.max()
                    total_records = len(valid_datetimes)
                    total_days = (date_range_end - date_range_start).days + 1
                    
                    # Determine species category
                    species_category = 'Other'
                    for category, species_list in all_species_categories.items():
                        if any(species in col for species in species_list):
                            species_category = category
                            break
                    
                    species_ranges[col] = {
                        'category': species_category,
                        'start_date': date_range_start,
                        'end_date': date_range_end,
                        'total_records': total_records,
                        'total_days': total_days,
                        'data_frequency': self._estimate_data_frequency(valid_datetimes, total_records)
                    }
            
        except Exception as e:
            print(f"      ⚠️ Error analyzing species date ranges: {e}")
        
        return species_ranges
    
    def _estimate_data_frequency(self, datetime_series, total_records):
        """Estimate the data frequency based on datetime intervals."""
        try:
            if len(datetime_series) < 2:
                return 'Unknown'
            
            # Calculate typical interval between measurements
            sorted_times = datetime_series.sort_values()
            intervals = sorted_times.diff().dropna()
            
            if len(intervals) == 0:
                return 'Unknown'
            
            # Get median interval to avoid outliers
            median_interval = intervals.median()
            
            # Convert to minutes
            interval_minutes = median_interval.total_seconds() / 60
            
            if interval_minutes <= 1:
                return 'Sub-minute'
            elif interval_minutes <= 5:
                return '5-minute'
            elif interval_minutes <= 15:
                return '15-minute'
            elif interval_minutes <= 30:
                return '30-minute'
            elif interval_minutes <= 60:
                return 'Hourly'
            elif interval_minutes <= 1440:  # 24 hours
                return 'Daily'
            else:
                return 'Less frequent'
                
        except Exception:
            return 'Unknown'
    
    def _calculate_completeness(self, df):
        """Calculate data completeness statistics."""
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        completeness_percent = ((total_cells - missing_cells) / total_cells) * 100
        
        # Per-column completeness
        column_completeness = {}
        for col in df.columns:
            if col.lower() not in ['datetime', 'timestamp', 'time']:
                valid_count = df[col].notna().sum()
                total_count = len(df)
                completeness = (valid_count / total_count) * 100
                column_completeness[col] = {
                    'valid_count': valid_count,
                    'total_count': total_count,
                    'completeness_percent': round(completeness, 1)
                }
        
        return {
            'overall_completeness_percent': round(completeness_percent, 1),
            'total_missing_cells': missing_cells,
            'column_completeness': column_completeness
        }
    
    def _get_detailed_column_stats(self, df):
        """Get detailed statistics for each column."""
        column_details = {}
        
        for col in df.columns:
            if col.lower() in ['datetime', 'timestamp', 'time']:
                continue
            
            col_data = df[col]
            details = {
                'dtype': str(col_data.dtype),
                'non_null_count': col_data.notna().sum(),
                'null_count': col_data.isna().sum(),
                'unique_count': col_data.nunique(),
                'completeness_percent': round((col_data.notna().sum() / len(df)) * 100, 1)
            }
            
            # Add numeric statistics if column is numeric
            if pd.api.types.is_numeric_dtype(col_data):
                numeric_data = col_data.dropna()
                if len(numeric_data) > 0:
                    details.update({
                        'min_value': float(numeric_data.min()),
                        'max_value': float(numeric_data.max()),
                        'mean_value': round(float(numeric_data.mean()), 3),
                        'std_value': round(float(numeric_data.std()), 3)
                    })
            
            column_details[col] = details
        
        return column_details
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        if not self.survey_results:
            print("❌ No survey results to report")
            return
        
        report_file = self.output_dir / "mmf_site_survey_report.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("MMF SITE DATA SURVEY REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Sites Surveyed: {len(self.survey_results)}\n\n")
            
            # Overall summary
            f.write("OVERALL SUMMARY\n")
            f.write("-" * 20 + "\n")
            total_records = sum(site['total_records'] for site in self.survey_results)
            total_size_mb = sum(site['file_size_mb'] for site in self.survey_results)
            f.write(f"Total Records Across All Sites: {total_records:,}\n")
            f.write(f"Total Data Size: {total_size_mb:.1f} MB\n")
            
            # Date range summary
            all_start_dates = [site['date_range_start'] for site in self.survey_results if site['date_range_start']]
            all_end_dates = [site['date_range_end'] for site in self.survey_results if site['date_range_end']]
            
            if all_start_dates and all_end_dates:
                overall_start = min(all_start_dates)
                overall_end = max(all_end_dates)
                f.write(f"Overall Date Range: {overall_start.strftime('%Y-%m-%d')} to {overall_end.strftime('%Y-%m-%d')}\n")
            f.write("\n")
            
            # Individual site details
            for site in sorted(self.survey_results, key=lambda x: x['site_code']):
                f.write(f"SITE: {site['display_name']}\n")
                f.write("-" * (len(site['display_name']) + 6) + "\n")
                f.write(f"File: {site['file_name']}\n")
                f.write(f"Size: {site['file_size_mb']} MB\n")
                f.write(f"Records: {site['total_records']:,}\n")
                
                if site['date_range_start'] and site['date_range_end'] and pd.notna(site['date_range_start']) and pd.notna(site['date_range_end']):
                    f.write(f"Date Range: {site['date_range_start'].strftime('%Y-%m-%d')} to {site['date_range_end'].strftime('%Y-%m-%d')} ({site['date_range_days']} days)\n")
                else:
                    f.write(f"Date Range: Not available\n")
                
                f.write(f"Total Columns: {site['total_columns']}\n")
                f.write(f"Overall Data Completeness: {site['completeness_stats']['overall_completeness_percent']}%\n")
                
                # Column categories
                f.write(f"\nData Categories:\n")
                f.write(f"  Gas Species ({len(site['gas_columns'])}): {', '.join(site['gas_columns'])}\n")
                f.write(f"  VOC Species ({len(site['voc_columns'])}): {', '.join(site['voc_columns'])}\n")  # Added VOC category
                f.write(f"  Particle Species ({len(site['particle_columns'])}): {', '.join(site['particle_columns'])}\n")
                f.write(f"  Meteorological ({len(site['meteorological_columns'])}): {', '.join(site['meteorological_columns'])}\n")
                f.write(f"  Quality/Status ({len(site['quality_columns'])}): {', '.join(site['quality_columns'])}\n")
                f.write(f"  Other ({len(site['other_columns'])}): {', '.join(site['other_columns'])}\n")
                
                # Species-specific date ranges
                if 'species_date_ranges' in site and site['species_date_ranges']:
                    f.write(f"\nSpecies-Specific Data Availability:\n")
                    f.write(f"{'-'*50}\n")
                    
                    # Group by category for better organization
                    ranges_by_category = {}
                    for species, range_info in site['species_date_ranges'].items():
                        category = range_info['category']
                        if category not in ranges_by_category:
                            ranges_by_category[category] = []
                        ranges_by_category[category].append((species, range_info))
                    
                    for category, species_list in ranges_by_category.items():
                        f.write(f"\n  {category}:\n")
                        for species, range_info in sorted(species_list):
                            start_str = range_info['start_date'].strftime('%Y-%m-%d') if pd.notna(range_info['start_date']) else 'Unknown'
                            end_str = range_info['end_date'].strftime('%Y-%m-%d') if pd.notna(range_info['end_date']) else 'Unknown'
                            f.write(f"    {species}: {start_str} to {end_str} ")
                            f.write(f"({range_info['total_records']:,} records, {range_info['data_frequency']} frequency)\n")
                
                f.write(f"\nAll Columns ({site['total_columns']}):\n")
                # Format columns in a readable way (4 columns per row)
                all_cols = site['all_columns']
                for i in range(0, len(all_cols), 4):
                    row_cols = all_cols[i:i+4]
                    f.write(f"  {', '.join(row_cols)}\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"📄 Summary report saved: {report_file}")
        return report_file
    
    def export_to_csv(self):
        """Export survey results to CSV format."""
        if not self.survey_results:
            print("❌ No survey results to export")
            return
        
        # Create main summary CSV
        summary_data = []
        for site in self.survey_results:
            row = {
                'Site_Code': site['site_code'],
                'Site_Name': site['site_name'],
                'Display_Name': site['display_name'],
                'File_Name': site['file_name'],
                'File_Size_MB': site['file_size_mb'],
                'Total_Records': site['total_records'],
                'Date_Range_Start': site['date_range_start'].strftime('%Y-%m-%d') if site['date_range_start'] and pd.notna(site['date_range_start']) else '',
                'Date_Range_End': site['date_range_end'].strftime('%Y-%m-%d') if site['date_range_end'] and pd.notna(site['date_range_end']) else '',
                'Date_Range_Days': site['date_range_days'],
                'Total_Columns': site['total_columns'],
                'Gas_Columns_Count': len(site['gas_columns']),
                'VOC_Columns_Count': len(site['voc_columns']),  # Added VOC count
                'Particle_Columns_Count': len(site['particle_columns']),
                'Meteorological_Columns_Count': len(site['meteorological_columns']),
                'Quality_Columns_Count': len(site['quality_columns']),
                'Other_Columns_Count': len(site['other_columns']),
                'Data_Completeness_Percent': site['completeness_stats']['overall_completeness_percent'],
                'Gas_Columns': '; '.join(site['gas_columns']),
                'VOC_Columns': '; '.join(site['voc_columns']),  # Added VOC columns
                'Particle_Columns': '; '.join(site['particle_columns']),
                'Meteorological_Columns': '; '.join(site['meteorological_columns']),
                'Quality_Columns': '; '.join(site['quality_columns']),
                'Other_Columns': '; '.join(site['other_columns']),
                'All_Columns': '; '.join(site['all_columns'])
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = self.output_dir / "mmf_site_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f"📊 Summary CSV saved: {summary_csv}")
        
        # Create detailed column completeness CSV if detailed analysis was run
        if self.detailed and any('column_details' in site for site in self.survey_results):
            completeness_data = []
            for site in self.survey_results:
                if 'column_details' not in site:
                    continue
                for col, details in site['column_details'].items():
                    row = {
                        'Site_Code': site['site_code'],
                        'Site_Name': site['site_name'],
                        'Column_Name': col,
                        'Data_Type': details['dtype'],
                        'Non_Null_Count': details['non_null_count'],
                        'Null_Count': details['null_count'],
                        'Completeness_Percent': details['completeness_percent'],
                        'Unique_Values': details['unique_count']
                    }
                    if 'min_value' in details:
                        row.update({
                            'Min_Value': details['min_value'],
                            'Max_Value': details['max_value'],
                            'Mean_Value': details['mean_value'],
                            'Std_Value': details['std_value']
                        })
                    completeness_data.append(row)
            
            if completeness_data:
                completeness_df = pd.DataFrame(completeness_data)
                completeness_csv = self.output_dir / "mmf_column_details.csv"
                completeness_df.to_csv(completeness_csv, index=False)
                print(f"📊 Detailed column CSV saved: {completeness_csv}")
    
    def export_to_html(self):
        """Export survey results to HTML format."""
        if not self.survey_results:
            print("❌ No survey results to export")
            return
        
        html_file = self.output_dir / "mmf_site_survey.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MMF Site Data Survey Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .site-card {{ border: 1px solid #ddd; margin: 20px 0; padding: 15px; border-radius: 5px; }}
                .site-title {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
                .summary-stats {{ background-color: #f8f9fa; padding: 10px; border-radius: 3px; margin: 10px 0; }}
                .column-category {{ margin: 10px 0; }}
                .column-list {{ font-family: monospace; font-size: 12px; background-color: #f1f1f1; padding: 5px; }}
                .stats {{ display: inline-block; margin-right: 20px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>MMF Site Data Survey Report</h1>
                <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Sites Surveyed:</strong> {len(self.survey_results)}</p>
            </div>
        """
        
        # Overall summary table
        total_records = sum(site['total_records'] for site in self.survey_results)
        total_size_mb = sum(site['file_size_mb'] for site in self.survey_results)
        
        html_content += f"""
            <div class="summary-stats">
                <h2>Overall Summary</h2>
                <div class="stats">Total Records: {total_records:,}</div>
                <div class="stats">Total Data Size: {total_size_mb:.1f} MB</div>
            </div>
        """
        
        # Site details
        for site in sorted(self.survey_results, key=lambda x: x['site_code']):
            html_content += f"""
            <div class="site-card">
                <h2 class="site-title">{site['display_name']}</h2>
                <div class="summary-stats">
                    <div class="stats">Records: {site['total_records']:,}</div>
                    <div class="stats">Size: {site['file_size_mb']} MB</div>
                    <div class="stats">Columns: {site['total_columns']}</div>
                    <div class="stats">Completeness: {site['completeness_stats']['overall_completeness_percent']}%</div>
                    """
            
            if site['date_range_start'] and site['date_range_end'] and pd.notna(site['date_range_start']) and pd.notna(site['date_range_end']):
                html_content += f"""
                    <div class="stats">Date Range: {site['date_range_start'].strftime('%Y-%m-%d')} to {site['date_range_end'].strftime('%Y-%m-%d')}</div>
                """
            
            html_content += "</div>"
            
            # Column categories
            categories = [
                ('Gas Species', site['gas_columns']),
                ('VOC Species', site['voc_columns']),  # Added VOC category
                ('Particle Species', site['particle_columns']),
                ('Meteorological', site['meteorological_columns']),
                ('Quality/Status', site['quality_columns']),
                ('Other', site['other_columns'])
            ]
            
            for cat_name, cat_columns in categories:
                if cat_columns:
                    html_content += f"""
                    <div class="column-category">
                        <h4>{cat_name} ({len(cat_columns)})</h4>
                        <div class="column-list">{', '.join(cat_columns)}</div>
                    </div>
                    """
            
            html_content += "</div>"
        
        html_content += "</body></html>"
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report saved: {html_file}")

def main():
    """Main function to run the MMF site survey."""
    parser = argparse.ArgumentParser(description='Survey MMF sites and generate comprehensive data summary')
    parser.add_argument('--output-dir', default='mmf_survey_results', help='Output directory (default: mmf_survey_results)')
    parser.add_argument('--detailed', action='store_true', help='Include detailed column statistics')
    parser.add_argument('--export-csv', action='store_true', help='Export results to CSV format')
    parser.add_argument('--export-html', action='store_true', help='Export results to HTML format')
    
    args = parser.parse_args()
    
    # Create and run survey
    survey = MMFSiteSurvey(
        output_dir=args.output_dir,
        detailed=args.detailed
    )
    
    # Survey all sites
    survey.survey_all_sites()
    
    # Generate reports
    survey.generate_summary_report()
    
    if args.export_csv:
        survey.export_to_csv()
    
    if args.export_html:
        survey.export_to_html()
    
    print(f"\n🎉 Survey complete! Results saved in: {survey.output_dir}")
    print("\n📋 Generated files:")
    for file in survey.output_dir.glob("*"):
        print(f"   📄 {file.name}")

if __name__ == "__main__":
    main()