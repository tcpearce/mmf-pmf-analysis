#!/usr/bin/env python3
"""
Enhanced 30-Minute Data Mapping Script

This script implements PMF best practices for temporal aggregation:
- Circular wind direction averaging (vector mean)
- Species-specific coverage gating
- Proper uncertainty handling
- Timezone management
- Comprehensive metadata

Usage:
    # Basic usage
    python enhanced_30min_data_mapper.py --station MMF2 --dry-run
    python enhanced_30min_data_mapper.py --all-stations --test
    
    # Conservative thresholds (keep more data)
    python enhanced_30min_data_mapper.py --station MMF9 --gas-min-samples 2 --pm-min-samples 1
    
    # Strict coverage gating (require full expected samples)
    python enhanced_30min_data_mapper.py --station MMF2 --enable-strict-gating
    
    # Disable coverage gating entirely
    python enhanced_30min_data_mapper.py --all-stations --disable-coverage-gating
    
    # Wind direction options
    python enhanced_30min_data_mapper.py --station MMF9 --disable-ws-weighting --wd-fallback-arithmetic
    
    # Custom configuration from JSON file
    python enhanced_30min_data_mapper.py --station MMF2 --coverage-config custom_coverage.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime
import argparse
import warnings
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Tuple, Optional, Union
import math

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_30min_mapping.log'),
        logging.StreamHandler()
    ]
)

class Enhanced30MinMapper:
    """Enhanced 30-minute data mapper implementing PMF best practices."""
    
    def __init__(self, output_dir="mmf_parquet_30min_enhanced", timezone=None, create_backup=True,
                 gas_min_samples=3, pm_min_samples=1, met_min_samples=2, voc_min_samples=1,
                 disable_coverage_gating=False, enable_strict_gating=False,
                 disable_ws_weighting=False, wd_fallback_arithmetic=False,
                 modal_interval_detection=False, verbose_coverage=False,
                 coverage_config=None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timezone = timezone
        self.create_backup = create_backup
        
        # Configuration options
        self.gas_min_samples = gas_min_samples
        self.pm_min_samples = pm_min_samples
        self.met_min_samples = met_min_samples
        self.voc_min_samples = voc_min_samples
        self.disable_coverage_gating = disable_coverage_gating
        self.enable_strict_gating = enable_strict_gating
        self.disable_ws_weighting = disable_ws_weighting
        self.wd_fallback_arithmetic = wd_fallback_arithmetic
        self.modal_interval_detection = modal_interval_detection
        self.verbose_coverage = verbose_coverage
        
        # Load custom coverage configuration if provided
        self.custom_coverage_config = None
        if coverage_config and Path(coverage_config).exists():
            try:
                with open(coverage_config, 'r') as f:
                    self.custom_coverage_config = json.load(f)
                logging.info(f"Loaded custom coverage config from {coverage_config}")
            except Exception as e:
                logging.warning(f"Failed to load coverage config: {str(e)}")
        
        # Production file mapping
        self.production_mapping = {
            'MMF1': 'mmf_parquet_final/MMF1_Cemetery_Road_combined_data.parquet',
            'MMF2': 'mmf_parquet_final/MMF2_Silverdale_Pumping_Station_combined_data.parquet',
            'MMF6': 'mmf_parquet_final/MMF6_Fire_Station_combined_data.parquet',
            'MMF9': 'mmf_parquet_final/MMF9_Galingale_View_combined_data.parquet',
            'Maries_Way': 'mmf_parquet_final/Maries_Way_combined_data.parquet'
        }
        
        # VOC mapping for integration
        self.voc_species = ['Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene']
        
        # Species configuration
        self.species_config = {
            # Gas species (typically 5-min native)
            'WD': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 3, 'aggregation': 'vector_mean'},
            'WS': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 3, 'aggregation': 'mean'},
            'CH4': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 3, 'aggregation': 'mean'},
            'NOX': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 3, 'aggregation': 'mean'},
            'NO': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 3, 'aggregation': 'mean'},
            'NO2': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 3, 'aggregation': 'mean'},
            'SO2': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 3, 'aggregation': 'mean'},
            'H2S': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 3, 'aggregation': 'mean'},
            
            # Particle species (typically 15-min native, but may be forward-filled to 5-min)
            'PM1 FIDAS': {'native_interval_min': 15, 'expected_n_30min': 2, 'min_samples': 1, 'aggregation': 'mean'},
            'PM2.5 FIDAS': {'native_interval_min': 15, 'expected_n_30min': 2, 'min_samples': 1, 'aggregation': 'mean'},
            'PM4 FIDAS': {'native_interval_min': 15, 'expected_n_30min': 2, 'min_samples': 1, 'aggregation': 'mean'},
            'PM10 FIDAS': {'native_interval_min': 15, 'expected_n_30min': 2, 'min_samples': 1, 'aggregation': 'mean'},
            'TSP FIDAS': {'native_interval_min': 15, 'expected_n_30min': 2, 'min_samples': 1, 'aggregation': 'mean'},
            
            # Meteorological (more lenient gating to preserve data for PMF)
            'TEMP': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 2, 'aggregation': 'mean'},
            'Pressure': {'native_interval_min': 5, 'expected_n_30min': 6, 'min_samples': 2, 'aggregation': 'mean'},
            
            # VOC species (typically 30-min native)
            'Benzene': {'native_interval_min': 30, 'expected_n_30min': 1, 'min_samples': 1, 'aggregation': 'mean'},
            'Toluene': {'native_interval_min': 30, 'expected_n_30min': 1, 'min_samples': 1, 'aggregation': 'mean'},
            'Ethylbenzene': {'native_interval_min': 30, 'expected_n_30min': 1, 'min_samples': 1, 'aggregation': 'mean'},
            'm&p-Xylene': {'native_interval_min': 30, 'expected_n_30min': 1, 'min_samples': 1, 'aggregation': 'mean'},
        }
        
        # Units mapping
        self.units_mapping = {
            'datetime': 'timestamp',
            'WD': 'degrees', 'WS': 'm/s',
            'CH4': 'mg/m3', 'NOX': 'ug/m3', 'NO': 'ug/m3', 'NO2': 'ug/m3',
            'SO2': 'ug/m3', 'H2S': 'ug/m3',
            'PM1 FIDAS': 'ug/m3', 'PM2.5 FIDAS': 'ug/m3', 'PM4 FIDAS': 'ug/m3', 
            'PM10 FIDAS': 'ug/m3', 'TSP FIDAS': 'ug/m3',
            'TEMP': 'degC', 'Pressure': 'hPa',
            'Benzene': 'ug/m3', 'Toluene': 'ug/m3', 'Ethylbenzene': 'ug/m3', 'm&p-Xylene': 'ug/m3',
            'gas_data_available': 'boolean', 'particle_data_available': 'boolean',
            'Odour_Reports': 'count'
        }
        
        # Update species configuration based on CLI parameters
        self._update_species_config()
    
    def _update_species_config(self):
        """Update species configuration based on CLI parameters and custom config."""
        
        # Define species categories
        gas_species = ['WD', 'WS', 'CH4', 'NOX', 'NO', 'NO2', 'SO2', 'H2S']
        pm_species = ['PM1 FIDAS', 'PM2.5 FIDAS', 'PM4 FIDAS', 'PM10 FIDAS', 'TSP FIDAS']
        met_species = ['TEMP', 'Pressure']
        voc_species = ['Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene']
        
        # Update minimum sample thresholds based on CLI parameters
        for species in gas_species:
            if species in self.species_config:
                if self.enable_strict_gating:
                    self.species_config[species]['min_samples'] = self.species_config[species]['expected_n_30min']
                elif not self.disable_coverage_gating:
                    self.species_config[species]['min_samples'] = self.gas_min_samples
                else:
                    self.species_config[species]['min_samples'] = 0
        
        for species in pm_species:
            if species in self.species_config:
                if self.enable_strict_gating:
                    self.species_config[species]['min_samples'] = self.species_config[species]['expected_n_30min']
                elif not self.disable_coverage_gating:
                    self.species_config[species]['min_samples'] = self.pm_min_samples
                else:
                    self.species_config[species]['min_samples'] = 0
        
        for species in met_species:
            if species in self.species_config:
                if self.enable_strict_gating:
                    self.species_config[species]['min_samples'] = self.species_config[species]['expected_n_30min']
                elif not self.disable_coverage_gating:
                    self.species_config[species]['min_samples'] = self.met_min_samples
                else:
                    self.species_config[species]['min_samples'] = 0
        
        for species in voc_species:
            if species in self.species_config:
                if self.enable_strict_gating:
                    self.species_config[species]['min_samples'] = self.species_config[species]['expected_n_30min']
                elif not self.disable_coverage_gating:
                    self.species_config[species]['min_samples'] = self.voc_min_samples
                else:
                    self.species_config[species]['min_samples'] = 0
        
        # Apply custom configuration overrides if provided
        if self.custom_coverage_config:
            for species, config in self.custom_coverage_config.items():
                if species in self.species_config:
                    for key, value in config.items():
                        if key in self.species_config[species]:
                            self.species_config[species][key] = value
                            logging.info(f"Custom config: {species}.{key} = {value}")
        
        # Log final configuration if verbose
        if self.verbose_coverage:
            logging.info("\nFinal species configuration:")
            for species, config in self.species_config.items():
                logging.info(f"  {species}: min_samples={config['min_samples']}, expected_n={config['expected_n_30min']}, aggregation={config['aggregation']}")
    
    def detect_modal_interval(self, df: pd.DataFrame, species: str) -> Optional[float]:
        """Detect modal interval for a species from actual data."""
        try:
            if species not in df.columns:
                return None
            
            # Get non-null data points
            species_data = df[df[species].notna()].copy()
            if len(species_data) < 2:
                return None
            
            # Calculate time differences
            species_data = species_data.sort_values('datetime')
            time_diffs = species_data['datetime'].diff().dropna()
            intervals_minutes = time_diffs.dt.total_seconds() / 60
            
            # Get mode (most common interval)
            if len(intervals_minutes) > 0:
                modal_interval = intervals_minutes.mode()
                if len(modal_interval) > 0:
                    return modal_interval.iloc[0]
            
            return None
            
        except Exception as e:
            logging.warning(f"Error detecting modal interval for {species}: {str(e)}")
            return None
    
    def vector_mean_wind_direction(self, wd_values: pd.Series, ws_values: pd.Series = None, 
                                  use_ws_weights: bool = None) -> float:
        """Compute vector mean wind direction, optionally weighted by wind speed."""
        try:
            # Use instance configuration if use_ws_weights not explicitly provided
            if use_ws_weights is None:
                use_ws_weights = not self.disable_ws_weighting
            
            # Remove NaN values
            if ws_values is not None and use_ws_weights:
                valid_mask = wd_values.notna() & ws_values.notna() & (ws_values > 0)
                wd = wd_values[valid_mask]
                ws = ws_values[valid_mask]
            else:
                valid_mask = wd_values.notna()
                wd = wd_values[valid_mask]
                ws = None
            
            if len(wd) == 0:
                return np.nan
            
            # Try vector mean calculation
            try:
                # Convert to radians
                wd_rad = np.deg2rad(wd)
                
                # Compute vector components
                if ws is not None and use_ws_weights:
                    u_components = ws * np.sin(wd_rad)
                    v_components = ws * np.cos(wd_rad)
                    mean_u = np.mean(u_components)
                    mean_v = np.mean(v_components)
                else:
                    mean_u = np.mean(np.sin(wd_rad))
                    mean_v = np.mean(np.cos(wd_rad))
                
                # Compute resultant direction
                mean_wd_rad = np.arctan2(mean_u, mean_v)
                mean_wd_deg = np.rad2deg(mean_wd_rad)
                
                # Ensure 0-360 range
                if mean_wd_deg < 0:
                    mean_wd_deg += 360
                
                # Check for invalid result
                if np.isnan(mean_wd_deg) and self.wd_fallback_arithmetic:
                    logging.info("Vector mean failed, falling back to arithmetic mean")
                    return np.mean(wd)
                
                return mean_wd_deg
                
            except Exception as vector_error:
                if self.wd_fallback_arithmetic:
                    logging.warning(f"Vector mean failed ({str(vector_error)}), falling back to arithmetic mean")
                    return np.mean(wd)
                else:
                    raise vector_error
            
        except Exception as e:
            logging.warning(f"Error in vector mean wind direction: {str(e)}")
            return np.nan
    
    def load_and_prepare_data(self, station_name: str) -> Optional[pd.DataFrame]:
        """Load production data and prepare for aggregation."""
        try:
            production_file = self.production_mapping.get(station_name)
            if not production_file:
                logging.error(f"No production file mapping for {station_name}")
                return None
                
            production_path = Path(production_file)
            if not production_path.exists():
                logging.error(f"Production file not found: {production_path}")
                return None
            
            logging.info(f"Loading production data: {production_path}")
            
            # Read production file
            df = pd.read_parquet(production_path)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Handle timezone if specified
            if self.timezone:
                if df['datetime'].dt.tz is None:
                    df['datetime'] = df['datetime'].dt.tz_localize(self.timezone)
                else:
                    df['datetime'] = df['datetime'].dt.tz_convert(self.timezone)
            
            # Sort and remove duplicates
            original_len = len(df)
            df = df.sort_values('datetime')
            df = df.drop_duplicates(subset=['datetime'], keep='first')
            
            if len(df) < original_len:
                logging.info(f"Removed {original_len - len(df)} duplicate timestamps")
            
            # Detect modal intervals for species
            self.update_species_config_from_data(df)
            
            logging.info(f"Loaded {len(df):,} records from {df['datetime'].min()} to {df['datetime'].max()}")
            logging.info(f"Available columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data for {station_name}: {str(e)}")
            return None
    
    def update_species_config_from_data(self, df: pd.DataFrame):
        """Update species configuration based on detected modal intervals."""
        for species in self.species_config:
            if species in df.columns:
                detected_interval = self.detect_modal_interval(df, species)
                if detected_interval is not None:
                    # Update expected_n based on detected interval
                    expected_n = max(1, round(30 / detected_interval))
                    self.species_config[species]['detected_interval_min'] = detected_interval
                    self.species_config[species]['expected_n_30min'] = expected_n
                    logging.debug(f"{species}: detected {detected_interval}min interval, expected_n={expected_n}")
    
    def aggregate_to_30min(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data to 30-minute timebase with PMF best practices."""
        try:
            logging.info("Aggregating to 30-minute timebase...")
            
            # Set datetime as index for resampling
            df_indexed = df.set_index('datetime')
            
            # Initialize result dataframe
            result_dict = {}
            count_dict = {}
            
            # Resample with explicit parameters
            resample_params = {
                'rule': '30min',
                'label': 'left',
                'closed': 'left',
                'origin': 'start_day'
            }
            
            # Define explicit numeric columns to process
            numeric_columns = ['WD', 'WS', 'CH4', 'NOX', 'NO', 'NO2', 'H2S', 'SO2', 
                              'Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene',
                              'PM1 FIDAS', 'PM2.5 FIDAS', 'PM4 FIDAS', 'PM10 FIDAS', 'TSP FIDAS',
                              'TEMP', 'Pressure']
            
            # Process each numeric column that exists in the data
            for col in numeric_columns:
                if col not in df_indexed.columns:
                    continue
                    
                logging.info(f"Processing column: {col} (dtype: {df_indexed[col].dtype})")
                
                # Get species configuration
                config = self.species_config.get(col, {
                    'native_interval_min': 5, 
                    'expected_n_30min': 6, 
                    'min_samples': 3, 
                    'aggregation': 'mean'
                })
                
                # Resample based on aggregation method
                if config['aggregation'] == 'vector_mean' and col == 'WD':
                    # Special handling for wind direction
                    ws_col = 'WS' if 'WS' in df_indexed.columns else None
                    ws_data = df_indexed[ws_col] if ws_col else None
                    
                    # Group by 30-min bins and apply vector mean
                    groups = df_indexed.groupby(pd.Grouper(freq=resample_params['rule'], 
                                                         label=resample_params['label'],
                                                         closed=resample_params['closed'],
                                                         origin=resample_params['origin']))
                    
                    wd_values = []
                    counts = []
                    
                    for name, group in groups:
                        if len(group) > 0:
                            if ws_data is not None:
                                wd_mean = self.vector_mean_wind_direction(group[col], group[ws_col])
                            else:
                                wd_mean = self.vector_mean_wind_direction(group[col])
                            wd_values.append(wd_mean)
                            counts.append(group[col].count())
                        else:
                            wd_values.append(np.nan)
                            counts.append(0)
                    
                    result_dict[col] = wd_values
                    # Only add count columns for non-VOC species (PMF compatibility)
                    if col not in self.voc_species:
                        count_dict[f'n_{col}'] = counts
                    
                else:
                    # Standard arithmetic mean aggregation
                    try:
                        resampled = df_indexed[[col]].resample(**resample_params)
                        result_dict[col] = resampled.mean()[col].values
                        # Only add count columns for non-VOC species (PMF compatibility)
                        if col not in self.voc_species:
                            count_dict[f'n_{col}'] = resampled.count()[col].values
                    except Exception as e:
                        logging.warning(f"Error aggregating {col}: {str(e)}, skipping column")
                        continue
            
            # Create result dataframe - get time index from first successful aggregation
            if result_dict:
                # Get time index from a sample resample operation
                sample_col = next(iter(result_dict.keys()))
                time_index = df_indexed[[sample_col]].resample(**resample_params).mean().index
                result_df = pd.DataFrame(result_dict, index=time_index)
            else:
                logging.error("No columns were successfully aggregated")
                return None
            
            # Add count columns
            for count_col, count_values in count_dict.items():
                result_df[count_col] = count_values
            
            # Reset index to get datetime back as column
            result_df = result_df.reset_index()
            
            logging.info(f"30-minute aggregation complete: {len(result_df):,} records")
            
            return result_df
            
        except Exception as e:
            logging.error(f"Error in 30-minute aggregation: {str(e)}")
            raise
    
    def apply_coverage_gating(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply species-specific coverage gating based on minimum sample requirements."""
        try:
            logging.info("Applying coverage gating...")
            
            gated_count = 0
            total_measurements = 0
            
            for species, config in self.species_config.items():
                if species in df.columns:
                    # Skip VOC species - they don't have n_ columns for PMF compatibility
                    if species in self.voc_species:
                        # VOC species at 30-min native interval don't need coverage gating
                        continue
                        
                    # For non-VOC species, apply gating if n_ column exists
                    if f'n_{species}' in df.columns:
                        min_samples = config['min_samples']
                        count_col = f'n_{species}'
                        
                        # Apply gating
                        mask = df[count_col] < min_samples
                        gated_values = mask.sum()
                        total_values = len(df)
                        
                        if gated_values > 0:
                            df.loc[mask, species] = np.nan
                            gated_count += gated_values
                            total_measurements += total_values
                            
                            logging.info(f"{species}: gated {gated_values}/{total_values} "
                                       f"({gated_values/total_values*100:.1f}%) values with n < {min_samples}")
            
            if gated_count > 0:
                logging.info(f"Total gating: {gated_count} values removed due to insufficient samples")
            
            return df
            
        except Exception as e:
            logging.error(f"Error in coverage gating: {str(e)}")
            return df
    
    def add_availability_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add data availability flags."""
        try:
            # Gas species
            gas_cols = [col for col in df.columns if col in ['WD', 'WS', 'CH4', 'NOX', 'NO', 'NO2', 'SO2', 'H2S']]
            if gas_cols:
                df['gas_data_available'] = df[gas_cols].notna().any(axis=1)
            else:
                df['gas_data_available'] = False
            
            # Particle species
            pm_cols = [col for col in df.columns if 'PM' in col or 'TSP' in col]
            if pm_cols:
                df['particle_data_available'] = df[pm_cols].notna().any(axis=1)
            else:
                df['particle_data_available'] = False
            
            logging.info(f"Added availability flags: {len(gas_cols)} gas, {len(pm_cols)} particle species")
            
            return df
            
        except Exception as e:
            logging.error(f"Error adding availability flags: {str(e)}")
            return df
    
    def integrate_voc_data(self, df: pd.DataFrame, station_name: str) -> pd.DataFrame:
        """Integrate VOC data if available in production files."""
        try:
            # Check if VOC data already exists
            existing_voc = [col for col in self.voc_species if col in df.columns]
            if existing_voc:
                logging.info(f"VOC data already present: {existing_voc}")
                
                # Skip n_ columns for VOC species to maintain PMF compatibility
                # VOC species are handled differently and don't need count columns
                logging.info("VOC species present - maintaining PMF compatibility by not adding n_ columns")
                
                return df
            
            # If no VOC data, check if production file has it
            production_file = self.production_mapping.get(station_name)
            if not production_file or not Path(production_file).exists():
                return df
            
            production_df = pd.read_parquet(production_file)
            production_df['datetime'] = pd.to_datetime(production_df['datetime'])
            
            # Check for VOC columns in production
            available_voc = [col for col in self.voc_species if col in production_df.columns]
            if not available_voc:
                logging.info(f"No VOC data available in production file for {station_name}")
                return df
            
            logging.info(f"Integrating VOC data: {available_voc}")
            
            # Extract VOC data with same datetime range as main data
            voc_data = production_df[['datetime'] + available_voc].copy()
            voc_data = voc_data[(voc_data['datetime'] >= df['datetime'].min()) & 
                               (voc_data['datetime'] <= df['datetime'].max())]
            
            # Merge with main dataframe
            result = df.merge(voc_data, on='datetime', how='left')
            
            # Skip n_ columns for VOC species to maintain PMF compatibility
            # The PMF app handles VOC species detection without needing count columns
            logging.info("VOC integration complete - no n_ columns added for PMF compatibility")
            
            # Log integration results
            for voc_species in available_voc:
                voc_count = result[voc_species].notna().sum()
                total_count = len(result)
                coverage = (voc_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"  {voc_species}: {voc_count:,}/{total_count:,} ({coverage:.1f}% coverage)")
            
            return result
            
        except Exception as e:
            logging.error(f"Error integrating VOC data: {str(e)}")
            return df
    
    def preserve_complaint_data(self, df: pd.DataFrame, station_name: str) -> pd.DataFrame:
        """Preserve existing complaint data from current 30-minute files or derive from production files."""
        try:
            # Check if complaint data already exists in the aggregated data
            if 'Odour_Reports' in df.columns:
                complaint_count = (df['Odour_Reports'] > 0).sum()
                logging.info(f"Complaint data already present in aggregated data: {complaint_count} complaint records")
                return df
            
            # Try to load complaint data from existing 30-minute file first
            existing_30min_file = self.output_dir / f"{station_name}_combined_data.parquet"
            if existing_30min_file.exists():
                logging.info(f"Loading complaint data from existing 30-minute file: {existing_30min_file}")
                try:
                    existing_df = pd.read_parquet(existing_30min_file)
                    existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                    
                    if 'Odour_Reports' in existing_df.columns:
                        # Merge complaint data based on datetime
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        complaint_data = existing_df[['datetime', 'Odour_Reports']].copy()
                        
                        # Merge preserving all records from df
                        result = df.merge(complaint_data, on='datetime', how='left')
                        
                        # Only fill missing with -1.0 if some complaint data exists
                        if (result['Odour_Reports'] > 0).sum() > 0:
                            result['Odour_Reports'] = result['Odour_Reports'].fillna(-1.0)
                            complaint_count = (result['Odour_Reports'] > 0).sum()
                            logging.info(f"Preserved {complaint_count} complaint records from existing 30-minute file")
                            return result
                        else:
                            logging.info("Existing file has no actual complaint data, checking production files")
                    else:
                        logging.info("No Odour_Reports column in existing file, checking production files")
                except Exception as e:
                    logging.warning(f"Error loading existing complaint data: {str(e)}, checking production files")
            
            # Try to derive complaint data from production file
            production_file = self.production_mapping.get(station_name)
            if production_file and Path(production_file).exists():
                logging.info(f"Checking for complaint data in production file: {production_file}")
                try:
                    production_df = pd.read_parquet(production_file)
                    production_df['datetime'] = pd.to_datetime(production_df['datetime'])
                    
                    if 'Odour_Reports' in production_df.columns:
                        # Resample complaint data to 30-minute intervals using sum aggregation
                        # (complaints should be additive within time windows)
                        production_indexed = production_df.set_index('datetime')
                        complaint_30min = production_indexed[['Odour_Reports']].resample('30min', 
                                                                                        label='left', 
                                                                                        closed='left', 
                                                                                        origin='start_day').sum()
                        complaint_30min = complaint_30min.reset_index()
                        
                        # Merge with main dataframe
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        result = df.merge(complaint_30min, on='datetime', how='left')
                        
                        # Fill missing complaint data with -1.0 only where no complaints occurred
                        result['Odour_Reports'] = result['Odour_Reports'].fillna(-1.0)
                        
                        complaint_count = (result['Odour_Reports'] > 0).sum()
                        total_complaints = result['Odour_Reports'][result['Odour_Reports'] > 0].sum()
                        logging.info(f"Derived complaint data from production: {complaint_count} time periods with complaints, {total_complaints} total complaints")
                        
                        return result
                    else:
                        logging.info(f"No complaint data in production file for {station_name}")
                except Exception as e:
                    logging.warning(f"Error deriving complaint data from production: {str(e)}")
            
            # If no complaint data found anywhere, return without adding Odour_Reports column
            logging.info(f"No complaint data found for {station_name} - not adding Odour_Reports column")
            return df
            
        except Exception as e:
            logging.error(f"Error preserving complaint data: {str(e)}")
            return df
    
    def create_metadata(self, df: pd.DataFrame, station_name: str) -> Dict:
        """Create comprehensive metadata for the parquet file."""
        metadata = {}
        
        # Processing metadata
        metadata['processing_date'] = datetime.now().isoformat()
        metadata['station'] = station_name
        metadata['source'] = 'enhanced_30min_mapper'
        metadata['aggregation_timebase'] = '30min'
        metadata['aggregation_method'] = 'species_specific'
        
        # Resample parameters
        metadata['resample_rule'] = '30min'
        metadata['resample_label'] = 'left'
        metadata['resample_closed'] = 'left'
        metadata['resample_origin'] = 'start_day'
        
        # Timezone
        metadata['timezone'] = str(self.timezone) if self.timezone else 'naive'
        
        # Species configuration
        metadata['species_config'] = json.dumps(self.species_config)
        
        # Column units
        for col in df.columns:
            if col in self.units_mapping:
                metadata[f"{col}_unit"] = self.units_mapping[col]
            elif col.startswith('n_'):
                metadata[f"{col}_unit"] = 'count'
            else:
                metadata[f"{col}_unit"] = 'unknown'
        
        # Coverage statistics
        total_records = len(df)
        for col in df.columns:
            if col not in ['datetime'] and pd.api.types.is_numeric_dtype(df[col]):
                non_null = df[col].notna().sum()
                coverage = (non_null / total_records) * 100 if total_records > 0 else 0
                metadata[f"{col}_coverage_pct"] = f"{coverage:.1f}"
        
        return metadata
    
    def save_enhanced_30min_file(self, df: pd.DataFrame, station_name: str, dry_run: bool = False) -> bool:
        """Save enhanced 30-minute parquet file with comprehensive metadata."""
        try:
            if dry_run:
                output_path = self.output_dir / f"{station_name}_combined_data_DRYRUN.parquet"
            else:
                output_path = self.output_dir / f"{station_name}_combined_data.parquet"
                
                # Create backup if file exists and not in dry-run
                if self.create_backup and output_path.exists():
                    backup_path = output_path.with_suffix(f'.parquet.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
                    output_path.rename(backup_path)
                    logging.info(f"Created backup: {backup_path}")
            
            # Create metadata
            metadata = self.create_metadata(df, station_name)
            
            # Create PyArrow table
            table = pa.Table.from_pandas(df)
            
            # Add metadata
            arrow_metadata = {k.encode(): str(v).encode() for k, v in metadata.items()}
            existing_metadata = table.schema.metadata or {}
            existing_metadata.update(arrow_metadata)
            table = table.replace_schema_metadata(existing_metadata)
            
            # Save parquet file
            pq.write_table(table, output_path)
            
            mode = "DRY-RUN" if dry_run else "SAVED"
            logging.info(f"{mode}: {output_path} ({len(df):,} records)")
            
            # Create summary file
            summary_path = output_path.with_suffix('.txt')
            self.create_summary_file(df, station_name, summary_path, dry_run)
            
            return True
            
        except Exception as e:
            logging.error(f"Error saving file for {station_name}: {str(e)}")
            return False
    
    def create_summary_file(self, df: pd.DataFrame, station_name: str, output_path: Path, dry_run: bool = False):
        """Create detailed summary file."""
        try:
            mode = "DRY-RUN " if dry_run else ""
            
            with open(output_path, 'w') as f:
                f.write(f"{mode}Enhanced 30min File Summary for {station_name}\n")
                f.write("=" * 70 + "\n\n")
                f.write(f"Created: {datetime.now().isoformat()}\n")
                f.write(f"Source: Enhanced 30min mapper with PMF best practices\n")
                f.write(f"Records: {len(df):,}\n")
                f.write(f"Columns: {len(df.columns)}\n")
                f.write(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}\n")
                f.write(f"Timezone: {self.timezone or 'naive'}\n\n")
                
                # Species configuration used
                f.write("Species Configuration Applied:\n")
                f.write("-" * 40 + "\n")
                for species, config in self.species_config.items():
                    if species in df.columns:
                        f.write(f"{species}:\n")
                        f.write(f"  Native interval: {config['native_interval_min']} min\n")
                        f.write(f"  Expected n/30min: {config['expected_n_30min']}\n")
                        f.write(f"  Min samples: {config['min_samples']}\n")
                        f.write(f"  Aggregation: {config['aggregation']}\n")
                        if 'detected_interval_min' in config:
                            f.write(f"  Detected interval: {config['detected_interval_min']} min\n")
                        f.write("\n")
                
                # Data coverage
                f.write("Data Coverage:\n")
                f.write("-" * 40 + "\n")
                gas_cols = [col for col in df.columns if col in ['WD', 'WS', 'CH4', 'NOX', 'NO', 'NO2', 'SO2', 'H2S']]
                pm_cols = [col for col in df.columns if 'PM' in col or 'TSP' in col]
                voc_cols = [col for col in df.columns if col in self.voc_species]
                meta_cols = [col for col in df.columns if col in ['TEMP', 'Pressure']]
                
                for category, cols in [("Gas", gas_cols), ("Particle", pm_cols), 
                                     ("VOC", voc_cols), ("Meteorological", meta_cols)]:
                    if cols:
                        f.write(f"\n{category} Species:\n")
                        for col in cols:
                            non_null = df[col].notna().sum()
                            percentage = (non_null / len(df)) * 100
                            f.write(f"  {col}: {percentage:.1f}% ({non_null:,}/{len(df):,})\n")
                
                # Sample counts summary
                f.write(f"\nSample Count Summary:\n")
                f.write("-" * 40 + "\n")
                count_cols = [col for col in df.columns if col.startswith('n_')]
                for count_col in sorted(count_cols):
                    species = count_col[2:]  # Remove 'n_' prefix
                    if species in self.species_config:
                        min_req = self.species_config[species]['min_samples']
                        counts = df[count_col]
                        above_min = (counts >= min_req).sum()
                        below_min = (counts < min_req).sum()
                        f.write(f"  {species}: {above_min} above min ({min_req}), {below_min} below\n")
            
            logging.info(f"Created summary: {output_path}")
            
        except Exception as e:
            logging.error(f"Error creating summary file: {str(e)}")
    
    def process_station(self, station_name: str, dry_run: bool = False) -> bool:
        """Process a single station with enhanced 30-minute mapping."""
        try:
            logging.info(f"\n{'='*70}")
            mode = "DRY-RUN " if dry_run else ""
            logging.info(f"{mode}Processing {station_name} with Enhanced 30min Mapping")
            logging.info(f"{'='*70}")
            
            # Step 1: Load and prepare data
            df = self.load_and_prepare_data(station_name)
            if df is None:
                return False
            
            # Step 2: Aggregate to 30-minute timebase
            df_30min = self.aggregate_to_30min(df)
            
            # Step 3: Apply coverage gating
            df_30min = self.apply_coverage_gating(df_30min)
            
            # Step 4: Add availability flags
            df_30min = self.add_availability_flags(df_30min)
            
            # Step 5: Integrate VOC data
            df_30min = self.integrate_voc_data(df_30min, station_name)
            
            # Step 6: Preserve existing complaint data or add placeholder
            df_30min = self.preserve_complaint_data(df_30min, station_name)
            
            # Step 7: Save file
            success = self.save_enhanced_30min_file(df_30min, station_name, dry_run)
            
            if success:
                mode_msg = "DRY-RUN completed" if dry_run else "Successfully processed"
                logging.info(f"{mode_msg} for {station_name}")
                return True
            else:
                logging.error(f"Failed to process {station_name}")
                return False
                
        except Exception as e:
            logging.error(f"Error processing {station_name}: {str(e)}")
            return False


def create_test_data():
    """Create synthetic test data for validation."""
    logging.info("Creating synthetic test data for validation...")
    
    # Create test timebase (5-min intervals)
    dates = pd.date_range('2024-01-01', periods=8640, freq='5min')  # 30 days
    
    test_data = {
        'datetime': dates,
        'WD': np.random.uniform(0, 360, len(dates)),  # Wind direction
        'WS': np.random.exponential(2, len(dates)),   # Wind speed
        'CH4': np.random.lognormal(2, 1, len(dates)), # Methane
        'PM2.5 FIDAS': np.random.gamma(2, 2, len(dates)),  # PM2.5
    }
    
    # Add some missing values
    missing_mask = np.random.random(len(dates)) < 0.1  # 10% missing
    test_data['CH4'] = np.where(missing_mask, np.nan, test_data['CH4'])
    
    # Create test DataFrame
    test_df = pd.DataFrame(test_data)
    
    # Save as temporary parquet file
    test_path = Path('test_production_data.parquet')
    test_df.to_parquet(test_path)
    
    logging.info(f"Created test data: {len(test_df)} records, saved to {test_path}")
    return test_path


def run_unit_tests():
    """Run unit tests for key functions."""
    logging.info("\n" + "="*50)
    logging.info("RUNNING UNIT TESTS")
    logging.info("="*50)
    
    mapper = Enhanced30MinMapper()
    
    # Test 1: Vector mean wind direction
    logging.info("\nTest 1: Vector mean wind direction")
    
    # Test case: winds from N (359°, 1°) should average to ~0°
    wd_test = pd.Series([359, 1, 0, 358, 2])
    ws_test = pd.Series([5, 5, 5, 5, 5])
    
    result = mapper.vector_mean_wind_direction(wd_test, ws_test)
    logging.info(f"Input WD: {list(wd_test)}")
    logging.info(f"Vector mean WD: {result:.1f}°")
    
    # Should be close to 0°
    assert abs(result) < 5 or abs(result - 360) < 5, f"Expected ~0°, got {result}°"
    logging.info("[PASS] Test 1 PASSED")
    
    # Test 2: Coverage gating
    logging.info("\nTest 2: Coverage gating")
    
    test_df = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=10, freq='30min'),
        'CH4': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'n_CH4': [1, 2, 3, 4, 5, 0, 1, 2, 3, 6]  # Some below threshold of 3
    })
    
    result_df = mapper.apply_coverage_gating(test_df.copy())
    gated_count = result_df['CH4'].isna().sum()
    expected_gated = (test_df['n_CH4'] < 3).sum()
    
    logging.info(f"Original CH4 values: {list(test_df['CH4'])}")
    logging.info(f"Sample counts: {list(test_df['n_CH4'])}")
    logging.info(f"Gated values: {gated_count}, Expected: {expected_gated}")
    
    assert gated_count == expected_gated, f"Expected {expected_gated} gated, got {gated_count}"
    logging.info("[PASS] Test 2 PASSED")
    
    # Test 3: Modal interval detection
    logging.info("\nTest 3: Modal interval detection")
    
    # Create data with 5-min intervals
    test_dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    test_df = pd.DataFrame({
        'datetime': test_dates,
        'CH4': np.random.random(len(test_dates))
    })
    
    detected_interval = mapper.detect_modal_interval(test_df, 'CH4')
    logging.info(f"Detected interval for CH4: {detected_interval} min")
    
    assert detected_interval == 5.0, f"Expected 5.0 min, got {detected_interval}"
    logging.info("[PASS] Test 3 PASSED")
    
    logging.info("\n[SUCCESS] ALL UNIT TESTS PASSED!")
    return True


def get_user_confirmation_for_batch_regeneration(stations: List[str], output_dir: str) -> bool:
    """Get user confirmation before batch regeneration with diff summaries."""
    print(f"\n{'='*60}")
    print("BATCH REGENERATION CONFIRMATION")
    print(f"{'='*60}")
    print(f"You are about to regenerate 30-minute parquet files for:")
    print(f"  Stations: {', '.join(stations)}")
    print(f"  Output directory: {output_dir}")
    print(f"\nThis will:")
    print(f"  1. Create backup copies of existing files (if any)")
    print(f"  2. Overwrite existing 30-minute parquet files")
    print(f"  3. Apply enhanced PMF aggregation methods")
    print(f"  4. Update metadata and processing provenance")
    
    # Check for existing files and show what will be affected
    output_path = Path(output_dir)
    existing_files = []
    for station in stations:
        station_file = output_path / f"{station}_combined_data.parquet"
        if station_file.exists():
            existing_files.append(str(station_file))
    
    if existing_files:
        print(f"\nExisting files that will be overwritten:")
        for file_path in existing_files:
            file_stat = Path(file_path).stat()
            file_size = file_stat.st_size / (1024*1024)  # MB
            file_mtime = datetime.fromtimestamp(file_stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  {file_path} ({file_size:.1f}MB, modified: {file_mtime})")
    else:
        print(f"\nNo existing files found - will create new files")
    
    print(f"\nSafety measures:")
    print(f"  - Original files will be backed up with .backup extension")
    print(f"  - Processing logs will be saved for review")
    print(f"  - Dry-run mode is available (--dry-run) to preview changes")
    
    while True:
        response = input(f"\nDo you want to proceed with batch regeneration? [y/N]: ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no', '']:
            print("Operation cancelled by user.")
            return False
        else:
            print("Please enter 'y' for yes or 'n' for no.")


def compare_with_existing(station_name: str, enhanced_path: Path, original_path: Path):
    """Compare enhanced output with existing 30-min file."""
    logging.info(f"\n{'='*50}")
    logging.info(f"COMPARING ENHANCED vs ORIGINAL: {station_name}")
    logging.info(f"{'='*50}")
    
    try:
        if not enhanced_path.exists():
            logging.error(f"Enhanced file not found: {enhanced_path}")
            return
        
        if not original_path.exists():
            logging.info(f"Original file not found: {original_path}")
            return
        
        # Load both files
        enhanced_df = pd.read_parquet(enhanced_path)
        original_df = pd.read_parquet(original_path)
        
        enhanced_df['datetime'] = pd.to_datetime(enhanced_df['datetime'])
        original_df['datetime'] = pd.to_datetime(original_df['datetime'])
        
        logging.info(f"Enhanced: {len(enhanced_df)} records, {len(enhanced_df.columns)} columns")
        logging.info(f"Original: {len(original_df)} records, {len(original_df.columns)} columns")
        
        # Compare common columns
        common_cols = set(enhanced_df.columns) & set(original_df.columns)
        enhanced_only = set(enhanced_df.columns) - set(original_df.columns)
        original_only = set(original_df.columns) - set(enhanced_df.columns)
        
        logging.info(f"\nColumn comparison:")
        logging.info(f"  Common columns: {len(common_cols)}")
        logging.info(f"  Enhanced only: {sorted(enhanced_only)}")
        logging.info(f"  Original only: {sorted(original_only)}")
        
        # Merge on datetime for comparison
        merged = enhanced_df.merge(original_df, on='datetime', how='inner', suffixes=('_enh', '_orig'))
        
        logging.info(f"\nOverlapping records: {len(merged)}")
        
        # Compare key species
        comparison_cols = ['WD', 'WS', 'CH4', 'NOX', 'PM2.5 FIDAS']
        
        for col in comparison_cols:
            if f'{col}_enh' in merged.columns and f'{col}_orig' in merged.columns:
                enh_vals = merged[f'{col}_enh'].dropna()
                orig_vals = merged[f'{col}_orig'].dropna()
                
                if len(enh_vals) > 0 and len(orig_vals) > 0:
                    # Compute differences
                    common_mask = merged[[f'{col}_enh', f'{col}_orig']].notna().all(axis=1)
                    if common_mask.sum() > 0:
                        diff = merged.loc[common_mask, f'{col}_enh'] - merged.loc[common_mask, f'{col}_orig']
                        
                        logging.info(f"\n{col} comparison:")
                        logging.info(f"  Enhanced mean: {enh_vals.mean():.3f}")
                        logging.info(f"  Original mean: {orig_vals.mean():.3f}")
                        logging.info(f"  Mean difference: {diff.mean():.3f}")
                        logging.info(f"  Max abs difference: {abs(diff).max():.3f}")
                        
                        # Special handling for wind direction
                        if col == 'WD':
                            # Check for 360° wraparound differences
                            circular_diff = np.minimum(abs(diff), 360 - abs(diff))
                            logging.info(f"  Max circular difference: {circular_diff.max():.1f}°")
        
        # Coverage comparison
        logging.info(f"\nCoverage comparison:")
        for col in ['CH4', 'NOX', 'PM2.5 FIDAS']:
            if col in enhanced_df.columns and col in original_df.columns:
                enh_coverage = enhanced_df[col].notna().mean() * 100
                orig_coverage = original_df[col].notna().mean() * 100
                logging.info(f"  {col}: Enhanced {enh_coverage:.1f}%, Original {orig_coverage:.1f}%")
        
    except Exception as e:
        logging.error(f"Error in comparison: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced 30-minute data mapper with PMF best practices")
    parser.add_argument('--station', choices=['MMF1', 'MMF2', 'MMF6', 'MMF9', 'Maries_Way'], 
                       help="Process specific station")
    parser.add_argument('--all-stations', action='store_true', help="Process all stations")
    parser.add_argument('--dry-run', action='store_true', help="Dry run mode (no overwrite)")
    parser.add_argument('--test', action='store_true', help="Run unit tests")
    parser.add_argument('--compare', action='store_true', help="Compare with existing files")
    parser.add_argument('--timezone', default=None, help="Timezone (e.g., 'Europe/London')")
    parser.add_argument('--output-dir', default='mmf_parquet_30min_enhanced', help="Output directory")
    
    # Species-specific threshold configuration
    parser.add_argument('--gas-min-samples', type=int, default=3, 
                       help='Minimum sample count for gas species (default: 3 out of 6 expected)')
    parser.add_argument('--pm-min-samples', type=int, default=1, 
                       help='Minimum sample count for PM species (default: 1 out of 2 expected)')
    parser.add_argument('--met-min-samples', type=int, default=2, 
                       help='Minimum sample count for meteorological species (default: 2 out of 6 expected)')
    parser.add_argument('--voc-min-samples', type=int, default=1, 
                       help='Minimum sample count for VOC species (default: 1 out of 1 expected)')
    
    # Coverage gating options
    parser.add_argument('--disable-coverage-gating', action='store_true',
                       help='Disable coverage gating (keep all data regardless of sample count)')
    parser.add_argument('--enable-strict-gating', action='store_true',
                       help='Enable strict coverage gating (require full expected samples)')
    parser.add_argument('--coverage-config', type=str, 
                       help='JSON file with custom coverage configuration per species')
    
    # Wind direction options
    parser.add_argument('--disable-ws-weighting', action='store_true',
                       help='Disable wind speed weighting in vector mean wind direction')
    parser.add_argument('--wd-fallback-arithmetic', action='store_true',
                       help='Use arithmetic mean as fallback for wind direction when vector mean fails')
    
    # Advanced options
    parser.add_argument('--modal-interval-detection', action='store_true',
                       help='Enable automatic modal interval detection from data')
    parser.add_argument('--verbose-coverage', action='store_true',
                       help='Enable verbose coverage reporting and warnings')
    
    # Batch processing options
    parser.add_argument('--force', action='store_true',
                       help='Skip user confirmation prompts for batch operations')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup files (not recommended)')
    
    args = parser.parse_args()
    
    # Run unit tests if requested
    if args.test:
        success = run_unit_tests()
        if not success:
            return 1
    
    # Create mapper with all configuration options
    mapper = Enhanced30MinMapper(
        output_dir=args.output_dir,
        timezone=args.timezone,
        create_backup=not args.dry_run and not args.no_backup,
        gas_min_samples=args.gas_min_samples,
        pm_min_samples=args.pm_min_samples,
        met_min_samples=args.met_min_samples,
        voc_min_samples=args.voc_min_samples,
        disable_coverage_gating=args.disable_coverage_gating,
        enable_strict_gating=args.enable_strict_gating,
        disable_ws_weighting=args.disable_ws_weighting,
        wd_fallback_arithmetic=args.wd_fallback_arithmetic,
        modal_interval_detection=args.modal_interval_detection,
        verbose_coverage=args.verbose_coverage,
        coverage_config=args.coverage_config
    )
    
    # Determine stations to process
    if args.station:
        stations = [args.station]
    elif args.all_stations:
        stations = list(mapper.production_mapping.keys())
    else:
        logging.error("Must specify --station or --all-stations")
        return 1
    
    # User confirmation for batch operations (unless dry-run or force flag)
    if not args.dry_run and not args.force and len(stations) > 1:
        if not get_user_confirmation_for_batch_regeneration(stations, args.output_dir):
            logging.info("Batch regeneration cancelled by user.")
            return 0
    elif not args.dry_run and not args.force and len(stations) == 1:
        # Single station confirmation for real runs
        station_file = Path(args.output_dir) / f"{stations[0]}_combined_data.parquet"
        if station_file.exists() and not args.no_backup:
            print(f"\nWARNING: This will overwrite existing file:")
            print(f"  {station_file}")
            response = input(f"Continue? [y/N]: ").strip().lower()
            if response not in ['y', 'yes']:
                logging.info("Operation cancelled by user.")
                return 0
    
    # Process stations
    successful = []
    failed = []
    
    for station in stations:
        success = mapper.process_station(station, dry_run=args.dry_run)
        if success:
            successful.append(station)
        else:
            failed.append(station)
    
    # Summary
    logging.info(f"\n{'='*50}")
    logging.info("PROCESSING SUMMARY")
    logging.info(f"{'='*50}")
    logging.info(f"Successful: {len(successful)} - {', '.join(successful)}")
    logging.info(f"Failed: {len(failed)} - {', '.join(failed) if failed else 'None'}")
    
    # Compare with existing files if requested
    if args.compare and successful:
        for station in successful:
            enhanced_path = Path(args.output_dir) / f"{station}_combined_data_DRYRUN.parquet"
            original_path = Path("mmf_parquet_30min") / f"{station}_combined_data.parquet"
            compare_with_existing(station, enhanced_path, original_path)
    
    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main())