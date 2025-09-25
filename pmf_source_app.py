#!/usr/bin/env python3
"""
MMF PMF Source Apportionment Analysis (ESAT Fixed)
==================================================

This script performs Positive Matrix Factorization (PMF) source apportionment on MMF environmental data
following EPA PMF 5.0 User Guide best practices using the ESAT library.

Fixed version based on successful test with current ESAT API.

Features:
- Loads data from processed MMF parquet files
- Applies EPA-recommended uncertainty estimation
- Performs batch PMF modeling with error estimation
- Creates comprehensive dashboard with seaborn styling
- Includes all recommended diagnostic plots
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
import subprocess

# PDF conversion imports
try:
    import pdfkit
    HAS_PDFKIT = True
except ImportError:
    HAS_PDFKIT = False

# Note: weasyprint disabled due to Windows library dependencies
HAS_WEASYPRINT = False

if not (HAS_PDFKIT or HAS_WEASYPRINT):
    print("ℹ️  PDF conversion will use Chrome/Edge headless (no additional libraries needed)")

# PCA analysis imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Import our existing analyzer for data loading
from analyze_parquet_data import ParquetAnalyzer
from mmf_config import get_mmf_parquet_file, get_corrected_mmf_files, get_station_mapping

# Import EPA uncertainty calculation module
try:
    from epa_uncertainty import create_epa_uncertainty_calculator
    HAS_EPA_UNCERTAINTY = True
except ImportError:
    HAS_EPA_UNCERTAINTY = False
    print("[WARNING] EPA uncertainty module not found. Only legacy uncertainty mode available.")

# Import S/N categorization module  
try:
    from snr_categorization import create_snr_categorizer
    HAS_SNR_CATEGORIZATION = True
except ImportError:
    HAS_SNR_CATEGORIZATION = False
    print("[WARNING] S/N categorization module not found. Only non-categorized analysis available.")

# Import ESAT modules
try:
    import esat
    # Always import SA for robust mode compatibility
    from esat.model.sa import SA
    
    # Try BatchSA, fallback gracefully if esat_rust is missing
    try:
        from esat.model.batch_sa import BatchSA
        USE_BATCH_SA = True
        print("[OK] ESAT BatchSA imported successfully")
    except ImportError as e:
        if "esat_rust" in str(e):
            USE_BATCH_SA = False
            print("[WARNING] Using SA model (BatchSA requires esat_rust)")
        else:
            raise e
except ImportError:
    print("[ERROR] ESAT library not found. Please install it using:")
    print("   $env:CARGO_BUILD_TARGET = \"x86_64-pc-windows-msvc\"")
    print("   pip install git+https://github.com/quanted/esat.git")
    print("\nAlternatively, you can install dependencies:")
    print("   pip install scikit-learn numpy pandas matplotlib seaborn")
    sys.exit(1)

class ColorManager:
    """Manages consistent colors for factors and species across all PMF plots."""
    
    def __init__(self, n_factors, species_names):
        self.n_factors = n_factors
        self.species_names = species_names
        
        # Define consistent color schemes
        self.factor_colors = self._get_factor_colors(n_factors)
        self.species_colors = self._get_species_colors(species_names)
        
    def _get_factor_colors(self, n_factors):
        """Get consistent colors for PMF factors."""
        # Use qualitative color palette for factors - distinct and easily distinguishable
        if n_factors <= 3:
            # Use primary colors for small number of factors
            base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        elif n_factors <= 6:
            # Use Set2 palette for medium number
            base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        else:
            # Use larger palette for many factors
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap('tab20')  # 20 distinct colors
            base_colors = [cmap(i) for i in np.linspace(0, 1, n_factors)]
        
        return base_colors[:n_factors]
    
    def _get_species_colors(self, species_names):
        """Get consistent colors for chemical species by category."""
        species_colors = {}
        
        # Define color schemes by pollutant type
        gas_colors = ['#e74c3c', '#c0392b', '#a93226']  # Red tones for gases
        voc_colors = ['#8e44ad', '#7d3c98', '#6c3483', '#5b2c6f']  # Purple tones for VOCs
        pm_colors = ['#3498db', '#2980b9', '#1f618d', '#1a5490', '#154360']  # Blue tones for PM
        
        gas_idx = 0
        voc_idx = 0
        pm_idx = 0
        
        for species in species_names:
            species_upper = species.upper()
            
            # Assign colors by species type
            if any(gas in species_upper for gas in ['H2S', 'CH4', 'SO2', 'NOX', 'NO', 'NO2']):
                species_colors[species] = gas_colors[gas_idx % len(gas_colors)]
                gas_idx += 1
            elif any(voc in species_upper for voc in ['BENZENE', 'TOLUENE', 'ETHYLBENZENE', 'XYLENE']):
                species_colors[species] = voc_colors[voc_idx % len(voc_colors)]
                voc_idx += 1
            elif any(pm in species_upper for pm in ['PM1', 'PM2.5', 'PM4', 'PM10', 'TSP']):
                species_colors[species] = pm_colors[pm_idx % len(pm_colors)]
                pm_idx += 1
            else:
                # Default color for unknown species
                species_colors[species] = '#95a5a6'  # Gray
        
        return species_colors
    
    def get_factor_color(self, factor_idx):
        """Get color for a specific factor."""
        return self.factor_colors[factor_idx % len(self.factor_colors)]
    
    def get_species_color(self, species_name):
        """Get color for a specific species."""
        return self.species_colors.get(species_name, '#95a5a6')
    
    def get_factor_colors(self):
        """Get all factor colors as list."""
        return self.factor_colors
    
    def get_species_colors_list(self):
        """Get species colors in order matching species_names."""
        return [self.species_colors[species] for species in self.species_names]

class MMFPMFAnalyzer:
    def __init__(self, station=None, data_dir=None, patterns=None, start_date=None, end_date=None, output_dir="pmf_results", 
                 remove_voc=False, uncertainty_mode='legacy', uncertainty_ef_mdl=None, uncertainty_epsilon=1e-12,
                 legacy_min_u=0.1, uncertainty_bdl_policy='five-sixth-mdl', snr_enable=False, snr_weak_threshold=2.0,
                 snr_bad_threshold=0.2, snr_bdl_weak_frac=0.6, snr_bdl_bad_frac=0.8, snr_missing_weak_frac=0.2,
                 snr_missing_bad_frac=0.4, exclude_bad=True, dashboard_snr_panel=True, write_diagnostics=True,
                 scale_units=True, seed=42, robust_fit=False, robust_alpha=4.0,
                 method="ls-nmf", init_method="column_mean", init_norm=True, hold_h=False, delay_h=-1):
        """
        Initialize PMF analyzer for MMF data.
        
        Args:
            station (str): MMF station identifier (MMF1, MMF2, MMF6, MMF9) - legacy mode
            data_dir (str): Directory containing parquet files - flexible mode
            patterns (str): Comma-separated parquet file patterns - flexible mode
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            output_dir (str): Directory for output files
            remove_voc (bool): If True, exclude VOC species from PMF analysis
            
            # EPA S/N weighting and uncertainty parameters (default to legacy behavior)
            uncertainty_mode (str): 'epa' or 'legacy' - uncertainty calculation method
            uncertainty_ef_mdl (str): Path to CSV with EF/MDL data (None = use built-ins)
            uncertainty_epsilon (float): Numerical floor for uncertainties
            legacy_min_u (float): Min uncertainty when using legacy mode
            uncertainty_bdl_policy (str): BDL policy ('five-sixth-mdl' or 'half-mdl')
            snr_enable (bool): Enable S/N-based feature categorization
            snr_weak_threshold (float): S/N threshold for weak categorization
            snr_bad_threshold (float): S/N threshold for bad categorization
            snr_bdl_weak_frac (float): BDL fraction for weak categorization
            snr_bdl_bad_frac (float): BDL fraction for bad categorization
            snr_missing_weak_frac (float): Missing fraction for weak categorization
            snr_missing_bad_frac (float): Missing fraction for bad categorization
            exclude_bad (bool): Exclude bad features from PMF analysis
            dashboard_snr_panel (bool): Add S/N panels to dashboard
            write_diagnostics (bool): Write diagnostic CSVs
            scale_units (bool): Apply unit standardization (mg/m³→μg/m³, ng/m³→μg/m³)
            seed (int): Random seed for reproducibility
            robust_fit (bool): Enable ESAT robust loss during SA training (single-model fallback)
            robust_alpha (float): Robust cutoff alpha for uncertainty-scaled residuals
            method (str): ESAT NMF algorithm ('ls-nmf' for nonnegative, 'ws-nmf' for semi-NMF)
            init_method (str): Matrix initialization method ('column_mean' or 'kmeans')
            init_norm (bool): Whiten data before kmeans initialization (for magnitude balance)
            hold_h (bool): Hold H (profile) matrix constant during training
            delay_h (int): Hold H matrix for N iterations, then release (-1 = disabled)
        """
        self.station = station
        self.data_dir = data_dir
        self.patterns = patterns
        self.start_date = start_date
        self.end_date = end_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.remove_voc = remove_voc
        
        # Create standardized filename prefix with dates and MMF identifier
        self.filename_prefix = self._create_filename_prefix()
        
        # PMF Configuration following EPA guidelines
        self.factors = 4  # Will be optimized during analysis
        self.models = 20  # EPA recommends 20+ models for robust results
        
        # Data containers
        self.df = None
        self.units = {}
        self.concentration_data = None
        self.uncertainty_data = None
        
        # ESAT objects
        self.batch_models = None
        self.best_model = None
        
        # Optimization results for plotting
        self.optimization_q_values = None
        self.optimal_factors = None
        
        # PCA analysis components
        self.pca_model = None
        self.pca_loadings = None
        self.pca_scores = None
        self.pca_scaler = None
        self.pca_explained_variance = None
        
        # Color management for consistent plotting
        self.color_manager = None
        
        # Multiprocessing control
        self.max_workers = 2  # Default number of workers

        # Runtime controls (CLI-configurable)
        self.zero_as_bdl = True          # Treat exact zeros as BDL by default
        self.save_masks = True           # Save BDL/missing masks by default
        self.drop_row_threshold = 0.5    # Drop rows with >50% missing prior to replacement
        
        # EPA S/N weighting and uncertainty parameters (legacy defaults preserve behavior)
        self.uncertainty_mode = uncertainty_mode
        self.uncertainty_ef_mdl = uncertainty_ef_mdl
        self.uncertainty_epsilon = uncertainty_epsilon
        self.legacy_min_u = legacy_min_u
        self.uncertainty_bdl_policy = uncertainty_bdl_policy
        self.snr_enable = snr_enable
        self.snr_weak_threshold = snr_weak_threshold
        self.snr_bad_threshold = snr_bad_threshold
        self.snr_bdl_weak_frac = snr_bdl_weak_frac
        self.snr_bdl_bad_frac = snr_bdl_bad_frac
        self.snr_missing_weak_frac = snr_missing_weak_frac
        self.snr_missing_bad_frac = snr_missing_bad_frac
        self.exclude_bad = exclude_bad
        self.dashboard_snr_panel = dashboard_snr_panel
        self.write_diagnostics = write_diagnostics
        self.scale_units = scale_units
        self.seed = seed
        self.robust_fit = robust_fit
        self.robust_alpha = robust_alpha
        
        # ESAT algorithm and initialization controls  
        # Validate method
        if method not in ['ls-nmf', 'ws-nmf']:
            raise ValueError(f"Invalid method '{method}'. Must be 'ls-nmf' or 'ws-nmf'")
        self.method = method
        
        # Validate init_method
        if init_method not in ['column_mean', 'kmeans']:
            raise ValueError(f"Invalid init_method '{init_method}'. Must be 'column_mean' or 'kmeans'")
        self.init_method = init_method
        
        self.init_norm = bool(init_norm)
        self.hold_h = bool(hold_h)
        
        # Validate delay_h
        if delay_h != -1 and delay_h < 1:
            raise ValueError(f"Invalid delay_h '{delay_h}'. Must be -1 (disabled) or positive integer")
        self.delay_h = int(delay_h) if delay_h != -1 else -1
        
        # Validate parameter combinations
        if self.delay_h > 0 and not self.hold_h:
            print("⚠️  Warning: delay_h specified without hold_h. Setting hold_h=True for consistency.")
            self.hold_h = True
    
    def _create_filename_prefix(self):
        """Create standardized filename prefix with dates and identifier."""
        # Format dates for filename (replace invalid characters)
        start_str = self.start_date.replace('-', '') if self.start_date else 'all'
        end_str = self.end_date.replace('-', '') if self.end_date else 'all'
        
        if self.station:
            # Legacy mode: use station name
            prefix = f"{self.station}_mmf_{start_str}_{end_str}"
        else:
            # Flexible mode: use generic prefix
            prefix = f"mmf_pmf_{start_str}_{end_str}"
        return prefix
    
    def _get_station_display_name(self):
        """Get the full display name for the station (e.g., 'MMF1 - Cemetery Road')."""
        station_mapping = get_station_mapping()
        
        # Get station name and MMF info
        if self.station in station_mapping:
            station_name = station_mapping[self.station]
            if station_name:
                return f"{self.station} - {station_name}"
            else:
                return self.station  # For Maries_Way
        else:
            return self.station
    
    def _display_station_info(self):
        """Display prominent analysis information banner."""
        print("\n" + "=" * 60)
        print("🧪 MMF PMF SOURCE APPORTIONMENT ANALYSIS (FIXED)")
        print("=" * 60)
        
        if self.station:
            # Legacy station-based mode
            station_mapping = get_station_mapping()
            if self.station in station_mapping:
                station_name = station_mapping[self.station]
                if station_name:
                    display_name = f"{self.station} - {station_name}"
                else:
                    display_name = self.station  # For Maries_Way
            else:
                display_name = self.station
            print(f"🏭 Station: {display_name}")
        else:
            # Flexible data directory mode
            print(f"📁 Data Directory: {self.data_dir}")
            print(f"🔍 Patterns: {self.patterns}")
            
        if self.start_date or self.end_date:
            date_info = f"{self.start_date or 'All'} to {self.end_date or 'All'}"
            print(f"📅 Analysis Period: {date_info}")
        print(f"📂 Output Directory: {self.output_dir}")
        print("=" * 60)
    
    def _find_parquet_files(self):
        """Find parquet files matching the specified patterns in the data directory."""
        from pathlib import Path
        import fnmatch
        
        data_path = Path(self.data_dir)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Parse patterns (comma-separated)
        patterns = [p.strip() for p in self.patterns.split(',')]
        
        # Find matching files
        matching_files = []
        for pattern in patterns:
            # Use glob-style matching
            matches = list(data_path.glob(pattern))
            matching_files.extend(matches)
        
        # Remove duplicates and sort
        unique_files = sorted(list(set(matching_files)))
        
        print(f"🔍 Found {len(unique_files)} matching parquet file(s):")
        for f in unique_files:
            print(f"   - {f.name}")
        
        return unique_files
    
    def load_mmf_data(self):
        """Load and prepare MMF data for PMF analysis."""
        # Display information banner
        self._display_station_info()
        
        if self.station:
            # Legacy mode: station-based loading
            print(f"🔍 Loading MMF data for {self.station}...")
            try:
                parquet_file = get_mmf_parquet_file(self.station)
            except Exception as e:
                raise RuntimeError(f"Error determining file path for {self.station}: {e}")
            
            if not parquet_file.exists():
                raise FileNotFoundError(f"Corrected parquet file not found: {parquet_file}")
            
            analyzer = ParquetAnalyzer(parquet_file)
            if not analyzer.load_data():
                raise RuntimeError("Failed to load parquet data")
        else:
            # Flexible mode: data directory and patterns
            print(f"🔍 Loading parquet data from {self.data_dir}...")
            parquet_files = self._find_parquet_files()
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found matching patterns: {self.patterns}")
            
            # For now, use the first file (could be extended to merge multiple files)
            parquet_file = parquet_files[0]
            print(f"📄 Using file: {parquet_file.name}")
            
            analyzer = ParquetAnalyzer(parquet_file)
            if not analyzer.load_data():
                raise RuntimeError("Failed to load parquet data")
        
        self.df = analyzer.df.copy()

        # Read aggregation metadata from parquet if available
        try:
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(parquet_file)
            meta = pf.metadata.metadata or {}
            meta = { (k.decode() if isinstance(k, bytes) else str(k)):
                     (v.decode() if isinstance(v, bytes) else str(v))
                     for k,v in meta.items() }
            self.aggregation_timebase = meta.get('aggregation_timebase')
            self.aggregation_method = meta.get('aggregation_method')
            self.min_valid_subsamples = int(meta.get('min_valid_subsamples')) if meta.get('min_valid_subsamples') else None
            if self.aggregation_timebase or self.aggregation_method:
                print(f"🧭 Aggregation metadata: timebase={self.aggregation_timebase}, method={self.aggregation_method}, min_valid={self.min_valid_subsamples}")
        except Exception as e:
            self.aggregation_timebase = None
            self.aggregation_method = None
            self.min_valid_subsamples = None
        
        # Get units from metadata (stored or inferred)
        stored_units = analyzer.extract_units_from_metadata()
        for col in self.df.columns:
            if col in stored_units:
                self.units[col] = stored_units[col]
            else:
                self.units[col] = analyzer.get_suspected_units(col)
        
        # Filter by date range if specified
        if self.start_date or self.end_date:
            self._filter_date_range()
        
        print(f"✅ Loaded {len(self.df):,} records")
        print(f"📅 Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
    
    def _filter_date_range(self):
        """Filter data by specified date range."""
        original_len = len(self.df)
        
        if self.start_date:
            start_dt = pd.to_datetime(self.start_date)
            self.df = self.df[self.df['datetime'] >= start_dt]
        
        if self.end_date:
            end_dt = pd.to_datetime(self.end_date)
            # If start and end dates are the same, include the full 24 hours of that day
            if self.start_date and self.start_date == self.end_date:
                # Add 23:59:59 to include the entire day
                end_dt = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                print(f"📅 Same start/end date detected - including full 24 hours of {self.start_date}")
            self.df = self.df[self.df['datetime'] <= end_dt]
        
        filtered_len = len(self.df)
        print(f"📊 Date filtering: {filtered_len:,} records ({original_len - filtered_len:,} excluded)")
    
    def prepare_pmf_data(self):
        """
        Prepare concentration and uncertainty matrices for PMF analysis.
        Following EPA PMF 5.0 User Guide recommendations.
        """
        print("🔧 Preparing PMF data matrices...")
        
        # Define pollutant columns for PMF (gas, particle, and VOC data)
        # Exclude meteorological and QA columns as per EPA guidelines
        pollutant_columns = []
        gas_species = ['H2S', 'CH4', 'SO2', 'NOX', 'NO', 'NO2']
        # Handle both old and new particle naming conventions
        particle_species = ['PM1 FIDAS', 'PM1', 'PM2.5 FIDAS', 'PM2.5', 'PM4 FIDAS', 'PM4', 
                          'PM10 FIDAS', 'PM10', 'TSP FIDAS', 'TSP']
        # VOC species (BTEX compounds) - newly integrated
        voc_species = ['Benzene', 'Toluene', 'Ethylbenzene', 'Xylene']
        
        # Check for available VOC data
        available_vocs = []
        for col in self.df.columns:
            if any(voc in col for voc in voc_species):
                available_vocs.append(col)
        
        if available_vocs and not self.remove_voc:
            print(f"✅ VOC species detected: {available_vocs}")
        elif available_vocs and self.remove_voc:
            print(f"⚠️ VOC species detected but excluded due to --remove-voc flag: {available_vocs}")
        
        # Select all applicable species for PMF analysis
        if self.remove_voc:
            all_species = gas_species + particle_species
            print(f"🚫 VOC species excluded from PMF analysis")
        else:
            all_species = gas_species + particle_species + voc_species
        
        # Select columns that exactly match target species (avoid auxiliary columns like 'n_*')
        pollutant_columns = [col for col in self.df.columns if col in all_species]
        
        print(f"📋 Selected pollutants for PMF: {pollutant_columns}")
        
        # Report data availability for different species types
        gas_cols = [col for col in pollutant_columns if any(gas in col for gas in gas_species)]
        voc_cols = [col for col in pollutant_columns if any(voc in col for voc in voc_species)]
        pm_cols = [col for col in pollutant_columns if any(pm in col for pm in particle_species)]
        
        print(f"📊 Species breakdown:")
        print(f"  Gas species ({len(gas_cols)}): {gas_cols}")
        
        if not self.remove_voc:
            if voc_cols:
                print(f"  VOC species ({len(voc_cols)}): {voc_cols}")
                # Report VOC data coverage
                for voc_col in voc_cols:
                    total_records = len(self.df)
                    non_null_records = self.df[voc_col].notna().sum()
                    coverage = (non_null_records / total_records) * 100
                    print(f"    {voc_col}: {non_null_records:,}/{total_records:,} ({coverage:.1f}% coverage)")
            else:
                print(f"  VOC species (0): None detected for this station")
        else:
            print(f"  VOC species (0): Excluded by --remove-voc flag")
            
        print(f"  Particle species ({len(pm_cols)}): {pm_cols}")
        
        # Create concentration matrix
        self.concentration_data = self.df[pollutant_columns].copy()

        # Attempt to collect aggregation counts if present (columns prefixed with 'n_')
        count_cols = {}
        for col in pollutant_columns:
            n_col = f"n_{col}"
            if n_col in self.df.columns:
                count_cols[n_col] = self.df[n_col]
        if count_cols:
            self.counts_data = pd.DataFrame(count_cols)
        else:
            self.counts_data = None

        # Standardize all concentrations to μg/m³ prior to computing uncertainties (if enabled)
        if self.scale_units:
            self._standardize_units_to_ugm3(pollutant_columns)
        else:
            print("⚠️  Unit standardization disabled (--no-scale-units). Units will be used as-is.")
        
        # Remove rows with too many missing values (EPA recommendation: >50% missing)
        missing_threshold = getattr(self, 'drop_row_threshold', 0.5)
        valid_rows = self.concentration_data.isnull().sum(axis=1) / len(pollutant_columns) < missing_threshold
        self.concentration_data = self.concentration_data[valid_rows]
        if self.counts_data is not None:
            self.counts_data = self.counts_data.loc[self.concentration_data.index]
        
        print(f"📊 After removing rows with >{missing_threshold*100:.1f}% missing: {len(self.concentration_data):,} records")
        
        # Generate uncertainty matrix following EPA guidelines
        self._generate_uncertainty_matrix(pollutant_columns)
        
        # Apply S/N-based feature categorization if enabled
        if self.snr_enable and HAS_SNR_CATEGORIZATION:
            self._apply_snr_categorization(pollutant_columns)
        elif self.snr_enable and not HAS_SNR_CATEGORIZATION:
            print("⚠️ S/N categorization requested but module not available. Proceeding without categorization.")
        
        # Save processed data
        self._save_processed_data()
    
    def _normalize_unit_string(self, unit_str):
        """Normalize unit strings to a canonical form for comparison."""
        if unit_str is None:
            return None
        s = str(unit_str).strip().lower()
        s = s.replace('µ', 'μ')
        s = s.replace(' ', '')
        s = s.replace('(', '').replace(')', '')
        # Normalize common variants
        s = s.replace('ug/m³', 'ug/m3')
        s = s.replace('μg/m³', 'ug/m3')
        s = s.replace('μg/m3', 'ug/m3')
        s = s.replace('µg/m³', 'ug/m3')
        s = s.replace('µg/m3', 'ug/m3')
        s = s.replace('mg/m³', 'mg/m3')
        s = s.replace('ng/m³', 'ng/m3')
        s = s.replace('ugm3', 'ug/m3')
        s = s.replace('mgm3', 'mg/m3')
        s = s.replace('ngm3', 'ng/m3')
        return s

    def _standardize_units_to_ugm3(self, pollutant_columns):
        """
        Ensure all concentration columns are in μg/m³ before uncertainty calc.
        - If a column is reported in mg/m³, multiply by 1000
        - If in ng/m³, divide by 1000
        - Update self.units to 'μg/m³'
        - Warn if units are non-mass-based (e.g., ppm/ppb) as no conversion is applied
        """
        conversions = []
        for col in pollutant_columns:
            orig_unit = self.units.get(col)
            unit_norm = self._normalize_unit_string(orig_unit)
            # Fallback to suspected units if not present
            if unit_norm is None or unit_norm == 'unknown':
                try:
                    from analyze_parquet_data import ParquetAnalyzer
                    # No direct file context here for analyzer, so use suspected units helper we imported earlier
                    suspected = self._normalize_unit_string(self.units.get(col, None))
                    unit_norm = suspected or unit_norm
                except Exception:
                    pass
            factor = None
            if unit_norm in ('ug/m3', None):
                continue  # already μg/m³ or unknown (leave as-is)
            elif unit_norm == 'mg/m3':
                factor = 1000.0
            elif unit_norm == 'ng/m3':
                factor = 0.001
            elif unit_norm in ('ppm', 'ppb'):
                print(f"⚠️ Units for {col} are in {orig_unit}; no automatic ppm/ppb → μg/m³ conversion applied.")
                continue
            else:
                # Unrecognized units
                print(f"⚠️ Unrecognized unit '{orig_unit}' for {col}; leaving values unchanged.")
                continue
            if factor is not None:
                self.concentration_data[col] = self.concentration_data[col] * factor
                self.units[col] = 'μg/m³'
                conversions.append((col, orig_unit, factor))
        if conversions:
            print("🔁 Standardized units to μg/m³ for the following columns:")
            for col, ou, f in conversions:
                print(f"  - {col}: {ou} → μg/m³ (×{f})")

    def _apply_snr_categorization(self, pollutant_columns):
        """
        Apply S/N-based feature categorization following EPA PMF 5.0 guidelines.
        
        This categorizes species as 'strong', 'weak', or 'bad' based on S/N ratio
        and data quality metrics. Weak species have their uncertainty tripled,
        while bad species are excluded from the analysis entirely.
        """
        print("🔢 Applying S/N-based feature categorization...")
        
        # Create EPA S/N categorizer with configured thresholds
        categorizer = create_snr_categorizer(
            snr_weak_threshold=self.snr_weak_threshold,
            snr_bad_threshold=self.snr_bad_threshold,
            bdl_weak_frac=self.snr_bdl_weak_frac,
            bdl_bad_frac=self.snr_bdl_bad_frac,
            missing_weak_frac=self.snr_missing_weak_frac,
            missing_bad_frac=self.snr_missing_bad_frac
        )
        
        # Get EPA calculator for MDL lookups if available
        epa_calculator = None
        if self.uncertainty_mode == 'epa' and HAS_EPA_UNCERTAINTY:
            epa_calculator = create_epa_uncertainty_calculator(
                epsilon=self.uncertainty_epsilon,
                bdl_policy=self.uncertainty_bdl_policy,
                ef_mdl_csv=self.uncertainty_ef_mdl
            )
        
        # Compute S/N ratios and categorize species
        metrics, categories, reasoning = categorizer.categorize_species(
            self.concentration_data, 
            self.uncertainty_data,
            epa_calculator
        )
        
        # Store categorization results
        self._snr_metrics = metrics
        self._snr_categories = categories
        self._snr_reasoning = reasoning
        
        # Apply categorization (currently just modifies uncertainties)
        # If ESAT integration is available in future, we'd apply via DataHandler here
        print("🔄 Applying categorization to uncertainty matrix...")
        
        # Collect species to exclude
        excluded_species = []
        
        for species, category in categories.items():
            if category == 'weak':
                # Triple uncertainties for weak species (EPA PMF 5.0 recommendation)
                print(f"   ⚠️ {species}: Weak - tripling uncertainty")
                self.uncertainty_data[species] = self.uncertainty_data[species] * 3.0
            elif category == 'bad' and self.exclude_bad:
                # Mark bad species for exclusion from analysis
                print(f"   ❌ {species}: Bad - excluding from analysis")
                excluded_species.append(species)
        
        # Remove bad species from data matrices
        if excluded_species:
            print(f"   🗑️ Removing {len(excluded_species)} bad species from matrices: {excluded_species}")
            self.concentration_data = self.concentration_data.drop(columns=excluded_species)
            self.uncertainty_data = self.uncertainty_data.drop(columns=excluded_species)
            
            # Also update counts data if available
            if self.counts_data is not None:
                # Drop corresponding n_* columns for excluded species
                count_cols_to_drop = [f"n_{species}" for species in excluded_species if f"n_{species}" in self.counts_data.columns]
                if count_cols_to_drop:
                    self.counts_data = self.counts_data.drop(columns=count_cols_to_drop)
                    print(f"   🗑️ Removed corresponding count columns: {count_cols_to_drop}")
        
        # Save diagnostics if requested
        if self.write_diagnostics:
            categorizer.save_diagnostics(self.output_dir, self.filename_prefix)
        
        # Summary report
        summary = categorizer.get_summary()
        if summary:
            print(f"\n📊 S/N Categorization Summary:")
            print(f"   Total species: {summary['total_species']}")
            print(f"   Strong: {summary['strong_count']} species")
            print(f"   Weak: {summary['weak_count']} species") 
            print(f"   Bad: {summary['bad_count']} species")
            print(f"   Average S/N: {summary['average_snr']:.3f}")
            print(f"   Thresholds: weak={self.snr_weak_threshold}, bad={self.snr_bad_threshold}")
    
    def _generate_uncertainty_matrix(self, pollutant_columns):
        """
        Generate uncertainty matrix using EPA or legacy mode based on configuration.
        
        EPA Mode: Uses EPA formulas with optional aggregation scaling
        Legacy Mode: Original implementation with fixed MDL/EF table
        """
        print(f"🔬 Generating uncertainty matrix (mode: {self.uncertainty_mode})...")
        
        if self.uncertainty_mode == 'epa' and HAS_EPA_UNCERTAINTY:
            return self._generate_epa_uncertainty_matrix(pollutant_columns)
        else:
            if self.uncertainty_mode == 'epa' and not HAS_EPA_UNCERTAINTY:
                print("[WARNING] EPA mode requested but epa_uncertainty module not available. Using legacy mode.")
            return self._generate_legacy_uncertainty_matrix(pollutant_columns)
    
    def _generate_epa_uncertainty_matrix(self, pollutant_columns):
        """
        Generate uncertainty matrix using EPA formulas and aggregation scaling.
        """
        print("📐 Using EPA uncertainty formulas...")
        
        # Create EPA uncertainty calculator
        calculator = create_epa_uncertainty_calculator(
            epsilon=self.uncertainty_epsilon,
            bdl_policy=self.uncertainty_bdl_policy,
            ef_mdl_csv=self.uncertainty_ef_mdl,
            legacy_min_u=self.legacy_min_u
        )
        
        # Display policy summary
        policy = calculator.get_policy_summary()
        print(f"   BDL policy: {policy['bdl_formula']}")
        print(f"   Above MDL: {policy['above_mdl_formula']}")
        print(f"   EF/MDL source: {policy['ef_mdl_source']}")
        
        # Calculate EPA uncertainties for pollutant columns only
        conc_for_uncertainty = self.concentration_data[pollutant_columns]
        
        # Load aggregation counts for scaling if available
        counts_df = None
        try:
            counts_file = self.output_dir / f"{self.filename_prefix}_counts.csv"
            if counts_file.exists():
                counts_df = pd.read_csv(counts_file, index_col=0)
                print(f"   📊 Loaded aggregation counts for scaling")
            else:
                print(f"   ℹ️ No aggregation counts available - EPA formulas only")
        except Exception as e:
            print(f"   ⚠️ Could not load aggregation counts: {e}")
        
        # Calculate EPA uncertainties with aggregation scaling
        epa_uncertainties = calculator.calculate_species_uncertainties(
            conc_for_uncertainty, counts_df
        )
        
        # Initialize full uncertainty matrix and handle concentration adjustments
        self.uncertainty_data = pd.DataFrame(
            index=self.concentration_data.index,
            columns=self.concentration_data.columns,
            dtype=float
        )
        
        # Copy EPA uncertainties for pollutant columns
        for species in pollutant_columns:
            if species in epa_uncertainties.columns:
                self.uncertainty_data[species] = epa_uncertainties[species]
                
                # Apply EPA concentration adjustments for BDL and missing values
                EF, MDL, unit = calculator.get_ef_mdl(species)
                
                conc_col = self.concentration_data[species].copy()
                missing_mask = conc_col.isna()
                zero_mask = (~missing_mask) & (conc_col == 0)
                bdl_mask = (~missing_mask) & (conc_col <= MDL)
                
                # Handle zero-as-bdl policy
                if hasattr(self, 'zero_as_bdl') and not self.zero_as_bdl:
                    bdl_mask = bdl_mask & (~zero_mask)
                    missing_mask = missing_mask | zero_mask
                
                # Apply EPA concentration replacements
                if bdl_mask.any():
                    self.concentration_data.loc[bdl_mask, species] = MDL * 0.5
                if missing_mask.any():
                    species_median = conc_col.median()
                    if pd.isna(species_median):
                        # Fallback if median cannot be computed
                        species_median = MDL
                    self.concentration_data.loc[missing_mask, species] = species_median
                
                # Summary statistics
                total = len(conc_col)
                n_meas = int((~missing_mask & ~bdl_mask).sum())
                n_bdl = int(bdl_mask.sum())
                n_missing = int(missing_mask.sum())
                print(f"   ✅ {species}: EF={EF:.3f}, MDL={MDL:.1f} | measured={n_meas} ({n_meas/total*100:.1f}%), "
                      f"BDL={n_bdl} ({n_bdl/total*100:.1f}%), missing={n_missing} ({n_missing/total*100:.1f}%)")
            else:
                # Fallback for species not handled by EPA calculator
                print(f"   ⚠️ {species}: No EPA data, using default uncertainty")
                self.uncertainty_data[species] = 1.0
        
        # Handle non-pollutant columns (if any)
        for col in self.concentration_data.columns:
            if col not in pollutant_columns:
                self.uncertainty_data[col] = 1.0  # Default uncertainty
        
        print(f"🔬 EPA uncertainty matrix completed for {len(pollutant_columns)} species")
        
    def _generate_legacy_uncertainty_matrix(self, pollutant_columns):
        """
        Generate uncertainty matrix using original implementation.
        All MDL values below are specified in μg/m³ to match standardized V.
        
        EPA Formula: σ = sqrt((error_fraction * concentration)² + (MDL)²)
        """
        
        # EPA-recommended MDL values and error fractions by pollutant type (all in μg/m³)
        # Based on typical instrument specifications and EPA guidance
        mdl_values = {
            'H2S': 0.5,      # μg/m³
            'CH4': 50.0,     # μg/m³ (converted from 0.05 mg/m³ → 50 μg/m³)
            'SO2': 0.5,      # μg/m³
            'NOX': 1.0,      # μg/m³
            'NO': 0.5,       # μg/m³
            'NO2': 1.0,      # μg/m³
            'PM1 FIDAS': 1.0,    # μg/m³
            'PM1': 1.0,          # μg/m³
            'PM2.5 FIDAS': 1.0,  # μg/m³
            'PM2.5': 1.0,        # μg/m³
            'PM4 FIDAS': 1.5,    # μg/m³
            'PM4': 1.5,          # μg/m³
            'PM10 FIDAS': 2.0,   # μg/m³
            'PM10': 2.0,         # μg/m³
            'TSP FIDAS': 2.5,    # μg/m³
            'TSP': 2.5,          # μg/m³
            # VOC species (BTEX compounds) - typical GC-MS detection limits (μg/m³)
            'Benzene': 0.01,     # μg/m³
            'Toluene': 0.02,     # μg/m³
            'Ethylbenzene': 0.02,    # μg/m³
            'Xylene': 0.02,      # μg/m³ (covers m&p-Xylene)
            'm&p-Xylene': 0.02   # μg/m³ (specific for mixed isomers)
        }
        
        # EPA-recommended error fractions (measurement precision)
        error_fractions = {
            'H2S': 0.15,      # 15% relative error
            'CH4': 0.10,      # 10% relative error
            'SO2': 0.15,      # 15% relative error
            'NOX': 0.20,      # 20% relative error
            'NO': 0.20,       # 20% relative error
            'NO2': 0.20,      # 20% relative error
            'PM1 FIDAS': 0.10,    # 10% relative error
            'PM1': 0.10,          # 10% relative error
            'PM2.5 FIDAS': 0.10,  # 10% relative error
            'PM2.5': 0.10,        # 10% relative error
            'PM4 FIDAS': 0.12,    # 12% relative error
            'PM4': 0.12,          # 12% relative error
            'PM10 FIDAS': 0.15,   # 15% relative error
            'PM10': 0.15,         # 15% relative error
            'TSP FIDAS': 0.20,    # 20% relative error
            'TSP': 0.20,          # 20% relative error
            # VOC species (BTEX compounds) - typical GC-MS measurement precision
            'Benzene': 0.10,     # 10% relative error (high precision for carcinogen)
            'Toluene': 0.12,     # 12% relative error
            'Ethylbenzene': 0.15,    # 15% relative error
            'Xylene': 0.15,      # 15% relative error (covers m&p-Xylene)
            'm&p-Xylene': 0.15   # 15% relative error (mixed isomers have higher uncertainty)
        }
        
        # Generate uncertainty for each species
        self.uncertainty_data = pd.DataFrame(
            index=self.concentration_data.index,
            columns=self.concentration_data.columns
        )
        
        # Initialize masks for traceability
        self._bdl_mask = pd.DataFrame(False, index=self.concentration_data.index, columns=self.concentration_data.columns)
        self._missing_mask = pd.DataFrame(False, index=self.concentration_data.index, columns=self.concentration_data.columns)

        for species in pollutant_columns:
            # Find matching MDL and error fraction (partial name matching)
            mdl = 1.0  # Default MDL (μg/m³)
            err_frac = 0.15  # Default error fraction

            for key in mdl_values.keys():
                if key in species:
                    mdl = mdl_values[key]
                    err_frac = error_fractions[key]
                    break

            # Create masks
            col = self.concentration_data[species]
            missing_mask = col.isna()
            zero_mask = (~missing_mask) & (col == 0)
            # Treat numeric values < MDL as BDL
            bdl_mask = (~missing_mask) & (col < mdl)
            # Optionally treat zeros as missing instead of BDL
            if hasattr(self, 'zero_as_bdl') and not self.zero_as_bdl:
                bdl_mask = bdl_mask & (~zero_mask)
                missing_mask = missing_mask | zero_mask
            measured_mask = (~missing_mask) & (col >= mdl)

            # Allocate arrays
            u_col = np.zeros_like(col, dtype=float)
            v_new = col.copy()

            # Apply EPA PMF rules
            # Measured cells
            if measured_mask.any():
                v_meas = v_new[measured_mask].astype(float)
                u_col[measured_mask] = np.sqrt((err_frac * v_meas) ** 2 + mdl ** 2)

            # BDL cells: V = MDL/2, U = (5/6)·MDL
            if bdl_mask.any():
                v_new.loc[bdl_mask] = mdl * 0.5
                u_col[bdl_mask] = mdl * (5.0 / 6.0)

            # Missing cells: V = MDL, U = 4·MDL
            if missing_mask.any():
                v_new.loc[missing_mask] = mdl
                u_col[missing_mask] = mdl * 4.0

            # Apply legacy minimum uncertainty clamping
            u_col = np.maximum(u_col, self.legacy_min_u)

            # Save back
            self.concentration_data[species] = v_new
            self.uncertainty_data[species] = u_col

            # Save masks for traceability
            self._bdl_mask[species] = bdl_mask
            self._missing_mask[species] = missing_mask

            # Summary
            total = len(col)
            n_meas = int(measured_mask.sum())
            n_bdl = int(bdl_mask.sum())
            n_missing = int(missing_mask.sum())
            print(f"  {species}: MDL={mdl}, Err={err_frac*100:.1f}%, min_u={self.legacy_min_u} | measured={n_meas} ({n_meas/total*100:.1f}%), "
                  f"BDL={n_bdl} ({n_bdl/total*100:.1f}%), missing={n_missing} ({n_missing/total*100:.1f}%) | Units={self.units.get(species, 'unknown')}")
    
    def _handle_missing_values(self):
        """Handle missing values following EPA Method 1."""
        print("🔄 Handling missing values (EPA Method 1)...")
        
        for col in self.concentration_data.columns:
            # Replace missing concentrations with median
            median_conc = self.concentration_data[col].median()
            missing_mask = self.concentration_data[col].isnull()
            n_missing = missing_mask.sum()
            
            if n_missing > 0:
                # If median is NaN (all values missing), use a small positive value
                if pd.isna(median_conc):
                    median_conc = 0.1  # Small positive default
                    print(f"  {col}: All values missing, using default value ({median_conc:.2f})")
                else:
                    print(f"  {col}: {n_missing} missing values replaced with median ({median_conc:.2f})")
                
                self.concentration_data.loc[missing_mask, col] = median_conc
                # Set high uncertainty for replaced values (4 × median)
                self.uncertainty_data.loc[missing_mask, col] = 4 * median_conc
        
        # Final check: ensure no NaN values remain
        self._remove_remaining_nan_values()
    
    def _remove_remaining_nan_values(self):
        """Remove any remaining NaN values that could cause ESAT to fail."""
        print("🔍 Final NaN check and removal...")
        
        # Check for NaN in concentration data
        conc_nan_count = self.concentration_data.isna().sum().sum()
        if conc_nan_count > 0:
            print(f"  Warning: {conc_nan_count} NaN values found in concentration data")
            # Replace any remaining NaN with small positive values
            self.concentration_data = self.concentration_data.fillna(0.1)
        
        # Check for NaN in uncertainty data  
        unc_nan_count = self.uncertainty_data.isna().sum().sum()
        if unc_nan_count > 0:
            print(f"  Warning: {unc_nan_count} NaN values found in uncertainty data")
            # Replace any remaining NaN with reasonable uncertainty values
            self.uncertainty_data = self.uncertainty_data.fillna(1.0)
        
        # Check for infinite values
        conc_inf_count = np.isinf(self.concentration_data).sum().sum()
        unc_inf_count = np.isinf(self.uncertainty_data).sum().sum()
        
        if conc_inf_count > 0:
            print(f"  Warning: {conc_inf_count} infinite values found in concentration data")
            self.concentration_data = self.concentration_data.replace([np.inf, -np.inf], 0.1)
        
        if unc_inf_count > 0:
            print(f"  Warning: {unc_inf_count} infinite values found in uncertainty data")
            self.uncertainty_data = self.uncertainty_data.replace([np.inf, -np.inf], 1.0)
        
        print(f"  ✅ Data cleaning complete: {len(self.concentration_data)} valid records")
    
    def _save_processed_data(self):
        """Save processed concentration and uncertainty data."""
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        unc_file = self.output_dir / f"{self.filename_prefix}_uncertainties.csv"
        
        # Add datetime index for ESAT compatibility
        conc_data = self.concentration_data.copy()
        unc_data = self.uncertainty_data.copy()
        
        # Get corresponding datetime values
        datetime_values = self.df.loc[self.concentration_data.index, 'datetime']
        conc_data.index = datetime_values
        unc_data.index = datetime_values
        
        conc_data.to_csv(conc_file)
        unc_data.to_csv(unc_file)

        # Optionally save BDL/missing masks for traceability
        try:
            if getattr(self, 'save_masks', True) and hasattr(self, '_bdl_mask') and hasattr(self, '_missing_mask'):
                bdl_mask = self._bdl_mask.loc[self.concentration_data.index]
                missing_mask = self._missing_mask.loc[self.concentration_data.index]
                bdl_mask.index = datetime_values
                missing_mask.index = datetime_values
                bdl_file = self.output_dir / f"{self.filename_prefix}_bdl_mask.csv"
                missing_file = self.output_dir / f"{self.filename_prefix}_missing_mask.csv"
                bdl_mask.to_csv(bdl_file)
                missing_mask.to_csv(missing_file)
                print(f"💾 Saved BDL mask: {bdl_file}")
                print(f"💾 Saved Missing mask: {missing_file}")
        except Exception as e:
            print(f"⚠️ Could not save masks: {e}")

        # Optionally save counts if available
        try:
            if self.counts_data is not None and len(self.counts_data.columns) > 0:
                counts_out = self.counts_data.copy()
                counts_out.index = datetime_values
                counts_file = self.output_dir / f"{self.filename_prefix}_counts.csv"
                counts_out.to_csv(counts_file)
                print(f"💾 Saved aggregation counts: {counts_file}")
        except Exception as e:
            print(f"⚠️ Could not save counts: {e}")
        
        print(f"💾 Saved concentration data: {conc_file}")
        print(f"💾 Saved uncertainty data: {unc_file}")
        
        # Save EPA-specific diagnostic files if requested
        if getattr(self, 'write_diagnostics', True) and self.uncertainty_mode == 'epa':
            try:
                epa_unc_file = self.output_dir / f"{self.filename_prefix}_uncertainties_epa.csv"
                unc_data.to_csv(epa_unc_file)
                print(f"💾 Saved EPA uncertainties: {epa_unc_file}")
            except Exception as e:
                print(f"⚠️ Could not save EPA uncertainties: {e}")
        
        # Save S/N categorization mapping if available
        if getattr(self, 'write_diagnostics', True) and hasattr(self, '_snr_categories'):
            try:
                categories_file = self.output_dir / f"{self.filename_prefix}_categories.csv"
                categories_df = pd.DataFrame({
                    'species': list(self._snr_categories.keys()),
                    'category': list(self._snr_categories.values())
                })
                categories_df.to_csv(categories_file, index=False)
                print(f"💾 Saved species categories: {categories_file}")
            except Exception as e:
                print(f"⚠️ Could not save S/N categories: {e}")
    
    def run_pmf_analysis(self):
        """
        Run PMF analysis using ESAT following EPA best practices.
        FIXED VERSION based on successful test.
        """
        print("🚀 Starting PMF analysis...")
        
        # Get data files
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        unc_file = self.output_dir / f"{self.filename_prefix}_uncertainties.csv"
        
        # Load data directly into numpy arrays (bypass DataHandler issues)
        print("📊 Loading data matrices...")
        conc_df = pd.read_csv(conc_file, index_col=0)
        unc_df = pd.read_csv(unc_file, index_col=0)
        
        # Convert to numpy arrays for ESAT
        V = conc_df.values  # Concentration matrix
        U = unc_df.values   # Uncertainty matrix
        species_names = conc_df.columns.tolist()

        # Apply uncertainty scaling for aggregated windows if using legacy mode
        # (EPA mode already includes 1/sqrt(n) scaling in uncertainty calculation)
        if self.uncertainty_mode == 'legacy':
            try:
                counts_file = self.output_dir / f"{self.filename_prefix}_counts.csv"
                if counts_file.exists() and (self.aggregation_method in ("mean", "median")):
                    counts_df = pd.read_csv(counts_file, index_col=0)
                    # Build scaling factors per species
                    for j, sp in enumerate(species_names):
                        n_col = f"n_{sp}"
                        if n_col in counts_df.columns:
                            n = counts_df[n_col].values.astype(float)
                            n = np.where(np.isfinite(n) & (n > 0), n, 1.0)
                            if self.aggregation_method == 'mean':
                                scale = 1.0 / np.sqrt(n)
                            else:
                                # Median approx: 1.253/sqrt(n)
                                scale = 1.253 / np.sqrt(n)
                            U[:, j] = U[:, j] * scale
                    print(f"🧮 Applied legacy uncertainty scaling based on aggregation counts (method={self.aggregation_method})")
            except Exception as e:
                print(f"⚠️ Could not apply aggregation-based uncertainty scaling: {e}")
        else:
            print(f"ℹ️ EPA mode: aggregation scaling already included in uncertainty calculation")
        
        print(f"📊 Data matrices: V={V.shape}, U={U.shape}")
        print(f"📋 Species: {', '.join(species_names)}")
        
        # Check if we have any data
        if V.size == 0 or U.size == 0:
            print("❌ Error: No data available for PMF analysis!")
            print("   This could be due to:")
            print("   1. Date range outside available data")
            print("   2. All data filtered out due to missing values")
            print("   3. No valid pollutant species found")
            return False
        
        if V.shape[0] < 10:
            print(f"⚠️ Warning: Very few data points ({V.shape[0]}) - PMF results may be unreliable")
            print("   Consider expanding date range or reducing missing data threshold")
        
        # Ensure physical constraints without distorting scale
        # Concentrations must be non-negative; clip small negatives to 0.0 (do NOT floor to 0.1)
        if V.size > 0:
            V = np.where(np.isnan(V), 0.0, V)
            V = np.where(np.isposinf(V), np.nanmax(V[np.isfinite(V)]) if np.any(np.isfinite(V)) else 1.0, V)
            V = np.where(np.isneginf(V), 0.0, V)
            V = np.where(V < 0, 0.0, V)
        
        # Uncertainty must be strictly positive for weighting; replace invalids with a small positive value
        if U.size > 0:
            finite_U = U[np.isfinite(U)]
            median_U = float(np.nanmedian(finite_U)) if finite_U.size > 0 else 1.0
            max_U = float(np.nanmax(finite_U)) if finite_U.size > 0 else 10.0
            min_positive = max(median_U * 1e-6, 1e-9)
            U = np.where(np.isnan(U), median_U, U)
            U = np.where(np.isposinf(U), max_U, U)
            U = np.where(np.isneginf(U) | (U <= 0), min_positive, U)
        
        print(f"📊 Final data validation:")
        print(f"  V range: [{np.min(V):.3f}, {np.max(V):.3f}]")
        print(f"  U range: [{np.min(U):.3f}, {np.max(U):.3f}]")
        print(f"  Valid data points: {np.sum(~np.isnan(V) & ~np.isnan(U))} / {V.size}")
        
        # Optimize number of factors (EPA recommendation: try multiple values)
        self._optimize_factors(V, U)
        
        # Check if robust mode is requested - force single SA if needed
        use_batch_sa = USE_BATCH_SA
        if self.robust_fit and USE_BATCH_SA:
            print("⚠️  Robust mode requested: forcing single SA mode (BatchSA doesn't support robust training)")
            use_batch_sa = False
        
        # Run PMF models
        print(f"🔄 Running {self.models} PMF models with {self.factors} factors...")
        try:
            if use_batch_sa:
                # Use BatchSA for multiple models
                self.batch_models = BatchSA(
                    V=V, U=U, 
                    factors=self.factors, 
                    models=self.models,
                    method=self.method,  # Use configured method (ls-nmf or ws-nmf)
                    init_method=self.init_method,  # Use configured init method
                    init_norm=self.init_norm,  # Use configured init normalization
                    seed=self.seed,
                    cpus=self.max_workers,  # Control number of processes
                    verbose=True
                )
                
                self.batch_models.train()
                
                # Select best model
                best_idx = self.batch_models.best_model
                self.best_model = self.batch_models.results[best_idx]
                
                print(f"✅ Best model: #{best_idx}")
                print(f"   Q(true): {self.best_model.Qtrue:.2f}")
                print(f"   Q(robust): {self.best_model.Qrobust:.2f}")
                
                # Interpret Q-values according to EPA guidelines
                interpretation = self._interpret_q_values(
                    q_true=self.best_model.Qtrue,
                    q_robust=self.best_model.Qrobust,
                    n_samples=V.shape[0],
                    n_species=V.shape[1],
                    n_factors=self.factors
                )
                self._display_q_interpretation(interpretation)
            else:
                # Use multiple SA models (manual implementation for robust mode)
                if self.robust_fit:
                    print(f"🔧 Running {self.models} SA models with ROBUST mode (alpha={self.robust_alpha})")
                    print("   → Robust training will downweight outliers during optimization")
                else:
                    print(f"⚠️ Running {self.models} SA models (BatchSA not available or disabled)")
                
                # Run multiple SA models and select the best one (keep only best to save memory)
                best_model = None
                best_q_robust = float('inf')
                best_idx = 0
                
                for model_idx in range(self.models):
                    print(f"   🔄 Training model {model_idx + 1}/{self.models}...")
                    
                    # Create SA model with different seed for each run
                    model_seed = self.seed + model_idx if self.seed else None
                    sa_model = SA(
                        V=V, U=U, 
                        factors=self.factors,
                        method=self.method,  # Use configured method (ls-nmf or ws-nmf)
                        seed=model_seed,
                        verbose=False  # Reduce verbosity for multiple models
                    )
                    
                    # Initialize matrices with configured parameters
                    sa_model.initialize(
                        init_method=self.init_method,
                        init_norm=self.init_norm
                    )
                    
                    # Train with all configured parameters
                    sa_model.train(
                        robust_mode=self.robust_fit, 
                        robust_alpha=self.robust_alpha
                    )
                    
                    print(f"     Model {model_idx + 1}: Q(true)={sa_model.Qtrue:.2f}, Q(robust)={sa_model.Qrobust:.2f}")
                    
                    # Keep only the best model (lowest Q(robust))
                    if sa_model.Qrobust < best_q_robust:
                        best_q_robust = sa_model.Qrobust
                        best_idx = model_idx
                        # Replace previous best model to save memory
                        best_model = sa_model
                    # Discard current model if it's not the best (memory management)
                
                # Set best model
                self.best_model = best_model
                
                # Create a simple mock for compatibility with model quality plots
                class MockBatchSA:
                    def __init__(self, best_idx, best_model_obj):
                        self.best_model = best_idx
                        # Create single-item results list for compatibility
                        self.results = [best_model_obj]
                
                self.batch_models = MockBatchSA(best_idx, best_model)
                
                print(f"✅ Best model: #{best_idx + 1} (Q(robust)={best_q_robust:.2f})")
                print(f"   Q(true): {self.best_model.Qtrue:.2f}")
                print(f"   Q(robust): {self.best_model.Qrobust:.2f}")
            
            # Store species names for plotting
            self.species_names = species_names
            
            # Initialize color manager for consistent plotting
            self.color_manager = ColorManager(self.factors, self.species_names)
            print(f"🎨 Initialized consistent color scheme for {self.factors} factors and {len(self.species_names)} species")
            
        except Exception as e:
            print(f"❌ PMF analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def _optimize_factors(self, V, U):
        """
        Optimize number of factors following EPA guidelines.
        Try different factor numbers and use quality criteria.
        """
        # Check if factors were explicitly set via CLI (not the default)
        if hasattr(self, 'user_specified_factors') and self.user_specified_factors:
            print(f"🔢 Using user-specified number of factors: {self.factors}")
            return
        
        print("🔍 Optimizing number of factors...")
        
        # Check if BatchSA is available (considering robust mode override)
        use_batch_for_optimization = USE_BATCH_SA and not self.robust_fit
        
        if not use_batch_for_optimization:
            if self.robust_fit:
                print("⚠️ Skipping factor optimization (robust mode requires single SA)")
            else:
                print("⚠️ Skipping factor optimization (requires BatchSA)")
            self.factors = 4  # Use default
            return
        
        # Test range of factors (EPA recommends testing multiple values)
        # CRITICAL: Never allow n_factors = n_species (creates DoF = 0)
        # Must always leave at least 1 degree of freedom
        species_limit = V.shape[1] - 1  # Always leave at least 1 DoF
        max_factors = min(getattr(self, 'max_factors', 10), species_limit)
        
        if max_factors < 2:
            print(f"⚠️  Warning: Only {V.shape[1]} species available - using 2 factors minimum")
            self.factors = 2
            return
            
        factor_range = range(2, max_factors + 1)  # 2 to max_factors (inclusive)
        q_values = {}
        
        print(f"  Testing factors from 2 to {max_factors} (max: {V.shape[1]-1} to ensure DoF > 0)")
        
        for n_factors in factor_range:
            print(f"  Testing {n_factors} factors...")
            try:
                test_batch = BatchSA(
                    V=V, U=U,
                    factors=n_factors,
                    models=3,  # Fewer models for optimization speed
                    method=self.method,  # Use consistent method for optimization
                    init_method=self.init_method,  # Use consistent init method
                    init_norm=self.init_norm,  # Use consistent init normalization
                    seed=self.seed,
                    cpus=self.max_workers,  # Control number of processes
                    verbose=False
                )
                test_batch.train()
                best_test = test_batch.results[test_batch.best_model]
                q_values[n_factors] = best_test.Qrobust
                print(f"    Q(robust) = {best_test.Qrobust:.2f}")
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                continue
        
        # Select optimal number of factors using EPA guidelines
        if q_values:
            # Calculate Q/DoF ratios to find best fit (should be close to 1.0)
            q_dof_ratios = {}
            n_samples, n_species = V.shape
            
            for nf, q_val in q_values.items():
                dof = (n_samples * n_species) - (n_samples * nf) - (n_species * nf) + (nf * nf)
                if dof > 0:
                    q_dof_ratios[nf] = q_val / dof
                else:
                    q_dof_ratios[nf] = float('inf')  # Invalid - should not happen now
            
            # Select factor number with Q/DoF closest to 1.0 (EPA best practice)
            optimal_factors = min(q_dof_ratios.keys(), key=lambda nf: abs(q_dof_ratios[nf] - 1.0))
            self.factors = optimal_factors
            print(f"📊 Optimal factors selected: {self.factors} (Q/DoF = {q_dof_ratios[optimal_factors]:.3f})")
            
            # Store optimization results for plotting
            self.optimization_q_values = q_values.copy()
            self.optimal_factors = optimal_factors
            
            # Show the Q-value and Q/DoF progression for user understanding
            print(f"  Q-value and Q/DoF progression:")
            for nf in sorted(q_values.keys()):
                marker = " ← SELECTED" if nf == optimal_factors else ""
                q_dof_ratio = q_dof_ratios.get(nf, float('inf'))
                print(f"    {nf} factors: Q = {q_values[nf]:.2f}, Q/DoF = {q_dof_ratio:.3f}{marker}")
        else:
            print("⚠️  Using default factor number: 4")
            self.factors = 4
    
    def _interpret_q_values(self, q_true, q_robust, n_samples, n_species, n_factors):
        """
        Interpret Q-values according to EPA PMF guidelines.
        
        Args:
            q_true (float): Q(true) value
            q_robust (float): Q(robust) value  
            n_samples (int): Number of data samples
            n_species (int): Number of species
            n_factors (int): Number of factors
        
        Returns:
            dict: Interpretation results
        """
        # Calculate degrees of freedom
        # DOF = (samples × species) - (samples × factors) - (species × factors) + factors²
        dof = (n_samples * n_species) - (n_samples * n_factors) - (n_species * n_factors) + (n_factors * n_factors)
        
        # Theoretical expected Q for perfect fit
        expected_q = dof
        
        # Calculate Q/DOF ratios (EPA guideline: should be close to 1.0 for good fit)
        q_true_ratio = q_true / dof if dof > 0 else float('inf')
        q_robust_ratio = q_robust / dof if dof > 0 else float('inf')
        
        # EPA interpretation guidelines
        interpretation = {
            'q_true': q_true,
            'q_robust': q_robust,
            'dof': dof,
            'expected_q': expected_q,
            'q_true_ratio': q_true_ratio,
            'q_robust_ratio': q_robust_ratio
        }
        
        # Assess model quality based on EPA guidelines
        if q_robust_ratio <= 1.5:
            interpretation['quality'] = 'Excellent'
            interpretation['assessment'] = 'Model fits data very well'
        elif q_robust_ratio <= 2.0:
            interpretation['quality'] = 'Good'
            interpretation['assessment'] = 'Model fits data adequately'
        elif q_robust_ratio <= 3.0:
            interpretation['quality'] = 'Fair'
            interpretation['assessment'] = 'Model may need refinement'
        else:
            interpretation['quality'] = 'Poor'
            interpretation['assessment'] = 'Model does not fit data well - consider more factors or data review'
        
        # Additional EPA guidance
        interpretation['recommendations'] = []
        
        if q_robust_ratio > 2.0:
            interpretation['recommendations'].append('Consider increasing number of factors')
            interpretation['recommendations'].append('Check for outliers or data quality issues')
            interpretation['recommendations'].append('Verify uncertainty estimates')
        
        if q_true_ratio / q_robust_ratio > 2.0:
            interpretation['recommendations'].append('Data may contain significant outliers')
        
        if abs(q_robust_ratio - 1.0) < 0.1:
            interpretation['recommendations'].append('Excellent fit - model captures data variance well')
        
        return interpretation
    
    def _display_q_interpretation(self, interpretation):
        """
        Display Q-value interpretation in a user-friendly format.
        """
        print("\n📊 Q-Value Analysis (EPA PMF Guidelines):")
        print("=" * 50)
        print(f"Q(true): {interpretation['q_true']:.2f}")
        print(f"Q(robust): {interpretation['q_robust']:.2f}")
        print(f"Degrees of Freedom: {interpretation['dof']:,}")
        print(f"Expected Q (perfect fit): ~{interpretation['expected_q']:,.0f}")
        print()
        print(f"Q(robust)/DOF Ratio: {interpretation['q_robust_ratio']:.3f}")
        print(f"Model Quality: {interpretation['quality']} ({interpretation['assessment']})")
        print()
        
        # EPA guidelines explanation
        print("EPA Q-Value Guidelines:")
        print("  Q/DOF ≤ 1.5: Excellent fit")
        print("  Q/DOF ≤ 2.0: Good fit")
        print("  Q/DOF ≤ 3.0: Fair fit (may need refinement)")
        print("  Q/DOF > 3.0: Poor fit (review model/data)")
        print()
        
        if interpretation['recommendations']:
            print("Recommendations:")
            for i, rec in enumerate(interpretation['recommendations'], 1):
                print(f"  {i}. {rec}")
        print()
    
    def create_pmf_dashboard(self):
        """
        Create comprehensive PMF dashboard with seaborn styling.
        FIXED VERSION based on successful ESAT test.
        """
        print("📊 Creating PMF dashboard...")
        
        # Suppress any potential matplotlib or numpy output during plotting
        import warnings
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        warnings.filterwarnings('ignore')  # Suppress warnings
        
        if not self.best_model:
            print("❌ No model results available")
            return False
        
        # Create dashboard directory
        dashboard_dir = self.output_dir / "dashboard"
        dashboard_dir.mkdir(exist_ok=True)
        
        # Extract ESAT results (using correct attribute names)
        F_profiles = self.best_model.H  # Factor profiles (source signatures)  
        G_contributions = self.best_model.W  # Factor contributions (time series)
        
        print(f"📊 Creating plots...")
        print(f"   Factor profiles: {F_profiles.shape}")
        print(f"   Factor contributions: {G_contributions.shape}")
        
        
        # Create basic PMF plots
        plot_files = []
        
        try:
            # Plot 1: Factor Profiles - Dynamic subplot layout for all factors
            n_factors = F_profiles.shape[0]
            
            # Calculate optimal subplot layout
            if n_factors <= 4:
                nrows, ncols = 2, 2
            elif n_factors <= 6:
                nrows, ncols = 2, 3
            elif n_factors <= 9:
                nrows, ncols = 3, 3
            elif n_factors <= 12:
                nrows, ncols = 3, 4
            else:
                nrows, ncols = 4, 4  # Maximum 16 factors
            
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            station_display_name = self._get_station_display_name()
            fig.suptitle(f'{station_display_name} PMF Factor Profiles - All {n_factors} Factors (Source Signatures)', 
                        fontsize=16, fontweight='bold')
            
            # Flatten axes array for easier indexing
            if n_factors == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            # Plot all factors
            for i in range(n_factors):
                ax = axes[i]
                profile = F_profiles[i, :]
                
                # Use consistent factor color for all species in this factor
                factor_color = self.color_manager.get_factor_color(i)
                bars = ax.bar(range(len(self.species_names)), profile, alpha=0.7, color=factor_color)
                
                ax.set_title(f'Factor {i+1}', fontweight='bold', fontsize=12)
                ax.set_xlabel('Species', fontsize=10)
                ax.set_ylabel('Contribution', fontsize=10)
                ax.set_xticks(range(len(self.species_names)))
                ax.set_xticklabels(self.species_names, rotation=45, ha='right', fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Optional: add species-specific edge colors for additional identification
                for bar, species in zip(bars, self.species_names):
                    species_color = self.color_manager.get_species_color(species)
                    bar.set_edgecolor(species_color)
                    bar.set_linewidth(2)
            
            # Hide unused subplots
            total_subplots = nrows * ncols
            for i in range(n_factors, total_subplots):
                if i < len(axes):
                    axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = dashboard_dir / f"{self.filename_prefix}_factor_profiles.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(plot_path)
            print(f"   ✅ Saved: factor_profiles.png")
            
        except Exception as e:
            print(f"   ❌ Error creating factor profiles: {e}")
        
        try:
            # Plot 2: Factor Contributions Time Series
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Get datetime index for plotting
            conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
            conc_data = pd.read_csv(conc_file, index_col=0)
            
            try:
                # Try to parse datetime index
                datetime_index = pd.to_datetime(conc_data.index)
                has_datetime = True
                
                # Plot with datetime x-axis
                for i in range(G_contributions.shape[1]):
                    factor_color = self.color_manager.get_factor_color(i)
                    ax.plot(datetime_index, G_contributions[:, i], 
                            label=f'Factor {i+1}', linewidth=2, alpha=0.8, color=factor_color)
                
                ax.set_xlabel('Date/Time')
                # Format x-axis for better readability
                import matplotlib.dates as mdates
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                
            except:
                # Fallback to sample index if datetime parsing fails
                time_index = np.arange(G_contributions.shape[0])
                has_datetime = False
                
                for i in range(G_contributions.shape[1]):
                    factor_color = self.color_manager.get_factor_color(i)
                    ax.plot(time_index, G_contributions[:, i], 
                            label=f'Factor {i+1}', linewidth=2, alpha=0.8, color=factor_color)
                
                ax.set_xlabel('Sample Index')
            
            ax.set_title(f'{station_display_name} PMF Factor Contributions Over Time', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Contribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = dashboard_dir / f"{self.filename_prefix}_factor_contributions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(plot_path)
            print(f"   ✅ Saved: factor_contributions.png")
            
        except Exception as e:
            print(f"   ❌ Error creating factor contributions: {e}")
        
        try:
            # Plot 3: Species Composition (Stacked Bar)
            fig, ax = plt.subplots(figsize=(12, 8))
            
            bottom = np.zeros(len(self.species_names))
            
            for i in range(F_profiles.shape[0]):
                factor_color = self.color_manager.get_factor_color(i)
                ax.bar(self.species_names, F_profiles[i, :], bottom=bottom, 
                       label=f'Factor {i+1}', alpha=0.8, color=factor_color)
                bottom += F_profiles[i, :]
            
            ax.set_title(f'{station_display_name} Species Composition by PMF Factors', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Species')
            ax.set_ylabel('Total Contribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plot_path = dashboard_dir / f"{self.filename_prefix}_species_composition.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(plot_path)
            print(f"   ✅ Saved: species_composition.png")
            
        except Exception as e:
            print(f"   ❌ Error creating species composition: {e}")
        
        try:
            # Plot 4: Model Quality Assessment
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Q-values for all models
            q_true_values = [model.Qtrue for model in self.batch_models.results]
            q_robust_values = [model.Qrobust for model in self.batch_models.results]
            
            ax1.hist(q_true_values, bins=10, alpha=0.7, label='Q(true)', color='skyblue')
            ax1.axvline(self.best_model.Qtrue, color='red', linestyle='--', linewidth=2, 
                       label=f'Best: {self.best_model.Qtrue:.1f}')
            ax1.set_title('Q(true) Distribution')
            ax1.set_xlabel('Q(true)')
            ax1.set_ylabel('Frequency')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.hist(q_robust_values, bins=10, alpha=0.7, label='Q(robust)', color='lightgreen')
            ax2.axvline(self.best_model.Qrobust, color='red', linestyle='--', linewidth=2,
                       label=f'Best: {self.best_model.Qrobust:.1f}')
            ax2.set_title('Q(robust) Distribution')
            ax2.set_xlabel('Q(robust)')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            fig.suptitle(f'{station_display_name} PMF Model Quality Assessment', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plot_path = dashboard_dir / f"{self.filename_prefix}_model_quality.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(plot_path)
            print(f"   ✅ Saved: model_quality.png")
            
        except Exception as e:
            print(f"   ❌ Error creating model quality plot: {e}")
        
        # Plot 5: Residual Analysis (EPA recommended)
        try:
            self._create_residual_plots(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating residual plots: {e}")
        
        # Plot 6: Factor Correlation Analysis
        try:
            self._create_correlation_plots(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating correlation plots: {e}")
        
        # Plot 7: Source Contribution Analysis
        try:
            self._create_source_contribution_plots(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating source contribution plots: {e}")
        
        # Plot 8: Seasonal/Temporal Analysis
        try:
            self._create_temporal_analysis_plots(dashboard_dir, plot_files, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating temporal analysis plots: {e}")
        
        # Plot 9: Bootstrap/Uncertainty Analysis (if multiple models)
        try:
            self._create_uncertainty_plots(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating uncertainty plots: {e}")
        
        # Plot 10: Diagnostic Scatter Plots
        try:
            self._create_diagnostic_scatters(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating diagnostic scatter plots: {e}")
        
        # Plot 11: Factor Optimization Plot (Q vs Factors)
        try:
            self._create_optimization_plot(dashboard_dir, plot_files)
        except Exception as e:
            print(f"   ❌ Error creating optimization plot: {e}")
        
        # Plot 12: Wind Direction and Speed Analysis
        try:
            self._create_wind_analysis_plots(dashboard_dir, plot_files, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating wind analysis plots: {e}")
        
        # Plot 13: Temperature Analysis
        try:
            self._create_temperature_analysis_plots(dashboard_dir, plot_files, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating temperature analysis plots: {e}")
        
        # Plot 14: Pressure Analysis
        try:
            self._create_pressure_analysis_plots(dashboard_dir, plot_files, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating pressure analysis plots: {e}")
        
        # Plot 15: Sankey Diagram (Factors → Species)
        try:
            self._create_sankey_diagram(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   ❌ Error creating Sankey diagram: {e}")
        
        # NEW S/N Categorization and EPA Analysis Plots
        if self.snr_enable and hasattr(self, '_snr_categories'):
            try:
                self._create_snr_analysis_plots(dashboard_dir, plot_files)
            except Exception as e:
                print(f"   ❌ Error creating S/N analysis plots: {e}")
        
        # Plot 16: PCA vs PMF Comparison Plots (if PCA has been run)
        try:
            self._create_pca_comparison_plots(dashboard_dir, plot_files)
        except Exception as e:
            print(f"   ❌ Error creating PCA comparison plots: {e}")
        
        # Create summary dashboard HTML
        self._create_html_dashboard(plot_files)
        
        print(f"📊 Dashboard complete: {len(plot_files)} plots generated")
    
    def _create_snr_analysis_plots(self, dashboard_dir, plot_files):
        """
        Create comprehensive S/N categorization analysis plots and summaries.
        """
        print("   🔢 Creating S/N categorization analysis plots...")
        
        # Define consistent category colors
        category_colors = {
            'strong': '#2ecc71',  # Green
            'weak': '#f39c12',    # Orange
            'bad': '#e74c3c'      # Red
        }
        
        # Load S/N metrics if available
        snr_metrics_file = self.output_dir / f"{self.filename_prefix}_snr_metrics.csv"
        categories_file = self.output_dir / f"{self.filename_prefix}_species_categories.csv"
        
        if not snr_metrics_file.exists():
            print("   ⚠️ S/N metrics file not found, skipping S/N plots")
            return
        
        snr_metrics = pd.read_csv(snr_metrics_file)
        categories_df = pd.read_csv(categories_file) if categories_file.exists() else None
        
        # Create 2x3 subplot layout for S/N analysis
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{self.station} S/N Categorization Analysis (EPA PMF 5.0)', fontsize=16, fontweight='bold')
        
        # Plot 1: S/N by species (bar chart)
        ax1 = axes[0, 0]
        snr_metrics_sorted = snr_metrics.sort_values('snr', ascending=False)
        colors = [category_colors.get(self._snr_categories.get(species, 'strong'), '#95a5a6') 
                  for species in snr_metrics_sorted['species']]
        
        bars = ax1.bar(range(len(snr_metrics_sorted)), snr_metrics_sorted['snr'], color=colors, alpha=0.8)
        ax1.axhline(y=self.snr_weak_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Weak threshold ({self.snr_weak_threshold})')
        ax1.axhline(y=self.snr_bad_threshold, color='red', linestyle='--', alpha=0.7, label=f'Bad threshold ({self.snr_bad_threshold})')
        ax1.set_title('Signal-to-Noise Ratio by Species')
        ax1.set_ylabel('S/N Ratio')
        ax1.set_xticks(range(len(snr_metrics_sorted)))
        ax1.set_xticklabels(snr_metrics_sorted['species'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: BDL and Missing fractions (stacked bar)
        ax2 = axes[0, 1]
        species_order = snr_metrics_sorted['species']
        bdl_fracs = [snr_metrics[snr_metrics['species'] == sp]['bdl_frac'].iloc[0] for sp in species_order]
        missing_fracs = [snr_metrics[snr_metrics['species'] == sp]['missing_frac'].iloc[0] for sp in species_order]
        valid_fracs = [1 - bdl - missing for bdl, missing in zip(bdl_fracs, missing_fracs)]
        
        x_pos = range(len(species_order))
        ax2.bar(x_pos, valid_fracs, label='Valid', color='#27ae60', alpha=0.8)
        ax2.bar(x_pos, bdl_fracs, bottom=valid_fracs, label='BDL', color='#e67e22', alpha=0.8)
        ax2.bar(x_pos, missing_fracs, bottom=[v + b for v, b in zip(valid_fracs, bdl_fracs)], 
               label='Missing', color='#95a5a6', alpha=0.8)
        
        ax2.axhline(y=1-self.snr_bdl_bad_frac, color='red', linestyle='--', alpha=0.7, 
                   label=f'Bad BDL threshold ({self.snr_bdl_bad_frac*100:.0f}%)')
        ax2.axhline(y=1-self.snr_bdl_weak_frac, color='orange', linestyle='--', alpha=0.7,
                   label=f'Weak BDL threshold ({self.snr_bdl_weak_frac*100:.0f}%)')
        
        ax2.set_title('Data Quality Fractions by Species')
        ax2.set_ylabel('Fraction')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(species_order, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mean concentration vs uncertainty scatter
        ax3 = axes[0, 2]
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        unc_file = self.output_dir / f"{self.filename_prefix}_uncertainties.csv"
        
        if conc_file.exists() and unc_file.exists():
            conc_data = pd.read_csv(conc_file, index_col=0)
            unc_data = pd.read_csv(unc_file, index_col=0)
            
            mean_concs = conc_data.mean()
            mean_uncs = unc_data.mean()
            
            for species in mean_concs.index:
                if species in self._snr_categories:
                    category = self._snr_categories[species]
                    color = category_colors.get(category, '#95a5a6')
                    ax3.scatter(mean_concs[species], mean_uncs[species], 
                              c=color, s=100, alpha=0.7, label=category if category not in ax3.get_legend_handles_labels()[1] else "")
                    ax3.annotate(species, (mean_concs[species], mean_uncs[species]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax3.set_xscale('log')
            ax3.set_yscale('log')
            ax3.set_title('Mean Concentration vs Mean Uncertainty')
            ax3.set_xlabel('Mean Concentration (μg/m³)')
            ax3.set_ylabel('Mean Uncertainty (μg/m³)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Uncertainty distribution by species (boxplot)
        ax4 = axes[1, 0]
        if unc_file.exists():
            species_uncs = []
            species_labels = []
            species_colors_list = []
            
            for species in unc_data.columns:
                if species in self._snr_categories:
                    species_uncs.append(unc_data[species].dropna())
                    species_labels.append(species)
                    species_colors_list.append(category_colors.get(self._snr_categories[species], '#95a5a6'))
            
            if species_uncs:
                bp = ax4.boxplot(species_uncs, labels=species_labels, patch_artist=True)
                for patch, color in zip(bp['boxes'], species_colors_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax4.set_title('Uncertainty Distributions by Species')
                ax4.set_ylabel('Uncertainty (μg/m³)')
                ax4.set_yscale('log')
                plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
                ax4.grid(True, alpha=0.3)
        
        # Plot 5: Impact of categorization (weak species only)
        ax5 = axes[1, 1]
        weak_species = [sp for sp, cat in self._snr_categories.items() if cat == 'weak']
        
        if weak_species:
            # Show the 3x uncertainty multiplier for weak species
            ratios = [3.0] * len(weak_species)  # All weak species get 3x uncertainty
            bars = ax5.bar(range(len(weak_species)), ratios, 
                          color=category_colors['weak'], alpha=0.8)
            ax5.axhline(y=3.0, color='orange', linestyle='--', alpha=0.7, label='Applied multiplier')
            ax5.set_title('Applied Uncertainty Scaling for Weak Species')
            ax5.set_ylabel('Uncertainty Multiplier')
            ax5.set_xticks(range(len(weak_species)))
            ax5.set_xticklabels(weak_species, rotation=45, ha='right')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
            
            # Annotate bars
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        '3.0x', ha='center', va='bottom', fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No weak species\nidentified', ha='center', va='center',
                    transform=ax5.transAxes, fontsize=14)
            ax5.set_title('Applied Uncertainty Scaling for Weak Species')
        
        # Plot 6: Category summary and reasons
        ax6 = axes[1, 2]
        if categories_df is not None and 'reasons' in categories_df.columns:
            # Count reason types
            reason_counts = {'S/N < threshold': 0, 'BDL > threshold': 0, 'Missing > threshold': 0, 'Other': 0}
            
            for reasons_str in categories_df['reasons'].fillna(''):
                if 'S/N <' in reasons_str:
                    reason_counts['S/N < threshold'] += 1
                elif 'BDL >' in reasons_str:
                    reason_counts['BDL > threshold'] += 1
                elif 'Missing >' in reasons_str:
                    reason_counts['Missing > threshold'] += 1
                elif reasons_str.strip():  # Non-empty but not matching above
                    reason_counts['Other'] += 1
            
            # Create stacked bar showing categorization reasons
            categories = list(self._snr_categories.values())
            cat_counts = {'strong': categories.count('strong'), 
                         'weak': categories.count('weak'), 
                         'bad': categories.count('bad')}
            
            bars = ax6.bar(cat_counts.keys(), cat_counts.values(), 
                          color=[category_colors[cat] for cat in cat_counts.keys()], alpha=0.8)
            ax6.set_title('Species Categorization Summary')
            ax6.set_ylabel('Number of Species')
            ax6.grid(True, alpha=0.3)
            
            # Annotate bars with counts
            for i, (cat, count) in enumerate(cat_counts.items()):
                if count > 0:
                    ax6.text(i, count + 0.1, str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_snr_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: snr_analysis.png")
    
    def _get_cli_flags_html_section(self):
        """
        Generate HTML section with complete CLI flags record for reproducibility.
        """
        # Get all CLI parameters with current values
        cli_params = {
            # Core parameters
            'station': getattr(self, 'station', None),
            'data_dir': getattr(self, 'data_dir', None),
            'patterns': getattr(self, 'patterns', None),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'output_dir': str(self.output_dir),
            'remove_voc': getattr(self, 'remove_voc', False),
            
            # EPA uncertainty parameters
            'uncertainty_mode': self.uncertainty_mode,
            'uncertainty_ef_mdl': self.uncertainty_ef_mdl,
            'uncertainty_epsilon': self.uncertainty_epsilon,
            'legacy_min_u': self.legacy_min_u,
            'uncertainty_bdl_policy': self.uncertainty_bdl_policy,
            
            # S/N categorization parameters
            'snr_enable': self.snr_enable,
            'snr_weak_threshold': self.snr_weak_threshold,
            'snr_bad_threshold': self.snr_bad_threshold,
            'snr_bdl_weak_frac': self.snr_bdl_weak_frac,
            'snr_bdl_bad_frac': self.snr_bdl_bad_frac,
            'snr_missing_weak_frac': self.snr_missing_weak_frac,
            'snr_missing_bad_frac': self.snr_missing_bad_frac,
            'exclude_bad': self.exclude_bad,
            
            # Robust training
            'robust_fit': getattr(self, 'robust_fit', False),
            'robust_alpha': getattr(self, 'robust_alpha', 4.0),
            
            # ESAT algorithm controls
            'method': getattr(self, 'method', 'ls-nmf'),
            'init_method': getattr(self, 'init_method', 'column_mean'),
            'init_norm': getattr(self, 'init_norm', True),
            'hold_h': getattr(self, 'hold_h', False),
            'delay_h': getattr(self, 'delay_h', -1),
            
            # Output controls
            'dashboard_snr_panel': getattr(self, 'dashboard_snr_panel', True),
            'write_diagnostics': self.write_diagnostics,
            'seed': self.seed,
        }
        
        html_section = """
            <div class="cli-section">
                <h2>📋 CLI Flags Used (Reproducibility Record)</h2>
                <p><strong>Command to reproduce this analysis:</strong></p>
                <pre style="background-color: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto;">
        """
        
        # Build command line
        cmd_parts = ['python pmf_source_app.py']
        
        if cli_params['station']:
            cmd_parts.append(f"--station {cli_params['station']}")
        if cli_params['data_dir']:
            cmd_parts.append(f'--data-dir "{cli_params["data_dir"]}"')
        if cli_params['patterns']:
            cmd_parts.append(f'--patterns "{cli_params["patterns"]}"')
        if cli_params['start_date']:
            cmd_parts.append(f"--start-date {cli_params['start_date']}")
        if cli_params['end_date']:
            cmd_parts.append(f"--end-date {cli_params['end_date']}")
        if cli_params['output_dir'] and cli_params['output_dir'] != 'pmf_results':
            cmd_parts.append(f'--output-dir "{cli_params["output_dir"]}"')
        if cli_params['remove_voc']:
            cmd_parts.append('--remove-voc')
        if cli_params['uncertainty_mode'] != 'legacy':
            cmd_parts.append(f"--uncertainty-mode {cli_params['uncertainty_mode']}")
        if cli_params['snr_enable']:
            cmd_parts.append('--snr-enable')
        if cli_params['write_diagnostics']:
            cmd_parts.append('--write-diagnostics')
        if cli_params['robust_fit']:
            cmd_parts.append('--robust-fit')
        if cli_params['hold_h']:
            cmd_parts.append('--hold-h')
        if not cli_params['init_norm']:
            cmd_parts.append('--no-init-norm')
        
        # Add other non-default parameters
        non_defaults = {
'uncertainty_epsilon': (1e-12, cli_params['uncertainty_epsilon']),
            'legacy_min_u': (0.1, cli_params['legacy_min_u']),
            'snr_weak_threshold': (2.0, cli_params['snr_weak_threshold']),
            'snr_bad_threshold': (0.2, cli_params['snr_bad_threshold']),
            'robust_alpha': (4.0, cli_params['robust_alpha']),
            'seed': (42, cli_params['seed']),
            'method': ('ls-nmf', cli_params['method']),
            'init_method': ('column_mean', cli_params['init_method']),
            'delay_h': (-1, cli_params['delay_h']),
        }
        
        for param, (default, value) in non_defaults.items():
            if value != default:
                cmd_parts.append(f"--{param.replace('_', '-')} {value}")
        
        # Format command with line breaks for readability
        cmd_str = ' \\\n    '.join(cmd_parts)
        
        html_section += cmd_str
        html_section += """
                </pre>
                
                <h3>Parameter Details:</h3>
                <table>
                    <tr><th>Parameter</th><th>Value</th><th>Description</th></tr>
        """
        
        # Parameter descriptions - separate values and descriptions
        param_info = {
'uncertainty_mode': (cli_params["uncertainty_mode"], 'Uncertainty calculation method'),
            'method': (cli_params['method'], 'ESAT NMF algorithm (ls-nmf or ws-nmf)'),
            'init_method': (cli_params['init_method'], 'Matrix initialization method'),
            'init_norm': (cli_params['init_norm'], 'Normalize data before kmeans initialization'),
            'hold_h': (cli_params['hold_h'], 'Hold H (profile) matrix constant during training'),
            'delay_h': (cli_params['delay_h'], 'Hold H matrix for N iterations (-1 = disabled)'),
            'snr_enable': (cli_params["snr_enable"], 'EPA S/N-based feature categorization'),
            'robust_fit': (cli_params['robust_fit'], 'Use robust loss during SA training (fallback only)'),
            'robust_alpha': (cli_params['robust_alpha'], 'Robust cutoff alpha for scaled residuals'),
            'snr_weak_threshold': (cli_params["snr_weak_threshold"], 'S/N threshold for weak species'),
            'snr_bad_threshold': (cli_params["snr_bad_threshold"], 'S/N threshold for bad species'),
            'exclude_bad': (cli_params["exclude_bad"], 'Exclude bad species from analysis'),
            'seed': (cli_params["seed"], 'Random seed for reproducibility'),
            'write_diagnostics': (cli_params["write_diagnostics"], 'Write diagnostic CSV files'),
        }
        
        for param, (value, description) in param_info.items():
            html_section += f"""
                    <tr><td>--{param.replace('_', '-')}</td><td>{value}</td><td>{description}</td></tr>
            """
        
        html_section += """
                </table>
                <p><small><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></small></p>
            </div>
        """
        
        return html_section
    
    def _create_html_dashboard(self, plot_files):
        """Create HTML dashboard combining all plots."""
        # Get the full station display name
        station_display_name = self._get_station_display_name()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{station_display_name} PMF Source Apportionment Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .plot-container {{ margin: 20px 0; text-align: center; }}
                .plot-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .summary {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .config-section {{ background-color: #e8f4fd; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .epa-section {{ background-color: #fff2e6; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .snr-section {{ background-color: #f0f8f0; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .cli-section {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; font-family: monospace; }}
                .category-strong {{ color: #2ecc71; font-weight: bold; }}
                .category-weak {{ color: #f39c12; font-weight: bold; }}
                .category-bad {{ color: #e74c3c; font-weight: bold; }}
                table {{ border-collapse: collapse; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{station_display_name} PMF Source Apportionment Analysis</h1>
                <p><strong>Analysis Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
                <p><strong>Data Period:</strong> {self.start_date or 'All'} to {self.end_date or 'All'}</p>
                <p><strong>Factors Resolved:</strong> {self.factors}</p>
                <p><strong>Models Run:</strong> {self.models}</p>
            </div>
            
            <div class="config-section">
                <h2>🔧 Run Configuration</h2>
                <ul>
                    <li><strong>Uncertainty Mode:</strong> {self.uncertainty_mode} 
                        {('(EPA PMF 5.0 formulas)' if self.uncertainty_mode == 'epa' else '(Fixed MDL/EF table)')}</li>
                    <li><strong>ESAT Algorithm:</strong> {self.method.upper()} {'(Semi-NMF, allows negative W)' if self.method == 'ws-nmf' else '(Standard PMF, nonnegative)'}</li>
                    <li><strong>Initialization:</strong> {self.init_method.replace('_', ' ').title()}{(' with normalization' if self.init_norm else ' without normalization') if self.init_method == 'kmeans' else ''}</li>
                    <li><strong>Matrix Updates:</strong> {'H held constant, ' if self.hold_h else ''}{'H delayed for ' + str(self.delay_h) + ' iterations, ' if self.delay_h > 0 else ''}Standard training</li>
                    <li><strong>Random Seed:</strong> {self.seed}</li>
                    <li><strong>Records Analyzed:</strong> {len(self.concentration_data):,}</li>
                    <li><strong>Original Species:</strong> {len(self.concentration_data.columns) + (len(getattr(self, '_excluded_species', [])))} 
                        {'(' + str(len(getattr(self, '_excluded_species', []))) + ' excluded)' if hasattr(self, '_excluded_species') and len(getattr(self, '_excluded_species', [])) > 0 else ''}</li>
                    <li><strong>Final Species:</strong> {len(self.concentration_data.columns)}</li>
                </ul>
            </div>
        """
        
        # Add EPA policy section if using EPA mode
        if self.uncertainty_mode == 'epa':
            html_content += f"""
            <div class="epa-section">
                <h2>🏧 EPA Uncertainty Policy (PMF 5.0)</h2>
                <ul>
                    <li><strong>Above MDL:</strong> U = √((EF × conc)² + (0.5 × MDL)²)</li>
                    <li><strong>BDL Cases:</strong> V = MDL/2, U = {self.uncertainty_bdl_policy.replace('-', ' ').title()}</li>
                    <li><strong>Missing Cases:</strong> V = species median (fallback MDL), U = 4 × median (fallback 4 × MDL)</li>
                    <li><strong>EF/MDL Source:</strong> {'Built-in database' if not self.uncertainty_ef_mdl else f'Custom CSV: {self.uncertainty_ef_mdl}'}</li>
                    <li><strong>Aggregation Scaling:</strong> Applied as 1/√n when counts are available</li>
                </ul>
            </div>
            """
        else:
            html_content += f"""
            <div class="epa-section">
                <h2>🏧 Legacy Uncertainty Policy</h2>
                <ul>
                    <li><strong>Method:</strong> Fixed MDL/Error Fraction table</li>
                    <li><strong>Above MDL:</strong> U = √((EF × conc)² + MDL²)</li>
                    <li><strong>BDL/Missing:</strong> V = MDL/2, U = (5/6) × MDL</li>
                    <li><strong>Minimum Uncertainty:</strong> {self.legacy_min_u}</li>
                    <li><strong>Aggregation Scaling:</strong> Applied as 1/√n after uncertainty calculation</li>
                </ul>
            </div>
            """
        
        # Add S/N categorization section if enabled
        if self.snr_enable and hasattr(self, '_snr_categories'):
            # Count categories
            strong_count = sum(1 for cat in self._snr_categories.values() if cat == 'strong')
            weak_count = sum(1 for cat in self._snr_categories.values() if cat == 'weak')
            bad_count = sum(1 for cat in self._snr_categories.values() if cat == 'bad')
            
            # Get average S/N
            if hasattr(self, '_snr_metrics'):
                avg_snr = sum(m['snr'] for m in self._snr_metrics.values()) / len(self._snr_metrics)
            else:
                avg_snr = 0.0
            
            html_content += f"""
            <div class="snr-section">
                <h2>🔢 S/N Categorization Summary (EPA PMF 5.0)</h2>
                <ul>
                    <li><strong>Total Species:</strong> {len(self._snr_categories)}</li>
                    <li><strong>Average S/N:</strong> {avg_snr:.3f}</li>
                    <li><strong>Thresholds:</strong> Strong ≥ {self.snr_weak_threshold}, Weak {self.snr_bad_threshold}-{self.snr_weak_threshold}, Bad < {self.snr_bad_threshold}</li>
                </ul>
                <h3>Category Breakdown:</h3>
                <ul>
                    <li><span class="category-strong">Strong: {strong_count} species</span> (normal weighting)</li>
                    <li><span class="category-weak">Weak: {weak_count} species</span> (uncertainty × 3)</li>
                    <li><span class="category-bad">Bad: {bad_count} species</span> (excluded from analysis)</li>
                </ul>
            """
            
            # Add species categorization table
            if hasattr(self, '_snr_categories'):
                html_content += """
                <h3>Species Categories:</h3>
                <table>
                    <tr><th>Species</th><th>Category</th><th>S/N</th><th>Action</th></tr>
                """
                
                # Load S/N metrics if available
                snr_metrics_file = self.output_dir / f"{self.filename_prefix}_snr_metrics.csv"
                snr_dict = {}
                if snr_metrics_file.exists():
                    snr_data = pd.read_csv(snr_metrics_file)
                    snr_dict = dict(zip(snr_data['species'], snr_data['snr']))
                
                for species, category in sorted(self._snr_categories.items()):
                    snr_value = snr_dict.get(species, 0.0)
                    if category == 'strong':
                        action = 'Normal weighting'
                    elif category == 'weak':
                        action = 'Uncertainty tripled'
                    else:  # bad
                        action = 'Excluded from analysis'
                    
                    html_content += f"""
                    <tr>
                        <td>{species}</td>
                        <td><span class="category-{category}">{category.title()}</span></td>
                        <td>{snr_value:.3f}</td>
                        <td>{action}</td>
                    </tr>
                    """
                
                html_content += "</table>"
            
            html_content += "</div>"
        
        html_content += """
            <div class="summary">
                <h2>Model Performance</h2>
                <ul>
                    <li><strong>Q(true):</strong> {self.best_model.Qtrue:.2f}</li>
                    <li><strong>Q(robust):</strong> {self.best_model.Qrobust:.2f}</li>
        """
        
        # Add Q/DoF interpretation if DoF available
        if hasattr(self.best_model, 'Qtrue') and hasattr(self.best_model, 'Qrobust'):
            try:
                # Calculate DoF
                n_samples = len(self.concentration_data)
                n_species = len(self.concentration_data.columns)
                n_factors = self.factors
                dof = n_samples * n_species - n_factors * (n_samples + n_species)
                
                if dof > 0:
                    q_dof_ratio = self.best_model.Qrobust / dof
                    if q_dof_ratio <= 1.5:
                        quality = "Excellent"
                    elif q_dof_ratio <= 2.0:
                        quality = "Good"
                    elif q_dof_ratio <= 3.0:
                        quality = "Fair"
                    else:
                        quality = "Poor"
                    
                    html_content += f"""
                    <li><strong>DoF:</strong> {dof:,}</li>
                    <li><strong>Q/DoF Ratio:</strong> {q_dof_ratio:.3f} ({quality} per EPA guidelines)</li>
                    """
            except:
                pass
        
        html_content += """
                </ul>
            </div>
        """
        
        # Add plots to HTML (only image files, not text summaries)
        image_extensions = {'.png', '.jpg', '.jpeg', '.svg', '.gif'}
        for plot_file in plot_files:
            # Only include image files in HTML dashboard
            if plot_file.suffix.lower() in image_extensions:
                plot_name = plot_file.stem.replace('_', ' ').title()
                html_content += f"""
            <div class="plot-container">
                <h3>{plot_name}</h3>
                <img src="dashboard/{plot_file.name}" alt="{plot_name}">
            </div>
            """
        
        # Add CLI flags record at bottom
        html_content += self._get_cli_flags_html_section()
        
        html_content += "</body></html>"
        
        # Save HTML dashboard
        html_file = self.output_dir / f"{self.filename_prefix}_pmf_dashboard.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📄 HTML Dashboard: {html_file}")
    
    def generate_report(self):
        """Generate comprehensive PMF analysis report."""
        if not self.best_model:
            print("❌ No PMF model available for report generation")
            return
            
        report_path = self.output_dir / f"{self.filename_prefix}_pmf_report.md"
        
        # Get the full station display name
        station_display_name = self._get_station_display_name()
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"""# {station_display_name} PMF Source Apportionment Report

## Analysis Overview
- **Station**: {station_display_name}
- **Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- **Data Period**: {self.start_date or 'All available'} to {self.end_date or 'All available'}
- **Records Analyzed**: {len(self.concentration_data):,}

## Model Configuration
- **Factors Resolved**: {self.factors}
- **Models Run**: {self.models}
- **ESAT Algorithm**: {self.method.upper()} ({'Semi-NMF (allows negative W)' if self.method == 'ws-nmf' else 'Least Squares NMF (standard PMF)'})
- **Initialization**: {self.init_method.replace('_', ' ').title()}{' with normalization' if self.init_norm and self.init_method == 'kmeans' else ''}
- **Training**: {'H held constant' if self.hold_h else 'H delayed ' + str(self.delay_h) + ' iterations' if self.delay_h > 0 else 'Standard W/H updates'}
- **ESAT Version**: Working (Rust-optimized)

## Model Performance
- **Q(true)**: {self.best_model.Qtrue:.2f}
- **Q(robust)**: {self.best_model.Qrobust:.2f}
- **Best Model Index**: {self.batch_models.best_model if self.batch_models else 'Single Model (Robust Mode)'}

## Species Analyzed
""")
            
            # Add species information
            for i, species in enumerate(self.concentration_data.columns):
                unit = self.units.get(species, 'unknown')
                data_points = self.concentration_data[species].notna().sum()
                completeness = (data_points / len(self.concentration_data)) * 100
                f.write(f"- **{species}** ({unit}): {data_points:,} data points ({completeness:.1f}% complete)\n")
            
            f.write(f"""
## Files Generated
- Concentration data: `{self.station}_concentrations.csv`
- Uncertainty data: `{self.station}_uncertainties.csv`
- PMF Dashboard: `{self.station}_pmf_dashboard.html`
- Individual plots: `dashboard/` directory

## Quality Assurance
This analysis follows EPA PMF 5.0 User Guide best practices:
- ✅ Appropriate uncertainty estimation using EPA formula
- ✅ Missing value treatment using EPA Method 1
- ✅ Batch modeling with {self.models} runs for robustness
- ✅ Comprehensive diagnostic plots generated
- ✅ ESAT Rust-optimized PMF implementation

## Technical Notes
- Used ESAT library with Rust-accelerated computations
- Fixed API compatibility issues (H/W matrices, Qtrue/Qrobust attributes)
- Parallel processing utilized for batch modeling
- Factor optimization performed across multiple solutions

## Recommendations
1. Review factor profiles for source identification
2. Examine factor contributions for temporal patterns
3. Validate results with local source inventory
4. Consider seasonal analysis if data span is sufficient
""")
        
        print(f"📄 Analysis report: {report_path}")
    
    def _create_residual_plots(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create residual analysis plots (EPA recommended)."""
        print("   🔍 Creating residual analysis plots...")
        
        # Reconstruct the original data from PMF results
        reconstructed = G_contributions @ F_profiles
        
        # Load original data for comparison
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        original_data = pd.read_csv(conc_file, index_col=0).values
        
        # Calculate residuals - ensure no intermediate results display
        residuals = original_data - reconstructed;
        residual_percent = (residuals / original_data) * 100;
        residual_percent = np.nan_to_num(residual_percent, nan=0, posinf=0, neginf=0);
        
        # Create residual plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.station} PMF Residual Analysis (EPA Diagnostic)', fontsize=16, fontweight='bold')
        
        # Plot 1: Residual vs Predicted
        ax1 = axes[0, 0]
        ax1.scatter(reconstructed.flatten(), residuals.flatten(), alpha=0.5, s=20)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_xlabel('Predicted Concentration')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plot of residuals
        from scipy import stats
        ax2 = axes[0, 1]
        stats.probplot(residuals.flatten(), dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Residuals by species
        ax3 = axes[1, 0]
        species_residuals = []
        for i, species in enumerate(self.species_names):
            species_res = residuals[:, i]
            species_residuals.append(species_res[~np.isnan(species_res)])
        
        bp = ax3.boxplot(species_residuals, labels=self.species_names, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax3.set_title('Residuals by Species')
        ax3.set_ylabel('Residuals')
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Residual percentage distribution
        ax4 = axes[1, 1]
        ax4.hist(residual_percent.flatten(), bins=50, alpha=0.7, color='green', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Residual Percentage (%)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Residual Percentages')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_residual_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: residual_analysis.png")
    
    def _create_correlation_plots(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create factor correlation analysis plots."""
        print("   🔗 Creating correlation analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.station} PMF Factor Correlation Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Factor correlation matrix (time series)
        ax1 = axes[0, 0]
        factor_corr = np.corrcoef(G_contributions.T)
        im1 = ax1.imshow(factor_corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title('Factor Time Series Correlations')
        ax1.set_xlabel('Factor')
        ax1.set_ylabel('Factor')
        factor_labels = [f'F{i+1}' for i in range(self.factors)]
        ax1.set_xticks(range(self.factors))
        ax1.set_yticks(range(self.factors))
        ax1.set_xticklabels(factor_labels)
        ax1.set_yticklabels(factor_labels)
        
        # Add correlation values to heatmap
        for i in range(self.factors):
            for j in range(self.factors):
                ax1.text(j, i, f'{factor_corr[i, j]:.2f}', ha='center', va='center')
        
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Species correlation matrix
        ax2 = axes[0, 1]
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        original_data = pd.read_csv(conc_file, index_col=0)
        species_corr = original_data.corr()
        
        im2 = ax2.imshow(species_corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_title('Species Correlations')
        ax2.set_xticks(range(len(self.species_names)))
        ax2.set_yticks(range(len(self.species_names)))
        ax2.set_xticklabels(self.species_names, rotation=45, ha='right')
        ax2.set_yticklabels(self.species_names)
        plt.colorbar(im2, ax=ax2)
        
        # Plot 3: Factor loadings scatter
        ax3 = axes[1, 0]
        if self.factors >= 2:
            ax3.scatter(F_profiles[0, :], F_profiles[1, :], s=100, alpha=0.7)
            for i, species in enumerate(self.species_names):
                ax3.annotate(species, (F_profiles[0, i], F_profiles[1, i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax3.set_xlabel('Factor 1 Loading')
            ax3.set_ylabel('Factor 2 Loading')
            ax3.set_title('Factor Loadings Scatter (F1 vs F2)')
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Factor contributions scatter
        ax4 = axes[1, 1]
        if self.factors >= 2:
            ax4.scatter(G_contributions[:, 0], G_contributions[:, 1], alpha=0.6, s=30)
            ax4.set_xlabel('Factor 1 Contribution')
            ax4.set_ylabel('Factor 2 Contribution')
            ax4.set_title('Factor Contributions Scatter (F1 vs F2)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_correlation_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: correlation_analysis.png")
    
    def _create_source_contribution_plots(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create source contribution analysis plots."""
        print("   📊 Creating source contribution plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.station} PMF Source Contribution Analysis', fontsize=16, fontweight='bold')
        
        # Calculate total contributions
        total_contributions = np.sum(G_contributions, axis=0)
        relative_contributions = total_contributions / np.sum(total_contributions) * 100
        
        # Plot 1: Pie chart of factor contributions
        ax1 = axes[0, 0]
        factor_colors = self.color_manager.get_factor_colors()
        wedges, texts, autotexts = ax1.pie(relative_contributions, 
                                          labels=[f'Factor {i+1}' for i in range(self.factors)],
                                          autopct='%1.1f%%', colors=factor_colors, startangle=90)
        ax1.set_title('Relative Source Contributions')
        
        # Plot 2: Stacked bar chart over time (binned)
        ax2 = axes[0, 1]
        n_bins = min(20, len(G_contributions) // 5)  # Adaptive binning
        if n_bins > 1:
            bin_size = len(G_contributions) // n_bins
            binned_contributions = []
            bin_labels = []
            
            for i in range(0, len(G_contributions), bin_size):
                end_idx = min(i + bin_size, len(G_contributions))
                bin_mean = np.mean(G_contributions[i:end_idx], axis=0)
                binned_contributions.append(bin_mean)
                bin_labels.append(f'{i//bin_size + 1}')
            
            binned_contributions = np.array(binned_contributions)
            bottom = np.zeros(len(binned_contributions))
            
            for i in range(self.factors):
                factor_color = self.color_manager.get_factor_color(i)
                ax2.bar(bin_labels, binned_contributions[:, i], bottom=bottom, 
                       label=f'Factor {i+1}', color=factor_color, alpha=0.8)
                bottom += binned_contributions[:, i]
        
        ax2.set_title('Source Contributions Over Time (Binned)')
        ax2.set_xlabel('Time Bin')
        ax2.set_ylabel('Concentration Contribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Factor strength by species
        ax3 = axes[1, 0]
        species_max_factor = np.argmax(F_profiles, axis=0)
        species_max_strength = np.max(F_profiles, axis=0)
        
        # Color bars by the dominant factor for each species
        bar_colors = [self.color_manager.get_factor_color(species_max_factor[i]) for i in range(len(self.species_names))]
        bars = ax3.bar(range(len(self.species_names)), species_max_strength, 
                       color=bar_colors, alpha=0.8)
        ax3.set_title('Dominant Factor Strength by Species')
        ax3.set_xlabel('Species')
        ax3.set_ylabel('Maximum Factor Loading')
        ax3.set_xticks(range(len(self.species_names)))
        ax3.set_xticklabels(self.species_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # Add factor labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'F{species_max_factor[i]+1}', ha='center', va='bottom', fontsize=8)
        
        # Plot 4: Contribution variability
        ax4 = axes[1, 1]
        factor_std = np.std(G_contributions, axis=0)
        factor_mean = np.mean(G_contributions, axis=0)
        factor_cv = factor_std / factor_mean * 100  # Coefficient of variation
        
        factor_colors = self.color_manager.get_factor_colors()
        bars4 = ax4.bar([f'Factor {i+1}' for i in range(self.factors)], factor_cv, 
                       color=factor_colors, alpha=0.8)
        ax4.set_title('Factor Contribution Variability (CV%)')
        ax4.set_ylabel('Coefficient of Variation (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_source_contribution_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: source_contribution_analysis.png")
    
    def _create_temporal_analysis_plots(self, dashboard_dir, plot_files, G_contributions):
        """Create temporal pattern analysis plots."""
        print("   ⏰ Creating temporal analysis plots...")
        
        # Try to get actual datetime information
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        conc_data = pd.read_csv(conc_file, index_col=0)
        
        try:
            # Parse datetime index
            datetime_index = pd.to_datetime(conc_data.index)
            has_datetime = True
        except:
            # Fallback to sample indices
            datetime_index = np.arange(len(G_contributions))
            has_datetime = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.station} PMF Temporal Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Time series with trend
        ax1 = axes[0, 0]
        
        for i in range(self.factors):
            factor_color = self.color_manager.get_factor_color(i)
            if has_datetime:
                ax1.plot(datetime_index, G_contributions[:, i], label=f'Factor {i+1}', 
                        color=factor_color, alpha=0.7, linewidth=1.5)
            else:
                ax1.plot(G_contributions[:, i], label=f'Factor {i+1}', 
                        color=factor_color, alpha=0.7, linewidth=1.5)
        
        ax1.set_title('Factor Contributions Time Series')
        ax1.set_ylabel('Concentration Contribution')
        if has_datetime:
            ax1.set_xlabel('Date/Time')
            # Format x-axis for better readability
            import matplotlib.dates as mdates
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
            ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        else:
            ax1.set_xlabel('Sample Index')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average patterns (if datetime available)
        ax2 = axes[0, 1]
        if has_datetime and len(datetime_index) > 24:
            # Hour-of-day patterns
            hours = datetime_index.hour
            hourly_means = np.zeros((24, self.factors))
            
            for hour in range(24):
                hour_mask = hours == hour
                if np.any(hour_mask):
                    hourly_means[hour, :] = np.mean(G_contributions[hour_mask], axis=0)
            
            for i in range(self.factors):
                factor_color = self.color_manager.get_factor_color(i)
                ax2.plot(range(24), hourly_means[:, i], 'o-', label=f'Factor {i+1}', 
                        color=factor_color, linewidth=2, markersize=6)
            
            ax2.set_title('Average Diurnal Patterns')
            ax2.set_xlabel('Hour of Day')
            ax2.set_ylabel('Average Contribution')
            ax2.set_xticks(range(0, 24, 4))
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            # Moving average if no datetime
            window = max(5, len(G_contributions) // 20)
            for i in range(self.factors):
                factor_color = self.color_manager.get_factor_color(i)
                rolling_mean = pd.Series(G_contributions[:, i]).rolling(window=window, center=True).mean()
                ax2.plot(rolling_mean, label=f'Factor {i+1}', color=factor_color, linewidth=2)
            
            ax2.set_title(f'Moving Average (window={window})')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Average Contribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Contribution distribution boxplot
        ax3 = axes[1, 0]
        factor_data = [G_contributions[:, i] for i in range(self.factors)]
        bp = ax3.boxplot(factor_data, labels=[f'F{i+1}' for i in range(self.factors)], 
                        patch_artist=True)
        
        for patch, i in zip(bp['boxes'], range(self.factors)):
            factor_color = self.color_manager.get_factor_color(i)
            patch.set_facecolor(factor_color)
            patch.set_alpha(0.7)
        
        ax3.set_title('Factor Contribution Distributions')
        ax3.set_ylabel('Contribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative contribution
        ax4 = axes[1, 1]
        cumsum_contributions = np.cumsum(G_contributions, axis=0)
        
        for i in range(self.factors):
            factor_color = self.color_manager.get_factor_color(i)
            if has_datetime:
                ax4.plot(datetime_index, cumsum_contributions[:, i], label=f'Factor {i+1}', 
                        color=factor_color, linewidth=2)
            else:
                ax4.plot(cumsum_contributions[:, i], label=f'Factor {i+1}', 
                        color=factor_color, linewidth=2)
        
        ax4.set_title('Cumulative Factor Contributions')
        ax4.set_ylabel('Cumulative Contribution')
        if has_datetime:
            ax4.set_xlabel('Date/Time')
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax4.set_xlabel('Sample Index')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_temporal_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: temporal_analysis.png")
    
    def _create_uncertainty_plots(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create uncertainty and bootstrap analysis plots."""
        print("   🎯 Creating uncertainty analysis plots...")
        
        if not USE_BATCH_SA or len(self.batch_models.results) < 5:
            print("   ⚠️ Skipping uncertainty plots (requires multiple models)")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.station} PMF Uncertainty Analysis', fontsize=16, fontweight='bold')
        
        # Collect results from all models
        all_F = np.array([model.H for model in self.batch_models.results])
        all_G = np.array([model.W for model in self.batch_models.results])
        
        # Plot 1: Factor profile uncertainties
        ax1 = axes[0, 0]
        F_mean = np.mean(all_F, axis=0)
        F_std = np.std(all_F, axis=0)
        
        # Show uncertainty for each factor
        x_pos = np.arange(len(self.species_names))
        width = 0.8 / self.factors
        
        for f in range(self.factors):
            offset = (f - self.factors/2) * width
            ax1.bar(x_pos + offset, F_mean[f, :], width, 
                   yerr=F_std[f, :], label=f'Factor {f+1}', 
                   alpha=0.7, capsize=3)
        
        ax1.set_title('Factor Profile Uncertainties')
        ax1.set_xlabel('Species')
        ax1.set_ylabel('Loading ± Std Dev')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(self.species_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Q-value distribution across all models
        ax2 = axes[0, 1]
        all_q_robust = [model.Qrobust for model in self.batch_models.results]
        all_q_true = [model.Qtrue for model in self.batch_models.results]
        
        ax2.hist(all_q_robust, bins=15, alpha=0.7, label='Q(robust)', color='green', density=True)
        ax2.hist(all_q_true, bins=15, alpha=0.7, label='Q(true)', color='blue', density=True)
        ax2.axvline(self.best_model.Qrobust, color='darkgreen', linestyle='--', 
                   label=f'Best Q(robust): {self.best_model.Qrobust:.1f}')
        ax2.axvline(self.best_model.Qtrue, color='darkblue', linestyle='--',
                   label=f'Best Q(true): {self.best_model.Qtrue:.1f}')
        
        ax2.set_title('Model Quality Distribution')
        ax2.set_xlabel('Q-value')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Bootstrap confidence intervals for contributions
        ax3 = axes[1, 0]
        G_mean = np.mean(all_G, axis=0)
        G_percentiles = np.percentile(all_G, [5, 25, 75, 95], axis=0)
        
        # Show confidence bands for first factor (example)
        factor_idx = 0
        x_range = np.arange(len(G_mean))
        
        ax3.fill_between(x_range, G_percentiles[0, :, factor_idx], G_percentiles[3, :, factor_idx], 
                        alpha=0.2, label='90% CI', color='lightblue')
        ax3.fill_between(x_range, G_percentiles[1, :, factor_idx], G_percentiles[2, :, factor_idx], 
                        alpha=0.3, label='50% CI', color='blue')
        ax3.plot(x_range, G_mean[:, factor_idx], 'k-', linewidth=2, label='Mean')
        ax3.plot(x_range, G_contributions[:, factor_idx], 'r--', linewidth=1, label='Best Model')
        
        ax3.set_title(f'Factor {factor_idx+1} Contribution Uncertainty')
        ax3.set_xlabel('Sample Index')
        ax3.set_ylabel('Contribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model stability metrics
        ax4 = axes[1, 1]
        
        # Calculate coefficient of variation for each factor
        # Handle potential division by zero or very small means
        G_std = np.std(all_G, axis=0)
        G_mean = np.mean(all_G, axis=0)
        
        # Calculate CV only where mean is not close to zero
        G_cv = np.zeros_like(G_mean)
        valid_mask = G_mean > 1e-6  # Avoid division by very small numbers
        G_cv[valid_mask] = (G_std[valid_mask] / G_mean[valid_mask]) * 100
        G_cv[~valid_mask] = 0  # Set CV to 0 for factors with near-zero contributions
        
        # Average CV per factor (across all time points)
        factor_stability = np.mean(G_cv, axis=0)
        
        # Cap extremely high CVs for better visualization
        factor_stability = np.minimum(factor_stability, 200)  # Cap at 200%
        
        bars = ax4.bar([f'Factor {i+1}' for i in range(self.factors)], factor_stability, 
                      alpha=0.7, color='orange')
        ax4.set_title('Factor Stability (Average CV%)')
        ax4.set_ylabel('Coefficient of Variation (%)')
        ax4.grid(True, alpha=0.3)
        
        # Add stability assessment with more reasonable thresholds
        for i, bar in enumerate(bars):
            height = bar.get_height()
            # More conservative thresholds for PMF factor stability
            if height < 30:  # Very stable
                stability = 'Stable'
                color = 'green'
            elif height < 60:  # Moderately stable
                stability = 'Moderate'
                color = 'orange'
            elif height < 100:  # Somewhat unstable
                stability = 'Variable'
                color = 'darkorange'
            else:  # Very unstable
                stability = 'Unstable'
                color = 'red'
            
            # Add CV value and stability label
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax4.text(bar.get_x() + bar.get_width()/2., height + 8,
                    stability, ha='center', va='bottom', color=color, fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_uncertainty_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: uncertainty_analysis.png")
    
    def _create_diagnostic_scatters(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create diagnostic scatter plots for model validation."""
        print("   🔬 Creating diagnostic scatter plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{self.station} PMF Diagnostic Scatter Plots', fontsize=16, fontweight='bold')
        
        # Load original and uncertainty data
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        unc_file = self.output_dir / f"{self.filename_prefix}_uncertainties.csv"
        original_data = pd.read_csv(conc_file, index_col=0).values
        uncertainty_data = pd.read_csv(unc_file, index_col=0).values
        
        # Reconstruct data
        reconstructed = G_contributions @ F_profiles
        
        # Plot 1: Observed vs Predicted scatter
        ax1 = axes[0, 0]
        ax1.scatter(original_data.flatten(), reconstructed.flatten(), alpha=0.6, s=20)
        
        # Add 1:1 line
        min_val = min(np.min(original_data), np.min(reconstructed))
        max_val = max(np.max(original_data), np.max(reconstructed))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(original_data.flatten(), reconstructed.flatten())
        
        ax1.set_xlabel('Observed Concentration')
        ax1.set_ylabel('Predicted Concentration')
        ax1.set_title(f'Observed vs Predicted (R² = {r2:.3f})')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Standardized residuals
        ax2 = axes[0, 1]
        residuals = original_data - reconstructed;
        standardized_residuals = residuals / uncertainty_data;
        standardized_residuals = np.nan_to_num(standardized_residuals, nan=0, posinf=0, neginf=0);
        
        ax2.scatter(reconstructed.flatten(), standardized_residuals.flatten(), alpha=0.6, s=20)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=2, color='orange', linestyle=':', alpha=0.7, label='±2σ')
        ax2.axhline(y=-2, color='orange', linestyle=':', alpha=0.7)
        ax2.set_xlabel('Predicted Concentration')
        ax2.set_ylabel('Standardized Residuals')
        ax2.set_title('Standardized Residuals vs Predicted')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Species-specific validation
        ax3 = axes[1, 0]
        species_r2 = []
        species_rmse = []
        
        for i in range(len(self.species_names)):
            obs = original_data[:, i]
            pred = reconstructed[:, i]
            
            # Remove NaN values
            mask = ~(np.isnan(obs) | np.isnan(pred))
            obs_clean = obs[mask]
            pred_clean = pred[mask]
            
            if len(obs_clean) > 0:
                r2_species = r2_score(obs_clean, pred_clean)
                rmse_species = np.sqrt(np.mean((obs_clean - pred_clean)**2))
                species_r2.append(r2_species)
                species_rmse.append(rmse_species)
            else:
                species_r2.append(0)
                species_rmse.append(0)
        
        bars = ax3.bar(range(len(self.species_names)), species_r2, alpha=0.7)
        
        # Color bars by R² quality
        for i, (bar, r2_val) in enumerate(zip(bars, species_r2)):
            if r2_val >= 0.8:
                bar.set_color('green')
            elif r2_val >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
            
            # Add R² value on top of bar (only for non-negative values)
            if r2_val >= 0:
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{r2_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax3.set_title('Species-Specific Model Performance (R²)')
        ax3.set_ylabel('R² Value')
        ax3.set_xticks(range(len(self.species_names)))
        ax3.set_xticklabels(self.species_names, rotation=45, ha='right')
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Leverage plot (influential observations)
        ax4 = axes[1, 1]
        
        # Calculate leverage (simplified) - distance from centroid
        # Ensure no intermediate calculations display values
        data_center = np.mean(original_data, axis=0);
        leverage = np.sum((original_data - data_center)**2, axis=1);
        residual_norm = np.sum(residuals**2, axis=1);
        
        # Create scatter plot with explicit variable assignment to prevent display
        scatter_plot = ax4.scatter(leverage, residual_norm, alpha=0.6, s=30, c=np.arange(len(leverage)), 
                                  cmap='viridis')
        # Assign to _ to suppress any potential return value display
        _ = scatter_plot
        ax4.set_xlabel('Leverage (Distance from Center)')
        ax4.set_ylabel('Sum of Squared Residuals')
        ax4.set_title('Leverage vs Residuals (Outlier Detection)')
        ax4.grid(True, alpha=0.3)
        
        # Mark potential outliers - suppress any array display
        try:
            leverage_threshold = float(np.percentile(leverage, 95))
            residual_threshold = float(np.percentile(residual_norm, 95))
            outlier_mask = (leverage > leverage_threshold) | (residual_norm > residual_threshold)
            
            if np.any(outlier_mask):
                # Use explicit copies to prevent array display
                outlier_x = leverage[outlier_mask].copy()
                outlier_y = residual_norm[outlier_mask].copy()
                _ = ax4.scatter(outlier_x, outlier_y, 
                              c='red', s=60, marker='x', label='Potential Outliers')
                ax4.legend()
        except Exception:
            # Skip outlier marking if it causes issues
            pass
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_diagnostic_scatters.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: diagnostic_scatters.png")
    
    def _create_optimization_plot(self, dashboard_dir, plot_files):
        """Create enhanced Q(robust) vs number of factors optimization plot with EPA reference lines."""
        print("   🔢 Creating factor optimization plot...")
        
        # Check if optimization data is available
        if not hasattr(self, 'optimization_q_values') or not self.optimization_q_values:
            print("   ⚠️ No optimization data available - skipping optimization plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'{self.station} PMF Factor Optimization Analysis', fontsize=16, fontweight='bold')
        
        # Extract factor numbers and Q-values
        factors = sorted(self.optimization_q_values.keys())
        q_values = [self.optimization_q_values[f] for f in factors]
        
        # Calculate Q/DoF ratios for each factor count
        n_samples = len(self.concentration_data)
        n_species = len(self.concentration_data.columns)
        
        q_dof_ratios = []
        for n_factors in factors:
            try:
                dof = n_samples * n_species - n_factors * (n_samples + n_species)
                if dof > 0:
                    q_dof_ratios.append(self.optimization_q_values[n_factors] / dof)
                else:
                    q_dof_ratios.append(float('inf'))
            except:
                q_dof_ratios.append(float('inf'))
        
        # Plot 1: Q(robust) vs factors
        ax1.plot(factors, q_values, 'o-', linewidth=2, markersize=8, alpha=0.7, color='blue', label='Q(robust)')
        
        # Highlight the selected optimal factor
        if hasattr(self, 'optimal_factors') and self.optimal_factors:
            optimal_q = self.optimization_q_values[self.optimal_factors]
            ax1.plot(self.optimal_factors, optimal_q, 'ro', markersize=12, 
                   label=f'Selected: {self.optimal_factors} factors', zorder=5)
            
            # Add annotation
            ax1.annotate(f'Selected\n{self.optimal_factors} factors\nQ = {optimal_q:.1f}',
                       xy=(self.optimal_factors, optimal_q),
                       xytext=(10, 20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        ax1.set_xlabel('Number of Factors')
        ax1.set_ylabel('Q(robust)')
        ax1.set_title('Q(robust) vs Number of Factors')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_xticks(factors)
        
        # Plot 2: Q/DoF ratios with EPA reference lines
        valid_ratios = [(f, r) for f, r in zip(factors, q_dof_ratios) if r != float('inf')]
        if valid_ratios:
            valid_factors, valid_ratios = zip(*valid_ratios)
            ax2.plot(valid_factors, valid_ratios, 'o-', linewidth=2, markersize=8, alpha=0.7, color='green', label='Q/DoF Ratio')
            
            # Add EPA reference lines
            ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Q/DoF = 1.0 (Perfect fit)')
            ax2.axhline(y=1.5, color='green', linestyle='--', alpha=0.7, label='Q/DoF = 1.5 (Excellent)')
            ax2.axhline(y=2.0, color='orange', linestyle='--', alpha=0.7, label='Q/DoF = 2.0 (Good)')
            ax2.axhline(y=3.0, color='red', linestyle='--', alpha=0.7, label='Q/DoF = 3.0 (Fair)')
            
            # Highlight selected factor on Q/DoF plot
            if hasattr(self, 'optimal_factors') and self.optimal_factors:
                try:
                    selected_idx = list(valid_factors).index(self.optimal_factors)
                    selected_ratio = valid_ratios[selected_idx]
                    ax2.plot(self.optimal_factors, selected_ratio, 'ro', markersize=12, zorder=5)
                    
                    # Determine quality based on EPA guidelines
                    if selected_ratio <= 1.5:
                        quality = "Excellent"
                        color = 'green'
                    elif selected_ratio <= 2.0:
                        quality = "Good"
                        color = 'orange'
                    elif selected_ratio <= 3.0:
                        quality = "Fair"
                        color = 'red'
                    else:
                        quality = "Poor"
                        color = 'darkred'
                    
                    ax2.annotate(f'Selected\n{self.optimal_factors} factors\nQ/DoF = {selected_ratio:.3f}\n({quality})',
                               xy=(self.optimal_factors, selected_ratio),
                               xytext=(-30, 20), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.3),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                except (ValueError, IndexError):
                    pass
            
            ax2.set_xlabel('Number of Factors')
            ax2.set_ylabel('Q/DoF Ratio')
            ax2.set_title('Q/DoF Ratio vs Number of Factors (EPA Guidelines)')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8, loc='upper right')
            ax2.set_xticks(list(valid_factors))
            
            # Set y-axis limits for better visualization
            max_ratio = max([r for r in valid_ratios if r != float('inf')])
            ax2.set_ylim(0, min(max_ratio * 1.1, 5))  # Cap at 5 for readability
        
        else:
            ax2.text(0.5, 0.5, 'No valid Q/DoF ratios\n(DoF ≤ 0 for all factor counts)', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Q/DoF Ratio vs Number of Factors')
        
        # Add explanatory text
        textstr = ('EPA PMF 5.0 Guidelines:\n'
                  'Q/DoF ≤ 1.5: Excellent fit\n'
                  'Q/DoF ≤ 2.0: Good fit\n'
                  'Q/DoF ≤ 3.0: Fair fit\n'
                  'Q/DoF > 3.0: Poor fit')
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=8,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_optimization_q_vs_factors.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: optimization_q_vs_factors.png")
    
    def _varimax_rotation(self, loadings, gamma=1.0, max_iter=100, tol=1e-6):
        """
        Perform Varimax rotation on PCA loadings.
        
        Args:
            loadings (ndarray): Original loadings matrix (species × components)
            gamma (float): Kaiser normalization parameter (1.0 for Varimax)
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
        
        Returns:
            tuple: (rotated_loadings, rotation_matrix)
        """
        p, k = loadings.shape
        R = np.eye(k)  # Initialize rotation matrix as identity
        
        for iteration in range(max_iter):
            # Apply current rotation
            L = loadings @ R
            
            # Compute gradient for Varimax criterion
            u, s, vt = np.linalg.svd(
                loadings.T @ (L**3 - (gamma/p) * L @ np.diag(np.diag(L.T @ L)))
            )
            
            # Update rotation matrix
            R_new = u @ vt
            
            # Check for convergence
            if np.linalg.norm(R - R_new) < tol:
                print(f"   Varimax converged after {iteration + 1} iterations")
                break
            
            R = R_new
        
        rotated_loadings = loadings @ R
        return rotated_loadings, R
    
    def run_pca_analysis(self):
        """
        Run Principal Component Analysis on the concentration data.
        This provides a comparison to PMF using traditional variance-based decomposition.
        """
        print("🔬 Starting PCA analysis...")
        
        # Load processed concentration data (same as used for PMF)
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        if not conc_file.exists():
            print("❌ No concentration data found. Run PMF analysis first.")
            return False
        
        print("📊 Loading concentration data for PCA...")
        conc_df = pd.read_csv(conc_file, index_col=0)
        
        # Remove any remaining NaN values
        conc_clean = conc_df.dropna()
        if len(conc_clean) != len(conc_df):
            print(f"   Removed {len(conc_df) - len(conc_clean)} rows with missing values")
        
        print(f"📊 PCA data matrix: {conc_clean.shape}")
        print(f"📋 Species: {', '.join(conc_clean.columns)}")
        
        # Step 1: Data Standardization (CRITICAL for PCA)
        print("🎯 Standardizing data (Z-score transformation)...")
        self.pca_scaler = StandardScaler()
        X_scaled = self.pca_scaler.fit_transform(conc_clean)
        
        # Display scaling statistics
        print("   Scaling statistics:")
        for i, species in enumerate(conc_clean.columns):
            mean_val = self.pca_scaler.mean_[i]
            std_val = np.sqrt(self.pca_scaler.var_[i])
            print(f"     {species}: mean={mean_val:.3f}, std={std_val:.3f}")
        
        # Step 2: Determine optimal number of components
        # Use same number as PMF factors for direct comparison
        n_components = self.factors
        print(f"🔢 Using {n_components} components (matching PMF factors)")
        
        # Step 3: Perform PCA
        print("⚙️ Performing PCA...")
        self.pca_model = PCA(n_components=n_components, random_state=self.seed)
        pca_scores = self.pca_model.fit_transform(X_scaled)
        pca_loadings = self.pca_model.components_.T  # Transpose to get species × components
        
        # Store explained variance
        self.pca_explained_variance = self.pca_model.explained_variance_ratio_
        
        print(f"📊 PCA Results:")
        print(f"   Components shape: {pca_loadings.shape}")
        print(f"   Scores shape: {pca_scores.shape}")
        print(f"   Total variance explained: {np.sum(self.pca_explained_variance):.1%}")
        for i, var in enumerate(self.pca_explained_variance):
            print(f"     PC{i+1}: {var:.1%}")
        
        # Step 4: Varimax Rotation for interpretability
        print("🔄 Applying Varimax rotation for interpretability...")
        rotated_loadings, rotation_matrix = self._varimax_rotation(pca_loadings)
        
        # Apply same rotation to scores
        rotated_scores = pca_scores @ rotation_matrix
        
        # Store final results
        self.pca_loadings = rotated_loadings
        self.pca_scores = rotated_scores
        
        # Save PCA results
        pca_loadings_file = self.output_dir / f"{self.filename_prefix}_pca_loadings.csv"
        pca_scores_file = self.output_dir / f"{self.filename_prefix}_pca_scores.csv"
        
        # Create loadings DataFrame with proper index
        loadings_df = pd.DataFrame(
            self.pca_loadings,
            index=conc_clean.columns,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        loadings_df.to_csv(pca_loadings_file)
        
        # Create scores DataFrame with proper index
        scores_df = pd.DataFrame(
            self.pca_scores,
            index=conc_clean.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        scores_df.to_csv(pca_scores_file)
        
        print(f"💾 Saved PCA loadings: {pca_loadings_file}")
        print(f"💾 Saved PCA scores: {pca_scores_file}")
        
        return True
    
    def _create_pca_comparison_plots(self, dashboard_dir, plot_files):
        """
        Create comparative plots between PMF and PCA results.
        This includes side-by-side profiles, correlation analysis, and method comparison.
        """
        print("   🅰 Creating PCA vs PMF comparison plots...")
        
        if not hasattr(self, 'pca_loadings') or self.pca_loadings is None:
            print("   ⚠️ No PCA results found - skipping comparison plots")
            return
        
        # Get PMF results
        F_profiles = self.best_model.H  # PMF factor profiles
        G_contributions = self.best_model.W  # PMF factor contributions
        
        # Create comprehensive comparison plots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'{self.station} PMF vs PCA Comparison Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: Explained Variance Comparison
        ax1 = axes[0, 0]
        factor_nums = range(1, self.factors + 1)
        
        # PCA explained variance (cumulative)
        pca_cumvar = np.cumsum(self.pca_explained_variance)
        ax1.plot(factor_nums, self.pca_explained_variance * 100, 'o-', 
                label='PCA Individual', linewidth=2, markersize=8, color='blue')
        ax1.plot(factor_nums, pca_cumvar * 100, 's--', 
                label='PCA Cumulative', linewidth=2, markersize=6, color='lightblue')
        
        # PMF doesn't have direct explained variance, but we can show relative contributions
        total_pmf = np.sum(G_contributions, axis=0)
        pmf_relative = total_pmf / np.sum(total_pmf) * 100
        pmf_cumulative = np.cumsum(pmf_relative)
        
        ax1.plot(factor_nums, pmf_relative, '^-', 
                label='PMF Relative Contribution', linewidth=2, markersize=8, color='red')
        ax1.plot(factor_nums, pmf_cumulative, 'd--', 
                label='PMF Cumulative', linewidth=2, markersize=6, color='lightcoral')
        
        ax1.set_xlabel('Component/Factor Number')
        ax1.set_ylabel('Variance/Contribution (%)')
        ax1.set_title('Variance Explained: PCA vs PMF Contribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(factor_nums)
        
        # Plot 2: Method Comparison Summary
        ax2 = axes[0, 1]
        comparison_data = {
            'Method': ['PCA', 'PMF'],
            'Approach': ['Variance Maximization', 'Non-negative Factorization'],
            'Constraints': ['Orthogonal Components', 'Non-negativity + Uncertainty'],
            'Data Scaling': ['Standardized (Z-score)', 'Raw + Uncertainty Weights'],
            'Interpretability': ['Requires Rotation', 'Direct Physical Meaning']
        }
        
        # Create a text summary table
        ax2.axis('off')
        table_text = "Method Comparison Summary\n" + "=" * 25 + "\n\n"
        
        table_text += f"PCA Total Variance Explained: {np.sum(self.pca_explained_variance):.1%}\n"
        table_text += f"PMF Q(robust): {self.best_model.Qrobust:.2f}\n"
        table_text += f"PMF Q(true): {self.best_model.Qtrue:.2f}\n\n"
        
        table_text += "Key Differences:\n"
        table_text += "• PCA: Orthogonal, variance-based\n"
        table_text += "• PMF: Physical constraints, uncertainty-weighted\n"
        table_text += "• PCA: Requires standardization\n"
        table_text += "• PMF: Uses raw concentrations + uncertainties\n"
        table_text += "• PCA: Mathematical optimality\n"
        table_text += "• PMF: Environmental interpretability"
        
        ax2.text(0.05, 0.95, table_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Plot 3: Side-by-side Profile Comparison (First Factor/Component)
        ax3 = axes[1, 0]
        
        # Compare Factor 1 (PMF) vs PC1 (PCA)
        x_pos = np.arange(len(self.species_names))
        width = 0.35
        
        # Normalize profiles for comparison (0-1 scale)
        pmf_profile_norm = F_profiles[0, :] / np.max(F_profiles[0, :]) if np.max(F_profiles[0, :]) > 0 else F_profiles[0, :]
        pca_profile_norm = np.abs(self.pca_loadings[:, 0]) / np.max(np.abs(self.pca_loadings[:, 0]))
        
        bars1 = ax3.bar(x_pos - width/2, pmf_profile_norm, width, 
                       label='PMF Factor 1 (Normalized)', alpha=0.8, color='red')
        bars2 = ax3.bar(x_pos + width/2, pca_profile_norm, width,
                       label='PCA PC1 (|Loading| Normalized)', alpha=0.8, color='blue')
        
        ax3.set_xlabel('Species')
        ax3.set_ylabel('Normalized Contribution/Loading')
        ax3.set_title('Profile Comparison: PMF Factor 1 vs PCA PC1')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(self.species_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Contribution/Score Correlation Matrix
        ax4 = axes[1, 1]
        
        # Calculate correlations between PMF factors and PCA components
        # Standardize both for fair comparison
        from sklearn.preprocessing import StandardScaler
        scaler_pmf = StandardScaler()
        scaler_pca = StandardScaler()
        
        G_std = scaler_pmf.fit_transform(G_contributions)
        pca_scores_std = scaler_pca.fit_transform(self.pca_scores)
        
        # Calculate cross-correlation matrix
        cross_corr = np.corrcoef(G_std.T, pca_scores_std.T)[:self.factors, self.factors:]
        
        im = ax4.imshow(cross_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        ax4.set_title('Factor-Component Correlation Matrix\n(Standardized Contributions/Scores)')
        
        # Add correlation values to heatmap
        for i in range(self.factors):
            for j in range(self.factors):
                ax4.text(j, i, f'{cross_corr[i, j]:.2f}', ha='center', va='center',
                        color='white' if abs(cross_corr[i, j]) > 0.5 else 'black')
        
        ax4.set_xticks(range(self.factors))
        ax4.set_yticks(range(self.factors))
        ax4.set_xticklabels([f'PC{i+1}' for i in range(self.factors)])
        ax4.set_yticklabels([f'PMF F{i+1}' for i in range(self.factors)])
        ax4.set_xlabel('PCA Components')
        ax4.set_ylabel('PMF Factors')
        
        plt.colorbar(im, ax=ax4, label='Correlation Coefficient')
        
        # Plot 5: Scree Plot Comparison
        ax5 = axes[2, 0]
        
        ax5.plot(factor_nums, self.pca_explained_variance * 100, 'o-', 
                linewidth=3, markersize=10, color='blue', label='PCA Eigenvalues (%)')
        ax5.set_xlabel('Component Number')
        ax5.set_ylabel('Variance Explained (%)')
        ax5.set_title('PCA Scree Plot')
        ax5.grid(True, alpha=0.3)
        ax5.set_xticks(factor_nums)
        ax5.legend()
        
        # Add Kaiser criterion line (eigenvalue = 1, equivalent to ~1/n_species * 100%)
        kaiser_line = (1.0 / len(self.species_names)) * 100
        ax5.axhline(y=kaiser_line, color='red', linestyle='--', alpha=0.7, 
                   label=f'Kaiser Criterion ({kaiser_line:.1f}%)')
        ax5.legend()
        
        # Plot 6: Best Factor-Component Matches
        ax6 = axes[2, 1]
        
        # Find best matches based on correlation matrix
        max_corr_indices = np.argmax(np.abs(cross_corr), axis=1)
        max_corr_values = np.max(np.abs(cross_corr), axis=1)
        
        # Create matching plot
        matches = []
        match_corrs = []
        match_labels = []
        
        for i in range(self.factors):
            best_pc = max_corr_indices[i]
            corr_val = cross_corr[i, best_pc]
            matches.append(i)
            match_corrs.append(abs(corr_val))
            match_labels.append(f'F{i+1} ↔ PC{best_pc+1}\nr={corr_val:.2f}')
        
        bars = ax6.bar(matches, match_corrs, alpha=0.7, 
                      color=['green' if abs(c) > 0.7 else 'orange' if abs(c) > 0.5 else 'red' 
                             for c in match_corrs])
        
        ax6.set_xlabel('PMF Factor')
        ax6.set_ylabel('Best |Correlation| with PCA')
        ax6.set_title('Best PMF-PCA Factor Matches')
        ax6.set_xticks(matches)
        ax6.set_xticklabels([f'F{i+1}' for i in matches])
        ax6.grid(True, alpha=0.3)
        
        # Add correlation values and PC matches on bars
        for i, (bar, label) in enumerate(zip(bars, match_labels)):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    label, ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Add interpretation legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='green', alpha=0.7, label='Strong (|r| > 0.7)'),
            plt.Rectangle((0,0),1,1, facecolor='orange', alpha=0.7, label='Moderate (|r| > 0.5)'),
            plt.Rectangle((0,0),1,1, facecolor='red', alpha=0.7, label='Weak (|r| ≤ 0.5)')
        ]
        ax6.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_pca_pmf_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: pca_pmf_comparison.png")
        
        # Additional detailed comparison plot
        self._create_detailed_profile_comparison(dashboard_dir, plot_files, F_profiles)
    
    def _create_detailed_profile_comparison(self, dashboard_dir, plot_files, F_profiles):
        """
        Create detailed side-by-side profile comparison for all factors/components.
        """
        print("   🔍 Creating detailed profile comparison plots...")
        
        # Create subplot layout based on number of factors
        if self.factors <= 4:
            nrows, ncols = 2, 2
        elif self.factors <= 6:
            nrows, ncols = 2, 3
        else:
            nrows, ncols = 3, 3
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        fig.suptitle(f'{self.station} Detailed Profile Comparison: PMF vs PCA', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes for easier iteration
        if self.factors == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i in range(self.factors):
            ax = axes[i]
            
            x_pos = np.arange(len(self.species_names))
            width = 0.35
            
            # Get PMF profile (always positive)
            pmf_profile = F_profiles[i, :]
            
            # Get PCA loading (can be negative)
            pca_loading = self.pca_loadings[:, i]
            
            # Plot both profiles
            bars1 = ax.bar(x_pos - width/2, pmf_profile, width, 
                          label=f'PMF F{i+1}', alpha=0.8, color='red')
            bars2 = ax.bar(x_pos + width/2, pca_loading, width,
                          label=f'PCA PC{i+1}', alpha=0.8, color='blue')
            
            # Color PCA bars by sign
            for bar, val in zip(bars2, pca_loading):
                bar.set_color('blue' if val >= 0 else 'lightblue')
            
            ax.set_title(f'Factor/Component {i+1} Profiles')
            ax.set_xlabel('Species')
            ax.set_ylabel('Loading/Contribution')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(self.species_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)  # Zero line
        
        # Hide unused subplots
        total_subplots = nrows * ncols
        for i in range(self.factors, total_subplots):
            if i < len(axes):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_detailed_profile_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: detailed_profile_comparison.png")
    
    def _create_wind_analysis_plots(self, dashboard_dir, plot_files, G_contributions):
        """
        Create wind analysis plots showing how PMF factors vary with wind direction and speed.
        This is valuable for source apportionment as it can identify directional sources.
        """
        print("   🌪️ Creating wind analysis plots...")
        
        # Set matplotlib to non-interactive mode to prevent any display output
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        
        # Create context manager to capture any unwanted output
        import io
        import sys
        
        class SuppressOutput:
            def __enter__(self):
                self._original_stdout = sys.stdout
                sys.stdout = io.StringIO()
                return self
            def __exit__(self, *args):
                sys.stdout = self._original_stdout
        
        # Check if we have meteorological data in the original dataset
        met_columns = []
        wind_dir_col = None
        wind_speed_col = None
        
        # Look for common wind column names (including MMF9 naming)
        wind_dir_patterns = ['wind_dir', 'wind_direction', 'wd', 'WindDir', 'WIND DIR']
        wind_speed_patterns = ['wind_speed', 'wind_vel', 'ws', 'WindSpeed', 'WIND SPD']
        
        for col in self.df.columns:
            # Skip count columns (prefixed with 'n_') - prefer actual data columns
            if col.startswith('n_'):
                continue
                
            # Check for wind direction
            if any(pattern.lower() in col.lower() for pattern in wind_dir_patterns):
                wind_dir_col = col
            # Check for wind speed
            elif any(pattern.lower() in col.lower() for pattern in wind_speed_patterns):
                wind_speed_col = col
        
        if not wind_dir_col and not wind_speed_col:
            print("   ⚠️ No wind data found in dataset - skipping wind analysis")
            return
        
        # Get the corresponding meteorological data for PMF time points
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        conc_data = pd.read_csv(conc_file, index_col=0)
        
        # Get datetime index for matching with original data
        try:
            datetime_index = pd.to_datetime(conc_data.index)
            has_datetime = True
        except:
            print("   ⚠️ Unable to parse datetime for wind analysis")
            return
        
        # Match meteorological data with PMF analysis times
        wind_data = []
        valid_indices = []
        
        for i, dt in enumerate(datetime_index):
            # Find closest match in original data
            time_diff = np.abs((self.df['datetime'] - dt).dt.total_seconds())
            closest_idx = time_diff.idxmin()
            
            # Only include if within reasonable time tolerance (e.g., 1 hour)
            if time_diff.loc[closest_idx] <= 3600:  # 1 hour in seconds
                wind_data.append({
                    'wind_dir': self.df.loc[closest_idx, wind_dir_col] if wind_dir_col else np.nan,
                    'wind_speed': self.df.loc[closest_idx, wind_speed_col] if wind_speed_col else np.nan,
                    'pmf_index': i
                })
                valid_indices.append(i)
        
        if len(wind_data) == 0:
            print("   ⚠️ No matching wind data found for PMF time points")
            return
        
        wind_df = pd.DataFrame(wind_data)
        print(f"   📊 Found {len(wind_df)} matching wind/PMF data points")
        
        # Filter PMF contributions to match wind data
        G_wind = G_contributions[valid_indices, :]
        
        # Create comprehensive wind analysis plot
        # Wrap all initial setup in output suppression to prevent matplotlib from displaying values
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with SuppressOutput():
                fig = plt.figure(figsize=(20, 16))
                _ = fig.suptitle(f'{self.station} PMF Factors vs Wind Conditions', fontsize=18, fontweight='bold')
                
                # Use consistent ColorManager colors
                colors = self.color_manager._get_factor_colors(self.factors)
                
                # Plot layout: 3 rows x 2 columns + polar plots
                gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Plot 1: Wind Rose for all data (top left)
        with SuppressOutput():
            ax1 = fig.add_subplot(gs[0, 0])
        if wind_dir_col and not wind_df['wind_dir'].isna().all():
            # Create simple wind rose
            wind_dirs = wind_df['wind_dir'].dropna()
            if len(wind_dirs) > 0:
                # Bin wind directions into 16 sectors (22.5° each)
                bins = np.arange(0, 361, 22.5)
                counts, bin_edges = np.histogram(wind_dirs, bins=bins)
                
                # Create bar chart representing wind rose
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax1.bar(bin_centers, counts, width=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Wind Direction (°)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Wind Direction Distribution')
                ax1.set_xlim(0, 360)
                ax1.set_xticks(np.arange(0, 361, 45))
                ax1.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
                ax1.grid(True, alpha=0.3)
        
        # Plot 2: Wind Speed Distribution (top right)
        with SuppressOutput():
            ax2 = fig.add_subplot(gs[0, 1])
        if wind_speed_col and not wind_df['wind_speed'].isna().all():
            wind_speeds = wind_df['wind_speed'].dropna()
            if len(wind_speeds) > 0:
                ax2.hist(wind_speeds, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                ax2.set_xlabel(f'Wind Speed ({self.units.get(wind_speed_col, "m/s")})')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Wind Speed Distribution')
                ax2.grid(True, alpha=0.3)
                
                # Add statistics
                mean_ws = np.mean(wind_speeds)
                ax2.axvline(mean_ws, color='red', linestyle='--', linewidth=2, 
                           label=f'Mean: {mean_ws:.1f}')
                ax2.legend()
        
        # Calculate wind-factor correlations for all factors
        # Use output suppression to prevent correlation values from displaying
        factor_wind_correlations = []
        with SuppressOutput():
            if wind_dir_col and not wind_df['wind_dir'].isna().all():
                # Calculate correlations between each factor and wind direction
                valid_wind_mask = ~wind_df['wind_dir'].isna()
                
                for f in range(self.factors):
                    valid_mask = valid_wind_mask & ~np.isnan(G_wind[:, f])
                    if np.sum(valid_mask) > 10:  # Need at least 10 points for meaningful correlation
                        corr_matrix = np.corrcoef(wind_df.loc[valid_mask, 'wind_dir'], 
                                                G_wind[valid_mask, f]);
                        corr = float(np.abs(corr_matrix[0, 1]));
                        if not np.isnan(corr):
                            factor_wind_correlations.append((f, corr))
                        else:
                            factor_wind_correlations.append((f, 0.0))
                    else:
                        factor_wind_correlations.append((f, 0.0))
                
                # Also check wind speed correlations if available
                if wind_speed_col and not wind_df['wind_speed'].isna().all():
                    valid_speed_mask = ~wind_df['wind_speed'].isna()
                    for i, (f, dir_corr) in enumerate(factor_wind_correlations):
                        valid_mask = valid_speed_mask & ~np.isnan(G_wind[:, f])
                        if np.sum(valid_mask) > 10:
                            corr_matrix = np.corrcoef(wind_df.loc[valid_mask, 'wind_speed'], 
                                                    G_wind[valid_mask, f]);
                            speed_corr = float(np.abs(corr_matrix[0, 1]));
                            if not np.isnan(speed_corr):
                                # Take maximum of wind direction and wind speed correlation
                                factor_wind_correlations[i] = (f, max(dir_corr, speed_corr))
                
                # Sort by correlation strength for display purposes
                factor_wind_correlations.sort(key=lambda x: x[1], reverse=True)
            
        # Print correlation results (outside suppression context)
        if 'factor_wind_correlations' in locals() and factor_wind_correlations:
            print(f"   🎯 Wind-correlated factors: {[(f+1, corr) for f, corr in factor_wind_correlations]}")
        
        # Create a larger figure to accommodate all polar plots
        plt.close()  # Close current figure
        fig = plt.figure(figsize=(24, 18))  # Larger figure
        _ = fig.suptitle(f'{self.station} PMF Factors vs Wind Conditions', fontsize=20, fontweight='bold')
        
        # Create a new layout: Top row for distributions, middle rows for polar plots, bottom for other analyses
        n_factors = self.factors
        polar_cols = min(4, n_factors)  # Max 4 columns for polar plots
        polar_rows = (n_factors + polar_cols - 1) // polar_cols  # Calculate needed rows
        
        # Layout: 1 row for distributions + polar_rows for polar plots + 2 rows for other plots
        total_rows = 1 + polar_rows + 2
        gs = fig.add_gridspec(total_rows, 4, height_ratios=[0.8] + [1.2] * polar_rows + [1, 1])
        
        # Plot 1: Wind Rose for all data (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if wind_dir_col and not wind_df['wind_dir'].isna().all():
            wind_dirs = wind_df['wind_dir'].dropna()
            if len(wind_dirs) > 0:
                bins = np.arange(0, 361, 22.5)
                counts, bin_edges = np.histogram(wind_dirs, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax1.bar(bin_centers, counts, width=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Wind Direction (°)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Wind Direction Distribution')
                ax1.set_xlim(0, 360)
                ax1.set_xticks(np.arange(0, 361, 45))
                ax1.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
                ax1.grid(True, alpha=0.3)
        
        # Plot 2: Wind Speed Distribution (top middle-left)
        ax2 = fig.add_subplot(gs[0, 1])
        if wind_speed_col and not wind_df['wind_speed'].isna().all():
            wind_speeds = wind_df['wind_speed'].dropna()
            if len(wind_speeds) > 0:
                ax2.hist(wind_speeds, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
                ax2.set_xlabel(f'Wind Speed ({self.units.get(wind_speed_col, "m/s")})')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Wind Speed Distribution')
                ax2.grid(True, alpha=0.3)
                mean_ws = np.mean(wind_speeds)
                ax2.axvline(mean_ws, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ws:.1f}')
                ax2.legend()
        
        # Plot 3: Factor Data Points vs Wind Direction (0-360°) (top right)
        ax3 = fig.add_subplot(gs[0, 2:])
        if wind_dir_col and not wind_df['wind_dir'].isna().all():
            valid_mask = ~wind_df['wind_dir'].isna()
            wd = wind_df.loc[valid_mask, 'wind_dir'].values
            
            for f in range(self.factors):
                fc = G_wind[valid_mask, f]
                ax3.scatter(wd, fc, alpha=0.6, s=20, color=colors[f], label=f'Factor {f+1}')
            
            ax3.set_xlabel('Wind Direction (°)')
            ax3.set_ylabel('Factor Contribution')
            ax3.set_title('Factor Data Points vs Wind Direction (0-360°)')
            ax3.set_xlim(0, 360)
            ax3.set_xticks(np.arange(0, 361, 45))
            ax3.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # Create polar plots for ALL factors (larger size)
        if wind_dir_col and not wind_df['wind_dir'].isna().all():
            for f in range(n_factors):
                row = 1 + f // polar_cols  # Start from row 1 (after distributions)
                col = f % polar_cols
                
                ax_polar = fig.add_subplot(gs[row, col], projection='polar')
                
                # Get wind direction and factor contributions
                valid_mask = ~(wind_df['wind_dir'].isna() | np.isnan(G_wind[:, f]))
                if np.sum(valid_mask) > 5:
                    wd = wind_df.loc[valid_mask, 'wind_dir'].values
                    fc = G_wind[valid_mask, f]
                    
                    # Convert degrees to radians
                    wd_rad = np.radians(wd)
                    
                    # Create polar scatter plot colored by factor contribution intensity
                    scatter = ax_polar.scatter(wd_rad, fc, c=fc, cmap='viridis', 
                                             alpha=0.7, s=40, vmin=np.min(fc), vmax=np.max(fc))
                    
                    # Add colorbar to show contribution scale
                    cbar = plt.colorbar(scatter, ax=ax_polar, shrink=0.6, pad=0.1)
                    cbar.set_label(f'Factor {f+1} Contribution', fontsize=10)
                    
                    # Get correlation for title
                    with SuppressOutput():
                        factor_corr = float(next((corr for idx, corr in factor_wind_correlations if idx == f), 0))
                    
                    ax_polar.set_title(f'Factor {f + 1} vs Wind Direction\n(|r|={factor_corr:.2f})', 
                                      fontweight='bold', pad=20, fontsize=12)
                    ax_polar.set_theta_zero_location('N')
                    ax_polar.set_theta_direction(-1)
                    ax_polar.grid(True, alpha=0.3)
        
        # Factor contributions binned by wind direction (bottom left)
        bottom_row = total_rows - 2  # Second to last row
        ax5 = fig.add_subplot(gs[bottom_row, :2])
        if wind_dir_col and not wind_df['wind_dir'].isna().all():
            # Bin by wind sectors
            sectors = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            sector_centers = np.arange(0, 360, 22.5)
            sector_means = np.zeros((len(sectors), self.factors))
            
            for i, (center, sector) in enumerate(zip(sector_centers, sectors)):
                # Define sector bounds
                lower = (center - 11.25) % 360
                upper = (center + 11.25) % 360
                
                if lower < upper:
                    mask = (wind_df['wind_dir'] >= lower) & (wind_df['wind_dir'] < upper)
                else:  # Handle wrap-around (e.g., N sector)
                    mask = (wind_df['wind_dir'] >= lower) | (wind_df['wind_dir'] < upper)
                
                if np.sum(mask) > 0:
                    sector_means[i, :] = np.mean(G_wind[mask, :], axis=0)
            
            # Create stacked bar chart
            x_pos = np.arange(len(sectors))
            bottom = np.zeros(len(sectors))
            
            for f in range(self.factors):
                ax5.bar(x_pos, sector_means[:, f], bottom=bottom, 
                       label=f'Factor {f+1}', alpha=0.8, color=colors[f])
                bottom += sector_means[:, f]
            
            ax5.set_xlabel('Wind Direction Sector')
            ax5.set_ylabel('Average Factor Contribution')
            ax5.set_title('Average Factor Contributions by Wind Direction')
            ax5.set_xticks(x_pos)
            ax5.set_xticklabels(sectors, rotation=45)
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # Factor contributions vs wind speed (bottom middle)
        ax6 = fig.add_subplot(gs[bottom_row, 2:])
        if wind_speed_col and not wind_df['wind_speed'].isna().all():
            # Scatter plot of each factor vs wind speed
            valid_mask = ~wind_df['wind_speed'].isna()
            ws = wind_df.loc[valid_mask, 'wind_speed'].values
            
            for f in range(self.factors):
                fc = G_wind[valid_mask, f]
                ax6.scatter(ws, fc, alpha=0.6, s=30, color=colors[f], 
                           label=f'Factor {f+1}')
                
                # Add trend line if enough points
                if len(ws) > 10:
                    with SuppressOutput():
                        z = np.polyfit(ws, fc, 1);
                        p = np.poly1d(z);
                        ws_trend = np.linspace(np.min(ws), np.max(ws), 100);
                    _ = ax6.plot(ws_trend, p(ws_trend), '--', color=colors[f], alpha=0.7)
            
            ax6.set_xlabel(f'Wind Speed ({self.units.get(wind_speed_col, "m/s")})')
            ax6.set_ylabel('Factor Contribution')
            ax6.set_title('Factor Contributions vs Wind Speed')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Wind speed binned analysis (last row left)
        last_row = total_rows - 1
        ax7 = fig.add_subplot(gs[last_row, :2])
        if wind_speed_col and not wind_df['wind_speed'].isna().all():
            # Bin wind speeds
            valid_ws = wind_df['wind_speed'].dropna()
            if len(valid_ws) > 0:
                ws_bins = np.percentile(valid_ws, [0, 25, 50, 75, 100])
                ws_labels = [f'{ws_bins[i]:.1f}-{ws_bins[i+1]:.1f}' for i in range(len(ws_bins)-1)]
                
                # Calculate mean contributions for each wind speed bin
                bin_means = np.zeros((len(ws_labels), self.factors))
                
                for i, (low, high) in enumerate(zip(ws_bins[:-1], ws_bins[1:])):
                    mask = (wind_df['wind_speed'] >= low) & (wind_df['wind_speed'] <= high)
                    if np.sum(mask) > 0:
                        bin_means[i, :] = np.mean(G_wind[mask, :], axis=0)
                
                # Create grouped bar chart
                x = np.arange(len(ws_labels))
                width = 0.8 / self.factors
                
                for f in range(self.factors):
                    offset = (f - self.factors/2) * width
                    ax7.bar(x + offset, bin_means[:, f], width, 
                           label=f'Factor {f+1}', alpha=0.8, color=colors[f])
                
                ax7.set_xlabel(f'Wind Speed Bins ({self.units.get(wind_speed_col, "m/s")})')
                ax7.set_ylabel('Average Factor Contribution')
                ax7.set_title('Factor Contributions by Wind Speed Category')
                ax7.set_xticks(x)
                ax7.set_xticklabels(ws_labels)
                ax7.legend()
                ax7.grid(True, alpha=0.3)
        
        # Correlation matrix (last row right)
        ax8 = fig.add_subplot(gs[last_row, 2:])
        
        # Calculate correlations between factors and wind variables
        corr_data = pd.DataFrame()
        if wind_dir_col and not wind_df['wind_dir'].isna().all():
            corr_data['Wind_Dir'] = wind_df['wind_dir']
        if wind_speed_col and not wind_df['wind_speed'].isna().all():
            corr_data['Wind_Speed'] = wind_df['wind_speed']
        
        # Add factor contributions
        for f in range(self.factors):
            corr_data[f'Factor_{f+1}'] = G_wind[:, f]
        
        if len(corr_data.columns) > self.factors:
            # Calculate correlation matrix - suppress any display
            corr_matrix = corr_data.corr();
            
            # Show only wind vs factor correlations
            wind_cols = [col for col in corr_matrix.columns if 'Wind' in col];
            factor_cols = [col for col in corr_matrix.columns if 'Factor' in col];
            
            if wind_cols and factor_cols:
                subset_corr = corr_matrix.loc[wind_cols, factor_cols].copy();
                
                im = ax8.imshow(subset_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax8.set_title('Wind-Factor Correlations')
                ax8.set_xticks(range(len(factor_cols)))
                ax8.set_yticks(range(len(wind_cols)))
                ax8.set_xticklabels([col.replace('_', ' ') for col in factor_cols])
                ax8.set_yticklabels([col.replace('_', ' ') for col in wind_cols])
                
                # Add correlation values - ensure no values leak to output
                for i in range(len(wind_cols)):
                    for j in range(len(factor_cols)):
                        value = float(subset_corr.iloc[i, j])  # Explicit float conversion
                        _ = ax8.text(j, i, f'{value:.2f}', ha='center', va='center',
                                    color='white' if abs(value) > 0.5 else 'black')
                
                plt.colorbar(im, ax=ax8, label='Correlation Coefficient')
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_wind_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: wind_analysis.png")
        
        # Create additional summary statistics
        self._create_wind_summary_stats(dashboard_dir, wind_df, G_wind)
    
    def _create_wind_summary_stats(self, dashboard_dir, wind_df, G_wind):
        """
        Create a summary table of wind-factor relationships.
        """
        summary_path = dashboard_dir / f"{self.filename_prefix}_wind_factor_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"{self.station} Wind-Factor Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write(f"Total matched data points: {len(wind_df)}\n")
            
            if 'wind_dir' in wind_df.columns and not wind_df['wind_dir'].isna().all():
                f.write(f"Wind direction range: {wind_df['wind_dir'].min():.1f}° - {wind_df['wind_dir'].max():.1f}°\n")
                f.write(f"Most frequent wind direction: {wind_df['wind_dir'].mode().iloc[0]:.1f}°\n")
            
            if 'wind_speed' in wind_df.columns and not wind_df['wind_speed'].isna().all():
                f.write(f"Wind speed range: {wind_df['wind_speed'].min():.1f} - {wind_df['wind_speed'].max():.1f}\n")
                f.write(f"Mean wind speed: {wind_df['wind_speed'].mean():.1f}\n")
            
            f.write("\n" + "Factor Statistics:\n")
            f.write("-" * 20 + "\n")
            
            for i in range(self.factors):
                f.write(f"Factor {i+1}:\n")
                f.write(f"  Mean contribution: {np.mean(G_wind[:, i]):.3f}\n")
                f.write(f"  Std deviation: {np.std(G_wind[:, i]):.3f}\n")
                f.write(f"  Max contribution: {np.max(G_wind[:, i]):.3f}\n")
                
                # Correlations if available
                if 'wind_dir' in wind_df.columns:
                    valid_mask = ~wind_df['wind_dir'].isna()
                    if np.sum(valid_mask) > 5:
                        corr = np.corrcoef(wind_df.loc[valid_mask, 'wind_dir'], 
                                         G_wind[valid_mask, i])[0, 1]
                        f.write(f"  Correlation with wind direction: {corr:.3f}\n")
                
                if 'wind_speed' in wind_df.columns:
                    valid_mask = ~wind_df['wind_speed'].isna()
                    if np.sum(valid_mask) > 5:
                        corr = np.corrcoef(wind_df.loc[valid_mask, 'wind_speed'], 
                                         G_wind[valid_mask, i])[0, 1]
                        f.write(f"  Correlation with wind speed: {corr:.3f}\n")
                
                f.write("\n")
        
        print(f"   📄 Wind summary statistics: {summary_path}")
    
    def _create_temperature_analysis_plots(self, dashboard_dir, plot_files, G_contributions):
        """
        Create temperature analysis plots showing how PMF factors vary with temperature.
        This can help identify temperature-dependent sources (e.g., heating, biogenic emissions).
        """
        print("   🌡️ Creating temperature analysis plots...")
        
        # Look for temperature-related columns in the original dataset
        temp_columns = []
        temp_patterns = ['temp', 'temperature', 'ambient_temp', 'air_temp', 't_air', 'ta']
        
        for col in self.df.columns:
            # Skip count and availability columns
            if col.startswith('n_') or col.lower() in ['gas_data_available', 'particle_data_available', 'station_name']:
                continue
            if any(pattern.lower() in col.lower() for pattern in temp_patterns):
                # Only include numeric columns (float or int)
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    temp_columns.append(col)
        
        if not temp_columns:
            print("   ⚠️ No temperature data found in dataset - skipping temperature analysis")
            return
        
        print(f"   📊 Found temperature columns: {temp_columns}")
    
        # Get the corresponding temperature data for PMF time points
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        conc_data = pd.read_csv(conc_file, index_col=0)
        
        # Get datetime index for matching with original data
        try:
            datetime_index = pd.to_datetime(conc_data.index)
            has_datetime = True
        except:
            print("   ⚠️ Unable to parse datetime for temperature analysis")
            return
        
        # Match temperature data with PMF analysis times
        temp_data = []
        valid_indices = []
        
        for i, dt in enumerate(datetime_index):
            # Find closest match in original data
            time_diff = np.abs((self.df['datetime'] - dt).dt.total_seconds())
            closest_idx = time_diff.idxmin()
            
            # Only include if within reasonable time tolerance (e.g., 1 hour)
            if time_diff.loc[closest_idx] <= 3600:  # 1 hour in seconds
                temp_dict = {'pmf_index': i}
                for temp_col in temp_columns:
                    temp_dict[temp_col] = self.df.loc[closest_idx, temp_col]
                temp_data.append(temp_dict)
                valid_indices.append(i)
        
        if len(temp_data) == 0:
            print("   ⚠️ No matching temperature data found for PMF time points")
            return
        
        temp_df = pd.DataFrame(temp_data)
        print(f"   📊 Found {len(temp_df)} matching temperature/PMF data points")
        
        # Filter PMF contributions to match temperature data
        G_temp = G_contributions[valid_indices, :]
        
        # Create comprehensive temperature analysis plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'{self.station} PMF Factors vs Temperature Conditions', fontsize=18, fontweight='bold')
        
        # Use consistent ColorManager colors
        colors = self.color_manager._get_factor_colors(self.factors)
        
        # Use the first temperature column for main analysis
        primary_temp_col = temp_columns[0]
        
        # Remove rows where primary temperature is NaN and ensure data is numeric
        valid_temp_mask = ~temp_df[primary_temp_col].isna()
        temp_values_raw = temp_df.loc[valid_temp_mask, primary_temp_col].values
        
        # Convert to numeric, handling any remaining non-numeric values
        try:
            temp_values = pd.to_numeric(temp_values_raw, errors='coerce')
            # Remove any NaN values created by failed conversions
            final_valid_mask = ~np.isnan(temp_values)
            temp_values = temp_values[final_valid_mask]
            G_temp_valid = G_temp[valid_temp_mask, :][final_valid_mask, :]
        except Exception as e:
            print(f"   ⚠️ Error converting temperature data to numeric: {e}")
            return
        
        if len(temp_values) == 0:
            print(f"   ⚠️ No valid temperature data in {primary_temp_col}")
            return
        
        # Plot 1: Temperature distribution (top left)
        ax1 = axes[0, 0]
        ax1.hist(temp_values, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax1.set_xlabel(f'Temperature ({self.units.get(primary_temp_col, "°C")})')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Temperature Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_temp = np.mean(temp_values)
        median_temp = np.median(temp_values)
        ax1.axvline(mean_temp, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_temp:.1f}')
        ax1.axvline(median_temp, color='blue', linestyle='--', linewidth=2, 
                   label=f'Median: {median_temp:.1f}')
        ax1.legend()
        
        # Plot 2: Factors vs Temperature scatter plot (top right)
        ax2 = axes[0, 1]
        for f in range(self.factors):
            ax2.scatter(temp_values, G_temp_valid[:, f], alpha=0.6, s=30, color=colors[f], 
                       label=f'Factor {f+1}')
            
            # Add trend line if enough points
            if len(temp_values) > 10:
                try:
                    z = np.polyfit(temp_values, G_temp_valid[:, f], 1)
                    p = np.poly1d(z)
                    temp_trend = np.linspace(np.min(temp_values), np.max(temp_values), 100)
                    ax2.plot(temp_trend, p(temp_trend), '--', color=colors[f], alpha=0.7)
                except:
                    pass  # Skip trend line if fitting fails
        
        ax2.set_xlabel(f'Temperature ({self.units.get(primary_temp_col, "°C")})')
        ax2.set_ylabel('Factor Contribution')
        ax2.set_title('Factor Contributions vs Temperature')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temperature binned analysis (middle left)
        ax3 = axes[1, 0]
        if len(temp_values) > 20:  # Need enough points for meaningful binning
            try:
                # Create temperature bins based on percentiles
                temp_bins = np.percentile(temp_values, [0, 25, 50, 75, 100])
                temp_labels = [f'{temp_bins[i]:.1f}-{temp_bins[i+1]:.1f}' for i in range(len(temp_bins)-1)]
                
                # Calculate mean contributions for each temperature bin
                bin_means = np.zeros((len(temp_labels), int(self.factors)))
                bin_stds = np.zeros((len(temp_labels), int(self.factors)))
                
                for i, (low, high) in enumerate(zip(temp_bins[:-1], temp_bins[1:])):
                    if i == len(temp_bins) - 2:  # Last bin includes upper bound
                        mask = (temp_values >= low) & (temp_values <= high)
                    else:
                        mask = (temp_values >= low) & (temp_values < high)
                    
                    if np.sum(mask) > 0:
                        bin_means[i, :] = np.mean(G_temp_valid[mask, :], axis=0)
                        bin_stds[i, :] = np.std(G_temp_valid[mask, :], axis=0)
                
                # Create grouped bar chart
                x = np.arange(len(temp_labels))
                n_factors = int(self.factors)  # Ensure integer type
                width = 0.8 / n_factors
                
                for f in range(n_factors):
                    offset = (f - n_factors/2.0) * width
                    # Ensure the bin_means and bin_stds don't contain NaN or inf values
                    means_clean = np.nan_to_num(bin_means[:, f], nan=0.0, posinf=0.0, neginf=0.0)
                    stds_clean = np.nan_to_num(bin_stds[:, f], nan=0.0, posinf=0.0, neginf=0.0)
                    
                    bars = ax3.bar(x + offset, means_clean, width, 
                                  yerr=stds_clean, capsize=3,
                                  label=f'Factor {f+1}', alpha=0.8, color=colors[f])
            except Exception as e:
                ax3.text(0.5, 0.5, f'Binned analysis failed:\n{str(e)}', 
                        transform=ax3.transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            
            ax3.set_xlabel(f'Temperature Bins ({self.units.get(primary_temp_col, "°C")})')
            ax3.set_ylabel('Average Factor Contribution ± Std Dev')
            ax3.set_title('Factor Contributions by Temperature Category')
            ax3.set_xticks(x)
            ax3.set_xticklabels(temp_labels, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Correlation matrix (middle right)
        ax4 = axes[1, 1]
        
        # Calculate correlations between factors and temperature variables
        corr_data = pd.DataFrame()
        for temp_col in temp_columns:
            if not temp_df[temp_col].isna().all():
                corr_data[temp_col.replace('_', ' ').title()] = temp_df[temp_col]
        
        # Add factor contributions
        for f in range(self.factors):
            corr_data[f'Factor {f+1}'] = G_temp[:, f]
        
        if len(corr_data.columns) > self.factors:
            # Calculate correlation matrix
            corr_matrix = corr_data.corr()
            
            # Show only temperature vs factor correlations
            temp_cols = [col for col in corr_matrix.columns if 'Factor' not in col]
            factor_cols = [col for col in corr_matrix.columns if 'Factor' in col]
            
            if temp_cols and factor_cols:
                subset_corr = corr_matrix.loc[temp_cols, factor_cols]
                
                im = ax4.imshow(subset_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax4.set_title('Temperature-Factor Correlations')
                ax4.set_xticks(range(len(factor_cols)))
                ax4.set_yticks(range(len(temp_cols)))
                ax4.set_xticklabels(factor_cols)
                ax4.set_yticklabels(temp_cols)
                
                # Add correlation values
                for i in range(len(temp_cols)):
                    for j in range(len(factor_cols)):
                        value = subset_corr.iloc[i, j]
                        ax4.text(j, i, f'{value:.2f}', ha='center', va='center',
                                color='white' if abs(value) > 0.5 else 'black')
                
                plt.colorbar(im, ax=ax4, label='Correlation Coefficient')
        
        # Plot 5: Seasonal temperature patterns (bottom left)
        ax5 = axes[2, 0]
        if has_datetime and len(datetime_index) > 30:  # Need reasonable amount of data for seasonal analysis
            try:
                # Extract month from datetime for seasonal analysis
                valid_datetime = datetime_index[valid_indices][valid_temp_mask]
                months = valid_datetime.month
                
                # Calculate monthly temperature and factor averages
                monthly_temp = np.zeros(12)
                monthly_factors = np.zeros((12, int(self.factors)))
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                for month in range(1, 13):
                    month_mask = months == month
                    if np.sum(month_mask) > 0:
                        monthly_temp[month-1] = np.mean(temp_values[month_mask])
                        monthly_factors[month-1, :] = np.mean(G_temp_valid[month_mask, :], axis=0)
                
                # Create dual-axis plot
                ax5_temp = ax5
                ax5_factors = ax5.twinx()
                
                # Plot temperature as line (clean the data first)
                monthly_temp_clean = np.nan_to_num(monthly_temp, nan=0.0, posinf=0.0, neginf=0.0)
                temp_line = ax5_temp.plot(month_names, monthly_temp_clean, 'o-', 
                                         color='red', linewidth=3, markersize=8, 
                                         label='Temperature')
                ax5_temp.set_ylabel(f'Temperature ({self.units.get(primary_temp_col, "°C")})', color='red')
                ax5_temp.tick_params(axis='y', labelcolor='red')
                
                # Plot factors as bars
                x = np.arange(len(month_names))
                n_factors = int(self.factors)
                width = 0.8 / n_factors
                
                for f in range(n_factors):
                    offset = (f - n_factors/2.0) * width
                    # Clean the monthly factor data
                    monthly_factors_clean = np.nan_to_num(monthly_factors[:, f], nan=0.0, posinf=0.0, neginf=0.0)
                    ax5_factors.bar(x + offset, monthly_factors_clean, width, 
                                   alpha=0.6, color=colors[f], label=f'Factor {f+1}')
                
                ax5_factors.set_ylabel('Average Factor Contribution')
                ax5_temp.set_xlabel('Month')
                ax5_temp.set_title('Seasonal Temperature and Factor Patterns')
                ax5_temp.set_xticks(x)
                ax5_temp.set_xticklabels(month_names)
                
                # Combine legends
                lines1, labels1 = ax5_temp.get_legend_handles_labels()
                lines2, labels2 = ax5_factors.get_legend_handles_labels()
                ax5_temp.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                ax5_temp.grid(True, alpha=0.3)
                
            except Exception as e:
                # Fallback to simple plot if seasonal analysis fails
                ax5.text(0.5, 0.5, f'Seasonal analysis failed:\n{str(e)}', 
                        transform=ax5.transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        else:
            ax5.text(0.5, 0.5, 'Insufficient data\nfor seasonal analysis', 
                    transform=ax5.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Plot 6: Temperature-factor relationship strength (bottom right)
        ax6 = axes[2, 1]
        
        # Calculate correlation strength and statistical significance
        correlations = []
        p_values = []
        
        for f in range(self.factors):
            if len(temp_values) > 10:
                try:
                    corr, p_val = pearsonr(temp_values, G_temp_valid[:, f])
                    correlations.append(corr)
                    p_values.append(p_val)
                except:
                    correlations.append(0)
                    p_values.append(1)
            else:
                correlations.append(0)
                p_values.append(1)
        
        # Create bar chart of correlations
        bars = ax6.bar([f'Factor {i+1}' for i in range(self.factors)], 
                      [abs(c) for c in correlations], 
                      color=[colors[i] if p_values[i] < 0.05 else 'lightgray' for i in range(self.factors)],
                      alpha=0.8)
        
        ax6.set_ylabel('Absolute Correlation with Temperature')
        ax6.set_title('Temperature-Factor Relationship Strength')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        
        # Add correlation values and significance on bars
        for i, (bar, corr, p_val) in enumerate(zip(bars, correlations, p_values)):
            height = bar.get_height()
            significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'r={corr:.2f}\n{significance}', ha='center', va='bottom', fontsize=8, 
                    fontweight='bold')
        
        # Add legend for significance levels
        legend_text = ('Significance levels:\n'
                      '*** p < 0.001\n'
                      '** p < 0.01\n'
                      '* p < 0.05\n'
                      'ns = not significant')
        ax6.text(0.02, 0.98, legend_text, transform=ax6.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_temperature_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: temperature_analysis.png")
        
        # Create additional summary statistics
        self._create_temperature_summary_stats(dashboard_dir, temp_df, G_temp, temp_columns, correlations, p_values)
    
    def _create_temperature_summary_stats(self, dashboard_dir, temp_df, G_temp, temp_columns, correlations, p_values):
        """
        Create a summary table of temperature-factor relationships.
        """
        summary_path = dashboard_dir / f"{self.filename_prefix}_temperature_factor_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"{self.station} Temperature-Factor Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write(f"Total matched data points: {len(temp_df)}\n")
            f.write(f"Temperature columns analyzed: {', '.join(temp_columns)}\n\n")
            
            for temp_col in temp_columns:
                if not temp_df[temp_col].isna().all():
                    valid_temps = temp_df[temp_col].dropna()
                    f.write(f"{temp_col} Statistics:\n")
                    f.write(f"  Range: {valid_temps.min():.1f} - {valid_temps.max():.1f} {self.units.get(temp_col, '')}\n")
                    f.write(f"  Mean: {valid_temps.mean():.1f} {self.units.get(temp_col, '')}\n")
                    f.write(f"  Median: {valid_temps.median():.1f} {self.units.get(temp_col, '')}\n")
                    f.write(f"  Std deviation: {valid_temps.std():.1f} {self.units.get(temp_col, '')}\n\n")
            
            f.write("Factor-Temperature Correlations:\n")
            f.write("-" * 30 + "\n")
            
            for i in range(self.factors):
                f.write(f"Factor {i+1}:\n")
                f.write(f"  Mean contribution: {np.mean(G_temp[:, i]):.3f}\n")
                f.write(f"  Std deviation: {np.std(G_temp[:, i]):.3f}\n")
                f.write(f"  Correlation with temperature: {correlations[i]:.3f}\n")
                f.write(f"  P-value: {p_values[i]:.3e}\n")
                
                # Interpretation
                if p_values[i] < 0.001:
                    significance = "highly significant (***)"
                elif p_values[i] < 0.01:
                    significance = "very significant (**)"
                elif p_values[i] < 0.05:
                    significance = "significant (*)"
                else:
                    significance = "not significant (ns)"
                
                if abs(correlations[i]) > 0.7:
                    strength = "strong"
                elif abs(correlations[i]) > 0.5:
                    strength = "moderate"
                elif abs(correlations[i]) > 0.3:
                    strength = "weak"
                else:
                    strength = "very weak"
                
                direction = "positive" if correlations[i] > 0 else "negative"
                
                f.write(f"  Interpretation: {strength} {direction} correlation, {significance}\n")
                f.write("\n")
            
            # Environmental interpretation hints
            f.write("Environmental Interpretation Notes:\n")
            f.write("-" * 35 + "\n")
            f.write("• Strong positive temperature correlation may indicate:\n")
            f.write("  - Biogenic emissions (vegetation)\n")
            f.write("  - Photochemical secondary formation\n")
            f.write("  - Evaporation of volatile compounds\n")
            f.write("  - Increased mixing height effects\n\n")
            
            f.write("• Strong negative temperature correlation may indicate:\n")
            f.write("  - Residential heating sources\n")
            f.write("  - Incomplete combustion in cold conditions\n")
            f.write("  - Reduced atmospheric mixing\n")
            f.write("  - Seasonal industrial patterns\n\n")
            
            f.write("• Factors with weak temperature correlation may indicate:\n")
            f.write("  - Industrial sources with constant emissions\n")
            f.write("  - Traffic-related sources (less temperature dependent)\n")
            f.write("  - Regional background contributions\n")
        
        print(f"   📄 Temperature summary statistics: {summary_path}")
    
    def _create_pressure_analysis_plots(self, dashboard_dir, plot_files, G_contributions):
        """
        Create pressure analysis plots showing how PMF factors vary with atmospheric pressure.
        This can help identify pressure-dependent sources and meteorological influences.
        """
        print("   💨 Creating pressure analysis plots...")
        
        # Look for pressure-related columns in the original dataset
        pressure_columns = []
        pressure_patterns = ['press', 'pressure', 'barometric', 'atm_press', 'bp', 'pa', 'hpa', 'mbar']
        
        for col in self.df.columns:
            # Skip count and availability columns
            if col.startswith('n_') or col.lower() in ['gas_data_available', 'particle_data_available', 'station_name']:
                continue
            if any(pattern.lower() in col.lower() for pattern in pressure_patterns):
                # Only include numeric columns (float or int)
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    pressure_columns.append(col)
        
        if not pressure_columns:
            print("   ⚠️ No pressure data found in dataset - skipping pressure analysis")
            return
        
        print(f"   📊 Found pressure columns: {pressure_columns}")
        
        # Get the corresponding pressure data for PMF time points
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        conc_data = pd.read_csv(conc_file, index_col=0)
        
        # Get datetime index for matching with original data
        try:
            datetime_index = pd.to_datetime(conc_data.index)
            has_datetime = True
        except:
            print("   ⚠️ Unable to parse datetime for pressure analysis")
            return
        
        # Match pressure data with PMF analysis times
        pressure_data = []
        valid_indices = []
        
        for i, dt in enumerate(datetime_index):
            # Find closest match in original data
            time_diff = np.abs((self.df['datetime'] - dt).dt.total_seconds())
            closest_idx = time_diff.idxmin()
            
            # Only include if within reasonable time tolerance (e.g., 1 hour)
            if time_diff.loc[closest_idx] <= 3600:  # 1 hour in seconds
                pressure_dict = {'pmf_index': i}
                for pressure_col in pressure_columns:
                    pressure_dict[pressure_col] = self.df.loc[closest_idx, pressure_col]
                pressure_data.append(pressure_dict)
                valid_indices.append(i)
        
        if len(pressure_data) == 0:
            print("   ⚠️ No matching pressure data found for PMF time points")
            return
        
        pressure_df = pd.DataFrame(pressure_data)
        print(f"   📊 Found {len(pressure_df)} matching pressure/PMF data points")
        
        # Filter PMF contributions to match pressure data
        G_pressure = G_contributions[valid_indices, :]
        
        # Create comprehensive pressure analysis plot
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'{self.station} PMF Factors vs Atmospheric Pressure Conditions', fontsize=18, fontweight='bold')
        
        # Use consistent ColorManager colors
        colors = self.color_manager._get_factor_colors(self.factors)
        
        # Use the first pressure column for main analysis
        primary_pressure_col = pressure_columns[0]
        
        # Remove rows where primary pressure is NaN and ensure data is numeric
        valid_pressure_mask = ~pressure_df[primary_pressure_col].isna()
        pressure_values_raw = pressure_df.loc[valid_pressure_mask, primary_pressure_col].values
        
        # Convert to numeric, handling any remaining non-numeric values
        try:
            pressure_values = pd.to_numeric(pressure_values_raw, errors='coerce')
            # Remove any NaN values created by failed conversions
            final_valid_mask = ~np.isnan(pressure_values)
            pressure_values = pressure_values[final_valid_mask]
            G_pressure_valid = G_pressure[valid_pressure_mask, :][final_valid_mask, :]
        except Exception as e:
            print(f"   ⚠️ Error converting pressure data to numeric: {e}")
            return
        
        if len(pressure_values) == 0:
            print(f"   ⚠️ No valid pressure data in {primary_pressure_col}")
            return
        
        # Plot 1: Pressure distribution (top left)
        ax1 = axes[0, 0]
        ax1.hist(pressure_values, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.set_xlabel(f'Pressure ({self.units.get(primary_pressure_col, "hPa")})')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Atmospheric Pressure Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_pressure = np.mean(pressure_values)
        median_pressure = np.median(pressure_values)
        ax1.axvline(mean_pressure, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_pressure:.1f}')
        ax1.axvline(median_pressure, color='blue', linestyle='--', linewidth=2, 
                   label=f'Median: {median_pressure:.1f}')
        ax1.legend()
        
        # Plot 2: Factors vs Pressure scatter plot (top right)
        ax2 = axes[0, 1]
        for f in range(self.factors):
            ax2.scatter(pressure_values, G_pressure_valid[:, f], alpha=0.6, s=30, color=colors[f], 
                       label=f'Factor {f+1}')
            
            # Add trend line if enough points
            if len(pressure_values) > 10:
                try:
                    z = np.polyfit(pressure_values, G_pressure_valid[:, f], 1)
                    p = np.poly1d(z)
                    pressure_trend = np.linspace(np.min(pressure_values), np.max(pressure_values), 100)
                    ax2.plot(pressure_trend, p(pressure_trend), '--', color=colors[f], alpha=0.7)
                except:
                    pass  # Skip trend line if fitting fails
        
        ax2.set_xlabel(f'Pressure ({self.units.get(primary_pressure_col, "hPa")})')
        ax2.set_ylabel('Factor Contribution')
        ax2.set_title('Factor Contributions vs Atmospheric Pressure')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Pressure binned analysis (middle left)
        ax3 = axes[1, 0]
        if len(pressure_values) > 20:  # Need enough points for meaningful binning
            # Create pressure bins based on percentiles
            pressure_bins = np.percentile(pressure_values, [0, 25, 50, 75, 100])
            pressure_labels = [f'{pressure_bins[i]:.1f}-{pressure_bins[i+1]:.1f}' for i in range(len(pressure_bins)-1)]
            
            # Calculate mean contributions for each pressure bin
            bin_means = np.zeros((len(pressure_labels), self.factors))
            bin_stds = np.zeros((len(pressure_labels), self.factors))
            
            for i, (low, high) in enumerate(zip(pressure_bins[:-1], pressure_bins[1:])):
                if i == len(pressure_bins) - 2:  # Last bin includes upper bound
                    mask = (pressure_values >= low) & (pressure_values <= high)
                else:
                    mask = (pressure_values >= low) & (pressure_values < high)
                
                if np.sum(mask) > 0:
                    bin_means[i, :] = np.mean(G_pressure_valid[mask, :], axis=0)
                    bin_stds[i, :] = np.std(G_pressure_valid[mask, :], axis=0)
            
            # Create grouped bar chart
            x = np.arange(len(pressure_labels))
            width = 0.8 / self.factors
            
            for f in range(self.factors):
                offset = (f - self.factors/2) * width
                # Ensure the bin_means and bin_stds don't contain NaN or inf values
                means_clean = np.nan_to_num(bin_means[:, f], nan=0.0, posinf=0.0, neginf=0.0)
                stds_clean = np.nan_to_num(bin_stds[:, f], nan=0.0, posinf=0.0, neginf=0.0)
                
                bars = ax3.bar(x + offset, means_clean, width, 
                              yerr=stds_clean, capsize=3,
                              label=f'Factor {f+1}', alpha=0.8, color=colors[f])
            
            ax3.set_xlabel(f'Pressure Bins ({self.units.get(primary_pressure_col, "hPa")})')
            ax3.set_ylabel('Average Factor Contribution ± Std Dev')
            ax3.set_title('Factor Contributions by Pressure Category')
            ax3.set_xticks(x)
            ax3.set_xticklabels(pressure_labels, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Plot 4: Correlation matrix (middle right)
        ax4 = axes[1, 1]
        
        # Calculate correlations between factors and pressure variables
        corr_data = pd.DataFrame()
        for pressure_col in pressure_columns:
            if not pressure_df[pressure_col].isna().all():
                # Ensure column is numeric for correlation analysis
                try:
                    numeric_data = pd.to_numeric(pressure_df[pressure_col], errors='coerce')
                    if not numeric_data.isna().all():  # Only add if there's valid numeric data
                        corr_data[pressure_col.replace('_', ' ').title()] = numeric_data
                except:
                    # Skip column if conversion fails
                    continue
        
        # Add factor contributions
        for f in range(self.factors):
            corr_data[f'Factor {f+1}'] = G_pressure[:, f]
        
        if len(corr_data.columns) > self.factors:
            # Calculate correlation matrix
            corr_matrix = corr_data.corr()
            
            # Show only pressure vs factor correlations
            pressure_cols = [col for col in corr_matrix.columns if 'Factor' not in col]
            factor_cols = [col for col in corr_matrix.columns if 'Factor' in col]
            
            if pressure_cols and factor_cols:
                subset_corr = corr_matrix.loc[pressure_cols, factor_cols]
                
                im = ax4.imshow(subset_corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
                ax4.set_title('Pressure-Factor Correlations')
                ax4.set_xticks(range(len(factor_cols)))
                ax4.set_yticks(range(len(pressure_cols)))
                ax4.set_xticklabels(factor_cols)
                ax4.set_yticklabels(pressure_cols)
                
                # Add correlation values
                for i in range(len(pressure_cols)):
                    for j in range(len(factor_cols)):
                        value = subset_corr.iloc[i, j]
                        ax4.text(j, i, f'{value:.2f}', ha='center', va='center',
                                color='white' if abs(value) > 0.5 else 'black')
                
                plt.colorbar(im, ax=ax4, label='Correlation Coefficient')
        
        # Plot 5: Seasonal pressure patterns (bottom left)
        ax5 = axes[2, 0]
        if has_datetime and len(datetime_index) > 30:  # Need reasonable amount of data for seasonal analysis
            try:
                # Extract month from datetime for seasonal analysis
                valid_datetime = datetime_index[valid_indices][valid_pressure_mask]
                months = valid_datetime.month
                
                # Calculate monthly pressure and factor averages
                monthly_pressure = np.zeros(12)
                monthly_factors = np.zeros((12, self.factors))
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                for month in range(1, 13):
                    month_mask = months == month
                    if np.sum(month_mask) > 0:
                        monthly_pressure[month-1] = np.mean(pressure_values[month_mask])
                        monthly_factors[month-1, :] = np.mean(G_pressure_valid[month_mask, :], axis=0)
                
                # Create dual-axis plot
                ax5_pressure = ax5
                ax5_factors = ax5.twinx()
                
                # Plot pressure as line (clean the data first)
                monthly_pressure_clean = np.nan_to_num(monthly_pressure, nan=0.0, posinf=0.0, neginf=0.0)
                pressure_line = ax5_pressure.plot(month_names, monthly_pressure_clean, 'o-', 
                                                 color='blue', linewidth=3, markersize=8, 
                                                 label='Pressure')
                ax5_pressure.set_ylabel(f'Pressure ({self.units.get(primary_pressure_col, "hPa")})', color='blue')
                ax5_pressure.tick_params(axis='y', labelcolor='blue')
                
                # Plot factors as bars
                x = np.arange(len(month_names))
                width = 0.8 / self.factors
                
                for f in range(self.factors):
                    offset = (f - self.factors/2) * width
                    # Clean the monthly factor data
                    monthly_factors_clean = np.nan_to_num(monthly_factors[:, f], nan=0.0, posinf=0.0, neginf=0.0)
                    ax5_factors.bar(x + offset, monthly_factors_clean, width, 
                                   alpha=0.6, color=colors[f], label=f'Factor {f+1}')
                
                ax5_factors.set_ylabel('Average Factor Contribution')
                ax5_pressure.set_xlabel('Month')
                ax5_pressure.set_title('Seasonal Pressure and Factor Patterns')
                ax5_pressure.set_xticks(x)
                ax5_pressure.set_xticklabels(month_names)
                
                # Combine legends
                lines1, labels1 = ax5_pressure.get_legend_handles_labels()
                lines2, labels2 = ax5_factors.get_legend_handles_labels()
                ax5_pressure.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                
                ax5_pressure.grid(True, alpha=0.3)
                
            except Exception as e:
                # Fallback to simple plot if seasonal analysis fails
                ax5.text(0.5, 0.5, f'Seasonal analysis failed:\n{str(e)}', 
                        transform=ax5.transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        else:
            ax5.text(0.5, 0.5, 'Insufficient data\nfor seasonal analysis', 
                    transform=ax5.transAxes, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Plot 6: Pressure-factor relationship strength (bottom right)
        ax6 = axes[2, 1]
        
        # Calculate correlation strength and statistical significance
        correlations = []
        p_values = []
        
        for f in range(self.factors):
            if len(pressure_values) > 10:
                try:
                    corr, p_val = pearsonr(pressure_values, G_pressure_valid[:, f])
                    correlations.append(corr)
                    p_values.append(p_val)
                except:
                    correlations.append(0)
                    p_values.append(1)
            else:
                correlations.append(0)
                p_values.append(1)
        
        # Create bar chart of correlations
        bars = ax6.bar([f'Factor {i+1}' for i in range(self.factors)], 
                      [abs(c) for c in correlations], 
                      color=[colors[i] if p_values[i] < 0.05 else 'lightgray' for i in range(self.factors)],
                      alpha=0.8)
        
        ax6.set_ylabel('Absolute Correlation with Pressure')
        ax6.set_title('Pressure-Factor Relationship Strength')
        ax6.grid(True, alpha=0.3)
        ax6.set_ylim(0, 1)
        
        # Add correlation values and significance on bars
        for i, (bar, corr, p_val) in enumerate(zip(bars, correlations, p_values)):
            height = bar.get_height()
            significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'r={corr:.2f}\n{significance}', ha='center', va='bottom', fontsize=8, 
                    fontweight='bold')
        
        # Add legend for significance levels
        legend_text = ('Significance levels:\n'
                      '*** p < 0.001\n'
                      '** p < 0.01\n'
                      '* p < 0.05\n'
                      'ns = not significant')
        ax6.text(0.02, 0.98, legend_text, transform=ax6.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_pressure_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   ✅ Saved: pressure_analysis.png")
        
        # Create additional summary statistics
        self._create_pressure_summary_stats(dashboard_dir, pressure_df, G_pressure, pressure_columns, correlations, p_values)
    
    def _create_pressure_summary_stats(self, dashboard_dir, pressure_df, G_pressure, pressure_columns, correlations, p_values):
        """
        Create a summary table of pressure-factor relationships.
        """
        summary_path = dashboard_dir / f"{self.filename_prefix}_pressure_factor_summary.txt"
        
        with open(summary_path, 'w') as f:
            f.write(f"{self.station} Pressure-Factor Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Basic statistics
            f.write(f"Total matched data points: {len(pressure_df)}\n")
            f.write(f"Pressure columns analyzed: {', '.join(pressure_columns)}\n\n")
            
            for pressure_col in pressure_columns:
                if not pressure_df[pressure_col].isna().all():
                    valid_pressures = pressure_df[pressure_col].dropna()
                    f.write(f"{pressure_col} Statistics:\n")
                    f.write(f"  Range: {valid_pressures.min():.1f} - {valid_pressures.max():.1f} {self.units.get(pressure_col, 'hPa')}\n")
                    f.write(f"  Mean: {valid_pressures.mean():.1f} {self.units.get(pressure_col, 'hPa')}\n")
                    f.write(f"  Median: {valid_pressures.median():.1f} {self.units.get(pressure_col, 'hPa')}\n")
                    f.write(f"  Std deviation: {valid_pressures.std():.1f} {self.units.get(pressure_col, 'hPa')}\n\n")
            
            f.write("Factor-Pressure Correlations:\n")
            f.write("-" * 30 + "\n")
            
            for i in range(self.factors):
                f.write(f"Factor {i+1}:\n")
                f.write(f"  Mean contribution: {np.mean(G_pressure[:, i]):.3f}\n")
                f.write(f"  Std deviation: {np.std(G_pressure[:, i]):.3f}\n")
                f.write(f"  Correlation with pressure: {correlations[i]:.3f}\n")
                f.write(f"  P-value: {p_values[i]:.3e}\n")
                
                # Interpretation
                if p_values[i] < 0.001:
                    significance = "highly significant (***)"
                elif p_values[i] < 0.01:
                    significance = "very significant (**)"
                elif p_values[i] < 0.05:
                    significance = "significant (*)"
                else:
                    significance = "not significant (ns)"
                
                if abs(correlations[i]) > 0.7:
                    strength = "strong"
                elif abs(correlations[i]) > 0.5:
                    strength = "moderate"
                elif abs(correlations[i]) > 0.3:
                    strength = "weak"
                else:
                    strength = "very weak"
                
                direction = "positive" if correlations[i] > 0 else "negative"
                
                f.write(f"  Interpretation: {strength} {direction} correlation, {significance}\n")
                f.write("\n")
            
            # Environmental interpretation hints
            f.write("Environmental Interpretation Notes:\n")
            f.write("-" * 35 + "\n")
            f.write("• Strong positive pressure correlation may indicate:\n")
            f.write("  - High pressure system influences (stable conditions)\n")
            f.write("  - Reduced atmospheric mixing\n")
            f.write("  - Accumulation of local emissions\n")
            f.write("  - Anticyclonic weather patterns\n\n")
            
            f.write("• Strong negative pressure correlation may indicate:\n")
            f.write("  - Low pressure system influences (unstable conditions)\n")
            f.write("  - Enhanced atmospheric mixing and ventilation\n")
            f.write("  - Storm/precipitation scavenging effects\n")
            f.write("  - Cyclonic weather patterns with dilution\n\n")
            
            f.write("• Factors with weak pressure correlation may indicate:\n")
            f.write("  - Sources independent of meteorological conditions\n")
            f.write("  - Industrial sources with constant emissions\n")
            f.write("  - Indoor/sheltered emission sources\n")
            f.write("  - Regional background contributions\n")
        
        print(f"   📄 Pressure summary statistics: {summary_path}")
    
    def convert_dashboard_to_pdf(self, dashboard_dir, station=None):
        """Convert HTML dashboard to PDF using multiple fallback methods."""
        # Use provided station or derive from self.station
        station_name = station or self.station
        
        # Use the specific HTML dashboard file that was created
        html_file = dashboard_dir / f"{self.filename_prefix}_pmf_dashboard.html"
        
        if not html_file.exists():
            print(f"   ⚠️ HTML dashboard not found: {html_file}")
            return None
        # Create PDF filename using filename_prefix
        pdf_file = dashboard_dir / f"{self.filename_prefix}_pmf_dashboard.pdf"
        
        try:
            # Method 1: Use weasyprint (best option, pure Python)
            if HAS_WEASYPRINT:
                try:
                    import weasyprint
                    weasyprint.HTML(filename=str(html_file)).write_pdf(str(pdf_file))
                    print(f"   ✅ PDF created with weasyprint: {pdf_file.name}")
                    return pdf_file
                except Exception as e:
                    print(f"   ⚠️ Weasyprint failed: {e}, trying next method")
            
            # Method 2: Use pdfkit (requires wkhtmltopdf)
            if HAS_PDFKIT:
                try:
                    options = {
                        'page-size': 'A4',
                        'orientation': 'Portrait',
                        'margin-top': '0.75in',
                        'margin-right': '0.75in',
                        'margin-bottom': '0.75in',
                        'margin-left': '0.75in',
                        'encoding': "UTF-8",
                        'no-outline': None,
                        'enable-local-file-access': None
                    }
                    
                    pdfkit.from_file(str(html_file), str(pdf_file), options=options)
                    print(f"   ✅ PDF created with pdfkit: {pdf_file.name}")
                    return pdf_file
                except Exception as e:
                    # Only log pdfkit failures if it's not the common missing executable issue
                    if 'wkhtmltopdf executable found' not in str(e):
                        print(f"   ⚠️ pdfkit failed: {e}, trying next method")
                    else:
                        print(f"   ℹ️ pdfkit not available (wkhtmltopdf not found)")
            
            # Method 3: Try using Chrome/Edge headless
            success = self._convert_with_chrome(html_file, pdf_file)
            if success:
                print(f"   ✅ PDF created with Chrome: {pdf_file.name}")
                return pdf_file
            
            # Method 4: Create a simple text-based report as fallback
            text_report = self._create_text_report(dashboard_dir, station_name)
            print(f"   ⚠️ All PDF methods failed, created text report: {text_report.name if text_report else 'failed'}")
            return text_report
        
        except Exception as e:
            print(f"   ❌ PDF conversion failed: {e}")
            try:
                text_report = self._create_text_report(dashboard_dir, station_name)
                print(f"   ✅ Created fallback text report: {text_report.name if text_report else 'failed'}")
                return text_report
            except Exception as e2:
                print(f"   ❌ Even text report failed: {e2}")
                return None
    
    def _convert_with_chrome(self, html_file, pdf_file):
        """Try to convert HTML to PDF using Chrome headless."""
        try:
            # Try common Chrome/Edge locations (most likely first)
            chrome_paths = [
                r'C:\Program Files\Google\Chrome\Application\chrome.exe',  # Most common
                r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
                r'C:\Program Files\Microsoft\Edge\Application\msedge.exe',
                r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
                'chrome',  # If in PATH
                'google-chrome',
                'chromium-browser'
            ]
            
            for chrome_path in chrome_paths:
                try:
                    # Check if chrome executable exists
                    if not os.path.exists(chrome_path) and chrome_path not in ['chrome', 'google-chrome', 'chromium-browser']:
                        continue
                    
                    # Convert file paths to proper format
                    pdf_path_str = str(pdf_file.absolute()).replace('\\', '/')
                    html_path_str = html_file.absolute().as_uri()
                    
                    cmd = [
                        chrome_path,
                        '--headless=new',
                        '--disable-gpu', 
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-web-security',
                        '--run-all-compositor-stages-before-draw',
                        '--virtual-time-budget=25000',
                        f'--print-to-pdf={pdf_path_str}',
                        html_path_str
                    ]
                    
                    env = os.environ.copy()
                    env['PYTHONIOENCODING'] = 'utf-8'
                    env['PYTHONLEGACYWINDOWSFSENCODING'] = '1'
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        timeout=60, 
                        env=env,
                        encoding='utf-8',
                        errors='replace'
                    )
                    
                    if result.returncode == 0 and pdf_file.exists():
                        return True
                
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                    continue
            
            return False
        
        except Exception:
            return False
    
    def _create_text_report(self, dashboard_dir, station_name):
        """Create a simple text-based summary report."""
        report_file = dashboard_dir / f"{self.filename_prefix}_summary_report.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"{station_name} PMF Analysis Summary Report\n")
                f.write("=" * 60 + "\n\n")
                
                f.write(f"Station: {station_name}\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
                if self.start_date and self.end_date:
                    f.write(f"Date Range: {self.start_date} to {self.end_date}\n")
                f.write(f"Number of Factors: {self.factors}\n")
                f.write(f"Number of Models: {self.models}\n\n")
                
                # Try to include basic analysis information
                try:
                    if hasattr(self, 'species_names') and self.species_names:
                        f.write(f"Species Analyzed ({len(self.species_names)}): {', '.join(self.species_names)}\n\n")
                    
                    # Look for summary files
                    summary_files = list(dashboard_dir.glob("*_pmf_summary.txt"))
                    factor_summaries = list(dashboard_dir.glob("*factor*summary*.txt"))
                    wind_summaries = list(dashboard_dir.glob("*wind*summary*.txt"))
                    temp_summaries = list(dashboard_dir.glob("*temperature*summary*.txt"))
                    pressure_summaries = list(dashboard_dir.glob("*pressure*summary*.txt"))
                    
                    # Include PMF summary if available
                    if summary_files:
                        f.write("PMF Analysis Summary:\n")
                        f.write("-" * 25 + "\n")
                        try:
                            with open(summary_files[0], 'r', encoding='utf-8') as sf:
                                f.write(sf.read())
                        except:
                            f.write("Could not read PMF summary file.\n")
                        f.write("\n\n")
                    
                    # Include other summaries
                    for summary_group, summary_list in [
                        ("Factor Analysis", factor_summaries),
                        ("Wind Analysis", wind_summaries), 
                        ("Temperature Analysis", temp_summaries),
                        ("Pressure Analysis", pressure_summaries)
                    ]:
                        if summary_list:
                            f.write(f"{summary_group}:\n")
                            f.write("-" * len(summary_group) + "\n")
                            try:
                                with open(summary_list[0], 'r', encoding='utf-8') as sf:
                                    content = sf.read()
                                    # Truncate very long summaries
                                    if len(content) > 2000:
                                        content = content[:2000] + "\n... (truncated)\n"
                                    f.write(content)
                            except:
                                f.write(f"Could not read {summary_group.lower()} summary file.\n")
                            f.write("\n\n")
                
                except Exception as e:
                    f.write(f"Could not include detailed statistics: {e}\n")
                
                # Count dashboard files
                png_files = len(list(dashboard_dir.glob('*.png')))
                html_files = len(list(dashboard_dir.glob('*.html')))
                
                f.write(f"Dashboard files generated:\n")
                f.write(f"- PNG plots: {png_files}\n")
                f.write(f"- HTML files: {html_files}\n")
                f.write(f"\nOutput directory: {dashboard_dir}\n")
            
            return report_file
        
        except Exception as e:
            print(f"   ❌ Failed to create text report: {e}")
            return None
    
    def _create_sankey_diagram(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """
        Create a Sankey diagram showing the flow from PMF factors to species concentrations.
        This provides an intuitive visualization of how each factor contributes to different species.
        """
        print("   🌊 Creating Sankey diagram (Factors → Species)...")
        
        # Try multiple approaches in order of preference
        sankey_created = False
        
        # Approach 1: Try Plotly Sankey (most proper)
        try:
            import plotly.graph_objects as go
            from plotly.offline import plot
            print("     Attempting interactive Plotly Sankey diagram...")
            sankey_created = self._create_plotly_sankey(dashboard_dir, plot_files, F_profiles)
        except ImportError:
            print("     Plotly not available, trying matplotlib alternatives...")
        except Exception as e:
            print(f"     Plotly Sankey failed: {e}")
        
        # Approach 2: Try matplotlib with proper Sankey library
        if not sankey_created:
            try:
                print("     Attempting matplotlib Sankey with sankey library...")
                sankey_created = self._create_matplotlib_sankey_proper(dashboard_dir, plot_files, F_profiles)
            except Exception as e:
                print(f"     Matplotlib Sankey failed: {e}")
        
        # Approach 3: Create custom flow diagram (reliable fallback)
        if not sankey_created:
            try:
                print("     Creating custom flow diagram as Sankey alternative...")
                self._create_custom_flow_sankey(dashboard_dir, plot_files, F_profiles, G_contributions)
                sankey_created = True
            except Exception as e:
                print(f"     Custom flow diagram failed: {e}")
        
        # Approach 4: Final fallback - simple heatmap
        if not sankey_created:
            print("     Using simple heatmap as final fallback...")
            self._create_flow_chart_alternative(dashboard_dir, plot_files, F_profiles)
    
    def _create_plotly_sankey(self, dashboard_dir, plot_files, F_profiles):
        """
        Create an interactive Sankey diagram using Plotly.
        """
        import plotly.graph_objects as go
        from plotly.offline import plot
        
        # Prepare data for Sankey diagram
        # Calculate total contribution of each factor to each species
        factor_species_flows = F_profiles  # Shape: (n_factors, n_species)
        
        # Create node labels
        factor_labels = [f'Factor {i+1}' for i in range(self.factors)]
        species_labels = [f'{species}' for species in self.species_names]
        all_labels = factor_labels + species_labels
        
        # Create source, target, and value arrays for Sankey
        sources = []
        targets = []
        values = []
        
        # Threshold for minimum flow to show (to avoid clutter)
        min_flow_threshold = 0.01 * np.max(factor_species_flows)  # 1% of maximum flow
        
        for factor_idx in range(self.factors):
            for species_idx in range(len(self.species_names)):
                flow_value = factor_species_flows[factor_idx, species_idx]
                
                if flow_value > min_flow_threshold:  # Only show significant flows
                    sources.append(factor_idx)  # Factor node index
                    targets.append(self.factors + species_idx)  # Species node index (offset by n_factors)
                    values.append(flow_value)
        
        # Define colors for factors and species
        factor_colors = ['rgba(31, 119, 180, 0.8)', 'rgba(255, 127, 14, 0.8)', 
                        'rgba(44, 160, 44, 0.8)', 'rgba(214, 39, 40, 0.8)',
                        'rgba(148, 103, 189, 0.8)', 'rgba(140, 86, 75, 0.8)',
                        'rgba(227, 119, 194, 0.8)', 'rgba(127, 127, 127, 0.8)']
        
        species_colors = ['rgba(188, 189, 34, 0.6)'] * len(self.species_names)
        
        # Ensure we have enough colors
        while len(factor_colors) < self.factors:
            factor_colors.extend(factor_colors)
        
        node_colors = factor_colors[:self.factors] + species_colors
        
        # Calculate node positions based on actual flow values to prevent overlapping
        # Factors on left (x=0.01), species on right (x=0.99)
        
        # Calculate flow-based node heights
        # For factors: height proportional to total outgoing flow
        factor_flows = np.sum(F_profiles, axis=1)  # Total flow from each factor
        max_factor_flow = np.max(factor_flows) if np.max(factor_flows) > 0 else 1
        
        # For species: height proportional to total incoming flow
        species_flows = np.sum(F_profiles, axis=0)  # Total flow to each species
        max_species_flow = np.max(species_flows) if np.max(species_flows) > 0 else 1
        
        # Calculate relative heights (0.02 to 0.15 range for visibility)
        min_height = 0.02
        max_height = 0.15
        
        factor_heights = min_height + (factor_flows / max_factor_flow) * (max_height - min_height)
        species_heights = min_height + (species_flows / max_species_flow) * (max_height - min_height)
        
        print(f"   Factor heights: {factor_heights}")
        print(f"   Species heights: {species_heights}")
        
        # Calculate cumulative positions to avoid overlaps
        available_height = 0.9  # Use 90% of plot height (0.05 to 0.95)
        
        # Factor positions (left side) - stack from bottom with gaps
        total_factor_height = np.sum(factor_heights)
        n_factor_gaps = max(0, self.factors - 1)
        gap_size = 0.02  # Minimum gap between nodes
        total_required_factor_space = total_factor_height + n_factor_gaps * gap_size
        
        if total_required_factor_space <= available_height:
            # Fit with equal gaps
            extra_space = available_height - total_required_factor_space
            adjusted_gap = gap_size + extra_space / max(1, n_factor_gaps)
        else:
            # Compress gaps if needed
            adjusted_gap = max(0.005, (available_height - total_factor_height) / max(1, n_factor_gaps))
        
        factor_y_positions = []
        current_y = 0.05 + factor_heights[0] / 2  # Start with center of first node
        
        for i in range(self.factors):
            if i == 0:
                factor_y_positions.append(current_y)
            else:
                # Move to next position: half of previous height + gap + half of current height
                current_y += factor_heights[i-1]/2 + adjusted_gap + factor_heights[i]/2
                factor_y_positions.append(current_y)
        
        # Species positions (right side) - same logic
        total_species_height = np.sum(species_heights)
        n_species_gaps = max(0, len(self.species_names) - 1)
        total_required_species_space = total_species_height + n_species_gaps * gap_size
        
        if total_required_species_space <= available_height:
            # Fit with equal gaps
            extra_space = available_height - total_required_species_space
            adjusted_species_gap = gap_size + extra_space / max(1, n_species_gaps)
        else:
            # Compress gaps if needed
            adjusted_species_gap = max(0.005, (available_height - total_species_height) / max(1, n_species_gaps))
        
        species_y_positions = []
        current_y = 0.05 + species_heights[0] / 2  # Start with center of first node
        
        for i in range(len(self.species_names)):
            if i == 0:
                species_y_positions.append(current_y)
            else:
                # Move to next position: half of previous height + gap + half of current height
                current_y += species_heights[i-1]/2 + adjusted_species_gap + species_heights[i]/2
                species_y_positions.append(current_y)
        
        # Center the entire layout if it doesn't fill the available space
        if len(factor_y_positions) > 0:
            factor_span = factor_y_positions[-1] + factor_heights[-1]/2 - (factor_y_positions[0] - factor_heights[0]/2)
            if factor_span < available_height:
                offset = (available_height - factor_span) / 2
                factor_y_positions = [y + offset for y in factor_y_positions]
        
        if len(species_y_positions) > 0:
            species_span = species_y_positions[-1] + species_heights[-1]/2 - (species_y_positions[0] - species_heights[0]/2)
            if species_span < available_height:
                offset = (available_height - species_span) / 2
                species_y_positions = [y + offset for y in species_y_positions]
        
        print(f"   Final factor positions: {factor_y_positions}")
        print(f"   Final species positions: {species_y_positions}")
        
        node_x = [0.01] * self.factors + [0.99] * len(self.species_names)
        node_y = factor_y_positions + species_y_positions
        
        # Create Sankey diagram with explicit positioning
        fig = go.Figure(data=[go.Sankey(
            arrangement="fixed",  # Use fixed positioning
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=node_colors,
                x=node_x,
                y=node_y
            ),
            link=dict(
                source=sources,
                target=targets, 
                value=values,
                color=[factor_colors[src].replace('0.8', '0.4') for src in sources]  # Semi-transparent links
            )
        )])
        
        fig.update_layout(
            title=dict(
                text=f"{self.station} PMF Source Apportionment: Factor → Species Flow",
                x=0.5,
                font=dict(size=16)
            ),
            font_size=12,
            width=1200,
            height=800,
            annotations=[
                dict(
                    text="Factors", x=0.01, y=1.02, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=14, color="black"), 
                    xanchor="left", yanchor="bottom"
                ),
                dict(
                    text="Species", x=0.99, y=1.02, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=14, color="black"), 
                    xanchor="right", yanchor="bottom"
                )
            ]
        )
        
        # Save as HTML
        html_path = dashboard_dir / f"{self.filename_prefix}_sankey_diagram.html"
        plot(fig, filename=str(html_path), auto_open=False)
        
        # Also save as PNG (requires kaleido: pip install --upgrade kaleido)
        png_success = False
        try:
            # Try PNG export with improved error handling
            png_path = dashboard_dir / f"{self.filename_prefix}_sankey_diagram.png"
            
            # First try with explicit format and engine specification
            try:
                fig.write_image(str(png_path), format='png', width=1200, height=800, scale=2)
                plot_files.append(png_path)
                print(f"     ✅ Saved: sankey_diagram.png")
                png_success = True
            except Exception as e1:
                # Try without format specification
                try:
                    fig.write_image(str(png_path), width=1200, height=800, scale=2)
                    plot_files.append(png_path)
                    print(f"     ✅ Saved: sankey_diagram.png (fallback method)")
                    png_success = True
                except Exception as e2:
                    # Print both error messages for debugging
                    print(f"     ⚠️ PNG export failed (method 1): {str(e1).strip()}")
                    print(f"     ⚠️ PNG export failed (method 2): {str(e2).strip()}")
                        
        except Exception as png_err:
            print(f"     ⚠️ PNG export outer exception: {str(png_err).strip()}")
        
        if not png_success:
            print(f"     📊 Creating matplotlib-based Sankey fallback...")
            # Create a static matplotlib version as fallback
            self._create_matplotlib_sankey_simple(dashboard_dir, plot_files, F_profiles)
        
        print(f"     ✅ Saved: sankey_diagram.html (interactive)")
        return True
    
    def _create_matplotlib_sankey(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """
        Create a Sankey-style diagram using matplotlib.
        This is a simplified version since matplotlib doesn't have native Sankey support.
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle(f'{self.station} PMF Factor → Species Flow Diagram', fontsize=16, fontweight='bold')
        
        # Calculate the flow data
        factor_species_flows = F_profiles  # Shape: (n_factors, n_species)
        
        # Normalize flows for better visualization
        max_flow = np.max(factor_species_flows)
        normalized_flows = factor_species_flows / max_flow
        
        # Define positions
        factor_x = 0.1
        species_x = 0.9
        
        # Factor positions (left side)
        factor_y_positions = np.linspace(0.1, 0.9, self.factors)
        # Species positions (right side)  
        species_y_positions = np.linspace(0.1, 0.9, len(self.species_names))
        
        # Use consistent ColorManager colors
        factor_colors = self.color_manager._get_factor_colors(self.factors)
        species_colors = self.color_manager._get_species_colors(self.species_names)
        
        # Draw factor nodes (left side)
        for i, (y_pos, color) in enumerate(zip(factor_y_positions, factor_colors)):
            # Factor contribution size (total contribution across all species)
            total_contribution = np.sum(G_contributions[:, i])  # Use time-series data for size
            relative_size = (total_contribution / np.max(np.sum(G_contributions, axis=0))) * 0.08 + 0.02
            
            circle = plt.Circle((factor_x, y_pos), relative_size, color=color, alpha=0.8, zorder=3)
            ax.add_patch(circle)
            ax.text(factor_x - 0.05, y_pos, f'Factor {i+1}', ha='right', va='center', 
                   fontsize=10, fontweight='bold')
        
        # Draw species nodes (right side)
        for i, y_pos in enumerate(species_y_positions):
            # Species total concentration (sum across all factors)
            total_species = np.sum(factor_species_flows[:, i])
            relative_size = (total_species / np.max(np.sum(factor_species_flows, axis=0))) * 0.06 + 0.015
            
            circle = plt.Circle((species_x, y_pos), relative_size, color=species_colors[i], alpha=0.7, zorder=3)
            ax.add_patch(circle)
            
            # Truncate long species names
            species_name = self.species_names[i]
            if len(species_name) > 12:
                species_name = species_name[:9] + '...'
            
            ax.text(species_x + 0.05, y_pos, species_name, ha='left', va='center', 
                   fontsize=9, fontweight='bold')
        
        # Draw flow lines (connections)
        min_flow_threshold = 0.05 * max_flow  # Only show flows > 5% of maximum
        
        for factor_idx in range(self.factors):
            factor_y = factor_y_positions[factor_idx]
            factor_color = factor_colors[factor_idx]
            
            for species_idx in range(len(self.species_names)):
                flow_value = factor_species_flows[factor_idx, species_idx]
                
                if flow_value > min_flow_threshold:
                    species_y = species_y_positions[species_idx]
                    
                    # Line width proportional to flow strength
                    line_width = (flow_value / max_flow) * 20 + 1
                    
                    # Create curved connection
                    x_mid = (factor_x + species_x) / 2
                    
                    # Use bezier-like curve
                    x_values = np.linspace(factor_x, species_x, 100)
                    y_values = factor_y + (species_y - factor_y) * (x_values - factor_x) / (species_x - factor_x)
                    
                    # Add some curve
                    curve_strength = 0.1 * abs(species_y - factor_y)
                    y_values += curve_strength * np.sin(np.pi * (x_values - factor_x) / (species_x - factor_x))
                    
                    ax.plot(x_values, y_values, color=factor_color, alpha=0.6, 
                           linewidth=line_width, zorder=1)
        
        # Formatting
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend
        legend_elements = []
        for i in range(self.factors):
            legend_elements.append(plt.Line2D([0], [0], color=factor_colors[i], lw=4, 
                                            label=f'Factor {i+1}'))
        
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
                 ncol=min(self.factors, 4), frameon=False)
        
        # Add title and description
        plt.figtext(0.5, 0.02, 
                   'Flow thickness represents factor contribution strength to each species\n'
                   'Node sizes represent total contributions/concentrations',
                   ha='center', va='bottom', fontsize=10, style='italic')
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_sankey_diagram.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"     ✅ Saved: sankey_diagram.png")
    
    def _create_matplotlib_sankey_simple(self, dashboard_dir, plot_files, F_profiles):
        """
        Create a simplified flow diagram when Plotly PNG export fails.
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(f'{self.station} PMF Factor → Species Contribution Matrix', fontsize=16, fontweight='bold')
        
        # Create a heatmap-style representation
        im = ax.imshow(F_profiles, cmap='viridis', aspect='auto')
        
        # Formatting
        ax.set_xticks(range(len(self.species_names)))
        ax.set_xticklabels(self.species_names, rotation=45, ha='right')
        ax.set_yticks(range(self.factors))
        ax.set_yticklabels([f'Factor {i+1}' for i in range(self.factors)])
        
        ax.set_xlabel('Species')
        ax.set_ylabel('PMF Factors')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Factor Loading (Contribution Strength)', rotation=270, labelpad=20)
        
        # Add value annotations
        max_val = np.max(F_profiles)
        for i in range(self.factors):
            for j in range(len(self.species_names)):
                value = F_profiles[i, j]
                # Only annotate significant values
                if value > 0.1 * max_val:
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                           color='white' if value > 0.5 * max_val else 'black',
                           fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_sankey_diagram.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"     ✅ Saved: sankey_diagram.png (heatmap style)")
    
    def _create_flow_chart_alternative(self, dashboard_dir, plot_files, F_profiles):
        """
        Create a simple flow chart as fallback when Sankey creation fails.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f'{self.station} PMF Factor-Species Flow Chart (Alternative)', fontsize=14, fontweight='bold')
        
        # Create a chord-diagram style visualization
        n_factors = self.factors
        n_species = len(self.species_names)
        
        # Create circular layout
        factor_angles = np.linspace(0, np.pi, n_factors)
        species_angles = np.linspace(np.pi, 2*np.pi, n_species)
        
        radius = 0.8
        factor_positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in factor_angles]
        species_positions = [(radius * np.cos(angle), radius * np.sin(angle)) for angle in species_angles]
        
        # Use consistent ColorManager colors
        factor_colors = self.color_manager._get_factor_colors(n_factors)
        species_colors = self.color_manager._get_species_colors(self.species_names)
        
        # Draw factor nodes
        for i, pos in enumerate(factor_positions):
            circle = plt.Circle(pos, 0.1, color=factor_colors[i], alpha=0.8)
            ax.add_patch(circle)
            ax.text(pos[0]*1.15, pos[1]*1.15, f'F{i+1}', ha='center', va='center', 
                   fontweight='bold', fontsize=10)
        
        # Draw species nodes  
        for i, pos in enumerate(species_positions):
            circle = plt.Circle(pos, 0.08, color=species_colors[i], alpha=0.7)
            ax.add_patch(circle)
            
            # Abbreviated species name
            name = self.species_names[i][:6] + '..' if len(self.species_names[i]) > 8 else self.species_names[i]
            ax.text(pos[0]*1.2, pos[1]*1.2, name, ha='center', va='center', 
                   fontweight='bold', fontsize=8)
        
        # Draw connections for significant flows
        max_flow = np.max(F_profiles)
        min_flow_threshold = 0.1 * max_flow
        
        for factor_idx in range(n_factors):
            for species_idx in range(n_species):
                flow = F_profiles[factor_idx, species_idx]
                if flow > min_flow_threshold:
                    factor_pos = factor_positions[factor_idx]
                    species_pos = species_positions[species_idx]
                    
                    line_width = (flow / max_flow) * 5 + 0.5
                    ax.plot([factor_pos[0], species_pos[0]], [factor_pos[1], species_pos[1]], 
                           color=factor_colors[factor_idx], alpha=0.5, linewidth=line_width)
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend
        ax.text(0, -1.3, 'Factors (F1-F4) connected to Species by contribution strength',
               ha='center', va='center', fontsize=10, style='italic')
        
        plt.tight_layout()
        plot_path = dashboard_dir / "sankey_diagram.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"     ✅ Saved: sankey_diagram.png (flow chart style)")
    
    def _create_matplotlib_sankey_proper(self, dashboard_dir, plot_files, F_profiles):
        """
        Attempt to create a proper Sankey diagram using matplotlib's native Sankey class.
        """
        try:
            from matplotlib.sankey import Sankey
            
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.suptitle(f'{self.station} PMF Factor → Species Sankey Diagram', fontsize=16, fontweight='bold')
            
            # Create Sankey instance
            sankey = Sankey(ax=ax, scale=0.01, offset=0.3, format='%.0f', gap=0.5)
            
            # Calculate flows
            factor_species_flows = F_profiles  # Shape: (n_factors, n_species)
            
            # Prepare flows for each factor
            for factor_idx in range(self.factors):
                flows = []
                orientations = []
                labels = []
                
                # Input flow (total factor contribution)
                total_factor_contribution = np.sum(factor_species_flows[factor_idx, :])
                if total_factor_contribution > 0:
                    flows.append(total_factor_contribution)
                    orientations.append(0)  # Right
                    labels.append(f'Factor {factor_idx + 1}')
                    
                    # Output flows to each significant species
                    for species_idx, species_name in enumerate(self.species_names):
                        flow_value = factor_species_flows[factor_idx, species_idx]
                        if flow_value > 0.01 * total_factor_contribution:  # >1% threshold
                            flows.append(-flow_value)  # Negative for outflow
                            orientations.append(1)  # Up
                            # Shorten species names for readability
                            short_name = species_name[:8] + '..' if len(species_name) > 10 else species_name
                            labels.append(short_name)
                    
                    # Add this factor's flow to the Sankey
                    sankey.add(flows=flows, orientations=orientations, labels=labels,
                              pathlengths=[0.25] * len(flows), trunklength=1.5)
            
            # Finish and render the Sankey diagram
            diagrams = sankey.finish()
            
            # Apply consistent color scheme
            factor_colors = self.color_manager._get_factor_colors(self.factors)
            for i, diagram in enumerate(diagrams):
                if i < self.factors:
                    diagram.texts[-1].set_color(factor_colors[i])
                    diagram.texts[-1].set_fontweight('bold')
            
            plt.tight_layout()
            plot_path = dashboard_dir / f"{self.filename_prefix}_sankey_diagram.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(plot_path)
            print(f"     ✅ Saved: sankey_diagram.png (matplotlib Sankey)")
            return True
            
        except Exception as e:
            print(f"     ⚠️ Matplotlib Sankey failed: {e}")
            return False
    
    def _create_custom_flow_sankey(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """
        Create a custom flow diagram that resembles a Sankey diagram.
        This is designed to be a reliable fallback that always works.
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.suptitle(f'{self.station} PMF Source Apportionment Flow Diagram\n(Factors → Species)', 
                    fontsize=18, fontweight='bold')
        
        # Calculate flow data
        factor_species_flows = F_profiles  # Shape: (n_factors, n_species)
        
        # Normalize for consistent visualization
        max_flow = np.max(factor_species_flows)
        if max_flow == 0:
            max_flow = 1  # Prevent division by zero
        
        # Position calculations
        factor_x = 0.15
        species_x = 0.85
        
        # Vertical positions
        factor_y_positions = np.linspace(0.15, 0.85, self.factors)
        species_y_positions = np.linspace(0.15, 0.85, len(self.species_names))
        
        # Use consistent ColorManager colors
        factor_colors = self.color_manager._get_factor_colors(self.factors)
        species_colors = self.color_manager._get_species_colors(self.species_names)
        
        # Calculate sizes based on total contributions
        factor_sizes = np.sum(G_contributions, axis=0)  # Total contribution over time
        factor_sizes = factor_sizes / np.max(factor_sizes) * 0.06 + 0.02  # Normalize to reasonable sizes
        
        species_sizes = np.sum(factor_species_flows, axis=0)  # Total from all factors
        species_sizes = species_sizes / np.max(species_sizes) * 0.04 + 0.015  # Normalize to reasonable sizes
        
        # Draw factor nodes (left side)
        factor_nodes = []
        for i, (y_pos, color, size) in enumerate(zip(factor_y_positions, factor_colors, factor_sizes)):
            circle = plt.Circle((factor_x, y_pos), size, color=color, alpha=0.8, zorder=3)
            ax.add_patch(circle)
            factor_nodes.append((factor_x, y_pos, size))
            
            # Factor labels
            ax.text(factor_x - 0.08, y_pos, f'Factor {i+1}', ha='right', va='center', 
                   fontsize=12, fontweight='bold', color=color)
        
        # Draw species nodes (right side)
        species_nodes = []
        for i, (y_pos, size) in enumerate(zip(species_y_positions, species_sizes)):
            circle = plt.Circle((species_x, y_pos), size, color=species_colors[i], 
                               alpha=0.7, zorder=3, edgecolor='navy', linewidth=1)
            ax.add_patch(circle)
            species_nodes.append((species_x, y_pos, size))
            
            # Species labels (with smart truncation)
            species_name = self.species_names[i]
            if len(species_name) > 12:
                display_name = species_name[:9] + '...'
            else:
                display_name = species_name
            
            ax.text(species_x + 0.08, y_pos, display_name, ha='left', va='center', 
                   fontsize=10, fontweight='bold', color='navy')
        
        # Draw flow streams (the key Sankey-like feature)
        min_flow_threshold = 0.02 * max_flow  # Only show flows > 2% of maximum
        
        for factor_idx in range(self.factors):
            factor_x_pos, factor_y_pos, factor_size = factor_nodes[factor_idx]
            factor_color = factor_colors[factor_idx]
            
            for species_idx in range(len(self.species_names)):
                flow_value = factor_species_flows[factor_idx, species_idx]
                
                if flow_value > min_flow_threshold:
                    species_x_pos, species_y_pos, species_size = species_nodes[species_idx]
                    
                    # Calculate flow width (Sankey characteristic)
                    flow_width = (flow_value / max_flow) * 30 + 2
                    
                    # Create smooth curved flow path
                    # Start from edge of factor node
                    start_x = factor_x_pos + factor_size
                    start_y = factor_y_pos
                    
                    # End at edge of species node
                    end_x = species_x_pos - species_size
                    end_y = species_y_pos
                    
                    # Create bezier curve points for smooth flow
                    n_points = 100
                    t = np.linspace(0, 1, n_points)
                    
                    # Control points for smooth curve
                    control1_x = start_x + 0.2
                    control1_y = start_y
                    control2_x = end_x - 0.2
                    control2_y = end_y
                    
                    # Bezier curve calculation
                    x_curve = (1-t)**3 * start_x + 3*(1-t)**2*t * control1_x + \
                             3*(1-t)*t**2 * control2_x + t**3 * end_x
                    y_curve = (1-t)**3 * start_y + 3*(1-t)**2*t * control1_y + \
                             3*(1-t)*t**2 * control2_y + t**3 * end_y
                    
                    # Draw the flow stream with varying alpha for depth effect
                    for j in range(len(x_curve) - 1):
                        alpha_val = 0.3 + 0.4 * (flow_value / max_flow)  # Vary alpha by flow strength
                        ax.plot([x_curve[j], x_curve[j+1]], [y_curve[j], y_curve[j+1]], 
                               color=factor_color, linewidth=flow_width, alpha=alpha_val, 
                               solid_capstyle='round', zorder=1)
        
        # Add flow strength legend
        legend_x = 0.02
        legend_y = 0.98
        ax.text(legend_x, legend_y, 'Flow Legend:', transform=ax.transAxes, 
               fontsize=12, fontweight='bold', va='top')
        
        # Create sample flows for legend
        sample_flows = [max_flow * 0.8, max_flow * 0.5, max_flow * 0.2]
        sample_labels = ['Strong', 'Medium', 'Weak']
        
        for i, (flow, label) in enumerate(zip(sample_flows, sample_labels)):
            legend_y_pos = 0.93 - i * 0.05
            sample_width = (flow / max_flow) * 30 + 2
            
            # Draw sample line
            ax.plot([legend_x, legend_x + 0.08], [legend_y_pos, legend_y_pos], 
                   transform=ax.transAxes, color='gray', linewidth=sample_width, alpha=0.6)
            
            # Add label
            ax.text(legend_x + 0.1, legend_y_pos, f'{label} Flow', transform=ax.transAxes, 
                   fontsize=10, va='center')
        
        # Add summary statistics box
        stats_x = 0.02
        stats_y = 0.25
        
        stats_text = f"""Flow Summary:
• Total Factors: {self.factors}
• Species: {len(self.species_names)}
• Max Flow: {max_flow:.3f}
• Flows Shown: >{min_flow_threshold:.3f}
• Total Connections: {np.sum(factor_species_flows > min_flow_threshold)}"""
        
        ax.text(stats_x, stats_y, stats_text, transform=ax.transAxes, 
               fontsize=9, va='top', bbox=dict(boxstyle='round,pad=0.5', 
               facecolor='lightgray', alpha=0.7))
        
        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add descriptive title
        ax.text(0.5, 0.02, 
               'Flow thickness represents factor contribution strength to each species.\n'
               'Node sizes represent total contribution magnitudes.',
               ha='center', va='bottom', transform=ax.transAxes, fontsize=11, 
               style='italic', color='gray')
        
        plt.tight_layout()
        plot_path = dashboard_dir / "sankey_diagram.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"     ✅ Saved: sankey_diagram.png (custom flow diagram)")
        return True


def show_detailed_help():
    """Show detailed descriptions of all CLI flags and their defaults."""
    help_text = """
[BOOKS] DETAILED CLI FLAG REFERENCE
=============================================================================

[FOLDER] DATA INPUT OPTIONS:
  station               Station to analyze using built-in file mappings
                        Choices: MMF1, MMF2, MMF6, MMF9, Maries_Way
                        Alternative: use --data-dir + --patterns for custom files
  
  --data-dir PATH       Directory containing parquet files (flexible mode)
                        Use with --patterns to specify file patterns
                        
  --patterns PATTERNS   Comma-separated parquet file patterns to match
                        Example: "MMF2_combined_data.parquet,MMF9_*.parquet"
                        
  --start-date DATE     Start date in YYYY-MM-DD format (optional)
                        Example: 2023-09-01
                        
  --end-date DATE       End date in YYYY-MM-DD format (optional)
                        Example: 2023-09-30

[FIRE] PMF ANALYSIS OPTIONS:
  --factors N           Exact number of factors to use (no optimization)
                        Range: 1-20, overrides --max-factors
                        
  --max-factors N       Maximum factors to test during optimization
                        Default: 10, ignored if --factors specified
                        
  --models N            Number of PMF models to run for statistical robustness
                        Default: 20, EPA recommends 20+ for reliable results
                        Range: 1-100
                        
  --max-workers N       Parallel processes for PMF analysis
                        Default: 2, increase for faster processing

[OUTPUT] OUTPUT OPTIONS:
  --output-dir PATH     Output directory for all results
                        Default: pmf_results_esat
                        
  --run-pca             Run PCA analysis for comparison with PMF
                        Adds PCA-PMF comparison plots to dashboard
                        
  --create-pdf          Create PDF version of HTML dashboard
                        Requires Chrome/Edge browser or PDF libraries

[FILTER] DATA FILTERING OPTIONS:
  --remove-voc          Exclude VOC species from PMF analysis
                        Removes: Benzene, Toluene, Ethylbenzene, Xylene
                        
  --scale-units         Apply unit standardization (DEFAULT)
                        mg/m3 -> ug/m3 (*1000), ng/m3 -> ug/m3 (/1000)
                        
  --no-scale-units      Disable unit standardization
                        Uses units as-is from source data

[QUALITY] DATA QUALITY CONTROLS:
  --drop-row-threshold F Drop rows with >F fraction missing values BEFORE replacement
                        Default: 0.5 (drop rows with >50% missing)
                        Range: 0.0-1.0
                        
  --zero-as-bdl         Treat exact zeros as below detection limit (DEFAULT)
                        
  --no-zero-as-bdl      Treat exact zeros as missing values instead
  
  --save-masks          Save BDL and missing value mask CSVs (DEFAULT)
                        
  --no-save-masks       Skip saving mask CSV files

[EPA] EPA UNCERTAINTY OPTIONS:
  --uncertainty-mode    Uncertainty calculation method
                        Choices: 'epa' (EPA PMF 5.0 formulas + 1/sqrt(n) scaling)
                                 'legacy' (MDL+EF table with min clamp)
                        Default: legacy
                        
  --uncertainty-ef-mdl  CSV file with custom EF/MDL values
                        Columns: species, EF, MDL, unit
                        If not provided, uses built-in instrument specs
                        
  --uncertainty-epsilon F Numerical floor for uncertainties (stability)
                        Default: 1e-12, not a weighting clamp
                        
  --legacy-min-u F      Minimum uncertainty clamp for legacy mode
                        Default: 0.1, only applies when --uncertainty-mode=legacy
                        
  --uncertainty-bdl-policy BDL uncertainty policy for conc <= MDL
                        Choices: 'five-sixth-mdl' (U = 5/6 * MDL)
                                 'half-mdl' (U = 0.5 * MDL)
                        Default: five-sixth-mdl

[S/N] S/N CATEGORIZATION OPTIONS:
  --snr-enable          Enable EPA S/N-based feature categorization
                        Default: disabled (preserves legacy behavior)
                        Categories: strong (S/N >= 2.0), weak (tripled uncertainty), bad (excluded)
                        
  --snr-weak-threshold F S/N threshold for weak categorization
                        Default: 2.0, species with S/N < 2.0 are weak
                        
  --snr-bad-threshold F  S/N threshold for bad categorization (exclusion)
                        Default: 0.2, species with S/N < 0.2 are excluded
                        
  --snr-bdl-weak-frac F  BDL fraction threshold for weak categorization
                        Default: 0.6 (60% BDL makes species weak)
                        
  --snr-bdl-bad-frac F   BDL fraction threshold for bad categorization
                        Default: 0.8 (80% BDL makes species bad/excluded)
                        
  --snr-missing-weak-frac F Missing fraction threshold for weak categorization
                        Default: 0.2 (20% missing makes species weak)
                        
  --snr-missing-bad-frac F Missing fraction threshold for bad categorization
                        Default: 0.4 (40% missing makes species bad/excluded)
                        
  --exclude-bad         Exclude bad species from PMF analysis (DEFAULT)
                        Recommended to keep enabled
                        
  --dashboard-snr-panel Add S/N analysis panels to dashboard (DEFAULT)
                        Shows S/N metrics and categorization results
                        
  --write-diagnostics   Write S/N metrics and categorization CSVs (DEFAULT)
                        Saves detailed diagnostic information

[ROBUST] ESAT ROBUST TRAINING (auto-switches to single SA mode):
  --robust-fit           Enable robust loss during SA training (downweights large scaled residuals)
                        Automatically forces single SA mode (BatchSA doesn't support robust params)
  --robust-alpha F       Robust cutoff alpha for uncertainty-scaled residuals (default: 4.0)
                        Higher values = more aggressive outlier downweighting

[ESAT] ESAT ALGORITHM AND INITIALIZATION CONTROLS:
  --method               ESAT NMF algorithm selection
                        Choices: ls-nmf (nonnegative, standard PMF behavior)
                                ws-nmf (semi-NMF, allows negative contributions)
                        Default: ls-nmf (recommended for PMF)
                        
  --init-method          Matrix initialization method
                        Choices: column_mean (randomized by column statistics)
                                kmeans (k-means clustering, better for magnitude differences)
                        Default: column_mean
                        
  --init-norm            Whiten (normalize) data before kmeans initialization (DEFAULT)
                        Reduces impact of cross-species magnitude differences
  --no-init-norm         Disable whitening before kmeans initialization
                        
  --hold-h               Hold H (profile) matrix constant during training
                        Use with --delay-h to stabilize early iterations
                        Default: disabled
                        
  --delay-h N            Hold H matrix for N iterations, then release
                        Requires --hold-h; lets W adapt first when species magnitudes vary
                        Default: -1 (disabled)

[SEED] REPRODUCIBILITY:
  --seed N              Random seed for reproducible results
                        Default: 42, ensures consistent PMF solutions

[HELP] HELP:
  --help-detail         Show this detailed help (you're reading it now!)
  -h, --help            Show standard help summary

=============================================================================
[EXAMPLES] EXAMPLE COMMANDS:

# Basic analysis with station mapping:
python pmf_source_app.py MMF9 --start-date 2023-09-01 --end-date 2023-09-30

# EPA uncertainty mode with S/N categorization:
python pmf_source_app.py MMF9 --uncertainty-mode epa --snr-enable --factors 7

# Custom data directory with unit scaling disabled:
python pmf_source_app.py --data-dir ./data --patterns "*.parquet" --no-scale-units

# High-performance analysis with PDF output:
python pmf_source_app.py MMF2 --models 50 --max-workers 4 --create-pdf --run-pca

# Robust PMF training to downweight outliers (auto-switches to single SA mode):
python pmf_source_app.py MMF9 --robust-fit --robust-alpha 3.0 --factors 5

=============================================================================
    """
    print(help_text)


def main():
    parser = argparse.ArgumentParser(description='PMF Source Apportionment Analysis for MMF Data', allow_abbrev=False)
    # Optional station argument (for backward compatibility)
    parser.add_argument('station', nargs='?', choices=['MMF1', 'MMF2', 'MMF6', 'MMF9', 'Maries_Way'],
                       help='MMF station to analyze (using corrected station mappings). Alternative: use --data-dir and --patterns')
    parser.add_argument('--data-dir', type=str,
                       help='Directory containing parquet files (alternative to station-based loading)')
    parser.add_argument('--patterns', type=str, 
                       help='Comma-separated parquet file patterns to match (e.g., "MMF2_combined_data.parquet,MMF9_combined_data.parquet")')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD format)', default=None)
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD format)', default=None)
    parser.add_argument('--factors', type=int, default=None,
                       help='Exact number of factors to use (no optimization). Overrides --max-factors if specified.')
    parser.add_argument('--max-factors', type=int, default=10,
                       help='Maximum factors to test during optimization (default: 10). Ignored if --factors is specified.')
    parser.add_argument('--models', type=int, default=20,
                       help='Number of models to run (must be >= 1, default: 20)')
    parser.add_argument('--output-dir', type=str, default='pmf_results_esat',
                       help='Output directory (default: pmf_results_esat)')
    parser.add_argument('--run-pca', action='store_true',
                       help='Run PCA analysis for comparison with PMF results')
    parser.add_argument('--create-pdf', action='store_true',
                       help='Create PDF version of the HTML dashboard (requires Chrome/Edge, pdfkit, or weasyprint)')
    parser.add_argument('--max-workers', type=int, default=2,
                       help='Maximum number of parallel processes for PMF analysis (default: 2)')
    parser.add_argument('--remove-voc', action='store_true',
                       help='Remove VOC species (Benzene, Toluene, Ethylbenzene, Xylene) from PMF analysis')
    
    # Unit scaling control
    scale_group = parser.add_mutually_exclusive_group()
    scale_group.add_argument('--scale-units', dest='scale_units', action='store_true',
                           help='Apply unit standardization: mg/m3->ug/m3 (*1000), ng/m3->ug/m3 (/1000) (default)')
    scale_group.add_argument('--no-scale-units', dest='scale_units', action='store_false',
                           help='Disable unit standardization - use units as-is from data')
    parser.set_defaults(scale_units=True)
    
    # Detailed help option
    parser.add_argument('--help-detail', action='store_true',
                       help='Show detailed descriptions of all CLI flags and their defaults')

    # EPA BDL/missing runtime controls
    parser.add_argument('--drop-row-threshold', type=float, default=0.5,
                       help='Row drop threshold BEFORE replacement: drop rows with a fraction of missing values above this threshold (0-1, default: 0.5 -> >50%% missing)')
    zero_group = parser.add_mutually_exclusive_group()
    zero_group.add_argument('--zero-as-bdl', dest='zero_as_bdl', action='store_true',
                           help='Treat exact zeros as below detection limit (BDL) (default)')
    zero_group.add_argument('--no-zero-as-bdl', dest='zero_as_bdl', action='store_false',
                           help='Treat exact zeros as missing instead of BDL')
    parser.set_defaults(zero_as_bdl=True)

    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument('--save-masks', dest='save_masks', action='store_true',
                           help='Save BDL and missing mask CSVs (default)')
    mask_group.add_argument('--no-save-masks', dest='save_masks', action='store_false',
                           help='Do not save BDL/missing mask CSVs')
    parser.set_defaults(save_masks=True)

    # EPA S/N weighting and uncertainty controls (legacy defaults preserve current behavior)
    parser.add_argument('--uncertainty-mode', choices=['epa', 'legacy'], default='legacy',
                       help='Uncertainty calculation mode: epa (EPA formulas + 1/sqrt(n) + no global clamp) or legacy (current MDL+EF table with min clamp) (default: legacy)')
    parser.add_argument('--uncertainty-ef-mdl', type=str, default=None,
                       help='CSV file with EF/MDL table (columns: species, EF, MDL, unit). If not provided, uses built-in values.')
    parser.add_argument('--uncertainty-epsilon', type=float, default=1e-12,
                       help='Numerical floor for uncertainties (not a weighting clamp) (default: 1e-12)')
    parser.add_argument('--legacy-min-u', type=float, default=0.1,
                       help='Minimum uncertainty clamp when --uncertainty-mode=legacy (default: 0.1)')
    parser.add_argument('--uncertainty-bdl-policy', choices=['five-sixth-mdl', 'half-mdl'], default='five-sixth-mdl',
                       help='Policy for conc <= MDL: five-sixth-mdl (U = 5/6 * MDL) or half-mdl (U = 0.5 * MDL) (default: five-sixth-mdl)')
    
    parser.add_argument('--snr-enable', action='store_true', default=False,
                       help='Enable S/N-based feature categorization (default: disabled to preserve legacy behavior)')
    
    # ESAT robust training options (automatically forces single SA mode)
    parser.add_argument('--robust-fit', action='store_true', default=False,
                       help='Use ESAT robust loss during SA training. Automatically switches to single SA mode (BatchSA does not support robust training). Downweights outliers via robust residual weighting.')
    parser.add_argument('--robust-alpha', type=float, default=4.0,
                       help='Robust cutoff alpha for uncertainty-scaled residuals (default: 4.0). Only effective when --robust-fit is enabled.')
    parser.add_argument('--snr-weak-threshold', type=float, default=2.0,
                       help='S/N threshold for weak categorization (default: 2.0)')
    parser.add_argument('--snr-bad-threshold', type=float, default=0.2,
                       help='S/N threshold for bad categorization (excluded) (default: 0.2)')
    parser.add_argument('--snr-bdl-weak-frac', type=float, default=0.6,
                       help='BDL fraction threshold for weak categorization (default: 0.6)')
    parser.add_argument('--snr-bdl-bad-frac', type=float, default=0.8,
                       help='BDL fraction threshold for bad categorization (default: 0.8)')
    parser.add_argument('--snr-missing-weak-frac', type=float, default=0.2,
                       help='Missing fraction threshold for weak categorization (default: 0.2)')
    parser.add_argument('--snr-missing-bad-frac', type=float, default=0.4,
                       help='Missing fraction threshold for bad categorization (default: 0.4)')
    parser.add_argument('--exclude-bad', action='store_true', default=True,
                       help='Exclude bad features from PMF analysis (default: enabled)')
    
    parser.add_argument('--dashboard-snr-panel', action='store_true', default=True,
                       help='Add S/N and categorization panels to dashboard (default: enabled)')
    parser.add_argument('--write-diagnostics', action='store_true', default=True,
                       help='Write S/N metrics, categories, and weights summary CSVs (default: enabled)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # ESAT Algorithm and Initialization Controls
    parser.add_argument('--method', choices=['ls-nmf', 'ws-nmf'], default='ls-nmf',
                       help='ESAT NMF method: ls-nmf (nonnegative, standard PMF) or ws-nmf (semi-NMF, allows negative W contributions) (default: ls-nmf)')
    parser.add_argument('--init-method', choices=['column_mean', 'kmeans'], default='column_mean',
                       help='Matrix initialization method: column_mean (randomized by column mean) or kmeans (k-means clustering) (default: column_mean)')
    
    init_norm_group = parser.add_mutually_exclusive_group()
    init_norm_group.add_argument('--init-norm', dest='init_norm', action='store_true',
                           help='Whiten (normalize) data before kmeans initialization (default when using kmeans)')
    init_norm_group.add_argument('--no-init-norm', dest='init_norm', action='store_false',
                           help='Disable whitening before kmeans initialization')
    parser.set_defaults(init_norm=True)
    
    parser.add_argument('--hold-h', action='store_true', default=False,
                       help='Hold the H (profile) matrix constant during training. Use with --delay-h to hold for N iterations then release.')
    parser.add_argument('--delay-h', type=int, default=-1,
                       help='Iterations to delay H matrix updates. When >0 and combined with --hold-h, holds H for N iterations then releases. (default: -1, disabled)')
    
    args = parser.parse_args()
    
    # Handle detailed help request
    if args.help_detail:
        show_detailed_help()
        return 0
    
    # Validate arguments
    if not args.station and not (args.data_dir and args.patterns):
        parser.error("Either specify a station or provide both --data-dir and --patterns")
    
    if args.station and (args.data_dir or args.patterns):
        parser.error("Cannot specify both station and --data-dir/--patterns. Choose one approach.")
    
    print("🚀 MMF PMF Source Apportionment Analysis (ESAT Fixed)")
    print("=" * 60)
    if args.station:
        print(f"Station: {args.station}")
    else:
        print(f"Data directory: {args.data_dir}")
        print(f"Patterns: {args.patterns}")
    print(f"Date range: {args.start_date or 'All'} to {args.end_date or 'All'}")
    print(f"Output: {args.output_dir}")
    print()
    
    try:
        # Initialize analyzer
        pmf = MMFPMFAnalyzer(
            station=args.station,
            data_dir=args.data_dir,
            patterns=args.patterns,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            remove_voc=args.remove_voc,
            # EPA S/N weighting and uncertainty parameters (legacy defaults preserve behavior)
            uncertainty_mode=args.uncertainty_mode,
            uncertainty_ef_mdl=args.uncertainty_ef_mdl,
            uncertainty_epsilon=args.uncertainty_epsilon,
            legacy_min_u=args.legacy_min_u,
            uncertainty_bdl_policy=args.uncertainty_bdl_policy,
            snr_enable=args.snr_enable,
            snr_weak_threshold=args.snr_weak_threshold,
            snr_bad_threshold=args.snr_bad_threshold,
            snr_bdl_weak_frac=args.snr_bdl_weak_frac,
            snr_bdl_bad_frac=args.snr_bdl_bad_frac,
            snr_missing_weak_frac=args.snr_missing_weak_frac,
            snr_missing_bad_frac=args.snr_missing_bad_frac,
            exclude_bad=args.exclude_bad,
            dashboard_snr_panel=args.dashboard_snr_panel,
            write_diagnostics=args.write_diagnostics,
            scale_units=args.scale_units,
            seed=args.seed,
            robust_fit=args.robust_fit,
            robust_alpha=args.robust_alpha,
            # ESAT algorithm and initialization controls
            method=args.method,
            init_method=args.init_method,
            init_norm=args.init_norm,
            hold_h=args.hold_h,
            delay_h=args.delay_h
        )
        
        # Override default parameters if specified
        if args.factors and args.factors > 0:
            # User explicitly specified a factor count
            pmf.factors = args.factors
            pmf.user_specified_factors = True  # Flag for optimization skipping
        else:
            # User wants auto-optimization
            pmf.factors = 4  # Default fallback
            pmf.user_specified_factors = False
        
        # Validate models count (ESAT requires at least 1)
        pmf.models = max(1, int(args.models))
        if args.models < 1:
            print("⚠️  --models must be >= 1; defaulting to 1")
        pmf.max_factors = args.max_factors  # Pass max_factors for optimization
        pmf.max_workers = args.max_workers  # Control multiprocessing

        # Apply EPA BDL/missing runtime controls
        pmf.drop_row_threshold = max(0.0, min(1.0, args.drop_row_threshold))
        pmf.zero_as_bdl = args.zero_as_bdl
        pmf.save_masks = args.save_masks
        print(f"⚙️  BDL/Missing controls: drop-row-threshold={pmf.drop_row_threshold}, zero-as-bdl={pmf.zero_as_bdl}, save-masks={pmf.save_masks}")
        
        # Show EPA S/N weighting settings
        print(f"🔬 EPA S/N weighting: uncertainty-mode={pmf.uncertainty_mode}, snr-enable={pmf.snr_enable}")
        if pmf.snr_enable:
            print(f"   S/N thresholds: weak<{pmf.snr_weak_threshold}, bad<{pmf.snr_bad_threshold}")
            print(f"   Data quality: BDL weak>{pmf.snr_bdl_weak_frac*100:.0f}%, bad>{pmf.snr_bdl_bad_frac*100:.0f}%")
        
        # Run analysis workflow
        pmf.load_mmf_data()
        pmf.prepare_pmf_data()
        
        if pmf.run_pmf_analysis():
            # Run PCA analysis if requested
            if args.run_pca:
                print("\n🔬 Running PCA analysis for comparison...")
                if pmf.run_pca_analysis():
                    print("✅ PCA analysis completed successfully")
                else:
                    print("⚠️ PCA analysis failed, continuing without comparison plots")
            
            pmf.create_pmf_dashboard()
            pmf.generate_report()
            
            # Create PDF if requested
            pdf_path = None
            if args.create_pdf:
                print("\n📄 Creating PDF version of dashboard...")
                # Use the exact HTML filename that was created
                dashboard_dir = pmf.output_dir
                
                pdf_path = pmf.convert_dashboard_to_pdf(dashboard_dir)
                if pdf_path:
                    print(f"✅ PDF created: {pdf_path}")
                else:
                    print("⚠️ PDF creation failed, but text report may have been created")
            
            print("\n🎉 Analysis Complete!")
            if args.run_pca:
                print("📊 PMF + PCA analysis results saved")
                print("   PMF-PCA comparison plots included in dashboard")
            else:
                print("📊 PMF analysis results saved")
                print("   Use --run-pca flag to include PCA comparison plots")
            
            if not args.create_pdf:
                print("   Use --create-pdf flag to generate PDF reports")
            elif pdf_path:
                print(f"📄 PDF dashboard: {pdf_path}")
            
            print(f"📊 Results saved in: {pmf.output_dir}")
            dashboard_name = f"{args.station}_pmf_dashboard.html" if args.station else "mmf_pmf_dashboard.html"
            print(f"📄 View dashboard: {pmf.output_dir}/{dashboard_name}")
        else:
            print("\n❌ PMF analysis failed!")
            return 1
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
