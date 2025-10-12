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
import json
import multiprocessing as mp

def validate_date_string(date_str, param_name="date", auto_correct=True):
    """
    Validate a date string in YYYY-MM-DD format with optional auto-correction.
    
    Args:
        date_str (str): Date string to validate
        param_name (str): Parameter name for error messages
        auto_correct (bool): If True, auto-correct invalid dates to last valid day of month
        
    Returns:
        str: The validated (and potentially corrected) date string
        
    Raises:
        ValueError: If date string is invalid and cannot be auto-corrected
    """
    if date_str is None:
        return None
        
    if not isinstance(date_str, str):
        raise ValueError(f"{param_name} must be a string in YYYY-MM-DD format, got {type(date_str).__name__}")
    
    # Check basic format
    if len(date_str) != 10 or date_str[4] != '-' or date_str[7] != '-':
        raise ValueError(f"Invalid {param_name} format '{date_str}'. Expected YYYY-MM-DD format (e.g., 2023-06-30)")
    
    # Check if parts are numeric
    parts = date_str.split('-')
    if len(parts) != 3:
        raise ValueError(f"Invalid {param_name} format '{date_str}'. Expected YYYY-MM-DD format (e.g., 2023-06-30)")
    
    try:
        year, month, day = parts
        year = int(year)
        month = int(month)
        day = int(day)
    except ValueError:
        raise ValueError(f"Invalid {param_name} '{date_str}'. Year, month, and day must be numeric. Expected YYYY-MM-DD format (e.g., 2023-06-30)")
    
    # Validate ranges
    if year < 1900 or year > 2100:
        raise ValueError(f"Invalid year in {param_name} '{date_str}'. Year must be between 1900 and 2100")
    
    if month < 1 or month > 12:
        raise ValueError(f"Invalid month in {param_name} '{date_str}'. Month must be between 01 and 12")
    
    if day < 1 or day > 31:
        raise ValueError(f"Invalid day in {param_name} '{date_str}'. Day must be between 01 and 31")
    
    # Helper function to get max days in month
    def get_max_days_in_month(year, month):
        if month == 2:  # February
            return 29 if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0) else 28
        elif month in [4, 6, 9, 11]:  # April, June, September, November
            return 30
        else:
            return 31
    
    # Try to parse as actual date to catch invalid dates like 2023-09-31
    try:
        from datetime import datetime
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str  # Valid date, return as-is
    except ValueError as e:
        if "day is out of range for month" in str(e) and auto_correct:
            # Auto-correct to last valid day of month
            max_days = get_max_days_in_month(year, month)
            corrected_date = f"{year}-{month:02d}-{max_days:02d}"
            
            # Get the month name for better message
            month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            month_name = month_names[month] if 1 <= month <= 12 else f"month {month:02d}"
            
            print(f"[WARNING] Auto-corrected invalid {param_name} '{date_str}' to '{corrected_date}'. {month_name} {year} has only {max_days} days.")
            return corrected_date
        elif "day is out of range for month" in str(e):
            # No auto-correction, provide helpful error message
            month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            month_name = month_names[month] if 1 <= month <= 12 else f"month {month:02d}"
            max_days = get_max_days_in_month(year, month)
            
            raise ValueError(f"Invalid {param_name} '{date_str}'. {month_name} {year} has only {max_days} days. Try {year}-{month:02d}-{max_days:02d}")
        else:
            raise ValueError(f"Invalid {param_name} '{date_str}': {e}")

def calculate_psychometric_fit(x_values, y_values, min_samples=6):
    """
    Fit psychometric sigmoid curve for concentration vs complaints.
    
    Fits a 4-parameter sigmoid: y = ymin + (ymax-ymin) / (1 + exp(-(x-x50)/slope))
    where:
    - ymin: minimum response (forced to 0 for zero concentration = zero complaints)
    - ymax: maximum response (upper asymptote)
    - x50: concentration at 50% of maximum response (threshold)
    - slope: steepness of the curve
    
    Args:
        x_values (array-like): Concentration values (continuous)
        y_values (array-like): Complaint values (continuous)
        min_samples (int): Minimum number of samples required for analysis
        
    Returns:
        dict: Dictionary containing psychometric fit metrics and predictions
    """
    try:
        from scipy.optimize import curve_fit
        import numpy as np
        
        # Convert to numpy arrays and remove NaN values
        x = np.array(x_values).flatten()
        y = np.array(y_values).flatten()
        
        # Remove NaN/inf values and ensure non-negative
        valid_mask = np.isfinite(x) & np.isfinite(y) & (x >= 0) & (y >= 0)
        x = x[valid_mask]
        y = y[valid_mask]
        
        # Check if we have enough samples and variation
        if len(x) < min_samples or len(np.unique(x)) < 3 or len(np.unique(y)) < 2:
            return None
        
        # Define 3-parameter sigmoid (constrained to pass through origin)
        def psychometric_sigmoid(x, ymax, x50, slope):
            """3-parameter sigmoid: ymin=0 (forced), ymax, x50 (threshold), slope"""
            return ymax / (1 + np.exp(-(x - x50) / slope))
        
        # Initial parameter estimates
        ymax_init = np.max(y) * 1.1  # Slightly above max
        x50_init = np.median(x)      # Threshold at median concentration
        slope_init = (np.max(x) - np.min(x)) / 4  # Reasonable slope
        
        # Parameter bounds (ymax>0, x50>0, slope>0)
        bounds = ([0.1, 0.001, 0.001], [np.max(y) * 2, np.max(x) * 2, np.max(x)])
        
        try:
            # Fit the psychometric curve
            params, covariance = curve_fit(
                psychometric_sigmoid, x, y,
                p0=[ymax_init, x50_init, slope_init],
                bounds=bounds,
                maxfev=2000
            )
            
            ymax, x50, slope = params
            
            # Generate smooth curve for plotting (including zero)
            x_plot = np.linspace(0, np.max(x) * 1.1, 100)
            y_pred_plot = psychometric_sigmoid(x_plot, ymax, x50, slope)
            
            # Calculate predictions for original data points
            y_pred = psychometric_sigmoid(x, ymax, x50, slope)
            
            # Calculate fit quality metrics
            # R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Root Mean Square Error
            rmse = np.sqrt(np.mean((y - y_pred) ** 2))
            
            # Normalized RMSE
            nrmse = rmse / (np.max(y) - np.min(y)) if (np.max(y) - np.min(y)) > 0 else float('inf')
            
            # Calculate parameter uncertainties from covariance
            param_errors = np.sqrt(np.diag(covariance))
            
            # Threshold metrics
            threshold_10 = x50 - slope * np.log(9)  # 10% of max response
            threshold_90 = x50 + slope * np.log(9)  # 90% of max response
            dynamic_range = threshold_90 - threshold_10
            
            print(f"   [PSYCHOMETRIC] Fitted sigmoid: ymax={ymax:.2f}, x50={x50:.3f}, slope={slope:.3f}")
            print(f"   [PSYCHOMETRIC] Thresholds: 10%={threshold_10:.3f}, 50%={x50:.3f}, 90%={threshold_90:.3f} μg/m³")
            
            return {
                'fit_type': 'psychometric_sigmoid',
                'parameters': {'ymax': ymax, 'x50_threshold': x50, 'slope': slope},
                'parameter_errors': {'ymax_err': param_errors[0], 'x50_err': param_errors[1], 'slope_err': param_errors[2]},
                'r_squared': r_squared,
                'rmse': rmse,
                'nrmse': nrmse,
                'n_samples': len(x),
                'sigmoid_x': x_plot,
                'sigmoid_y': y_pred_plot,
                'predicted_values': y_pred,
                'thresholds': {
                    'threshold_10': max(0, threshold_10),
                    'threshold_50': x50,
                    'threshold_90': threshold_90,
                    'dynamic_range': dynamic_range
                },
                'passes_origin': True,  # By design
                'max_response': ymax
            }
            
        except (RuntimeError, ValueError) as fit_error:
            print(f"   [PSYCHOMETRIC] Curve fitting failed: {fit_error}")
            return None
        
    except Exception as e:
        print(f"   [WARN] Psychometric curve fitting failed: {e}")
        return None

# Hosmer-Lemeshow function removed - no longer needed for psychometric fitting

# PDF conversion imports
try:
    import pdfkit
    HAS_PDFKIT = True
except ImportError:
    HAS_PDFKIT = False

# Note: weasyprint disabled due to Windows library dependencies
HAS_WEASYPRINT = False

if not (HAS_PDFKIT or HAS_WEASYPRINT):
    print("[INFO]  PDF conversion will use Chrome/Edge headless (no additional libraries needed)")

# PCA analysis imports
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, log_loss, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from scipy.stats import pearsonr, chisquare

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
    
    # Import bootstrap error estimation
    try:
        from esat.error.bootstrap import Bootstrap
        from esat.data.datahandler import DataHandler
        HAS_BOOTSTRAP = True
        print("[OK] ESAT Bootstrap imported successfully")
    except ImportError as e:
        HAS_BOOTSTRAP = False
        print(f"[WARNING] ESAT Bootstrap not available: {e}")
except ImportError:
    print("[ERROR] ESAT library not found. Please install it using:")
    print("   $env:CARGO_BUILD_TARGET = \"x86_64-pc-windows-msvc\"")
    print("   pip install git+https://github.com/quanted/esat.git")
    print("\nAlternatively, you can install dependencies:")
    print("   pip install scikit-learn numpy pandas matplotlib seaborn")
    sys.exit(1)

class ColorManager:
    """Manages consistent colors for factors and species across all PMF plots."""
    
    def __init__(self, n_factors, species_names, factor_profiles=None):
        self.n_factors = n_factors
        self.species_names = species_names
        self.factor_profiles = factor_profiles
        self.h2s_factor_idx = None
        
        # Identify H2S-dominant factor if profiles are provided
        self._identify_h2s_factor()
        
        # Define consistent color schemes
        self.factor_colors = self._get_factor_colors(n_factors)
        self.species_colors = self._get_species_colors(species_names)
        
    def _identify_h2s_factor(self):
        """Identify which factor has the highest H2S contribution."""
        if self.factor_profiles is None:
            return
        
        # Find H2S column index
        h2s_col_idx = None
        for i, species in enumerate(self.species_names):
            if 'H2S' in species.upper():
                h2s_col_idx = i
                break
        
        if h2s_col_idx is not None:
            # Find factor with highest H2S contribution
            h2s_contributions = self.factor_profiles[:, h2s_col_idx]
            self.h2s_factor_idx = int(np.argmax(h2s_contributions))
            print(f"[COLOR] H2S-dominant factor identified: Factor {self.h2s_factor_idx + 1} (H2S contribution: {h2s_contributions[self.h2s_factor_idx]:.3f})")
        else:
            print("[COLOR] H2S species not found in dataset - using standard coloring")
        
    def _get_factor_colors(self, n_factors):
        """Get consistent colors for PMF factors with H2S-dominant factor in red."""
        # Define color palette EXCLUDING red - red is reserved for H2S factor only
        non_red_colors = [
            '#1f77b4',  # Blue
            '#ff7f0e',  # Orange
            '#2ca02c',  # Green
            '#9467bd',  # Purple
            '#8c564b',  # Brown
            '#e377c2',  # Pink
            '#7f7f7f',  # Gray
            '#bcbd22',  # Olive
            '#17becf',  # Cyan
            '#ffbb78',  # Light Orange
            '#98df8a',  # Light Green
            '#c5b0d5',  # Light Purple
            '#c49c94',  # Light Brown
            '#f7b6d3',  # Light Pink
            '#c7c7c7',  # Light Gray
            '#dbdb8d',  # Light Olive
            '#9edae5',  # Light Cyan
            '#ff9896',  # Light Red (but not pure red)
            '#aec7e8',  # Light Blue
            '#ffcc99'   # Peach
        ]
        
        # Use appropriate number of colors based on factors needed
        if n_factors <= len(non_red_colors):
            factor_colors = non_red_colors[:n_factors]
        else:
            # For many factors, use matplotlib colormap but exclude red range
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
            
            # Generate colors avoiding red hues (0.8-1.2 in HSV space)
            factor_colors = []
            for i in range(n_factors):
                hue = (i / n_factors) * 0.8  # Use only 0-0.8 of hue range to avoid red
                color = mcolors.hsv_to_rgb([hue, 0.8, 0.8])
                factor_colors.append(mcolors.rgb2hex(color))
        
        # Now assign red color EXCLUSIVELY to H2S-dominant factor
        if self.h2s_factor_idx is not None:
            red_color = '#d62728'  # Standard matplotlib red - RESERVED FOR H2S ONLY
            factor_colors[self.h2s_factor_idx] = red_color
            print(f"[COLOR] Factor {self.h2s_factor_idx + 1} assigned red color (H2S-dominant) - red reserved exclusively for H2S factor")
        
        return factor_colors
    
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
    
    def get_factor_plot_order(self):
        """Get factor indices in plotting order with H2S-dominant factor last."""
        factor_order = list(range(self.n_factors))
        
        # Move H2S factor to the end for top-layer plotting
        if self.h2s_factor_idx is not None:
            factor_order.remove(self.h2s_factor_idx)
            factor_order.append(self.h2s_factor_idx)
            
        return factor_order
    
    def is_h2s_factor(self, factor_idx):
        """Check if the given factor index is the H2S-dominant factor."""
        return self.h2s_factor_idx is not None and factor_idx == self.h2s_factor_idx

class MMFPMFAnalyzer:
    def __init__(self, station=None, data_dir=None, patterns=None, start_date=None, end_date=None, output_dir="pmf_results", 
                 remove_voc=False, uncertainty_mode='legacy', uncertainty_ef_mdl=None, uncertainty_epsilon=1e-12,
                 legacy_min_u=0.1, uncertainty_bdl_policy='five-sixth-mdl', snr_enable=False, snr_weak_threshold=2.0,
                 snr_bad_threshold=0.2, snr_bdl_weak_frac=0.6, snr_bdl_bad_frac=0.8, snr_missing_weak_frac=0.2,
                 snr_missing_bad_frac=0.4, exclude_bad=True, dashboard_snr_panel=True, write_diagnostics=True,
                 scale_units=True, seed=42, robust_fit=False, robust_alpha=4.0,
                 method="ls-nmf", init_method="column_mean", init_norm=True, hold_h=False, delay_h=-1, 
                 species_weight=None, exclude_species=None, weight_aware_init=None,
                 reg_species=None, reg_lambda=None, reg_template=None, reg_template_files=None,
                 reg_bursts=5, reg_iter_per_burst=50, reg_tol=1e-4, reg_elastic_l1=0.0,
                 # Bootstrap error estimation parameters
                 bootstrap=False, bootstrap_n=100, bootstrap_block_size=None, bootstrap_threshold=0.6,
                 bootstrap_parallel=True, bootstrap_cpus=None, bootstrap_seed=None, bootstrap_keep_h=True,
                 bootstrap_reuse_seed=True, bootstrap_overlapping=False,
                 # Complaint correlation analysis parameters
                 complaint_correlation_hours=0, complaint_window_method='average'):
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
            scale_units (bool): Apply unit standardization (mg/m3->ug/m3, ng/m3->ug/m3)
            seed (int): Random seed for reproducibility
            robust_fit (bool): Enable ESAT robust loss during SA training (single-model fallback)
            robust_alpha (float): Robust cutoff alpha for uncertainty-scaled residuals
            method (str): ESAT NMF algorithm ('ls-nmf' for nonnegative, 'ws-nmf' for semi-NMF)
            init_method (str): Matrix initialization method ('column_mean' or 'kmeans')
            init_norm (bool): Whiten data before kmeans initialization (for magnitude balance)
            hold_h (bool): Hold H (profile) matrix constant during training
            delay_h (int): Hold H matrix for N iterations, then release (-1 = disabled)
            species_weight (list): List of species uncertainty multipliers (e.g., ['CH4=5', 'H2S=2'])
            exclude_species (list): List of species to exclude from PMF analysis entirely (e.g., ['CH4', 'H2S'])
            weight_aware_init (bool): Enable weight-aware initialization for weighted species (None = auto-detect from species_weight)
            reg_species (list): List of species names to regularize (e.g., ['CH4', 'H2S'])
            reg_lambda (list): Regularization strength lambda per species (broadcast if single value)
            reg_template (list): Template types per species ('zero', 'uniform', 'from-file')
            reg_template_files (list): CSV file paths for from-file templates
            reg_bursts (int): Number of train->prox cycles for regularization
            reg_iter_per_burst (int): Max iterations per training burst
            reg_tol (float): Early stop tolerance for relative change in regulated columns
            reg_elastic_l1 (float): Elastic-net L1 penalty on deviation from h0
            
            # Bootstrap error estimation parameters
            bootstrap (bool): Enable bootstrap error estimation after PMF analysis
            bootstrap_n (int): Number of bootstrap samples to run (default: 100)
            bootstrap_block_size (int): Block size for temporal bootstrap resampling (None = auto-estimate)
            bootstrap_threshold (float): Factor mapping threshold for bootstrap correlation (default: 0.6)
            bootstrap_parallel (bool): Enable parallel processing for bootstrap (default: True)
            bootstrap_cpus (int): Number of CPUs for bootstrap parallel processing (None = use all)
            bootstrap_seed (int): Random seed for bootstrap resampling (None = use main seed)
            bootstrap_keep_h (bool): Keep factor profiles (H matrix) from bootstrap samples (default: True)
            bootstrap_reuse_seed (bool): Reuse seed across bootstrap samples for deterministic resampling
            bootstrap_overlapping (bool): Allow overlapping blocks in bootstrap resampling (default: True)
            
            # Complaint correlation analysis parameters
            complaint_correlation_hours (int): Time window in hours for complaint correlation analysis.
                Default 0 uses daily aggregation. Positive values correlate complaints with 
                ±N hours of concentration data around each complaint timestamp.
            complaint_window_method (str): Statistical aggregation method for data within complaint
                correlation time windows: 'peak' (maximum), 'average' (mean), 'median', 'mode'
                (most frequent), 'range' (max-min). Only used when complaint_correlation_hours > 0.
        """
        self.station = station
        self.data_dir = data_dir
        self.patterns = patterns
        
        # Validate date formats before storing them
        self.start_date = validate_date_string(start_date, "start_date")
        self.end_date = validate_date_string(end_date, "end_date")
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
            print("[WARN]  Warning: delay_h specified without hold_h. Setting hold_h=True for consistency.")
            self.hold_h = True
        
        # Species-specific uncertainty weighting
        self.species_weight_raw = species_weight or []
        self.species_weight_dict = self._parse_species_weights(self.species_weight_raw)
        
        # Species exclusion from PMF analysis
        self.exclude_species_raw = exclude_species or []
        self.exclude_species_set = self._parse_species_exclusions(self.exclude_species_raw)
        
        # Weight-aware initialization control (auto-enable if species weights are applied)
        if weight_aware_init is None:
            self.weight_aware_init = bool(self.species_weight_dict)  # Auto-enable if weights specified
        else:
            self.weight_aware_init = bool(weight_aware_init)
        
        if self.weight_aware_init and not self.species_weight_dict:
            print("[WARN] Warning: Weight-aware initialization enabled but no species weights specified")
        elif self.weight_aware_init:
            print(f"[INIT] Weight-aware initialization enabled for {len(self.species_weight_dict)} weighted species")
        
        # Species regularization control
        self.reg_species_raw = reg_species or []
        self.reg_lambda_raw = reg_lambda or []
        self.reg_template_raw = reg_template or []
        self.reg_template_files_raw = reg_template_files or []
        self.reg_bursts = int(reg_bursts)
        self.reg_iter_per_burst = int(reg_iter_per_burst)
        self.reg_tol = float(reg_tol)
        self.reg_elastic_l1 = float(reg_elastic_l1)
        
        # Validate and normalize regularization parameters
        self._validate_regularization_parameters()
        
        # Regularization plan will be computed in _prepare_regularization()
        self._reg_plan = []
        self._reg_enabled = bool(self.reg_species_raw)
        
        # Bootstrap error estimation parameters
        self.bootstrap = bool(bootstrap)
        self.bootstrap_n = int(bootstrap_n) if bootstrap_n > 0 else 100
        self.bootstrap_block_size = int(bootstrap_block_size) if bootstrap_block_size is not None and bootstrap_block_size > 0 else None
        self.bootstrap_threshold = float(bootstrap_threshold) if 0 < bootstrap_threshold <= 1 else 0.6
        self.bootstrap_parallel = bool(bootstrap_parallel)
        # Default to CPU-1 for performance (leave one CPU free)
        if bootstrap_cpus is None and bootstrap_parallel:
            import multiprocessing as mp
            total_cpus = mp.cpu_count()
            self.bootstrap_cpus = max(1, total_cpus - 1)  # Use CPU-1, minimum 1
        else:
            self.bootstrap_cpus = int(bootstrap_cpus) if bootstrap_cpus is not None and bootstrap_cpus > 0 else None
        self.bootstrap_seed = int(bootstrap_seed) if bootstrap_seed is not None else None
        self.bootstrap_keep_h = bool(bootstrap_keep_h)
        self.bootstrap_reuse_seed = bool(bootstrap_reuse_seed)  # Default: True for stability
        self.bootstrap_overlapping = bool(bootstrap_overlapping)  # Default: False per ESAT recommendations
        
        # Bootstrap results storage
        self.bootstrap_results = None
        
        # Complaint correlation analysis parameters
        self.complaint_correlation_hours = int(complaint_correlation_hours) if complaint_correlation_hours >= 0 else 0
        self.complaint_window_method = complaint_window_method if complaint_window_method in ['peak', 'average', 'median', 'mode', 'range'] else 'average'
    
    def _aggregate_window_data(self, data_window, method='average'):
        """Apply statistical aggregation to data within a complaint time window.
        
        Args:
            data_window (pd.DataFrame): Data within the time window
            method (str): Aggregation method - 'peak', 'average', 'median', 'mode', 'range'
            
        Returns:
            pd.Series: Aggregated values for each column
        """
        if len(data_window) == 0:
            return pd.Series(index=data_window.columns, dtype=float)
            
        try:
            if method == 'peak':
                # Maximum value
                return data_window.max()
            elif method == 'average':
                # Mean value (default)
                return data_window.mean()
            elif method == 'median':
                # Median value
                return data_window.median()
            elif method == 'mode':
                # Most frequent value (for continuous data, use median as fallback)
                mode_result = data_window.mode()
                if len(mode_result) > 0:
                    return mode_result.iloc[0]  # Take first mode if multiple exist
                else:
                    return data_window.median()  # Fallback to median
            elif method == 'range':
                # Range (max - min)
                return data_window.max() - data_window.min()
            else:
                # Default to average
                return data_window.mean()
        except Exception as e:
            # Fallback to mean if any aggregation method fails
            print(f"[WARN] Aggregation method '{method}' failed, using mean: {e}")
            return data_window.mean()
    
    def _calculate_window_uncertainty(self, data_window, aggregated_value, method='average'):
        """Calculate appropriate uncertainty measure for the aggregated data.
        
        Args:
            data_window (pd.DataFrame): Data within the time window
            aggregated_value (pd.Series): The aggregated values
            method (str): Aggregation method used
            
        Returns:
            pd.Series: Uncertainty measures for each column
        """
        if len(data_window) == 0:
            return pd.Series(index=data_window.columns, dtype=float)
            
        try:
            if method == 'peak':
                # For peak, use standard deviation as uncertainty measure
                return data_window.std()
            elif method == 'average':
                # For average, standard deviation is appropriate
                return data_window.std()
            elif method == 'median':
                # For median, use MAD (Median Absolute Deviation) scaled to std
                mad = data_window.sub(data_window.median()).abs().median()
                return mad * 1.4826  # Scale factor to approximate standard deviation
            elif method == 'mode':
                # For mode, use standard deviation
                return data_window.std()
            elif method == 'range':
                # For range, uncertainty is the interquartile range
                return data_window.quantile(0.75) - data_window.quantile(0.25)
            else:
                # Default to standard deviation
                return data_window.std()
        except Exception as e:
            # Fallback to standard deviation if calculation fails
            print(f"[WARN] Uncertainty calculation for method '{method}' failed, using std: {e}")
            return data_window.std()
    
    def _parse_species_weights(self, species_weight_list):
        """Parse species weight specifications into a dictionary.
        
        Args:
            species_weight_list (list): List of weight specifications (e.g., ['CH4=5', 'H2S=2,NO2=3'])
            
        Returns:
            dict: Mapping of species names (case-insensitive) to weight factors
        """
        weight_dict = {}
        
        if not species_weight_list:
            return weight_dict
        
        for spec in species_weight_list:
            # Handle comma-separated multiple weights in single spec
            for item in spec.split(','):
                item = item.strip()
                if '=' not in item:
                    print(f"Warning: Invalid species weight specification '{item}' (missing '='). Skipping.")
                    continue
                    
                parts = item.split('=', 1)  # Split on first '=' only
                if len(parts) != 2:
                    print(f"Warning: Invalid species weight specification '{item}'. Skipping.")
                    continue
                    
                species_name = parts[0].strip()
                weight_str = parts[1].strip()
                
                # Validate species name
                if not species_name:
                    print(f"Warning: Empty species name in '{item}'. Skipping.")
                    continue
                    
                # Parse weight factor
                try:
                    weight_factor = float(weight_str)
                    if weight_factor <= 0:
                        print(f"Warning: Weight factor must be > 0 for {species_name} (got {weight_factor}). Skipping.")
                        continue
                    if weight_factor > 100:
                        print(f"Warning: Very large weight factor for {species_name} ({weight_factor}). This may cause numerical instability.")
                    
                    # Store with case-insensitive key (convert to uppercase for matching)
                    key = species_name.upper()
                    if key in weight_dict:
                        print(f"Warning: Duplicate species weight for {species_name}. Using latest value: {weight_factor}")
                    weight_dict[key] = weight_factor
                    
                except ValueError:
                    print(f"Warning: Invalid weight factor '{weight_str}' for {species_name}. Skipping.")
                    continue
        
        return weight_dict
    
    def _parse_species_exclusions(self, exclude_species_list):
        """Parse species exclusion specifications into a set.
        
        Args:
            exclude_species_list (list): List of species to exclude (e.g., ['CH4', 'H2S,NO2'])
            
        Returns:
            set: Set of species names (case-insensitive uppercase) to exclude
        """
        exclude_set = set()
        
        if not exclude_species_list:
            return exclude_set
        
        for spec in exclude_species_list:
            # Handle comma-separated multiple species in single spec
            for item in spec.split(','):
                species_name = item.strip()
                
                # Validate species name
                if not species_name:
                    print(f"Warning: Empty species name in exclusion list. Skipping.")
                    continue
                
                # Store with case-insensitive key (convert to uppercase for matching)
                key = species_name.upper()
                if key in exclude_set:
                    print(f"Warning: Duplicate species exclusion for {species_name}. Ignoring duplicate.")
                else:
                    exclude_set.add(key)
                    print(f"[OK] Species scheduled for exclusion: {species_name}")
        
        return exclude_set
    
    def _validate_regularization_parameters(self):
        """Validate and normalize regularization parameters with broadcasting.
        
        This method handles:
        - Broadcasting singleton values to all species
        - Validating list length consistency
        - Checking template/file combinations
        - Providing helpful error messages
        """
        if not self.reg_species_raw:
            return  # No regularization, nothing to validate
            
        n_species = len(self.reg_species_raw)
        
        # Helper function to broadcast singleton lists
        def broadcast_list(lst, name, default_value=None):
            if not lst:
                if default_value is not None:
                    return [default_value] * n_species
                else:
                    raise ValueError(f"{name} list is empty but species are specified for regularization")
            elif len(lst) == 1:
                print(f"[BROADCAST] Broadcasting single {name} value {lst[0]} to {n_species} species")
                return lst * n_species
            elif len(lst) == n_species:
                return lst
            else:
                raise ValueError(
                    f"{name} list length ({len(lst)}) must be 1 (broadcast) or {n_species} (per-species). "
                    f"Species: {self.reg_species_raw}"
                )
        
        # Validate and broadcast lists
        try:
            self.reg_lambda_norm = broadcast_list(self.reg_lambda_raw, "lambda", 1.0)
            self.reg_template_norm = broadcast_list(self.reg_template_raw, "template", "zero")
            
            # Validate lambda values
            for i, lam in enumerate(self.reg_lambda_norm):
                if lam <= 0:
                    raise ValueError(f"Lambda value must be > 0 for species {self.reg_species_raw[i]} (got {lam})")
                if lam > 1e6:
                    print(f"[WARN] Warning: Very large lambda {lam} for {self.reg_species_raw[i]} may cause numerical issues")
            
            # Validate template choices
            valid_templates = {'zero', 'uniform', 'from-file'}
            for i, template in enumerate(self.reg_template_norm):
                if template not in valid_templates:
                    raise ValueError(
                        f"Invalid template '{template}' for species {self.reg_species_raw[i]}. "
                        f"Must be one of: {valid_templates}"
                    )
            
            # Handle template files for from-file templates
            from_file_count = sum(1 for t in self.reg_template_norm if t == 'from-file')
            if from_file_count > 0:
                if len(self.reg_template_files_raw) != from_file_count:
                    raise ValueError(
                        f"Need {from_file_count} template files for 'from-file' templates, "
                        f"but got {len(self.reg_template_files_raw)} file paths"
                    )
                # Check that files exist (will be done later in _prepare_regularization)
            
            print(f"[OK] Regularization parameters validated for {n_species} species")
            for i, species in enumerate(self.reg_species_raw):
                template_desc = self.reg_template_norm[i]
                if template_desc == 'from-file':
                    file_idx = sum(1 for j in range(i) if self.reg_template_norm[j] == 'from-file')
                    template_desc += f" ({self.reg_template_files_raw[file_idx]})"
                print(f"   {species}: lambda={self.reg_lambda_norm[i]}, template={template_desc}")
                
        except Exception as e:
            print(f"[ERROR] Regularization parameter validation failed: {e}")
            raise
    
    def _construct_regularization_template(self, template_type, template_file_path, factors, species_name):
        """Construct regularization template h0 for a specific species.
        
        Args:
            template_type (str): 'zero', 'uniform', or 'from-file'
            template_file_path (str): Path to CSV file for 'from-file' type
            factors (int): Number of factors (k)
            species_name (str): Species name for error messages
            
        Returns:
            numpy.ndarray: Template vector h0 of shape (factors,)
        """
        import numpy as np
        
        if template_type == 'zero':
            h0 = np.zeros(factors, dtype=np.float64)
            print(f"   {species_name}: Zero template (all zeros)")
            
        elif template_type == 'uniform':
            # Small uniform vector, sum-normalized to be tiny
            uniform_value = 1e-6  # Tiny positive value
            h0 = np.full(factors, uniform_value, dtype=np.float64)
            h0 = h0 / np.sum(h0) * uniform_value * factors  # Keep total mass small
            print(f"   {species_name}: Uniform template (value={uniform_value:.1e} each)")
            
        elif template_type == 'from-file':
            if not template_file_path:
                raise ValueError(f"Template file path required for 'from-file' template for {species_name}")
            
            try:
                import pandas as pd
                from pathlib import Path
                
                file_path = Path(template_file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"Template file not found: {template_file_path}")
                
                # Read CSV - expect single column with k rows
                df = pd.read_csv(file_path)
                if df.shape[1] != 1:
                    raise ValueError(f"Template file must have exactly 1 column, got {df.shape[1]} in {template_file_path}")
                if df.shape[0] != factors:
                    raise ValueError(f"Template file must have {factors} rows (one per factor), got {df.shape[0]} in {template_file_path}")
                
                h0 = df.iloc[:, 0].values.astype(np.float64)
                
                # Validate: must be non-negative and finite
                if np.any(h0 < 0):
                    raise ValueError(f"Template values must be non-negative in {template_file_path}")
                if not np.all(np.isfinite(h0)):
                    raise ValueError(f"Template values must be finite in {template_file_path}")
                
                print(f"   {species_name}: From-file template (range: {np.min(h0):.3e} to {np.max(h0):.3e})")
                
            except Exception as e:
                raise ValueError(f"Failed to load template file {template_file_path} for {species_name}: {e}")
                
        else:
            raise ValueError(f"Unknown template type '{template_type}' for {species_name}")
        
        # Final validation
        if h0.shape != (factors,):
            raise ValueError(f"Template shape mismatch: expected ({factors},), got {h0.shape} for {species_name}")
        if not np.all(np.isfinite(h0)):
            raise ValueError(f"Template contains non-finite values for {species_name}")
        if np.any(h0 < 0):
            raise ValueError(f"Template contains negative values for {species_name}")
        
        return h0
    
    def _prepare_regularization(self):
        """Prepare the regularization plan by mapping species to column indices and building templates.
        
        This method:
        - Maps regulated species names to concentration matrix column indices
        - Builds template vectors for each regulated species
        - Stores the complete regularization plan in self._reg_plan
        - Warns about species not found in data
        
        Should be called after prepare_pmf_data() when species list and factors are known.
        """
        if not self._reg_enabled:
            return
            
        print(f"[CONFIG] Preparing regularization for {len(self.reg_species_raw)} species...")
        
        # Map species names to concentration matrix column indices
        if not hasattr(self, 'concentration_data') or self.concentration_data is None:
            raise RuntimeError("Cannot prepare regularization before concentration data is loaded")
            
        species_columns = list(self.concentration_data.columns)
        name_to_idx = {col.upper(): j for j, col in enumerate(species_columns)}
        
        print(f"   Available species columns: {species_columns}")
        
        # Build regularization plan
        self._reg_plan = []
        file_counter = 0  # Track from-file template file index
        
        for i, species in enumerate(self.reg_species_raw):
            species_upper = species.upper()
            
            # Check if species exists in data
            if species_upper not in name_to_idx:
                print(f"   [WARN] Warning: Regularization target '{species}' not found in data. Skipping.")
                continue
                
            col_idx = name_to_idx[species_upper]
            lambda_val = self.reg_lambda_norm[i]
            template_type = self.reg_template_norm[i]
            
            # Handle template file path for from-file templates
            template_file_path = None
            if template_type == 'from-file':
                if file_counter >= len(self.reg_template_files_raw):
                    print(f"   [ERROR] Error: Missing template file for {species}. Skipping.")
                    continue
                template_file_path = self.reg_template_files_raw[file_counter]
                file_counter += 1
            
            # Build template (requires factors to be set)
            if not hasattr(self, 'factors') or self.factors is None:
                raise RuntimeError("Cannot prepare regularization before number of factors is determined")
                
            try:
                h0 = self._construct_regularization_template(
                    template_type, template_file_path, self.factors, species
                )
                
                # Store in regularization plan
                reg_item = {
                    'species': species,
                    'species_upper': species_upper,
                    'col_idx': col_idx,
                    'lambda': lambda_val,
                    'template_type': template_type,
                    'template_file': template_file_path,
                    'h0': h0
                }
                self._reg_plan.append(reg_item)
                
                print(f"   [OK] Mapped {species} to column {col_idx}: lambda={lambda_val}, template={template_type}")
                
            except Exception as e:
                print(f"   [ERROR] Error preparing template for {species}: {e}. Skipping.")
                continue
        
        if not self._reg_plan:
            print(f"   [WARN] Warning: No valid regularization targets found. Regularization will be disabled.")
            self._reg_enabled = False
        else:
            print(f"   [INIT] Regularization prepared for {len(self._reg_plan)} species")
            
        return len(self._reg_plan)
    
    def _compute_uncertainty_weights(self, U_col, species_name):
        """Compute uncertainty weights (We = 1/U^2) for a species column with numerical guards.
        
        Args:
            U_col (array-like): Uncertainty values for a single species (n_samples,)
            species_name (str): Species name for error messages
            
        Returns:
            numpy.ndarray: Uncertainty weights We of shape (n_samples,)
            
        This method implements the guards specified in the plan:
        - Replace 0/NaN/inf U with species median positive U
        - Ensure We stays finite
        - Add epsilon floors in denominators
        """
        import numpy as np
        
        # Convert to float64 array
        u = np.asarray(U_col, dtype=np.float64)
        
        # Step 1: Identify problematic values (zero, NaN, inf)
        problematic_mask = ~(np.isfinite(u) & (u > 0))
        n_problematic = np.sum(problematic_mask)
        
        if n_problematic > 0:
            # Find replacement value: median of positive finite values
            good_values = u[~problematic_mask]
            if len(good_values) > 0:
                replacement_value = np.median(good_values)
                print(f"   [WARN] {species_name}: Found {n_problematic}/{len(u)} problematic U values, replacing with median={replacement_value:.3e}")
            else:
                # All values are problematic - use a reasonable default
                replacement_value = 1.0  # Conservative uncertainty
                print(f"   [WARN] {species_name}: All U values problematic, using default={replacement_value}")
            
            # Replace problematic values
            u_clean = u.copy()
            u_clean[problematic_mask] = replacement_value
        else:
            u_clean = u
            print(f"   [OK] {species_name}: All {len(u)} uncertainty values are positive and finite")
        
        # Step 2: Compute weights with epsilon floor to avoid division issues
        epsilon = 1e-12  # Minimum uncertainty floor
        u_safe = np.maximum(u_clean, epsilon)
        
        # We = 1 / U^2 
        we = 1.0 / np.square(u_safe)
        
        # Step 3: Final validation
        if not np.all(np.isfinite(we)):
            print(f"   [ERROR] {species_name}: Non-finite weights after computation - this should not happen")
            # Emergency fallback: uniform weights
            we = np.ones_like(we, dtype=np.float64)
            
        # Report statistics
        print(f"   [DATA] {species_name} weights: min={np.min(we):.3e}, median={np.median(we):.3e}, max={np.max(we):.3e}")
        
        return we
    
    def _ridge_proximal_update(self, W, V_col, U_col, h_current, lambda_val, h0, species_name):
        """Perform closed-form ridge regularization proximal update for a single species column.
        
        Solves: (W^T D W + lambda I) h = W^T D v + lambda h0
        Then projects: h <- max(h, 0)
        
        Args:
            W (np.ndarray): Factor contributions matrix (n_samples, k)
            V_col (np.ndarray): Concentration data for species (n_samples,)
            U_col (np.ndarray): Uncertainty data for species (n_samples,)
            h_current (np.ndarray): Current H column for species (k,)
            lambda_val (float): Regularization strength
            h0 (np.ndarray): Template vector (k,)
            species_name (str): Species name for logging
            
        Returns:
            np.ndarray: Updated H column h_new (k,)
            
        This implements the closed-form solution from the mathematical plan:
        - Compute D = diag(We) where We = 1/U^2
        - Solve normal equations with regularization
        - Project to nonnegativity
        - Validate objective decrease
        """
        import numpy as np
        
        # Convert inputs to float64 for numerical stability
        W = np.asarray(W, dtype=np.float64)
        V_col = np.asarray(V_col, dtype=np.float64)
        h_current = np.asarray(h_current, dtype=np.float64)
        h0 = np.asarray(h0, dtype=np.float64)
        
        k = W.shape[1]  # Number of factors
        n = W.shape[0]  # Number of samples
        
        # Step 1: Compute uncertainty weights D = diag(We)
        we = self._compute_uncertainty_weights(U_col, species_name)
        
        # Step 2: Compute normal equations components efficiently
        # A = W^T D W + lambda I
        # b = W^T D v + lambda h0
        
        # Vectorized computation: W^T * we (broadcasting)
        WT_scaled = W.T * we  # Shape: (k, n)
        
        # A = WT_scaled @ W + lambda * I
        A = WT_scaled @ W + lambda_val * np.eye(k, dtype=np.float64)
        
        # b = WT_scaled @ V_col + lambda * h0
        b = WT_scaled @ V_col + lambda_val * h0
        
        print(f"   [NUMBERS] {species_name}: Solving ({k}x{k}) system, lambda={lambda_val}, cond(A)={np.linalg.cond(A):.2e}")
        
        # Step 3: Solve linear system with numerical stability checks
        try:
            # Check condition number
            cond_A = np.linalg.cond(A)
            if cond_A > 1e12:
                print(f"   [WARN] {species_name}: High condition number {cond_A:.2e}, adding jitter")
                # Add small diagonal jitter for numerical stability
                jitter = 1e-12 * np.trace(A) / k
                A += jitter * np.eye(k)
                
            # Solve normal equations
            h_new = np.linalg.solve(A, b)
            
        except np.linalg.LinAlgError as e:
            print(f"   [ERROR] {species_name}: Linear solve failed: {e}. Using fallback.")
            # Fallback: use current h (no update)
            h_new = h_current.copy()
            
        # Step 4: Project to nonnegativity
        h_new_proj = np.maximum(h_new, 0.0)
        n_negative = np.sum(h_new < 0)
        if n_negative > 0:
            print(f"   [SYMBOL]️ {species_name}: Projected {n_negative}/{k} negative values to zero")
            
        # Step 5: Validate objective decrease (as specified in plan)
        objective_before = self._ridge_objective(W, V_col, we, h_current, lambda_val, h0)
        objective_after = self._ridge_objective(W, V_col, we, h_new_proj, lambda_val, h0)
        
        if objective_after > objective_before + 1e-9 * max(1.0, objective_before):
            print(f"   [WARN] {species_name}: Objective increased {objective_before:.3e} -> {objective_after:.3e}, keeping current h")
            h_new_proj = h_current.copy()
        else:
            obj_decrease = objective_before - objective_after
            rel_decrease = obj_decrease / max(1e-12, objective_before)
            print(f"   [SYMBOL] {species_name}: Objective decreased by {obj_decrease:.3e} ({rel_decrease:.2%})")
            
        # Step 6: Report update statistics
        delta_norm = np.linalg.norm(h_new_proj - h_current)
        rel_change = delta_norm / (np.linalg.norm(h_current) + 1e-12)
        
        print(f"   [RESULTS] {species_name}: ||h_new - h_old||={delta_norm:.3e}, rel_change={rel_change:.3e}")
        
        return h_new_proj
    
    def _ridge_objective(self, W, V_col, we, h, lambda_val, h0):
        """Compute ridge regularized objective value for validation.
        
        Objective: 0.5 * ||sqrt(D)(v - Wh)||^2 + 0.5 * lambda * ||h - h0||^2
        """
        import numpy as np
        
        # Data fidelity term: 0.5 * sum(we * (v - Wh)^2)
        residual = V_col - W @ h
        data_term = 0.5 * np.sum(we * residual**2)
        
        # Regularization term: 0.5 * lambda * ||h - h0||^2
        reg_term = 0.5 * lambda_val * np.sum((h - h0)**2)
        
        return data_term + reg_term
    
    def _train_with_regularization(self, sa_model, V, U, species_names):
        """Execute staged training loop with regularization proximal updates.
        
        This implements the core regularization algorithm:
        1. Train ESAT model for reg_iter_per_burst iterations
        2. Extract W, H matrices  
        3. Apply proximal updates to regulated species columns in H
        4. Re-initialize ESAT with updated H, W matrices
        5. Repeat until convergence or max bursts reached
        
        Args:
            sa_model: ESAT SA model instance
            V (np.ndarray): Concentration matrix (n_samples, n_species)
            U (np.ndarray): Uncertainty matrix (n_samples, n_species) 
            species_names (list): Species names matching V columns
            
        Returns:
            bool: True if training completed successfully
            
        This method forces single SA mode and Python update path as planned.
        """
        import numpy as np
        
        if not self._reg_enabled or not self._reg_plan:
            raise RuntimeError("Cannot run regularized training: regularization not enabled or prepared")
            
        print(f"[PROC] Starting staged regularization training: {self.reg_bursts} bursts, {self.reg_iter_per_burst} iter/burst")
        print(f"   Regularizing {len(self._reg_plan)} species: {[item['species'] for item in self._reg_plan]}")
        print(f"   Convergence tolerance: {self.reg_tol}")
        
        # Initialize Stage 9 diagnostics if available
        try:
            from regularization_diagnostics import create_regularization_diagnostics
            self._reg_diagnostics = create_regularization_diagnostics()
            diagnostics_available = True
            print(f"   [DIAG] Stage 9 regularization diagnostics enabled")
        except ImportError:
            self._reg_diagnostics = None
            diagnostics_available = False
            print(f"   [INFO] Stage 9 diagnostics module not available")
        
        # Storage for burst diagnostics
        burst_diagnostics = []
        
        # Main staged training loop
        for burst_idx in range(self.reg_bursts):
            print(f"\n[RUN] Burst {burst_idx + 1}/{self.reg_bursts}:")
            
            # Step 1: Train ESAT model for one burst
            print(f"   [LEARN] Training ESAT for {self.reg_iter_per_burst} iterations...")
            
            # Record Q values before training for diagnostics
            q_before = (float(getattr(sa_model, 'Qtrue', np.nan)) if getattr(sa_model, 'Qtrue', None) is not None else np.nan, 
                       float(getattr(sa_model, 'Qrobust', np.nan)) if getattr(sa_model, 'Qrobust', None) is not None else np.nan)
            
            try:
                sa_model.train(
                    max_iter=self.reg_iter_per_burst,
                    robust_mode=self.robust_fit, 
                    robust_alpha=self.robust_alpha
                )
                print(f"   [SYMBOL] ESAT training completed")
            except Exception as e:
                print(f"   [ERROR] ESAT training failed: {e}")
                return False
                
            # Record Q values after training for diagnostics
            q_after = (float(getattr(sa_model, 'Qtrue', np.nan)) if getattr(sa_model, 'Qtrue', None) is not None else np.nan, 
                      float(getattr(sa_model, 'Qrobust', np.nan)) if getattr(sa_model, 'Qrobust', None) is not None else np.nan)
            
            # Step 2: Extract current W, H matrices
            try:
                W = sa_model.W.astype(np.float64)
                H = sa_model.H.astype(np.float64) 
                
                print(f"   [RESULTS] Extracted matrices: W{W.shape}, H{H.shape}")
                
                # Validate matrix shapes
                if W.shape != (V.shape[0], self.factors):
                    raise ValueError(f"W shape mismatch: expected ({V.shape[0]}, {self.factors}), got {W.shape}")
                if H.shape != (self.factors, V.shape[1]):
                    raise ValueError(f"H shape mismatch: expected ({self.factors}, {V.shape[1]}), got {H.shape}")
                    
            except Exception as e:
                print(f"   [ERROR] Matrix extraction failed: {e}")
                return False
            
            # Step 3: Apply proximal updates to regulated species
            max_rel_change = 0.0
            burst_species_changes = []
            
            print(f"   [INIT] Applying proximal updates to {len(self._reg_plan)} regulated species...")
            
            for reg_item in self._reg_plan:
                species = reg_item['species']
                col_idx = reg_item['col_idx']
                lambda_val = reg_item['lambda']
                h0 = reg_item['h0']
                
                # Extract current H column for this species
                h_current = H[:, col_idx].copy()
                
                # Extract V, U columns for this species
                V_col = V[:, col_idx]
                U_col = U[:, col_idx]
                
                # Apply proximal update
                try:
                    h_updated = self._ridge_proximal_update(
                        W, V_col, U_col, h_current, lambda_val, h0, species
                    )
                    
                    # Compute relative change
                    change_norm = np.linalg.norm(h_updated - h_current)
                    rel_change = change_norm / (np.linalg.norm(h_current) + 1e-12)
                    
                    # Update H matrix
                    H[:, col_idx] = h_updated
                    
                    # Track changes
                    max_rel_change = max(max_rel_change, rel_change)
                    burst_species_changes.append({
                        'species': species,
                        'col_idx': col_idx,
                        'lambda': lambda_val,
                        'change_norm': change_norm,
                        'rel_change': rel_change,
                        'h_norm': np.linalg.norm(h_updated),
                        'h_to_template_dist': np.linalg.norm(h_updated - h0)
                    })
                    
                    print(f"     [SYMBOL] {species}: rel_change={rel_change:.3e}, ||h||={np.linalg.norm(h_updated):.3e}")
                    
                except Exception as e:
                    print(f"     [ERROR] {species}: Proximal update failed: {e}")
                    # Continue with other species
                    continue
            
            # Step 4: Re-initialize ESAT with updated matrices
            print(f"   [PROC] Re-initializing ESAT with updated H matrix...")
            try:
                # Use ESAT's public initialize method with updated H, W
                sa_model.initialize(
                    H=H.astype(np.float64),
                    W=W.astype(np.float64), 
                    init_method='column_mean',  # Ensure ESAT doesn't override
                    init_norm=self.init_norm
                )
                
                # Force Python update path to avoid dtype issues
                sa_model.optimized = False
                
                # Verify matrices were set correctly
                H_check = sa_model.H.astype(np.float64)
                W_check = sa_model.W.astype(np.float64)
                
                h_diff = np.max(np.abs(H_check - H))
                w_diff = np.max(np.abs(W_check - W))
                
                if h_diff > 1e-10 or w_diff > 1e-10:
                    print(f"   [WARN] Warning: Matrix roundtrip error - H diff: {h_diff:.2e}, W diff: {w_diff:.2e}")
                else:
                    print(f"   [SYMBOL] Matrix re-initialization successful")
                    
            except Exception as e:
                print(f"   [ERROR] ESAT re-initialization failed: {e}")
                return False
            
            # Step 5: Record burst diagnostics
            burst_diag = {
                'burst': burst_idx + 1,
                'max_rel_change': max_rel_change,
                'n_species_updated': len(burst_species_changes),
                'Qtrue': float(getattr(sa_model, 'Qtrue', np.nan)),
                'Qrobust': float(getattr(sa_model, 'Qrobust', np.nan)),
                'species_changes': burst_species_changes
            }
            burst_diagnostics.append(burst_diag)
            
            print(f"   [SUMMARY] Burst {burst_idx + 1} summary: max_rel_change={max_rel_change:.3e}, Qtrue={burst_diag['Qtrue']:.3f}")
            
            # Stage 9: Track convergence diagnostics for each regulated species
            if diagnostics_available and self._reg_diagnostics:
                for reg_item in self._reg_plan:
                    species = reg_item['species']
                    lambda_val = reg_item['lambda']
                    
                    # Calculate objective reduction for this species (approximate)
                    obj_reduction = 0.0
                    if not np.isnan(q_before[1]) and not np.isnan(q_after[1]):
                        obj_reduction = q_before[1] - q_after[1]
                    
                    # Track convergence for this burst
                    self._reg_diagnostics.track_convergence(
                        burst_num=burst_idx + 1,
                        species_name=species,
                        lambda_val=lambda_val,
                        q_start=q_before,
                        q_end=q_after,
                        obj_reduction=obj_reduction,
                        rel_change=max_rel_change,  # Use max across all species
                        converged=(max_rel_change < self.reg_tol),
                        iterations=self.reg_iter_per_burst
                    )
            
            # Step 6: Check convergence
            if max_rel_change < self.reg_tol:
                print(f"   [OK] Regularization converged in burst {burst_idx + 1}: max_rel_change={max_rel_change:.3e} < tol={self.reg_tol}")
                break
            else:
                print(f"   [CONTINUE] Continuing: max_rel_change={max_rel_change:.3e} > tol={self.reg_tol}")
        
        # Store diagnostics for later saving
        self._reg_burst_diagnostics = burst_diagnostics
        
        # Stage 9: Generate regularization diagnostics if available
        if diagnostics_available and self._reg_diagnostics:
            try:
                # Save convergence and diagnostic data
                self._reg_diagnostics.save_diagnostics_csv(self.output_dir, self.filename_prefix)
                
                # Generate convergence plots
                conv_plot = self._reg_diagnostics.generate_convergence_plots(self.output_dir, self.filename_prefix)
                if conv_plot:
                    print(f"   [SAVE] Convergence plots: {conv_plot}")
                
                # Generate diagnostic summary report  
                report = self._reg_diagnostics.generate_diagnostic_summary_report(self.output_dir, self.filename_prefix)
                if report:
                    print(f"   [SAVE] Diagnostic report: {report}")
                    
            except Exception as e:
                print(f"   [WARN] Stage 9 diagnostics generation failed: {e}")
        
        print(f"\n[COMPLETE] Staged regularization training completed:")
        print(f"   Total bursts: {len(burst_diagnostics)}")
        print(f"   Final max_rel_change: {max_rel_change:.3e}")
        print(f"   Converged: {'Yes' if max_rel_change < self.reg_tol else 'No'}")
        
        return True
    
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
        print("[INFO] MMF PMF SOURCE APPORTIONMENT ANALYSIS (FIXED)")
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
            print(f"[STATION] Station: {display_name}")
        else:
            # Flexible data directory mode
            print(f"[DATA] Data Directory: {self.data_dir}")
            print(f"[SEARCH] Patterns: {self.patterns}")
            
        if self.start_date or self.end_date:
            date_info = f"{self.start_date or 'All'} to {self.end_date or 'All'}"
            print(f"[DATE] Analysis Period: {date_info}")
        print(f"[FOLDER] Output Directory: {self.output_dir}")
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
        
        print(f"[SEARCH] Found {len(unique_files)} matching parquet file(s):")
        for f in unique_files:
            print(f"   - {f.name}")
        
        return unique_files
    
    def load_mmf_data(self):
        """Load and prepare MMF data for PMF analysis."""
        # Display information banner
        self._display_station_info()
        
        if self.station:
            # Legacy mode: station-based loading
            print(f"[SEARCH] Loading MMF data for {self.station}...")
            try:
                parquet_file = get_mmf_parquet_file(self.station, use_test_data=True)
            except Exception as e:
                raise RuntimeError(f"Error determining file path for {self.station}: {e}")
            
            if not parquet_file.exists():
                raise FileNotFoundError(f"Corrected parquet file not found: {parquet_file}")
            
            analyzer = ParquetAnalyzer(parquet_file)
            if not analyzer.load_data():
                raise RuntimeError("Failed to load parquet data")
        else:
            # Flexible mode: data directory and patterns
            print(f"[SEARCH] Loading parquet data from {self.data_dir}...")
            parquet_files = self._find_parquet_files()
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found matching patterns: {self.patterns}")
            
            # For now, use the first file (could be extended to merge multiple files)
            parquet_file = parquet_files[0]
            print(f"[FILE] Using file: {parquet_file.name}")
            
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
                print(f"[META] Aggregation metadata: timebase={self.aggregation_timebase}, method={self.aggregation_method}, min_valid={self.min_valid_subsamples}")
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
        
        print(f"[OK] Loaded {len(self.df):,} records")
        # Report date range based on where datetime information is located
        if hasattr(self.df.index, 'min') and hasattr(self.df.index, 'max'):
            try:
                print(f"[DATE] Date range: {self.df.index.min()} to {self.df.index.max()}")
            except:
                print(f"[DATE] Index range: {self.df.index[0]} to {self.df.index[-1]}")
        elif 'datetime' in self.df.columns:
            print(f"[DATE] Date range: {self.df['datetime'].min()} to {self.df['datetime'].max()}")
        else:
            print(f"[DATE] No datetime information available")
    
    def _filter_date_range(self):
        """Filter data by specified date range."""
        original_len = len(self.df)
        
        # Check if datetime is in the index (new behavior) or as a column (legacy)
        if hasattr(self.df.index, 'min') and hasattr(self.df.index, 'max'):
            # Datetime is in the index - filter by index
            if self.start_date:
                start_dt = pd.to_datetime(self.start_date)
                self.df = self.df[self.df.index >= start_dt]
            
            if self.end_date:
                end_dt = pd.to_datetime(self.end_date)
                # If start and end dates are the same, include the full 24 hours of that day
                if self.start_date and self.start_date == self.end_date:
                    # Add 23:59:59 to include the entire day
                    end_dt = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    print(f"[DATE] Same start/end date detected - including full 24 hours of {self.start_date}")
                self.df = self.df[self.df.index <= end_dt]
        elif 'datetime' in self.df.columns:
            # Legacy: datetime is a column
            if self.start_date:
                start_dt = pd.to_datetime(self.start_date)
                self.df = self.df[self.df['datetime'] >= start_dt]
            
            if self.end_date:
                end_dt = pd.to_datetime(self.end_date)
                # If start and end dates are the same, include the full 24 hours of that day
                if self.start_date and self.start_date == self.end_date:
                    # Add 23:59:59 to include the entire day
                    end_dt = end_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
                    print(f"[DATE] Same start/end date detected - including full 24 hours of {self.start_date}")
                self.df = self.df[self.df['datetime'] <= end_dt]
        else:
            print(f"[WARN] No datetime information found - cannot filter by date range")
            return
        
        filtered_len = len(self.df)
        print(f"[DATA] Date filtering: {filtered_len:,} records ({original_len - filtered_len:,} excluded)")
    
    def prepare_pmf_data(self):
        """
        Prepare concentration and uncertainty matrices for PMF analysis.
        Following EPA PMF 5.0 User Guide recommendations.
        """
        print("[CONFIG] Preparing PMF data matrices...")
        
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
            print(f"[OK] VOC species detected: {available_vocs}")
        elif available_vocs and self.remove_voc:
            print(f"[WARN] VOC species detected but excluded due to --remove-voc flag: {available_vocs}")
        
        # Select all applicable species for PMF analysis
        if self.remove_voc:
            all_species = gas_species + particle_species
            print(f"[EXCLUDE] VOC species excluded from PMF analysis")
        else:
            # Use actual detected VOC column names instead of generic names
            all_species = gas_species + particle_species + available_vocs
        
        # Select columns that exactly match target species (avoid auxiliary columns like 'n_*')
        pollutant_columns = [col for col in self.df.columns if col in all_species]
        
        # Apply species exclusions if specified
        if self.exclude_species_set:
            excluded_columns = []
            remaining_columns = []
            
            for col in pollutant_columns:
                if col.upper() in self.exclude_species_set:
                    excluded_columns.append(col)
                else:
                    remaining_columns.append(col)
            
            # Determine which requested species were not found
            found_species_upper = {col.upper() for col in pollutant_columns}
            not_found_species = [species for species in self.exclude_species_set 
                               if species not in found_species_upper]
            
            # Store results for provenance tracking
            self._excluded_species_applied = excluded_columns
            self._excluded_species_not_found = list(not_found_species)
            
            if excluded_columns:
                print(f"[EXCLUDE] Excluding {len(excluded_columns)} species from PMF analysis: {excluded_columns}")
                pollutant_columns = remaining_columns
            else:
                print(f"[WARN] No species matched exclusion list: {list(self.exclude_species_set)}")
            
            if not_found_species:
                print(f"[WARN] {len(not_found_species)} requested exclusions not found in data: {list(not_found_species)}")
        
        print(f"[SPECIES] Final pollutants for PMF: {pollutant_columns}")
        
        # Report data availability for different species types
        gas_cols = [col for col in pollutant_columns if any(gas in col for gas in gas_species)]
        voc_cols = [col for col in pollutant_columns if any(voc in col for voc in voc_species)]
        pm_cols = [col for col in pollutant_columns if any(pm in col for pm in particle_species)]
        
        print(f"[SPECIES] Species breakdown:")
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

        # Standardize all concentrations to ug/m3 prior to computing uncertainties (if enabled)
        if self.scale_units:
            self._standardize_units_to_ugm3(pollutant_columns)
        else:
            print("[UNITS] Unit standardization disabled (--no-scale-units). Units will be used as-is.")
        
        # Remove rows with too many missing values (EPA recommendation: >50% missing)
        missing_threshold = getattr(self, 'drop_row_threshold', 0.5)
        valid_rows = self.concentration_data.isnull().sum(axis=1) / len(pollutant_columns) < missing_threshold
        self.concentration_data = self.concentration_data[valid_rows]
        if self.counts_data is not None:
            self.counts_data = self.counts_data.loc[self.concentration_data.index]
        
        print(f"[DATA] After removing rows with >{missing_threshold*100:.1f}% missing: {len(self.concentration_data):,} records")
        
        # Generate uncertainty matrix following EPA guidelines
        self._generate_uncertainty_matrix(pollutant_columns)
        
        # Apply S/N-based feature categorization if enabled
        if self.snr_enable and HAS_SNR_CATEGORIZATION:
            self._apply_snr_categorization(pollutant_columns)
        elif self.snr_enable and not HAS_SNR_CATEGORIZATION:
            print("[WARN] S/N categorization requested but module not available. Proceeding without categorization.")
        
        # Apply species-specific uncertainty weighting if specified
        if self.species_weight_dict:
            self._apply_species_weighting()
        
        # Save species exclusions provenance if any exclusions were specified
        if self.exclude_species_set:
            self._save_species_exclusions_csv()
        
        # Save processed data
        self._save_processed_data()
    
    def _safe_unicode_clean(self, text):
        """Clean Unicode characters for Windows cp1252 compatibility."""
        if text is None:
            return 'unknown'
        try:
            # Convert to string and clean common problematic Unicode chars
            s = str(text)
            # Replace Greek mu (μ) with latin u
            s = s.replace('\u03bc', 'u')
            # Replace superscript 2 and 3
            s = s.replace('\u00b2', '2').replace('\u00b3', '3') 
            # Replace arrow characters
            s = s.replace('\u2192', '->')
            # Try to encode/decode to catch other problematic characters
            s = s.encode('ascii', errors='replace').decode('ascii')
            return s
        except Exception:
            return 'unknown'
    
    def _compute_closure_metrics(self, V, U, W, H, species_names):
        """Compute species-level closure metrics for mass balance analysis.
        
        Args:
            V (np.ndarray): Concentration matrix (n_samples, n_species)
            U (np.ndarray): Uncertainty matrix (n_samples, n_species) 
            W (np.ndarray): Factor contributions (n_samples, n_factors)
            H (np.ndarray): Factor profiles (n_factors, n_species)
            species_names (list): Species names matching V columns
            
        Returns:
            tuple: (closure_df, group_summary) where closure_df is a DataFrame with
                   per-species metrics and group_summary is a dict with group closures
        """
        import numpy as np
        import pandas as pd
        
        # Guard against numerical issues
        eps = 1e-12
        U_safe = np.maximum(U, eps)
        
        # Reconstruct data matrix
        R = W @ H  # (n_samples, n_species)
        
        # Compute residuals
        res = V - R
        
        # Uncertainty weights
        w = 1.0 / (U_safe ** 2)
        
        # Per-species metrics
        meas_sum = np.maximum(np.sum(V, axis=0), eps)
        reco_sum = np.sum(R, axis=0)
        closure_pct = 100.0 * reco_sum / meas_sum
        
        # Uncertainty-weighted closure
        meas_w_sum = np.maximum(np.sum(w * V, axis=0), eps)
        reco_w_sum = np.sum(w * R, axis=0)
        closure_w_pct = 100.0 * reco_w_sum / meas_w_sum
        
        # Species-wise Q contribution
        Q_species = np.sum(((V - R) / U_safe) ** 2, axis=0)
        total_Q = np.maximum(np.sum(Q_species), eps)
        q_share_pct = 100.0 * Q_species / total_Q
        
        # RMSE and normalized RMSE
        rmse = np.sqrt(np.mean(res ** 2, axis=0))
        mean_conc = np.maximum(np.mean(V, axis=0), eps)
        nrmse = 100.0 * rmse / mean_conc
        
        # Median residuals
        med_res = np.median(res, axis=0)
        
        # Create DataFrame
        closure_df = pd.DataFrame({
            'species': species_names,
            'measured_sum': meas_sum,
            'reconstructed_sum': reco_sum,
            'closure_pct': closure_pct,
            'closure_w_pct': closure_w_pct,
            'Q_species': Q_species,
            'q_share_pct': q_share_pct,
            'rmse': rmse,
            'nrmse': nrmse,
            'median_residual': med_res
        })
        
        # Compute group closures (avoid mixed-unit issues)
        def group_closure(species_list, name):
            """Compute closure for a group of species."""
            indices = [i for i, sp in enumerate(species_names) if sp in species_list]
            if not indices:
                return name, np.nan, 0
            
            group_meas = np.sum(V[:, indices])
            group_reco = np.sum(R[:, indices])
            group_closure = 100.0 * group_reco / np.maximum(group_meas, eps)
            
            return name, group_closure, len(indices)
        
        # Define species groups
        gas_species = ['CH4', 'NOX', 'NO', 'NO2', 'SO2', 'H2S']
        voc_species = ['Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene']
        pm_species = [sp for sp in species_names if 'FIDAS' in sp]
        
        # Compute group closures
        group_summary = {}
        for group_name, species_list in [('Gases', gas_species), ('VOCs', voc_species), ('PM', pm_species)]:
            name, closure, count = group_closure(species_list, group_name)
            if count > 0:
                group_summary[name] = {'closure_pct': closure, 'n_species': count}
        
        return closure_df, group_summary
    
    def _plot_closure_summary(self, closure_df, group_summary, dashboard_dir):
        """Create closure summary plot highlighting regularized species.
        
        Args:
            closure_df (pd.DataFrame): Per-species closure metrics
            group_summary (dict): Group-level closure summary
            dashboard_dir (Path): Dashboard directory for saving plots
            
        Returns:
            Path: Path to the saved plot file
        """
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Set matplotlib to non-interactive mode
        plt.ioff()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Get species and closure data
        species = closure_df['species'].values
        closure_pct = closure_df['closure_pct'].values
        closure_w_pct = closure_df['closure_w_pct'].values
        q_share_pct = closure_df['q_share_pct'].values
        
        # Default colors
        colors = ['#1f77b4'] * len(species)  # Blue for normal species
        
        # Highlight regularized species in red
        reg_targets = set()
        if hasattr(self, '_reg_enabled') and self._reg_enabled and hasattr(self, '_reg_plan'):
            reg_targets = set([ri['species'] for ri in self._reg_plan])
        
        for i, sp in enumerate(species):
            if sp in reg_targets:
                colors[i] = '#d62728'  # Red for regularized species
        
        # Create bar plot
        x = np.arange(len(species))
        bars = ax.bar(x, closure_pct, color=colors, alpha=0.7, label='Closure (%)', edgecolor='black', linewidth=0.5)
        
        # Add uncertainty-weighted closure as line
        ax.plot(x, closure_w_pct, color='black', marker='o', linewidth=2, markersize=4, 
                label='Weighted Closure (%)', alpha=0.8)
        
        # Add reference line at 100%
        ax.axhline(100.0, color='gray', linestyle='--', alpha=0.6, label='Perfect Closure')
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels(species, rotation=45, ha='right')
        ax.set_ylabel('Closure (%)')
        ax.set_title('Species-Level Mass Closure (Reconstructed / Measured)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text annotation for regularized species
        if reg_targets:
            reg_species_str = ', '.join(sorted(reg_targets))
            ax.text(0.02, 0.98, f'Regularized: {reg_species_str}', transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Add group closure summary in text box
        if group_summary:
            group_text = 'Group Closures:\n'
            for group, data in group_summary.items():
                group_text += f'{group}: {data["closure_pct"]:.1f}% ({data["n_species"]} spp)\n'
            
            ax.text(0.98, 0.98, group_text.strip(), transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        plot_file = dashboard_dir / f"{self.filename_prefix}_closure_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return plot_file
    
    def _normalize_unit_string(self, unit_str):
        """Normalize unit strings to a canonical form for comparison."""
        if unit_str is None:
            return None
        s = str(unit_str).strip().lower()
        # Handle Greek letters and Unicode characters
        s = s.replace('μ', 'u')
        # Handle superscript characters (³ → 3, ² → 2)
        s = s.replace('³', '3')
        s = s.replace('²', '2')
        # Remove spaces and parentheses
        s = s.replace(' ', '')
        s = s.replace('(', '').replace(')', '')
        # Normalize common unit variants
        s = s.replace('ugm3', 'ug/m3')
        s = s.replace('mgm3', 'mg/m3')
        s = s.replace('ngm3', 'ng/m3')
        return s

    def _standardize_units_to_ugm3(self, pollutant_columns):
        """
        Ensure all concentration columns are in ug/m3 before uncertainty calc.
        - If a column is reported in mg/m3, multiply by 1000
        - If in ng/m3, divide by 1000
        - Update self.units to 'ug/m3'
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
                continue  # already ug/m3 or unknown (leave as-is)
            elif unit_norm == 'mg/m3':
                factor = 1000.0
            elif unit_norm == 'ng/m3':
                factor = 0.001
            elif unit_norm in ('ppm', 'ppb'):
                safe_unit = self._safe_unicode_clean(orig_unit)
                print(f"[WARN] Units for {col} are in {safe_unit}; no automatic ppm/ppb -> ug/m3 conversion applied.")
                continue
            else:
                # Unrecognized units - sanitize unicode for Windows compatibility
                safe_unit = self._safe_unicode_clean(orig_unit)
                print(f"[WARN] Unrecognized unit '{safe_unit}' for {col}; leaving values unchanged.")
                continue
            if factor is not None:
                self.concentration_data[col] = self.concentration_data[col] * factor
                self.units[col] = 'ug/m3'
                conversions.append((col, orig_unit, factor))
        if conversions:
            print("[CYCLE] Standardized units to ug/m3 for the following columns:")
            for col, ou, f in conversions:
                # Sanitize units for Windows compatibility
                safe_ou = self._safe_unicode_clean(ou)
                print(f"  - {col}: {safe_ou} -> ug/m3 (x{f})")

    def _apply_snr_categorization(self, pollutant_columns):
        """
        Apply S/N-based feature categorization following EPA PMF 5.0 guidelines.
        
        This categorizes species as 'strong', 'weak', or 'bad' based on S/N ratio
        and data quality metrics. Weak species have their uncertainty tripled,
        while bad species are excluded from the analysis entirely.
        """
        print("[NUMBERS] Applying S/N-based feature categorization...")
        
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
        print("[PROC] Applying categorization to uncertainty matrix...")
        
        # Collect species to exclude
        excluded_species = []
        
        for species, category in categories.items():
            if category == 'weak':
                # Triple uncertainties for weak species (EPA PMF 5.0 recommendation)
                print(f"   [WARN] {species}: Weak - tripling uncertainty")
                self.uncertainty_data[species] = self.uncertainty_data[species] * 3.0
            elif category == 'bad' and self.exclude_bad:
                # Mark bad species for exclusion from analysis
                print(f"   [ERROR] {species}: Bad - excluding from analysis")
                excluded_species.append(species)
        
        # Remove bad species from data matrices
        if excluded_species:
            print(f"   [REMOVE] Removing {len(excluded_species)} bad species from matrices: {excluded_species}")
            self.concentration_data = self.concentration_data.drop(columns=excluded_species)
            self.uncertainty_data = self.uncertainty_data.drop(columns=excluded_species)
            
            # Also update counts data if available
            if self.counts_data is not None:
                # Drop corresponding n_* columns for excluded species
                count_cols_to_drop = [f"n_{species}" for species in excluded_species if f"n_{species}" in self.counts_data.columns]
                if count_cols_to_drop:
                    self.counts_data = self.counts_data.drop(columns=count_cols_to_drop)
                    print(f"   [REMOVE] Removed corresponding count columns: {count_cols_to_drop}")
        
        # Save diagnostics if requested
        if self.write_diagnostics:
            categorizer.save_diagnostics(self.output_dir, self.filename_prefix)
        
        # Summary report
        summary = categorizer.get_summary()
        if summary:
            print(f"\n[DATA] S/N Categorization Summary:")
            print(f"   Total species: {summary['total_species']}")
            print(f"   Strong: {summary['strong_count']} species")
            print(f"   Weak: {summary['weak_count']} species") 
            print(f"   Bad: {summary['bad_count']} species")
            print(f"   Average S/N: {summary['average_snr']:.3f}")
            print(f"   Thresholds: weak={self.snr_weak_threshold}, bad={self.snr_bad_threshold}")
    
    def _apply_species_weighting(self):
        """
        Apply species-specific uncertainty multipliers to downweight selected species.
        
        This multiplies uncertainties for specified species, which effectively
        downweights them in the ESAT LS-PMF objective function. Applied after
        S/N categorization and before saving data for ESAT.
        """
        print("[INIT] Applying species-specific uncertainty weighting...")
        
        # Build mapping of actual column names to weight factors
        species_weights_applied = {}
        species_weights_not_found = {}
        
        for species_key, weight_factor in self.species_weight_dict.items():
            # Find matching column name (case-insensitive)
            matched_column = None
            for col in self.uncertainty_data.columns:
                if col.upper() == species_key.upper():
                    matched_column = col
                    break
            
            if matched_column:
                # Apply weight factor to uncertainty column
                original_uncertainty = self.uncertainty_data[matched_column].copy()
                self.uncertainty_data[matched_column] = original_uncertainty * weight_factor
                species_weights_applied[matched_column] = weight_factor
                print(f"   [OK] {matched_column}: Uncertainty multiplied by {weight_factor}")
            else:
                species_weights_not_found[species_key] = weight_factor
                print(f"   [WARN] {species_key}: Species not found in data (skipped)")
        
        # Store results for provenance tracking
        self._species_weights_applied = species_weights_applied
        self._species_weights_not_found = species_weights_not_found
        
        # Summary
        if species_weights_applied:
            print(f"   [DATA] Applied weights to {len(species_weights_applied)} species")
        if species_weights_not_found:
            print(f"   [WARN] {len(species_weights_not_found)} requested species not found in data")
        
        # Write species weights CSV for provenance
        self._save_species_weights_csv()
    
    def _save_species_weights_csv(self):
        """Save species weight application results to CSV for provenance."""
        species_weights_data = []
        
        # Add applied weights
        for species, weight in getattr(self, '_species_weights_applied', {}).items():
            species_weights_data.append({
                'species': species,
                'multiplier': weight,
                'was_present': True,
                'applied': True
            })
        
        # Add not found species
        for species, weight in getattr(self, '_species_weights_not_found', {}).items():
            species_weights_data.append({
                'species': species,
                'multiplier': weight,
                'was_present': False,
                'applied': False
            })
        
        if species_weights_data:
            weights_df = pd.DataFrame(species_weights_data)
            weights_file = self.output_dir / f"{self.filename_prefix}_species_weights.csv"
            weights_df.to_csv(weights_file, index=False)
            print(f"   [SAVE] Species weights saved: {weights_file.name}")
    
    def _save_species_exclusions_csv(self):
        """Save species exclusion application results to CSV for provenance."""
        species_exclusion_data = []
        
        # Add excluded species (from current state)
        excluded_species = getattr(self, '_excluded_species_applied', [])
        for species in excluded_species:
            species_exclusion_data.append({
                'species': species,
                'was_present': True,
                'excluded': True,
                'reason': 'CLI exclusion flag'
            })
        
        # Add requested species that were not found
        not_found_species = getattr(self, '_excluded_species_not_found', [])
        for species in not_found_species:
            species_exclusion_data.append({
                'species': species,
                'was_present': False,
                'excluded': False,
                'reason': 'Species not found in data'
            })
        
        if species_exclusion_data:
            exclusions_df = pd.DataFrame(species_exclusion_data)
            exclusions_file = self.output_dir / f"{self.filename_prefix}_species_exclusions.csv"
            exclusions_df.to_csv(exclusions_file, index=False)
            print(f"   [SAVE] Species exclusions saved: {exclusions_file.name}")
    
    def _generate_uncertainty_matrix(self, pollutant_columns):
        """
        Generate uncertainty matrix using EPA or legacy mode based on configuration.
        
        EPA Mode: Uses EPA formulas with optional aggregation scaling
        Legacy Mode: Original implementation with fixed MDL/EF table
        """
        print(f"[ANALYSIS] Generating uncertainty matrix (mode: {self.uncertainty_mode})...")
        
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
        print("[GEOMETRY] Using EPA uncertainty formulas...")
        
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
                print(f"   [DATA] Loaded aggregation counts for scaling")
            else:
                print(f"   [INFO] No aggregation counts available - EPA formulas only")
        except Exception as e:
            print(f"   [WARN] Could not load aggregation counts: {e}")
        
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
                
                # Prevent division by zero when no data records exist
                if total > 0:
                    print(f"   [OK] {species}: EF={EF:.3f}, MDL={MDL:.1f} | measured={n_meas} ({n_meas/total*100:.1f}%), "
                          f"BDL={n_bdl} ({n_bdl/total*100:.1f}%), missing={n_missing} ({n_missing/total*100:.1f}%)")
                else:
                    print(f"   [WARN] {species}: EF={EF:.3f}, MDL={MDL:.1f} | No data records available (total=0)")
            else:
                # Fallback for species not handled by EPA calculator
                print(f"   [WARN] {species}: No EPA data, using default uncertainty")
                self.uncertainty_data[species] = 1.0
        
        # Handle non-pollutant columns (if any)
        for col in self.concentration_data.columns:
            if col not in pollutant_columns:
                self.uncertainty_data[col] = 1.0  # Default uncertainty
        
        print(f"[ANALYSIS] EPA uncertainty matrix completed for {len(pollutant_columns)} species")
        
    def _generate_legacy_uncertainty_matrix(self, pollutant_columns):
        """
        Generate uncertainty matrix using original implementation.
        All MDL values below are specified in ug/m3 to match standardized V.
        
        EPA Formula: σ = sqrt((error_fraction * concentration)2 + (MDL)2)
        """
        
        # EPA-recommended MDL values and error fractions by pollutant type (all in ug/m3)
        # Based on typical instrument specifications and EPA guidance
        mdl_values = {
            'H2S': 0.5,      # ug/m3
            'CH4': 50.0,     # ug/m3 (converted from 0.05 mg/m3 -> 50 ug/m3)
            'SO2': 0.5,      # ug/m3
            'NOX': 1.0,      # ug/m3
            'NO': 0.5,       # ug/m3
            'NO2': 1.0,      # ug/m3
            'PM1 FIDAS': 1.0,    # ug/m3
            'PM1': 1.0,          # ug/m3
            'PM2.5 FIDAS': 1.0,  # ug/m3
            'PM2.5': 1.0,        # ug/m3
            'PM4 FIDAS': 1.5,    # ug/m3
            'PM4': 1.5,          # ug/m3
            'PM10 FIDAS': 2.0,   # ug/m3
            'PM10': 2.0,         # ug/m3
            'TSP FIDAS': 2.5,    # ug/m3
            'TSP': 2.5,          # ug/m3
            # VOC species (BTEX compounds) - typical GC-MS detection limits (ug/m3)
            'Benzene': 0.01,     # ug/m3
            'Toluene': 0.02,     # ug/m3
            'Ethylbenzene': 0.02,    # ug/m3
            'Xylene': 0.02,      # ug/m3 (covers m&p-Xylene)
            'm&p-Xylene': 0.02   # ug/m3 (specific for mixed isomers)
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
            mdl = 1.0  # Default MDL (ug/m3)
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
            safe_units = self._safe_unicode_clean(self.units.get(species, 'unknown'))
            print(f"  {species}: MDL={mdl}, Err={err_frac*100:.1f}%, min_u={self.legacy_min_u} | measured={n_meas} ({n_meas/total*100:.1f}%), "
                  f"BDL={n_bdl} ({n_bdl/total*100:.1f}%), missing={n_missing} ({n_missing/total*100:.1f}%) | Units={safe_units}")
    
    def _handle_missing_values(self):
        """Handle missing values following EPA Method 1."""
        print("[PROC] Handling missing values (EPA Method 1)...")
        
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
        print("[SEARCH] Final NaN check and removal...")
        
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
        
        print(f"  [OK] Data cleaning complete: {len(self.concentration_data)} valid records")
    
    def _save_processed_data(self):
        """Save processed concentration and uncertainty data."""
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        unc_file = self.output_dir / f"{self.filename_prefix}_uncertainties.csv"
        
        # Add datetime index for ESAT compatibility
        conc_data = self.concentration_data.copy()
        unc_data = self.uncertainty_data.copy()
        
        # Get corresponding datetime values
        if hasattr(self.df.index, 'min') and hasattr(self.df.index, 'max'):
            # Datetime is in the index - use index directly
            datetime_values = self.df.loc[self.concentration_data.index].index
        elif 'datetime' in self.df.columns:
            # Legacy: datetime is a column
            datetime_values = self.df.loc[self.concentration_data.index, 'datetime']
        else:
            # Fallback: use concentration data index as-is
            datetime_values = self.concentration_data.index
        
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
                print(f"[SAVE] Saved BDL mask: {bdl_file}")
                print(f"[SAVE] Saved Missing mask: {missing_file}")
        except Exception as e:
            print(f"[WARN] Could not save masks: {e}")

        # Optionally save counts if available
        try:
            if self.counts_data is not None and len(self.counts_data.columns) > 0:
                counts_out = self.counts_data.copy()
                counts_out.index = datetime_values
                counts_file = self.output_dir / f"{self.filename_prefix}_counts.csv"
                counts_out.to_csv(counts_file)
                print(f"[SAVE] Saved aggregation counts: {counts_file}")
        except Exception as e:
            print(f"[WARN] Could not save counts: {e}")
        
        print(f"[SAVE] Saved concentration data: {conc_file}")
        print(f"[SAVE] Saved uncertainty data: {unc_file}")
        
        # Save EPA-specific diagnostic files if requested
        if getattr(self, 'write_diagnostics', True) and self.uncertainty_mode == 'epa':
            try:
                epa_unc_file = self.output_dir / f"{self.filename_prefix}_uncertainties_epa.csv"
                unc_data.to_csv(epa_unc_file)
                print(f"[SAVE] Saved EPA uncertainties: {epa_unc_file}")
            except Exception as e:
                print(f"[WARN] Could not save EPA uncertainties: {e}")
        
        # Save S/N categorization mapping if available
        if getattr(self, 'write_diagnostics', True) and hasattr(self, '_snr_categories'):
            try:
                categories_file = self.output_dir / f"{self.filename_prefix}_categories.csv"
                categories_df = pd.DataFrame({
                    'species': list(self._snr_categories.keys()),
                    'category': list(self._snr_categories.values())
                })
                categories_df.to_csv(categories_file, index=False)
                print(f"[SAVE] Saved species categories: {categories_file}")
            except Exception as e:
                print(f"[WARN] Could not save S/N categories: {e}")
    
    def run_pmf_analysis(self):
        """
        Run PMF analysis using ESAT following EPA best practices.
        FIXED VERSION based on successful test.
        """
        print("[START] Starting PMF analysis...")
        
        # Get data files
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        unc_file = self.output_dir / f"{self.filename_prefix}_uncertainties.csv"
        
        # Load data directly into numpy arrays (bypass DataHandler issues)
        print("[DATA] Loading data matrices...")
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
                    print(f"[CALC] Applied legacy uncertainty scaling based on aggregation counts (method={self.aggregation_method})")
            except Exception as e:
                print(f"[WARN] Could not apply aggregation-based uncertainty scaling: {e}")
        else:
            print(f"[INFO] EPA mode: aggregation scaling already included in uncertainty calculation")
        
        print(f"[DATA] Data matrices: V={V.shape}, U={U.shape}")
        print(f"[INFO] Species: {', '.join(species_names)}")
        
        # Check if we have any data
        if V.size == 0 or U.size == 0:
            print("[ERROR] Error: No data available for PMF analysis!")
            print("   This could be due to:")
            print("   1. Date range outside available data")
            print("   2. All data filtered out due to missing values")
            print("   3. No valid pollutant species found")
            return False
        
        if V.shape[0] < 10:
            print(f"[WARN] Warning: Very few data points ({V.shape[0]}) - PMF results may be unreliable")
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
        
        print(f"[DATA] Final data validation:")
        print(f"  V range: [{np.min(V):.3f}, {np.max(V):.3f}]")
        print(f"  U range: [{np.min(U):.3f}, {np.max(U):.3f}]")
        print(f"  Valid data points: {np.sum(~np.isnan(V) & ~np.isnan(U))} / {V.size}")
        
        # Optimize number of factors (EPA recommendation: try multiple values)
        self._optimize_factors(V, U)
        
        # Prepare regularization after factors are determined
        if self._reg_enabled:
            try:
                n_reg_targets = self._prepare_regularization()
                if n_reg_targets == 0:
                    print("[WARN] Regularization disabled: no valid targets found")
                    self._reg_enabled = False
                else:
                    print(f"[OK] Regularization preparation complete: {n_reg_targets} targets mapped")
            except Exception as e:
                print(f"[ERROR] Regularization preparation failed: {e}")
                print("   Disabling regularization and continuing with standard PMF")
                self._reg_enabled = False
        
        # Check if regularization, robust mode, or weight-aware init is enabled - force single SA if needed
        use_batch_sa = USE_BATCH_SA
        if self._reg_enabled and USE_BATCH_SA:
            print("[REG] Regularization enabled: forcing single SA mode (BatchSA doesn't support proximal updates)")
            use_batch_sa = False
        elif self.robust_fit and USE_BATCH_SA:
            print("[WARN]  Robust mode requested: forcing single SA mode (BatchSA doesn't support robust training)")
            use_batch_sa = False
        elif self.weight_aware_init and self.species_weight_dict and USE_BATCH_SA:
            print("[INIT] Weight-aware initialization enabled: forcing single SA mode (BatchSA doesn't support custom initialization)")
            use_batch_sa = False
        
        # Run PMF models
        print(f"[PROC] Running {self.models} PMF models with {self.factors} factors...")
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
                
                print(f"[OK] Best model: #{best_idx}")
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
                # Use single SA model (manual implementation for regularization, robust mode, or weight-aware init)
                if self._reg_enabled:
                    print(f"[REG] Running regularized PMF training with staged proximal updates")
                    print(f"   -> Regularizing {len(self._reg_plan)} species over {self.reg_bursts} bursts")
                elif self.robust_fit:
                    print(f"[CONFIG] Running {self.models} SA models with ROBUST mode (alpha={self.robust_alpha})")
                    print("   -> Robust training will downweight outliers during optimization")
                elif self.weight_aware_init and self.species_weight_dict:
                    print(f"[INIT] Running {self.models} SA models with WEIGHT-AWARE initialization")
                    print("   -> Custom initialization accounts for species uncertainty weights")
                else:
                    print(f"[WARN] Running {self.models} SA models (BatchSA not available or disabled)")
                
                if self._reg_enabled:
                    # Regularized training: single model with staged proximal updates
                    print(f"   [START] Creating SA model for regularized training...")
                    
                    sa_model = SA(
                        V=V, U=U, 
                        factors=self.factors,
                        method=self.method,
                        seed=self.seed,
                        verbose=True  # More verbose for regularized runs
                    )
                    
                    # Initialize matrices (weight-aware or standard)
                    self._weight_aware_initialize(sa_model, species_names, V, U)
                    
                    # Run staged regularization training
                    regularization_success = self._train_with_regularization(sa_model, V, U, species_names)
                    
                    if not regularization_success:
                        print(f"[ERROR] Regularized training failed")
                        return False
                    
                    # Use the regularized model as the best model
                    best_model = sa_model
                    best_idx = 0
                    best_q_robust = sa_model.Qrobust
                    
                else:
                    # Run multiple SA models and select the best one (keep only best to save memory)
                    best_model = None
                    best_q_robust = float('inf')
                    best_idx = 0
                    
                    for model_idx in range(self.models):
                        print(f"   [PROC] Training model {model_idx + 1}/{self.models}...")
                        
                        # Create SA model with different seed for each run
                        model_seed = self.seed + model_idx if self.seed else None
                        sa_model = SA(
                            V=V, U=U, 
                            factors=self.factors,
                            method=self.method,  # Use configured method (ls-nmf or ws-nmf)
                            seed=model_seed,
                            verbose=False  # Reduce verbosity for multiple models
                        )
                        
                        # Initialize matrices with weight-aware method if enabled
                        self._weight_aware_initialize(sa_model, species_names, V, U)
                        
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
                
                print(f"[OK] Best model: #{best_idx + 1} (Q(robust)={best_q_robust:.2f})")
                print(f"   Q(true): {self.best_model.Qtrue:.2f}")
                print(f"   Q(robust): {self.best_model.Qrobust:.2f}")
            
            # Store species names for plotting
            self.species_names = species_names
            
            # Initialize color manager for consistent plotting (with H matrix for H2S factor identification)
            factor_profiles = self.best_model.H  # H matrix: factors x species
            self.color_manager = ColorManager(self.factors, self.species_names, factor_profiles)
            print(f"[UI] Initialized consistent color scheme for {self.factors} factors and {len(self.species_names)} species")
            
        except Exception as e:
            print(f"[ERROR] PMF analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def _weight_aware_initialize(self, sa_model, species_names, V, U):
        """
        Custom weight-aware initialization for ESAT SA models.
        
        This method implements a weight-aware k-means initialization that considers
        uncertainty weights when clustering data, providing better initial factor
        profiles when species have dramatically different uncertainty weights.
        
        Args:
            sa_model: ESAT SA model instance
            species_names (list): List of species names matching V columns
            V (np.ndarray): Concentration data matrix
            U (np.ndarray): Uncertainty data matrix
            
        Returns:
            None: Modifies sa_model.W and sa_model.H in place
        """
        if not self.weight_aware_init or not self.species_weight_dict:
            # Fallback to standard initialization
            sa_model.initialize(
                init_method=self.init_method,
                init_norm=self.init_norm
            )
            return
        
        print(f"[INIT] Applying weight-aware initialization...")
        
        # Create weight mapping for current species
        species_weights = np.ones(len(species_names))  # Default weight = 1
        
        for i, species in enumerate(species_names):
            species_upper = species.upper()
            if species_upper in self.species_weight_dict:
                weight_factor = self.species_weight_dict[species_upper]
                # Weight affects uncertainty: U_weighted = U_original * weight_factor
                # For initialization, we want to reduce the influence of high-weight (high-uncertainty) species
                # So we scale the concentration by inverse weight to balance magnitudes
                species_weights[i] = 1.0 / weight_factor
                print(f"  {species}: weight factor = {weight_factor:.1f}, init scaling = {species_weights[i]:.3f}")
        
        # Create weight-scaled concentration matrix for initialization
        # Scale each column (species) by its inverse weight to balance magnitudes
        V_scaled = V.copy()
        for i in range(V_scaled.shape[1]):
            V_scaled[:, i] = V_scaled[:, i] * species_weights[i]
        
        print(f"  Original V range: [{np.min(V):.3f}, {np.max(V):.3f}]")
        print(f"  Scaled V range: [{np.min(V_scaled):.3f}, {np.max(V_scaled):.3f}]")
        
        # Apply k-means clustering on weight-scaled data
        from scipy.cluster.vq import kmeans2, whiten
        
        # Use the same random seed as the SA model
        rng = np.random.default_rng(sa_model.seed)
        
        # Optionally whiten the scaled data if init_norm is enabled
        obs = V_scaled
        if self.init_norm:
            obs = whiten(obs=V_scaled)
            print(f"  Applied whitening: V range after whitening: [{np.min(obs):.3f}, {np.max(obs):.3f}]")
        
        try:
            # Perform k-means clustering
            centroids, clusters = kmeans2(data=obs, k=sa_model.factors, seed=sa_model.seed)
            print(f"  K-means clustering completed: {sa_model.factors} factors")
            
            # Initialize W (factor contributions) based on cluster assignments
            W = np.zeros(shape=(V.shape[0], sa_model.factors)) + (1.0 / sa_model.factors)
            for i, cluster_id in enumerate(clusters):
                # Add some noise to avoid identical contributions
                W[i, cluster_id] = rng.normal(1.0, 0.1, None)
            
            # Ensure non-negative W for ls-nmf
            if sa_model.method == "ls-nmf":
                W[W <= 0.0] = 1e-12
            
            # Initialize H (factor profiles) from centroids, scaled back to original units
            H = centroids.copy()
            
            # Scale H back to original concentration units by reversing the weight scaling
            for i in range(H.shape[1]):
                H[:, i] = H[:, i] / species_weights[i]
            
            # Ensure non-negative H
            H[H <= 0.0] = 1e-12
            
            # Normalize H profiles (sum to 1 for each species across factors)
            H = H / H.sum(axis=0)
            
            # Initialize ESAT model with the prepared matrices using the public API
            sa_model.initialize(
                H=H.astype(np.float64),
                W=W.astype(np.float64),
                init_method='column_mean',  # ensure ESAT does not override provided H/W
                init_norm=self.init_norm
            )
            
            # Force Python update path to avoid dtype mismatch in Rust-optimized update
            sa_model.optimized = False
            
            print("  Weight-aware initialization complete (via SA.initialize, Python update path)")
            print(f"    W shape: {W.shape}, range: [{np.min(W):.6f}, {np.max(W):.6f}]")
            print(f"    H shape: {H.shape}, range: [{np.min(H):.6f}, {np.max(H):.6f}]")
            
        except Exception as e:
            print(f"  [WARN] Weight-aware initialization failed: {e}")
            print(f"  Falling back to standard initialization")
            # Fallback to standard initialization
            sa_model.initialize(
                init_method=self.init_method,
                init_norm=self.init_norm
            )
    
    def _optimize_factors(self, V, U):
        """
        Optimize number of factors following EPA guidelines.
        Try different factor numbers and use quality criteria.
        """
        # Check if factors were explicitly set via CLI (not the default)
        if hasattr(self, 'user_specified_factors') and self.user_specified_factors:
            print(f"[NUMBERS] Using user-specified number of factors: {self.factors}")
            return
        
        print("[SEARCH] Optimizing number of factors...")
        
        # Check if BatchSA is available (considering robust mode override)
        use_batch_for_optimization = USE_BATCH_SA and not self.robust_fit
        
        if not use_batch_for_optimization:
            if self.robust_fit:
                print("[WARN] Skipping factor optimization (robust mode requires single SA)")
            else:
                print("[WARN] Skipping factor optimization (requires BatchSA)")
            self.factors = 4  # Use default
            return
        
        # Test range of factors (EPA recommends testing multiple values)
        # CRITICAL: Never allow n_factors = n_species (creates DoF = 0)
        # Must always leave at least 1 degree of freedom
        species_limit = V.shape[1] - 1  # Always leave at least 1 DoF
        max_factors = min(getattr(self, 'max_factors', 10), species_limit)
        
        if max_factors < 2:
            print(f"[WARN]  Warning: Only {V.shape[1]} species available - using 2 factors minimum")
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
                print(f"    [ERROR] Failed: {e}")
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
            print(f"[OPTIMIZE] Optimal factors selected: {self.factors} (Q/DoF = {q_dof_ratios[optimal_factors]:.3f})")
            
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
            print("[WARN]  Using default factor number: 4")
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
        # DOF = (samples × species) - (samples × factors) - (species × factors) + factors2
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
        print("\n[ANALYSIS] Q-Value Analysis (EPA PMF Guidelines):")
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
        print("  Q/DOF <= 1.5: Excellent fit")
        print("  Q/DOF <= 2.0: Good fit")
        print("  Q/DOF <= 3.0: Fair fit (may need refinement)")
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
        print("[DASHBOARD] Creating PMF dashboard...")
        
        # Suppress any potential matplotlib or numpy output during plotting
        import warnings
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        warnings.filterwarnings('ignore')  # Suppress warnings
        
        if not self.best_model:
            print("[ERROR] No model results available")
            return False
        
        # Create dashboard directory
        dashboard_dir = self.output_dir / "dashboard"
        dashboard_dir.mkdir(exist_ok=True)
        
        # Extract ESAT results (using correct attribute names)
        F_profiles = self.best_model.H  # Factor profiles (source signatures)  
        G_contributions = self.best_model.W  # Factor contributions (time series)
        
        print(f"[PLOTS] Creating plots...")
        print(f"   Factor profiles: {F_profiles.shape}")
        print(f"   Factor contributions: {G_contributions.shape}")
        
        # Save factor profiles (H matrix) for analysis
        try:
            factor_profiles_file = self.output_dir / f"{self.filename_prefix}_factor_profiles.csv"
            factor_names = [f"Factor_{i+1}" for i in range(F_profiles.shape[0])]
            
            profiles_df = pd.DataFrame(
                F_profiles, 
                index=factor_names, 
                columns=self.species_names
            )
            
            profiles_df.to_csv(factor_profiles_file, index=True)
            print(f"[SAVE] Saved factor profiles: {factor_profiles_file.name}")
        except Exception as e:
            print(f"[WARN] Could not save factor profiles: {e}")
        
        # Create basic PMF plots
        plot_files = []
        
        # Compute pressure derivative analysis has been moved to the pressure analysis section
        
        # Compute and save closure metrics for mass balance analysis
        try:
            print("   [CLOSURE] Computing species-level closure metrics...")
            
            # Load original concentration and uncertainty data
            conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
            unc_file = self.output_dir / f"{self.filename_prefix}_uncertainties.csv"
            
            conc_df = pd.read_csv(conc_file, index_col=0)
            unc_df = pd.read_csv(unc_file, index_col=0)
            
            V = conc_df.values  # (n_samples, n_species)
            U = unc_df.values   # (n_samples, n_species)
            
            # Compute closure metrics
            closure_df, group_summary = self._compute_closure_metrics(
                V, U, G_contributions, F_profiles, self.species_names
            )
            
            # Save closure metrics CSV
            closure_file = self.output_dir / f"{self.filename_prefix}_closure_summary.csv"
            closure_df.to_csv(closure_file, index=False)
            print(f"[SAVE] Saved closure summary: {closure_file.name}")
            
            # Create closure plot
            closure_plot = self._plot_closure_summary(closure_df, group_summary, dashboard_dir)
            plot_files.append(closure_plot)
            print(f"   [OK] Saved: closure_summary.png")
            
            # Store closure data for HTML dashboard
            self._closure_df = closure_df
            self._group_summary = group_summary
            
        except Exception as e:
            print(f"[WARN] Could not compute closure metrics: {e}")
            self._closure_df = None
            self._group_summary = None
        
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
                ax.set_ylabel('Contribution (ug/m3)', fontsize=10)
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
            print(f"   [OK] Saved: factor_profiles.png")
            
        except Exception as e:
            print(f"   [ERROR] Error creating factor profiles: {e}")
        
        # New: Relative (composition) profiles per factor for scale-invariant view
        try:
            import numpy as np  # Ensure numpy is available
            n_factors = F_profiles.shape[0]
            if n_factors <= 4:
                nrows, ncols = 2, 2
            elif n_factors <= 6:
                nrows, ncols = 2, 3
            elif n_factors <= 9:
                nrows, ncols = 3, 3
            elif n_factors <= 12:
                nrows, ncols = 3, 4
            else:
                nrows, ncols = 4, 4
            fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
            station_display_name = self._get_station_display_name()
            fig.suptitle(f'{station_display_name} PMF Factor Profiles (Relative Composition)', fontsize=16, fontweight='bold')
            if n_factors == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            for i in range(n_factors):
                ax = axes[i]
                profile = F_profiles[i, :]
                s = np.sum(profile)
                rel = profile / s if s > 0 else profile
                
                # Replace zeros with small positive values for log scale
                rel_log = np.where(rel <= 0, 1e-6, rel)
                
                factor_color = self.color_manager.get_factor_color(i)
                bars = ax.bar(range(len(self.species_names)), rel_log, alpha=0.8, color=factor_color)
                ax.set_title(f'Factor {i+1}', fontweight='bold', fontsize=12)
                ax.set_xlabel('Species', fontsize=10)
                ax.set_ylabel('Relative Composition (log scale)', fontsize=10)
                ax.set_xticks(range(len(self.species_names)))
                ax.set_xticklabels(self.species_names, rotation=45, ha='right', fontsize=8)
                ax.set_yscale('log')
                ax.set_ylim(1e-6, 1.0)
                ax.grid(True, alpha=0.3)
            total_subplots = nrows * ncols
            for i in range(n_factors, total_subplots):
                if i < len(axes):
                    axes[i].set_visible(False)
            plt.tight_layout()
            plot_path_rel = dashboard_dir / f"{self.filename_prefix}_factor_profiles_relative.png"
            plt.savefig(plot_path_rel, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(plot_path_rel)
            print(f"   [OK] Saved: factor_profiles_relative.png")
        except Exception as e:
            print(f"   [ERROR] Error creating relative factor profiles: {e}")
        
        try:
            # Plot 2: Factor Contributions Time Series with Complaint Overlay
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Load complaint data for overlay
            complaint_data = self._load_complaint_data_for_overlay()
            
            # Get datetime index for plotting
            conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
            conc_data = pd.read_csv(conc_file, index_col=0)
            
            try:
                # Try to parse datetime index
                datetime_index = pd.to_datetime(conc_data.index)
                has_datetime = True
                
                # Plot factors in order with H2S factor last (for top layer visibility)
                factor_plot_order = self.color_manager.get_factor_plot_order()
                
                for i in factor_plot_order:
                    factor_color = self.color_manager.get_factor_color(i)
                    # Make H2S factor more prominent
                    is_h2s = self.color_manager.is_h2s_factor(i)
                    linewidth = 2.5 if is_h2s else 2
                    alpha_val = 0.9 if is_h2s else 0.8
                    
                    ax.plot(datetime_index, G_contributions[:, i], 
                            label=f'Factor {i+1}', linewidth=linewidth, alpha=alpha_val, color=factor_color)
                
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
                
                for i in factor_plot_order:
                    factor_color = self.color_manager.get_factor_color(i)
                    # Make H2S factor more prominent
                    is_h2s = self.color_manager.is_h2s_factor(i)
                    linewidth = 2.5 if is_h2s else 2
                    alpha_val = 0.9 if is_h2s else 0.8
                    
                    ax.plot(time_index, G_contributions[:, i], 
                            label=f'Factor {i+1}', linewidth=linewidth, alpha=alpha_val, color=factor_color)
                
                ax.set_xlabel('Sample Index')
            
            # Add complaint data overlay if available
            if complaint_data is not None and has_datetime:
                print(f"   [COMPLAINTS] Adding overlay to Factor Contributions plot")
                try:
                    # Create secondary y-axis for complaints (right side)
                    ax_complaints = ax.twinx()
                    
                    # Plot complaints as bars
                    valid_complaints = complaint_data.dropna()
                    print(f"   [COMPLAINTS] Plotting {len(valid_complaints)} complaint data points")
                    
                    if len(valid_complaints) > 0:
                        # Calculate max complaints for scaling
                        max_complaints = valid_complaints.max()
                        
                        # Set up compressed scale using top 30% of plot area
                        # Y-axis: 0-100, where 100 is top, 70 is bottom of complaint area
                        plot_top = 100.0
                        plot_complaint_bottom = 70.0  # Complaints use top 30% (70-100)
                        
                        # Scale complaint values to the compressed range (0 complaints = plot_top)
                        # Higher complaint values extend down from the top
                        scaled_heights = (valid_complaints / max_complaints) * (plot_top - plot_complaint_bottom)
                        
                        # Create hanging bars starting from top (plot_top) going down by scaled_heights
                        bars = ax_complaints.bar(valid_complaints.index, 
                                               scaled_heights,  # Bar heights (how far down from top)
                                               bottom=plot_top - scaled_heights,  # Start position (top minus height)
                                               alpha=0.7, color='red', width=pd.Timedelta(hours=18),
                                               label='Daily Complaints', zorder=10)
                        
                        # Format the right y-axis for complaints
                        ax_complaints.set_ylabel('Number of Complaints', color='red', fontweight='bold')
                        ax_complaints.tick_params(axis='y', labelcolor='red', colors='red')
                        ax_complaints.yaxis.label.set_color('red')
                        
                        # Ensure complaints y-axis is on the right and visible
                        ax_complaints.yaxis.set_label_position('right')
                        ax_complaints.yaxis.tick_right()
                        
                        # Set y-axis limits
                        ax_complaints.set_ylim(0, 100)
                        
                        # Create custom y-tick labels - show actual complaint values
                        # Ticks from top (0 complaints) down to max complaints
                        import numpy as np
                        n_ticks = 5
                        # Create tick positions from plot_top down to plot_complaint_bottom
                        tick_positions = np.linspace(plot_top, plot_complaint_bottom, n_ticks)
                        # Convert positions to complaint values (0 at top, max at bottom)
                        tick_labels = [f'{int(max_complaints * (plot_top - pos) / (plot_top - plot_complaint_bottom))}' 
                                     for pos in tick_positions]
                        ax_complaints.set_yticks(tick_positions)
                        ax_complaints.set_yticklabels(tick_labels)
                        
                        # Add complaint legend
                        complaint_legend = ax_complaints.legend(loc='upper right', frameon=True)
                        complaint_legend.get_frame().set_facecolor('white')
                        complaint_legend.get_frame().set_alpha(0.8)
                        
                        print(f"   [COMPLAINTS] Successfully added overlay: {len(valid_complaints)} days with complaints")
                        
                        # Highlight non-zero complaint days
                        non_zero_complaints = valid_complaints[valid_complaints > 0]
                        print(f"   [COMPLAINTS] Non-zero complaint days: {len(non_zero_complaints)}")
                    
                except Exception as e:
                    print(f"   [WARN] Could not add complaint overlay: {e}")
            
            ax.set_title(f'{station_display_name} PMF Factor Contributions Over Time with Complaints', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Contribution')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = dashboard_dir / f"{self.filename_prefix}_factor_contributions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(plot_path)
            print(f"   [OK] Saved: factor_contributions.png")
            
        except Exception as e:
            print(f"   [ERROR] Error creating factor contributions: {e}")
        
        # Plot 2b: H2S & CH4 Concentrations with Complaint Time Series - Include excluded species
        # Always try to show both H2S and CH4, even if excluded from PMF analysis
        available_species = []
        
        # Load concentration data from PMF analysis
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        conc_data = pd.read_csv(conc_file, index_col=0)
        datetime_index = pd.to_datetime(conc_data.index)
        
        # Extended data container for plotting (includes excluded species)
        extended_conc_data = conc_data.copy()
        
        # Check for H2S availability in PMF data
        if 'H2S' in conc_data.columns:
            available_species.append('H2S')
        elif 'H2S' in self.df.columns:
            # H2S was excluded from PMF but exists in original data
            try:
                # Handle both datetime column and index cases
                if 'datetime' in self.df.columns:
                    df_indexed = self.df.set_index('datetime')
                elif hasattr(self.df.index, 'min') and pd.api.types.is_datetime64_any_dtype(self.df.index):
                    df_indexed = self.df  # Already has datetime index
                else:
                    raise ValueError("No datetime information available")
                
                h2s_original = df_indexed['H2S'].reindex(datetime_index, method='nearest', tolerance=pd.Timedelta('1H'))
                if h2s_original.notna().sum() > len(h2s_original) * 0.5:  # At least 50% data available
                    extended_conc_data['H2S'] = h2s_original
                    available_species.append('H2S')
                    print(f"   [H2S] Loading excluded H2S from original data for visualization ({h2s_original.notna().sum()}/{len(h2s_original)} valid points)")
            except Exception as e:
                print(f"   [WARN] Could not load excluded H2S: {e}")
        
        # Check for CH4 availability in PMF data, or load from original if excluded
        if 'CH4' in conc_data.columns:
            available_species.append('CH4')
        elif 'CH4' in self.df.columns:
            # CH4 was excluded from PMF but exists in original data
            try:
                # Handle both datetime column and index cases
                if 'datetime' in self.df.columns:
                    df_indexed = self.df.set_index('datetime')
                elif hasattr(self.df.index, 'min') and pd.api.types.is_datetime64_any_dtype(self.df.index):
                    df_indexed = self.df  # Already has datetime index
                else:
                    raise ValueError("No datetime information available")
                
                ch4_original = df_indexed['CH4'].reindex(datetime_index, method='nearest', tolerance=pd.Timedelta('1H'))
                if ch4_original.notna().sum() > len(ch4_original) * 0.5:  # At least 50% data available
                    extended_conc_data['CH4'] = ch4_original
                    available_species.append('CH4')
                    print(f"   [CH4] Loading excluded CH4 from original data for visualization ({ch4_original.notna().sum()}/{len(ch4_original)} valid points)")
            except Exception as e:
                print(f"   [WARN] Could not load excluded CH4: {e}")
        
        if not available_species:
            print("   [SKIP] No H2S or CH4 data available for concentration plot")
        else:
            species_names = ' & '.join(available_species)
            print(f"   [PLOT] Creating {species_names} concentrations plot...")
            fig, ax = plt.subplots(figsize=(15, 8))
            
            # Plot H2S if available
            if 'H2S' in available_species:
                h2s_concentrations = extended_conc_data['H2S']
                h2s_min, h2s_max = h2s_concentrations.min(), h2s_concentrations.max()
                h2s_normalized = (h2s_concentrations - h2s_min) / (h2s_max - h2s_min)
                ax.plot(datetime_index, h2s_normalized, 
                       color='darkgreen', linewidth=2, alpha=0.8, 
                       label=f'H2S Normalized ({h2s_min:.1f}-{h2s_max:.1f} ppb)')
                print(f"   [H2S] Plotted H2S: {h2s_min:.2f} to {h2s_max:.2f} ppb")
            
            # Plot CH4 if available
            if 'CH4' in available_species:
                ch4_concentrations = extended_conc_data['CH4']
                ch4_min, ch4_max = ch4_concentrations.min(), ch4_concentrations.max()
                ch4_normalized = (ch4_concentrations - ch4_min) / (ch4_max - ch4_min)
                ax.plot(datetime_index, ch4_normalized, 
                       color='blue', linewidth=3, alpha=0.9, 
                       label=f'CH4 Normalized ({ch4_min:.0f}-{ch4_max:.0f} ppm)', marker='o', markersize=2)
                print(f"   [CH4] Plotted CH4: {ch4_min:.2f} to {ch4_max:.2f} ppm")
            
            # Format x-axis
            ax.set_xlabel('Date/Time')
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Add complaint overlay
            complaint_data = self._load_complaint_data_for_overlay()
            if complaint_data is not None:
                ax_complaints = ax.twinx()
                valid_complaints = complaint_data.dropna()
                if len(valid_complaints) > 0:
                    ax_complaints.plot(valid_complaints.index, valid_complaints.values, 
                                     color='red', linewidth=2, alpha=0.8, 
                                     label='Daily Complaints', marker='s', markersize=4)
                    ax_complaints.set_ylabel('Number of Complaints', color='red', fontweight='bold')
                    ax_complaints.tick_params(axis='y', labelcolor='red', colors='red')
                    ax_complaints.yaxis.set_label_position('right')
                    ax_complaints.yaxis.tick_right()
                    ax_complaints.set_ylim(0, valid_complaints.max() * 1.1)
                    complaint_legend = ax_complaints.legend(loc='upper right', frameon=True)
                    complaint_legend.get_frame().set_facecolor('white')
                    complaint_legend.get_frame().set_alpha(0.8)
                    print(f"   [COMPLAINTS] Added complaint overlay: {len(valid_complaints)} days")
            
            # Set axis properties
            ax.set_ylim(0, 1.1)
            station_display_name = self._get_station_display_name()
            # Dynamic title based on available species
            ax.set_title(f'{station_display_name} Normalized {species_names} Concentrations with Complaint Events', 
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Normalized Concentration (0-1)', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.4)
            ax.minorticks_on()
            
            # Add daily vertical reference lines
            start_date = datetime_index.min()
            end_date = datetime_index.max()
            daily_dates = pd.date_range(start_date.date(), end_date.date(), freq='D')
            for date in daily_dates:
                ax.axvline(x=date, color='lightgray', alpha=0.5, linewidth=0.8, linestyle='-')
            
            plt.tight_layout()
            
            # Save plot - dynamic filename based on available species
            species_filename = '_'.join([s.lower() for s in available_species])
            plot_path = dashboard_dir / f"{self.filename_prefix}_{species_filename}_concentrations.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(plot_path)
            print(f"   [OK] Saved: {species_filename}_concentrations.png")
        
        try:
            # Plot 2c: Correlation Table - Factors/Species vs Complaints
            print("   [CORR] Creating complaint correlation analysis...")
            
            # Load complaint data for correlation analysis
            complaint_data = self._load_complaint_data_for_overlay()
            
            if complaint_data is not None:
                # Get daily averages for factors (G_contributions)
                conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
                conc_data = pd.read_csv(conc_file, index_col=0)
                conc_data.index = pd.to_datetime(conc_data.index)
                
                # Create extended concentration data for correlation analysis (includes excluded species)
                extended_conc_data_corr = conc_data.copy()
                
                # Add excluded species from original data for correlation analysis
                datetime_index_corr = pd.to_datetime(conc_data.index)
                
                # Add CH4 if it was excluded but exists in original data
                if 'CH4' not in conc_data.columns and 'CH4' in self.df.columns:
                    try:
                        # Create a properly indexed original data series for alignment
                        # Handle both datetime column and index cases
                        if 'datetime' in self.df.columns:
                            df_indexed = self.df.set_index('datetime')
                        elif hasattr(self.df.index, 'min') and pd.api.types.is_datetime64_any_dtype(self.df.index):
                            df_indexed = self.df  # Already has datetime index
                        else:
                            raise ValueError("No datetime information available")
                        
                        # Use reindex to align with concentration data index, forward fill gaps
                        ch4_original = df_indexed['CH4'].reindex(datetime_index_corr, method='nearest', tolerance=pd.Timedelta('1H'))
                        # Only add if we have sufficient data
                        if ch4_original.notna().sum() > len(ch4_original) * 0.5:  # At least 50% data available
                            extended_conc_data_corr['CH4'] = ch4_original
                            print(f"   [CORR] Including excluded CH4 in correlation analysis ({ch4_original.notna().sum()}/{len(ch4_original)} valid points)")
                        else:
                            print(f"   [CORR] Insufficient CH4 data for correlation analysis ({ch4_original.notna().sum()}/{len(ch4_original)} valid points)")
                    except Exception as e:
                        print(f"   [WARN] Could not include CH4 in correlation analysis: {e}")
                
                # Add H2S if it was excluded but exists in original data
                if 'H2S' not in conc_data.columns and 'H2S' in self.df.columns:
                    try:
                        # Create a properly indexed original data series for alignment
                        # Handle both datetime column and index cases
                        if 'datetime' in self.df.columns:
                            df_indexed = self.df.set_index('datetime')
                        elif hasattr(self.df.index, 'min') and pd.api.types.is_datetime64_any_dtype(self.df.index):
                            df_indexed = self.df  # Already has datetime index
                        else:
                            raise ValueError("No datetime information available")
                        
                        # Use reindex to align with concentration data index, forward fill gaps
                        h2s_original = df_indexed['H2S'].reindex(datetime_index_corr, method='nearest', tolerance=pd.Timedelta('1H'))
                        # Only add if we have sufficient data
                        if h2s_original.notna().sum() > len(h2s_original) * 0.5:  # At least 50% data available
                            extended_conc_data_corr['H2S'] = h2s_original
                            print(f"   [CORR] Including excluded H2S in correlation analysis ({h2s_original.notna().sum()}/{len(h2s_original)} valid points)")
                        else:
                            print(f"   [CORR] Insufficient H2S data for correlation analysis ({h2s_original.notna().sum()}/{len(h2s_original)} valid points)")
                    except Exception as e:
                        print(f"   [WARN] Could not include H2S in correlation analysis: {e}")
                
                # Apply complaint correlation time window logic
                valid_complaints = complaint_data.dropna()
                
                print(f"   [CORR] Complaint correlation window: ±{self.complaint_correlation_hours} hours")
                
                if self.complaint_correlation_hours == 0:
                    # Original behavior: use daily aggregation
                    daily_concentrations = extended_conc_data_corr.resample('D').mean()
                    daily_concentrations_std = extended_conc_data_corr.resample('D').std()
                    
                    # Load factor contributions and aggregate to daily averages
                    factor_contribs_file = self.output_dir / f"{self.filename_prefix}_factor_contributions.csv"
                    if factor_contribs_file.exists():
                        factor_data = pd.read_csv(factor_contribs_file, index_col=0)
                        factor_data.index = pd.to_datetime(factor_data.index)
                        daily_factors = factor_data.resample('D').mean()
                        daily_factors_std = factor_data.resample('D').std()
                    else:
                        # Fallback: create daily factors from G_contributions
                        datetime_index = pd.to_datetime(conc_data.index)
                        factor_df = pd.DataFrame(G_contributions, index=datetime_index,
                                               columns=[f'Factor_{i+1}' for i in range(self.factors)])
                        daily_factors = factor_df.resample('D').mean()
                        daily_factors_std = factor_df.resample('D').std()
                else:
                    # New behavior: use time windows around complaints
                    print(f"   [CORR] Using windowed correlation: ±{self.complaint_correlation_hours}h around each complaint")
                    print(f"   [CORR] Aggregation method: {self.complaint_window_method}")
                    
                    # Load factor contributions for windowing
                    factor_contribs_file = self.output_dir / f"{self.filename_prefix}_factor_contributions.csv"
                    if factor_contribs_file.exists():
                        factor_data = pd.read_csv(factor_contribs_file, index_col=0)
                        factor_data.index = pd.to_datetime(factor_data.index)
                    else:
                        # Fallback: create factor data from G_contributions
                        datetime_index = pd.to_datetime(conc_data.index)
                        factor_data = pd.DataFrame(G_contributions, index=datetime_index,
                                                 columns=[f'Factor_{i+1}' for i in range(self.factors)])
                    
                    # Create windowed averages and standard deviations for each complaint day
                    windowed_concentrations = []
                    windowed_factors = []
                    windowed_concentrations_std = []
                    windowed_factors_std = []
                    windowed_complaint_dates = []
                    
                    for complaint_date, complaint_count in valid_complaints.items():
                        if complaint_count > 0:  # Only process days with complaints
                            # Define time window around complaint date (assuming complaints occur at noon)
                            center_time = pd.Timestamp(complaint_date.date()) + pd.Timedelta(hours=12)
                            window_start = center_time - pd.Timedelta(hours=self.complaint_correlation_hours)
                            window_end = center_time + pd.Timedelta(hours=self.complaint_correlation_hours)
                            
                            # Extract concentration data within window
                            conc_window = extended_conc_data_corr[
                                (extended_conc_data_corr.index >= window_start) & 
                                (extended_conc_data_corr.index <= window_end)
                            ]
                            
                            # Extract factor data within window
                            factor_window = factor_data[
                                (factor_data.index >= window_start) & 
                                (factor_data.index <= window_end)
                            ]
                            
                            # Calculate aggregation and uncertainties for the window (if data exists)
                            if len(conc_window) > 0:
                                # Apply selected aggregation method
                                conc_aggregated = self._aggregate_window_data(conc_window, self.complaint_window_method)
                                factor_aggregated = self._aggregate_window_data(factor_window, self.complaint_window_method)
                                
                                # Calculate appropriate uncertainty measures
                                conc_uncertainty = self._calculate_window_uncertainty(conc_window, conc_aggregated, self.complaint_window_method)
                                factor_uncertainty = self._calculate_window_uncertainty(factor_window, factor_aggregated, self.complaint_window_method)
                                
                                windowed_concentrations.append(conc_aggregated)
                                windowed_factors.append(factor_aggregated)
                                windowed_concentrations_std.append(conc_uncertainty)
                                windowed_factors_std.append(factor_uncertainty)
                                windowed_complaint_dates.append(complaint_date)
                    
                    # Convert to DataFrames with proper indexing
                    if windowed_concentrations:
                        daily_concentrations = pd.DataFrame(windowed_concentrations, index=windowed_complaint_dates)
                        daily_factors = pd.DataFrame(windowed_factors, index=windowed_complaint_dates)
                        daily_concentrations_std = pd.DataFrame(windowed_concentrations_std, index=windowed_complaint_dates)
                        daily_factors_std = pd.DataFrame(windowed_factors_std, index=windowed_complaint_dates)
                        print(f"   [CORR] Generated {len(daily_concentrations)} windowed correlation points using '{self.complaint_window_method}' aggregation with uncertainties")
                    else:
                        print(f"   [WARN] No valid windowed data found for correlation analysis")
                        daily_concentrations = pd.DataFrame()
                        daily_factors = pd.DataFrame()
                        daily_concentrations_std = pd.DataFrame()
                        daily_factors_std = pd.DataFrame()
                
                # Align data by date and calculate correlations
                
                correlations = {}
                
                # Calculate factor correlations
                print(f"   [CORR] Calculating factor correlations...")
                for factor_col in daily_factors.columns:
                    # Align dates between factors and complaints
                    aligned_data = pd.concat([daily_factors[factor_col], valid_complaints], 
                                           axis=1, join='inner')
                    aligned_data.columns = ['factor_value', 'complaints']
                    
                    if len(aligned_data) > 1:
                        correlation = aligned_data['factor_value'].corr(aligned_data['complaints'])
                        correlations[factor_col] = correlation
                    else:
                        correlations[factor_col] = float('nan')
                
                # Calculate species correlations
                print(f"   [CORR] Calculating species correlations...")
                for species in daily_concentrations.columns:
                    # Align dates between species and complaints
                    aligned_data = pd.concat([daily_concentrations[species], valid_complaints], 
                                           axis=1, join='inner')
                    aligned_data.columns = ['species_value', 'complaints']
                    
                    if len(aligned_data) > 1:
                        correlation = aligned_data['species_value'].corr(aligned_data['complaints'])
                        correlations[f'{species}_Species'] = correlation
                    else:
                        correlations[f'{species}_Species'] = float('nan')
                
                # Create correlation table visualization
                # matplotlib.pyplot as plt already imported at module level
                import numpy as np
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
                
                # Separate factor and species correlations
                factor_corrs = {k: v for k, v in correlations.items() if 'Factor_' in k}
                species_corrs = {k.replace('_Species', ''): v for k, v in correlations.items() if '_Species' in k}
                
                # Plot 1: Factor correlations
                if factor_corrs:
                    factor_names = list(factor_corrs.keys())
                    factor_values = [factor_corrs[name] for name in factor_names]
                    
                    # Color bars by correlation strength
                    colors = ['red' if v < -0.3 else 'blue' if v > 0.3 else 'gray' for v in factor_values]
                    
                    bars1 = ax1.bar(range(len(factor_names)), factor_values, color=colors, alpha=0.7)
                    
                    # Dynamic title based on correlation mode
                    if self.complaint_correlation_hours == 0:
                        title1 = 'PMF Factors vs Daily Complaints\nCorrelation Analysis (Daily Aggregation)'
                    else:
                        title1 = f'PMF Factors vs Complaints\nCorrelation Analysis (±{self.complaint_correlation_hours}h {self.complaint_window_method.title()} Windows)'
                    ax1.set_title(title1, fontweight='bold')
                    ax1.set_xlabel('PMF Factors')
                    ax1.set_ylabel('Pearson Correlation Coefficient')
                    ax1.set_xticks(range(len(factor_names)))
                    ax1.set_xticklabels([f'F{i+1}' for i in range(len(factor_names))], rotation=0)
                    ax1.grid(True, alpha=0.3)
                    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    ax1.set_ylim(-1, 1)
                    
                    # Add correlation values as text on bars
                    for i, (bar, val) in enumerate(zip(bars1, factor_values)):
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
                                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                                fontsize=9, fontweight='bold')
                
                # Plot 2: Species correlations
                if species_corrs:
                    species_names = list(species_corrs.keys())
                    species_values = [species_corrs[name] for name in species_names]
                    
                    # Color bars by correlation strength
                    colors = ['red' if v < -0.3 else 'blue' if v > 0.3 else 'gray' for v in species_values]
                    
                    bars2 = ax2.bar(range(len(species_names)), species_values, color=colors, alpha=0.7)
                    
                    # Dynamic title based on correlation mode
                    if self.complaint_correlation_hours == 0:
                        title2 = 'Chemical Species vs Daily Complaints\nCorrelation Analysis (Daily Aggregation)'
                    else:
                        title2 = f'Chemical Species vs Complaints\nCorrelation Analysis (±{self.complaint_correlation_hours}h {self.complaint_window_method.title()} Windows)'
                    ax2.set_title(title2, fontweight='bold')
                    ax2.set_xlabel('Chemical Species')
                    ax2.set_ylabel('Pearson Correlation Coefficient')
                    ax2.set_xticks(range(len(species_names)))
                    ax2.set_xticklabels(species_names, rotation=45, ha='right')
                    ax2.grid(True, alpha=0.3)
                    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                    ax2.set_ylim(-1, 1)
                    
                    # Add correlation values as text on bars
                    for i, (bar, val) in enumerate(zip(bars2, species_values)):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height + (0.02 if height >= 0 else -0.05),
                                f'{val:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                                fontsize=9, fontweight='bold')
                
                plt.tight_layout()
                plot_path = dashboard_dir / f"{self.filename_prefix}_complaint_correlations.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                plot_files.append(plot_path)
                
                # CREATE SCATTERPLOTS: Species/Factor Values vs Complaints
                print(f"   [CORR] Creating scatterplots for species and factors vs complaints...")
                
                # Get top correlated variables (both positive and negative) for scatterplots
                all_correlations = list(correlations.items())
                sorted_correlations = sorted(all_correlations, key=lambda x: abs(x[1]), reverse=True)
                top_correlations = [item for item in sorted_correlations if not pd.isna(item[1])][:12]  # Top 12 for 3x4 grid
                
                if len(top_correlations) > 0:
                    # Create scatterplot figure
                    n_plots = len(top_correlations)
                    n_cols = 4
                    n_rows = max(1, (n_plots + n_cols - 1) // n_cols)
                    
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
                    if n_rows == 1:
                        axes = axes.reshape(1, -1) if n_plots > 1 else [[axes]]
                    
                    for idx, (var_name, correlation) in enumerate(top_correlations):
                        row = idx // n_cols
                        col = idx % n_cols
                        ax = axes[row, col]
                        
                        # Determine if it's a factor or species
                        if 'Factor_' in var_name:
                            # Factor scatterplot
                            factor_col = var_name
                            if factor_col in daily_factors.columns:
                                # Align dates between factors and complaints
                                aligned_data = pd.concat([daily_factors[factor_col], valid_complaints], 
                                                       axis=1, join='inner')
                                aligned_data.columns = ['factor_value', 'complaints']
                                
                                if len(aligned_data) > 1:
                                    x_values = aligned_data['factor_value']
                                    y_values = aligned_data['complaints']
                                    
                                    # Get standard deviations for error bars
                                    if factor_col in daily_factors_std.columns:
                                        # Align standard deviations with the same dates
                                        aligned_std = daily_factors_std[factor_col].reindex(aligned_data.index)
                                        # Use small non-zero value for missing/zero std to ensure error bars are visible
                                        x_errors = aligned_std.fillna(x_values.std() * 0.01)  # Use 1% of data std for missing values
                                        x_errors = x_errors.replace(0, x_values.std() * 0.01)  # Replace zeros too
                                    else:
                                        x_errors = None
                                    
                                    # Color points by correlation strength
                                    color = 'blue' if correlation > 0.3 else 'red' if correlation < -0.3 else 'gray'
                                    
                                    # Use errorbar instead of scatter to show error bars
                                    if x_errors is not None:
                                        ax.errorbar(x_values, y_values, xerr=x_errors, fmt='o', 
                                                   alpha=0.6, color=color, markersize=5, 
                                                   ecolor='lightgray', capsize=2, capthick=1)
                                    else:
                                        ax.scatter(x_values, y_values, alpha=0.6, color=color, s=30)
                                    
                                    # Calculate psychometric sigmoid metrics
                                    print(f"   [PSYCHOMETRIC] Calculating sigmoid fit for {factor_col}: {len(x_values)} samples")
                                    psychometric_metrics = calculate_psychometric_fit(x_values, y_values)
                                    
                                    title_text = f'{factor_col} vs Complaints\nr = {correlation:.3f}'
                                    
                                    if psychometric_metrics is not None:
                                        print(f"   [PSYCHOMETRIC] {factor_col}: R²={psychometric_metrics['r_squared']:.3f}, Threshold={psychometric_metrics['parameters']['x50_threshold']:.3f} μg/m³")
                                        
                                        # Plot psychometric sigmoid curve
                                        sigmoid_color = 'darkgreen' if psychometric_metrics['r_squared'] > 0.7 else 'orange'
                                        
                                        # Plot the fitted sigmoid curve
                                        ax.plot(psychometric_metrics['sigmoid_x'], 
                                               psychometric_metrics['sigmoid_y'],
                                               '-', alpha=0.8, color=sigmoid_color, linewidth=3, 
                                               label='Psychometric Sigmoid')
                                        
                                        # Add threshold markers
                                        threshold_50 = psychometric_metrics['parameters']['x50_threshold']
                                        ymax = psychometric_metrics['parameters']['ymax']
                                        ax.axvline(x=threshold_50, color=sigmoid_color, linestyle='--', alpha=0.6, 
                                                  label=f'50% Threshold ({threshold_50:.2f})')
                                        ax.axhline(y=ymax/2, color=sigmoid_color, linestyle=':', alpha=0.6)
                                        
                                        # Add metrics to title
                                        title_text += f'\nR²={psychometric_metrics["r_squared"]:.3f}'  
                                        title_text += f', Threshold={threshold_50:.2f} μg/m³'
                                        
                                        # Add legend
                                        ax.legend(fontsize=8)
                                    else:
                                        print(f"   [PSYCHOMETRIC] {factor_col}: No sigmoid fit (insufficient data or poor fit)")
                                    
                                    # Add linear trend line for comparison if correlation is strong
                                    if abs(correlation) > 0.3:
                                        try:
                                            z = np.polyfit(x_values, y_values, 1)
                                            p = np.poly1d(z)
                                            x_line = np.linspace(x_values.min(), x_values.max(), 100)
                                            ax.plot(x_line, p(x_line), "--", alpha=0.5, color='gray', linewidth=1, label='Linear')
                                        except np.linalg.LinAlgError:
                                            pass
                                    
                                    ax.set_xlabel(f'{factor_col} (μg/m³)')
                                    ax.set_ylabel('Daily Complaints')
                                    ax.set_title(title_text, fontsize=9, fontweight='bold')
                                    ax.grid(True, alpha=0.3)
                        
                        elif '_Species' in var_name:
                            # Species scatterplot
                            species_name = var_name.replace('_Species', '')
                            if species_name in daily_concentrations.columns:
                                # Align dates between species and complaints
                                aligned_data = pd.concat([daily_concentrations[species_name], valid_complaints], 
                                                       axis=1, join='inner')
                                aligned_data.columns = ['species_value', 'complaints']
                                
                                if len(aligned_data) > 1:
                                    x_values = aligned_data['species_value']
                                    y_values = aligned_data['complaints']
                                    
                                    # Get standard deviations for error bars
                                    if species_name in daily_concentrations_std.columns:
                                        # Align standard deviations with the same dates
                                        aligned_std = daily_concentrations_std[species_name].reindex(aligned_data.index)
                                        # Use small non-zero value for missing/zero std to ensure error bars are visible
                                        x_errors = aligned_std.fillna(x_values.std() * 0.01)  # Use 1% of data std for missing values
                                        x_errors = x_errors.replace(0, x_values.std() * 0.01)  # Replace zeros too
                                    else:
                                        x_errors = None
                                    
                                    # Color points by correlation strength
                                    color = 'blue' if correlation > 0.3 else 'red' if correlation < -0.3 else 'gray'
                                    
                                    # Use errorbar instead of scatter to show error bars
                                    if x_errors is not None:
                                        ax.errorbar(x_values, y_values, xerr=x_errors, fmt='o', 
                                                   alpha=0.6, color=color, markersize=5, 
                                                   ecolor='lightgray', capsize=2, capthick=1)
                                    else:
                                        ax.scatter(x_values, y_values, alpha=0.6, color=color, s=30)
                                    
                                    # Calculate psychometric sigmoid metrics
                                    print(f"   [PSYCHOMETRIC] Calculating sigmoid fit for {species_name}: {len(x_values)} samples")
                                    psychometric_metrics = calculate_psychometric_fit(x_values, y_values)
                                    
                                    title_text = f'{species_name} vs Complaints\nr = {correlation:.3f}'
                                    
                                    if psychometric_metrics is not None:
                                        print(f"   [PSYCHOMETRIC] {species_name}: R²={psychometric_metrics['r_squared']:.3f}, Threshold={psychometric_metrics['parameters']['x50_threshold']:.3f} μg/m³")
                                        
                                        # Plot psychometric sigmoid curve
                                        sigmoid_color = 'darkgreen' if psychometric_metrics['r_squared'] > 0.7 else 'orange'
                                        
                                        # Plot the fitted sigmoid curve
                                        ax.plot(psychometric_metrics['sigmoid_x'], 
                                               psychometric_metrics['sigmoid_y'],
                                               '-', alpha=0.8, color=sigmoid_color, linewidth=3, 
                                               label='Psychometric Sigmoid')
                                        
                                        # Add threshold markers
                                        threshold_50 = psychometric_metrics['parameters']['x50_threshold']
                                        ymax = psychometric_metrics['parameters']['ymax']
                                        ax.axvline(x=threshold_50, color=sigmoid_color, linestyle='--', alpha=0.6, 
                                                  label=f'50% Threshold ({threshold_50:.2f})')
                                        ax.axhline(y=ymax/2, color=sigmoid_color, linestyle=':', alpha=0.6)
                                        
                                        # Add metrics to title
                                        title_text += f'\nR²={psychometric_metrics["r_squared"]:.3f}'  
                                        title_text += f', Threshold={threshold_50:.2f} μg/m³'
                                        
                                        # Add legend
                                        ax.legend(fontsize=8)
                                    else:
                                        print(f"   [PSYCHOMETRIC] {species_name}: No sigmoid fit (insufficient data or poor fit)")
                                    
                                    # Add linear trend line for comparison if correlation is strong
                                    if abs(correlation) > 0.3:
                                        try:
                                            z = np.polyfit(x_values, y_values, 1)
                                            p = np.poly1d(z)
                                            x_line = np.linspace(x_values.min(), x_values.max(), 100)
                                            ax.plot(x_line, p(x_line), "--", alpha=0.5, color='gray', linewidth=1, label='Linear')
                                        except np.linalg.LinAlgError:
                                            pass
                                    
                                    ax.set_xlabel(f'{species_name} (μg/m³)')
                                    ax.set_ylabel('Daily Complaints')
                                    ax.set_title(title_text, fontsize=9, fontweight='bold')
                                    ax.grid(True, alpha=0.3)
                    
                    # Hide unused subplots
                    for idx in range(len(top_correlations), n_rows * n_cols):
                        row = idx // n_cols
                        col = idx % n_cols
                        axes[row, col].set_visible(False)
                    
                    # Add overall title based on correlation window mode
                    if self.complaint_correlation_hours == 0:
                        fig.suptitle(f'Species & Factor Values vs Daily Complaints\nScatterplot Analysis (Daily Aggregation)', 
                                   fontsize=14, fontweight='bold', y=0.98)
                    else:
                        fig.suptitle(f'Species & Factor Values vs Complaints\nScatterplot Analysis (±{self.complaint_correlation_hours}h {self.complaint_window_method.title()} Windows)', 
                                   fontsize=14, fontweight='bold', y=0.98)
                    
                    plt.tight_layout()
                    scatter_plot_path = dashboard_dir / f"{self.filename_prefix}_complaint_scatterplots.png"
                    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                    plt.close()
                    plot_files.append(scatter_plot_path)
                    print(f"   [OK] Saved: complaint_scatterplots.png ({len(top_correlations)} plots)")
                
                # Save correlation results to CSV
                corr_df = pd.DataFrame(list(correlations.items()), columns=['Variable', 'Correlation_with_Complaints'])
                corr_df = corr_df.sort_values('Correlation_with_Complaints', key=abs, ascending=False)
                
                corr_csv_path = self.output_dir / f"{self.filename_prefix}_complaint_correlations.csv"
                corr_df.to_csv(corr_csv_path, index=False)
                
                print(f"   [OK] Saved: complaint_correlations.png")
                print(f"   [OK] Saved: complaint_correlations.csv")
                
                # Print top correlations
                print(f"   [CORR] Top positive correlations:")
                positive_corrs = corr_df[corr_df['Correlation_with_Complaints'] > 0].head(3)
                for _, row in positive_corrs.iterrows():
                    print(f"     {row['Variable']}: {row['Correlation_with_Complaints']:.3f}")
                
                print(f"   [CORR] Top negative correlations:")
                negative_corrs = corr_df[corr_df['Correlation_with_Complaints'] < 0].head(3)
                for _, row in negative_corrs.iterrows():
                    print(f"     {row['Variable']}: {row['Correlation_with_Complaints']:.3f}")
                
                # Print psychometric sigmoid fit metrics for top correlations
                print(f"   [PSYCHOMETRIC] Psychometric sigmoid fit metrics:")
                for var_name, correlation in top_correlations[:5]:  # Top 5 variables
                    if 'Factor_' in var_name:
                        factor_col = var_name
                        if factor_col in daily_factors.columns:
                            aligned_data = pd.concat([daily_factors[factor_col], valid_complaints], axis=1, join='inner')
                            if len(aligned_data) > 1:
                                psychometric_metrics = calculate_psychometric_fit(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                                if psychometric_metrics is not None:
                                    print(f"     {var_name}: R²={psychometric_metrics['r_squared']:.3f}, "
                                          f"Threshold={psychometric_metrics['parameters']['x50_threshold']:.3f} μg/m³, "
                                          f"Max Response={psychometric_metrics['parameters']['ymax']:.1f}")
                    elif '_Species' in var_name:
                        species_name = var_name.replace('_Species', '')
                        if species_name in daily_concentrations.columns:
                            aligned_data = pd.concat([daily_concentrations[species_name], valid_complaints], axis=1, join='inner')
                            if len(aligned_data) > 1:
                                psychometric_metrics = calculate_psychometric_fit(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                                if psychometric_metrics is not None:
                                    print(f"     {var_name}: R²={psychometric_metrics['r_squared']:.3f}, "
                                          f"Threshold={psychometric_metrics['parameters']['x50_threshold']:.3f} μg/m³, "
                                          f"Max Response={psychometric_metrics['parameters']['ymax']:.1f}")
            
            else:
                print(f"   [INFO] No complaint data available for correlation analysis")
            
        except Exception as e:
            print(f"   [ERROR] Error creating complaint correlation analysis: {e}")
        
        try:
            # Plot 3: Species Composition (Stacked Bar)
            import numpy as np  # Ensure numpy is available
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
            print(f"   [OK] Saved: species_composition.png")
            
        except Exception as e:
            print(f"   [ERROR] Error creating species composition: {e}")
        
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
            print(f"   [OK] Saved: model_quality.png")
            
        except Exception as e:
            print(f"   [ERROR] Error creating model quality plot: {e}")
        
        # Plot 5: Residual Analysis (EPA recommended)
        try:
            self._create_residual_plots(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating residual plots: {e}")
        
        # Plot 6: Factor Correlation Analysis
        try:
            self._create_correlation_plots(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating correlation plots: {e}")
        
        # Plot 7: Source Contribution Analysis
        try:
            self._create_source_contribution_plots(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating source contribution plots: {e}")
        
        # Plot 8: Seasonal/Temporal Analysis
        try:
            self._create_temporal_analysis_plots(dashboard_dir, plot_files, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating temporal analysis plots: {e}")
        
        # Plot 9: Bootstrap/Uncertainty Analysis (if multiple models)
        try:
            self._create_uncertainty_plots(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating uncertainty plots: {e}")
        
        # Plot 10: Diagnostic Scatter Plots
        try:
            self._create_diagnostic_scatters(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating diagnostic scatter plots: {e}")
        
        # Plot 11: Factor Optimization Plot (Q vs Factors)
        try:
            self._create_optimization_plot(dashboard_dir, plot_files)
        except Exception as e:
            print(f"   [ERROR] Error creating optimization plot: {e}")
        
        # Plot 12: Wind Direction and Speed Analysis
        try:
            self._create_wind_analysis_plots(dashboard_dir, plot_files, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating wind analysis plots: {e}")
        
        # Plot 13: Temperature Analysis
        try:
            self._create_temperature_analysis_plots(dashboard_dir, plot_files, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating temperature analysis plots: {e}")
        
        # Plot 14: Pressure Analysis
        try:
            self._create_pressure_analysis_plots(dashboard_dir, plot_files, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating pressure analysis plots: {e}")
        
        # Plot 15: Sankey Diagram (Factors -> Species)
        try:
            self._create_sankey_diagram(dashboard_dir, plot_files, F_profiles, G_contributions)
        except Exception as e:
            print(f"   [ERROR] Error creating Sankey diagram: {e}")
        
        # NEW S/N Categorization and EPA Analysis Plots
        if self.snr_enable and hasattr(self, '_snr_categories'):
            try:
                self._create_snr_analysis_plots(dashboard_dir, plot_files)
            except Exception as e:
                print(f"   [ERROR] Error creating S/N analysis plots: {e}")
        
        # Plot 16: PCA vs PMF Comparison Plots (if PCA has been run)
        try:
            self._create_pca_comparison_plots(dashboard_dir, plot_files)
        except Exception as e:
            print(f"   [ERROR] Error creating PCA comparison plots: {e}")
        
        # Plot 17: PCA Loadings Plots (if PCA has been run)
        try:
            self._create_pca_loadings_plot(dashboard_dir, plot_files)
        except Exception as e:
            print(f"   [ERROR] Error creating PCA loadings plots: {e}")
        
        # Generate factor structure diagnostics summary
        try:
            self._generate_factor_structure_summary()
        except Exception as e:
            print(f"   [WARN] Error generating factor structure summary: {e}")
        
        # Create summary dashboard HTML
        self._create_html_dashboard(plot_files)
        
        print(f"[DATA] Dashboard complete: {len(plot_files)} plots generated")
    
    def _create_snr_analysis_plots(self, dashboard_dir, plot_files):
        """
        Create comprehensive S/N categorization analysis plots and summaries.
        """
        print("   [NUMBERS] Creating S/N categorization analysis plots...")
        
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
            print("   [WARN] S/N metrics file not found, skipping S/N plots")
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
            ax3.set_xlabel('Mean Concentration (ug/m3)')
            ax3.set_ylabel('Mean Uncertainty (ug/m3)')
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
                ax4.set_ylabel('Uncertainty (ug/m3)')
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
        print(f"   [OK] Saved: snr_analysis.png")
    
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
            'factors': getattr(self, 'factors', None),
            'max_factors': getattr(self, 'max_factors', 10),
            'models': getattr(self, 'models', 20),
            'output_dir': str(self.output_dir),
            'remove_voc': getattr(self, 'remove_voc', False),
            'run_pca': getattr(self, 'run_pca', False),
            'create_pdf': getattr(self, 'create_pdf', False),
            'max_workers': getattr(self, 'max_workers', 2),
            'scale_units': getattr(self, 'scale_units', True),
            
            # Data processing parameters
            'drop_row_threshold': getattr(self, 'drop_row_threshold', 0.5),
            'zero_as_bdl': getattr(self, 'zero_as_bdl', True),
            'save_masks': getattr(self, 'save_masks', True),
            
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
            'weight_aware_init': getattr(self, 'weight_aware_init', None),
            
            # Output controls
            'dashboard_snr_panel': getattr(self, 'dashboard_snr_panel', True),
            'write_diagnostics': self.write_diagnostics,
            'seed': self.seed,
            'help_detail': getattr(self, 'help_detail', False),
            
            # Regularization parameters
            'reg_species': getattr(self, 'reg_species', []),
            'reg_lambda': getattr(self, 'reg_lambda', []),
            'reg_template': getattr(self, 'reg_template', []),
            'reg_template_file': getattr(self, 'reg_template_file', []),
            'reg_bursts': getattr(self, 'reg_bursts', 5),
            'reg_iter_per_burst': getattr(self, 'reg_iter_per_burst', 50),
            'reg_tol': getattr(self, 'reg_tol', 1e-4),
            'reg_elastic_l1': getattr(self, 'reg_elastic_l1', 0.0),
            
            # Bootstrap error estimation parameters
            'bootstrap': getattr(self, 'bootstrap', False),
            'bootstrap_n': getattr(self, 'bootstrap_n', 100),
            'bootstrap_block_size': getattr(self, 'bootstrap_block_size', None),
            'bootstrap_threshold': getattr(self, 'bootstrap_threshold', 0.6),
            'bootstrap_parallel': getattr(self, 'bootstrap_parallel', True),
            'bootstrap_cpus': getattr(self, 'bootstrap_cpus', None),
            'bootstrap_seed': getattr(self, 'bootstrap_seed', None),
            'bootstrap_keep_h': getattr(self, 'bootstrap_keep_h', True),
            'bootstrap_reuse_seed': getattr(self, 'bootstrap_reuse_seed', True),
            'bootstrap_overlapping': getattr(self, 'bootstrap_overlapping', False),
            
            # Complaint correlation analysis parameters
            'complaint_correlation_hours': getattr(self, 'complaint_correlation_hours', 0),
        }
        
        html_section = """
            <div class="cli-section">
                <h2>[INFO] CLI Flags Used (Reproducibility Record)</h2>
                <p><strong>Command to reproduce this analysis:</strong></p>
                <pre style="background-color: #f8f8f8; padding: 10px; border-radius: 3px; overflow-x: auto;">
        """
        
        # Build command line
        cmd_parts = ['python pmf_source_app.py']
        
        if cli_params['station']:
            cmd_parts.append(f"{cli_params['station']}")
        if cli_params['data_dir']:
            cmd_parts.append(f'--data-dir "{cli_params["data_dir"]}"')
        if cli_params['patterns']:
            cmd_parts.append(f'--patterns "{cli_params["patterns"]}"')
        if cli_params['start_date']:
            cmd_parts.append(f"--start-date {cli_params['start_date']}")
        if cli_params['end_date']:
            cmd_parts.append(f"--end-date {cli_params['end_date']}")
        if cli_params['factors']:
            cmd_parts.append(f"--factors {cli_params['factors']}")
        elif cli_params['max_factors'] != 10:
            cmd_parts.append(f"--max-factors {cli_params['max_factors']}")
        if cli_params['models'] != 20:
            cmd_parts.append(f"--models {cli_params['models']}")
        if cli_params['max_workers'] != 2:
            cmd_parts.append(f"--max-workers {cli_params['max_workers']}")
        if cli_params['output_dir'] and cli_params['output_dir'] != 'pmf_results_esat':
            cmd_parts.append(f'--output-dir "{cli_params["output_dir"]}"')
        if cli_params['run_pca']:
            cmd_parts.append('--run-pca')
        if cli_params['create_pdf']:
            cmd_parts.append('--create-pdf')
        if cli_params['remove_voc']:
            cmd_parts.append('--remove-voc')
        if not cli_params['scale_units']:
            cmd_parts.append('--no-scale-units')
        if cli_params['drop_row_threshold'] != 0.5:
            cmd_parts.append(f"--drop-row-threshold {cli_params['drop_row_threshold']}")
        if not cli_params['zero_as_bdl']:
            cmd_parts.append('--no-zero-as-bdl')
        if not cli_params['save_masks']:
            cmd_parts.append('--no-save-masks')
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
        if cli_params['weight_aware_init'] is True:
            cmd_parts.append('--weight-aware-init')
        elif cli_params['weight_aware_init'] is False:
            cmd_parts.append('--no-weight-aware-init')
        
        # Add species weighting parameters
        if hasattr(self, '_species_weights_applied') and self._species_weights_applied:
            for species, weight in self._species_weights_applied.items():
                cmd_parts.append(f'--species-weight {species}={weight}')
        
        # Add species exclusion parameters
        if hasattr(self, '_excluded_species_applied') and self._excluded_species_applied:
            for species in self._excluded_species_applied:
                cmd_parts.append(f'--exclude-species {species}')
        
        # Add regularization parameters
        if hasattr(self, '_reg_enabled') and self._reg_enabled and hasattr(self, '_reg_plan'):
            for reg_item in self._reg_plan:
                cmd_parts.append(f'--reg-species {reg_item["species"]}')
                cmd_parts.append(f'--reg-lambda {reg_item["lambda"]}')
                cmd_parts.append(f'--reg-template {reg_item["template_type"]}')
        
        # Add bootstrap parameters
        if cli_params['bootstrap']:
            cmd_parts.append('--bootstrap')
            if cli_params['bootstrap_n'] != 100:
                cmd_parts.append(f"--bootstrap-n {cli_params['bootstrap_n']}")
            if cli_params['bootstrap_block_size'] is not None:
                cmd_parts.append(f"--bootstrap-block-size {cli_params['bootstrap_block_size']}")
            if cli_params['bootstrap_threshold'] != 0.6:
                cmd_parts.append(f"--bootstrap-threshold {cli_params['bootstrap_threshold']}")
            if not cli_params['bootstrap_parallel']:
                cmd_parts.append('--bootstrap-parallel false')
            if cli_params['bootstrap_cpus'] is not None:
                cmd_parts.append(f"--bootstrap-cpus {cli_params['bootstrap_cpus']}")
            if cli_params['bootstrap_seed'] is not None:
                cmd_parts.append(f"--bootstrap-seed {cli_params['bootstrap_seed']}")
            if not cli_params['bootstrap_keep_h']:
                cmd_parts.append('--no-bootstrap-keep-h')
            if not cli_params['bootstrap_reuse_seed']:
                cmd_parts.append('--no-bootstrap-reuse-seed')
            if cli_params['bootstrap_overlapping']:
                cmd_parts.append('--bootstrap-overlapping')
        
        # Add complaint correlation parameters
        if cli_params['complaint_correlation_hours'] != 0:
            cmd_parts.append(f"--complaint-correlation-hours {cli_params['complaint_correlation_hours']}")
        
        # Add help detail flag if enabled  
        if cli_params['help_detail']:
            cmd_parts.append('--help-detail')
        
        # Add other non-default parameters
        non_defaults = {
            'uncertainty_epsilon': (1e-12, cli_params['uncertainty_epsilon']),
            'legacy_min_u': (0.1, cli_params['legacy_min_u']),
            'snr_weak_threshold': (2.0, cli_params['snr_weak_threshold']),
            'snr_bad_threshold': (0.2, cli_params['snr_bad_threshold']),
            'snr_bdl_weak_frac': (0.6, cli_params['snr_bdl_weak_frac']),
            'snr_bdl_bad_frac': (0.8, cli_params['snr_bdl_bad_frac']),
            'snr_missing_weak_frac': (0.2, cli_params['snr_missing_weak_frac']),
            'snr_missing_bad_frac': (0.4, cli_params['snr_missing_bad_frac']),
            'robust_alpha': (4.0, cli_params['robust_alpha']),
            'seed': (42, cli_params['seed']),
            'method': ('ls-nmf', cli_params['method']),
            'init_method': ('column_mean', cli_params['init_method']),
            'delay_h': (-1, cli_params['delay_h']),
            'reg_bursts': (5, cli_params['reg_bursts']),
            'reg_iter_per_burst': (50, cli_params['reg_iter_per_burst']),
            'reg_tol': (1e-4, cli_params['reg_tol']),
            'reg_elastic_l1': (0.0, cli_params['reg_elastic_l1']),
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
            # Core analysis parameters
            'station': (cli_params.get('station', 'N/A'), 'Station identifier (positional argument)'),
            'data_dir': (cli_params.get('data_dir', 'N/A'), 'Data directory (alternative to station)'),
            'patterns': (cli_params.get('patterns', 'N/A'), 'File patterns to match'),
            'start_date': (cli_params.get('start_date', 'All data'), 'Analysis start date (YYYY-MM-DD)'),
            'end_date': (cli_params.get('end_date', 'All data'), 'Analysis end date (YYYY-MM-DD)'),
            'factors': (cli_params.get('factors', 'Auto-optimized'), 'Number of PMF factors'),
            'max_factors': (cli_params.get('max_factors', 10), 'Maximum factors to test during optimization'),
            'models': (cli_params.get('models', 20), 'Number of PMF models to run'),
            'max_workers': (cli_params.get('max_workers', 2), 'Maximum parallel processes'),
            'output_dir': (cli_params.get('output_dir', 'pmf_results_esat'), 'Output directory'),
            'run_pca': (cli_params.get('run_pca', False), 'Run PCA analysis for comparison'),
            'create_pdf': (cli_params.get('create_pdf', False), 'Create PDF version of dashboard'),
            
            # Data processing parameters
            'remove_voc': (cli_params.get('remove_voc', False), 'Remove VOC species from analysis'),
            'scale_units': (cli_params.get('scale_units', True), 'Apply unit standardization to μg/m³'),
            'drop_row_threshold': (cli_params.get('drop_row_threshold', 0.5), 'Row drop threshold for missing values'),
            'zero_as_bdl': (cli_params.get('zero_as_bdl', True), 'Treat zeros as below detection limit'),
            'save_masks': (cli_params.get('save_masks', True), 'Save BDL and missing mask CSVs'),
            
            # Uncertainty parameters
            'uncertainty_mode': (cli_params["uncertainty_mode"], 'Uncertainty calculation method'),
            'uncertainty_ef_mdl': (cli_params.get('uncertainty_ef_mdl') or 'Built-in values', 'EF/MDL data source'),
            'uncertainty_epsilon': (cli_params.get('uncertainty_epsilon', 1e-12), 'Numerical uncertainty floor'),
            'legacy_min_u': (cli_params.get('legacy_min_u', 0.1), 'Minimum uncertainty (legacy mode)'),
            'uncertainty_bdl_policy': (cli_params.get('uncertainty_bdl_policy', 'five-sixth-mdl'), 'BDL uncertainty policy'),
            
            # S/N categorization parameters
            'snr_enable': (cli_params["snr_enable"], 'EPA S/N-based feature categorization'),
            'snr_weak_threshold': (cli_params["snr_weak_threshold"], 'S/N threshold for weak species'),
            'snr_bad_threshold': (cli_params["snr_bad_threshold"], 'S/N threshold for bad species'),
            'snr_bdl_weak_frac': (cli_params.get('snr_bdl_weak_frac', 0.6), 'BDL fraction for weak categorization'),
            'snr_bdl_bad_frac': (cli_params.get('snr_bdl_bad_frac', 0.8), 'BDL fraction for bad categorization'),
            'snr_missing_weak_frac': (cli_params.get('snr_missing_weak_frac', 0.2), 'Missing fraction for weak categorization'),
            'snr_missing_bad_frac': (cli_params.get('snr_missing_bad_frac', 0.4), 'Missing fraction for bad categorization'),
            'exclude_bad': (cli_params["exclude_bad"], 'Exclude bad species from analysis'),
            
            # ESAT algorithm parameters
            'method': (cli_params['method'], 'ESAT NMF algorithm (ls-nmf or ws-nmf)'),
            'init_method': (cli_params['init_method'], 'Matrix initialization method'),
            'init_norm': (cli_params['init_norm'], 'Normalize data before kmeans initialization'),
            'hold_h': (cli_params['hold_h'], 'Hold H (profile) matrix constant during training'),
            'delay_h': (cli_params['delay_h'], 'Hold H matrix for N iterations (-1 = disabled)'),
            'weight_aware_init': (cli_params.get('weight_aware_init', 'Auto'), 'Weight-aware initialization for species weighting'),
            
            # Robust training parameters
            'robust_fit': (cli_params['robust_fit'], 'Use robust loss during SA training (fallback only)'),
            'robust_alpha': (cli_params['robust_alpha'], 'Robust cutoff alpha for scaled residuals'),
            
            # Output and diagnostics
            'dashboard_snr_panel': (cli_params.get('dashboard_snr_panel', True), 'Include S/N panels in dashboard'),
            'write_diagnostics': (cli_params["write_diagnostics"], 'Write diagnostic CSV files'),
            'seed': (cli_params["seed"], 'Random seed for reproducibility'),
            'help_detail': (cli_params.get('help_detail', False), 'Show detailed CLI help'),
            
            # Regularization parameters
            'reg_species': (', '.join(cli_params.get('reg_species', [])) or 'None', 'Species to regularize'),
            'reg_lambda': (', '.join(map(str, cli_params.get('reg_lambda', []))) or 'Default', 'Regularization strength per species'),
            'reg_template': (', '.join(cli_params.get('reg_template', [])) or 'zero', 'Template type per species'),
            'reg_template_file': (', '.join(cli_params.get('reg_template_file', [])) or 'None', 'Template CSV files'),
            'reg_bursts': (cli_params.get('reg_bursts', 5), 'Number of train->prox cycles'),
            'reg_iter_per_burst': (cli_params.get('reg_iter_per_burst', 50), 'Max iterations per burst'),
            'reg_tol': (cli_params.get('reg_tol', 1e-4), 'Early stop tolerance'),
            'reg_elastic_l1': (cli_params.get('reg_elastic_l1', 0.0), 'Elastic-net L1 penalty'),
            
            # Bootstrap error estimation parameters
            'bootstrap': (cli_params.get('bootstrap', False), 'Enable bootstrap error estimation after PMF analysis'),
            'bootstrap_n': (cli_params.get('bootstrap_n', 100), 'Number of bootstrap samples to run'),
            'bootstrap_block_size': (cli_params.get('bootstrap_block_size') or 'Auto-estimated', 'Block size for temporal bootstrap resampling'),
            'bootstrap_threshold': (cli_params.get('bootstrap_threshold', 0.6), 'Factor mapping threshold for bootstrap correlation'),
            'bootstrap_parallel': (cli_params.get('bootstrap_parallel', True), 'Enable parallel processing for bootstrap'),
            'bootstrap_cpus': (cli_params.get('bootstrap_cpus') or 'All available', 'Number of CPUs for bootstrap parallel processing'),
            'bootstrap_seed': (cli_params.get('bootstrap_seed') or 'Main seed', 'Random seed for bootstrap resampling'),
            'bootstrap_keep_h': (cli_params.get('bootstrap_keep_h', True), 'Keep factor profiles (H matrix) from bootstrap samples'),
            'bootstrap_reuse_seed': (cli_params.get('bootstrap_reuse_seed', True), 'Reuse seed across bootstrap samples'),
            'bootstrap_overlapping': (cli_params.get('bootstrap_overlapping', False), 'Allow overlapping blocks in bootstrap resampling'),
            
            # Complaint correlation analysis parameters
            'complaint_correlation_hours': (cli_params.get('complaint_correlation_hours', 0), 'Time window in hours for complaint correlation analysis (0 = daily aggregation)'),
        }
        
        # Add species weighting if any applied
        if hasattr(self, '_species_weights_applied') and self._species_weights_applied:
            for species, weight in self._species_weights_applied.items():
                param_info[f'species_weight_{species}'] = (weight, f'Uncertainty multiplier for {species}')
        
        # Add species exclusions if any applied
        if hasattr(self, '_excluded_species_applied') and self._excluded_species_applied:
            excluded_list = ', '.join(sorted(self._excluded_species_applied))
            param_info['excluded_species'] = (excluded_list, 'Species completely removed from PMF analysis')
        
        for param, (value, description) in param_info.items():
            html_section += f"""
                    <tr><td>--{param.replace('_', '-')}</td><td>{value}</td><td>{description}</td></tr>
            """
        
        html_section += f"""
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
                <h2>[CONFIG] Run Configuration</h2>
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
                <h2>[UNCERTAINTY MODE] EPA Uncertainty Policy (PMF 5.0)</h2>
                <ul>
                    <li><strong>Above MDL:</strong> U = √((EF × conc)2 + (0.5 × MDL)2)</li>
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
                <h2>[UNCERTAINTY MODE] Legacy Uncertainty Policy</h2>
                <ul>
                    <li><strong>Method:</strong> Fixed MDL/Error Fraction table</li>
                    <li><strong>Above MDL:</strong> U = √((EF × conc)2 + MDL2)</li>
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
                <h2>[NUMBERS] S/N Categorization Summary (EPA PMF 5.0)</h2>
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
        
        # Add species weighting section if any weights were applied
        if hasattr(self, '_species_weights_applied') and self._species_weights_applied:
            html_content += f"""
            <div class="epa-section">
                <h2>[SYMBOL]️ Species Uncertainty Weighting</h2>
                <p>The following species had their uncertainties multiplied to downweight them in the PMF objective function:</p>
                <table>
                    <tr><th>Species</th><th>Uncertainty Multiplier</th><th>Effect</th></tr>
            """
            
            for species, weight in sorted(self._species_weights_applied.items()):
                effect = f"Downweighted {weight}× (less influence on factors)"
                html_content += f"""
                    <tr>
                        <td><strong>{species}</strong></td>
                        <td>{weight}</td>
                        <td>{effect}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                <p><small><em>Note: Uncertainty multiplication reduces species influence in ESAT LS-PMF optimization without changing concentrations.</em></small></p>
            </div>
            """
        
        # Add species exclusion section if any species were excluded
        if hasattr(self, '_excluded_species_applied') and self._excluded_species_applied:
            html_content += f"""
            <div class="epa-section">
                <h2>[EXCLUSIONS] Species Exclusions</h2>
                <p>The following species were completely removed from PMF analysis via the --exclude-species flag:</p>
                <table>
                    <tr><th>Species</th><th>Reason</th></tr>
            """
            
            for species in sorted(self._excluded_species_applied):
                html_content += f"""
                    <tr>
                        <td><strong>{species}</strong></td>
                        <td>CLI exclusion flag</td>
                    </tr>
                """
            
            # Add any species that were requested but not found
            if hasattr(self, '_excluded_species_not_found') and self._excluded_species_not_found:
                for species in sorted(self._excluded_species_not_found):
                    html_content += f"""
                        <tr>
                            <td><em>{species}</em></td>
                            <td>Requested but not found in data</td>
                        </tr>
                    """
            
            html_content += """
                </table>
                <p><small><em>Note: Excluded species are completely removed from concentration and uncertainty matrices before PMF analysis.</em></small></p>
            </div>
            """
        
        # Add regularization section if regularization was used
        if hasattr(self, '_reg_enabled') and self._reg_enabled and hasattr(self, '_reg_plan'):
            html_content += f"""
            <div class="epa-section">
                <h2>[REG] Species Regularization Applied</h2>
                <p><strong>Regularization Mode:</strong> Ridge regularization with zero template for species push-out</p>
                <table>
                    <tr><th>Species</th><th>Lambda (λ)</th><th>Template</th><th>Effect</th></tr>
            """
            
            for reg_item in self._reg_plan:
                species = reg_item['species']
                lambda_val = reg_item['lambda']
                template_type = reg_item.get('template_type', 'zero')
                html_content += f"""
                    <tr>
                        <td><strong>{species}</strong></td>
                        <td>{lambda_val}</td>
                        <td>{template_type.title()}</td>
                        <td>Push-out (minimize {species} factor loadings)</td>
                    </tr>
                """
            
            # Check if convergence info is available
            convergence_info = "Not available"
            if hasattr(self, '_reg_burst_diagnostics') and self._reg_burst_diagnostics:
                final_burst = self._reg_burst_diagnostics[-1]
                max_rel_change = final_burst.get('max_rel_change', 0)
                converged = max_rel_change < getattr(self, 'reg_tol', 0.0001)
                convergence_info = f"{'Converged' if converged else 'Not converged'} (final rel_change={max_rel_change:.3e})"
            
            html_content += f"""
                </table>
                <p><strong>Convergence Status:</strong> {convergence_info}</p>
                <p><strong>Method:</strong> Staged training with {getattr(self, 'reg_bursts', 5)} bursts of {getattr(self, 'reg_iter_per_burst', 50)} iterations each</p>
                <p><strong>Mathematical Approach:</strong> Ridge regularization with proximal updates - min ||V-WH||² + λ||H[:,species]-template||²</p>
            </div>
            """
        
        # Add closure metrics section if available
        if hasattr(self, '_closure_df') and self._closure_df is not None:
            html_content += f"""
            <div class="epa-section">
                <h2>[CLOSURE] Mass Closure / Fit Divergence Analysis</h2>
                <p><strong>Purpose:</strong> Quantify how well the PMF model reconstructs measured concentrations (closure) and identify species where regularization affects fit quality.</p>
            """
            
            # Add group closure summary
            if hasattr(self, '_group_summary') and self._group_summary:
                html_content += "<h3>Group-Level Closure:</h3><ul>"
                for group, data in self._group_summary.items():
                    html_content += f"<li><strong>{group}:</strong> {data['closure_pct']:.1f}% ({data['n_species']} species)</li>"
                html_content += "</ul>"
            
            # Add regularization context if active
            if hasattr(self, '_reg_enabled') and self._reg_enabled and hasattr(self, '_reg_plan'):
                reg_species = [ri['species'] for ri in self._reg_plan]
                reg_species_str = ', '.join(reg_species)
                html_content += f"""
                <p><strong>Regularization Impact:</strong> {reg_species_str} closure may decrease as lambda (λ) increases due to push-out regularization forcing smaller factor loadings.</p>
                """
            
            # Add interpretation guide
            html_content += f"""
            <h3>Interpretation Guide:</h3>
            <ul>
                <li><strong>Closure %:</strong> (Reconstructed Sum / Measured Sum) × 100. Values near 100% indicate good fit.</li>
                <li><strong>Weighted Closure %:</strong> Same calculation but weighted by 1/uncertainty². More robust to outliers.</li>
                <li><strong>Red bars:</strong> Regularized species (expected to show closure reduction as λ increases).</li>
                <li><strong>Q Share %:</strong> Fraction of total model Q contributed by each species (indicates fit difficulty).</li>
            </ul>
            """
            
            html_content += f"""
                <p><strong>Files:</strong> See <code>{self.filename_prefix}_closure_summary.csv</code> for detailed per-species metrics.</p>
            </div>
            """
        
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
        
        # Add bootstrap uncertainty plots if available
        if self.bootstrap and self.bootstrap_results:
            print("   [BOOTSTRAP] Adding bootstrap uncertainty plots to dashboard...")
            bootstrap_plots = self.create_bootstrap_dashboard()
            
            if bootstrap_plots:
                html_content += """
            <h2 class="section-header">Bootstrap Error Estimation</h2>
            <p>Bootstrap error estimation provides uncertainty quantification for PMF factors using resampling methods.</p>
            
            <style>
                .bootstrap-grid {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    grid-gap: 15px;
                    width: 100%;
                }
                .bootstrap-grid .plot-panel {
                    background-color: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 4px;
                    padding: 10px;
                }
                .bootstrap-grid h3 {
                    font-size: 14px;
                    margin-top: 5px;
                    margin-bottom: 8px;
                }
                .bootstrap-grid img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 0 auto;
                }
            </style>
            
            <div class="bootstrap-grid">
            """
                
                for plot_file in bootstrap_plots:
                    if plot_file.suffix.lower() in image_extensions:
                        plot_name = plot_file.stem.replace('_', ' ').title()
                        # Adjust path for bootstrap plots
                        plot_rel_path = f"bootstrap_plots/{plot_file.name}"
                        html_content += f"""
                <div class="plot-panel">
                    <h3>{plot_name}</h3>
                    <img src="{plot_rel_path}" alt="{plot_name}">
                </div>
                """
                
                html_content += "</div>"  # Close bootstrap grid
                
                print(f"   [OK] Added {len(bootstrap_plots)} bootstrap plots to dashboard")
        
        # Add CLI flags record at bottom
        html_content += self._get_cli_flags_html_section()
        
        html_content += "</body></html>"
        
        # Save HTML dashboard
        html_file = self.output_dir / f"{self.filename_prefix}_pmf_dashboard.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[FILE] HTML Dashboard: {html_file}")
    
    def generate_report(self):
        """Generate comprehensive PMF analysis report."""
        if not self.best_model:
            print("[ERROR] No PMF model available for report generation")
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
- [OK] Appropriate uncertainty estimation using EPA formula
- [OK] Missing value treatment using EPA Method 1
- [OK] Batch modeling with {self.models} runs for robustness
- [OK] Comprehensive diagnostic plots generated
- [OK] ESAT Rust-optimized PMF implementation

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
        
        print(f"[FILE] Analysis report: {report_path}")
    
    def _create_residual_plots(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create residual analysis plots (EPA recommended)."""
        print("   [SEARCH] Creating residual analysis plots...")
        
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
        print(f"   [OK] Saved: residual_analysis.png")
    
    def _create_correlation_plots(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create factor correlation analysis plots."""
        print("   [LINK] Creating correlation analysis plots...")
        
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
        print(f"   [OK] Saved: correlation_analysis.png")
    
    def _create_source_contribution_plots(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create source contribution analysis plots."""
        print("   [DATA] Creating source contribution plots...")
        
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
        print(f"   [OK] Saved: source_contribution_analysis.png")
    
    def _load_complaint_data_for_overlay(self):
        """Load complaint data from parquet file for overlay plots."""
        try:
            from mmf_config import get_mmf_parquet_file
            
            # Get the parquet file path (using test data)
            parquet_file = get_mmf_parquet_file(self.station, use_test_data=True)
            
            # Load the full parquet data
            full_data = pd.read_parquet(parquet_file)
            
            # Check if complaint data exists
            if 'Odour_Reports' not in full_data.columns:
                print("   [INFO] No complaint data available for overlay")
                return None
            
            # Convert datetime column to proper index
            full_data['datetime'] = pd.to_datetime(full_data['datetime'])
            full_data = full_data.set_index('datetime')
            
            # Filter to the analysis time period
            if hasattr(self, 'start_date') and hasattr(self, 'end_date'):
                start_date = pd.to_datetime(self.start_date)
                end_date = pd.to_datetime(self.end_date)
                period_data = full_data[(full_data.index >= start_date) & (full_data.index <= end_date)]
            else:
                period_data = full_data
            
            # Extract complaint data
            complaint_data = period_data['Odour_Reports'].copy()
            
            # Handle missing data markers (-1) and convert to NaN
            complaint_data = complaint_data.replace(-1, np.nan)
            
            # Aggregate daily (since complaints are daily)
            daily_complaints = complaint_data.resample('D').first()
            
            # Keep timestamps at midnight for data alignment
            # (The noon shift was causing alignment issues with concentration data)
            # daily_complaints.index = daily_complaints.index + pd.Timedelta(hours=12)
            
            print(f"   [COMPLAINTS] Loaded {daily_complaints.notna().sum()} days with complaint data")
            return daily_complaints
            
        except Exception as e:
            print(f"   [WARN] Could not load complaint data: {e}")
            return None
    
    def _create_temporal_analysis_plots(self, dashboard_dir, plot_files, G_contributions):
        """Create temporal pattern analysis plots with optional complaint data overlay."""
        print("   [TIME] Creating temporal analysis plots...")
        
        # Note: Complaint data overlay moved to Factor Contributions plot
        
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
        
        # Plot 1: Time series with trend and complaint overlay
        ax1 = axes[0, 0]
        
        # Plot factor contributions
        for i in range(self.factors):
            factor_color = self.color_manager.get_factor_color(i)
            if has_datetime:
                ax1.plot(datetime_index, G_contributions[:, i], label=f'Factor {i+1}', 
                        color=factor_color, alpha=0.7, linewidth=1.5)
            else:
                ax1.plot(G_contributions[:, i], label=f'Factor {i+1}', 
                        color=factor_color, alpha=0.7, linewidth=1.5)
        
        # Note: Complaint overlay moved to main Factor Contributions plot
        
        ax1.set_title('Factor Contributions Time Series with Complaint Overlay')
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
        
        # Plot 4: Cumulative contribution with complaint overlay
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
        
        # Note: Complaint overlay moved to main Factor Contributions plot
        
        ax4.set_title('Cumulative Factor Contributions with Complaint Events')
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
        print(f"   [OK] Saved: temporal_analysis.png")
    
    def _create_uncertainty_plots(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create uncertainty and bootstrap analysis plots."""
        print("   [INIT] Creating uncertainty analysis plots...")
        
        if not USE_BATCH_SA or len(self.batch_models.results) < 5:
            print("   [WARN] Skipping uncertainty plots (requires multiple models)")
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
        print(f"   [OK] Saved: uncertainty_analysis.png")
    
    def _create_diagnostic_scatters(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """Create diagnostic scatter plots for model validation."""
        print("   [ANALYSIS] Creating diagnostic scatter plots...")
        
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
        
        # Calculate R2
        from sklearn.metrics import r2_score
        r2 = r2_score(original_data.flatten(), reconstructed.flatten())
        
        ax1.set_xlabel('Observed Concentration')
        ax1.set_ylabel('Predicted Concentration')
        ax1.set_title(f'Observed vs Predicted (R2 = {r2:.3f})')
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
        
        # Color bars by R2 quality
        for i, (bar, r2_val) in enumerate(zip(bars, species_r2)):
            if r2_val >= 0.8:
                bar.set_color('green')
            elif r2_val >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('red')
            
            # Add R2 value on top of bar (only for non-negative values)
            if r2_val >= 0:
                ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{r2_val:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax3.set_title('Species-Specific Model Performance (R2)')
        ax3.set_ylabel('R2 Value')
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
        print(f"   [OK] Saved: diagnostic_scatters.png")
    
    def _create_optimization_plot(self, dashboard_dir, plot_files):
        """Create enhanced Q(robust) vs number of factors optimization plot with EPA reference lines."""
        print("   [NUMBERS] Creating factor optimization plot...")
        
        # Check if optimization data is available
        if not hasattr(self, 'optimization_q_values') or not self.optimization_q_values:
            print("   [WARN] No optimization data available - skipping optimization plot")
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
                  'Q/DoF <= 1.5: Excellent fit\n'
                  'Q/DoF <= 2.0: Good fit\n'
                  'Q/DoF <= 3.0: Fair fit\n'
                  'Q/DoF > 3.0: Poor fit')
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=8,
               verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_optimization_q_vs_factors.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   [OK] Saved: optimization_q_vs_factors.png")
    
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
    
    def prepare_bootstrap_inputs(self):
        """
        Prepare inputs for bootstrap error estimation.
        
        Returns:
            tuple: (best_sa, feature_labels, data_handler, block_size, seed)
        """
        if not HAS_BOOTSTRAP:
            raise RuntimeError("Bootstrap functionality not available. Please check ESAT installation.")
        
        if self.best_model is None:
            raise RuntimeError("No PMF model available. Run PMF analysis first.")
        
        print("[BOOTSTRAP] Preparing bootstrap inputs...")
        
        # Get best SA model
        if hasattr(self.best_model, 'sa'):
            best_sa = self.best_model.sa  # For BatchSA models
        else:
            best_sa = self.best_model  # For single SA models
        
        # Get feature labels (species names)
        feature_labels = list(self.concentration_data.columns)
        print(f"   Feature labels: {feature_labels}")
        
        # Create DataHandler for optimal block size estimation
        print("   Creating DataHandler for block size estimation...")
        try:
            # Create DataFrames with proper index for DataHandler
            concentration_df = pd.DataFrame(self.concentration_data.values, 
                                          columns=feature_labels, 
                                          index=self.concentration_data.index)
            uncertainty_df = pd.DataFrame(self.uncertainty_data.values, 
                                        columns=feature_labels, 
                                        index=self.uncertainty_data.index)
            
            # Use DataHandler.load_dataframe for proper initialization
            data_handler = DataHandler.load_dataframe(concentration_df, uncertainty_df)
        except Exception as e:
            print(f"   [WARN] DataHandler creation failed: {e}")
            data_handler = None
        
        # Estimate optimal block size if not specified
        if self.bootstrap_block_size is None:
            print("   Estimating optimal block size using DataHandler...")
            try:
                if data_handler and hasattr(data_handler, 'optimal_block'):
                    optimal_block_size = data_handler.optimal_block
                else:
                    raise RuntimeError("DataHandler optimal_block not available")
                print(f"   Estimated optimal block size: {optimal_block_size}")
                
                # Sanity check and warnings for block size
                n_samples = self.concentration_data.shape[0]
                if optimal_block_size >= n_samples / 10:
                    print(f"   [WARN] Large block size ({optimal_block_size}) relative to dataset ({n_samples} samples)")
                    print(f"   [WARN] This may reduce bootstrap effectiveness. Consider more data or smaller block size.")
                elif optimal_block_size < 3:
                    print(f"   [WARN] Very small block size ({optimal_block_size}). May indicate insufficient temporal correlation.")
                    print(f"   [WARN] Consider using block_size=5-10 manually if results seem unstable.")
                
            except Exception as e:
                print(f"   [ERROR] Block size estimation failed: {e}")
                optimal_block_size = min(10, max(3, len(self.concentration_data) // 20))  # Adaptive fallback
                print(f"   [FALLBACK] Using adaptive default block size: {optimal_block_size}")
        else:
            optimal_block_size = self.bootstrap_block_size
            print(f"   Using user-specified block size: {optimal_block_size}")
            
            # Validate user-specified block size
            n_samples = self.concentration_data.shape[0]
            if optimal_block_size >= n_samples / 5:
                print(f"   [WARN] User block size ({optimal_block_size}) is very large for dataset ({n_samples} samples)")
        
        # Determine seed for bootstrap
        bootstrap_seed = self.bootstrap_seed if self.bootstrap_seed is not None else self.seed
        print(f"   Bootstrap seed: {bootstrap_seed}")
        
        # Store actual values for later reference
        self._actual_bootstrap_block_size = optimal_block_size
        self._actual_bootstrap_seed = bootstrap_seed
        
        return best_sa, feature_labels, data_handler, optimal_block_size, bootstrap_seed
    
    def run_bootstrap_analysis(self):
        """
        Run bootstrap error estimation on the best PMF model.
        
        Returns:
            dict: Bootstrap results containing paths to output files
        """
        if not self.bootstrap:
            print("[BOOTSTRAP] Bootstrap disabled, skipping...")
            return None
        
        if not HAS_BOOTSTRAP:
            print("[ERROR] Bootstrap functionality not available. Please check ESAT installation.")
            return None
        
        print(f"[BOOTSTRAP] Starting bootstrap error estimation with {self.bootstrap_n} samples...")
        
        try:
            # Prepare bootstrap inputs
            best_sa, feature_labels, data_handler, block_size, seed = self.prepare_bootstrap_inputs()
            
            # Create Bootstrap instance
            print("   Creating Bootstrap instance...")
            bootstrap = Bootstrap(
                sa=best_sa,
                feature_labels=feature_labels,
                model_selected=0,  # Use the best model
                bootstrap_n=self.bootstrap_n,
                block_size=block_size,
                threshold=self.bootstrap_threshold,
                parallel=self.bootstrap_parallel,
                cpus=self.bootstrap_cpus,
                seed=seed
            )
            
            print(f"   Bootstrap configuration:")
            print(f"     Samples: {self.bootstrap_n}")
            print(f"     Block size: {block_size}")
            print(f"     Threshold: {self.bootstrap_threshold}")
            print(f"     Parallel: {self.bootstrap_parallel}")
            print(f"     CPUs: {self.bootstrap_cpus or 'all available'}")
            print(f"     Seed: {seed}")
            
            # Provide runtime guidance
            if self.bootstrap_n >= 200:
                print(f"   [INFO] Large bootstrap sample size ({self.bootstrap_n}) - expect longer runtime")
                print(f"   [INFO] Consider starting with --bootstrap-n 20 for testing, then scale up")
            if self.bootstrap_parallel and self.bootstrap_cpus and self.bootstrap_cpus >= mp.cpu_count():
                print(f"   [WARN] Using all CPUs may impact system responsiveness during bootstrap")
            
            # Run bootstrap analysis
            print("   Running bootstrap analysis...")
            bootstrap.run(
                keep_H=self.bootstrap_keep_h,
                reuse_seed=self.bootstrap_reuse_seed,
                block=True,  # Use block resampling
                overlapping=self.bootstrap_overlapping
            )
            
            print(f"[OK] Bootstrap analysis completed successfully")
            
            # Save bootstrap outputs
            saved_results = self.save_bootstrap_outputs(bootstrap)
            
            # Store results
            self.bootstrap_results = saved_results
            
            return saved_results
            
        except Exception as e:
            print(f"[ERROR] Bootstrap analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_bootstrap_outputs(self, bootstrap_obj):
        """
        Save ESAT Bootstrap object results to organized output structure.
        
        Args:
            bootstrap_obj (Bootstrap): ESAT Bootstrap object with completed analysis
            
        Returns:
            dict: Paths to saved bootstrap output files
        """
        print("[BOOTSTRAP] Saving bootstrap outputs...")
        
        # Create error estimation output directory
        error_dir = self.output_dir / "error"
        error_dir.mkdir(exist_ok=True, parents=True)
        
        saved_files = {}
        
        try:
            # Save bootstrap object as pickle (full state) - use absolute path
            bootstrap_name = f"bootstrap-{self.filename_prefix}"
            error_dir_abs = error_dir.resolve()  # Convert to absolute path
            
            print(f"   Saving bootstrap files to: {error_dir_abs}")
            
            # Save bootstrap pickle - ESAT Bootstrap.save() often returns None but creates files successfully
            expected_pickle = error_dir_abs / f"{bootstrap_name}.pkl"
            pickle_path = bootstrap_obj.save(bootstrap_name, str(error_dir_abs), pickle_result=True)
            
            # Handle ESAT Bootstrap.save() return behavior (often returns None even when successful)
            if expected_pickle.exists():
                saved_files['pickle'] = expected_pickle
                print(f"   Saved: {expected_pickle.name}")
                if pickle_path is None:
                    print(f"   [INFO] ESAT Bootstrap.save() returned None but file was created successfully")
            else:
                print(f"   [ERROR] Bootstrap pickle file not found: {expected_pickle}")
                print(f"   [DEBUG] ESAT Bootstrap.save() returned: {pickle_path}")
            
            # Save as JSON/CSV artifacts for dashboard consumption 
            json_path = bootstrap_obj.save(bootstrap_name, str(error_dir_abs), pickle_result=False)
            if json_path:
                # Bootstrap.save() returns directory path when pickle_result=False
                saved_files['json_artifacts'] = Path(json_path)
                print(f"   Saved: JSON/CSV artifacts to {Path(json_path).name}")
            else:
                print(f"   [WARN] Bootstrap JSON/CSV save returned None")
            
            # Create summary information file
            summary_file = error_dir / f"{self.filename_prefix}_bootstrap_summary.json"
            # Get actual parameters used (including computed block size)
            actual_block_size = getattr(self, '_actual_bootstrap_block_size', self.bootstrap_block_size)
            actual_seed = getattr(self, '_actual_bootstrap_seed', self.bootstrap_seed)
            
            summary_info = {
                "bootstrap_parameters": {
                    "n_samples": self.bootstrap_n,
                    "block_size": actual_block_size,  # Use actual computed value
                    "threshold": self.bootstrap_threshold,
                    "parallel": self.bootstrap_parallel,
                    "cpus": self.bootstrap_cpus,
                    "seed": actual_seed,  # Use actual seed value
                    "keep_h": self.bootstrap_keep_h,
                    "reuse_seed": self.bootstrap_reuse_seed,
                    "overlapping": self.bootstrap_overlapping
                },
                "base_model": {
                    "factors": self.factors,
                    "models": self.models,
                    "species": list(self.concentration_data.columns),
                    "n_samples": len(self.concentration_data),
                    "date_range": f"{self.start_date} to {self.end_date}"
                },
                "output_files": {k: str(v) for k, v in saved_files.items()},
                "created": datetime.now().isoformat()
            }
            
            with open(summary_file, 'w') as f:
                json.dump(summary_info, f, indent=2)
            
            saved_files['summary'] = summary_file
            print(f"   Saved: {summary_file.name}")
            
            print(f"[OK] Bootstrap outputs saved to: {error_dir}")
            return saved_files
            
        except Exception as e:
            print(f"[ERROR] Failed to save bootstrap outputs: {e}")
            return {}
    
    def create_bootstrap_dashboard(self):
        """
        Create bootstrap error estimation dashboard with uncertainty visualizations.
        
        Returns:
            list: Paths to created bootstrap plot files
        """
        if not self.bootstrap or self.bootstrap_results is None:
            print("[BOOTSTRAP] No bootstrap results available for dashboard creation")
            return []
        
        print("[DASHBOARD] Creating bootstrap uncertainty dashboard...")
        
        # Create bootstrap plots directory
        bootstrap_dir = self.output_dir / "bootstrap_plots"
        bootstrap_dir.mkdir(exist_ok=True, parents=True)
        
        plot_files = []
        
        try:
            # Load bootstrap object from pickle for full access to results
            if 'pickle' in self.bootstrap_results and self.bootstrap_results['pickle']:
                pickle_path = Path(self.bootstrap_results['pickle']).resolve()  # Ensure absolute path
                print(f"   Loading bootstrap object from: {pickle_path}")
                
                # Import Bootstrap class for loading
                from esat.error.bootstrap import Bootstrap
                bootstrap_obj = Bootstrap.load(str(pickle_path))
                
                if bootstrap_obj is None:
                    print("   [ERROR] Failed to load bootstrap pickle file")
                    # Try alternative approach using summary data
                    return self._create_bootstrap_plots_from_summary(bootstrap_dir)
                
                print(f"   [OK] Bootstrap object loaded successfully")
                
                # Create factor variability plots (per-factor profile uncertainty)
                for factor_idx in range(self.factors):
                    factor_plot = self._create_factor_variability_plot(bootstrap_obj, bootstrap_dir, factor_idx)
                    if factor_plot:
                        plot_files.append(factor_plot)
                
                # Create species uncertainty plot (across all factors)
                species_plot = self._create_species_uncertainty_plot(bootstrap_obj, bootstrap_dir)
                if species_plot:
                    plot_files.append(species_plot)
                
                # Create contribution uncertainty plot (time series of factor contributions)
                contrib_plot = self._create_contribution_uncertainty_plot(bootstrap_obj, bootstrap_dir)
                if contrib_plot:
                    plot_files.append(contrib_plot)
                
                # Create bootstrap summary statistics plot
                summary_plot = self._create_bootstrap_summary_plot(bootstrap_obj, bootstrap_dir)
                if summary_plot:
                    plot_files.append(summary_plot)
            else:
                print("   [WARN] No bootstrap pickle file available, creating summary plots instead")
                return self._create_bootstrap_plots_from_summary(bootstrap_dir)
            
            print(f"[OK] Created {len(plot_files)} bootstrap dashboard plots")
            return plot_files
            
        except Exception as e:
            print(f"[ERROR] Failed to create bootstrap dashboard: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _create_bootstrap_plots_from_summary(self, bootstrap_dir):
        """
        Create basic bootstrap plots from summary information when pickle loading fails.
        
        Args:
            bootstrap_dir (Path): Output directory for plots
            
        Returns:
            list: Paths to created plot files
        """
        print("   [FALLBACK] Creating basic bootstrap summary plots...")
        
        plot_files = []
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Set larger font sizes for bootstrap plots
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12
            })
            
            # Create a basic bootstrap summary plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get bootstrap parameters from summary
            if 'summary' in self.bootstrap_results:
                summary_file = self.bootstrap_results['summary']
                with open(summary_file, 'r') as f:
                    import json
                    summary = json.load(f)
                
                bs_params = summary.get('bootstrap_parameters', {})
                base_model = summary.get('base_model', {})
                
                # Create text summary plot
                ax.text(0.5, 0.8, 'Bootstrap Error Estimation Completed', 
                       ha='center', va='center', fontsize=20, fontweight='bold')
                
                info_text = f"""Bootstrap Configuration:
• Samples: {bs_params.get('n_samples', 'N/A')}
• Block Size: {bs_params.get('block_size', 'N/A')}
• Threshold: {bs_params.get('threshold', 'N/A')}
• Parallel: {bs_params.get('parallel', 'N/A')}
• CPUs: {bs_params.get('cpus', 'N/A')}
• Seed: {bs_params.get('seed', 'N/A')}

Base Model:
• Factors: {base_model.get('factors', 'N/A')}
• Species: {len(base_model.get('species', []))}
• Samples: {base_model.get('n_samples', 'N/A')}
• Date Range: {base_model.get('date_range', 'N/A')}

Note: Bootstrap uncertainty plots require pickle files.
Bootstrap analysis completed successfully but
visualization data is not available."""
                
                ax.text(0.5, 0.4, info_text, ha='center', va='center', 
                       fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_title('Bootstrap Error Estimation Summary', fontsize=18, fontweight='bold')
                ax.axis('off')
                
                plt.tight_layout()
                plot_path = bootstrap_dir / f"{self.filename_prefix}_bootstrap_summary.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                plot_files.append(plot_path)
                print(f"   [OK] Created bootstrap summary plot: {plot_path.name}")
            
            return plot_files
            
        except Exception as e:
            print(f"   [ERROR] Failed to create fallback bootstrap plots: {e}")
            return []
    
    def _create_factor_variability_plot(self, bootstrap_obj, output_dir, factor_idx):
        """
        Create factor variability plot showing bootstrap uncertainty in factor profiles.
        
        Args:
            bootstrap_obj (Bootstrap): ESAT Bootstrap object
            output_dir (Path): Output directory for plots
            factor_idx (int): Factor index to plot
            
        Returns:
            Path: Path to created plot file
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Set larger font sizes for bootstrap plots
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12
            })
            
            # Get factor profile distributions from bootstrap
            if not hasattr(bootstrap_obj, 'bs_profiles') or factor_idx not in bootstrap_obj.bs_profiles:
                print(f"   [WARN] No bootstrap profiles available for factor {factor_idx}")
                return None
                
            factor_profiles = np.array(bootstrap_obj.bs_profiles[factor_idx])
            base_profile = bootstrap_obj.base_H[factor_idx] / np.sum(bootstrap_obj.base_H[factor_idx])  # Normalize
            species_names = list(self.concentration_data.columns)
            
            # Create box plot showing profile uncertainty
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Box plots for each species
            box_data = [factor_profiles[:, i] * 100 for i in range(len(species_names))]
            bp = ax.boxplot(box_data, positions=range(len(species_names)), patch_artist=True)
            
            # Color H2S factor red, others blue
            if self.color_manager and self.color_manager.is_h2s_factor(factor_idx):
                box_color = '#d62728'  # Red for H2S factor
                factor_label = f"Factor {factor_idx + 1} (H2S-dominant)"
            else:
                box_color = '#1f77b4'  # Blue for other factors
                factor_label = f"Factor {factor_idx + 1}"
            
            for patch in bp['boxes']:
                patch.set_facecolor(box_color)
                patch.set_alpha(0.6)
            
            # Overlay base model profile as red points
            base_percentages = base_profile * 100
            ax.scatter(range(len(species_names)), base_percentages, 
                      color='red', s=40, zorder=5, label='Base Model')
            
            # Format plot
            ax.set_xticks(range(len(species_names)))
            ax.set_xticklabels(species_names, rotation=45, ha='right', fontsize=12)
            ax.set_ylabel('Percentage of Species (%)', fontsize=14)
            ax.set_title(f'{factor_label} - Profile Variability from Bootstrap Analysis', fontsize=16)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            plt.tight_layout()
            plot_path = output_dir / f"{self.filename_prefix}_bootstrap_factor_{factor_idx + 1}_profile.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"   Error creating factor variability plot for factor {factor_idx}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_species_uncertainty_plot(self, bootstrap_obj, output_dir):
        """
        Create species profile uncertainty plot across all factors.
        
        Args:
            bootstrap_obj (Bootstrap): ESAT Bootstrap object
            output_dir (Path): Output directory for plots
            
        Returns:
            Path: Path to created plot file
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Set larger font sizes for bootstrap plots
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12
            })
            
            if not hasattr(bootstrap_obj, 'bs_profiles'):
                print("   [WARN] No bootstrap profiles available for species uncertainty plot")
                return None
                
            species_names = list(self.concentration_data.columns)
            n_species = len(species_names)
            n_factors = self.factors
            
            # Create subplot grid
            fig, axes = plt.subplots(n_factors, 1, figsize=(14, 4 * n_factors), sharex=True)
            if n_factors == 1:
                axes = [axes]
            
            for factor_idx in range(n_factors):
                ax = axes[factor_idx]
                
                if factor_idx in bootstrap_obj.bs_profiles:
                    factor_profiles = np.array(bootstrap_obj.bs_profiles[factor_idx])
                    base_profile = bootstrap_obj.base_H[factor_idx] / np.sum(bootstrap_obj.base_H[factor_idx])
                    
                    # Calculate percentiles for uncertainty bands
                    p5 = np.percentile(factor_profiles, 5, axis=0) * 100
                    p25 = np.percentile(factor_profiles, 25, axis=0) * 100
                    p75 = np.percentile(factor_profiles, 75, axis=0) * 100
                    p95 = np.percentile(factor_profiles, 95, axis=0) * 100
                    median = np.percentile(factor_profiles, 50, axis=0) * 100
                    base_pct = base_profile * 100
                    
                    x = range(n_species)
                    
                    # Plot uncertainty bands
                    ax.fill_between(x, p5, p95, alpha=0.2, color='lightgray', label='5-95% range')
                    ax.fill_between(x, p25, p75, alpha=0.4, color='lightblue', label='25-75% range')
                    
                    # Plot median and base model
                    ax.plot(x, median, 'b-', linewidth=2, label='Bootstrap Median')
                    
                    # Color H2S factor red, others blue
                    if self.color_manager and self.color_manager.is_h2s_factor(factor_idx):
                        base_color = '#d62728'  # Red for H2S factor
                        factor_label = f"Factor {factor_idx + 1} (H2S-dominant)"
                    else:
                        base_color = '#1f77b4'  # Blue for other factors
                        factor_label = f"Factor {factor_idx + 1}"
                    
                    ax.scatter(x, base_pct, color=base_color, s=40, zorder=5, label='Base Model')
                    
                    ax.set_ylabel('Percentage (%)', fontsize=14)
                    ax.set_title(factor_label, fontsize=14)
                    ax.grid(True, alpha=0.3)
                    
                    if factor_idx == 0:  # Only show legend on first subplot
                        ax.legend(loc='upper right', fontsize=12)
                
            # Format x-axis for bottom subplot
            axes[-1].set_xticks(range(n_species))
            axes[-1].set_xticklabels(species_names, rotation=45, ha='right', fontsize=12)
            axes[-1].set_xlabel('Species', fontsize=14)
            
            plt.suptitle('Species Profile Uncertainty from Bootstrap Analysis', fontsize=18)
            plt.tight_layout()
            
            plot_path = output_dir / f"{self.filename_prefix}_bootstrap_species_uncertainty.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"   Error creating species uncertainty plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_contribution_uncertainty_plot(self, bootstrap_obj, output_dir):
        """
        Create factor contribution uncertainty plot.
        
        Args:
            bootstrap_obj (Bootstrap): ESAT Bootstrap object
            output_dir (Path): Output directory for plots
            
        Returns:
            Path: Path to created plot file
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Set larger font sizes for bootstrap plots
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 16,
                'axes.labelsize': 14,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12
            })
            
            if not hasattr(bootstrap_obj, 'bs_factor_contributions'):
                print("   [WARN] No bootstrap factor contributions available")
                return None
                
            species_names = list(self.concentration_data.columns)
            n_species = len(species_names)
            n_factors = self.factors
            
            # Create summary plot showing total contribution uncertainty per factor
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
            
            # Plot 1: Total contribution variability by factor
            factor_totals = []
            factor_labels = []
            factor_colors = []
            
            for factor_idx in range(n_factors):
                if factor_idx in bootstrap_obj.bs_factor_contributions:
                    contributions = np.array(bootstrap_obj.bs_factor_contributions[factor_idx])
                    # Sum across all species for each bootstrap sample
                    total_contributions = np.sum(contributions, axis=1)
                    factor_totals.append(total_contributions)
                    
                    # Color H2S factor red, others blue
                    if self.color_manager and self.color_manager.is_h2s_factor(factor_idx):
                        factor_colors.append('#d62728')  # Red for H2S factor
                        factor_labels.append(f"Factor {factor_idx + 1} (H2S)")
                    else:
                        factor_colors.append('#1f77b4')  # Blue for other factors
                        factor_labels.append(f"Factor {factor_idx + 1}")
            
            if factor_totals:
                # Box plot of total contributions
                bp1 = ax1.boxplot(factor_totals, positions=range(len(factor_totals)), patch_artist=True)
                
                for patch, color in zip(bp1['boxes'], factor_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                
                ax1.set_xticks(range(len(factor_labels)))
                ax1.set_xticklabels(factor_labels, rotation=0, fontsize=12)
                ax1.set_ylabel('Total Contribution', fontsize=14)
                ax1.set_title('Factor Contribution Uncertainty (Total across all species)', fontsize=14)
                ax1.grid(True, alpha=0.3)
            
            # Plot 2: Species-wise contribution uncertainty for H2S factor (if available)
            if self.color_manager and self.color_manager.h2s_factor_idx is not None:
                h2s_idx = self.color_manager.h2s_factor_idx
                if h2s_idx in bootstrap_obj.bs_factor_contributions:
                    h2s_contributions = np.array(bootstrap_obj.bs_factor_contributions[h2s_idx])
                    
                    # Box plot for each species contribution
                    species_box_data = [h2s_contributions[:, i] for i in range(n_species)]
                    bp2 = ax2.boxplot(species_box_data, positions=range(n_species), patch_artist=True)
                    
                    for patch in bp2['boxes']:
                        patch.set_facecolor('#d62728')  # Red for H2S factor
                        patch.set_alpha(0.6)
                    
                    # Base model contributions for comparison
                    base_h2s_contribs = np.sum(bootstrap_obj.base_W[:, h2s_idx].reshape(-1, 1) @ 
                                              bootstrap_obj.base_H[h2s_idx].reshape(1, -1), axis=0)
                    ax2.scatter(range(n_species), base_h2s_contribs, 
                              color='red', s=40, zorder=5, label='Base Model')
                    
                    ax2.set_xticks(range(n_species))
                    ax2.set_xticklabels(species_names, rotation=45, ha='right', fontsize=12)
                    ax2.set_ylabel('Species Contribution', fontsize=14)
                    ax2.set_title(f'Factor {h2s_idx + 1} (H2S-dominant) - Species-wise Contribution Uncertainty', fontsize=14)
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(fontsize=12)
            
            plt.suptitle('Factor Contribution Uncertainty from Bootstrap Analysis', fontsize=16)
            plt.tight_layout()
            
            plot_path = output_dir / f"{self.filename_prefix}_bootstrap_contribution_uncertainty.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"   Error creating contribution uncertainty plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_bootstrap_summary_plot(self, bootstrap_obj, output_dir):
        """
        Create bootstrap summary statistics plot.
        
        Args:
            bootstrap_obj (Bootstrap): ESAT Bootstrap object
            output_dir (Path): Output directory for plots
            
        Returns:
            Path: Path to created plot file
        """
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left panel: Bootstrap parameters and summary
            summary_text = f"""Bootstrap Error Estimation Summary
            
Bootstrap Parameters:
• Samples: {self.bootstrap_n}
• Block Size: {self.bootstrap_block_size or 'Auto-estimated'}
• Threshold: {self.bootstrap_threshold}
• Parallel: {self.bootstrap_parallel}
• CPUs: {self.bootstrap_cpus or 'All available'}
• Keep H: {self.bootstrap_keep_h}
• Reuse Seed: {self.bootstrap_reuse_seed}

Base Model:
• Factors: {self.factors}
• Models: {self.models}
• Species: {len(self.concentration_data.columns)}
• Samples: {len(self.concentration_data)}
            """
            
            if hasattr(bootstrap_obj, 'q_results') and bootstrap_obj.q_results is not None:
                q_stats = bootstrap_obj.q_results['Q(robust)']
                summary_text += f"""

Q(robust) Statistics:
• Base Model: {bootstrap_obj.base_Q:.2f}
• Bootstrap Mean: {q_stats.mean():.2f}
• Bootstrap Std: {q_stats.std():.2f}
• Bootstrap Range: {q_stats.min():.2f} - {q_stats.max():.2f}
                """
            
            ax1.text(0.05, 0.95, summary_text, transform=ax1.transAxes, 
                   fontsize=13, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            ax1.set_xlim(0, 1)
            ax1.set_ylim(0, 1)
            ax1.axis('off')
            ax1.set_title('Bootstrap Configuration & Statistics', fontsize=16, pad=20)
            
            # Right panel: Factor mapping table visualization
            if hasattr(bootstrap_obj, 'mapping_df') and bootstrap_obj.mapping_df is not None:
                mapping_data = bootstrap_obj.mapping_df.copy()
                
                # Remove the 'Boot Factors' column for plotting (it's just row labels)
                if 'Boot Factors' in mapping_data.columns:
                    mapping_data = mapping_data.drop('Boot Factors', axis=1)
                
                # Create heatmap of mapping counts
                import seaborn as sns
                sns.heatmap(mapping_data.values, 
                           xticklabels=mapping_data.columns, 
                           yticklabels=[f'Boot F{i+1}' for i in range(len(mapping_data))],
                           annot=True, fmt='.1f', cmap='Blues', ax=ax2,
                           annot_kws={'fontsize': 12},
                           cbar_kws={'label': 'Bootstrap Mapping Count'})
                
                ax2.set_title('Factor Mapping: Bootstrap → Base Factors', fontsize=16, pad=20)
                ax2.set_xlabel('Base Model Factors', fontsize=14)
                ax2.set_ylabel('Bootstrap Factors', fontsize=14)
                ax2.tick_params(axis='both', which='major', labelsize=12)
                
                # Add interpretation text
                interpretation = "Higher counts indicate more consistent\nfactor identification across bootstrap samples"
                ax2.text(0.02, 0.98, interpretation, transform=ax2.transAxes, 
                        fontsize=11, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
            else:
                ax2.text(0.5, 0.5, 'Factor mapping data not available', 
                        ha='center', va='center', transform=ax2.transAxes, fontsize=12)
                ax2.set_title('Factor Mapping Unavailable', fontsize=14)
                ax2.axis('off')
            
            plt.suptitle('Bootstrap Error Estimation Summary', fontsize=18, y=0.95)
            plt.tight_layout()
            
            plot_path = output_dir / f"{self.filename_prefix}_bootstrap_summary.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return plot_path
            
        except Exception as e:
            print(f"   Error creating bootstrap summary plot: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_pca_analysis(self):
        """
        Run Principal Component Analysis on the concentration data.
        This provides a comparison to PMF using traditional variance-based decomposition.
        """
        print("[ANALYSIS] Starting PCA analysis...")
        
        # Load processed concentration data (same as used for PMF)
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        if not conc_file.exists():
            print("[ERROR] No concentration data found. Run PMF analysis first.")
            return False
        
        print("[DATA] Loading concentration data for PCA...")
        conc_df = pd.read_csv(conc_file, index_col=0)
        
        # Remove any remaining NaN values
        conc_clean = conc_df.dropna()
        if len(conc_clean) != len(conc_df):
            print(f"   Removed {len(conc_df) - len(conc_clean)} rows with missing values")
        
        print(f"[DATA] PCA data matrix: {conc_clean.shape}")
        print(f"[INFO] Species: {', '.join(conc_clean.columns)}")
        
        # Step 1: Data Standardization (CRITICAL for PCA)
        print("[INIT] Standardizing data (Z-score transformation)...")
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
        print(f"[NUMBERS] Using {n_components} components (matching PMF factors)")
        
        # Step 3: Perform PCA
        print("[SYMBOL]️ Performing PCA...")
        self.pca_model = PCA(n_components=n_components, random_state=self.seed)
        pca_scores = self.pca_model.fit_transform(X_scaled)
        pca_loadings = self.pca_model.components_.T  # Transpose to get species × components
        
        # Store explained variance
        self.pca_explained_variance = self.pca_model.explained_variance_ratio_
        
        print(f"[DATA] PCA Results:")
        print(f"   Components shape: {pca_loadings.shape}")
        print(f"   Scores shape: {pca_scores.shape}")
        print(f"   Total variance explained: {np.sum(self.pca_explained_variance):.1%}")
        for i, var in enumerate(self.pca_explained_variance):
            print(f"     PC{i+1}: {var:.1%}")
        
        # Step 4: Varimax Rotation for interpretability
        print("[PROC] Applying Varimax rotation for interpretability...")
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
        
        print(f"[SAVE] Saved PCA loadings: {pca_loadings_file}")
        print(f"[SAVE] Saved PCA scores: {pca_scores_file}")
        
        return True
    
    def _create_pca_comparison_plots(self, dashboard_dir, plot_files):
        """
        Create comparative plots between PMF and PCA results.
        This includes side-by-side profiles, correlation analysis, and method comparison.
        """
        print("   [A] Creating PCA vs PMF comparison plots...")
        
        if not hasattr(self, 'pca_loadings') or self.pca_loadings is None:
            print("   [WARN] No PCA results found - skipping comparison plots")
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
        print(f"   [OK] Saved: pca_pmf_comparison.png")
        
        # Additional detailed comparison plot
        self._create_detailed_profile_comparison(dashboard_dir, plot_files, F_profiles)
    
    def _create_detailed_profile_comparison(self, dashboard_dir, plot_files, F_profiles):
        """
        Create detailed side-by-side profile comparison for all factors/components.
        """
        print("   [SEARCH] Creating detailed profile comparison plots...")
        
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
        print(f"   [OK] Saved: detailed_profile_comparison.png")
    
    def _create_pca_loadings_plot(self, dashboard_dir, plot_files):
        """
        Create dedicated PCA loadings plots at the end of analysis when PCA is run.
        """
        print("   [DATA] Creating PCA loadings plots...")
        
        if not hasattr(self, 'pca_loadings') or self.pca_loadings is None:
            print("   [WARN] No PCA results found - skipping PCA loadings plots")
            return
        
        # Calculate optimal subplot layout for PCA components
        n_components = self.factors
        if n_components <= 4:
            nrows, ncols = 2, 2
        elif n_components <= 6:
            nrows, ncols = 2, 3
        elif n_components <= 9:
            nrows, ncols = 3, 3
        elif n_components <= 12:
            nrows, ncols = 3, 4
        else:
            nrows, ncols = 4, 4
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
        station_display_name = self._get_station_display_name()
        fig.suptitle(f'{station_display_name} PCA Component Loadings (after Varimax rotation)', 
                    fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if n_components == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Plot all PCA components
        for i in range(n_components):
            ax = axes[i]
            loadings = self.pca_loadings[:, i]
            
            # Use consistent component color (same color for positive and negative)
            component_color = self.color_manager.get_factor_color(i)
            
            bars = ax.bar(range(len(self.species_names)), loadings, alpha=0.8, color=component_color)
            
            ax.set_title(f'PC{i+1} (Explains {self.pca_explained_variance[i]:.1%} variance)', 
                        fontweight='bold', fontsize=12)
            ax.set_xlabel('Species', fontsize=10)
            ax.set_ylabel('Loading', fontsize=10)
            ax.set_xticks(range(len(self.species_names)))
            ax.set_xticklabels(self.species_names, rotation=45, ha='right', fontsize=8)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)  # Zero reference line
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        total_subplots = nrows * ncols
        for i in range(n_components, total_subplots):
            if i < len(axes):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_path = dashboard_dir / f"{self.filename_prefix}_pca_loadings.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_path)
        print(f"   [OK] Saved: pca_loadings.png")
    
    def _generate_factor_structure_summary(self):
        """Generate factor structure diagnostics for model validation.
        
        Creates a summary file with sparsity metrics, W/H correlations,
        and other structural diagnostics useful for comparing PMF runs.
        """
        if not self.best_model:
            return
        
        print("[DATA] Generating factor structure diagnostics...")
        
        # Get factor profiles (H) and contributions (W) from ESAT model
        # H shape: (n_factors, n_species), W shape: (n_samples, n_factors)
        F_profiles = getattr(self.best_model, 'H', None)
        G_contributions = getattr(self.best_model, 'W', None)
        
        if F_profiles is None or G_contributions is None:
            print("[WARN] Factor structure diagnostics unavailable: model lacks H/W matrices")
            return
        
        # Calculate sparsity metrics
        def calculate_sparsity(matrix, threshold=0.01):
            """Calculate sparsity as fraction of elements below threshold."""
            return float(np.mean(np.abs(matrix) < threshold))
        
        def calculate_gini_coefficient(array):
            """Calculate Gini coefficient as sparsity measure."""
            arr = np.abs(np.asarray(array)).flatten()
            arr = arr[arr > 0]
            if arr.size == 0:
                return 0.0
            arr.sort()
            n = arr.size
            index = np.arange(1, n + 1)
            return float((2 * np.sum(index * arr)) / (n * np.sum(arr)) - (n + 1) / n)
        
        # W (contributions) sparsity analysis
        w_sparsity_01 = calculate_sparsity(G_contributions, 0.01)
        w_sparsity_05 = calculate_sparsity(G_contributions, 0.05)
        w_gini = float(np.mean([calculate_gini_coefficient(G_contributions[:, i]) for i in range(self.factors)]))
        
        # H (profiles) sparsity analysis
        h_sparsity_01 = calculate_sparsity(F_profiles, 0.01)
        h_sparsity_05 = calculate_sparsity(F_profiles, 0.05)
        h_gini = float(np.mean([calculate_gini_coefficient(F_profiles[i, :]) for i in range(self.factors)]))
        
        # W correlation matrix (factor time series correlations)
        w_corr_matrix = np.corrcoef(G_contributions.T)
        w_off = np.abs(w_corr_matrix[np.triu_indices_from(w_corr_matrix, k=1)])
        w_max_off_diagonal = float(np.max(w_off)) if w_off.size else 0.0
        w_mean_off_diagonal = float(np.mean(w_off)) if w_off.size else 0.0
        
        # H correlation matrix (factor profile correlations)
        h_corr_matrix = np.corrcoef(F_profiles)
        h_off = np.abs(h_corr_matrix[np.triu_indices_from(h_corr_matrix, k=1)])
        h_max_off_diagonal = float(np.max(h_off)) if h_off.size else 0.0
        h_mean_off_diagonal = float(np.mean(h_off)) if h_off.size else 0.0
        
        # Individual factor diagnostics
        factor_diagnostics = []
        species_cols = list(self.concentration_data.columns)
        for i in range(self.factors):
            row = F_profiles[i, :]
            factor_diag = {
                'factor_id': i + 1,
                'w_variance': float(np.var(G_contributions[:, i])),
                'w_mean': float(np.mean(G_contributions[:, i])),
                'w_sparsity_01': calculate_sparsity(G_contributions[:, i], 0.01),
                'h_sparsity_01': calculate_sparsity(row, 0.01),
                'h_dominant_species': species_cols[int(np.argmax(row))] if len(species_cols) == row.shape[0] else str(int(np.argmax(row))),
                'h_max_loading': float(np.max(row)),
                'h_gini': calculate_gini_coefficient(row)
            }
            factor_diagnostics.append(factor_diag)
        
        # Save structured diagnostics to file
        summary_file = self.output_dir / f"{self.filename_prefix}_factor_structure_summary.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Factor Structure Diagnostics Summary\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"=" * 50 + "\n\n")
            
            f.write(f"Model Information:\n")
            f.write(f"  Factors: {self.factors}\n")
            f.write(f"  Species: {len(self.concentration_data.columns)}\n")
            f.write(f"  Samples: {len(self.concentration_data)}\n")
            f.write(f"  Q(true): {self.best_model.Qtrue:.2f}\n")
            f.write(f"  Q(robust): {self.best_model.Qrobust:.2f}\n\n")
            
            f.write(f"Sparsity Metrics:\n")
            f.write(f"  W Matrix (Contributions):\n")
            f.write(f"    Sparsity (< 1%): {w_sparsity_01:.3f}\n")
            f.write(f"    Sparsity (< 5%): {w_sparsity_05:.3f}\n")
            f.write(f"    Mean Gini: {w_gini:.3f}\n")
            
            f.write(f"  H Matrix (Profiles):\n")
            f.write(f"    Sparsity (< 1%): {h_sparsity_01:.3f}\n")
            f.write(f"    Sparsity (< 5%): {h_sparsity_05:.3f}\n")
            f.write(f"    Mean Gini: {h_gini:.3f}\n\n")
            
            f.write(f"Correlation Analysis:\n")
            f.write(f"  W Correlations (Time Series):\n")
            f.write(f"    Max Off-Diagonal: {w_max_off_diagonal:.3f}\n")
            f.write(f"    Mean Off-Diagonal: {w_mean_off_diagonal:.3f}\n")
            
            f.write(f"  H Correlations (Profiles):\n")
            f.write(f"    Max Off-Diagonal: {h_max_off_diagonal:.3f}\n")
            f.write(f"    Mean Off-Diagonal: {h_mean_off_diagonal:.3f}\n\n")
            
            f.write(f"Individual Factor Diagnostics:\n")
            for factor in factor_diagnostics:
                f.write(f"  Factor {factor['factor_id']}:\n")
                f.write(f"    W Variance: {factor['w_variance']:.3f}\n")
                f.write(f"    W Mean: {factor['w_mean']:.3f}\n")
                f.write(f"    W Sparsity: {factor['w_sparsity_01']:.3f}\n")
                f.write(f"    H Sparsity: {factor['h_sparsity_01']:.3f}\n")
                f.write(f"    Dominant Species: {factor['h_dominant_species']}\n")
                f.write(f"    Max H Loading: {factor['h_max_loading']:.3f}\n")
                f.write(f"    H Gini: {factor['h_gini']:.3f}\n")
        
        print(f"   [SAVE] Factor structure summary: {summary_file.name}")
        
        # Store diagnostics for potential dashboard use
        self._factor_structure_diagnostics = {
            'w_corr_max_offdiag': w_max_off_diagonal,
            'w_corr_mean_offdiag': w_mean_off_diagonal,
            'h_corr_max_offdiag': h_max_off_diagonal,
            'h_corr_mean_offdiag': h_mean_off_diagonal
        }
    
    def _create_wind_analysis_plots(self, dashboard_dir, plot_files, G_contributions):
        """
        Create wind analysis plots showing how PMF factors vary with wind direction and speed.
        This is valuable for source apportionment as it can identify directional sources.
        """
        print("   [WIND] Creating wind analysis plots...")
        
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
            print("   [WARN] No wind data found in dataset - skipping wind analysis")
            return
        
        # Get the corresponding meteorological data for PMF time points
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        conc_data = pd.read_csv(conc_file, index_col=0)
        
        # Get datetime index for matching with original data
        try:
            datetime_index = pd.to_datetime(conc_data.index)
            has_datetime = True
        except:
            print("   [WARN] Unable to parse datetime for wind analysis")
            return
        
        # Ensure datetime information is available - handle both column and index cases
        datetime_series = None
        if 'datetime' in self.df.columns:
            # Case 1: datetime is a column
            if not pd.api.types.is_datetime64_any_dtype(self.df['datetime']):
                self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            datetime_series = self.df['datetime']
        elif hasattr(self.df.index, 'min') and pd.api.types.is_datetime64_any_dtype(self.df.index):
            # Case 2: datetime is in the index
            datetime_series = self.df.index.to_series(name='datetime')
        else:
            print("   [WARN] No datetime information found in dataset - skipping wind analysis")
            return
        
        # Match meteorological data with PMF analysis times
        wind_data = []
        valid_indices = []
        
        for i, dt in enumerate(datetime_index):
            # Find closest match in original data
            time_diff = np.abs((datetime_series - dt).dt.total_seconds())
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
            print("   [WARN] No matching wind data found for PMF time points")
            return
        
        wind_df = pd.DataFrame(wind_data)
        print(f"   [DATA] Found {len(wind_df)} matching wind/PMF data points")
        
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
                # Bin wind directions into 16 sectors (22.5 deg each)
                bins = np.arange(0, 361, 22.5)
                counts, bin_edges = np.histogram(wind_dirs, bins=bins)
                
                # Create bar chart representing wind rose
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                ax1.bar(bin_centers, counts, width=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.set_xlabel('Wind Direction ( deg)')
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
            print(f"   [INIT] Wind-correlated factors: {[(f+1, corr) for f, corr in factor_wind_correlations]}")
        
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
                ax1.set_xlabel('Wind Direction ( deg)')
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
        
        # Plot 3: Factor Data Points vs Wind Direction (0-360 deg) (top right)
        ax3 = fig.add_subplot(gs[0, 2:])
        if wind_dir_col and not wind_df['wind_dir'].isna().all():
            valid_mask = ~wind_df['wind_dir'].isna()
            wd = wind_df.loc[valid_mask, 'wind_dir'].values
            
            # Plot factors in order with H2S factor last (for top layer visibility)
            factor_plot_order = self.color_manager.get_factor_plot_order()
            
            for f in factor_plot_order:
                fc = G_wind[valid_mask, f]
                # Make H2S factor more prominent
                is_h2s = self.color_manager.is_h2s_factor(f)
                alpha_val = 0.8 if is_h2s else 0.6
                marker_size = 25 if is_h2s else 20
                
                ax3.scatter(wd, fc, alpha=alpha_val, s=marker_size, color=colors[f], 
                           label=f'Factor {f+1}', edgecolors='black', linewidth=0.5 if is_h2s else 0)
            
            ax3.set_xlabel('Wind Direction ( deg)')
            ax3.set_ylabel('Factor Contribution')
            ax3.set_title('Factor Data Points vs Wind Direction (0-360 deg)')
            ax3.set_xlim(0, 360)
            ax3.set_xticks(np.arange(0, 361, 45))
            ax3.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'])
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # Create polar plots for ALL factors (larger size) with consistent color scale
        if wind_dir_col and not wind_df['wind_dir'].isna().all():
            # Calculate global min/max across all factors for consistent color scaling
            global_min = float('inf')
            global_max = float('-inf')
            
            # First pass: find global min/max across all factors
            for f in range(n_factors):
                valid_mask = ~(wind_df['wind_dir'].isna() | np.isnan(G_wind[:, f]))
                if np.sum(valid_mask) > 5:
                    fc = G_wind[valid_mask, f]
                    global_min = min(global_min, np.min(fc))
                    global_max = max(global_max, np.max(fc))
            
            # Ensure valid range found
            if global_min == float('inf') or global_max == float('-inf'):
                global_min, global_max = 0, 1  # Fallback range
            
            print(f"   [UI] Using consistent color scale for polar plots: {global_min:.3f} to {global_max:.3f}")
            
            # Second pass: create polar plots with consistent scale
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
                    # Use global scale for consistent color comparison across all factors
                    scatter = ax_polar.scatter(wd_rad, fc, c=fc, cmap='viridis', 
                                             alpha=0.7, s=40, vmin=global_min, vmax=global_max)
                    
                    # Add colorbar to show contribution scale
                    cbar = plt.colorbar(scatter, ax=ax_polar, shrink=0.6, pad=0.1)
                    cbar.set_label('Factor Contribution\n(Global Scale)', fontsize=10)
                    
                    # Get correlation for title
                    with SuppressOutput():
                        factor_corr = float(next((corr for idx, corr in factor_wind_correlations if idx == f), 0))
                    
                    ax_polar.set_title(f'Factor {f + 1} vs Wind Direction\n(|r|={factor_corr:.2f})', 
                                      fontweight='bold', pad=20, fontsize=12)
                    ax_polar.set_theta_zero_location('N')
                    ax_polar.set_theta_direction(-1)
                    ax_polar.grid(True, alpha=0.3)
                    
                    # Set consistent radial scale across all polar plots for better comparison
                    ax_polar.set_ylim(0, global_max * 1.05)  # Slight margin above max
        
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
            
            # Create stacked bar chart (H2S factor will be on top as it's plotted last)
            x_pos = np.arange(len(sectors))
            bottom = np.zeros(len(sectors))
            factor_plot_order = self.color_manager.get_factor_plot_order()
            
            for f in factor_plot_order:
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
            
            for f in factor_plot_order:
                fc = G_wind[valid_mask, f]
                # Make H2S factor more prominent
                is_h2s = self.color_manager.is_h2s_factor(f)
                alpha_val = 0.8 if is_h2s else 0.6
                marker_size = 35 if is_h2s else 30
                
                ax6.scatter(ws, fc, alpha=alpha_val, s=marker_size, color=colors[f], 
                           label=f'Factor {f+1}', edgecolors='black', linewidth=0.5 if is_h2s else 0)
                
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
                
                for f in factor_plot_order:
                    # Use original factor index for offset calculation to maintain spacing
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
        print(f"   [OK] Saved: wind_analysis.png")
        
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
                f.write(f"Wind direction range: {wind_df['wind_dir'].min():.1f} deg - {wind_df['wind_dir'].max():.1f} deg\n")
                f.write(f"Most frequent wind direction: {wind_df['wind_dir'].mode().iloc[0]:.1f} deg\n")
            
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
        
        print(f"   [FILE] Wind summary statistics: {summary_path}")
    
    def _create_temperature_analysis_plots(self, dashboard_dir, plot_files, G_contributions):
        """
        Create temperature analysis plots showing how PMF factors vary with temperature.
        This can help identify temperature-dependent sources (e.g., heating, biogenic emissions).
        """
        print("   [TEMP] Creating temperature analysis plots...")
        
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
            print("   [WARN] No temperature data found in dataset - skipping temperature analysis")
            return
        
        print(f"   [DATA] Found temperature columns: {temp_columns}")
    
        # Get the corresponding temperature data for PMF time points
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        conc_data = pd.read_csv(conc_file, index_col=0)
        
        # Get datetime index for matching with original data
        try:
            datetime_index = pd.to_datetime(conc_data.index)
            has_datetime = True
        except:
            print("   [WARN] Unable to parse datetime for temperature analysis")
            return
        
        # Ensure datetime information is available - handle both column and index cases
        datetime_series = None
        if 'datetime' in self.df.columns:
            # Case 1: datetime is a column
            if not pd.api.types.is_datetime64_any_dtype(self.df['datetime']):
                self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            datetime_series = self.df['datetime']
        elif hasattr(self.df.index, 'min') and pd.api.types.is_datetime64_any_dtype(self.df.index):
            # Case 2: datetime is in the index
            datetime_series = self.df.index.to_series(name='datetime')
        else:
            print("   [WARN] No datetime information found in dataset - skipping temperature analysis")
            return
        
        # Match temperature data with PMF analysis times
        temp_data = []
        valid_indices = []
        
        for i, dt in enumerate(datetime_index):
            # Find closest match in original data
            time_diff = np.abs((datetime_series - dt).dt.total_seconds())
            closest_idx = time_diff.idxmin()
            
            # Only include if within reasonable time tolerance (e.g., 1 hour)
            if time_diff.loc[closest_idx] <= 3600:  # 1 hour in seconds
                temp_dict = {'pmf_index': i}
                for temp_col in temp_columns:
                    temp_dict[temp_col] = self.df.loc[closest_idx, temp_col]
                temp_data.append(temp_dict)
                valid_indices.append(i)
        
        if len(temp_data) == 0:
            print("   [WARN] No matching temperature data found for PMF time points")
            return
        
        temp_df = pd.DataFrame(temp_data)
        print(f"   [DATA] Found {len(temp_df)} matching temperature/PMF data points")
        
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
            print(f"   [WARN] Error converting temperature data to numeric: {e}")
            return
        
        if len(temp_values) == 0:
            print(f"   [WARN] No valid temperature data in {primary_temp_col}")
            return
        
        # Plot 1: Temperature distribution (top left)
        ax1 = axes[0, 0]
        ax1.hist(temp_values, bins=30, alpha=0.7, color='orange', edgecolor='black')
        ax1.set_xlabel(f'Temperature ({self.units.get(primary_temp_col, " degC")})')
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
        
        ax2.set_xlabel(f'Temperature ({self.units.get(primary_temp_col, " degC")})')
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
            
            ax3.set_xlabel(f'Temperature Bins ({self.units.get(primary_temp_col, " degC")})')
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
                ax5_temp.set_ylabel(f'Temperature ({self.units.get(primary_temp_col, " degC")})', color='red')
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
        print(f"   [OK] Saved: temperature_analysis.png")
        
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
        
        print(f"   [FILE] Temperature summary statistics: {summary_path}")
    
    def _create_pressure_analysis_plots(self, dashboard_dir, plot_files, G_contributions):
        """
        Create pressure analysis plots showing how PMF factors vary with atmospheric pressure.
        This can help identify pressure-dependent sources and meteorological influences.
        """
        print("   [PRESSURE] Creating pressure analysis plots...")
        
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
            print("   [WARN] No pressure data found in dataset - skipping pressure analysis")
            return
        
        print(f"   [DATA] Found pressure columns: {pressure_columns}")
        
        # Get the corresponding pressure data for PMF time points
        conc_file = self.output_dir / f"{self.filename_prefix}_concentrations.csv"
        conc_data = pd.read_csv(conc_file, index_col=0)
        
        # Get datetime index for matching with original data
        try:
            datetime_index = pd.to_datetime(conc_data.index)
            has_datetime = True
        except:
            print("   [WARN] Unable to parse datetime for pressure analysis")
            return
        
        # Ensure datetime information is available - handle both column and index cases
        datetime_series = None
        if 'datetime' in self.df.columns:
            # Case 1: datetime is a column
            if not pd.api.types.is_datetime64_any_dtype(self.df['datetime']):
                self.df['datetime'] = pd.to_datetime(self.df['datetime'])
            datetime_series = self.df['datetime']
        elif hasattr(self.df.index, 'min') and pd.api.types.is_datetime64_any_dtype(self.df.index):
            # Case 2: datetime is in the index
            datetime_series = self.df.index.to_series(name='datetime')
        else:
            print("   [WARN] No datetime information found in dataset - skipping pressure analysis")
            return
        
        # Match pressure data with PMF analysis times
        pressure_data = []
        valid_indices = []
        
        for i, dt in enumerate(datetime_index):
            # Find closest match in original data
            time_diff = np.abs((datetime_series - dt).dt.total_seconds())
            closest_idx = time_diff.idxmin()
            
            # Only include if within reasonable time tolerance (e.g., 1 hour)
            if time_diff.loc[closest_idx] <= 3600:  # 1 hour in seconds
                pressure_dict = {'pmf_index': i}
                for pressure_col in pressure_columns:
                    pressure_dict[pressure_col] = self.df.loc[closest_idx, pressure_col]
                pressure_data.append(pressure_dict)
                valid_indices.append(i)
        
        if len(pressure_data) == 0:
            print("   [WARN] No matching pressure data found for PMF time points")
            return
        
        pressure_df = pd.DataFrame(pressure_data)
        print(f"   [DATA] Found {len(pressure_df)} matching pressure/PMF data points")
        
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
            print(f"   [WARN] Error converting pressure data to numeric: {e}")
            return
        
        if len(pressure_values) == 0:
            print(f"   [WARN] No valid pressure data in {primary_pressure_col}")
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
        print(f"   [OK] Saved: pressure_analysis.png")
        
        # Add pressure derivative analysis plots
        self._create_pressure_derivative_plots(dashboard_dir, plot_files, G_contributions)
        
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
        
        print(f"   [FILE] Pressure summary statistics: {summary_path}")
    
    def _create_pressure_derivative_plots(self, dashboard_dir, plot_files, G_contributions):
        """
        Create pressure derivative (dP/dt) analysis plots showing barometric pumping effects.
        """
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            
            # ensure Pressure exists
            if 'Pressure' not in self.df.columns:
                print("[WARN] Pressure column not found; skipping dP/dt plots")
                return
            
            # Get the datetime index from the concentration data (which matches PMF analysis)
            if not hasattr(self, 'concentration_data'):
                print("[WARN] No concentration data available; skipping dP/dt plots")
                return
            
            # SAFE APPROACH: Create independent pressure analysis without affecting PMF data
            # Work with a completely separate copy to avoid affecting PMF analysis
            pressure_df = self.df.copy()
            
            # Handle both datetime column and index cases
            if 'datetime' in pressure_df.columns:
                pressure_df['datetime'] = pd.to_datetime(pressure_df['datetime'])
            elif hasattr(pressure_df.index, 'min') and pd.api.types.is_datetime64_any_dtype(pressure_df.index):
                # Datetime is in index - add as column for processing
                pressure_df['datetime'] = pressure_df.index
            else:
                print("[WARN] No datetime information available for pressure derivative analysis")
                return
            
            # Create pressure time series from current filtered data (original sparse measurements)
            pressure_indexed = pressure_df.set_index('datetime')['Pressure'].sort_index()
            
            # FIXED: Use actual concentration data timestamps instead of sparse regular grid
            # This preserves pressure variation by matching actual measurement times
            
            # Get the datetime information properly
            if hasattr(self.df.index, 'min') and pd.api.types.is_datetime64_any_dtype(self.df.index):
                # Case: datetime is in index - use concentration data index directly
                concentration_indices = self.concentration_data.index
                matching_indices = self.df.index.intersection(concentration_indices)
                idx = pd.to_datetime(matching_indices).sort_values()
            elif 'datetime' in self.df.columns:
                # Case: datetime is a column
                concentration_rows = self.df[self.df.index.isin(self.concentration_data.index)]
                concentration_rows = concentration_rows.sort_values('datetime')
                idx = pd.to_datetime(concentration_rows['datetime'].values).sort_values()
            else:
                # Fallback: use concentration data index as is
                idx = pd.to_datetime(self.concentration_data.index).sort_values()
            # Remove duplicates and ensure unique timestamps
            idx = pd.Index(idx).drop_duplicates().sort_values()
            
            # Debug datetime handling
            print(f"   [DEBUG] Created datetime index: {len(idx)} timestamps from {idx.min()} to {idx.max()}")
            
            print(f"   [DEBUG] Pressure analysis: {pressure_indexed.notna().sum()} raw points")
            print(f"   [DEBUG] Using {len(idx)} concentration data timestamps for pressure alignment")
            
            # FIXED APPROACH: Calculate derivatives on original sparse measurements FIRST
            # Get only the non-NaN pressure data points (the actual 15-minute measurements)
            p_valid = pressure_indexed.dropna()  # Original sparse measurements
            
            if len(p_valid) < 3:
                print(f"   [WARN] Insufficient pressure data points ({len(p_valid)}) for derivative calculation")
                return
            
            # Apply zero-phase low-pass filtering to remove noise before derivative calculation
            try:
                from scipy import signal
                
                # Design Butterworth low-pass filter (cutoff = 1/(6 hours) for smoother signal)
                # 15-min data -> Nyquist = 1/(30 min) = 48 cycles/day
                # Cutoff at 4 cycles/day (6-hour period) for smoother weather patterns
                fs = 1.0 / (15 * 60)  # Sampling frequency in Hz (15-minute intervals)
                cutoff = 1.0 / (6 * 3600)  # Cutoff frequency: 1/(6 hours) in Hz
                nyquist = fs / 2
                normalized_cutoff = cutoff / nyquist
                
                # Create 4th order Butterworth filter
                b, a = signal.butter(4, normalized_cutoff, btype='low', analog=False)
                
                # Apply zero-phase filtering (forward + backward pass, no phase lag)
                p_filtered = signal.filtfilt(b, a, p_valid.values)
                p_valid_filtered = pd.Series(p_filtered, index=p_valid.index)
                
                print(f"   [DEBUG] Applied zero-phase low-pass filter (cutoff: 6-hour period)")
                print(f"   [DEBUG] Pressure range after filtering: {p_valid_filtered.min():.2f} - {p_valid_filtered.max():.2f} hPa")
                
            except ImportError:
                print(f"   [WARN] scipy not available - using raw pressure data (will be noisy)")
                p_valid_filtered = p_valid
            except Exception as e:
                print(f"   [WARN] Filtering failed: {e} - using raw pressure data")
                p_valid_filtered = p_valid
            
            # ENHANCED: Calculate 6-hour window derivatives for ALL PMF timestamps to maximize data usage
            # This approach uses all available pressure measurements around each PMF timestamp
            
            window_hours = 6.0
            window_timedelta = pd.Timedelta(hours=window_hours)
            
            print(f"   [DEBUG] Calculating derivatives using {window_hours}-hour window for ALL PMF timestamps")
            print(f"   [DEBUG] This maximizes use of all {len(p_valid_filtered)} available pressure measurements")
            
            # Create derivative series for ALL concentration timestamps (not just pressure timestamps)
            dpdt_1h = pd.Series(index=idx, dtype=float)
            derivatives_calculated = 0
            insufficient_data_count = 0
            
            # Calculate derivatives at each PMF analysis timestamp
            for timestamp in idx:
                # Find all available pressure values within ±3 hours of current timestamp
                start_time = timestamp - window_timedelta / 2
                end_time = timestamp + window_timedelta / 2
                
                # Get ALL pressure data within the window (using original unfiltered data for maximum coverage)
                window_data_raw = pressure_indexed[(pressure_indexed.index >= start_time) & 
                                                  (pressure_indexed.index <= end_time) &
                                                  (pressure_indexed.notna())]
                
                # Also get filtered data in the window if available
                window_data_filtered = p_valid_filtered[(p_valid_filtered.index >= start_time) & 
                                                       (p_valid_filtered.index <= end_time)]
                
                # Use filtered data if we have enough, otherwise fall back to raw data
                if len(window_data_filtered) >= 3:
                    window_data = window_data_filtered
                    data_source = "filtered"
                elif len(window_data_raw) >= 3:
                    window_data = window_data_raw
                    data_source = "raw"
                else:
                    # Try expanding the window if we don't have enough data
                    extended_window = window_timedelta * 1.5  # Extend to 9 hours
                    start_extended = timestamp - extended_window / 2
                    end_extended = timestamp + extended_window / 2
                    
                    window_data_extended = pressure_indexed[(pressure_indexed.index >= start_extended) & 
                                                           (pressure_indexed.index <= end_extended) &
                                                           (pressure_indexed.notna())]
                    
                    if len(window_data_extended) >= 3:
                        window_data = window_data_extended
                        data_source = "extended-9h"
                    else:
                        insufficient_data_count += 1
                        continue
                
                # Calculate weighted linear regression slope over the window
                # Weight points by their temporal distance from the center timestamp
                times_sec = np.array([(t - timestamp).total_seconds() for t in window_data.index])
                pressures = window_data.values
                
                # Apply distance-based weighting (closer points get higher weight)
                max_distance = np.max(np.abs(times_sec)) if len(times_sec) > 1 else 1
                if max_distance > 0:
                    weights = 1.0 - (np.abs(times_sec) / max_distance) ** 0.5  # Sqrt weighting
                    weights = np.maximum(weights, 0.1)  # Minimum weight of 0.1
                else:
                    weights = np.ones(len(times_sec))
                
                # Weighted linear regression
                if len(times_sec) >= 3:  # Need at least 3 points
                    # Compute weighted means
                    w_sum = np.sum(weights)
                    mean_time = np.sum(weights * times_sec) / w_sum
                    mean_pressure = np.sum(weights * pressures) / w_sum
                    
                    # Compute weighted slope
                    numerator = np.sum(weights * (times_sec - mean_time) * (pressures - mean_pressure))
                    denominator = np.sum(weights * (times_sec - mean_time) ** 2)
                    
                    if denominator > 0:
                        slope_per_second = numerator / denominator
                        slope_per_hour = slope_per_second * 3600.0  # Convert to hPa/hr
                        dpdt_1h.loc[timestamp] = slope_per_hour
                        derivatives_calculated += 1
            
            print(f"   [DEBUG] Calculated {derivatives_calculated} derivatives for {len(idx)} PMF timestamps")
            print(f"   [DEBUG] Insufficient data for {insufficient_data_count} timestamps")
            
            # Remove any remaining NaN values
            dpdt_valid_count = dpdt_1h.notna().sum()
            if dpdt_valid_count > 0:
                print(f"   [DEBUG] Raw dP/dt range: {dpdt_1h.min():.3f} to {dpdt_1h.max():.3f} hPa/hr")
            else:
                print(f"   [DEBUG] No valid derivatives calculated")
            
            # Apply additional zero-phase low-pass filtering to the derivative signal itself
            # Use lower cutoff frequency for smoother barometric analysis
            dpdt_1h_filtered = None
            if dpdt_valid_count > 5:  # Need sufficient points for filtering
                try:
                    from scipy import signal
                    
                    # Design even lower cutoff filter for derivative smoothing
                    # Use 48-hour period cutoff for very smooth barometric analysis
                    derivative_cutoff = 1.0 / (48 * 3600)  # 1/(48 hours) in Hz
                    
                    # Estimate sampling frequency from derivative timestamps
                    valid_dpdt = dpdt_1h.dropna()
                    if len(valid_dpdt) >= 3:
                        time_diffs_dt = np.diff(valid_dpdt.index)
                        # Handle both pandas.Timedelta and numpy.timedelta64 objects
                        interval_seconds = []
                        for td in time_diffs_dt:
                            if hasattr(td, 'total_seconds'):
                                interval_seconds.append(td.total_seconds())
                            else:
                                # Handle numpy.timedelta64 by converting to seconds
                                interval_seconds.append(td / np.timedelta64(1, 's'))
                        median_interval = np.median(interval_seconds)
                        fs_derivative = 1.0 / median_interval  # Sampling frequency in Hz
                        nyquist_derivative = fs_derivative / 2
                        normalized_cutoff_derivative = derivative_cutoff / nyquist_derivative
                        
                        print(f"   [DEBUG] Derivative filtering: {len(valid_dpdt)} points, median interval: {median_interval/3600:.2f}h")
                        print(f"   [DEBUG] Applying 48-hour cutoff filter to derivative signal")
                        
                        # Only filter if we have reasonable parameters
                        if 0 < normalized_cutoff_derivative < 0.45:  # Nyquist limit with safety margin
                            # Create 4th order Butterworth filter for derivatives
                            b_deriv, a_deriv = signal.butter(4, normalized_cutoff_derivative, btype='low', analog=False)
                            
                            # Apply zero-phase filtering to derivative values
                            dpdt_filtered_values = signal.filtfilt(b_deriv, a_deriv, valid_dpdt.values)
                            
                            # Create filtered derivative series
                            dpdt_1h_filtered = pd.Series(index=idx, dtype=float)
                            dpdt_1h_filtered.loc[valid_dpdt.index] = dpdt_filtered_values
                            
                            # Interpolate to full PMF timeline
                            dpdt_1h_filtered = dpdt_1h_filtered.reindex(idx).interpolate(method='time', limit_direction='both')
                            if dpdt_1h_filtered.isna().any():
                                dpdt_1h_filtered = dpdt_1h_filtered.ffill().bfill()
                            
                            print(f"   [DEBUG] Filtered dP/dt range: {dpdt_1h_filtered.min():.3f} to {dpdt_1h_filtered.max():.3f} hPa/hr")
                        else:
                            print(f"   [DEBUG] Normalized cutoff {normalized_cutoff_derivative:.3f} outside valid range - skipping derivative filtering")
                    else:
                        print(f"   [DEBUG] Insufficient points for derivative filtering")
                        
                except Exception as e:
                    print(f"   [DEBUG] Derivative filtering failed: {e} - using unfiltered derivatives")
            else:
                print(f"   [DEBUG] Insufficient derivatives ({dpdt_valid_count}) for additional filtering")
            
            # Calculate smoother hourly derivative for comparison
            try:
                if len(p_valid_filtered) >= 3:
                    # Resample filtered pressure to hourly, then calculate derivatives
                    p_hourly = p_valid_filtered.resample('1H').mean().interpolate(limit_direction='both')
                    if len(p_hourly) >= 2:
                        # Simple hourly derivative
                        slope3_hourly = p_hourly.diff() / 1.0  # 1 hour intervals
                        slope3 = slope3_hourly.reindex(idx).interpolate(method='time', limit_direction='both')
                        if slope3.isna().any():
                            slope3 = slope3.ffill().bfill()
                    else:
                        slope3 = None
                else:
                    slope3 = None
            except Exception:
                slope3 = None
            
            # Create interpolated pressure for visualization (both raw and filtered)
            p_on_idx_raw = pressure_indexed.reindex(idx).interpolate(method='time', limit_direction='both')
            
            if len(p_valid_filtered) > 0:
                # Create filtered pressure series for interpolation
                pressure_indexed_filtered = pressure_indexed.copy()
                pressure_indexed_filtered.loc[p_valid_filtered.index] = p_valid_filtered
                p_on_idx_filtered = pressure_indexed_filtered.reindex(idx).interpolate(method='time', limit_direction='both')
                p_on_idx = p_on_idx_filtered  # Use filtered for derivative calculations
                print(f"   [DEBUG] Prepared both raw and filtered pressure data for visualization")
            else:
                # Fallback to original data if filtering failed
                p_on_idx_filtered = None
                p_on_idx = p_on_idx_raw
            
            # Debug: Check data ranges before plotting
            p_valid_count = p_on_idx.notna().sum()
            dpdt_valid_count = dpdt_1h.notna().sum() if dpdt_1h is not None else 0
            if p_valid_count > 0:
                p_min, p_max = p_on_idx.min(), p_on_idx.max()
            else:
                p_min, p_max = float('nan'), float('nan')
            if dpdt_valid_count > 0:
                dpdt_min, dpdt_max = dpdt_1h.min(), dpdt_1h.max()
            else:
                dpdt_min, dpdt_max = float('nan'), float('nan')
                
            print(f"   [DEBUG] Pressure data: {p_valid_count}/{len(p_on_idx)} valid points, range: {p_min:.2f} - {p_max:.2f} hPa")
            print(f"   [DEBUG] dP/dt data: {dpdt_valid_count}/{len(dpdt_1h) if dpdt_1h is not None else 0} valid points, range: {dpdt_min:.4f} - {dpdt_max:.4f} hPa/hr")
            
            # Plot time series with proper axis scaling (3 subplots)
            # Note: ax3 should NOT share x-axis as it plots pressure derivatives vs factors (not time-based)
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
            # Share x-axis only between ax1 and ax2 (both are time series)
            ax2.sharex(ax1)
            
            # Configure datetime formatting for matplotlib with improved readability
            import matplotlib.dates as mdates
            
            # Determine appropriate tick spacing based on data duration
            time_span_days = (idx.max() - idx.min()).days
            
            if time_span_days <= 2:  # 1-2 days: show every 6 hours
                major_interval = 6
                minor_interval = 2
                date_format = '%m-%d\n%H:%M'
            elif time_span_days <= 7:  # 3-7 days: show daily
                major_interval = 24  
                minor_interval = 6
                date_format = '%m-%d\n%H:%M'
            elif time_span_days <= 30:  # 8-30 days: show every 2-3 days
                major_interval = 72  # Every 3 days
                minor_interval = 24  # Daily minor ticks
                date_format = '%m-%d'
            else:  # > 30 days: show weekly
                major_interval = 168  # Weekly (7*24 hours)
                minor_interval = 24   # Daily minor ticks
                date_format = '%m-%d'
            
            # Apply formatting to shared time axes (ax1 and ax2)
            for ax_time in [ax1, ax2]:
                ax_time.xaxis.set_major_locator(mdates.HourLocator(interval=major_interval))
                ax_time.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
                ax_time.xaxis.set_minor_locator(mdates.HourLocator(interval=minor_interval))
                # Rotate labels if needed and add padding
                if time_span_days > 7:
                    ax_time.tick_params(axis='x', rotation=45)
                # Add some padding to prevent label overlap
                ax_time.tick_params(axis='x', pad=8)
            
            # Plot 1: Pressure time series (both raw and filtered if available) - REVERT TO WORKING VERSION
            if p_on_idx_filtered is not None:
                # Plot raw data first (background)
                ax1.plot(idx, p_on_idx_raw, color='lightgray', alpha=0.6, label=f'Raw Pressure (hPa)', linewidth=1, zorder=1)
                # Plot filtered data on top (foreground)
                ax1.plot(idx, p_on_idx_filtered, color='tab:blue', label=f'Filtered Pressure (hPa): {p_min:.1f}-{p_max:.1f}', linewidth=2, zorder=2)
            else:
                # Only raw data available
                ax1.plot(idx, p_on_idx, color='tab:blue', label=f'Pressure (hPa): {p_min:.1f}-{p_max:.1f}', linewidth=1.5)
            
            # Add red dots at y=0 for missing pressure data points
            missing_mask = p_on_idx.isna()
            if missing_mask.sum() > 0:
                missing_times = idx[missing_mask]
                ax1.scatter(missing_times, [0] * len(missing_times), 
                           color='red', s=2, alpha=0.7, 
                           label=f'Missing data ({missing_mask.sum()} points)', zorder=5)
            
            ax1.set_ylabel('Pressure (hPa)')
            if p_on_idx_filtered is not None:
                ax1.set_title(f'Pressure Time Series: Raw vs 6-Hour Low-Pass Filtered (Range: {p_min:.2f} - {p_max:.2f} hPa)')
            else:
                ax1.set_title(f'Pressure Time Series (Range: {p_min:.2f} - {p_max:.2f} hPa)')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            
            # Plot 2: Pressure derivative - Show both raw and filtered (if available)
            if dpdt_valid_count > 0:
                # Plot raw derivatives
                ax2.plot(idx, dpdt_1h, color='lightcoral', alpha=0.6, label=f'Raw dP/dt (hPa/hr): {dpdt_min:.3f}-{dpdt_max:.3f}', linewidth=1, zorder=1)
                
                # Plot filtered derivatives if available
                if dpdt_1h_filtered is not None:
                    dpdt_filtered_min, dpdt_filtered_max = dpdt_1h_filtered.min(), dpdt_1h_filtered.max()
                    ax2.plot(idx, dpdt_1h_filtered, color='tab:red', label=f'48-Hour Filtered dP/dt (hPa/hr): {dpdt_filtered_min:.3f}-{dpdt_filtered_max:.3f}', linewidth=2, zorder=2)
                else:
                    # If no filtering applied, use original styling
                    ax2.lines[0].set_color('tab:red')
                    ax2.lines[0].set_linewidth(1.5)
                    ax2.lines[0].set_alpha(1.0)
                    ax2.lines[0].set_label(f'dP/dt (hPa/hr): {dpdt_min:.3f}-{dpdt_max:.3f}')
            else:
                ax2.plot(idx, dpdt_1h, color='tab:red', label='dP/dt (hPa/hr): No data', linewidth=1.5)
            
            # Add red dots at y=0 for missing dP/dt data points
            if dpdt_1h is not None:
                dpdt_missing_mask = dpdt_1h.isna()
                if dpdt_missing_mask.sum() > 0:
                    dpdt_missing_times = idx[dpdt_missing_mask]
                    ax2.scatter(dpdt_missing_times, [0] * len(dpdt_missing_times), 
                               color='red', s=2, alpha=0.7, 
                               label=f'Missing dP/dt ({dpdt_missing_mask.sum()} points)', zorder=5)
                
            # Plot hourly slope if available
            if slope3 is not None and slope3.notna().sum() > 0:
                ax2.plot(idx, slope3, color='tab:orange', alpha=0.6, label='1h slope (hPa/hr)', linewidth=1)
                
            ax2.axhline(0, color='k', linewidth=1, alpha=0.5)
            ax2.set_ylabel('dP/dt (hPa/hr)')
            ax2.set_xlabel('Time')
            # Update title based on filtering status
            if dpdt_1h_filtered is not None:
                dpdt_filtered_min, dpdt_filtered_max = dpdt_1h_filtered.min(), dpdt_1h_filtered.max()
                ax2.set_title(f'Pressure Derivatives: 6-Hour Window + 48-Hour Filter (Range: {dpdt_filtered_min:.4f} - {dpdt_filtered_max:.4f} hPa/hr)')
            else:
                ax2.set_title(f'Pressure Derivative - 6-Hour Window (Range: {dpdt_min:.4f} - {dpdt_max:.4f} hPa/hr)')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            
            # Plot 3: Factor contributions vs pressure derivatives (scatter plot)
            n_factors = G_contributions.shape[1]
            if dpdt_1h is not None and dpdt_valid_count > 10:
                # Get consistent colors for factors
                if hasattr(self, 'color_manager'):
                    factor_colors = self.color_manager._get_factor_colors(n_factors)
                else:
                    # Fallback colors if color manager not available
                    factor_colors = plt.cm.Set1(np.linspace(0, 1, n_factors))
                
                # Plot factors in order with H2S factor last (for top layer visibility)
                factor_plot_order = self.color_manager.get_factor_plot_order() if hasattr(self, 'color_manager') else list(range(n_factors))
                
                for f in factor_plot_order:
                    factor_contrib = G_contributions[:, f]
                    
                    # Create scatter plot with factor-colored points
                    # Extract actual pressure derivative VALUES (not datetime index)
                    # Use filtered derivatives for correlation analysis if available
                    if dpdt_1h_filtered is not None:
                        dpdt_values = dpdt_1h_filtered.values  # Use filtered derivatives for cleaner correlations
                    else:
                        dpdt_values = dpdt_1h.values  # Fall back to raw derivatives
                    valid_data_mask = ~(np.isnan(dpdt_values) | np.isnan(factor_contrib))
                    if valid_data_mask.sum() > 0:
                        # Adjust alpha and marker size for H2S factor (red) to make it more prominent
                        is_h2s = hasattr(self, 'color_manager') and self.color_manager.is_h2s_factor(f)
                        alpha_val = 0.8 if is_h2s else 0.6
                        marker_size = 25 if is_h2s else 20
                        edge_width = 1.0 if is_h2s else 0.5
                        
                        ax3.scatter(dpdt_values[valid_data_mask], factor_contrib[valid_data_mask], 
                                  c=[factor_colors[f]], alpha=alpha_val, s=marker_size,
                                  label=f'Factor {f+1}', edgecolors='black', linewidth=edge_width)
                
                ax3.set_xlabel('Pressure Derivative (hPa/hr)')
                ax3.set_ylabel('Factor Contribution')
                # Update title based on filtering used
                if dpdt_1h_filtered is not None:
                    ax3.set_title('Factor Contributions vs Filtered Pressure Derivatives (Barometric Pumping Analysis)')
                else:
                    ax3.set_title('Factor Contributions vs 6-Hour Pressure Derivative (Barometric Pumping Analysis)')
                ax3.grid(True, alpha=0.3)
                ax3.legend(loc='upper right', ncol=min(n_factors, 3))  # Max 3 columns for legend
                
                # Add zero reference lines
                ax3.axhline(0, color='k', linewidth=0.5, alpha=0.3)
                ax3.axvline(0, color='k', linewidth=0.5, alpha=0.3)
                
            else:
                ax3.text(0.5, 0.5, 'Insufficient pressure derivative data\nfor factor correlation analysis', 
                        transform=ax3.transAxes, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
                ax3.set_xlabel('Pressure Derivative (hPa/hr)')
                ax3.set_ylabel('Factor Contribution')
                ax3.set_title('Factor Contributions vs Pressure Derivative')
            
            # Ensure proper y-axis limits to show the data
            if p_valid_count > 0 and not (pd.isna(p_min) or pd.isna(p_max)) and p_max != p_min:
                p_range = p_max - p_min
                ax1.set_ylim(p_min - 0.05*p_range, p_max + 0.05*p_range)
            
            if (dpdt_valid_count > 0 and dpdt_1h is not None and 
                not (pd.isna(dpdt_min) or pd.isna(dpdt_max)) and dpdt_max != dpdt_min):
                dpdt_range = dpdt_max - dpdt_min
                ax2.set_ylim(dpdt_min - 0.05*dpdt_range, dpdt_max + 0.05*dpdt_range)
            
            # Set appropriate limits for factor contributions plot
            if dpdt_1h is not None and dpdt_valid_count > 10:
                # Set reasonable limits for factor contributions based on data
                factor_min = np.min(G_contributions)
                factor_max = np.max(G_contributions)
                if factor_max != factor_min:
                    factor_range = factor_max - factor_min
                    ax3.set_ylim(factor_min - 0.05*factor_range, factor_max + 0.05*factor_range)
                
            plt.tight_layout()
            dpdt_plot_path = dashboard_dir / f"{self.filename_prefix}_pressure_derivative.png"
            plt.savefig(dpdt_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(dpdt_plot_path)
            print(f"   [PRESSURE] Saved: pressure_derivative.png")
            
            # Compute lagged correlations vs factor contributions (0-3 h)
            try:
                n_factors = G_contributions.shape[1]
                factor_names = [f"Factor_{i+1}" for i in range(n_factors)]
                G_df = pd.DataFrame(G_contributions, index=idx, columns=factor_names)
                
                # Only proceed if we have valid pressure derivative data
                if dpdt_1h is not None and dpdt_valid_count > 10:  # Need at least 10 valid points
                    # Determine steps per hour from median dt
                    med_dt = np.median(np.diff(idx.values).astype('timedelta64[s]').astype(float))
                    med_dt_h = med_dt / 3600.0 if med_dt and med_dt > 0 else 1.0
                    step_per_hour = max(1, int(round(1.0 / med_dt_h)))
                    lags_hours = [0, 1, 2, 3]
                    corr_mat = np.full((n_factors, len(lags_hours)), np.nan)
                    
                    for j, lag in enumerate(lags_hours):
                        shifted = dpdt_1h.shift(-lag * step_per_hour)
                        for i in range(n_factors):
                            # Combine and drop NaN values (important for sparse pressure data)
                            combined = pd.DataFrame({
                                'dpdt': shifted,
                                'factor': G_df.iloc[:, i]
                            }).dropna()
                            
                            if len(combined) > 5:  # Need sufficient overlap for reliable correlation
                                corr_mat[i, j] = combined['dpdt'].corr(combined['factor'])
                else:
                    print(f"   [WARN] Insufficient pressure derivative data ({dpdt_valid_count} points) for correlation analysis")
                    n_factors = G_contributions.shape[1]
                    lags_hours = [0, 1, 2, 3]
                    corr_mat = np.full((n_factors, len(lags_hours)), np.nan)
                    
                # Plot heatmap with adaptive color scale
                max_abs_corr = np.nanmax(np.abs(corr_mat))
                if max_abs_corr < 0.2:  # If all correlations are small, use a more sensitive scale
                    vmin, vmax = -0.2, 0.2
                    title_suffix = " (enhanced scale: ±0.2)"
                else:
                    vmin, vmax = -1, 1
                    title_suffix = ""
                
                fig, ax = plt.subplots(figsize=(2.0 + 1.0*len(lags_hours), 0.8 + 0.5*n_factors))
                im = ax.imshow(corr_mat, aspect='auto', cmap='RdBu_r', vmin=vmin, vmax=vmax)
                ax.set_xticks(range(len(lags_hours)))
                ax.set_xticklabels([f"lag {h}h" for h in lags_hours])
                ax.set_yticks(range(n_factors))
                ax.set_yticklabels(factor_names)
                ax.set_title(f'Correlation: dP/dt vs Factor contributions{title_suffix}')
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Pearson r')
                
                # Add correlation values as text for better readability
                for i in range(n_factors):
                    for j in range(len(lags_hours)):
                        if not np.isnan(corr_mat[i, j]):
                            # Use white text for strong correlations, black for weak ones
                            text_color = 'white' if abs(corr_mat[i, j]) > 0.5 * max(abs(vmin), abs(vmax)) else 'black'
                            ax.text(j, i, f'{corr_mat[i, j]:.3f}', ha='center', va='center', 
                                    color=text_color, fontsize=9, weight='bold')
                
                plt.tight_layout()
                corr_plot_path = dashboard_dir / f"{self.filename_prefix}_dpdt_factor_corr.png"
                plt.savefig(corr_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                plot_files.append(corr_plot_path)
                print(f"   [PRESSURE] Saved: dpdt_factor_corr.png")
                
            except Exception as e:
                print(f"[WARN] Could not compute dP/dt factor correlations: {e}")
                
            # Store for HTML context (optional)
            self._dpdt_series = dpdt_1h
            
        except Exception as e:
            import traceback
            print(f"[WARN] Failed to generate dP/dt analysis: {e}")
            print(f"[DEBUG] Error details: {traceback.format_exc()}")
            # Continue without pressure derivative plots
            pass
    
    def convert_dashboard_to_pdf(self, dashboard_dir, station=None):
        """Convert HTML dashboard to PDF using multiple fallback methods."""
        # Use provided station or derive from self.station
        station_name = station or self.station
        
        # Use the specific HTML dashboard file that was created
        html_file = dashboard_dir / f"{self.filename_prefix}_pmf_dashboard.html"
        
        if not html_file.exists():
            print(f"   [WARN] HTML dashboard not found: {html_file}")
            return None
        # Create PDF filename using filename_prefix
        pdf_file = dashboard_dir / f"{self.filename_prefix}_pmf_dashboard.pdf"
        
        try:
            # Method 1: Use weasyprint (best option, pure Python)
            if HAS_WEASYPRINT:
                try:
                    import weasyprint
                    weasyprint.HTML(filename=str(html_file)).write_pdf(str(pdf_file))
                    print(f"   [OK] PDF created with weasyprint: {pdf_file.name}")
                    return pdf_file
                except Exception as e:
                    print(f"   [WARN] Weasyprint failed: {e}, trying next method")
            
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
                    print(f"   [OK] PDF created with pdfkit: {pdf_file.name}")
                    return pdf_file
                except Exception as e:
                    # Only log pdfkit failures if it's not the common missing executable issue
                    if 'wkhtmltopdf executable found' not in str(e):
                        print(f"   [WARN] pdfkit failed: {e}, trying next method")
                    else:
                        print(f"   [INFO] pdfkit not available (wkhtmltopdf not found)")
            
            # Method 3: Try using Chrome/Edge headless
            success = self._convert_with_chrome(html_file, pdf_file)
            if success:
                print(f"   [OK] PDF created with Chrome: {pdf_file.name}")
                return pdf_file
            
            # Method 4: Create a simple text-based report as fallback
            text_report = self._create_text_report(dashboard_dir, station_name)
            print(f"   [WARN] All PDF methods failed, created text report: {text_report.name if text_report else 'failed'}")
            return text_report
        
        except Exception as e:
            print(f"   [ERROR] PDF conversion failed: {e}")
            try:
                text_report = self._create_text_report(dashboard_dir, station_name)
                print(f"   [OK] Created fallback text report: {text_report.name if text_report else 'failed'}")
                return text_report
            except Exception as e2:
                print(f"   [ERROR] Even text report failed: {e2}")
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
            print(f"   [ERROR] Failed to create text report: {e}")
            return None
    
    def _create_sankey_diagram(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """
        Create a Sankey diagram showing the flow from PMF factors to species concentrations.
        This provides an intuitive visualization of how each factor contributes to different species.
        """
        print("   [FLOW] Creating Sankey diagram (Factors -> Species)...")
        
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
        
        # Use H2S plotting order to ensure H2S factor connections are drawn last (most prominent)
        factor_plot_order = self.color_manager.get_factor_plot_order()
        
        for factor_idx in factor_plot_order:
            for species_idx in range(len(self.species_names)):
                flow_value = factor_species_flows[factor_idx, species_idx]
                
                if flow_value > min_flow_threshold:  # Only show significant flows
                    sources.append(factor_idx)  # Factor node index
                    targets.append(self.factors + species_idx)  # Species node index (offset by n_factors)
                    values.append(flow_value)
        
        # Use ColorManager for consistent colors (including H2S red factor)
        factor_colors_hex = self.color_manager.get_factor_colors()
        
        # Convert hex colors to rgba format for Plotly
        def hex_to_rgba(hex_color, alpha=0.8):
            hex_color = hex_color.lstrip('#')
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f'rgba({r}, {g}, {b}, {alpha})'
        
        factor_colors = [hex_to_rgba(color) for color in factor_colors_hex]
        
        # Use ColorManager for species colors too
        species_colors_hex = [self.color_manager.get_species_color(species) for species in self.species_names]
        species_colors = [hex_to_rgba(color, 0.6) for color in species_colors_hex]
        
        node_colors = factor_colors[:self.factors] + species_colors
        
        # Dynamic layout parameters to avoid vertical overlaps regardless of flow size
        # Compute node padding and thickness in pixels based on figure height and node counts
        layout_height_px = 900
        top_margin_px = 80
        bottom_margin_px = 50
        available_px = max(200, layout_height_px - (top_margin_px + bottom_margin_px))
        
        left_nodes = self.factors
        right_nodes = len(self.species_names)
        max_nodes = max(left_nodes, right_nodes)
        
        # Start with desired thickness and compute pad to fit all nodes
        thickness_px = 18
        min_thickness_px = 12
        min_pad_px = 8
        
        # Adjust thickness/pad to fit within available height
        if max_nodes > 1:
            pad_px = int((available_px - max_nodes * thickness_px) / (max_nodes - 1))
        else:
            pad_px = available_px - thickness_px
        
        if pad_px < min_pad_px:
            # Reduce thickness to make room
            thickness_px = max(min_thickness_px, int(available_px / max_nodes) - min_pad_px)
            if max_nodes > 1:
                pad_px = max(min_pad_px, int((available_px - max_nodes * thickness_px) / (max_nodes - 1)))
            else:
                pad_px = min_pad_px
        
        # Final safety clamp
        pad_px = max(min_pad_px, pad_px)
        
        print(f"   [CALC] Sankey layout: thickness={thickness_px}px, pad={pad_px}px, nodes_left={left_nodes}, nodes_right={right_nodes}, available_px={available_px}")
        
        # Fix node columns; let Plotly auto-position y to respect pad and avoid overlaps
        node_x = [0.01] * self.factors + [0.99] * len(self.species_names)
        
        
        # Create Sankey diagram with optimized positioning (Plotly manages y to avoid overlaps)
        fig = go.Figure(data=[go.Sankey(
            arrangement="snap",  # Let Plotly optimize y positions and avoid overlaps
            node=dict(
                pad=pad_px,
                thickness=thickness_px,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=node_colors,
                x=node_x
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
                text=f"{self.station} PMF Source Apportionment: Factor -> Species Flow",
                x=0.5,
                font=dict(size=16)
            ),
            font_size=10,  # Slightly smaller font to fit better
            width=1200,
            height=900,  # Increased height to accommodate better spacing
            margin=dict(t=80, b=50, l=50, r=50),  # Add top margin for annotations
            annotations=[
                dict(
                    text="Factors", x=0.01, y=0.98, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=14, color="black"), 
                    xanchor="left", yanchor="top"
                ),
                dict(
                    text="Species", x=0.99, y=0.98, xref="paper", yref="paper",
                    showarrow=False, font=dict(size=14, color="black"), 
                    xanchor="right", yanchor="top"
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
                fig.write_image(str(png_path), format='png', width=1200, height=900, scale=2)
                plot_files.append(png_path)
                print(f"     [OK] Saved: sankey_diagram.png")
                png_success = True
            except Exception as e1:
                # Try without format specification
                try:
                    fig.write_image(str(png_path), width=1200, height=900, scale=2)
                    plot_files.append(png_path)
                    print(f"     [OK] Saved: sankey_diagram.png (fallback method)")
                    png_success = True
                except Exception as e2:
                    # Print both error messages for debugging
                    print(f"     [WARN] PNG export failed (method 1): {str(e1).strip()}")
                    print(f"     [WARN] PNG export failed (method 2): {str(e2).strip()}")
                        
        except Exception as png_err:
            print(f"     [WARN] PNG export outer exception: {str(png_err).strip()}")
        
        if not png_success:
            print(f"     [DATA] Creating matplotlib-based Sankey fallback...")
            # Create a static matplotlib version as fallback
            self._create_matplotlib_sankey_simple(dashboard_dir, plot_files, F_profiles)
        
        print(f"     [OK] Saved: sankey_diagram.html (interactive)")
        return True
    
    def _create_matplotlib_sankey(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """
        Create a Sankey-style diagram using matplotlib.
        This is a simplified version since matplotlib doesn't have native Sankey support.
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle(f'{self.station} PMF Factor -> Species Flow Diagram', fontsize=16, fontweight='bold')
        
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
        print(f"     [OK] Saved: sankey_diagram.png")
    
    def _create_matplotlib_sankey_simple(self, dashboard_dir, plot_files, F_profiles):
        """
        Create a simplified flow diagram when Plotly PNG export fails.
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle(f'{self.station} PMF Factor -> Species Contribution Matrix', fontsize=16, fontweight='bold')
        
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
        print(f"     [OK] Saved: sankey_diagram.png (heatmap style)")
    
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
        
        # Draw factor nodes using plotting order for consistency
        factor_plot_order = self.color_manager.get_factor_plot_order()
        
        for i in factor_plot_order:
            pos = factor_positions[i]
            is_h2s = self.color_manager.is_h2s_factor(i)
            
            # Enhanced styling for H2S factor
            alpha_val = 0.9 if is_h2s else 0.8
            edge_width = 2 if is_h2s else 1
            size = 0.12 if is_h2s else 0.1
            
            circle = plt.Circle(pos, size, color=factor_colors[i], alpha=alpha_val, 
                               edgecolor='black', linewidth=edge_width)
            ax.add_patch(circle)
            
            font_size = 11 if is_h2s else 10
            ax.text(pos[0]*1.15, pos[1]*1.15, f'F{i+1}', ha='center', va='center', 
                   fontweight='bold', fontsize=font_size)
        
        # Draw species nodes  
        for i, pos in enumerate(species_positions):
            circle = plt.Circle(pos, 0.08, color=species_colors[i], alpha=0.7)
            ax.add_patch(circle)
            
            # Abbreviated species name
            name = self.species_names[i][:6] + '..' if len(self.species_names[i]) > 8 else self.species_names[i]
            ax.text(pos[0]*1.2, pos[1]*1.2, name, ha='center', va='center', 
                   fontweight='bold', fontsize=8)
        
        # Draw connections for significant flows - use plotting order for layering
        max_flow = np.max(F_profiles)
        min_flow_threshold = 0.1 * max_flow
        
        for factor_idx in factor_plot_order:
            for species_idx in range(n_species):
                flow = F_profiles[factor_idx, species_idx]
                if flow > min_flow_threshold:
                    factor_pos = factor_positions[factor_idx]
                    species_pos = species_positions[species_idx]
                    
                    # Enhanced styling for H2S factor connections
                    is_h2s = self.color_manager.is_h2s_factor(factor_idx)
                    base_width = (flow / max_flow) * 5 + 0.5
                    line_width = base_width * 1.3 if is_h2s else base_width
                    alpha_val = 0.7 if is_h2s else 0.5
                    
                    ax.plot([factor_pos[0], species_pos[0]], [factor_pos[1], species_pos[1]], 
                           color=factor_colors[factor_idx], alpha=alpha_val, linewidth=line_width)
        
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
        print(f"     [OK] Saved: sankey_diagram.png (flow chart style)")
    
    def _create_matplotlib_sankey_proper(self, dashboard_dir, plot_files, F_profiles):
        """
        Attempt to create a proper Sankey diagram using matplotlib's native Sankey class.
        """
        try:
            from matplotlib.sankey import Sankey
            
            fig, ax = plt.subplots(figsize=(14, 10))
            fig.suptitle(f'{self.station} PMF Factor -> Species Sankey Diagram', fontsize=16, fontweight='bold')
            
            # Create Sankey instance
            sankey = Sankey(ax=ax, scale=0.01, offset=0.3, format='%.0f', gap=0.5)
            
            # Calculate flows
            factor_species_flows = F_profiles  # Shape: (n_factors, n_species)
            
            # Prepare flows for each factor - use plotting order
            factor_plot_order = self.color_manager.get_factor_plot_order()
            
            for factor_idx in factor_plot_order:
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
            
            # Apply consistent color scheme using ColorManager
            factor_colors_hex = self.color_manager.get_factor_colors()
            for i, diagram in enumerate(diagrams):
                if i < len(factor_colors_hex):
                    diagram.texts[-1].set_color(factor_colors_hex[i])
                    diagram.texts[-1].set_fontweight('bold')
                    
                    # Enhance H2S factor styling
                    if self.color_manager.is_h2s_factor(i):
                        diagram.texts[-1].set_fontsize(12)
                        diagram.texts[-1].set_bbox(dict(boxstyle='round,pad=0.3', 
                                                       facecolor=factor_colors_hex[i], 
                                                       alpha=0.3))
            
            plt.tight_layout()
            plot_path = dashboard_dir / f"{self.filename_prefix}_sankey_diagram.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            plot_files.append(plot_path)
            print(f"     [OK] Saved: sankey_diagram.png (matplotlib Sankey)")
            return True
            
        except Exception as e:
            print(f"     [WARN] Matplotlib Sankey failed: {e}")
            return False
    
    def _create_custom_flow_sankey(self, dashboard_dir, plot_files, F_profiles, G_contributions):
        """
        Create a custom flow diagram that resembles a Sankey diagram.
        This is designed to be a reliable fallback that always works.
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.suptitle(f'{self.station} PMF Source Apportionment Flow Diagram\n(Factors -> Species)', 
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
        
        # Draw factor nodes (left side) - use plotting order for consistent layering
        factor_nodes = [None] * self.factors  # Pre-allocate to maintain indexing
        factor_plot_order = self.color_manager.get_factor_plot_order()
        
        for plot_order_idx, factor_idx in enumerate(factor_plot_order):
            y_pos = factor_y_positions[factor_idx]
            color = factor_colors[factor_idx]
            size = factor_sizes[factor_idx]
            
            # H2S factor gets enhanced styling
            is_h2s = self.color_manager.is_h2s_factor(factor_idx)
            alpha_val = 0.9 if is_h2s else 0.8
            edge_width = 2 if is_h2s else 1
            zorder = 4 if is_h2s else 3
            
            circle = plt.Circle((factor_x, y_pos), size, color=color, alpha=alpha_val, 
                               zorder=zorder, edgecolor='black', linewidth=edge_width)
            ax.add_patch(circle)
            factor_nodes[factor_idx] = (factor_x, y_pos, size)  # Store at original index
            
            # Factor labels - emphasize H2S factor
            font_size = 13 if is_h2s else 12
            ax.text(factor_x - 0.08, y_pos, f'Factor {factor_idx+1}', ha='right', va='center', 
                   fontsize=font_size, fontweight='bold', color=color)
        
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
        
        # Draw flow streams (the key Sankey-like feature) - use plotting order for layering
        min_flow_threshold = 0.02 * max_flow  # Only show flows > 2% of maximum
        
        for factor_idx in factor_plot_order:
            factor_x_pos, factor_y_pos, factor_size = factor_nodes[factor_idx]
            factor_color = factor_colors[factor_idx]
            is_h2s = self.color_manager.is_h2s_factor(factor_idx)
            
            for species_idx in range(len(self.species_names)):
                flow_value = factor_species_flows[factor_idx, species_idx]
                
                if flow_value > min_flow_threshold:
                    species_x_pos, species_y_pos, species_size = species_nodes[species_idx]
                    
                    # Calculate flow width (Sankey characteristic) - enhance H2S flows
                    base_width = (flow_value / max_flow) * 30 + 2
                    flow_width = base_width * 1.2 if is_h2s else base_width
                    
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
        print(f"     [OK] Saved: sankey_diagram.png (custom flow diagram)")
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

[WEIGHTING] SPECIES-SPECIFIC UNCERTAINTY WEIGHTING:
  --species-weight      Multiply uncertainties for specific species to downweight them in PMF
                        Format: SPECIES=FACTOR (e.g., CH4=5 or CH4=5,H2S=2)
                        Can be used multiple times: --species-weight CH4=5 --species-weight H2S=2
                        Applied AFTER S/N categorization but BEFORE saving uncertainties for ESAT
                        Higher multipliers = lower influence in factor optimization
                        
                        Examples:
                        --species-weight CH4=5          # Downweight CH4 by 5x
                        --species-weight CH4=5,H2S=2    # Multiple in one flag
                        --species-weight CH4=10         # Strong downweighting
                        
                        Effects on PMF:
                        - Increases uncertainty -> reduces species weight in LS objective
                        - Does NOT change concentration values
                        - Helps address extreme dynamic range differences
                        - Preserves PMF additive assumptions (unlike log-transform)

[COMPLAINTS] COMPLAINT CORRELATION ANALYSIS:
  --complaint-correlation-hours N  Time window in hours for complaint correlation analysis
                        Default: 0 (uses daily aggregation - legacy behavior)
                        
                        Modes:
                        --complaint-correlation-hours 0   # Daily aggregation (default)
                        --complaint-correlation-hours 6   # ±6 hour window around noon
                        --complaint-correlation-hours 12  # ±12 hour window around noon
                        
  --complaint-window METHOD     Statistical aggregation method for data within complaint time windows
                        Default: average
                        
                        Methods:
                        --complaint-window average    # Mean value (default)
                        --complaint-window peak       # Maximum value
                        --complaint-window median     # Median value  
                        --complaint-window mode       # Most frequent value
                        --complaint-window range      # Range (max - min)
                        
                        Note: Only used when --complaint-correlation-hours > 0
                        
                        Window Logic:
                        - 0: Correlates daily averaged concentrations/factors with daily complaint counts
                        - N>0: For each complaint day, aggregates concentrations/factors from 
                               noon±N hours using selected method and correlates with complaint count
                        - Complaint plotting: Shows complaints at noon (12:00) instead of midnight
                        
                        Use Cases:
                        - 6-8 hours: Capture diurnal patterns around complaint times
                        - 12+ hours: Full day correlation with centered time window
                        - 0: Preserve legacy daily correlation behavior
                        - peak: Focus on maximum pollutant concentrations during complaint periods
                        - median: Robust to outliers in concentration data

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

# Species weighting to handle extreme concentration ranges (e.g., CH4 >> other species):
python pmf_source_app.py MMF9 --species-weight CH4=5 --species-weight H2S=2 --uncertainty-mode epa

# Weight-aware initialization with species weighting (addresses factor degeneracy):
python pmf_source_app.py MMF9 --species-weight CH4=5 --weight-aware-init --uncertainty-mode epa

# Complaint correlation analysis with time windows (instead of daily aggregation):
python pmf_source_app.py MMF9 --complaint-correlation-hours 6 --uncertainty-mode epa --snr-enable

# Complaint correlation using peak concentrations within time windows:
python pmf_source_app.py MMF9 --complaint-correlation-hours 12 --complaint-window peak --uncertainty-mode epa

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
    
    # Species-specific uncertainty weighting
    parser.add_argument('--species-weight', action='append', default=[],
                       help='Multiply uncertainties for specific species (e.g., --species-weight CH4=5 or --species-weight CH4=5,H2S=2). Applied after S/N adjustments. Can be used multiple times.')
    
    # Species exclusion from analysis
    parser.add_argument('--exclude-species', action='append', default=[],
                       help='Remove specific species from PMF analysis entirely (e.g., --exclude-species CH4 or --exclude-species CH4,H2S). Case-insensitive. Can be used multiple times: --exclude-species CH4 --exclude-species H2S')
    
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
    
    # Weight-aware initialization control
    weight_init_group = parser.add_mutually_exclusive_group()
    weight_init_group.add_argument('--weight-aware-init', dest='weight_aware_init', action='store_true',
                          help='Enable weight-aware initialization for weighted species (auto-enabled if --species-weight is used)')
    weight_init_group.add_argument('--no-weight-aware-init', dest='weight_aware_init', action='store_false',
                          help='Disable weight-aware initialization even when species weights are applied')
    parser.set_defaults(weight_aware_init=None)
    
    # Species regularization control
    parser.add_argument('--reg-species', action='append', default=[],
                       help='Species to regularize (repeatable). Example: --reg-species CH4 --reg-species H2S')
    parser.add_argument('--reg-lambda', action='append', type=float, default=[],
                       help='Regularization strength lambda per species. If single value provided, broadcast to all reg-species. Example: --reg-lambda 10')
    parser.add_argument('--reg-template', action='append', default=[],
                       choices=['zero', 'uniform', 'from-file'],
                       help='Template type per regulated species: zero (h0=0), uniform (small uniform vector), from-file (CSV with k values). Broadcast if single value.')
    parser.add_argument('--reg-template-file', action='append', default=[],
                       help='CSV file path for from-file template (k rows, 1 column). Must match count of from-file template entries.')
    parser.add_argument('--reg-bursts', type=int, default=5,
                       help='Number of train->prox cycles for regularization (default: 5)')
    parser.add_argument('--reg-iter-per-burst', type=int, default=50,
                       help='Max iterations per training burst (default: 50)')
    parser.add_argument('--reg-tol', type=float, default=1e-4,
                       help='Early stop tolerance: relative change in regulated columns (default: 1e-4)')
    parser.add_argument('--reg-elastic-l1', type=float, default=0.0,
                       help='Elastic-net L1 penalty on deviation from h0 (default: 0.0, disabled)')
    
    # Bootstrap error estimation options
    parser.add_argument('--bootstrap', action='store_true', default=False,
                       help='Run bootstrap error estimation after PMF analysis (default: disabled)')
    parser.add_argument('--bootstrap-n', type=int, default=100,
                       help='Number of bootstrap samples to run (default: 100)')
    parser.add_argument('--bootstrap-block-size', type=int, default=None,
                       help='Block size for temporal bootstrap resampling. If None, auto-estimated using optimal_block_length (default: None)')
    parser.add_argument('--bootstrap-threshold', type=float, default=0.6,
                       help='Factor mapping threshold for bootstrap correlation (default: 0.6)')
    parser.add_argument('--bootstrap-parallel', action='store_true', default=True,
                       help='Enable parallel processing for bootstrap (default: enabled)')
    parser.add_argument('--bootstrap-cpus', type=int, default=None,
                       help='Number of CPUs for bootstrap parallel processing. If None, uses all available (default: None)')
    parser.add_argument('--bootstrap-seed', type=int, default=None,
                       help='Random seed for bootstrap resampling. If None, uses main --seed value (default: None)')
    parser.add_argument('--bootstrap-keep-h', action='store_true', default=True,
                       help='Keep factor profiles (H matrix) from bootstrap samples (default: enabled)')
    parser.add_argument('--bootstrap-reuse-seed', action='store_true', default=True,
                       help='Reuse seed across bootstrap samples for deterministic resampling (default: enabled)')
    parser.add_argument('--bootstrap-overlapping', action='store_true', default=False,
                       help='Allow overlapping blocks in bootstrap resampling (default: disabled)')
    
    # Complaint correlation analysis controls
    parser.add_argument('--complaint-correlation-hours', type=int, default=0,
                       help='Time window in hours for complaint correlation analysis. Default 0 uses daily aggregation. Positive values (e.g., 6) correlate complaints with ±N hours of concentration data around each complaint timestamp (default: 0)')
    parser.add_argument('--complaint-window', choices=['peak', 'average', 'median', 'mode', 'range'], default='average',
                       help='Statistical aggregation method for data within complaint correlation time windows: peak (maximum), average (mean), median, mode (most frequent), range (max-min). Only used when --complaint-correlation-hours > 0 (default: average)')
    
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
    
    print("MMF PMF Source Apportionment Analysis (ESAT Fixed)")
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
            delay_h=args.delay_h,
            # Species-specific uncertainty weighting
            species_weight=args.species_weight,
            # Species exclusion from analysis
            exclude_species=args.exclude_species,
            # Weight-aware initialization control
            weight_aware_init=args.weight_aware_init,
            # Species regularization control
            reg_species=args.reg_species,
            reg_lambda=args.reg_lambda,
            reg_template=args.reg_template,
            reg_template_files=args.reg_template_file,
            reg_bursts=args.reg_bursts,
            reg_iter_per_burst=args.reg_iter_per_burst,
            reg_tol=args.reg_tol,
            reg_elastic_l1=args.reg_elastic_l1,
            # Bootstrap error estimation parameters
            bootstrap=args.bootstrap,
            bootstrap_n=args.bootstrap_n,
            bootstrap_block_size=args.bootstrap_block_size,
            bootstrap_threshold=args.bootstrap_threshold,
            bootstrap_parallel=args.bootstrap_parallel,
            bootstrap_cpus=args.bootstrap_cpus,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_keep_h=args.bootstrap_keep_h,
            bootstrap_reuse_seed=args.bootstrap_reuse_seed,
            bootstrap_overlapping=args.bootstrap_overlapping,
            # Complaint correlation analysis parameters
            complaint_correlation_hours=args.complaint_correlation_hours,
            complaint_window_method=args.complaint_window
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
            print("[WARN]  --models must be >= 1; defaulting to 1")
        pmf.max_factors = args.max_factors  # Pass max_factors for optimization
        pmf.max_workers = args.max_workers  # Control multiprocessing

        # Apply EPA BDL/missing runtime controls
        pmf.drop_row_threshold = max(0.0, min(1.0, args.drop_row_threshold))
        pmf.zero_as_bdl = args.zero_as_bdl
        pmf.save_masks = args.save_masks
        print(f"[CONFIG] BDL/Missing controls: drop-row-threshold={pmf.drop_row_threshold}, zero-as-bdl={pmf.zero_as_bdl}, save-masks={pmf.save_masks}")
        
        # Show EPA S/N weighting settings
        print(f"[EPA] S/N weighting: uncertainty-mode={pmf.uncertainty_mode}, snr-enable={pmf.snr_enable}")
        if pmf.snr_enable:
            print(f"   S/N thresholds: weak<{pmf.snr_weak_threshold}, bad<{pmf.snr_bad_threshold}")
            print(f"   Data quality: BDL weak>{pmf.snr_bdl_weak_frac*100:.0f}%, bad>{pmf.snr_bdl_bad_frac*100:.0f}%")
        
        # Run analysis workflow
        pmf.load_mmf_data()
        pmf.prepare_pmf_data()
        
        if pmf.run_pmf_analysis():
            # Run bootstrap error estimation if requested
            if args.bootstrap:
                print(f"\n[BOOTSTRAP] Bootstrap parameters: n={args.bootstrap_n}, parallel={args.bootstrap_parallel}")
                if pmf.bootstrap_seed:
                    print(f"   Using bootstrap seed: {pmf.bootstrap_seed}")
                else:
                    print(f"   Using main seed for bootstrap: {pmf.seed}")
                
                bootstrap_results = pmf.run_bootstrap_analysis()
                if bootstrap_results:
                    print(f"[OK] Bootstrap error estimation completed")
                    print(f"[DATA] Bootstrap results saved in: {pmf.output_dir / 'error'}")
                else:
                    print(f"[WARN] Bootstrap analysis failed, continuing without error estimation")
            
            # Run PCA analysis if requested
            if args.run_pca:
                print("\n[PCA] Running PCA analysis for comparison...")
                if pmf.run_pca_analysis():
                    print("[OK] PCA analysis completed successfully")
                else:
                    print("[WARN] PCA analysis failed, continuing without comparison plots")
            
            pmf.create_pmf_dashboard()
            pmf.generate_report()
            
            # Create PDF if requested
            pdf_path = None
            if args.create_pdf:
                print("\n[FILE] Creating PDF version of dashboard...")
                # Use the exact HTML filename that was created
                dashboard_dir = pmf.output_dir
                
                pdf_path = pmf.convert_dashboard_to_pdf(dashboard_dir)
                if pdf_path:
                    print(f"[OK] PDF created: {pdf_path}")
                else:
                    print("[WARN] PDF creation failed, but text report may have been created")
            
            print("\n[SUCCESS] Analysis Complete!")
            if args.run_pca:
                print("[DATA] PMF + PCA analysis results saved")
                print("   PMF-PCA comparison plots included in dashboard")
            else:
                print("[DATA] PMF analysis results saved")
                print("   Use --run-pca flag to include PCA comparison plots")
            
            if not args.create_pdf:
                print("   Use --create-pdf flag to generate PDF reports")
            elif pdf_path:
                print(f"[FILE] PDF dashboard: {pdf_path}")
            
            print(f"[DATA] Results saved in: {pmf.output_dir}")
            dashboard_name = f"{args.station}_pmf_dashboard.html" if args.station else "mmf_pmf_dashboard.html"
            print(f"[FILE] View dashboard: {pmf.output_dir}/{dashboard_name}")
        else:
            print("\n[ERROR] PMF analysis failed!")
            return 1
        
    except Exception as e:
        print(f"\n[ERROR] Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
