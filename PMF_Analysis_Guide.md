# MMF PMF Source Apportionment Analysis Guide

## Overview

The `pmf_source_apportionment.py` script performs Positive Matrix Factorization (PMF) source apportionment analysis on MMF environmental data following EPA PMF 5.0 User Guide best practices.

## Features

### 🎯 **EPA-Compliant Analysis**
- Follows EPA PMF 5.0 User Guide recommendations
- Uses proper uncertainty estimation formulas
- Implements Signal-to-Noise ratio categorization
- Applies EPA Method 1 for missing value handling

### 📊 **Comprehensive Dashboard**
- All required diagnostic plots from your specification:
  - Batch Models Loss Distribution
  - Residual Histogram
  - Observed vs. Predicted Scatter
  - Observed vs. Predicted Time Series
  - All Factor Profiles
  - Factor Fingerprints
  - G-Space Plot (F1 vs F2)
  - Factor Composition Radar

### 🔬 **Advanced Features**
- Automatic factor optimization
- Batch modeling with 20+ runs for robustness
- Error estimation (Displacement & Bootstrap)
- HTML dashboard with all plots
- Comprehensive analysis report

## Installation

### 1. Install ESAT Library
```bash
pip install git+https://github.com/quanted/esat.git
```

### 2. Install Additional Dependencies (if needed)
```bash
pip install scikit-learn numpy pandas matplotlib seaborn
```

## Usage

### Basic Usage
```bash
# Analyze full dataset for MMF1
python pmf_source_apportionment.py MMF1

# Analyze specific date range
python pmf_source_apportionment.py MMF1 --start-date 2022-01-01 --end-date 2022-12-31

# Quick analysis (skip error estimation)
python pmf_source_apportionment.py MMF2 --start-date 2022-06-01 --end-date 2022-08-31 --skip-errors
```

### Advanced Options
```bash
# Customize PMF parameters
python pmf_source_apportionment.py MMF1 \
    --factors 5 \
    --models 30 \
    --output-dir custom_pmf_results \
    --start-date 2022-01-01 \
    --end-date 2022-06-30
```

### Command Line Arguments
- `station`: MMF station (MMF1, MMF2, MMF6, MMF9)
- `--start-date`: Start date (YYYY-MM-DD format)
- `--end-date`: End date (YYYY-MM-DD format)
- `--factors`: Number of factors to resolve (default: 4, optimized automatically)
- `--models`: Number of models to run (default: 20)
- `--output-dir`: Output directory (default: pmf_results)
- `--skip-errors`: Skip error estimation for faster execution

## Output Files

### Generated Files Structure
```
pmf_results/
├── MMF1_concentrations.csv          # Processed concentration data
├── MMF1_uncertainties.csv           # EPA-compliant uncertainty data
├── MMF1_pmf_dashboard.html          # Interactive HTML dashboard
├── MMF1_pmf_report.md               # Comprehensive analysis report
├── displacement_error_f1.png        # Error analysis plots
├── bootstrap_error_f1.png
├── error_summary_f1.png
└── dashboard/                       # Individual diagnostic plots
    ├── batch_models_loss_distribution.png
    ├── residual_histogram.png
    ├── observed_vs_predicted_scatter.png
    ├── observed_vs_predicted_time_series.png
    ├── all_factor_profiles.png
    ├── factor_fingerprints.png
    ├── g-space_plot_f1_vs_f2.png
    └── factor_composition_radar.png
```

## Analysis Workflow

### 1. Data Preparation
- Loads MMF parquet data with units
- Filters by date range (if specified)
- Selects gas and particle species for PMF
- Removes rows with >50% missing data

### 2. Uncertainty Estimation
Following EPA PMF 5.0 formula: **σ = √[(error_fraction × concentration)² + (MDL)²]**

**Default Parameters:**
```python
# Method Detection Limits (MDL)
H2S: 0.5 μg/m³,    CH4: 0.05 mg/m³,   SO2: 0.5 μg/m³
PM1: 1.0 μg/m³,    PM2.5: 1.0 μg/m³,  PM10: 2.0 μg/m³
TSP: 2.5 μg/m³

# Error Fractions (Measurement Precision)
H2S: 15%,  CH4: 10%,  SO2: 15%
PM1: 10%,  PM2.5: 10%, PM10: 15%,  TSP: 20%
```

### 3. PMF Modeling
- **Method**: Least Squares Non-negative Matrix Factorization (LS-NMF)
- **Factor Optimization**: Tests 2-7 factors, selects optimal based on Q values
- **Batch Modeling**: Runs 20+ models for robust results
- **Convergence**: Δ < 0.01 for 50 iterations

### 4. Species Categorization
Based on Signal-to-Noise ratios:
- **Strong**: S/N ≥ 2.0 (included with full weight)
- **Weak**: 0.5 ≤ S/N < 2.0 (down-weighted)
- **Bad**: S/N < 0.5 (excluded from analysis)

### 5. Dashboard Generation
Creates seaborn-styled plots with proper formatting:
- Enhanced titles with station information
- Date range annotations
- High-resolution PNG outputs (300 DPI)
- Interactive HTML dashboard

## EPA Best Practices Implemented

### ✅ **Data Quality**
- Appropriate uncertainty estimation
- Missing value handling (EPA Method 1)
- Signal-to-noise evaluation
- Species categorization

### ✅ **Modeling**
- Batch modeling for robustness
- Factor optimization
- Convergence criteria
- Model selection based on Q values

### ✅ **Validation**
- Comprehensive diagnostic plots
- Error estimation (when not skipped)
- Residual analysis
- Factor interpretation tools

## Interpretation Guide

### Key Plots for Source Identification

1. **Factor Profiles**: Shows species concentrations in each factor
2. **Factor Fingerprints**: Normalized factor contributions
3. **G-Space Plot**: Factor correlation analysis
4. **Observed vs. Predicted**: Model performance validation

### Quality Indicators
- **Q(robust) < Q(true)**: Good model fit
- **Low residuals**: Accurate predictions
- **Distinct factor profiles**: Clear source separation

## Troubleshooting

### Common Issues

1. **ESAT Installation Fails**
   ```bash
   # Try alternative installation
   pip install --upgrade pip setuptools wheel
   pip install git+https://github.com/quanted/esat.git
   ```

2. **Memory Issues with Large Datasets**
   - Use date range filtering
   - Add `--skip-errors` flag
   - Reduce number of models with `--models 10`

3. **Poor Factor Separation**
   - Try different number of factors
   - Check data quality and completeness
   - Review species selection criteria

### Performance Tips

- Use `--skip-errors` for initial analysis (much faster)
- Filter to specific seasons or periods of interest
- Start with fewer models (10-15) for testing
- Use SSD storage for better I/O performance

## Example Analysis Session

```bash
# 1. Quick exploratory analysis
python pmf_source_apportionment.py MMF1 \
    --start-date 2022-06-01 --end-date 2022-08-31 \
    --skip-errors --models 10

# 2. Full analysis with error estimation
python pmf_source_apportionment.py MMF1 \
    --start-date 2022-01-01 --end-date 2022-12-31 \
    --factors 5 --models 25

# 3. Compare stations
python pmf_source_apportionment.py MMF2 \
    --start-date 2022-01-01 --end-date 2022-12-31 \
    --output-dir pmf_mmf2_results
```

## Contact & Support

This script integrates with the existing MMF data analysis infrastructure:
- Uses `analyze_parquet_data.py` for data loading
- Works with processed parquet files
- Maintains unit information from Excel sources

For issues or questions about the PMF analysis, refer to the EPA PMF 5.0 User Guide for theoretical background and interpretation guidelines.
