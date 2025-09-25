# Environmental Monitoring and PMF Analysis

This repository contains comprehensive Python tooling for processing and analyzing environmental monitoring data from the Walleys MMF stations. It includes:

- **Data preparation and processing**: Robust parquet processing with 30-minute timebase aggregation
- **EPA PMF 5.0 compliance**: Full implementation of EPA uncertainty methods and S/N categorization
- **Dual-mode architecture**: Legacy mode (default) and EPA mode for backwards compatibility
- **Enhanced uncertainty handling**: Timebase averaging with 1/√n scaling for improved sensitivity
- **Comprehensive dashboards**: 16-plot interactive HTML dashboards with complete provenance
- **BTEX VOC integration**: Optional 30‑minute BTEX data integration for MMF2/MMF9
- **Advanced source apportionment**: PMF using ESAT with optimization, validation, and diagnostics
- **Site survey and validation tools**: Complete data coverage analysis

Data files are not included in this repository by design. The .gitignore excludes large/raw data and generated outputs.

## Key Scripts and Modules

### **Core Analysis Scripts**
- **pmf_source_apportionment_fixed.py** — Complete EPA PMF 5.0 implementation with S/N categorization, dual uncertainty modes, 16-plot dashboards, and CLI reproducibility
- **process_mmf_fixed.py** — Advanced parquet processing with 30-minute timebase aggregation, metadata propagation, and uncertainty scaling support
- **epa_uncertainty.py** — EPA PMF 5.0 uncertainty calculator with configurable EF/MDL tables and 1/√n aggregation scaling
- **snr_categorization.py** — EPA S/N-based feature categorization (strong/weak/bad) with comprehensive diagnostics

### **Data Integration and Utilities**
- **mmf_config.py** — Centralized configuration with corrected station mappings and file paths
- **integrate_btex_data.py** — BTEX VOC data integration with exact timestamp matching (no interpolation)
- **plot_mmf_data.py** — Multi‑panel plotting with statistical overlays and data quality indicators
- **verify_uncertainty_scaling.py** — Validation script for timebase averaging uncertainty improvements
- **legacy_methods_analysis.md** — Comprehensive documentation of legacy vs EPA methods

### **Survey and Validation**
- **mmf_site_survey.py** — Station coverage analysis with date ranges and species availability
- **mmf_data_validation.py** — Data validation against official specifications
- **weekly_pmf_analysis.py** — Batch processing for weekly PMF analysis campaigns

### **Generated Outputs**
All outputs are written to your chosen directory with comprehensive provenance:
- **Interactive HTML dashboards** (16 plots + metadata)
- **Diagnostic CSV files** (S/N metrics, species categories, uncertainties, concentrations)
- **Analysis reports** (Markdown summaries with key findings)
- **Plot collections** (PNG exports for all visualizations)
- **Sankey diagrams** (Interactive HTML + static PNG versions)

## Environment Requirements

### **System Requirements**
- **Python**: 3.9–3.11 (recommend 3.10+ for optimal performance)
- **OS**: Windows (tested), Linux/macOS compatible with minor path adjustments
- **Memory**: 4GB+ RAM recommended for large datasets (30-day analyses)
- **Storage**: ~100MB per analysis output directory

### **Dependencies**

**Core packages:**
```bash
numpy pandas pyarrow matplotlib seaborn scikit-learn
```

**Interactive plotting:**
```bash
plotly kaleido  # For Sankey diagrams and interactive dashboards
```

**PMF analysis:**
```bash
git+https://github.com/quanted/esat.git  # EPA ESAT library
```

**Optional (PDF generation):**
```bash
wkhtmltopdf  # For PDF dashboard exports (alternative: Chrome headless)
```

### **Quick Setup**

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/macOS)  
source .venv/bin/activate

# Install core dependencies
pip install --upgrade pip
pip install numpy pandas pyarrow matplotlib seaborn scikit-learn
pip install plotly kaleido
pip install git+https://github.com/quanted/esat.git
```

### **Verified Compatibility**
- **ESAT**: Current GitHub version (BatchSA with parallel processing)
- **Kaleido**: 0.2.1+ (Sankey diagram PNG export)
- **Plotly**: 5.0+ (Interactive dashboard components)

## Data locations and configuration

Place corrected parquet files under mmf_parquet_final/ and (optionally) corrected raw under mmf_data_corrected/. Use mmf_config.py helpers to resolve paths and station mappings consistently.

No data files are committed to the repository; adjust paths or mmf_config.py if your layout differs.

## Usage Examples

### **1. Data Processing with Timebase Aggregation**

```bash
# Process raw Excel files to 30-minute aggregated parquet
python process_mmf_fixed.py --station MMF2 \\
  --timebase 30min --aggregate mean --min-valid-subsamples 2 \\
  --include-voc --output-dir mmf_test_30min
```

### **2. PMF Source Apportionment (EPA Mode)**

**Modern flexible approach:**
```bash
# EPA PMF 5.0 with S/N categorization (recommended)
python pmf_source_apportionment_fixed.py \\
  --data-dir "mmf_test_30min" \\
  --patterns "*mmf2*.parquet" \\
  --start-date 2023-09-01 --end-date 2023-09-30 \\
  --output-dir "pmf_results_epa" \\
  --uncertainty-mode epa \\
  --snr-enable \\
  --write-diagnostics
```

**Legacy station-based approach:**
```bash
# Traditional method (backwards compatible)
python pmf_source_apportionment_fixed.py MMF2 \\
  --start-date 2023-09-01 --end-date 2023-09-14 \\
  --factors 4 --models 20 --max-workers 4 \\
  --output-dir pmf_results_legacy --run-pca --create-pdf
```

### **3. Advanced PMF Options**

**Comprehensive analysis with all features:**
```bash
python pmf_source_apportionment_fixed.py \\
  --data-dir "mmf_test_30min" \\
  --patterns "*mmf9*.parquet" \\
  --start-date 2023-09-01 --end-date 2023-09-30 \\
  --output-dir "pmf_comprehensive" \\
  --uncertainty-mode epa \\
  --snr-enable \\
  --snr-weak-threshold 2.5 \\
  --factors 6 --models 100 \\
  --run-pca --create-pdf \\
  --write-diagnostics
```

**Compare uncertainty methods:**
```bash
# Legacy uncertainty calculation
python pmf_source_apportionment_fixed.py --data-dir mmf_data --patterns "*mmf2*" \\
  --uncertainty-mode legacy --output-dir results_legacy

# EPA uncertainty calculation  
python pmf_source_apportionment_fixed.py --data-dir mmf_data --patterns "*mmf2*" \\
  --uncertainty-mode epa --output-dir results_epa
```

### **4. Key CLI Parameters**

#### **Data Selection:**
- `--data-dir DIR` + `--patterns "*.parquet"` (flexible approach)
- `station` (MMF1|MMF2|MMF6|MMF9|Maries_Way) (legacy approach)
- `--start-date`, `--end-date` (YYYY-MM-DD format)

#### **EPA S/N Categorization:**
- `--snr-enable` (enable EPA S/N-based feature categorization)
- `--snr-weak-threshold 2.0` (S/N threshold for weak categorization)
- `--snr-bad-threshold 0.2` (S/N threshold for bad categorization, species excluded)
- `--exclude-bad` (exclude bad species from analysis)

#### **Uncertainty Calculation:**
- `--uncertainty-mode legacy|epa` (calculation method)
- `--uncertainty-ef-mdl FILE.csv` (custom EF/MDL table)
- `--legacy-min-u 0.1` (minimum uncertainty clamping in legacy mode)
- `--uncertainty-bdl-policy five-sixth-mdl|half-mdl` (BDL uncertainty policy)

#### **PMF Model Configuration:**
- `--factors N` (exact factor count, skips optimization)
- `--max-factors N` (upper bound for optimization)
- `--models M` (BatchSA model count, default: 20)
- `--max-workers K` (parallel processes)

#### **Output and Analysis:**
- `--run-pca` (add PCA comparison plots)
- `--create-pdf` (generate PDF dashboard)
- `--write-diagnostics` (save S/N metrics, categories, uncertainties)
- `--remove-voc` (exclude VOC species if present)

### **5. Data Validation and Diagnostics**

```bash
# Verify uncertainty scaling implementation
python verify_uncertainty_scaling.py

# Comprehensive site survey
python mmf_site_survey.py --detailed --export-csv --out survey_results

# Data validation
python mmf_data_validation.py --survey survey_results/mmf_site_summary.csv
```

### **6. BTEX VOC Integration**

```bash
# Integrate 30-minute BTEX data (MMF2/MMF9 only)
python integrate_btex_data.py \\
  --excel "mmf_data_corrected/BTEX/BTEX data for UKHSA.xlsx" \\
  --outdir mmf_parquet_final
```

*Note: Integration aligns 30‑minute VOC timestamps to existing grids by exact match; no interpolation. Units stored in parquet metadata.*

### **7. Plotting and Visualization**

```bash
# Multi-panel data visualization
python plot_mmf_data.py --station MMF2 --start 2024-01-01 --end 2024-01-31 --out plots/MMF2_jan2024
```

## Dashboard Features

### **Interactive HTML Dashboards (16 Plots + Metadata)**
- **Factor Analysis**: Profiles, contributions, temporal patterns, optimization diagnostics
- **Data Quality**: S/N categorization, uncertainty analysis, BDL/missing patterns
- **Environmental Context**: Wind roses, temperature/pressure correlations, seasonal patterns
- **Model Validation**: Residual analysis, correlation matrices, diagnostic scatterplots
- **Flow Visualization**: Interactive Sankey diagrams (HTML + PNG via Kaleido)
- **Policy Transparency**: EPA vs legacy method explanations, CLI parameter records

### **Complete Provenance and Reproducibility**
- **CLI Command Reconstruction**: Exact parameters used for analysis reproduction
- **Data Processing History**: Timebase aggregation, uncertainty scaling, species categorization
- **Model Selection Rationale**: Q/DoF progression, EPA quality guidelines
- **Scientific Validation**: S/N metrics, categorization reasoning, uncertainty improvements

## Current Implementation Status

### **✅ Completed Features (EPA PMF 5.0 Compliant)**
- **Dual uncertainty modes**: Legacy (backwards compatible) and EPA (PMF 5.0 compliant)
- **S/N categorization**: Strong/weak/bad species classification with comprehensive diagnostics
- **Timebase aggregation**: 30-minute averaging with 1/√n uncertainty scaling (29-59% improvements)
- **Enhanced dashboards**: 16-plot HTML dashboards with complete transparency
- **Advanced PMF**: Factor optimization, BatchSA parallel processing, Q/DoF EPA guidelines
- **Data validation**: BDL/missing handling, uncertainty propagation, scientific integrity

### **🔬 Advanced Features Ready for Use**
- **A/B validation**: Compare legacy vs EPA methods side-by-side
- **Custom EF/MDL tables**: CSV-configurable detection limits and error fractions
- **Flexible data loading**: Pattern-based parquet discovery vs hardcoded station paths
- **Comprehensive diagnostics**: Complete CSV exports for external analysis
- **PDF generation**: Dashboard export via Chrome headless or wkhtmltopdf

## Future Enhancements (Non-Breaking)

### **Planned Additions**
- **Bootstrap uncertainty**: Factor stability and rotational ambiguity quantification
- **Constrained PMF**: Anchored rotations using ESAT's ConstrainedModel
- **Batch processing**: Multi-station, multi-period campaign analysis
- **Advanced validation**: Cross-validation, sensitivity analysis, model diagnostics

*All future features will be opt-in via CLI flags and will not change existing default behavior.*

## Contributing

- Please open an issue or pull request with a clear description of changes.
- Do not include data files in PRs; the .gitignore excludes large data and outputs by default.

## License

If you need a license added (e.g., MIT), please specify and we’ll include it in a follow‑up PR.
