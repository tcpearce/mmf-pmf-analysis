# Environmental Monitoring and PMF Analysis

This repository contains Python tooling for processing and analyzing environmental monitoring data from the Walleys MMF stations. It includes:

- Data preparation and correction scripts (consistent MMF mappings, parquet regeneration)
- Optional integration of 30‑minute BTEX VOC data (MMF2, MMF9)
- Plotting utilities for exploratory analysis
- PMF (Positive Matrix Factorization) source apportionment using the ESAT library
- Site survey and validation tools

Data files are not included in this repository by design. The .gitignore excludes large/raw data and generated outputs.

## Key directories and scripts

- mmf_config.py — Centralized paths and helpers for corrected data locations (mmf_parquet_final, mmf_data_corrected)
- process_mmf_corrected.py — Reuses robust existing parsing to regenerate corrected parquet with station metadata
- integrate_btex_data.py — Integrates 30‑minute BTEX VOC channels into MMF2/MMF9 parquet files (no interpolation)
- plot_mmf_data.py — Multi‑panel plotting with numeric‑safe stats overlays
- pmf_source_apportionment_fixed.py — ESAT‑based PMF analysis; saves dashboards, reports, and images
- mmf_site_survey.py — Reports station coverage, date ranges, and columns (with optional VOC details)
- mmf_data_validation.py — Cross‑checks data availability vs official statements

Outputs (dashboards, images, CSV summaries) are written under your chosen output directory (e.g., pmf_results_esat/). These folders are ignored by git.

## Environment

- Python: 3.9–3.11 (recommend 3.10+)
- OS: Windows (tested), Linux/macOS should work with minor path tweaks

Recommended packages:

- Core: numpy, pandas, pyarrow, matplotlib, seaborn, scikit‑learn
- Plotting: plotly, kaleido
- ESAT (PMF): install from GitHub

Example setup:

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install numpy pandas pyarrow matplotlib seaborn scikit-learn plotly kaleido
pip install git+https://github.com/quanted/esat.git
```

Note: Depending on ESAT version, attributes and APIs may differ. The fixed PMF script targets current ESAT (as tested in this project).

## Data locations and configuration

Place corrected parquet files under mmf_parquet_final/ and (optionally) corrected raw under mmf_data_corrected/. Use mmf_config.py helpers to resolve paths and station mappings consistently.

No data files are committed to the repository; adjust paths or mmf_config.py if your layout differs.

## Usage examples

1) Plotting

```bash
python plot_mmf_data.py --station MMF2 --start 2024-01-01 --end 2024-01-31 --out plots/MMF2_jan2024
```

2) PMF (ESAT) source apportionment (fixed script)

```bash
python pmf_source_apportionment_fixed.py MMF2 \
  --start-date 2023-09-01 --end-date 2023-09-14 \
  --factors 4 --models 5 --max-workers 4 \
  --output-dir pmf_results_esat --run-pca --create-pdf
```

Common flags in pmf_source_apportionment_fixed.py:

- station (positional): MMF1 | MMF2 | MMF6 | MMF9 | Maries_Way
- --start-date, --end-date (YYYY-MM-DD)
- --factors N (use exactly N factors; skips optimization)
- --max-factors N (upper bound for optimization when --factors is not provided)
- --models M (BatchSA model count)
- --max-workers K (parallel processes)
- --run-pca (optional PCA comparison and plots)
- --create-pdf (render dashboard to PDF using headless Chrome/Edge)
- --remove-voc (exclude VOC species from PMF analysis if present)

3) Integrate BTEX VOC data (MMF2/MMF9 only)

```bash
python integrate_btex_data.py \
  --excel "mmf_data_corrected/BTEX/BTEX data for UKHSA.xlsx" \
  --outdir mmf_parquet_final
```

(Integration aligns 30‑minute VOC timestamps to existing 5‑minute grids by exact match; no interpolation. Units are stored in parquet metadata.)

4) Site survey

```bash
python mmf_site_survey.py --detailed --export-csv --out mmf_survey_results
```

5) Validation against statements

```bash
python mmf_data_validation.py --survey mmf_survey_results/mmf_site_summary.csv
```

## Notes on ESAT and dashboards

- ESAT PMF outputs: Q(true), Q(robust), W (contributions), H (profiles) are used to generate diagnostic plots, temporal and meteorological analyses, and Sankey diagrams (HTML + PNG via Kaleido).
- If PNG export for Sankey fails in your environment, the script produces an HTML version and a static fallback image.
- PDF conversion of dashboards relies on headless Chrome/Edge when wkhtmltopdf/pdfkit are not available.

## Roadmap (non‑breaking additions planned)

- BatchSA spread analysis across many seeds with factor matching and stability summaries
- Bootstrap and BS‑DISP integration to quantify solution and rotational ambiguity
- Constrained/anchored rotations using ESAT’s ConstrainedModel (no EPA Fpeak control yet)
- Optional species category tuning (strong/weak/bad) via DataHandler

These will be exposed as optional CLI flags once implemented and will not change existing default behavior.

## Contributing

- Please open an issue or pull request with a clear description of changes.
- Do not include data files in PRs; the .gitignore excludes large data and outputs by default.

## License

If you need a license added (e.g., MIT), please specify and we’ll include it in a follow‑up PR.
