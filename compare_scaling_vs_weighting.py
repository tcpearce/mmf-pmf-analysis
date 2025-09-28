#!/usr/bin/env python3
"""
Compare the effect of unit scaling vs. large species uncertainty weighting on CH4.

This script runs two scenarios on the same date range using the existing
MMFPMFAnalyzer:

Scenario A (Units scaled):
    - scale_units=True (converts mg/m3 → μg/m3 before uncertainty/BDL logic)
    - no species uncertainty weighting

Scenario B (Weight-only, no unit scaling):
    - scale_units=False (leave CH4 in mg/m3)
    - species_weight=['CH4=1000'] (multiply uncertainties by 1000 after they are computed)

For each scenario, the script prints for CH4:
    - BDL count (if available via analyzer._bdl_mask)
    - V (concentration) min / median / max
    - U (uncertainty) min / median / max
    - Units recorded for CH4

Notes:
- The default uncertainty_mode is legacy so that BDL masks are available.
- This does not run PMF model training; it only prepares matrices.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Local import from repository
from pmf_source_app import MMFPMFAnalyzer

def _safe_stat(s: pd.Series, func: str):
    try:
        if func == 'min':
            return float(np.nanmin(s.values))
        if func == 'median':
            return float(np.nanmedian(s.values))
        if func == 'max':
            return float(np.nanmax(s.values))
    except Exception:
        return float('nan')


def _compute_metrics(analyzer: MMFPMFAnalyzer, species: str):
    metrics = {}

    if species not in analyzer.concentration_data.columns:
        metrics['present'] = False
        return metrics

    metrics['present'] = True
    v = analyzer.concentration_data[species]
    u = analyzer.uncertainty_data[species]

    metrics['total_rows'] = int(len(v))

    # BDL count if available (legacy path creates _bdl_mask)
    bdl_count = None
    if hasattr(analyzer, '_bdl_mask') and species in analyzer._bdl_mask.columns:
        try:
            bdl_count = int(analyzer._bdl_mask[species].sum())
        except Exception:
            bdl_count = None
    metrics['bdl_count'] = bdl_count

    # Concentration stats
    metrics['V_min'] = _safe_stat(v, 'min')
    metrics['V_median'] = _safe_stat(v, 'median')
    metrics['V_max'] = _safe_stat(v, 'max')

    # Uncertainty stats
    metrics['U_min'] = _safe_stat(u, 'min')
    metrics['U_median'] = _safe_stat(u, 'median')
    metrics['U_max'] = _safe_stat(u, 'max')

    # Units (as recorded during load/prepare)
    try:
        metrics['units'] = analyzer.units.get(species, 'unknown')
    except Exception:
        metrics['units'] = 'unknown'

    return metrics


def _print_metrics(title: str, metrics: dict):
    print("=" * 72)
    print(title)
    print("=" * 72)
    if not metrics.get('present', False):
        print("Species not present in processed data.")
        return
    print(f"Total rows: {metrics['total_rows']}")
    print(f"BDL count:  {metrics['bdl_count'] if metrics['bdl_count'] is not None else 'n/a'}")
    print(f"Units:      {metrics.get('units', 'unknown')}")
    print("-")
    print(f"V (conc):   min={metrics['V_min']:.6g}  median={metrics['V_median']:.6g}  max={metrics['V_max']:.6g}")
    print(f"U (unc):    min={metrics['U_min']:.6g}  median={metrics['U_median']:.6g}  max={metrics['U_max']:.6g}")


def run_scenario(
    station: str,
    data_dir: str,
    patterns: str,
    start_date: str,
    end_date: str,
    output_dir: Path,
    species: str,
    scale_units: bool,
    species_weight_list,
    uncertainty_mode: str,
):
    analyzer = MMFPMFAnalyzer(
        station=station,
        data_dir=data_dir,
        patterns=patterns,
        start_date=start_date,
        end_date=end_date,
        output_dir=str(output_dir),
        remove_voc=False,
        uncertainty_mode=uncertainty_mode,
        scale_units=scale_units,
        species_weight=species_weight_list,
        snr_enable=False,
    )

    analyzer.load_mmf_data()
    analyzer.prepare_pmf_data()

    return analyzer


def main():
    parser = argparse.ArgumentParser(
        description="Compare unit scaling vs. uncertainty weighting for CH4",
        allow_abbrev=False,
    )
    parser.add_argument('--station', choices=['MMF1','MMF2','MMF6','MMF9','Maries_Way'], default='MMF1',
                        help='MMF station to analyze (alternative: use --data-dir and --patterns)')
    parser.add_argument('--data-dir', default=None, help='Directory containing parquet files (flexible mode)')
    parser.add_argument('--patterns', default=None, help='Comma-separated parquet filename patterns (flexible mode)')
    parser.add_argument('--start-date', required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end-date', required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--species', default='CH4', help='Species to check (default: CH4)')
    parser.add_argument('--uncertainty-mode', choices=['legacy','epa'], default='legacy',
                        help='Uncertainty mode for both runs (default: legacy)')
    parser.add_argument('--base-output-dir', default='comparison_outputs', help='Base output directory for both runs')

    args = parser.parse_args()

    base_out = Path(args.base_output_dir)
    out_scale = base_out / 'scenario_scale_units'
    out_weight = base_out / 'scenario_weight_only'
    out_scale.mkdir(parents=True, exist_ok=True)
    out_weight.mkdir(parents=True, exist_ok=True)

    # Scenario A: Scale units, no species weighting
    print("\nRunning Scenario A: scale_units=True, no species weighting")
    analyzer_a = run_scenario(
        station=args.station,
        data_dir=args.data_dir,
        patterns=args.patterns,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=out_scale,
        species=args.species,
        scale_units=True,
        species_weight_list=None,
        uncertainty_mode=args.uncertainty_mode,
    )

    metrics_a = _compute_metrics(analyzer_a, args.species)
    _print_metrics("Scenario A (scale_units=True)", metrics_a)

    # Scenario B: No scaling, CH4 uncertainty x1000
    print("\nRunning Scenario B: scale_units=False, species_weight CH4=1000")
    analyzer_b = run_scenario(
        station=args.station,
        data_dir=args.data_dir,
        patterns=args.patterns,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=out_weight,
        species=args.species,
        scale_units=False,
        species_weight_list=[f"{args.species}=1000"],
        uncertainty_mode=args.uncertainty_mode,
    )

    metrics_b = _compute_metrics(analyzer_b, args.species)
    _print_metrics("Scenario B (scale_units=False, species_weight CH4=1000)", metrics_b)

    # Summary comparison
    print("\n" + "#" * 72)
    print("Summary of differences")
    print("#" * 72)
    if metrics_a.get('present') and metrics_b.get('present'):
        def fmt(v):
            try:
                return f"{v:.6g}"
            except Exception:
                return str(v)
        print(f"BDL count:  A={metrics_a['bdl_count']}  B={metrics_b['bdl_count']}")
        print(f"V median:   A={fmt(metrics_a['V_median'])}  B={fmt(metrics_b['V_median'])}")
        print(f"U median:   A={fmt(metrics_a['U_median'])}  B={fmt(metrics_b['U_median'])}")
        print(f"Units:      A={metrics_a.get('units','unknown')}  B={metrics_b.get('units','unknown')}")
    else:
        print("Species is missing in one of the processed scenarios.")

    print("\nDone.")


if __name__ == '__main__':
    sys.exit(main())
