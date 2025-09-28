#!/usr/bin/env python3
"""
Sweep CH4 species uncertainty multipliers and quantify suppression.

This script runs the source apportionment pipeline multiple times over the same
period, with different CH4 uncertainty multipliers (k), to verify that CH4 is
progressively downweighted despite its large absolute magnitude.

For each k, it:
  - Prepares data with --scale-units (recommended),
  - Applies species_weight CH4=k (if k != 1),
  - Runs a (small) PMF analysis to obtain W (contributions) and H (profiles),
  - Computes residuals R = V - W·H and the species-wise Q share:
        Q_species[j] = sum_i ((R_ij / U_ij)**2)
        CH4_share = Q_species[CH4] / sum_j Q_species[j]
  - Prints CH4 Q share, and CH4 uncertainty summary per k.

Notes:
- This is intended for quick comparative diagnostics; it uses a small number of
  models for speed. Increase --models for more robust fits.
- Ensure CH4 is present in the selected period and species set.
- Unit scaling is enabled by default to avoid MDL/BDL misclassification.
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from pmf_source_app import MMFPMFAnalyzer


def _corr_stats(mat: np.ndarray, rowvar_factors: bool) -> dict:
    """Compute factor-factor correlation stats.
    - If rowvar_factors is True, rows are factors (H shape: n_factors x n_species)
    - If False, columns are factors (W shape: n_samples x n_factors)
    Returns max and mean absolute off-diagonal correlation and full matrix.
    """
    try:
        if rowvar_factors:
            corr = np.corrcoef(mat)
        else:
            # columns are factors
            corr = np.corrcoef(mat, rowvar=False)
        # Handle degenerate cases
        if corr.ndim != 2 or corr.shape[0] != corr.shape[1]:
            return {
                'corr': corr,
                'max_abs_off': np.nan,
                'mean_abs_off': np.nan,
            }
        n = corr.shape[0]
        # Off-diagonal absolute values
        off = np.abs(corr[np.triu_indices(n, k=1)])
        return {
            'corr': corr,
            'max_abs_off': float(np.nanmax(off)) if off.size else np.nan,
            'mean_abs_off': float(np.nanmean(off)) if off.size else np.nan,
        }
    except Exception:
        return {
            'corr': np.array([[np.nan]]),
            'max_abs_off': np.nan,
            'mean_abs_off': np.nan,
        }


def _effective_rank_from_array(X: np.ndarray, rel_thresh: float = 0.01) -> int:
    try:
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        if X.size == 0:
            return 0
        s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
        if s.size == 0:
            return 0
        max_s = s[0] if s[0] > 0 else (np.max(s) if np.max(s) > 0 else 1.0)
        return int(np.sum(s >= rel_thresh * max_s))
    except Exception:
        return 0


def run_once(
    station: str,
    data_dir: str,
    patterns: str,
    start_date: str,
    end_date: str,
    out_dir: Path,
    factors: int,
    models: int,
    uncertainty_mode: str,
    scale_units: bool,
    species_weight_list: List[str] | None,
    exclude_species_list: List[str] | None,
    species: str,
    weight_aware_init: bool | None = True,
    optimize_factors: bool = False,
    max_factors: int = 8,
):
    analyzer = MMFPMFAnalyzer(
        station=station,
        data_dir=data_dir,
        patterns=patterns,
        start_date=start_date,
        end_date=end_date,
        output_dir=str(out_dir),
        uncertainty_mode=uncertainty_mode,
        scale_units=scale_units,
        species_weight=species_weight_list,
        exclude_species=exclude_species_list,
        weight_aware_init=weight_aware_init,
        snr_enable=False,
    )

    analyzer.load_mmf_data()
    analyzer.prepare_pmf_data()

    # Configure PMF run size
    if optimize_factors:
        analyzer.factors = 4  # Default fallback 
        analyzer.user_specified_factors = False  # Let optimizer run
        analyzer.max_factors = max_factors
    else:
        analyzer.factors = int(factors)
        analyzer.user_specified_factors = True  # ensure optimizer does not override
    analyzer.models = int(models)

    ok = analyzer.run_pmf_analysis()
    if not ok or analyzer.best_model is None:
        return {
            'ok': False,
            'reason': 'PMF run failed or best_model is None'
        }

    # Paths for V and U used by run_pmf_analysis
    conc_file = Path(analyzer.output_dir) / f"{analyzer.filename_prefix}_concentrations.csv"
    unc_file  = Path(analyzer.output_dir) / f"{analyzer.filename_prefix}_uncertainties.csv"

    conc_df = pd.read_csv(conc_file, index_col=0)
    unc_df  = pd.read_csv(unc_file, index_col=0)

    V = conc_df.values
    U = unc_df.values

    # ESAT model matrices: W (n_samples x n_factors), H (n_factors x n_species)
    W = getattr(analyzer.best_model, 'W', None)
    H = getattr(analyzer.best_model, 'H', None)
    if W is None or H is None:
        return {
            'ok': False,
            'reason': 'Model missing W/H matrices'
        }

    V_hat = W @ H
    R = V - V_hat

    # Guard against zeros in U
    U_safe = np.where(U > 0, U, np.nan)
    norm_resid = (R / U_safe) ** 2

    # Sum over samples to get per-species Q share
    Q_species = np.nansum(norm_resid, axis=0)
    Q_total = float(np.nansum(Q_species)) if np.isfinite(np.nansum(Q_species)) else np.nan

    # Factor-factor correlations
    w_stats = _corr_stats(W, rowvar_factors=False)
    h_stats = _corr_stats(H, rowvar_factors=True)
    
    species_names = conc_df.columns.tolist()
    
    # H correlation excluding target species (if present)
    h_stats_no_species = {'max_abs_off': np.nan, 'mean_abs_off': np.nan}
    if species in species_names:
        species_idx = species_names.index(species)
        # Remove the species column from H (n_factors x n_species)
        H_no_species = np.delete(H, species_idx, axis=1)
        if H_no_species.shape[1] > 1:  # Need at least 2 species for correlation
            h_stats_no_species = _corr_stats(H_no_species, rowvar_factors=True)

    # Save correlation matrices to CSV for inspection
    try:
        pd.DataFrame(w_stats['corr']).to_csv(out_dir / 'factor_corr_W.csv', index=False)
        pd.DataFrame(h_stats['corr']).to_csv(out_dir / 'factor_corr_H.csv', index=False)
    except Exception:
        pass

    # Species-species correlation heatmap (V)
    try:
        species_corr = conc_df.corr()
        species_corr.to_csv(out_dir / 'species_corr_V.csv')
    except Exception:
        pass

    # Effective rank of V (with and without the target species)
    eff_rank_with = _effective_rank_from_array(V)
    if species in conc_df.columns:
        V_no = conc_df.drop(columns=[species]).values
        eff_rank_without = _effective_rank_from_array(V_no)
        try:
            species_corr_no = conc_df.drop(columns=[species]).corr()
            species_corr_no.to_csv(out_dir / f'species_corr_V_no_{species}.csv')
        except Exception:
            pass
    else:
        eff_rank_without = eff_rank_with

    return {
        'ok': True,
        'species_names': species_names,
        'Q_species': Q_species,
        'Q_total': Q_total,
        'Qrobust': float(getattr(analyzer.best_model, 'Qrobust', np.nan)),
        'Qtrue': float(getattr(analyzer.best_model, 'Qtrue', np.nan)),
        'U_df': unc_df,
        'V_df': conc_df,
        'prefix': analyzer.filename_prefix,
        'output_dir': str(analyzer.output_dir),
        'W_max_abs_corr_off': w_stats['max_abs_off'],
        'W_mean_abs_corr_off': w_stats['mean_abs_off'],
        'H_max_abs_corr_off': h_stats['max_abs_off'],
        'H_mean_abs_corr_off': h_stats['mean_abs_off'],
        'H_max_abs_corr_off_no_species': h_stats_no_species['max_abs_off'],
        'H_mean_abs_corr_off_no_species': h_stats_no_species['mean_abs_off'],
        'eff_rank_V': eff_rank_with,
        'eff_rank_V_no_species': eff_rank_without,
        'n_factors': int(getattr(analyzer, 'factors', factors)),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Sweep CH4 uncertainty multipliers and report Q share suppression',
        allow_abbrev=False,
    )
    parser.add_argument('--station', choices=['MMF1','MMF2','MMF6','MMF9','Maries_Way'], default='MMF1')
    parser.add_argument('--data-dir', default=None, help='Directory with parquet files (flex mode)')
    parser.add_argument('--patterns', default=None, help='Comma-separated parquet glob patterns (flex mode)')
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--end-date', required=True)
    parser.add_argument('--species', default='CH4')
    parser.add_argument('--factors', type=int, default=4)
    parser.add_argument('--models', type=int, default=1, help='Number of PMF models per k (default: 1 for speed)')
    parser.add_argument('--uncertainty-mode', choices=['legacy','epa'], default='legacy')
    parser.add_argument('--no-scale-units', action='store_true', help='Disable unit scaling (not recommended)')
    parser.add_argument('--weights', default='1,3,5,10,20,30,50,100',
                        help='Comma-separated list of CH4 uncertainty multipliers to test')
    parser.add_argument('--base-output-dir', default='sweep_ch4_weight_outputs')
    parser.add_argument('--no-weight-aware-init', action='store_true', 
                        help='Disable weight-aware initialization (enabled by default for weighted species)')
    parser.add_argument('--optimize-factors', action='store_true',
                        help='Enable factor number optimization (tests 2 to max-factors, slower)')
    parser.add_argument('--max-factors', type=int, default=8,
                        help='Maximum factors to test during optimization (default: 8)')

    args = parser.parse_args()

    species = args.species
    weights = [w.strip() for w in args.weights.split(',') if w.strip()]
    weights = [float(w) for w in weights]

    base_out = Path(args.base_output_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    print(f"Species: {species}")
    print(f"Weights (k): {weights}")
    print(f"Unit scaling: {'ON' if not args.no_scale_units else 'OFF'}")

    rows = []

    # Also append an explicit exclude scenario at the end
    weight_scenarios = [('exclude', None)] + [(k, k) for k in weights]

    for label, k in weight_scenarios:
        # Build run-specific output dir
        if label == 'exclude':
            out_dir = base_out / 'exclude_species'
        else:
            out_dir = base_out / f"k_{int(k) if float(k).is_integer() else str(k).replace('.', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build species weight list and exclusion list
        if label == 'exclude':
            sw = None
            ex = [species]
        else:
            sw = None if (k == 1 or k is None) else [f"{species}={k}"]
            ex = None

        print("\n" + "="*70)
        if label == 'exclude':
            print(f"Running EXCLUDE scenario (exclude {species}) ...")
        else:
            print(f"Running k={k} ...")
        print("="*70)

        result = run_once(
            station=args.station,
            data_dir=args.data_dir,
            patterns=args.patterns,
            start_date=args.start_date,
            end_date=args.end_date,
            out_dir=out_dir,
            factors=args.factors,
            models=args.models,
            uncertainty_mode=args.uncertainty_mode,
            scale_units=(not args.no_scale_units),
            species_weight_list=sw,
            exclude_species_list=ex,
            species=species,
            weight_aware_init=(not args.no_weight_aware_init) if sw else False,
            optimize_factors=args.optimize_factors,
            max_factors=args.max_factors,
        )

        if not result.get('ok', False):
            reason = result.get('reason','unknown')
            print(f"  ❌ Failed: {reason}")
            rows.append({'scenario': label if label=='exclude' else k, 'k': k, 'ok': False, 'reason': reason})
            continue

        names = result['species_names']
        Q_species = result['Q_species']
        Q_total = result['Q_total']

        # Compute CH4 metrics if present; if excluded, use NaN placeholders
        ch4_present = True
        try:
            j = names.index(species)
        except ValueError:
            ch4_present = False
            print(f"  ℹ️ Species {species} not present (e.g., excluded scenario) — CH4-specific metrics will be reported as NaN")

        if ch4_present:
            ch4_Q = float(Q_species[j])
            ch4_share = float(ch4_Q / Q_total) if (Q_total and np.isfinite(Q_total) and Q_total > 0) else np.nan
            # Also report CH4 uncertainty summary
            U = result['U_df'][species]
            U_min = float(np.nanmin(U.values))
            U_med = float(np.nanmedian(U.values))
            U_max = float(np.nanmax(U.values))
        else:
            ch4_Q = np.nan
            ch4_share = np.nan
            U_min = np.nan
            U_med = np.nan
            U_max = np.nan

        print(f"  Qtrue={result['Qtrue']:.3f}  Qrobust={result['Qrobust']:.3f}")
        print(f"  CH4 Q share: {ch4_share:.4f}  (Q_ch4={ch4_Q:.3f}, Q_total={Q_total:.3f})")
        print(f"  CH4 U summary: min={U_min:.6g}, median={U_med:.6g}, max={U_max:.6g}")
        print(f"  Factor correlation (W time series): max|r_off|={result['W_max_abs_corr_off']:.3f}, mean|r_off|={result['W_mean_abs_corr_off']:.3f}")
        print(f"  Factor correlation (H profiles):   max|r_off|={result['H_max_abs_corr_off']:.3f}, mean|r_off|={result['H_mean_abs_corr_off']:.3f}")
        print(f"  Factor correlation (H excl. {species}):  max|r_off|={result['H_max_abs_corr_off_no_species']:.3f}, mean|r_off|={result['H_mean_abs_corr_off_no_species']:.3f}")
        print(f"  Effective rank: V={result['eff_rank_V']}, V_no_{species}={result['eff_rank_V_no_species']}")
        print(f"  Factors used: {result['n_factors']}")

        rows.append({
            'scenario': label if label=='exclude' else k,
            'k': k,
            'ok': True,
            'Qtrue': result['Qtrue'],
            'Qrobust': result['Qrobust'],
            'Q_total': Q_total,
            'Q_ch4': ch4_Q,
            'Q_share_ch4': ch4_share,
            'U_min_ch4': U_min,
            'U_med_ch4': U_med,
            'U_max_ch4': U_max,
            'W_max_abs_corr_off': result['W_max_abs_corr_off'],
            'W_mean_abs_corr_off': result['W_mean_abs_corr_off'],
            'H_max_abs_corr_off': result['H_max_abs_corr_off'],
            'H_mean_abs_corr_off': result['H_mean_abs_corr_off'],
            'H_max_abs_corr_off_no_species': result['H_max_abs_corr_off_no_species'],
            'H_mean_abs_corr_off_no_species': result['H_mean_abs_corr_off_no_species'],
            'eff_rank_V': result['eff_rank_V'],
            'eff_rank_V_no_species': result['eff_rank_V_no_species'],
            'n_factors': result['n_factors'],
            'output_dir': result['output_dir'],
            'prefix': result['prefix'],
        })

    # Save summary CSV
    summary_df = pd.DataFrame(rows)
    summary_csv = base_out / 'sweep_summary.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")

    # Pretty print brief table
    print("\nSummary (scenario, Q_share_ch4, U_med_ch4, W_max|r_off|, H_max|r_off|, rank_V, rank_V_no_species):")
    for r in rows:
        label = 'exclude' if r.get('scenario') == 'exclude' else f"k={r.get('k')}"
        if r.get('ok'):
            print(
                f"  {label}: Q_share_ch4={r['Q_share_ch4']:.4f}, "
                f"U_med_ch4={r['U_med_ch4']:.6g}, "
                f"W_max|r_off|={r['W_max_abs_corr_off']:.3f}, H_max|r_off|={r['H_max_abs_corr_off']:.3f}, "
                f"rank_V={r['eff_rank_V']}, rank_V_no_{species}={r['eff_rank_V_no_species']}"
            )
        else:
            print(f"  {label}: FAILED ({r.get('reason','')})")

    print("\nDone.")


if __name__ == '__main__':
    sys.exit(main())
