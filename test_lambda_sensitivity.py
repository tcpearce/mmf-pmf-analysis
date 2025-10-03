#!/usr/bin/env python3
"""
Stage 9: Lambda Sensitivity Analysis Tool

This script performs comprehensive lambda sweep analysis to validate regularization
behavior and find optimal lambda values for target species push-out.

Usage:
    python test_lambda_sensitivity.py --species CH4 --lambda-range 0.1,0.5,1.0,2.0,5.0
    python test_lambda_sensitivity.py --species CH4 --lambda-range auto
    
Key Features:
- Automated lambda sweep across specified ranges
- Push-out effectiveness metrics  
- Factor correlation analysis
- Model fit quality tracking (Q-values)
- Comprehensive diagnostic plots
- CSV results export for further analysis

Mathematical Validation:
- As λ increases, target species factor loadings should decrease
- Model fit quality (Q-values) should remain reasonable
- Factor correlations should decrease (better separation)
- Convergence should be maintained across lambda range
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json

def create_lambda_range(range_type: str, custom_values: str = None) -> List[float]:
    """Create lambda range for sensitivity analysis."""
    
    if range_type == "auto":
        # Automatic range covering multiple scales
        return [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    elif range_type == "fine":
        # Fine-grained analysis around lambda=1
        return [0.1, 0.3, 0.5, 0.7, 1.0, 1.3, 1.5, 2.0, 3.0, 5.0]
    elif range_type == "coarse":
        # Coarse sweep for quick testing
        return [0.5, 1.0, 2.0, 5.0]
    elif range_type == "custom" and custom_values:
        # Parse custom comma-separated values
        try:
            return [float(x.strip()) for x in custom_values.split(',')]
        except ValueError as e:
            raise ValueError(f"Invalid custom lambda values: {e}")
    else:
        raise ValueError(f"Unknown lambda range type: {range_type}")

def run_single_lambda_analysis(species: str, lambda_val: float, base_cmd: str,
                              output_dir: Path, verbose: bool = True) -> Dict:
    """Run PMF analysis with single lambda value and extract metrics."""
    
    import subprocess
    import tempfile
    
    # Create temporary output directory for this lambda
    lambda_dir = output_dir / f"lambda_{lambda_val:.3f}".replace('.', '_')
    lambda_dir.mkdir(exist_ok=True, parents=True)
    
    # Build command with regularization parameters
    cmd_parts = base_cmd.split()
    reg_cmd = cmd_parts + [
        "--reg-species", species,
        "--reg-lambda", str(lambda_val),
        "--reg-template", "zero",
        "--output-dir", str(lambda_dir)
    ]
    
    if verbose:
        print(f"   [RUN] λ = {lambda_val:.3f} -> {lambda_dir.name}")
    
    try:
        # Run PMF analysis
        result = subprocess.run(
            reg_cmd, 
            capture_output=True, 
            text=True, 
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            print(f"   [ERROR] Lambda {lambda_val} failed with return code {result.returncode}")
            if verbose:
                print(f"   STDERR: {result.stderr[-500:]}")  # Last 500 chars
            return None
            
        # Extract metrics from results
        metrics = extract_metrics_from_output(lambda_dir, species, lambda_val, result.stdout)
        
        if verbose and metrics:
            print(f"   [METRICS] Q(robust)={metrics.get('q_robust', 'N/A'):.2f}, "
                  f"Max Loading={metrics.get('max_loading', 'N/A'):.3f}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"   [TIMEOUT] Lambda {lambda_val} timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"   [ERROR] Lambda {lambda_val} failed: {e}")
        return None

def extract_metrics_from_output(output_dir: Path, species: str, lambda_val: float, 
                               stdout: str) -> Dict:
    """Extract key metrics from PMF analysis output."""
    
    metrics = {
        'lambda_value': lambda_val,
        'species_name': species,
        'q_robust': np.nan,
        'q_true': np.nan,
        'max_loading': np.nan,
        'factor_correlation': np.nan,
        'converged': False,
        'bursts_completed': 0
    }
    
    try:
        # Parse Q values from stdout
        import re
        
        # Look for final Q values
        q_matches = re.findall(r'Q\(robust\):\s*([\d.]+)', stdout)
        if q_matches:
            metrics['q_robust'] = float(q_matches[-1])
            
        qtrue_matches = re.findall(r'Q\(true\):\s*([\d.]+)', stdout)
        if qtrue_matches:
            metrics['q_true'] = float(qtrue_matches[-1])
            
        # Look for convergence information
        if "Regularization converged" in stdout:
            metrics['converged'] = True
            
        # Count completed bursts
        burst_matches = re.findall(r'Burst (\d+)/\d+ summary', stdout)
        if burst_matches:
            metrics['bursts_completed'] = len(burst_matches)
        
        # Try to extract factor matrix information if available
        try:
            # Look for saved concentration/uncertainty files
            conc_files = list(output_dir.glob("*_concentrations.csv"))
            if conc_files:
                # Load and analyze factor profiles (this is approximate)
                conc_df = pd.read_csv(conc_files[0])
                if species in conc_df.columns:
                    species_values = conc_df[species].values
                    metrics['max_loading'] = float(np.max(species_values))
                    
                    # Simple correlation analysis
                    other_cols = [col for col in conc_df.columns if col != species and col != 'datetime']
                    if other_cols:
                        correlations = [abs(np.corrcoef(species_values, conc_df[col].values)[0,1]) 
                                      for col in other_cols if not np.isnan(np.corrcoef(species_values, conc_df[col].values)[0,1])]
                        if correlations:
                            metrics['factor_correlation'] = float(np.mean(correlations))
                            
        except Exception as e:
            # Factor analysis failed, use stdout parsing as fallback
            pass
            
    except Exception as e:
        print(f"   [WARN] Metric extraction failed for λ={lambda_val}: {e}")
        
    return metrics

def analyze_lambda_sensitivity_results(results: List[Dict], species: str, 
                                     output_dir: Path) -> Dict:
    """Analyze lambda sensitivity results and generate summary."""
    
    if not results:
        print("[ERROR] No valid results to analyze")
        return {}
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(results)
    df = df.dropna(subset=['q_robust'])  # Remove failed runs
    
    if len(df) == 0:
        print("[ERROR] No successful runs with valid Q values")
        return {}
    
    print(f"\n[ANALYSIS] Lambda Sensitivity Analysis Results for {species}")
    print("=" * 70)
    
    # Find optimal lambda values based on different criteria
    analysis = {}
    
    # Best model fit (lowest Q)
    best_fit_idx = df['q_robust'].idxmin()
    analysis['optimal_lambda_fit'] = df.loc[best_fit_idx, 'lambda_value']
    analysis['best_q_robust'] = df.loc[best_fit_idx, 'q_robust']
    
    # Best push-out (lowest max loading, if available)
    if not df['max_loading'].isna().all():
        best_pushout_idx = df['max_loading'].idxmin()
        analysis['optimal_lambda_pushout'] = df.loc[best_pushout_idx, 'lambda_value']
        analysis['best_max_loading'] = df.loc[best_pushout_idx, 'max_loading']
    
    # Best correlation reduction (lowest factor correlation, if available)
    if not df['factor_correlation'].isna().all():
        best_corr_idx = df['factor_correlation'].idxmin()
        analysis['optimal_lambda_correlation'] = df.loc[best_corr_idx, 'lambda_value']
        analysis['best_correlation'] = df.loc[best_corr_idx, 'factor_correlation']
    
    # Convergence analysis
    converged_count = df['converged'].sum()
    analysis['convergence_rate'] = converged_count / len(df)
    analysis['total_runs'] = len(df)
    analysis['successful_runs'] = len(df)
    
    # Lambda range analysis
    analysis['lambda_range'] = (df['lambda_value'].min(), df['lambda_value'].max())
    analysis['lambda_values_tested'] = df['lambda_value'].tolist()
    
    # Print summary
    print(f"Successful runs: {analysis['successful_runs']} / {analysis['total_runs']}")
    print(f"Convergence rate: {analysis['convergence_rate']:.1%}")
    print(f"Lambda range tested: {analysis['lambda_range'][0]:.3f} - {analysis['lambda_range'][1]:.3f}")
    print(f"Optimal λ (best fit): {analysis['optimal_lambda_fit']:.3f} (Q={analysis['best_q_robust']:.2f})")
    
    if 'optimal_lambda_pushout' in analysis:
        print(f"Optimal λ (push-out): {analysis['optimal_lambda_pushout']:.3f} (max_loading={analysis['best_max_loading']:.3f})")
    
    if 'optimal_lambda_correlation' in analysis:
        print(f"Optimal λ (correlation): {analysis['optimal_lambda_correlation']:.3f} (corr={analysis['best_correlation']:.3f})")
    
    # Save detailed results
    results_file = output_dir / f"{species}_lambda_sensitivity_results.csv"
    df.to_csv(results_file, index=False)
    print(f"\n[SAVE] Detailed results: {results_file}")
    
    # Save analysis summary
    summary_file = output_dir / f"{species}_lambda_sensitivity_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"[SAVE] Analysis summary: {summary_file}")
    
    return analysis

def create_sensitivity_plots(results: List[Dict], species: str, output_dir: Path) -> None:
    """Create comprehensive lambda sensitivity plots."""
    
    if not results:
        return
    
    df = pd.DataFrame(results)
    df = df.dropna(subset=['q_robust'])  # Remove failed runs
    
    if len(df) == 0:
        print("[WARN] No valid data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Lambda Sensitivity Analysis: {species}', fontsize=16, fontweight='bold')
    
    # Plot 1: Q(robust) vs Lambda (model fit quality)
    axes[0, 0].semilogx(df['lambda_value'], df['q_robust'], 'o-', color='blue', linewidth=2, markersize=8)
    axes[0, 0].set_title('Model Fit Quality vs Lambda')
    axes[0, 0].set_xlabel('Lambda (log scale)')
    axes[0, 0].set_ylabel('Q(robust)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Highlight optimal point
    best_idx = df['q_robust'].idxmin()
    axes[0, 0].scatter(df.loc[best_idx, 'lambda_value'], df.loc[best_idx, 'q_robust'], 
                      color='red', s=100, zorder=5, label=f'Optimal λ={df.loc[best_idx, "lambda_value"]:.3f}')
    axes[0, 0].legend()
    
    # Plot 2: Max factor loading vs Lambda (push-out effectiveness)
    if not df['max_loading'].isna().all():
        axes[0, 1].semilogx(df['lambda_value'], df['max_loading'], 'o-', color='green', linewidth=2, markersize=8)
        axes[0, 1].set_title('Push-Out Effectiveness vs Lambda')
        axes[0, 1].set_xlabel('Lambda (log scale)')
        axes[0, 1].set_ylabel('Maximum Factor Loading')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Highlight optimal point
        pushout_best_idx = df['max_loading'].idxmin()
        axes[0, 1].scatter(df.loc[pushout_best_idx, 'lambda_value'], df.loc[pushout_best_idx, 'max_loading'],
                          color='red', s=100, zorder=5, label=f'Optimal λ={df.loc[pushout_best_idx, "lambda_value"]:.3f}')
        axes[0, 1].legend()
    else:
        axes[0, 1].text(0.5, 0.5, 'Max Loading\nData Not Available', ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Push-Out Effectiveness vs Lambda')
    
    # Plot 3: Factor correlation vs Lambda (separation quality)
    if not df['factor_correlation'].isna().all():
        axes[1, 0].semilogx(df['lambda_value'], df['factor_correlation'], 'o-', color='orange', linewidth=2, markersize=8)
        axes[1, 0].set_title('Factor Separation vs Lambda')
        axes[1, 0].set_xlabel('Lambda (log scale)')
        axes[1, 0].set_ylabel('Average Factor Correlation')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Highlight optimal point
        corr_best_idx = df['factor_correlation'].idxmin()
        axes[1, 0].scatter(df.loc[corr_best_idx, 'lambda_value'], df.loc[corr_best_idx, 'factor_correlation'],
                          color='red', s=100, zorder=5, label=f'Optimal λ={df.loc[corr_best_idx, "lambda_value"]:.3f}')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'Factor Correlation\nData Not Available', ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Factor Separation vs Lambda')
    
    # Plot 4: Convergence analysis
    convergence_colors = ['green' if conv else 'red' for conv in df['converged']]
    axes[1, 1].scatter(df['lambda_value'], df['bursts_completed'], c=convergence_colors, s=80, alpha=0.7)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_title('Convergence Analysis vs Lambda')
    axes[1, 1].set_xlabel('Lambda (log scale)')
    axes[1, 1].set_ylabel('Bursts Completed')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add legend for convergence
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Converged'),
                      Patch(facecolor='red', label='Not Converged')]
    axes[1, 1].legend(handles=legend_elements)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = output_dir / f"{species}_lambda_sensitivity_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVE] Sensitivity plots: {plot_file}")

def main():
    """Main lambda sensitivity analysis script."""
    
    parser = argparse.ArgumentParser(description="Lambda sensitivity analysis for species regularization")
    
    parser.add_argument("--species", required=True, help="Target species for regularization (e.g., CH4)")
    parser.add_argument("--lambda-range", default="auto", help="Lambda range: auto, fine, coarse, or custom values (comma-separated)")
    parser.add_argument("--base-cmd", required=True, help="Base PMF command without regularization flags")
    parser.add_argument("--output-dir", default="lambda_sensitivity_analysis", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"🔍 LAMBDA SENSITIVITY ANALYSIS")
    print("=" * 50)
    print(f"Target Species: {args.species}")
    print(f"Output Directory: {output_dir}")
    print(f"Base Command: {args.base_cmd}")
    
    # Create lambda range
    try:
        if "," in args.lambda_range and args.lambda_range != "auto":
            lambda_values = create_lambda_range("custom", args.lambda_range)
        else:
            lambda_values = create_lambda_range(args.lambda_range)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    
    print(f"Lambda Values: {lambda_values}")
    print("=" * 50)
    
    # Run sensitivity analysis
    results = []
    
    for i, lambda_val in enumerate(lambda_values):
        print(f"\n[STEP] {i+1}/{len(lambda_values)}: Testing λ = {lambda_val}")
        
        metrics = run_single_lambda_analysis(
            species=args.species,
            lambda_val=lambda_val,
            base_cmd=args.base_cmd,
            output_dir=output_dir,
            verbose=args.verbose
        )
        
        if metrics:
            results.append(metrics)
        else:
            print(f"   [FAILED] Lambda {lambda_val} analysis failed")
    
    # Analyze results
    if results:
        print(f"\n[COMPLETE] Lambda sensitivity analysis completed")
        print(f"Successful runs: {len(results)} / {len(lambda_values)}")
        
        analysis = analyze_lambda_sensitivity_results(results, args.species, output_dir)
        create_sensitivity_plots(results, args.species, output_dir)
        
        print(f"\n[SUCCESS] Analysis complete. Results saved in: {output_dir}")
        
        # Print key recommendations
        if analysis:
            print(f"\n🎯 RECOMMENDATIONS:")
            print(f"   • Best model fit: λ = {analysis.get('optimal_lambda_fit', 'N/A')}")
            if 'optimal_lambda_pushout' in analysis:
                print(f"   • Best push-out: λ = {analysis['optimal_lambda_pushout']}")
            if 'optimal_lambda_correlation' in analysis:
                print(f"   • Best separation: λ = {analysis['optimal_lambda_correlation']}")
            print(f"   • Convergence rate: {analysis.get('convergence_rate', 0):.1%}")
        
    else:
        print(f"\n[ERROR] All lambda values failed. Check base command and PMF setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()