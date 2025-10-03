#!/usr/bin/env python3
"""
Process Existing Lambda Sweep Results

This processes the already-completed PMF runs to generate the comparison analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from lambda_sweep_analysis import LambdaSweepAnalyzer

def main():
    analyzer = LambdaSweepAnalyzer()
    
    print("Processing existing lambda sweep results...")
    
    # Process all existing scenarios
    lambda_values = [0, 1, 2, 5, 10, 20, 50, 100]
    
    results = []
    
    # Process CH4 exclusion baseline
    exclusion_dir = Path('lambda_sweep_analysis/scenario_ch4_excluded')
    if exclusion_dir.exists():
        print("\nProcessing CH4 exclusion baseline...")
        profiles = analyzer.extract_factor_profiles(exclusion_dir)
        corr_matrix, diversity = analyzer.compute_factor_correlations(profiles)
        closure = analyzer.extract_closure_metrics(exclusion_dir)
        
        result = {
            'scenario': 'CH4_Exclusion',
            'lambda': None,
            'diversity_score': diversity['diversity_score'] if diversity else None,
            'mean_abs_corr': diversity['mean_abs_correlation'] if diversity else None,
            'max_abs_corr': diversity['max_abs_correlation'] if diversity else None,
            'ch4_closure': closure['ch4_closure_pct'] if closure else None,
            'ch4_q_share': closure['ch4_q_share_pct'] if closure else None,
            'mean_closure': closure['mean_closure_pct'] if closure else None
        }
        results.append(result)
        
        if diversity:
            print(f"CH4 Exclusion - Diversity: {diversity['diversity_score']:.3f}, Mean |r|: {diversity['mean_abs_correlation']:.3f}")
    
    # Process lambda scenarios
    for lambda_val in lambda_values:
        scenario_dir = Path(f'lambda_sweep_analysis/scenario_lambda_{lambda_val}')
        if scenario_dir.exists():
            print(f"\nProcessing Lambda = {lambda_val}...")
            profiles = analyzer.extract_factor_profiles(scenario_dir)
            corr_matrix, diversity = analyzer.compute_factor_correlations(profiles)
            closure = analyzer.extract_closure_metrics(scenario_dir)
            
            result = {
                'scenario': f'Lambda_{lambda_val}',
                'lambda': lambda_val,
                'diversity_score': diversity['diversity_score'] if diversity else None,
                'mean_abs_corr': diversity['mean_abs_correlation'] if diversity else None,
                'max_abs_corr': diversity['max_abs_correlation'] if diversity else None,
                'ch4_closure': closure['ch4_closure_pct'] if closure else None,
                'ch4_q_share': closure['ch4_q_share_pct'] if closure else None,
                'mean_closure': closure['mean_closure_pct'] if closure else None
            }
            results.append(result)
            
            if diversity:
                print(f"Lambda {lambda_val} - Diversity: {diversity['diversity_score']:.3f}, Mean |r|: {diversity['mean_abs_correlation']:.3f}")
            if closure and closure['ch4_closure_pct']:
                print(f"Lambda {lambda_val} - CH4 Closure: {closure['ch4_closure_pct']:.1f}%, Q Share: {closure['ch4_q_share_pct']:.1f}%")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('lambda_sweep_analysis/results_summary.csv', index=False)
    print(f"\nResults saved to lambda_sweep_analysis/results_summary.csv")
    
    # Create plots
    create_summary_plots(results_df)
    
    # Print summary
    print("\n" + "="*80)
    print("LAMBDA SWEEP ANALYSIS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False, float_format='%.3f'))

def create_summary_plots(results_df):
    """Create summary plots from results"""
    
    # Filter regularization results (exclude CH4 exclusion)
    reg_results = results_df[results_df['lambda'].notna()].copy()
    reg_results = reg_results.sort_values('lambda')
    
    # Get exclusion baseline
    excl_result = results_df[results_df['lambda'].isna()]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Lambda Sweep Results: CH4 Regularization vs Exclusion', fontsize=16, fontweight='bold')
    
    # Plot 1: Diversity Score vs Lambda
    ax1 = axes[0, 0]
    ax1.plot(reg_results['lambda'], reg_results['diversity_score'], 'bo-', linewidth=2, markersize=8, label='Regularization')
    
    if len(excl_result) > 0 and not pd.isna(excl_result['diversity_score'].iloc[0]):
        excl_div = excl_result['diversity_score'].iloc[0]
        ax1.axhline(y=excl_div, color='red', linestyle='--', linewidth=2, label=f'CH4 Exclusion ({excl_div:.3f})')
    
    ax1.set_xlabel('Lambda (λ)')
    ax1.set_ylabel('Diversity Score')
    ax1.set_title('Factor Diversity vs Lambda')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Mean Abs Correlation vs Lambda  
    ax2 = axes[0, 1]
    ax2.plot(reg_results['lambda'], reg_results['mean_abs_corr'], 'go-', linewidth=2, markersize=8, label='Regularization')
    
    if len(excl_result) > 0 and not pd.isna(excl_result['mean_abs_corr'].iloc[0]):
        excl_corr = excl_result['mean_abs_corr'].iloc[0]
        ax2.axhline(y=excl_corr, color='red', linestyle='--', linewidth=2, label=f'CH4 Exclusion ({excl_corr:.3f})')
    
    ax2.set_xlabel('Lambda (λ)')
    ax2.set_ylabel('Mean Absolute Correlation')
    ax2.set_title('Factor Correlation vs Lambda')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: CH4 Closure vs Lambda
    ax3 = axes[1, 0]
    valid_closure = reg_results.dropna(subset=['ch4_closure'])
    if len(valid_closure) > 0:
        ax3.plot(valid_closure['lambda'], valid_closure['ch4_closure'], 'ro-', linewidth=2, markersize=8)
        ax3.set_xlabel('Lambda (λ)')
        ax3.set_ylabel('CH4 Closure %')
        ax3.set_title('CH4 Closure vs Lambda')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: CH4 Q Share vs Lambda
    ax4 = axes[1, 1]
    valid_q_share = reg_results.dropna(subset=['ch4_q_share'])
    if len(valid_q_share) > 0:
        ax4.plot(valid_q_share['lambda'], valid_q_share['ch4_q_share'], 'mo-', linewidth=2, markersize=8)
        ax4.set_xlabel('Lambda (λ)')
        ax4.set_ylabel('CH4 Q Share %')
        ax4.set_title('CH4 Q Share vs Lambda')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lambda_sweep_analysis/lambda_sweep_summary.png', dpi=300, bbox_inches='tight')
    print("\nSummary plot saved to lambda_sweep_analysis/lambda_sweep_summary.png")
    plt.show()

if __name__ == "__main__":
    main()