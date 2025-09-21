#!/usr/bin/env python3
"""
Factor 3 Outlier Analysis Script
===============================

This script investigates whether the Factor 3 outlier (128.084 contribution) 
is caused by:
1. Outliers in raw data (extreme concentrations)
2. Poor model fitting (mathematical artifact)

Following EPA PMF 5.0 best practices for outlier investigation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from datetime import datetime, timedelta

# Import ESAT to recreate the analysis
from esat.model.batch_sa import BatchSA

def load_pmf_data():
    """Load the concentration and uncertainty data"""
    conc_df = pd.read_csv('pmf_results_esat/MMF2_mmf_20230901_20230930_concentrations.csv')
    unc_df = pd.read_csv('pmf_results_esat/MMF2_mmf_20230901_20230930_uncertainties.csv')
    
    conc_df['datetime'] = pd.to_datetime(conc_df['datetime'])
    unc_df['datetime'] = pd.to_datetime(unc_df['datetime'])
    
    return conc_df, unc_df

def run_quick_pmf():
    """Run a quick PMF analysis to get factor contributions"""
    conc_df, unc_df = load_pmf_data()
    
    # Prepare matrices
    species_cols = [col for col in conc_df.columns if col != 'datetime']
    V = conc_df[species_cols].values
    U = unc_df[species_cols].values
    
    print(f"Running PMF on {V.shape[0]} samples, {V.shape[1]} species")
    
    # Run single PMF model (quick analysis)
    batch_sa = BatchSA(
        V=V, U=U, factors=6, models=1, seed=65457,  # Use same seed as best model
        method="ls-nmf", max_iter=5000, verbose=False
    )
    
    batch_sa.train()
    best_model = batch_sa.details(0)  # Get the single model
    
    # Extract factor contributions (G matrix)
    factor_contributions = best_model['factors']  # Shape: (n_samples, n_factors)
    
    return factor_contributions, conc_df, species_cols

def analyze_factor3_outlier():
    """Main analysis function"""
    print("=== FACTOR 3 OUTLIER ROOT CAUSE ANALYSIS ===")
    print()
    
    # Load data and run PMF
    factor_contributions, conc_df, species_cols = run_quick_pmf()
    
    # Find Factor 3 outlier
    factor3_contributions = factor_contributions[:, 2]  # Factor 3 (0-indexed)
    outlier_idx = np.argmax(factor3_contributions)
    outlier_value = factor3_contributions[outlier_idx]
    outlier_timestamp = conc_df['datetime'].iloc[outlier_idx]
    
    print(f"Factor 3 Outlier Identified:")
    print(f"  Index: {outlier_idx}")
    print(f"  Timestamp: {outlier_timestamp}")
    print(f"  Factor 3 contribution: {outlier_value:.3f}")
    print(f"  Factor 3 mean: {np.mean(factor3_contributions):.3f}")
    print(f"  Factor 3 std: {np.std(factor3_contributions):.3f}")
    print(f"  Z-score: {(outlier_value - np.mean(factor3_contributions)) / np.std(factor3_contributions):.1f}")
    print()
    
    # ANALYSIS 1: Raw Data Examination
    print("1. RAW DATA ANALYSIS AT OUTLIER TIMESTAMP")
    print("=" * 50)
    
    # Get raw concentrations at outlier time
    outlier_concentrations = conc_df.iloc[outlier_idx]
    
    # Calculate Z-scores for each species at outlier time
    print("Species concentrations at outlier timestamp:")
    print("Species".ljust(15) + "Value".ljust(12) + "Mean".ljust(12) + "Std".ljust(12) + "Z-score".ljust(10) + "Outlier?")
    print("-" * 75)
    
    raw_data_outliers = []
    for species in species_cols:
        value = outlier_concentrations[species]
        mean_val = conc_df[species].mean()
        std_val = conc_df[species].std()
        z_score = (value - mean_val) / std_val if std_val > 0 else 0
        
        is_outlier = abs(z_score) > 3.0  # EPA threshold
        if is_outlier:
            raw_data_outliers.append(species)
            
        print(f"{species:<15}{value:<12.3f}{mean_val:<12.3f}{std_val:<12.3f}{z_score:<10.2f}{'YES' if is_outlier else 'NO'}")
    
    print()
    if raw_data_outliers:
        print(f"⚠️ RAW DATA OUTLIERS DETECTED: {raw_data_outliers}")
    else:
        print("✅ No extreme raw data outliers found")
    print()
    
    # ANALYSIS 2: Temporal Context
    print("2. TEMPORAL CONTEXT ANALYSIS")
    print("=" * 30)
    
    # Look at ±2 hours around outlier
    outlier_time = conc_df['datetime'].iloc[outlier_idx]
    time_window = timedelta(hours=2)
    
    mask = (conc_df['datetime'] >= outlier_time - time_window) & \
           (conc_df['datetime'] <= outlier_time + time_window)
    
    window_data = conc_df[mask]
    window_indices = window_data.index.tolist()
    
    print(f"Time window: {outlier_time - time_window} to {outlier_time + time_window}")
    print(f"Data points in window: {len(window_data)}")
    
    if len(window_data) > 1:
        # Check if there are other unusual values in the time window
        print("\\nFactor 3 contributions in time window:")
        window_factor3 = factor3_contributions[window_indices]
        for i, (idx, contrib) in enumerate(zip(window_indices, window_factor3)):
            marker = " <-- OUTLIER" if idx == outlier_idx else ""
            print(f"  {conc_df['datetime'].iloc[idx]}: {contrib:.3f}{marker}")
    
    print()
    
    # ANALYSIS 3: Factor Contribution Patterns
    print("3. FACTOR CONTRIBUTION PATTERNS")
    print("=" * 35)
    
    # Check contributions of all factors at outlier time
    all_factor_contributions = factor_contributions[outlier_idx, :]
    
    print("All factor contributions at outlier timestamp:")
    for f_idx, contrib in enumerate(all_factor_contributions):
        print(f"  Factor {f_idx+1}: {contrib:.3f}")
    
    total_contribution = np.sum(all_factor_contributions)
    print(f"\\nTotal contribution: {total_contribution:.3f}")
    print(f"Factor 3 percentage: {(all_factor_contributions[2] / total_contribution * 100):.1f}%")
    print()
    
    # ANALYSIS 4: Model Quality Assessment
    print("4. MODEL FITTING QUALITY")
    print("=" * 25)
    
    # Calculate residuals at outlier point
    # Reconstruct concentrations using PMF factors
    from sklearn.decomposition import NMF
    
    # Get factor profiles and contributions
    # For simplicity, let's examine the residuals pattern
    mean_factor3 = np.mean(factor3_contributions)
    outlier_deviation = outlier_value - mean_factor3
    
    print(f"Factor 3 contribution statistics:")
    print(f"  Minimum: {np.min(factor3_contributions):.3f}")
    print(f"  Maximum: {np.max(factor3_contributions):.3f}")
    print(f"  Median: {np.median(factor3_contributions):.3f}")
    print(f"  95th percentile: {np.percentile(factor3_contributions, 95):.3f}")
    print(f"  99th percentile: {np.percentile(factor3_contributions, 99):.3f}")
    print()
    
    # Count extreme outliers
    extreme_outliers = np.sum(factor3_contributions > np.percentile(factor3_contributions, 99))
    print(f"Data points above 99th percentile: {extreme_outliers}")
    
    # ANALYSIS 5: Determine Root Cause
    print()
    print("5. ROOT CAUSE DETERMINATION")
    print("=" * 30)
    
    # Decision logic based on EPA guidelines
    has_raw_outliers = len(raw_data_outliers) > 0
    extreme_factor_outlier = outlier_value > (mean_factor3 + 10 * np.std(factor3_contributions))
    isolated_event = extreme_outliers <= 3  # Very few extreme events
    
    print(f"Raw data outliers present: {'YES' if has_raw_outliers else 'NO'}")
    print(f"Extreme factor outlier (>10σ): {'YES' if extreme_factor_outlier else 'NO'}")
    print(f"Isolated event (<3 extreme): {'YES' if isolated_event else 'NO'}")
    print()
    
    if has_raw_outliers and extreme_factor_outlier:
        conclusion = "RAW DATA OUTLIER DRIVEN"
        explanation = "Extreme concentrations in raw data are causing the PMF model to assign very high contributions to Factor 3"
        action = "EPA Method 1: Investigate physical cause → Apply uncertainty weighting → Consider exclusion if no explanation"
        
    elif extreme_factor_outlier and not has_raw_outliers:
        conclusion = "MODEL FITTING ISSUE"
        explanation = "Normal raw data but extreme factor contribution suggests mathematical fitting problem"
        action = "EPA Method 2: Check model constraints → Reduce factors → Apply robust fitting"
        
    else:
        conclusion = "MODERATE OUTLIER"
        explanation = "Outlier present but not extreme in both raw data and model"
        action = "EPA Method 3: Apply uncertainty weighting → Monitor in future analyses"
    
    print(f"🔍 CONCLUSION: {conclusion}")
    print(f"📝 Explanation: {explanation}")
    print(f"⚡ Recommended Action: {action}")
    
    return {
        'outlier_idx': outlier_idx,
        'outlier_timestamp': outlier_timestamp,
        'outlier_value': outlier_value,
        'raw_outliers': raw_data_outliers,
        'conclusion': conclusion,
        'explanation': explanation,
        'action': action
    }

if __name__ == "__main__":
    results = analyze_factor3_outlier()
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - Results available for further investigation")