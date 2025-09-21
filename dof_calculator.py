#!/usr/bin/env python3
"""
PMF Degrees of Freedom (DOF) Calculator and Analyzer
===================================================

This tool helps understand and calculate degrees of freedom for PMF models,
and identifies optimal factor/species combinations.

Usage:
    python dof_calculator.py --samples 100 --species 8 --factors 4
    python dof_calculator.py --samples 100 --species 8 --analyze
"""

import argparse
import math

def calculate_dof(n_samples, n_species, n_factors):
    """
    Calculate degrees of freedom for PMF model.
    
    DOF = (samples × species) - (samples × factors) - (species × factors) + factors²
    
    This represents:
    - Data points: samples × species
    - Model parameters: (samples × factors) + (species × factors) - factors²
    - DOF = Data points - Model parameters
    """
    data_points = n_samples * n_species
    factor_contributions = n_samples * n_factors  # W matrix (time series)
    factor_profiles = n_species * n_factors       # H matrix (source profiles)
    factor_overlap = n_factors * n_factors        # Overlap correction
    
    model_parameters = factor_contributions + factor_profiles - factor_overlap
    dof = data_points - model_parameters
    
    return {
        'dof': dof,
        'data_points': data_points,
        'model_parameters': model_parameters,
        'factor_contributions': factor_contributions,
        'factor_profiles': factor_profiles,
        'factor_overlap': factor_overlap
    }

def analyze_factor_range(n_samples, n_species, max_factors=None):
    """Analyze DOF for different numbers of factors."""
    if max_factors is None:
        max_factors = min(n_species, n_samples // 2)
    
    print(f"🔍 DOF Analysis for {n_samples} samples × {n_species} species:")
    print("=" * 60)
    print(f"{'Factors':<8} {'DOF':<8} {'Status':<12} {'Recommendation'}")
    print("-" * 60)
    
    recommendations = []
    
    for factors in range(1, max_factors + 1):
        result = calculate_dof(n_samples, n_species, factors)
        dof = result['dof']
        
        if dof <= 0:
            status = "❌ Invalid"
            recommendation = "Too many factors"
        elif dof < 10:
            status = "⚠️ Risky"
            recommendation = "Very few DOF"
        elif dof < 50:
            status = "🟡 Caution"
            recommendation = "Limited DOF"
        else:
            status = "✅ Good"
            recommendation = "Adequate DOF"
            
        print(f"{factors:<8} {dof:<8} {status:<12} {recommendation}")
        
        if dof > 0:
            recommendations.append((factors, dof, status))
    
    return recommendations

def find_optimal_factors(n_samples, n_species):
    """Find the range of reasonable factor numbers."""
    max_possible = min(n_species - 1, n_samples // 3)  # Conservative limits
    
    optimal_range = []
    for factors in range(2, max_possible + 1):
        result = calculate_dof(n_samples, n_species, factors)
        if result['dof'] >= 20:  # Minimum recommended DOF
            optimal_range.append(factors)
    
    return optimal_range

def main():
    parser = argparse.ArgumentParser(description='PMF Degrees of Freedom Calculator')
    parser.add_argument('--samples', type=int, required=True,
                       help='Number of data samples (time points)')
    parser.add_argument('--species', type=int, required=True,
                       help='Number of species/pollutants')
    parser.add_argument('--factors', type=int, default=None,
                       help='Number of factors (optional - for specific calculation)')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze DOF for range of factor numbers')
    parser.add_argument('--max-factors', type=int, default=None,
                       help='Maximum factors to analyze (default: auto)')
    
    args = parser.parse_args()
    
    print("🧮 PMF Degrees of Freedom (DOF) Analysis")
    print("=" * 50)
    print()
    
    if args.factors:
        # Calculate DOF for specific configuration
        result = calculate_dof(args.samples, args.species, args.factors)
        
        print(f"📊 Model Configuration:")
        print(f"   Samples: {args.samples:,}")
        print(f"   Species: {args.species}")
        print(f"   Factors: {args.factors}")
        print()
        
        print(f"🔢 DOF Calculation Breakdown:")
        print(f"   Data points (S×P): {result['data_points']:,}")
        print(f"   Factor contributions (S×F): {result['factor_contributions']:,}")
        print(f"   Factor profiles (P×F): {result['factor_profiles']:,}")
        print(f"   Factor overlap correction (F²): {result['factor_overlap']:,}")
        print(f"   Model parameters: {result['model_parameters']:,}")
        print()
        
        print(f"📈 Result:")
        print(f"   Degrees of Freedom: {result['dof']:,}")
        
        if result['dof'] <= 0:
            print(f"   Status: ❌ Invalid model - too many factors!")
            print(f"   Problem: More parameters than data points")
            print(f"   Solution: Reduce factors to {max(1, args.species-1)} or fewer")
        elif result['dof'] < 10:
            print(f"   Status: ⚠️  Very low DOF - model may be unstable")
            print(f"   Recommendation: Consider reducing factors or increasing data")
        elif result['dof'] < 50:
            print(f"   Status: 🟡 Low DOF - use with caution")
            print(f"   Recommendation: More data or fewer factors would be better")
        else:
            print(f"   Status: ✅ Good DOF - model should be stable")
    
    if args.analyze:
        print("\n" + "="*50)
        recommendations = analyze_factor_range(args.samples, args.species, args.max_factors)
        
        print()
        print("🎯 Recommendations:")
        
        optimal = find_optimal_factors(args.samples, args.species)
        if optimal:
            print(f"   ✅ Optimal factor range: {min(optimal)} - {max(optimal)}")
            print(f"   💡 Start with {min(optimal)+1} factors and test upward")
        else:
            print(f"   ⚠️  No optimal range found - consider:")
            print(f"      • More data samples (current: {args.samples})")
            print(f"      • Fewer species (current: {args.species})")
            print(f"      • Accept lower DOF with caution")
    
    print()
    print("📋 DOF Guidelines:")
    print("   • DOF > 50: Excellent - stable model")
    print("   • DOF 20-50: Good - acceptable for most purposes")
    print("   • DOF 10-20: Caution - model may be sensitive")
    print("   • DOF 1-10: Risky - very unstable model")
    print("   • DOF ≤ 0: Invalid - impossible to solve")
    print()
    
    print("🔧 DOF Formula Explained:")
    print("   DOF = (Samples × Species) - (Samples × Factors) - (Species × Factors) + Factors²")
    print("   DOF = Data Points - Model Parameters")
    print("   • More samples/species → Higher DOF ✅")
    print("   • More factors → Lower DOF ⚠️")
    print("   • DOF represents 'statistical freedom' for model fitting")

if __name__ == "__main__":
    main()
