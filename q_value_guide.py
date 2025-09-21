#!/usr/bin/env python3
"""
Q-Value Interpretation Guide for PMF Analysis
=============================================

This script helps understand what Q-values mean in PMF (Positive Matrix Factorization) 
source apportionment analysis according to EPA PMF 5.0 User Guide.

Usage:
    python q_value_guide.py --q-robust 850 --samples 100 --species 8 --factors 4
"""

import argparse
import math

def calculate_dof(n_samples, n_species, n_factors):
    """
    Calculate degrees of freedom for PMF model.
    
    EPA Formula: DOF = (samples × species) - (samples × factors) - (species × factors) + factors²
    """
    return (n_samples * n_species) - (n_samples * n_factors) - (n_species * n_factors) + (n_factors * n_factors)

def interpret_q_value(q_robust, n_samples, n_species, n_factors):
    """
    Interpret Q-value according to EPA PMF guidelines.
    """
    dof = calculate_dof(n_samples, n_species, n_factors)
    q_ratio = q_robust / dof if dof > 0 else float('inf')
    
    # EPA quality assessment
    if q_ratio <= 1.5:
        quality = "Excellent"
        color = "🟢"
        assessment = "Model fits data very well - excellent reconstruction"
    elif q_ratio <= 2.0:
        quality = "Good"
        color = "🟡"
        assessment = "Model fits data adequately - acceptable for source apportionment"
    elif q_ratio <= 3.0:
        quality = "Fair"
        color = "🟠"
        assessment = "Model may need refinement - consider increasing factors"
    else:
        quality = "Poor"
        color = "🔴"
        assessment = "Model does not fit data well - major issues to address"
    
    return {
        'dof': dof,
        'q_ratio': q_ratio,
        'quality': quality,
        'color': color,
        'assessment': assessment
    }

def main():
    parser = argparse.ArgumentParser(description='Q-Value Interpretation Guide for PMF Analysis')
    parser.add_argument('--q-robust', type=float, required=True,
                       help='Q(robust) value from PMF analysis')
    parser.add_argument('--q-true', type=float, default=None,
                       help='Q(true) value from PMF analysis (optional)')
    parser.add_argument('--samples', type=int, required=True,
                       help='Number of data samples')
    parser.add_argument('--species', type=int, required=True,
                       help='Number of species/pollutants')
    parser.add_argument('--factors', type=int, required=True,
                       help='Number of factors resolved')
    
    args = parser.parse_args()
    
    print("🔬 PMF Q-Value Interpretation Guide")
    print("=" * 50)
    print()
    
    # Basic model info
    print(f"📊 Model Configuration:")
    print(f"   Data samples: {args.samples:,}")
    print(f"   Species: {args.species}")
    print(f"   Factors: {args.factors}")
    print()
    
    # Q-value interpretation
    result = interpret_q_value(args.q_robust, args.samples, args.species, args.factors)
    
    print(f"📈 Q-Value Analysis:")
    print(f"   Q(robust): {args.q_robust:.2f}")
    if args.q_true:
        print(f"   Q(true): {args.q_true:.2f}")
        outlier_ratio = args.q_true / args.q_robust if args.q_robust > 0 else 1
        if outlier_ratio > 2:
            print(f"   ⚠️  Q(true)/Q(robust) = {outlier_ratio:.2f} (may indicate outliers)")
    print(f"   Degrees of Freedom: {result['dof']:,}")
    print(f"   Expected Q (perfect fit): ~{result['dof']:,}")
    print()
    
    print(f"🎯 Model Quality Assessment:")
    print(f"   Q(robust)/DOF Ratio: {result['q_ratio']:.3f}")
    print(f"   Quality Rating: {result['color']} {result['quality']}")
    print(f"   Assessment: {result['assessment']}")
    print()
    
    # EPA Guidelines
    print("📋 EPA PMF Q-Value Guidelines:")
    print("   🟢 Q/DOF ≤ 1.5: Excellent fit")
    print("   🟡 Q/DOF ≤ 2.0: Good fit") 
    print("   🟠 Q/DOF ≤ 3.0: Fair fit (may need refinement)")
    print("   🔴 Q/DOF > 3.0: Poor fit (review model/data)")
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    
    if result['q_ratio'] <= 1.5:
        print("   ✅ Excellent model quality - proceed with confidence")
        print("   ✅ Results are suitable for source identification")
        print("   ✅ Factor profiles and contributions are reliable")
    elif result['q_ratio'] <= 2.0:
        print("   ✅ Good model quality - acceptable for source apportionment")
        print("   ⚠️  Consider validating with external data if possible")
        print("   ⚠️  Review factor profiles for physical meaning")
    elif result['q_ratio'] <= 3.0:
        print("   ⚠️  Fair model quality - some improvements needed:")
        print("   📈 Try increasing the number of factors")
        print("   🔍 Check for outliers in the data")
        print("   🔧 Review uncertainty estimates")
        print("   📊 Consider data pre-processing improvements")
    else:
        print("   🚨 Poor model quality - significant issues to address:")
        print("   📈 Increase number of factors (try +1 or +2)")
        print("   🔍 Identify and handle outliers")
        print("   🔧 Review uncertainty calculation method")
        print("   📊 Check data quality and preprocessing")
        print("   📋 Consider removing problematic species")
        print("   🕒 Try different time periods or data subsets")
    
    print()
    
    # Context and typical values
    print("🎯 Typical Q-Values in Environmental PMF:")
    print("   • Urban air quality: Q/DOF = 0.8 - 2.5")
    print("   • Rural/background: Q/DOF = 0.5 - 1.8")  
    print("   • Indoor air: Q/DOF = 1.0 - 3.0")
    print("   • Source samples: Q/DOF = 0.3 - 1.5")
    print()
    
    # Mathematical background
    print("🧮 Q-Value Mathematics:")
    print("   Q = Σ[(observed - modeled)² / uncertainty²]")
    print("   • Lower Q = better fit between model and data")
    print("   • Q/DOF ≈ 1.0 indicates model captures data variance optimally")
    print("   • Q/DOF >> 1.0 suggests systematic model deficiencies")
    print("   • Q/DOF << 1.0 may indicate over-fitting or underestimated uncertainties")

if __name__ == "__main__":
    main()
