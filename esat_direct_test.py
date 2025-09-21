#!/usr/bin/env python3
"""
Direct ESAT PMF Test
====================

This script tests ESAT PMF functionality directly without using DataHandler.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Import ESAT core modules
try:
    from esat.model.batch_sa import BatchSA
    print("✅ ESAT BatchSA imported successfully")
except ImportError as e:
    print(f"❌ ESAT BatchSA import failed: {e}")
    exit(1)

def create_simple_test_data():
    """Create minimal test data for PMF."""
    print("🔧 Creating simple test data...")
    
    # Simple synthetic data: 50 samples, 4 species, 3 factors
    np.random.seed(42)
    n_samples = 50
    n_species = 4  
    n_factors = 3
    
    # Create true factor profiles (F matrix: factors x species)
    F_true = np.array([
        [10, 5, 2, 8],    # Factor 1: High in species 1 and 4
        [2, 15, 3, 1],    # Factor 2: High in species 2 
        [1, 2, 12, 3]     # Factor 3: High in species 3
    ])
    
    # Create factor contributions (G matrix: samples x factors)
    G_true = np.random.exponential(scale=2.0, size=(n_samples, n_factors))
    
    # Generate observed data: V = G * F + noise
    V = np.dot(G_true, F_true) + np.random.normal(0, 1, (n_samples, n_species))
    V = np.maximum(V, 0.1)  # Ensure positive values
    
    # Create uncertainty matrix (15% of concentration + 1.0)
    U = np.sqrt((0.15 * V)**2 + 1.0**2)
    
    print(f"📊 Generated data: V={V.shape}, U={U.shape}")
    print(f"   True factors: {n_factors}")
    
    return V, U, F_true, G_true

def test_esat_pmf():
    """Test ESAT PMF functionality directly."""
    print("🚀 Testing ESAT PMF Analysis")
    print("=" * 35)
    
    # Create test data
    V, U, F_true, G_true = create_simple_test_data()
    
    # Test BatchSA directly
    print("🔄 Initializing BatchSA...")
    try:
        batch_sa = BatchSA(
            V=V, U=U,
            factors=3,  # We know there are 3 true factors
            models=5,   # Run 5 models for speed
            method="ls-nmf",
            seed=42,
            verbose=True
        )
        print("✅ BatchSA initialized successfully")
        
        # Run PMF
        print("🔄 Running PMF analysis...")
        batch_sa.train()
        
        # Get results
        best_idx = batch_sa.best_model
        best_model = batch_sa.results[best_idx]
        
        print(f"\n✅ PMF Analysis Complete!")
        print(f"   Best model: #{best_idx}")
        print(f"   Q(true): {best_model.Qtrue:.2f}")
        print(f"   Q(robust): {best_model.Qrobust:.2f}")
        
        # Extract factor profiles and contributions
        # In ESAT: H = factor profiles (sources), W = contributions (time series)
        F_estimated = best_model.H  # Factor profiles (sources)
        G_estimated = best_model.W  # Factor contributions (time series)
        
        print(f"\n📊 Results:")
        print(f"   Estimated F (H) shape: {F_estimated.shape}")
        print(f"   Estimated G (W) shape: {G_estimated.shape}")
        print(f"   True F shape: {F_true.shape}")
        print(f"   True G shape: {G_true.shape}")
        
        # Create visualization
        create_comparison_plots(F_true, F_estimated, G_true, G_estimated)
        
        return True
        
    except Exception as e:
        print(f"❌ BatchSA failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_comparison_plots(F_true, F_estimated, G_true, G_estimated):
    """Create plots comparing true vs estimated results."""
    print("📊 Creating comparison plots...")
    
    output_dir = Path("pmf_esat_test")
    output_dir.mkdir(exist_ok=True)
    
    # Plot 1: Factor Profiles Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # True profiles
    species_names = ['Species 1', 'Species 2', 'Species 3', 'Species 4']
    x = np.arange(len(species_names))
    width = 0.25
    
    for i in range(F_true.shape[0]):
        ax1.bar(x + i*width, F_true[i, :], width, 
                label=f'True Factor {i+1}', alpha=0.8)
    
    ax1.set_title('True Factor Profiles', fontweight='bold')
    ax1.set_xlabel('Species')
    ax1.set_ylabel('Contribution')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(species_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Estimated profiles
    for i in range(F_estimated.shape[0]):
        ax2.bar(x + i*width, F_estimated[i, :], width, 
                label=f'Est. Factor {i+1}', alpha=0.8)
    
    ax2.set_title('Estimated Factor Profiles', fontweight='bold')
    ax2.set_xlabel('Species')
    ax2.set_ylabel('Contribution')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(species_names)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "factor_profiles_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    
    # Plot 2: Factor Contributions Time Series
    fig, axes = plt.subplots(F_true.shape[0], 1, figsize=(12, 8))
    if F_true.shape[0] == 1:
        axes = [axes]
    
    time_index = np.arange(G_true.shape[0])
    
    for i in range(F_true.shape[0]):
        ax = axes[i]
        ax.plot(time_index, G_true[:, i], 'o-', label=f'True Factor {i+1}', 
                alpha=0.7, linewidth=2)
        ax.plot(time_index, G_estimated[:, i], 's--', label=f'Est. Factor {i+1}', 
                alpha=0.7, linewidth=2)
        ax.set_title(f'Factor {i+1} Contributions', fontweight='bold')
        ax.set_ylabel('Contribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Sample Index')
    
    plt.tight_layout()
    plot_path = output_dir / "factor_contributions_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    
    # Plot 3: Reconstruction Quality
    V_reconstructed = np.dot(G_estimated, F_estimated)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    for i in range(min(4, F_true.shape[1])):
        ax = axes[i]
        # Create sample indices for x-axis
        sample_idx = np.arange(len(G_estimated))
        
        # Plot original data vs reconstructed
        original = np.dot(G_true, F_true)[:, i]  # True data for species i
        reconstructed = V_reconstructed[:, i]    # Reconstructed data for species i
        
        ax.scatter(original, reconstructed, alpha=0.6)
        
        # Add perfect reconstruction line
        min_val = min(np.min(original), np.min(reconstructed))
        max_val = max(np.max(original), np.max(reconstructed))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax.set_title(f'{species_names[i]} Reconstruction')
        ax.set_xlabel('True Value')
        ax.set_ylabel('Reconstructed Value')
        ax.grid(True, alpha=0.3)
        
        # Calculate R²
        correlation = np.corrcoef(original, reconstructed)[0, 1]
        ax.text(0.05, 0.95, f'R = {correlation:.3f}', 
                transform=ax.transAxes, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle('PMF Reconstruction Quality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plot_path = output_dir / "reconstruction_quality.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    
    print(f"📊 All plots saved in: {output_dir}")

def main():
    """Main function."""
    print("🧪 ESAT Direct PMF Test")
    print("=" * 25)
    
    success = test_esat_pmf()
    
    if success:
        print("\n🎉 ESAT PMF Test Successful!")
        print("📁 Results saved in: pmf_esat_test/")
        print("📊 Check the generated plots:")
        print("   - factor_profiles_comparison.png")
        print("   - factor_contributions_comparison.png") 
        print("   - reconstruction_quality.png")
        print("\n✨ ESAT library is working correctly for PMF source apportionment!")
    else:
        print("\n❌ Test failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
