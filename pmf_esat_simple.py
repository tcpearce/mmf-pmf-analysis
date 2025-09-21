#!/usr/bin/env python3
"""
Simple ESAT PMF Source Apportionment Test
=========================================

This script demonstrates PMF source apportionment using the current ESAT library API.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import ESAT modules
try:
    import esat
    from esat.data.datahandler import DataHandler
    from esat.model.batch_sa import BatchSA
    print("✅ ESAT library imported successfully")
except ImportError as e:
    print(f"❌ ESAT library not found: {e}")
    sys.exit(1)

def create_test_data():
    """Create simple test data for PMF analysis."""
    print("🔧 Creating test environmental data...")
    
    # Generate synthetic environmental data (simulating hourly measurements)
    np.random.seed(42)
    n_samples = 100  # 100 hours of data
    
    # Define source profiles (4 sources, 6 species)
    sources = {
        'Traffic': {'PM2.5': 15, 'NOx': 30, 'CO': 5, 'SO2': 2, 'O3': 10, 'PM10': 20},
        'Industrial': {'PM2.5': 8, 'NOx': 10, 'CO': 2, 'SO2': 25, 'O3': 5, 'PM10': 15},
        'Residential': {'PM2.5': 12, 'NOx': 5, 'CO': 15, 'SO2': 3, 'O3': 8, 'PM10': 18},
        'Background': {'PM2.5': 5, 'NOx': 3, 'CO': 1, 'SO2': 1, 'O3': 20, 'PM10': 8}
    }
    
    # Generate source contributions (G matrix)
    source_names = list(sources.keys())
    species_names = list(sources['Traffic'].keys())
    
    # Random source contributions over time
    contributions = np.random.exponential(scale=2.0, size=(n_samples, len(source_names)))
    
    # Create concentration data by mixing sources
    concentration_data = np.zeros((n_samples, len(species_names)))
    
    for i, species in enumerate(species_names):
        for j, source in enumerate(source_names):
            # Add contribution from each source
            concentration_data[:, i] += contributions[:, j] * sources[source][species]
    
    # Add measurement noise
    concentration_data += np.random.normal(0, 1, concentration_data.shape)
    concentration_data = np.maximum(concentration_data, 0.1)  # Ensure positive values
    
    # Create DataFrame
    df = pd.DataFrame(concentration_data, columns=species_names)
    
    # Add datetime index
    start_time = pd.Timestamp('2023-01-01')
    df.index = pd.date_range(start_time, periods=n_samples, freq='H')
    
    # Create uncertainty data (EPA method: 10-20% of concentration + MDL)
    uncertainty_data = df.copy()
    for col in df.columns:
        # 15% relative error + small MDL
        uncertainty_data[col] = np.sqrt((0.15 * df[col])**2 + 1.0**2)
    
    print(f"✅ Generated {n_samples} samples with {len(species_names)} species")
    print(f"📊 Species: {', '.join(species_names)}")
    print(f"🏭 True sources: {', '.join(source_names)}")
    
    return df, uncertainty_data

def run_esat_pmf_analysis():
    """Run PMF analysis using ESAT."""
    print("🚀 Starting ESAT PMF Analysis")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("pmf_esat_test")
    output_dir.mkdir(exist_ok=True)
    
    # Generate test data
    conc_data, unc_data = create_test_data()
    
    # Save data to CSV files (ESAT requirement)
    conc_file = output_dir / "concentrations.csv"
    unc_file = output_dir / "uncertainties.csv"
    
    conc_data.to_csv(conc_file)
    unc_data.to_csv(unc_file)
    
    print(f"💾 Saved concentration data: {conc_file}")
    print(f"💾 Saved uncertainty data: {unc_file}")
    
    # Initialize ESAT DataHandler
    print("🔄 Initializing ESAT DataHandler...")
    try:
        # Use absolute paths to avoid ESAT path issues
        abs_conc_file = conc_file.resolve()
        abs_unc_file = unc_file.resolve()
        
        data_handler = DataHandler(
            input_path=str(abs_conc_file),
            uncertainty_path=str(abs_unc_file),
            index_col=0  # Use first column (datetime) as index
        )
        print("✅ DataHandler initialized successfully")
        
        # Get data matrices
        V, U = data_handler.get_data()
        print(f"📊 Data matrices: V={V.shape}, U={U.shape}")
        print(f"   Features: {data_handler.features}")
        
    except Exception as e:
        print(f"❌ DataHandler initialization failed: {e}")
        return False
    
    # Run PMF analysis
    print("🔄 Running PMF analysis...")
    try:
        # Configure PMF parameters
        n_factors = 4  # We know there are 4 sources
        n_models = 10  # Run 10 models for robustness
        
        # Initialize BatchSA
        batch_sa = BatchSA(
            V=V, U=U,
            factors=n_factors,
            models=n_models,
            method="ls-nmf",  # Least squares NMF
            seed=42,
            verbose=True
        )
        
        print(f"🔧 Configuration: {n_factors} factors, {n_models} models")
        print("   Running batch PMF analysis...")
        
        # Train models
        batch_sa.train()
        
        # Get best model
        best_model_idx = batch_sa.best_model
        best_model = batch_sa.results[best_model_idx]
        
        print(f"✅ PMF analysis complete!")
        print(f"   Best model: #{best_model_idx}")
        print(f"   Q(true): {best_model.Q_true:.2f}")
        print(f"   Q(robust): {best_model.Q_robust:.2f}")
        
        # Extract results
        factor_profiles = best_model.F  # Factor profiles (sources)
        factor_contributions = best_model.G  # Time series contributions
        
        print(f"📊 Results:")
        print(f"   Factor profiles (F): {factor_profiles.shape}")
        print(f"   Factor contributions (G): {factor_contributions.shape}")
        
        # Create basic visualization
        create_basic_plots(factor_profiles, factor_contributions, data_handler.features, output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ PMF analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_basic_plots(factor_profiles, factor_contributions, feature_names, output_dir):
    """Create basic PMF visualization plots."""
    print("📊 Creating visualization plots...")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Plot 1: Factor Profiles
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('PMF Factor Profiles (Source Signatures)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    for i in range(min(4, factor_profiles.shape[0])):
        ax = axes[i]
        profile = factor_profiles[i, :]
        bars = ax.bar(range(len(feature_names)), profile, alpha=0.7)
        ax.set_title(f'Factor {i+1}', fontweight='bold')
        ax.set_xlabel('Species')
        ax.set_ylabel('Contribution')
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Color bars by magnitude
        max_val = np.max(profile)
        for bar, val in zip(bars, profile):
            bar.set_color(plt.cm.viridis(val / max_val))
    
    plt.tight_layout()
    plot_path = output_dir / "factor_profiles.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    
    # Plot 2: Factor Contributions Time Series
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create time index for plotting
    time_index = pd.date_range('2023-01-01', periods=factor_contributions.shape[0], freq='H')
    
    for i in range(factor_contributions.shape[1]):
        ax.plot(time_index, factor_contributions[:, i], 
                label=f'Factor {i+1}', linewidth=2, alpha=0.8)
    
    ax.set_title('PMF Factor Contributions Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Contribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:00'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plot_path = output_dir / "factor_contributions.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    
    # Plot 3: Factor Composition (Stacked Bar)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create stacked bar plot showing contribution of each factor to each species
    bottom = np.zeros(len(feature_names))
    colors = plt.cm.Set3(np.linspace(0, 1, factor_profiles.shape[0]))
    
    for i in range(factor_profiles.shape[0]):
        ax.bar(feature_names, factor_profiles[i, :], bottom=bottom, 
               label=f'Factor {i+1}', alpha=0.8, color=colors[i])
        bottom += factor_profiles[i, :]
    
    ax.set_title('Species Composition by PMF Factors', fontsize=14, fontweight='bold')
    ax.set_xlabel('Species')
    ax.set_ylabel('Total Contribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plot_path = output_dir / "species_composition.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   ✅ Saved: {plot_path}")
    
    print(f"📊 All plots saved in: {output_dir}")

def main():
    """Main function."""
    print("🧪 ESAT PMF Test Analysis")
    print("=" * 30)
    
    # Test ESAT import
    print(f"📦 ESAT version: {esat.__version__ if hasattr(esat, '__version__') else 'Unknown'}")
    
    # Run PMF analysis
    success = run_esat_pmf_analysis()
    
    if success:
        print("\n🎉 ESAT PMF Analysis Complete!")
        print("📁 Results saved in: pmf_esat_test/")
        print("📊 Check the generated plots:")
        print("   - factor_profiles.png")
        print("   - factor_contributions.png") 
        print("   - species_composition.png")
    else:
        print("\n❌ Analysis failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
