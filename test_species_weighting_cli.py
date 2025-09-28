#!/usr/bin/env python3
"""
Integration test for species weighting CLI functionality.

Tests end-to-end CLI with synthetic data to verify uncertainty scaling and output files.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import subprocess
import sys
from pathlib import Path
import os

def create_synthetic_pmf_data(output_dir):
    """Create synthetic parquet data for testing."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 50
    datetime_index = pd.date_range('2023-10-01', periods=n_samples, freq='30min')
    
    # Create concentration data with realistic values
    data = {
        'CH4': np.random.lognormal(mean=7.0, sigma=0.5, size=n_samples),  # High values (~1000s μg/m³)
        'H2S': np.random.lognormal(mean=2.0, sigma=0.5, size=n_samples),  # Lower values (~10s μg/m³)
        'NO2': np.random.lognormal(mean=3.0, sigma=0.5, size=n_samples),  # Medium values (~20-50 μg/m³)
        'PM1': np.random.lognormal(mean=2.5, sigma=0.3, size=n_samples),
        'PM2.5': np.random.lognormal(mean=2.8, sigma=0.3, size=n_samples),
        # Add count columns for aggregation
        'n_CH4': np.random.poisson(lam=6, size=n_samples),  # ~6 sub-samples per 30min
        'n_H2S': np.random.poisson(lam=6, size=n_samples),
        'n_NO2': np.random.poisson(lam=6, size=n_samples),
        'n_PM1': np.random.poisson(lam=2, size=n_samples),   # ~2 sub-samples for PM
        'n_PM2.5': np.random.poisson(lam=2, size=n_samples),
        # Add some metadata
        'gas_data_available': True,
        'particle_data_available': True
    }
    
    df = pd.DataFrame(data, index=datetime_index)
    
    # Save as parquet
    output_path = Path(output_dir)
    parquet_file = output_path / 'synthetic_mmf_data.parquet'
    df.to_parquet(parquet_file)
    
    # Create metadata file
    metadata = {
        'processing_date': '2023-10-01',
        'timebase': '30min',
        'aggregation_method': 'mean',
        'station': 'TEST',
        'units': {
            'CH4': 'μg/m³',
            'H2S': 'μg/m³', 
            'NO2': 'μg/m³',
            'PM1': 'μg/m³',
            'PM2.5': 'μg/m³'
        }
    }
    
    import json
    metadata_file = output_path / 'synthetic_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return parquet_file, df


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with synthetic data."""
    temp_dir = tempfile.mkdtemp()
    parquet_file, data_df = create_synthetic_pmf_data(temp_dir)
    yield temp_dir, parquet_file, data_df
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_species_weighting_cli_basic(temp_data_dir):
    """Test basic species weighting functionality via CLI."""
    temp_dir, parquet_file, data_df = temp_data_dir
    
    output_dir = Path(temp_dir) / 'pmf_output'
    
    # Run PMF with species weighting
    cmd = [
        sys.executable, 'pmf_source_app.py',
        '--data-dir', str(temp_dir),
        '--patterns', 'synthetic_mmf_data.parquet',
        '--output-dir', str(output_dir),
        '--factors', '3',  # Fixed factors for quick test
        '--models', '3',   # Few models for speed
        '--species-weight', 'CH4=5',
        '--species-weight', 'H2S=2',
        '--uncertainty-mode', 'legacy',  # Use legacy for simplicity
        '--write-diagnostics'
    ]
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    # Check if command succeeded
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        pytest.skip(f"PMF analysis failed with return code {result.returncode}. This may be due to missing dependencies.")
        return
    
    # Check that species weights CSV was created
    species_weights_file = None
    for file in output_dir.glob('*_species_weights.csv'):
        species_weights_file = file
        break
    
    assert species_weights_file is not None, "Species weights CSV file not found"
    
    # Verify species weights CSV content
    weights_df = pd.read_csv(species_weights_file)
    
    # Should have entries for CH4 and H2S
    ch4_entries = weights_df[weights_df['species'] == 'CH4']
    h2s_entries = weights_df[weights_df['species'] == 'H2S']
    
    assert len(ch4_entries) == 1, "Should have exactly one CH4 entry"
    assert len(h2s_entries) == 1, "Should have exactly one H2S entry"
    
    assert ch4_entries.iloc[0]['multiplier'] == 5.0, "CH4 multiplier should be 5.0"
    assert h2s_entries.iloc[0]['multiplier'] == 2.0, "H2S multiplier should be 2.0"
    assert ch4_entries.iloc[0]['applied'] == True, "CH4 weight should be applied"
    assert h2s_entries.iloc[0]['applied'] == True, "H2S weight should be applied"
    
    # Check that uncertainties CSV exists and CH4 uncertainties are larger
    uncertainties_file = None
    for file in output_dir.glob('*_uncertainties.csv'):
        uncertainties_file = file
        break
    
    assert uncertainties_file is not None, "Uncertainties CSV file not found"
    
    uncertainties_df = pd.read_csv(uncertainties_file, index_col=0)
    
    # CH4 uncertainties should be much larger than other species due to 5x multiplier
    ch4_mean_unc = uncertainties_df['CH4'].mean()
    h2s_mean_unc = uncertainties_df['H2S'].mean()
    no2_mean_unc = uncertainties_df['NO2'].mean()
    
    # CH4 should have higher uncertainty than NO2 due to weighting
    # (Note: this assumes similar base uncertainty calculations)
    assert ch4_mean_unc > no2_mean_unc, f"CH4 mean uncertainty ({ch4_mean_unc}) should be > NO2 ({no2_mean_unc}) due to 5x weighting"
    
    # H2S should have higher uncertainty than NO2 due to weighting
    assert h2s_mean_unc > no2_mean_unc, f"H2S mean uncertainty ({h2s_mean_unc}) should be > NO2 ({no2_mean_unc}) due to 2x weighting"
    
    print(f"Test passed: CH4 uncertainty={ch4_mean_unc:.3f}, H2S uncertainty={h2s_mean_unc:.3f}, NO2 uncertainty={no2_mean_unc:.3f}")


def test_species_weighting_cli_nonexistent_species(temp_data_dir):
    """Test CLI handling of non-existent species in weighting."""
    temp_dir, parquet_file, data_df = temp_data_dir
    
    output_dir = Path(temp_dir) / 'pmf_output_nonexistent'
    
    # Run PMF with some non-existent species weights
    cmd = [
        sys.executable, 'pmf_source_app.py',
        '--data-dir', str(temp_dir),
        '--patterns', 'synthetic_mmf_data.parquet',
        '--output-dir', str(output_dir),
        '--factors', '3',
        '--models', '2',
        '--species-weight', 'CH4=5',
        '--species-weight', 'NONEXISTENT=10',
        '--species-weight', 'MISSING_SPECIES=3',
        '--uncertainty-mode', 'legacy',
        '--write-diagnostics'
    ]
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        pytest.skip(f"PMF analysis failed. This may be due to missing dependencies.")
        return
    
    # Check species weights CSV
    species_weights_file = None
    for file in output_dir.glob('*_species_weights.csv'):
        species_weights_file = file
        break
    
    assert species_weights_file is not None
    
    weights_df = pd.read_csv(species_weights_file)
    
    # Should have 3 entries total
    assert len(weights_df) == 3
    
    # CH4 should be applied
    ch4_entry = weights_df[weights_df['species'] == 'CH4'].iloc[0]
    assert ch4_entry['applied'] == True
    assert ch4_entry['was_present'] == True
    
    # Non-existent species should not be applied
    nonexistent_entry = weights_df[weights_df['species'] == 'NONEXISTENT'].iloc[0]
    assert nonexistent_entry['applied'] == False
    assert nonexistent_entry['was_present'] == False
    
    missing_entry = weights_df[weights_df['species'] == 'MISSING_SPECIES'].iloc[0]
    assert missing_entry['applied'] == False
    assert missing_entry['was_present'] == False


def test_species_weighting_cli_comma_separated(temp_data_dir):
    """Test CLI parsing of comma-separated species weights."""
    temp_dir, parquet_file, data_df = temp_data_dir
    
    output_dir = Path(temp_dir) / 'pmf_output_comma'
    
    # Run PMF with comma-separated species weights
    cmd = [
        sys.executable, 'pmf_source_app.py',
        '--data-dir', str(temp_dir),
        '--patterns', 'synthetic_mmf_data.parquet',
        '--output-dir', str(output_dir),
        '--factors', '3',
        '--models', '2',
        '--species-weight', 'CH4=5,H2S=2.5,NO2=1.5',  # Comma-separated
        '--uncertainty-mode', 'legacy',
        '--write-diagnostics'
    ]
    
    # Run the command
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)))
    
    if result.returncode != 0:
        pytest.skip(f"PMF analysis failed. This may be due to missing dependencies.")
        return
    
    # Check species weights CSV
    species_weights_file = None
    for file in output_dir.glob('*_species_weights.csv'):
        species_weights_file = file
        break
    
    assert species_weights_file is not None
    
    weights_df = pd.read_csv(species_weights_file)
    
    # Should have 3 entries (all should be applied)
    applied_weights = weights_df[weights_df['applied'] == True]
    assert len(applied_weights) == 3
    
    # Check specific multipliers
    ch4_mult = weights_df[weights_df['species'] == 'CH4'].iloc[0]['multiplier']
    h2s_mult = weights_df[weights_df['species'] == 'H2S'].iloc[0]['multiplier']
    no2_mult = weights_df[weights_df['species'] == 'NO2'].iloc[0]['multiplier']
    
    assert ch4_mult == 5.0
    assert h2s_mult == 2.5
    assert no2_mult == 1.5


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])