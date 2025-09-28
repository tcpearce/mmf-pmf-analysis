#!/usr/bin/env python3
"""
Unit tests for species weighting functionality in PMF analysis.

Tests the CLI parsing, uncertainty application logic, and interaction with S/N categorization.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add current directory to path to import pmf_source_app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pmf_source_app import MMFPMFAnalyzer

class TestSpeciesWeightParsing:
    """Test the parsing of --species-weight CLI arguments."""
    
    def test_parse_single_species_weight(self):
        """Test parsing a single species weight specification."""
        analyzer = MMFPMFAnalyzer(species_weight=['CH4=5'])
        
        expected = {'CH4': 5.0}
        assert analyzer.species_weight_dict == expected
    
    def test_parse_multiple_species_single_flag(self):
        """Test parsing multiple species in a single flag with comma separation."""
        analyzer = MMFPMFAnalyzer(species_weight=['CH4=5,H2S=2.5'])
        
        expected = {'CH4': 5.0, 'H2S': 2.5}
        assert analyzer.species_weight_dict == expected
    
    def test_parse_multiple_flags(self):
        """Test parsing multiple --species-weight flags."""
        analyzer = MMFPMFAnalyzer(species_weight=['CH4=5', 'H2S=2.5', 'NO2=1.5'])
        
        expected = {'CH4': 5.0, 'H2S': 2.5, 'NO2': 1.5}
        assert analyzer.species_weight_dict == expected
    
    def test_case_insensitive_parsing(self):
        """Test that species names are stored case-insensitively (uppercase keys)."""
        analyzer = MMFPMFAnalyzer(species_weight=['ch4=5', 'H2s=2.5', 'nO2=1.5'])
        
        expected = {'CH4': 5.0, 'H2S': 2.5, 'NO2': 1.5}
        assert analyzer.species_weight_dict == expected
    
    def test_duplicate_species_latest_wins(self):
        """Test that when a species appears multiple times, the latest value wins."""
        analyzer = MMFPMFAnalyzer(species_weight=['CH4=5', 'CH4=10', 'CH4=3'])
        
        expected = {'CH4': 3.0}
        assert analyzer.species_weight_dict == expected
    
    def test_invalid_specifications_skipped(self):
        """Test that invalid specifications are skipped with warnings."""
        # This should only keep the valid CH4=5 specification
        analyzer = MMFPMFAnalyzer(species_weight=['CH4=5', 'INVALID', 'H2S=', '=2.5', 'NO2=abc'])
        
        expected = {'CH4': 5.0}
        assert analyzer.species_weight_dict == expected
    
    def test_zero_negative_weights_rejected(self):
        """Test that zero and negative weights are rejected."""
        analyzer = MMFPMFAnalyzer(species_weight=['CH4=5', 'H2S=0', 'NO2=-1.5'])
        
        expected = {'CH4': 5.0}  # Only positive weight should be kept
        assert analyzer.species_weight_dict == expected
    
    def test_empty_species_weight_list(self):
        """Test that empty or None species_weight lists result in empty dict."""
        analyzer1 = MMFPMFAnalyzer(species_weight=[])
        analyzer2 = MMFPMFAnalyzer(species_weight=None)
        
        assert analyzer1.species_weight_dict == {}
        assert analyzer2.species_weight_dict == {}


class TestSpeciesWeightApplication:
    """Test the application of species weights to uncertainty matrices."""
    
    def setup_method(self):
        """Set up test data for each test."""
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
        self.output_path = Path(self.temp_dir)
        
        # Create synthetic concentration and uncertainty data
        self.species_names = ['CH4', 'H2S', 'NO2', 'PM1', 'PM2.5']
        self.n_samples = 100
        
        # Create synthetic data
        np.random.seed(42)
        self.concentration_data = pd.DataFrame(
            np.random.lognormal(mean=2.0, sigma=1.0, size=(self.n_samples, len(self.species_names))),
            columns=self.species_names
        )
        
        self.uncertainty_data = pd.DataFrame(
            np.random.lognormal(mean=0.5, sigma=0.3, size=(self.n_samples, len(self.species_names))),
            columns=self.species_names
        )
    
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_species_weight_application_simple(self):
        """Test that species weights are correctly applied to uncertainty matrix."""
        analyzer = MMFPMFAnalyzer(
            output_dir=str(self.output_path),
            species_weight=['CH4=5', 'H2S=2']
        )
        
        # Set up the test data
        analyzer.concentration_data = self.concentration_data.copy()
        analyzer.uncertainty_data = self.uncertainty_data.copy()
        
        # Store original uncertainties for comparison
        original_ch4_uncertainty = analyzer.uncertainty_data['CH4'].copy()
        original_h2s_uncertainty = analyzer.uncertainty_data['H2S'].copy()
        original_no2_uncertainty = analyzer.uncertainty_data['NO2'].copy()
        
        # Apply species weighting
        analyzer._apply_species_weighting()
        
        # Check that weights were applied correctly
        np.testing.assert_array_almost_equal(
            analyzer.uncertainty_data['CH4'], 
            original_ch4_uncertainty * 5.0
        )
        np.testing.assert_array_almost_equal(
            analyzer.uncertainty_data['H2S'], 
            original_h2s_uncertainty * 2.0
        )
        # NO2 should be unchanged
        np.testing.assert_array_equal(
            analyzer.uncertainty_data['NO2'], 
            original_no2_uncertainty
        )
        
        # Check that tracking attributes are set
        assert hasattr(analyzer, '_species_weights_applied')
        assert hasattr(analyzer, '_species_weights_not_found')
        assert 'CH4' in analyzer._species_weights_applied
        assert 'H2S' in analyzer._species_weights_applied
        assert analyzer._species_weights_applied['CH4'] == 5.0
        assert analyzer._species_weights_applied['H2S'] == 2.0
    
    def test_species_not_found_handling(self):
        """Test that species not found in data are tracked appropriately."""
        analyzer = MMFPMFAnalyzer(
            output_dir=str(self.output_path),
            species_weight=['CH4=5', 'NONEXISTENT=10', 'MISSING=3']
        )
        
        # Set up the test data
        analyzer.concentration_data = self.concentration_data.copy()
        analyzer.uncertainty_data = self.uncertainty_data.copy()
        
        # Apply species weighting
        analyzer._apply_species_weighting()
        
        # Check that existing species was applied
        assert 'CH4' in analyzer._species_weights_applied
        assert analyzer._species_weights_applied['CH4'] == 5.0
        
        # Check that non-existent species are tracked
        assert 'NONEXISTENT' in analyzer._species_weights_not_found
        assert 'MISSING' in analyzer._species_weights_not_found
        assert analyzer._species_weights_not_found['NONEXISTENT'] == 10.0
        assert analyzer._species_weights_not_found['MISSING'] == 3.0
    
    def test_species_weights_csv_creation(self):
        """Test that species_weights.csv is created with correct content."""
        analyzer = MMFPMFAnalyzer(
            output_dir=str(self.output_path),
            species_weight=['CH4=5', 'NONEXISTENT=10']
        )
        analyzer.filename_prefix = 'test'
        
        # Set up the test data
        analyzer.concentration_data = self.concentration_data.copy()
        analyzer.uncertainty_data = self.uncertainty_data.copy()
        
        # Apply species weighting
        analyzer._apply_species_weighting()
        
        # Check that CSV file was created
        csv_file = self.output_path / 'test_species_weights.csv'
        assert csv_file.exists()
        
        # Read and verify CSV content
        weights_df = pd.read_csv(csv_file)
        
        # Should have 2 rows
        assert len(weights_df) == 2
        
        # Check applied weight
        ch4_row = weights_df[weights_df['species'] == 'CH4'].iloc[0]
        assert ch4_row['multiplier'] == 5.0
        assert ch4_row['was_present'] == True
        assert ch4_row['applied'] == True
        
        # Check not found weight
        missing_row = weights_df[weights_df['species'] == 'NONEXISTENT'].iloc[0]
        assert missing_row['multiplier'] == 10.0
        assert missing_row['was_present'] == False
        assert missing_row['applied'] == False


class TestSpeciesWeightIntegration:
    """Test integration with S/N categorization and other pipeline components."""
    
    def test_species_weight_after_snr_categorization(self):
        """Test that species weights are applied after S/N categorization adjustments."""
        # This would require setting up S/N categorization mocks
        # For now, just test that the method exists and can be called
        analyzer = MMFPMFAnalyzer(species_weight=['CH4=5'])
        
        # Create minimal test data
        analyzer.concentration_data = pd.DataFrame({
            'CH4': [1.0, 2.0, 3.0],
            'H2S': [0.1, 0.2, 0.3]
        })
        analyzer.uncertainty_data = pd.DataFrame({
            'CH4': [0.1, 0.2, 0.3],
            'H2S': [0.01, 0.02, 0.03]
        })
        analyzer.output_dir = Path(tempfile.mkdtemp())
        analyzer.filename_prefix = 'test'
        
        # Should be able to apply weighting without error
        analyzer._apply_species_weighting()
        
        # Clean up
        import shutil
        shutil.rmtree(analyzer.output_dir, ignore_errors=True)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])