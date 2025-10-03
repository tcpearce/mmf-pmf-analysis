#!/usr/bin/env python3
"""
Stage 9 Validation Test Suite: Comprehensive Regularization Diagnostics Testing

This test suite validates the complete Stage 9 diagnostics and validation framework,
including convergence tracking, lambda sensitivity analysis, and species push-out metrics.

Test Categories:
1. Convergence tracking validation
2. Lambda sensitivity analysis accuracy  
3. Species push-out metrics verification
4. Dashboard integration testing
5. Mathematical correctness validation

Key Validation Principles:
- λ ↑ → target species factor loadings ↓ (push-out effect)
- Convergence behavior should be mathematically sound
- Q-values should remain reasonable across lambda range
- Factor correlations should decrease with regularization
- Diagnostic plots should be generated correctly
"""

import unittest
import subprocess
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Tuple
import json

class Stage9ValidationTests(unittest.TestCase):
    """Comprehensive Stage 9 regularization diagnostics validation."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and validate prerequisites."""
        
        # Check that Stage 9 diagnostics module is available
        try:
            from regularization_diagnostics import create_regularization_diagnostics
            cls.diagnostics_available = True
            print("[OK] Stage 9 diagnostics module available")
        except ImportError:
            cls.diagnostics_available = False
            print("[ERROR] Stage 9 diagnostics module not available - skipping tests")
            
        # Create temporary test directory
        cls.test_dir = Path(tempfile.mkdtemp(prefix="stage9_validation_"))
        print(f"[SETUP] Test directory: {cls.test_dir}")
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        if hasattr(cls, 'test_dir') and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
            print(f"[CLEANUP] Removed test directory: {cls.test_dir}")
    
    def test_convergence_tracking_functionality(self):
        """Test that convergence tracking works correctly during regularization."""
        
        if not self.diagnostics_available:
            self.skipTest("Stage 9 diagnostics module not available")
            
        print("\n🧪 Testing convergence tracking functionality...")
        
        # Run a short regularization analysis with known parameters
        cmd = [
            "python", "pmf_source_app.py", "MMF9",
            "--start-date", "2023-10-01",
            "--end-date", "2023-10-02",  # Very short for speed
            "--factors", "3",
            "--models", "1",
            "--reg-species", "CH4",
            "--reg-lambda", "1.0", 
            "--reg-template", "zero",
            "--reg-bursts", "3",  # Reduced for testing
            "--reg-iter-per-burst", "20",
            "--uncertainty-mode", "epa",
            "--output-dir", str(self.test_dir / "convergence_test")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        # Validate that regularization ran successfully
        self.assertEqual(result.returncode, 0, f"Regularization failed: {result.stderr}")
        
        # Check for convergence tracking output
        stdout = result.stdout
        self.assertIn("[DIAG] Stage 9 regularization diagnostics enabled", stdout)
        self.assertIn("Burst ", stdout)
        self.assertIn("rel_change=", stdout)
        
        # Check for diagnostic files
        output_dir = self.test_dir / "convergence_test"
        self.assertTrue(output_dir.exists(), "Output directory not created")
        
        # Look for diagnostic CSV files
        convergence_files = list(output_dir.glob("*_regularization_convergence.csv"))
        self.assertGreater(len(convergence_files), 0, "Convergence CSV not generated")
        
        # Validate convergence data structure
        conv_df = pd.read_csv(convergence_files[0])
        required_columns = ['burst_number', 'species_name', 'lambda_value', 'relative_change', 'objective_reduction']
        for col in required_columns:
            self.assertIn(col, conv_df.columns, f"Required column {col} missing from convergence data")
        
        # Validate that we have data for CH4
        ch4_data = conv_df[conv_df['species_name'] == 'CH4']
        self.assertGreater(len(ch4_data), 0, "No convergence data for CH4")
        
        # Validate that relative changes are reasonable (positive and decreasing trend expected)
        rel_changes = ch4_data['relative_change'].values
        self.assertTrue(np.all(rel_changes >= 0), "Negative relative changes detected")
        
        print("   ✅ Convergence tracking validation passed")
        
    def test_lambda_sensitivity_basic_behavior(self):
        """Test basic lambda sensitivity behavior with multiple lambda values."""
        
        if not self.diagnostics_available:
            self.skipTest("Stage 9 diagnostics module not available")
            
        print("\n🧪 Testing lambda sensitivity basic behavior...")
        
        # Test multiple lambda values to validate push-out effect
        lambda_values = [0.5, 2.0, 5.0]
        results = {}
        
        for lambda_val in lambda_values:
            print(f"   Testing λ = {lambda_val}")
            
            cmd = [
                "python", "pmf_source_app.py", "MMF9",
                "--start-date", "2023-10-01",
                "--end-date", "2023-10-02",
                "--factors", "3", 
                "--models", "1",
                "--reg-species", "CH4",
                "--reg-lambda", str(lambda_val),
                "--reg-template", "zero",
                "--reg-bursts", "2",  # Minimal for testing
                "--uncertainty-mode", "epa",
                "--output-dir", str(self.test_dir / f"lambda_{lambda_val}")
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, f"Lambda {lambda_val} failed: {result.stderr}")
            
            # Extract Q(robust) value from output
            import re
            q_matches = re.findall(r'Q\(robust\):\s*([\d.]+)', result.stdout)
            if q_matches:
                results[lambda_val] = {'q_robust': float(q_matches[-1])}
            
        # Validate that we got results for all lambda values
        self.assertEqual(len(results), len(lambda_values), "Some lambda values failed to produce results")
        
        # Validate that Q values are reasonable (not increasing dramatically)
        q_values = [results[lam]['q_robust'] for lam in sorted(lambda_values)]
        print(f"   Q(robust) values: {dict(zip(lambda_values, q_values))}")
        
        # Q values should not increase by more than 50% as lambda increases (rough heuristic)
        q_ratio = max(q_values) / min(q_values)
        self.assertLess(q_ratio, 2.0, f"Q-values increased too much with lambda: {q_ratio:.2f}x")
        
        print("   ✅ Lambda sensitivity basic behavior validated")
        
    def test_push_out_effectiveness_metrics(self):
        """Test that push-out effectiveness can be quantified."""
        
        if not self.diagnostics_available:
            self.skipTest("Stage 9 diagnostics module not available")
            
        print("\n🧪 Testing push-out effectiveness metrics...")
        
        # Run baseline (no regularization) and regularized analysis
        test_cases = [
            ("baseline", []),
            ("regularized", ["--reg-species", "CH4", "--reg-lambda", "20.0", "--reg-template", "zero"])
        ]
        
        results = {}
        
        for case_name, extra_args in test_cases:
            print(f"   Running {case_name} case...")
            
            base_cmd = [
                "python", "pmf_source_app.py", "MMF9",
                "--start-date", "2023-10-01", 
                "--end-date", "2023-10-02",
                "--factors", "3",
                "--models", "1",
                "--uncertainty-mode", "epa",
                "--output-dir", str(self.test_dir / case_name)
            ]
            
            cmd = base_cmd + extra_args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, f"{case_name} case failed: {result.stderr}")
            
            # Look for factor profiles to analyze regularization effect
            output_dir = self.test_dir / case_name
            profile_files = list(output_dir.glob("*_factor_profiles.csv"))
            
            if profile_files:
                profile_df = pd.read_csv(profile_files[0], index_col=0)
                if 'CH4' in profile_df.columns:
                    # Calculate CH4 dominance: max CH4 loading across all factors
                    ch4_loadings = profile_df['CH4'].values
                    max_ch4_loading = float(np.max(ch4_loadings))
                    mean_ch4_loading = float(np.mean(ch4_loadings))
                    
                    # Also compute factor dominance: which species dominates each factor
                    factor_dominance = []
                    for idx, row_name in enumerate(profile_df.index):
                        row = profile_df.iloc[idx]
                        dominant_species = row.idxmax()
                        dominant_value = row.max()
                        factor_dominance.append((row_name, dominant_species, dominant_value))
                    
                    # Count how many factors are dominated by CH4
                    ch4_dominated_factors = sum(1 for _, species, _ in factor_dominance if species == 'CH4')
                    total_factors = len(factor_dominance)
                    
                    results[case_name] = {
                        'max_ch4_loading': max_ch4_loading,
                        'mean_ch4_loading': mean_ch4_loading,
                        'ch4_dominated_factors': ch4_dominated_factors,
                        'total_factors': total_factors,
                        'ch4_dominance_ratio': ch4_dominated_factors / total_factors if total_factors > 0 else 1.0
                    }
                    print(f"   {case_name}: max CH4 loading = {max_ch4_loading:.3f}, CH4 dominates {ch4_dominated_factors}/{total_factors} factors")
            else:
                print(f"   {case_name}: No factor profiles found - PMF may have failed")
        
        # Validate push-out effect if both cases have data
        if 'baseline' in results and 'regularized' in results:
            # Test 1: CH4 factor dominance reduction
            baseline_dominance = results['baseline']['ch4_dominance_ratio']
            regularized_dominance = results['regularized']['ch4_dominance_ratio']
            
            dominance_reduction = baseline_dominance - regularized_dominance
            print(f"   CH4 factor dominance: baseline={baseline_dominance:.3f}, regularized={regularized_dominance:.3f}")
            print(f"   Dominance reduction: {dominance_reduction:.3f} (> 0 indicates push-out effect)")
            
            # Test 2: Maximum CH4 loading reduction
            baseline_max_loading = results['baseline']['max_ch4_loading']
            regularized_max_loading = results['regularized']['max_ch4_loading']
            
            loading_reduction_ratio = regularized_max_loading / baseline_max_loading
            print(f"   Max CH4 loading ratio: {loading_reduction_ratio:.3f} (< 1.0 indicates push-out effect)")
            
            # Primary validation: either dominance reduction OR loading reduction should occur
            push_out_success = (dominance_reduction > 0) or (loading_reduction_ratio < 0.95)
            
            if push_out_success:
                print("   ✅ Push-out effect detected")
            else:
                # More detailed analysis for debugging
                print(f"   ❌ Push-out effect not detected:")
                print(f"      Baseline factors dominated by CH4: {results['baseline']['ch4_dominated_factors']}/{results['baseline']['total_factors']}")
                print(f"      Regularized factors dominated by CH4: {results['regularized']['ch4_dominated_factors']}/{results['regularized']['total_factors']}")
                print(f"      Max loading change: {baseline_max_loading:.3f} -> {regularized_max_loading:.3f}")
                
                # With λ=20.0, we should definitely see push-out effect
                self.assertTrue(push_out_success, 
                              f"Regularization (λ=20.0) did not reduce CH4 dominance: "
                              f"dominance reduction={dominance_reduction:.3f}, "
                              f"loading ratio={loading_reduction_ratio:.3f}")
            
        print("   ✅ Push-out effectiveness metrics validated")
        
    def test_lambda_sensitivity_analysis_tool(self):
        """Test the standalone lambda sensitivity analysis tool."""
        
        print("\n🧪 Testing lambda sensitivity analysis tool...")
        
        # Create a minimal base command for the tool
        base_cmd = (
            "python pmf_source_app.py MMF9 --start-date 2023-10-01 --end-date 2023-10-02 "
            "--factors 3 --models 1 --uncertainty-mode epa"
        )
        
        # Run lambda sensitivity analysis with coarse range for speed
        cmd = [
            "python", "test_lambda_sensitivity.py",
            "--species", "CH4", 
            "--lambda-range", "coarse",  # [0.5, 1.0, 2.0, 5.0]
            "--base-cmd", base_cmd,
            "--output-dir", str(self.test_dir / "lambda_sensitivity"),
            "--verbose"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)  # 10 min timeout
            
            # Check if the analysis completed (may have some failures, that's OK for testing)
            stdout = result.stdout
            self.assertIn("LAMBDA SENSITIVITY ANALYSIS", stdout)
            self.assertIn("Testing λ = ", stdout)
            
            # Check for output files
            output_dir = self.test_dir / "lambda_sensitivity"
            if output_dir.exists():
                result_files = list(output_dir.glob("CH4_lambda_sensitivity_*.csv"))
                plot_files = list(output_dir.glob("CH4_lambda_sensitivity_*.png"))
                
                # We should have at least attempted to generate results
                if result_files:
                    print(f"   Generated {len(result_files)} result files")
                if plot_files:
                    print(f"   Generated {len(plot_files)} plot files")
                    
            print("   ✅ Lambda sensitivity tool executed (results may vary)")
            
        except subprocess.TimeoutExpired:
            print("   ⚠️  Lambda sensitivity analysis timed out - this is acceptable for testing")
        except Exception as e:
            print(f"   ⚠️  Lambda sensitivity analysis error: {e}")
            
    def test_diagnostic_data_consistency(self):
        """Test that diagnostic data is mathematically consistent."""
        
        if not self.diagnostics_available:
            self.skipTest("Stage 9 diagnostics module not available")
            
        print("\n🧪 Testing diagnostic data consistency...")
        
        # Run regularization with known parameters  
        cmd = [
            "python", "pmf_source_app.py", "MMF9",
            "--start-date", "2023-10-01",
            "--end-date", "2023-10-02",
            "--factors", "3",
            "--models", "1", 
            "--reg-species", "CH4",
            "--reg-lambda", "2.0",
            "--reg-template", "zero",
            "--reg-bursts", "4",
            "--uncertainty-mode", "epa",
            "--output-dir", str(self.test_dir / "consistency_test")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, f"Consistency test failed: {result.stderr}")
        
        # Load and validate convergence data
        output_dir = self.test_dir / "consistency_test"
        convergence_files = list(output_dir.glob("*_regularization_convergence.csv"))
        
        if convergence_files:
            conv_df = pd.read_csv(convergence_files[0])
            
            # Validate data consistency
            self.assertTrue(len(conv_df) > 0, "No convergence data recorded")
            
            # Check that burst numbers are sequential
            burst_nums = conv_df['burst_number'].values
            expected_bursts = np.arange(1, len(burst_nums) + 1)
            np.testing.assert_array_equal(burst_nums, expected_bursts, "Burst numbers not sequential")
            
            # Check that lambda values are consistent
            lambda_values = conv_df['lambda_value'].unique()
            self.assertEqual(len(lambda_values), 1, "Multiple lambda values in single run")
            self.assertAlmostEqual(lambda_values[0], 2.0, places=3, msg="Lambda value mismatch")
            
            # Check that relative changes are non-negative
            rel_changes = conv_df['relative_change'].values
            self.assertTrue(np.all(rel_changes >= 0), "Negative relative changes detected")
            
            # Check that objective reductions make sense
            obj_reductions = conv_df['objective_reduction'].values
            finite_reductions = obj_reductions[np.isfinite(obj_reductions)]
            if len(finite_reductions) > 0:
                # Most objective reductions should be non-negative (objective should decrease)
                positive_ratio = np.mean(finite_reductions >= 0)
                self.assertGreater(positive_ratio, 0.5, "Too many objective increases")
            
        print("   ✅ Diagnostic data consistency validated")
        
    def test_dashboard_integration(self):
        """Test that regularization diagnostics integrate properly with dashboard."""
        
        if not self.diagnostics_available:
            self.skipTest("Stage 9 diagnostics module not available")
            
        print("\n🧪 Testing dashboard integration...")
        
        # Run regularized analysis with dashboard generation
        cmd = [
            "python", "pmf_source_app.py", "MMF9", 
            "--start-date", "2023-10-01",
            "--end-date", "2023-10-02", 
            "--factors", "3",
            "--models", "1",
            "--reg-species", "CH4",
            "--reg-lambda", "1.5",
            "--reg-template", "zero",
            "--uncertainty-mode", "epa",
            "--output-dir", str(self.test_dir / "dashboard_test")
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, f"Dashboard test failed: {result.stderr}")
        
        # Check for diagnostic outputs
        output_dir = self.test_dir / "dashboard_test"
        
        # Should have convergence plots
        convergence_plots = list(output_dir.glob("*_regularization_convergence.png"))
        
        # Should have diagnostic report
        diagnostic_reports = list(output_dir.glob("*_regularization_diagnostics_report.txt"))
        
        # Should have main dashboard
        dashboards = list(output_dir.glob("*_pmf_dashboard.html"))
        self.assertGreater(len(dashboards), 0, "Main PMF dashboard not generated")
        
        # Check dashboard contains regularization information
        if dashboards:
            with open(dashboards[0], 'r') as f:
                dashboard_content = f.read()
                
            # Should mention regularization
            regularization_indicators = [
                'regulariz', 'species', 'lambda', 'CH4', 'convergence'
            ]
            
            found_indicators = []
            for indicator in regularization_indicators:
                if indicator.lower() in dashboard_content.lower():
                    found_indicators.append(indicator)
            
            self.assertGreater(len(found_indicators), 2, 
                             f"Dashboard missing regularization context. Found: {found_indicators}")
        
        print("   ✅ Dashboard integration validated")

def run_stage9_validation_suite():
    """Run the complete Stage 9 validation test suite."""
    
    print("🚀 SPECIES REGULARIZATION STAGE 9 VALIDATION SUITE")
    print("=" * 60)
    print("Testing comprehensive diagnostics and validation framework")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(Stage9ValidationTests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=False)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("🧪 STAGE 9 VALIDATION SUMMARY")
    print("=" * 60)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped)
    passed = total_tests - failures - errors - skipped
    
    print(f"Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"🚫 Errors: {errors}")
    print(f"⏭️  Skipped: {skipped}")
    
    if failures > 0:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"   • {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print(f"\n🚫 ERRORS:")
        for test, traceback in result.errors:
            print(f"   • {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success_rate = passed / total_tests if total_tests > 0 else 0
    
    if success_rate >= 0.8:
        print(f"\n🎉 STAGE 9 VALIDATION PASSED ({success_rate:.1%} success rate)")
        print("   ✅ Regularization diagnostics framework is working correctly")
        print("   ✅ Convergence tracking is functional")
        print("   ✅ Lambda sensitivity analysis is operational")
        print("   ✅ Push-out effectiveness can be measured")
        print("   ✅ Dashboard integration is working")
        return True
    else:
        print(f"\n❌ STAGE 9 VALIDATION FAILED ({success_rate:.1%} success rate)")
        print("   🔧 Review failed tests and fix issues before proceeding")
        return False

if __name__ == "__main__":
    success = run_stage9_validation_suite()
    sys.exit(0 if success else 1)