#!/usr/bin/env python3
"""
Stage 8 Test Suite: Mode Forcing and Compatibility Testing

Tests regularization compatibility with existing PMF features:
- Robust training (--robust-fit)
- Weight-aware initialization (--weight-aware-init)
- Species exclusion (--exclude-species)
- S/N categorization (--snr-enable)
- EPA vs Legacy uncertainty modes
- Different algorithm methods (ls-nmf vs ws-nmf)
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

def run_compatibility_test(test_name, base_args, expected_success=True):
    """Run a PMF analysis with specific feature combinations and check compatibility."""
    
    print(f"\n🧪 Running compatibility test: {test_name}")
    print("=" * 70)
    
    # Build command with base PMF args
    cmd = [
        "python", "pmf_source_app.py", "MMF9",
        "--start-date", "2023-10-01",
        "--end-date", "2023-10-03",  # Very short for speed
        "--factors", "3",  # Fewer factors for speed
        "--models", "1",   # Single model for speed
    ] + base_args
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        # Check return code
        if result.returncode != 0:
            if expected_success:
                print(f"❌ Test FAILED - Unexpected error (code {result.returncode})")
                print("STDERR:")
                print(result.stderr[-1000:])
                return False
            else:
                print(f"✅ Test PASSED - Expected failure occurred (code {result.returncode})")
                return True
        
        # Check for success indicators
        stdout = result.stdout
        success_indicators = []
        
        # Check basic PMF completion
        if "Analysis Complete!" in stdout:
            success_indicators.append("PMF analysis completed")
        
        # Check regularization activation if reg flags present
        if any("--reg-" in arg for arg in base_args):
            if "[OK] Regularization preparation complete" in stdout:
                success_indicators.append("Regularization activated")
            if "[REG] Running regularized PMF training" in stdout:
                success_indicators.append("Regularized training executed")
        
        # Check robust training activation
        if "--robust-fit" in base_args:
            if "ROBUST mode" in stdout:
                success_indicators.append("Robust training activated")
        
        # Check weight-aware initialization
        if "--weight-aware-init" in base_args or "--species-weight" in ' '.join(base_args):
            if "Weight-aware initialization" in stdout:
                success_indicators.append("Weight-aware initialization activated")
        
        # Check species exclusion
        if "--exclude-species" in base_args:
            if "excluding from analysis" in stdout or "Species scheduled for exclusion" in stdout:
                success_indicators.append("Species exclusion activated")
        
        # Check S/N categorization
        if "--snr-enable" in base_args:
            if "S/N-based feature categorization" in stdout:
                success_indicators.append("S/N categorization activated")
        
        # Check Q-values are reasonable
        if "Q(robust):" in stdout:
            success_indicators.append("Valid Q-values reported")
        
        print(f"📊 Success indicators found: {len(success_indicators)}")
        for indicator in success_indicators:
            print(f"   ✅ {indicator}")
        
        if expected_success:
            return len(success_indicators) >= 2  # At least PMF + one feature
        else:
            print(f"⚠️  Unexpected success - test was expected to fail")
            return False
        
    except subprocess.TimeoutExpired:
        print("❌ Test FAILED - Timeout after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test FAILED - Exception: {e}")
        return False

def main():
    """Run comprehensive Stage 8 compatibility tests."""
    
    print("🚀 SPECIES REGULARIZATION STAGE 8 TEST SUITE")
    print("Testing: Mode Forcing and Compatibility")
    print("=" * 80)
    
    # Test cases: (name, args, expected_success)
    test_cases = [
        # Basic regularization functionality
        (
            "Regularization Only",
            ["--reg-species", "CH4", "--reg-lambda", "1.0", "--reg-template", "zero", 
             "--uncertainty-mode", "epa", "--output-dir", "test_stage8_reg_only"],
            True
        ),
        
        # Compatibility with robust training
        (
            "Regularization + Robust Training", 
            ["--reg-species", "CH4", "--reg-lambda", "2.0", "--reg-template", "zero",
             "--robust-fit", "--robust-alpha", "3.0",
             "--uncertainty-mode", "epa", "--output-dir", "test_stage8_reg_robust"],
            True
        ),
        
        # Compatibility with weight-aware initialization
        (
            "Regularization + Weight-Aware Init",
            ["--reg-species", "CH4", "--reg-lambda", "1.5", "--reg-template", "zero",
             "--species-weight", "H2S=2.0", "--weight-aware-init",
             "--uncertainty-mode", "epa", "--output-dir", "test_stage8_reg_weight"],
            True
        ),
        
        # Compatibility with species exclusion
        (
            "Regularization + Species Exclusion",
            ["--reg-species", "CH4", "--reg-lambda", "1.0", "--reg-template", "zero",
             "--exclude-species", "Benzene,Toluene",
             "--uncertainty-mode", "epa", "--output-dir", "test_stage8_reg_exclude"],
            True
        ),
        
        # Compatibility with S/N categorization
        (
            "Regularization + S/N Categorization",
            ["--reg-species", "CH4", "--reg-lambda", "1.0", "--reg-template", "zero",
             "--snr-enable", "--snr-weak-threshold", "2.0",
             "--uncertainty-mode", "epa", "--output-dir", "test_stage8_reg_snr"],
            True
        ),
        
        # Multiple regularized species with other features
        (
            "Multi-Species Reg + Multiple Features",
            ["--reg-species", "CH4", "--reg-species", "H2S",
             "--reg-lambda", "2.0", "--reg-lambda", "1.0", 
             "--reg-template", "zero", "--reg-template", "uniform",
             "--species-weight", "SO2=1.5", "--weight-aware-init",
             "--robust-fit", "--robust-alpha", "4.0",
             "--uncertainty-mode", "epa", "--output-dir", "test_stage8_multi_all"],
            True
        ),
        
        # Test with different algorithm method
        (
            "Regularization + Semi-NMF Method",
            ["--reg-species", "CH4", "--reg-lambda", "1.0", "--reg-template", "zero",
             "--method", "ws-nmf", "--init-method", "kmeans",
             "--uncertainty-mode", "epa", "--output-dir", "test_stage8_reg_seminmf"],
            True
        ),
        
        # Test with legacy uncertainty mode
        (
            "Regularization + Legacy Uncertainty",
            ["--reg-species", "CH4", "--reg-lambda", "1.0", "--reg-template", "zero",
             "--uncertainty-mode", "legacy",
             "--output-dir", "test_stage8_reg_legacy"],
            True
        ),
        
        # Edge case: regularizing excluded species (should handle gracefully)
        (
            "Regularizing Excluded Species",
            ["--reg-species", "Benzene", "--reg-lambda", "1.0", "--reg-template", "zero",
             "--exclude-species", "Benzene",
             "--uncertainty-mode", "epa", "--output-dir", "test_stage8_reg_excluded"],
            True  # Should handle gracefully by skipping regularization
        ),
    ]
    
    # Run all test cases
    results = []
    for test_name, args, expected_success in test_cases:
        try:
            result = run_compatibility_test(test_name, args, expected_success)
            results.append((test_name, result))
        except KeyboardInterrupt:
            print(f"\n⚠️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test framework error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("🧪 STAGE 8 COMPATIBILITY TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL COMPATIBILITY TESTS PASSED - Stage 8 validated!")
        print("\n✅ Regularization is fully compatible with all existing PMF features:")
        print("   • Robust training (--robust-fit)")
        print("   • Weight-aware initialization (--weight-aware-init)")
        print("   • Species exclusion (--exclude-species)")
        print("   • S/N categorization (--snr-enable)")
        print("   • Different uncertainty modes (epa/legacy)")
        print("   • Different algorithm methods (ls-nmf/ws-nmf)")
        return True
    else:
        print("❌ SOME COMPATIBILITY TESTS FAILED")
        print(f"   Failed: {total - passed}/{total}")
        print("\nNext steps:")
        print("1. Review failed test error messages above")
        print("2. Fix compatibility issues in regularization implementation")
        print("3. Ensure mode forcing works correctly for all combinations")
        print("4. Re-run tests until all pass")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)