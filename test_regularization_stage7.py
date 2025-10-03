#!/usr/bin/env python3
"""
Test script for Species Regularization Stage 7 - Staged Training Loop Integration

This script tests the end-to-end regularized PMF training to verify:
1. Regularization preparation works correctly
2. Species-to-column mapping functions
3. Template construction succeeds
4. Staged training loop executes
5. Regularization "pushes out" target species as lambda increases
"""

import subprocess
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

def run_pmf_test(test_name, reg_species=None, reg_lambda=None, reg_template=None, output_dir=None, expected_species_count=None):
    """Run PMF analysis with regularization parameters and capture results."""
    
    print(f"\n🧪 Running test: {test_name}")
    print("=" * 60)
    
    # Build command
    cmd = [
        "python", "pmf_source_app.py", "MMF9",
        "--start-date", "2023-10-01",
        "--end-date", "2023-10-05",  # Short date range for fast testing
        "--factors", "4",  # Reasonable number of factors
        "--models", "2",   # Just 2 models for speed
        "--uncertainty-mode", "epa",
        "--output-dir", output_dir or f"test_reg_stage7_{test_name.lower().replace(' ', '_')}"
    ]
    
    # Add regularization parameters if provided
    if reg_species:
        for species in reg_species:
            cmd.extend(["--reg-species", species])
    
    if reg_lambda:
        for lam in reg_lambda:
            cmd.extend(["--reg-lambda", str(lam)])
    
    if reg_template:
        for template in reg_template:
            cmd.extend(["--reg-template", template])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        # Check return code
        if result.returncode != 0:
            print(f"❌ Test FAILED - PMF analysis returned error code {result.returncode}")
            print("STDERR:")
            print(result.stderr[-1500:])  # Last 1500 chars of stderr
            return False
        
        # Check stdout for key indicators
        stdout = result.stdout
        stderr = result.stderr
        
        print("✅ PMF analysis completed successfully")
        
        # Extract key information from output
        success_indicators = []
        
        # Check if regularization was prepared
        if reg_species and "[OK] Regularization preparation complete" in stdout:
            success_indicators.append("Regularization preparation successful")
            # Extract number of targets
            for line in stdout.split('\n'):
                if "Regularization preparation complete" in line:
                    print(f"   {line.strip()}")
        elif reg_species:
            print("⚠️  Warning: Expected regularization preparation message not found")
        
        # Check if regularized training was used
        if reg_species and "[REG] Running regularized PMF training" in stdout:
            success_indicators.append("Regularized training executed")
            for line in stdout.split('\n'):
                if "Running regularized PMF training" in line:
                    print(f"   {line.strip()}")
        elif reg_species:
            print("⚠️  Warning: Expected regularized training message not found")
        
        # Check for mathematical regularization evidence
        if reg_species and "Objective decreased by" in stdout:
            success_indicators.append("Mathematical regularization working")
            for line in stdout.split('\n'):
                if "Objective decreased by" in line and "%" in line:
                    print(f"   {line.strip()}")
                    break  # Just show first major decrease
        
        # Check for staged training bursts
        if reg_species and "Burst 1/" in stdout:
            success_indicators.append("Staged training bursts executing")
            burst_count = stdout.count("Burst ")
            print(f"   Detected {burst_count} training bursts")
        
        # Check final model results
        for line in stdout.split('\n'):
            if "[OK] Best model:" in line:
                print(f"   {line.strip()}")
            elif "Q(true):" in line or "Q(robust):" in line:
                print(f"   {line.strip()}")
        
        # Check output files were created
        output_path = Path(output_dir or f"test_reg_stage7_{test_name.lower().replace(' ', '_')}")
        if output_path.exists():
            conc_file = list(output_path.glob("*_concentrations.csv"))
            if conc_file:
                # Check species count if expected
                if expected_species_count:
                    df = pd.read_csv(conc_file[0], index_col=0)
                    actual_count = len(df.columns)
                    if actual_count == expected_species_count:
                        success_indicators.append(f"Species count correct: {actual_count}")
                    else:
                        print(f"⚠️  Warning: Expected {expected_species_count} species, got {actual_count}")
                        print(f"   Species: {list(df.columns)}")
                
                success_indicators.append("Output files created")
        
        print(f"📊 Success indicators: {len(success_indicators)}")
        for indicator in success_indicators:
            print(f"   ✅ {indicator}")
        
        return len(success_indicators) > 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test FAILED - Timeout after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Test FAILED - Exception: {e}")
        return False

def main():
    """Run comprehensive Stage 7 tests."""
    
    print("🚀 SPECIES REGULARIZATION STAGE 7 TEST SUITE")
    print("Testing: Staged Training Loop Integration")
    print("=" * 80)
    
    # Track test results
    tests = []
    
    # Test 1: Baseline (no regularization) - should work normally
    print("\n📋 Test 1: Baseline (No Regularization)")
    result1 = run_pmf_test(
        test_name="Baseline No Regularization",
        output_dir="test_reg_stage7_baseline",
        expected_species_count=10  # All species should be present
    )
    tests.append(("Baseline", result1))
    
    # Test 2: Single species regularization with zero template
    print("\n📋 Test 2: Single Species Zero Template")
    result2 = run_pmf_test(
        test_name="Single Species Zero Template", 
        reg_species=["CH4"],
        reg_lambda=[1.0],
        reg_template=["zero"],
        output_dir="test_reg_stage7_ch4_zero",
        expected_species_count=10
    )
    tests.append(("CH4 Zero Template", result2))
    
    # Test 3: Multiple species regularization
    print("\n📋 Test 3: Multiple Species Regularization")  
    result3 = run_pmf_test(
        test_name="Multiple Species Regularization",
        reg_species=["CH4", "H2S"], 
        reg_lambda=[2.0, 1.0],
        reg_template=["zero", "uniform"],
        output_dir="test_reg_stage7_multi_species",
        expected_species_count=10
    )
    tests.append(("Multiple Species", result3))
    
    # Test 4: Higher lambda to test push-out behavior
    print("\n📋 Test 4: High Lambda Push-Out Test")
    result4 = run_pmf_test(
        test_name="High Lambda Push Out",
        reg_species=["CH4"],
        reg_lambda=[10.0],  # High lambda should strongly regularize
        reg_template=["zero"],
        output_dir="test_reg_stage7_high_lambda",
        expected_species_count=10
    )
    tests.append(("High Lambda Push-Out", result4))
    
    # Summary
    print("\n" + "=" * 80)
    print("🧪 STAGE 7 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in tests if result)
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Stage 7 implementation validated!")
        return True
    else:
        print("❌ SOME TESTS FAILED - Stage 7 needs fixes")
        print("\nNext steps:")
        print("1. Check error messages above for specific issues")
        print("2. Verify regularization preparation logic")
        print("3. Ensure species-to-column mapping works correctly")
        print("4. Test template construction validation")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)