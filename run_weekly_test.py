#!/usr/bin/env python3
"""
Simple wrapper to run weekly PMF analysis with proper encoding
"""
import os
import sys

# Set UTF-8 encoding for Windows compatibility
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Run the analysis
if __name__ == "__main__":
    # Test with a 3-week period, single station, reduced settings
    cmd = [
        sys.executable, "weekly_pmf_analysis.py",
        "--start-date", "2025-01-09", 
        "--end-date", "2025-01-15", 
        "--stations", "Maries_Way", 
        "--factors", "3", 
        "--models", "3",  # Very fast for testing
        "--output-dir", "test_weekly"
    ]
    
    print("Running weekly PMF analysis test...")
    print("Command:", " ".join(cmd))
    
    import subprocess
    result = subprocess.run(cmd, env=os.environ)
    sys.exit(result.returncode)