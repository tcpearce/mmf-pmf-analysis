import sys
sys.path.append('.')
from pmf_source_app import MMFPMFAnalyzer
import pandas as pd

# Create analyzer and load data
analyzer = MMFPMFAnalyzer(station='MMF9', start_date='2023-10-01', end_date='2023-10-05')
analyzer.load_mmf_data()

print("=== VOC UNITS DEBUG ===")
voc_columns = ['Benzene', 'Toluene', 'Ethylbenzene', 'm&p-Xylene']

for col in voc_columns:
    if col in analyzer.units:
        unit_raw = analyzer.units[col]
        unit_norm = analyzer._normalize_unit_string(unit_raw)
        print(f"{col}:")
        print(f"  Raw unit: '{unit_raw}' (type: {type(unit_raw)})")
        print(f"  Normalized: '{unit_norm}'")
        
        # Check if it matches the expected patterns
        if unit_norm in ('ug/m3', None):
            print(f"  Status: Should be RECOGNIZED as ug/m3")
        else:
            print(f"  Status: UNRECOGNIZED - will trigger warning")
        print()

print("=== NORMALIZATION FUNCTION TEST ===")
test_cases = ['ug/m3', 'μg/m3', 'ugm3', 'mg/m3', 'ng/m3']
for test in test_cases:
    normalized = analyzer._normalize_unit_string(test)
    print(f"'{test}' -> '{normalized}'")