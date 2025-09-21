# MMF Data Organization and Requirements

## Data Sources

### MMF Stations and Time Ranges
1. **MMF1**
   - Period: March 2021 - August 2023
   - Location: TBD from data

2. **MMF2**
   - Period: March 2021 - August 2023
   - Location: TBD from data

3. **MMF6**
   - Period: March 2021 - June 2023
   - Location: TBD from data

4. **MMF9**
   - Period: March 2021 - August 2023
   - Location: TBD from data

## Data Types

### Raw Data
- Original monitoring data
- Pre-calibration values
- Greater degree of uncertainty for H₂S measurements
- Stored in `/raw` subdirectories

### Processed Data
- Rectified data with adjusted calibration
- Quality assured measurements
- More reliable H₂S values
- Stored in `/processed` subdirectories

## Directory Structure
```
mmf_data/
├── MMF1/
│   ├── raw/
│   └── processed/
├── MMF2/
│   ├── raw/
│   └── processed/
├── MMF6/
│   ├── raw/
│   └── processed/
├── MMF9/
│   ├── raw/
│   └── processed/
└── combined_analysis/
    ├── raw/
    └── processed/
```

## Data Files Needed

### Required Files
For each MMF station (1, 2, 6, and 9):
1. Rectified data files covering the entire monitoring period
2. Data should include:
   - H₂S measurements with adjusted calibration
   - Quality assurance flags/notes
   - Timestamp information
   - Any relevant metadata

### Additional Information
- Data files should be in Excel (.xlsx) format
- Each file should contain clear headers and units
- Quality assurance information should be included
- Any calibration adjustments should be documented

## Data Quality Notes
- H₂S data prior to recalibration has greater uncertainty
- Only rectified data (with adjusted calibration) should be used for analysis
- Data quality flags should be checked before use
- Any gaps in monitoring should be documented

## Usage Requirements
1. Check timestamps for continuity
2. Verify units and calibration factors
3. Note any quality assurance flags
4. Document any data gaps or anomalies
5. Cross-reference with station locations
6. Maintain data versioning
