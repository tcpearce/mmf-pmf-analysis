# MMF Data Organization Summary

## Completed Tasks

### 1. Data Download ✅
Successfully downloaded all four MMF rectified data files:
- **MMF1**: Silverdale Ambient Air Monitoring Data - MMF1 - Mar 21 to Aug 23.xlsx (23.66 MB)
- **MMF2**: Silverdale Ambient Air Monitoring Data - MMF2 - Mar 21 to Aug 23.xlsx (30.85 MB)  
- **MMF6**: Silverdale Ambient Air Monitoring Data - MMF6 - Mar 21 to June 23.xlsx (19.63 MB)
- **MMF9**: Silverdale Ambient Air Monitoring Data - MMF9 - Mar 21 to Aug 23.xlsx (33.18 MB)

### 2. Directory Structure ✅
Organized files into proper directory structure:
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

### 3. Data Structure Analysis ✅
Identified the structure of each Excel file:
- **Sheet 1**: "Rectified data" - Metadata and description (not processed)
- **Sheet 2**: Gas measurements at 5-minute intervals
  - Columns: DATE, TIME, WD (degr), WS (m/s), H2S (ug/m3), CH4 (mg/m3), SO2 (ug/m3)
- **Sheet 3**: Particle measurements at 15-minute intervals  
  - Columns: DATE, TIME, PM1 FIDAS (ug/m3), PM2.5 (ug/m3), PM4 FIDAS (ug/m3), PM10 (ug/m3), TSP (ug/m3), TEMP (oC), AMB_PRES (hPa)

## Data Processing Challenges

### Issues Encountered
1. **Large file sizes**: Files are 20-35 MB each with 250,000+ rows
2. **Date/time parsing**: Complex datetime formats causing parsing errors
3. **Memory limitations**: Processing large Excel files with multiple sheets
4. **Mixed timezone data**: Some datetime columns have timezone-aware and timezone-naive data mixed

### Current Status
- Files are successfully downloaded and organized
- Data structure is fully documented
- Processing scripts are created but need refinement for large file handling

## Data Overview

### Coverage Periods
- **MMF1, MMF2, MMF9**: March 2021 - August 2023 (30 months)
- **MMF6**: March 2021 - June 2023 (28 months)

### Data Volume (Estimated)
- **Gas measurements**: ~250,000 records per station (5-minute intervals)
- **Particle measurements**: ~83,000 records per station (15-minute intervals)
- **Total combined**: ~1.3 million individual measurements across all stations

### Parameters Monitored

#### Meteorological
- Wind Direction (WD) - degrees
- Wind Speed (WS) - m/s
- Temperature (TEMP) - °C
- Ambient Pressure (AMB_PRES) - hPa

#### Gas Pollutants
- Hydrogen Sulfide (H2S) - ug/m3
- Methane (CH4) - mg/m3
- Sulfur Dioxide (SO2) - ug/m3

#### Particulate Matter
- PM1 FIDAS - ug/m3
- PM2.5 - ug/m3
- PM4 FIDAS - ug/m3
- PM10 - ug/m3
- Total Suspended Particles (TSP) - ug/m3

## Next Steps Required

### For Complete Processing
1. **Optimize memory usage**: Process files in chunks to handle large datasets
2. **Fix datetime parsing**: Handle mixed timezone data properly
3. **Create parquet files**: Convert to efficient storage format with metadata
4. **Data validation**: Verify all original data values are preserved
5. **Time alignment**: Merge 5-minute and 15-minute data on common timebase

### Recommended Approach
1. Process one station at a time to manage memory
2. Use chunked reading for large Excel sheets
3. Standardize datetime format before processing
4. Create separate parquet files for each station
5. Include comprehensive metadata in parquet files

## Files Created
- `mmf_data_requirements.md` - Data structure documentation
- `download_mmf_data.py` - Download script
- `process_mmf_improved.py` - Processing script (needs optimization)
- `final_data_summary.md` - This summary

## Data Quality Notes
- These are **rectified data** files with adjusted H2S calibration
- Original uncalibrated H2S data had higher uncertainty
- Data quality flags should be preserved during processing
- Missing data periods should be clearly marked in final datasets
