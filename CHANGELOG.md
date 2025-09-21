# CHANGELOG - MMF PMF Analysis Pipeline

This file tracks all changes made to the codebase with timestamps and descriptions.

## 2025-09-21

### process_mmf_fixed.py

**13:16 - Fixed file path references and I/O handling**
- **Changed**: Updated mmf_files dictionary to use correct raw/ directory paths with hash-prefixed filenames
- **Fixed**: MMF file paths now point to:
  - MMF1: `mmf_data_corrected/MMF1_Cemetery_Road/raw/7969ed6f77e41d4fd840a70cd840d42f_Silverdale_Ambient_Air_Monitoring_Data_-_Cemetery_Road_-_Mar_2021_-_Aug_2024.xlsx`
  - MMF2: `mmf_data_corrected/MMF2_Silverdale_Pumping_Station/raw/c39163361bc4854cac6f969b148b4c64_Silverdale Ambient Air Monitoring Data - MMF Silverdale Pumping Station - Mar 2021 to July 2025.xlsx`
  - MMF6: Set to None (no raw Excel file available)
  - MMF9: `mmf_data_corrected/MMF9_Galingale_View/raw/61379dace1c94403959b18fbd97184b7_Silverdale Ambient Air Monitoring Data -MMF Galingale View - Mar 2021 to Jul 2025.xlsx`
- **Added**: None check in processing loop to handle MMF6 gracefully
- **Fixed**: save_to_parquet method - moved all metadata file writes inside the `with` block to prevent "I/O operation on closed file" error
- **Removed**: Duplicate error handling lines at end of save_to_parquet method
- **Reason**: User specified to never use processed/ subdirectories, always use raw Excel files

### Previous Changes (from conversation history):

**Earlier - EPA BDL and Missing Value Implementation**
- Implemented EPA-consistent BDL and missing value handling
- Added unit standardization (all concentrations to μg/m³)
- Added CLI flags: --drop-row-threshold, --zero-as-bdl, --save-masks
- Updated MDL table with standardized units

**Earlier - Temporal Alignment and Aggregation Pipeline**  
- Added timebase aggregation to replace forward-fill approach
- Added CLI flags: --timebase, --aggregate, --min-valid-subsamples
- Implemented proper resampling with count tracking
- Added metadata propagation to parquet files

## Next Planned Changes

**Immediate Priority**:
- Test MMF2 processing with 30min timebase aggregation
- Process MMF9 with same parameters
- Verify metadata propagation in PMF analysis script
- Test uncertainty scaling based on aggregation counts

**Context for Planning**:
The current focus is on completing the temporal alignment pipeline test to ensure:
1. Raw Excel files are processed correctly (not processed/ subdirectories)
2. 30min timebase aggregation works properly
3. Metadata is propagated to PMF analysis
4. Uncertainty scaling is applied based on aggregation counts

**14:00 - Completed PMF script modifications for flexible data directory input**
- **Issue**: pmf_source_apportionment_fixed.py only accepted hardcoded station choices (MMF1, MMF2, MMF6, MMF9, Maries_Way)
- **Solution**: Added flexible data directory and pattern matching capabilities
- **Changes Made**: 
  - Made station argument optional, added --data-dir and --patterns options
  - Added argument validation to ensure either station OR data-dir/patterns are provided
  - Updated MMFPMFAnalyzer constructor to handle both modes
  - Added _find_parquet_files() method for pattern-based file discovery
  - Updated load_mmf_data() for both legacy station-based and flexible data directory loading
  - Updated _display_station_info() and _create_filename_prefix() for flexible mode
  - Fixed HTML dashboard filename generation for both modes
- **Next**: Test with MMF2 30min aggregated parquet file and verify metadata reading

**13:30 - Discovered and fixed units row parsing bug**
- **Issue Found**: MMF2 processing completed but output parquet only contains datetime and availability flags, no concentration data or count columns
- **Root Cause**: Excel files have units in row immediately after headers (row 1), causing all columns to be treated as strings instead of numeric
- **Fixed**: Updated read_sheet_data to skip units row (header_row + 1) during data reading while preserving units extraction
- **Fixed**: Updated availability flag logic to handle column names with suffixes (e.g., 'PM2.5 FIDAS')
- **Result**: MMF2 processing now successful with 31 columns (concentrations + counts + metadata), 75,830 records at 30min timebase
- **Verification**: Gas data points: 70,869, Particle data points: 70,473
