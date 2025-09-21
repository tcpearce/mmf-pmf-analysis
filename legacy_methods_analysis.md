# Legacy Methods in PMF Source Apportionment Code

This document catalogs all legacy methods implemented in the PMF source apportionment code and how they differ from EPA methods.

## Overview

The PMF source apportionment script (`pmf_source_apportionment_fixed.py`) implements a dual-mode system with:
- **Legacy mode** (default): Preserves original behavior for backwards compatibility
- **EPA mode**: Implements EPA PMF 5.0 recommended methods

## 1. Legacy Uncertainty Calculation (`--uncertainty-mode=legacy`)

### Method: `_generate_legacy_uncertainty_matrix()` (lines 806-937)

**Implementation Details:**
- **Formula**: σ = sqrt((error_fraction × concentration)² + (MDL)²)
- **Built-in MDL/EF tables**: Hardcoded values for all species types
- **Minimum uncertainty clamping**: `--legacy-min-u` (default: 0.1) applied to all uncertainties
- **Aggregation scaling**: Applied post-calculation during PMF analysis (lines 1086-1107)

**Gas Species MDLs** (μg/m³):
- H2S: 0.5, CH4: 50.0, SO2: 0.5, NOX: 1.0, NO: 0.5, NO2: 1.0

**Particle Species MDLs** (μg/m³):  
- PM1: 1.0, PM2.5: 1.0, PM4: 1.5, PM10: 2.0, TSP: 2.5

**VOC Species MDLs** (μg/m³):
- Benzene: 0.01, Toluene: 0.02, Ethylbenzene: 0.02, Xylene: 0.02

**Error Fractions** (%):
- Gas: 10-20%, Particles: 10-20%, VOCs: 10-15%

**Legacy Specific Features:**
```python
# Apply legacy minimum uncertainty clamping
u_col = np.maximum(u_col, self.legacy_min_u)
```

## 2. Legacy BDL/Missing Value Handling

### BDL (Below Detection Limit) Rules:
- **Concentration replacement**: V = MDL/2
- **Uncertainty assignment**: U = (5/6) × MDL
- **Consistent with EPA Method 3** but uses hardcoded MDL values

### Missing Value Rules:  
- **Concentration replacement**: V = MDL
- **Uncertainty assignment**: U = 4 × MDL
- **Consistent with EPA Method 1**

### Legacy Controls:
- `--zero-as-bdl` (default: true): Treat exact zeros as BDL
- `--drop-row-threshold` (default: 0.5): Drop rows with >50% missing before replacement
- `--save-masks` (default: true): Save BDL/missing masks for traceability

## 3. Legacy Missing Value Handling

### Method: `_handle_missing_values()` (lines 939-962)

**Note**: Despite name reference to "EPA Method 1", this is actually a **legacy implementation**:

```python
def _handle_missing_values(self):
    """Handle missing values following EPA Method 1."""
    # Replace missing concentrations with median
    median_conc = self.concentration_data[col].median()
    # Set high uncertainty for replaced values (4 × median)
    self.uncertainty_data.loc[missing_mask, col] = 4 * median_conc
```

**Legacy Characteristics:**
- **Median replacement**: Uses column median instead of MDL
- **4× median uncertainty**: Different from EPA 4×MDL approach
- **Applied after MDL-based processing**: Secondary fallback method

## 4. Legacy Aggregation Scaling

### Method: Applied in `run_pmf_analysis()` (lines 1086-1107)

```python
if self.uncertainty_mode == 'legacy':
    # Apply uncertainty scaling for aggregated windows
    if self.aggregation_method == 'mean':
        scale = 1.0 / np.sqrt(n)
    else:
        scale = 1.253 / np.sqrt(n)  # Median approximation
    U[:, j] = U[:, j] * scale
```

**Legacy Approach:**
- **Post-calculation scaling**: Applied after uncertainty matrix generation
- **Separate from core uncertainty**: Not integrated into uncertainty formulas
- **Median scaling approximation**: 1.253/√n factor for median aggregation

## 5. Legacy Station-Based Data Loading

### Method: Traditional station mapping approach

**Legacy Stations:**
- MMF1, MMF2, MMF6, MMF9, Maries_Way
- **Hardcoded paths**: Fixed directory structure assumptions
- **Station-specific processing**: Different handling per station

**vs Modern Approach:**
- `--data-dir` and `--patterns`: Flexible file discovery
- **Pattern matching**: Supports arbitrary parquet file organization

## 6. Legacy Default Parameters

### All EPA features disabled by default:
```python
uncertainty_mode='legacy'           # Use legacy uncertainty calculation
snr_enable=False                   # Disable S/N categorization 
legacy_min_u=0.1                  # Apply minimum uncertainty clamping
uncertainty_bdl_policy='five-sixth-mdl'  # 5/6 MDL for BDL uncertainty
```

### Legacy CLI Defaults:
- **No S/N categorization**: `--snr-enable=false` by default
- **Legacy uncertainty**: `--uncertainty-mode=legacy` by default  
- **Minimum uncertainty clamping**: `--legacy-min-u=0.1` preserved
- **Backwards compatibility**: All new features opt-in only

## 7. Legacy Dashboard Elements

### Dashboard sections that reference legacy behavior:
- **Policy explanations**: Show "Legacy Mode" vs "EPA Mode" formulas
- **CLI parameter documentation**: Explains legacy defaults
- **S/N categorization**: Conditionally shown only when `--snr-enable=true`

## Key Differences: Legacy vs EPA

| Aspect | Legacy Mode | EPA Mode |
|--------|-------------|----------|
| **Uncertainty Formula** | Built-in MDL/EF tables | CSV-configurable EF/MDL data |
| **Minimum Clamping** | 0.1 μg/m³ global minimum | No global minimum (ε=1e-12 numerical floor) |
| **Aggregation Scaling** | Post-calculation application | Integrated into uncertainty formulas |
| **BDL Policy** | 5/6 × MDL (hardcoded) | Configurable (5/6 or 1/2) × MDL |
| **Missing Handling** | Median + 4×median uncertainty | MDL + 4×MDL uncertainty |
| **S/N Categorization** | Disabled by default | Available when enabled |
| **Data Sources** | Hardcoded station paths | Flexible directory patterns |

## Usage Examples

### Legacy Mode (Default):
```bash
python pmf_source_apportionment_fixed.py MMF2 --start-date 2023-09-01 --end-date 2023-09-03
# Uses: legacy uncertainty, no S/N categorization, hardcoded MDL/EF, min uncertainty clamping
```

### EPA Mode:
```bash  
python pmf_source_apportionment_fixed.py --data-dir mmf_test_30min --patterns "*mmf2*.parquet" \
  --start-date 2023-09-01 --end-date 2023-09-03 \
  --uncertainty-mode epa --snr-enable --write-diagnostics
# Uses: EPA uncertainty formulas, S/N categorization, configurable EF/MDL, no min clamping
```

## Implementation Status

- ✅ **Legacy mode fully preserved**: All original behavior maintained
- ✅ **EPA mode fully implemented**: All EPA PMF 5.0 methods available  
- ✅ **Backwards compatibility**: Default parameters ensure no breaking changes
- ✅ **Migration path**: Users can gradually adopt EPA features via CLI flags

## Conclusion

The code maintains comprehensive legacy support while providing modern EPA-compliant alternatives. Legacy methods remain the default to ensure backwards compatibility, with EPA methods available as opt-in enhancements through CLI configuration.