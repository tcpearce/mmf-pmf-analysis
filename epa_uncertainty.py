#!/usr/bin/env python3
"""
EPA Uncertainty Calculation Module
==================================

This module implements EPA-recommended uncertainty calculations for PMF analysis,
following EPA PMF 5.0 User Guide formulas. Supports both concentration-dependent
and BDL (Below Detection Limit) uncertainty calculations.

Key Features:
- EPA formulas: sqrt((EF * conc)^2 + (0.5 * MDL)^2) for conc > MDL
- BDL handling: 5/6 * MDL or 0.5 * MDL for conc <= MDL  
- Temporal aggregation scaling: 1/sqrt(n) applied after EPA formulas
- Numerical stability with configurable epsilon floor
- Unit validation and standardization support
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import warnings


class EPAUncertaintyCalculator:
    """
    Implements EPA-recommended uncertainty calculations for PMF analysis.
    """
    
    def __init__(self, epsilon: float = 1e-12, bdl_policy: str = 'five-sixth-mdl', legacy_min_u: float = 0.1):
        """
        Initialize EPA uncertainty calculator.
        
        Args:
            epsilon: Numerical floor for uncertainties (not a weighting clamp)
            bdl_policy: Policy for BDL handling ('five-sixth-mdl' or 'half-mdl')
            legacy_min_u: Minimum uncertainty clamp (aligned with legacy mode)
        """
        self.epsilon = epsilon
        self.bdl_policy = bdl_policy
        self.legacy_min_u = legacy_min_u
        
        # Built-in EF and MDL values (ALIGNED WITH LEGACY MODE - pmf_source_app.py)
        # FIXED 2025-01-25: Synchronized with legacy uncertainty parameters for consistent PMF fitting
        self.default_ef_mdl = {
            # Gas species - aligned with legacy mode values
            'CH4': {'EF': 0.10, 'MDL': 50.0, 'unit': 'μg/m³'},   # 10% error fraction, 50 μg/m³ MDL (legacy)
            'H2S': {'EF': 0.15, 'MDL': 0.5, 'unit': 'μg/m³'},    # 15% error fraction, 0.5 μg/m³ MDL (legacy)  
            'NOX': {'EF': 0.20, 'MDL': 1.0, 'unit': 'μg/m³'},    # 20% error fraction, 1.0 μg/m³ MDL (legacy)
            'NO': {'EF': 0.20, 'MDL': 0.5, 'unit': 'μg/m³'},     # 20% error fraction, 0.5 μg/m³ MDL (legacy)
            'NO2': {'EF': 0.20, 'MDL': 1.0, 'unit': 'μg/m³'},    # 20% error fraction, 1.0 μg/m³ MDL (legacy)
            'SO2': {'EF': 0.15, 'MDL': 0.5, 'unit': 'μg/m³'},    # 15% error fraction, 0.5 μg/m³ MDL (legacy)
            
            # VOC species - aligned with legacy mode values
            'Benzene': {'EF': 0.10, 'MDL': 0.01, 'unit': 'μg/m³'},     # 10% error fraction, 0.01 μg/m³ MDL (legacy)
            'Toluene': {'EF': 0.12, 'MDL': 0.02, 'unit': 'μg/m³'},      # 12% error fraction, 0.02 μg/m³ MDL (legacy)
            'Ethylbenzene': {'EF': 0.15, 'MDL': 0.02, 'unit': 'μg/m³'}, # 15% error fraction, 0.02 μg/m³ MDL (legacy)
            'Xylene': {'EF': 0.15, 'MDL': 0.02, 'unit': 'μg/m³'},       # 15% error fraction, 0.02 μg/m³ MDL (legacy)
            'm&p-Xylene': {'EF': 0.15, 'MDL': 0.02, 'unit': 'μg/m³'},   # 15% error fraction, 0.02 μg/m³ MDL (legacy)
            
            # Particle species - aligned with legacy mode values
            'PM1 FIDAS': {'EF': 0.10, 'MDL': 1.0, 'unit': 'μg/m³'},   # 10% error fraction, 1.0 μg/m³ MDL (legacy)
            'PM1': {'EF': 0.10, 'MDL': 1.0, 'unit': 'μg/m³'},         # 10% error fraction, 1.0 μg/m³ MDL (legacy)
            'PM2.5 FIDAS': {'EF': 0.10, 'MDL': 1.0, 'unit': 'μg/m³'}, # 10% error fraction, 1.0 μg/m³ MDL (legacy)
            'PM2.5': {'EF': 0.10, 'MDL': 1.0, 'unit': 'μg/m³'},       # 10% error fraction, 1.0 μg/m³ MDL (legacy)
            'PM4 FIDAS': {'EF': 0.12, 'MDL': 1.5, 'unit': 'μg/m³'},   # 12% error fraction, 1.5 μg/m³ MDL (legacy)
            'PM4': {'EF': 0.12, 'MDL': 1.5, 'unit': 'μg/m³'},         # 12% error fraction, 1.5 μg/m³ MDL (legacy)
            'PM10 FIDAS': {'EF': 0.15, 'MDL': 2.0, 'unit': 'μg/m³'},  # 15% error fraction, 2.0 μg/m³ MDL (legacy)
            'PM10': {'EF': 0.15, 'MDL': 2.0, 'unit': 'μg/m³'},        # 15% error fraction, 2.0 μg/m³ MDL (legacy)
            'TSP FIDAS': {'EF': 0.20, 'MDL': 2.5, 'unit': 'μg/m³'},   # 20% error fraction, 2.5 μg/m³ MDL (legacy)
            'TSP': {'EF': 0.20, 'MDL': 2.5, 'unit': 'μg/m³'},         # 20% error fraction, 2.5 μg/m³ MDL (legacy)
        }
        
        self.ef_mdl_data = None  # Will be loaded from CSV if provided
        
    def load_ef_mdl_table(self, csv_path: str) -> bool:
        """
        Load EF/MDL table from CSV file.
        
        Expected columns: species, EF, MDL, unit
        
        Args:
            csv_path: Path to CSV file with EF/MDL data
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            if not Path(csv_path).exists():
                warnings.warn(f"EF/MDL CSV not found: {csv_path}. Using built-in values.")
                return False
                
            df = pd.read_csv(csv_path)
            required_cols = ['species', 'EF', 'MDL', 'unit']
            
            if not all(col in df.columns for col in required_cols):
                warnings.warn(f"EF/MDL CSV missing required columns {required_cols}. Using built-in values.")
                return False
                
            # Convert to dict format
            self.ef_mdl_data = {}
            for _, row in df.iterrows():
                species = row['species']
                self.ef_mdl_data[species] = {
                    'EF': float(row['EF']),
                    'MDL': float(row['MDL']),
                    'unit': str(row['unit'])
                }
                
            print(f"✅ Loaded EF/MDL data for {len(self.ef_mdl_data)} species from {csv_path}")
            return True
            
        except Exception as e:
            warnings.warn(f"Failed to load EF/MDL CSV {csv_path}: {e}. Using built-in values.")
            return False
    
    def get_ef_mdl(self, species: str) -> Tuple[float, float, str]:
        """
        Get error fraction, MDL, and unit for a species.
        
        Args:
            species: Species name
            
        Returns:
            Tuple of (EF, MDL, unit)
        """
        # Use loaded CSV data if available, otherwise fall back to defaults
        source_data = self.ef_mdl_data if self.ef_mdl_data is not None else self.default_ef_mdl
        
        if species in source_data:
            data = source_data[species]
            return data['EF'], data['MDL'], data['unit']
        else:
            # Default fallback for unknown species
            warnings.warn(f"No EF/MDL data for species '{species}'. Using default values.")
            return 0.20, 5.0, 'μg/m³'  # Conservative defaults
    
    def calculate_epa_uncertainty(self, concentrations: np.ndarray, species: str) -> np.ndarray:
        """
        Calculate EPA-recommended uncertainties for a species.
        
        EPA Formula (per EPA PMF 5.0):
        - If conc <= MDL: U = 5/6 * MDL (or 0.5 * MDL based on policy)
        - If conc > MDL:  U = sqrt((EF * conc)^2 + (0.5 * MDL)^2)
        - Missing values: high uncertainty; common practice is U = 4 × species median concentration
        
        Args:
            concentrations: Array of concentration values
            species: Species name for EF/MDL lookup
            
        Returns:
            Array of EPA uncertainties (before aggregation scaling)
        """
        EF, MDL, unit = self.get_ef_mdl(species)
        
        # Ensure we have valid numeric data
        conc = np.asarray(concentrations, dtype=float)
        uncertainties = np.full_like(conc, np.nan)
        
        # Handle finite values only
        valid_mask = np.isfinite(conc)
        
        if not np.any(valid_mask):
            # No valid data - return array of epsilon values
            return np.full_like(conc, self.epsilon)
        
        valid_conc = conc[valid_mask]
        
        # Apply EPA formulas
        if self.bdl_policy == 'five-sixth-mdl':
            bdl_uncertainty = (5.0 / 6.0) * MDL
        else:  # 'half-mdl'
            bdl_uncertainty = 0.5 * MDL
        
        # BDL case: conc <= MDL
        bdl_mask = valid_conc <= MDL
        
        # Above MDL case: conc > MDL
        above_mdl_mask = valid_conc > MDL
        
        # Calculate uncertainties
        valid_uncertainties = np.full_like(valid_conc, np.nan)
        
        # BDL uncertainties
        valid_uncertainties[bdl_mask] = bdl_uncertainty
        
        # Above MDL uncertainties: sqrt((EF * conc)^2 + (0.5 * MDL)^2) [EPA PMF 5.0 Eq. 5-2]
        if np.any(above_mdl_mask):
            conc_above = valid_conc[above_mdl_mask]
            valid_uncertainties[above_mdl_mask] = np.sqrt(
                (EF * conc_above) ** 2 + (0.5 * MDL) ** 2
            )
        
        # Apply epsilon floor for numerical stability (EPA guidance does not prescribe a global clamp)
        valid_uncertainties = np.maximum(valid_uncertainties, self.epsilon)
        
        # Fill back into full array
        uncertainties[valid_mask] = valid_uncertainties
        
        # For invalid/missing concentrations, set uncertainty high per EPA guidance
        # EPA PMF practice: replace missing concentrations with species median and set Unc = 4 × median
        # Here we set uncertainty to 4 × species median (fall back to 4 × MDL if median is not finite)
        species_median = np.nanmedian(conc) if np.isfinite(np.nanmedian(conc)) else np.nan
        fallback = 4.0 * MDL
        missing_unc = 4.0 * species_median if np.isfinite(species_median) else fallback
        # Enforce epsilon floor only
        missing_unc = max(missing_unc, self.epsilon)
        uncertainties[~valid_mask] = missing_unc
        
        return uncertainties
    
    def apply_aggregation_scaling(self, uncertainties: np.ndarray, 
                                counts: np.ndarray) -> np.ndarray:
        """
        Apply temporal aggregation uncertainty scaling: U_scaled = U * 1/sqrt(n)
        
        Args:
            uncertainties: EPA uncertainties before scaling
            counts: Number of sub-samples aggregated for each time point
            
        Returns:
            Scaled uncertainties
        """
        # Ensure counts are valid and > 0
        safe_counts = np.where(np.isfinite(counts) & (counts > 0), counts, 1.0)
        
        # Apply 1/sqrt(n) scaling  
        scaling_factors = 1.0 / np.sqrt(safe_counts)
        
        return uncertainties * scaling_factors
    
    def calculate_species_uncertainties(self, concentrations_df: pd.DataFrame,
                                     counts_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate EPA uncertainties for all species in a concentration DataFrame.
        
        Args:
            concentrations_df: DataFrame with species as columns
            counts_df: Optional DataFrame with aggregation counts (n_species columns)
            
        Returns:
            DataFrame with EPA uncertainties (same shape as input)
        """
        uncertainties_df = pd.DataFrame(index=concentrations_df.index)
        
        print(f"🧮 Calculating EPA uncertainties for {len(concentrations_df.columns)} species...")
        
        for species in concentrations_df.columns:
            # Calculate base EPA uncertainties
            conc_values = concentrations_df[species].values
            epa_uncertainties = self.calculate_epa_uncertainty(conc_values, species)
            
            # Apply aggregation scaling if counts are available
            if counts_df is not None:
                count_col = f"n_{species}"
                if count_col in counts_df.columns:
                    count_values = counts_df[count_col].values
                    epa_uncertainties = self.apply_aggregation_scaling(
                        epa_uncertainties, count_values
                    )
                    print(f"   ✅ {species}: EPA + 1/√n scaling applied")
                else:
                    print(f"   ⚠️ {species}: EPA only (no count data for scaling)")
            else:
                print(f"   ✅ {species}: EPA formula applied")
            
            uncertainties_df[species] = epa_uncertainties
        
        return uncertainties_df
    
    def get_policy_summary(self) -> Dict:
        """Get summary of current EPA uncertainty policy settings."""
        return {
            'bdl_policy': self.bdl_policy,
            'bdl_formula': '5/6 * MDL' if self.bdl_policy == 'five-sixth-mdl' else '0.5 * MDL',
            'above_mdl_formula': 'sqrt((EF * conc)^2 + (0.5 * MDL)^2)',
            'epsilon': self.epsilon,
            'aggregation_scaling': '1/sqrt(n) applied after EPA formulas',
            'ef_mdl_source': 'CSV file' if self.ef_mdl_data is not None else 'built-in defaults'
        }


def create_epa_uncertainty_calculator(epsilon: float = 1e-12, 
                                    bdl_policy: str = 'five-sixth-mdl',
                                    ef_mdl_csv: Optional[str] = None,
                                    legacy_min_u: float = 0.1) -> EPAUncertaintyCalculator:
    """
    Factory function to create and configure EPA uncertainty calculator.
    
    Args:
        epsilon: Numerical floor for uncertainties  
        bdl_policy: BDL handling policy ('five-sixth-mdl' or 'half-mdl')
        ef_mdl_csv: Optional path to CSV with EF/MDL data
        legacy_min_u: Minimum uncertainty clamp (aligned with legacy mode)
        
    Returns:
        Configured EPAUncertaintyCalculator instance
    """
    calculator = EPAUncertaintyCalculator(epsilon=epsilon, bdl_policy=bdl_policy, legacy_min_u=legacy_min_u)
    
    if ef_mdl_csv:
        calculator.load_ef_mdl_table(ef_mdl_csv)
    
    return calculator