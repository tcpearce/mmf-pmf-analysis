#!/usr/bin/env python3
"""
S/N Categorization Module
=========================

This module implements EPA-recommended S/N (Signal-to-Noise) ratio computation and
feature categorization for PMF analysis, using ESAT DataHandler API where possible.

Key Features:
- S/N computation using ESAT DataHandler.compute_snr() or equivalent
- EPA categorization: strong (S/N >= 2.0), weak (0.2 <= S/N < 2.0), bad (S/N < 0.2)
- Data quality assessment: BDL fraction, missing fraction, variance checks
- Integration with ESAT DataHandler.set_category() for automatic weak/bad handling
- Comprehensive diagnostics and reasoning for all categorization decisions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import warnings


class SNRCategorizer:
    """
    Implements EPA-recommended S/N categorization for PMF analysis using ESAT API.
    """
    
    def __init__(self, snr_weak_threshold: float = 2.0, snr_bad_threshold: float = 0.2,
                 bdl_weak_frac: float = 0.6, bdl_bad_frac: float = 0.8,
                 missing_weak_frac: float = 0.2, missing_bad_frac: float = 0.4):
        """
        Initialize S/N categorizer with EPA thresholds.
        
        Args:
            snr_weak_threshold: S/N threshold for weak categorization (default: 2.0)
            snr_bad_threshold: S/N threshold for bad categorization (default: 0.2)  
            bdl_weak_frac: BDL fraction threshold for weak (default: 0.6)
            bdl_bad_frac: BDL fraction threshold for bad (default: 0.8)
            missing_weak_frac: Missing fraction threshold for weak (default: 0.2)
            missing_bad_frac: Missing fraction threshold for bad (default: 0.4)
        """
        self.snr_weak_threshold = snr_weak_threshold
        self.snr_bad_threshold = snr_bad_threshold
        self.bdl_weak_frac = bdl_weak_frac
        self.bdl_bad_frac = bdl_bad_frac
        self.missing_weak_frac = missing_weak_frac
        self.missing_bad_frac = missing_bad_frac
        
        # Storage for diagnostics
        self.snr_metrics = {}
        self.categories = {}
        self.reasoning = {}
        
    def compute_snr_manual(self, concentrations: np.ndarray, uncertainties: np.ndarray) -> float:
        """
        Manual S/N computation matching EPA PMF 5.0 revised method (Eq. 5-3, 5-4).
        
        For each sample i with finite values and s_i > 0:
          d_i = max((x_i - s_i) / s_i, 0)   if x_i > s_i else 0
        S/N = mean(d_i) over all valid samples (zeros included for x_i <= s_i)
        
        Args:
            concentrations: Array of concentration values (x)
            uncertainties: Array of uncertainty values (s)
            
        Returns:
            S/N ratio (float)
        """
        # Ensure numeric arrays
        conc = np.asarray(concentrations, dtype=float)
        unc = np.asarray(uncertainties, dtype=float)
        
        # Valid if both finite and s > 0
        valid_mask = (np.isfinite(conc) & np.isfinite(unc) & (unc > 0))
        if not np.any(valid_mask):
            return 0.0
        
        x = conc[valid_mask]
        s = unc[valid_mask]
        
        # d_i per EPA revised method
        d = (x - s) / s
        d[d < 0] = 0.0  # zero out when x <= s
        
        # S/N is mean of d (zeros included)
        return float(np.mean(d))
    
    def compute_data_quality_metrics(self, concentrations: np.ndarray, 
                                   mdl: float) -> Dict[str, float]:
        """
        Compute data quality metrics for categorization.
        
        Args:
            concentrations: Array of concentration values
            mdl: Method Detection Limit
            
        Returns:
            Dictionary with quality metrics
        """
        conc = np.asarray(concentrations, dtype=float)
        
        total_count = len(conc)
        if total_count == 0:
            return {'missing_frac': 1.0, 'bdl_frac': 0.0, 'valid_frac': 0.0, 'variance': 0.0}
        
        # Count missing values
        missing_mask = ~np.isfinite(conc)
        missing_count = np.sum(missing_mask)
        
        # Count BDL values (finite but <= MDL)
        finite_mask = np.isfinite(conc)
        bdl_mask = finite_mask & (conc <= mdl)
        bdl_count = np.sum(bdl_mask)
        
        # Count valid measurements (finite and > MDL)
        valid_mask = finite_mask & (conc > mdl)
        valid_count = np.sum(valid_mask)
        
        # Calculate variance for valid data
        if valid_count > 1:
            valid_data = conc[valid_mask]
            variance = np.var(valid_data, ddof=1)  # Sample variance
        else:
            variance = 0.0
        
        return {
            'missing_frac': missing_count / total_count,
            'bdl_frac': bdl_count / total_count,
            'valid_frac': valid_count / total_count,
            'variance': variance,
            'total_count': total_count,
            'missing_count': missing_count,
            'bdl_count': bdl_count,
            'valid_count': valid_count
        }
    
    def categorize_species(self, concentrations_df: pd.DataFrame, 
                          uncertainties_df: pd.DataFrame,
                          epa_calculator=None) -> Tuple[Dict, Dict, Dict]:
        """
        Categorize all species based on S/N and data quality metrics.
        
        Args:
            concentrations_df: DataFrame with species concentrations
            uncertainties_df: DataFrame with species uncertainties  
            epa_calculator: Optional EPA uncertainty calculator for MDL lookup
            
        Returns:
            Tuple of (snr_metrics, categories, reasoning)
        """
        print("🔍 Computing S/N ratios and categorizing features...")
        
        species_names = concentrations_df.columns.tolist()
        snr_metrics = {}
        categories = {}
        reasoning = {}
        
        for species in species_names:
            print(f"   📊 Analyzing {species}...")
            
            # Get data arrays
            conc_data = concentrations_df[species].values
            unc_data = uncertainties_df[species].values
            
            # Compute S/N ratio
            snr = self.compute_snr_manual(conc_data, unc_data)
            
            # Get MDL for data quality assessment
            if epa_calculator:
                try:
                    _, mdl, _ = epa_calculator.get_ef_mdl(species)
                except:
                    mdl = 1.0  # Default fallback
            else:
                # Use some reasonable defaults based on species type
                if 'CH4' in species:
                    mdl = 50.0
                elif any(pm in species for pm in ['PM1', 'PM2.5', 'PM4', 'PM10', 'TSP']):
                    mdl = 2.0
                else:
                    mdl = 1.0
            
            # Compute data quality metrics
            quality = self.compute_data_quality_metrics(conc_data, mdl)
            
            # Store S/N metrics
            snr_metrics[species] = {
                'snr': snr,
                'missing_frac': quality['missing_frac'],
                'bdl_frac': quality['bdl_frac'],
                'valid_frac': quality['valid_frac'],
                'variance': quality['variance'],
                'total_count': quality['total_count'],
                'mdl': mdl
            }
            
            # Apply categorization logic
            category, reasons = self._apply_categorization_rules(
                snr, quality['missing_frac'], quality['bdl_frac'], quality['variance']
            )
            
            categories[species] = category
            reasoning[species] = reasons
            
            # Display results
            status_icon = "✅" if category == 'strong' else "⚠️" if category == 'weak' else "❌"
            print(f"      {status_icon} S/N = {snr:.3f}, category = {category}")
            print(f"         Data: {quality['valid_count']} valid, {quality['bdl_count']} BDL, {quality['missing_count']} missing")
            if len(reasons) > 0:
                print(f"         Reasons: {', '.join(reasons)}")
        
        # Store results
        self.snr_metrics = snr_metrics
        self.categories = categories  
        self.reasoning = reasoning
        
        # Summary statistics
        category_counts = {'strong': 0, 'weak': 0, 'bad': 0}
        for cat in categories.values():
            category_counts[cat] += 1
            
        print(f"\n📊 Categorization Summary:")
        print(f"   Strong: {category_counts['strong']} species")
        print(f"   Weak: {category_counts['weak']} species") 
        print(f"   Bad: {category_counts['bad']} species")
        
        return snr_metrics, categories, reasoning
    
    def _apply_categorization_rules(self, snr: float, missing_frac: float, 
                                  bdl_frac: float, variance: float) -> Tuple[str, List[str]]:
        """
        Apply EPA categorization rules to determine species category.
        
        Args:
            snr: Signal-to-noise ratio
            missing_frac: Fraction of missing data
            bdl_frac: Fraction of BDL data
            variance: Data variance
            
        Returns:
            Tuple of (category, list_of_reasons)
        """
        reasons = []
        
        # Check for "bad" conditions (any one is sufficient)
        if snr < self.snr_bad_threshold:
            return 'bad', [f'S/N < {self.snr_bad_threshold}']
            
        if missing_frac > self.missing_bad_frac:
            return 'bad', [f'Missing > {self.missing_bad_frac*100:.0f}%']
            
        if bdl_frac > self.bdl_bad_frac:
            return 'bad', [f'BDL > {self.bdl_bad_frac*100:.0f}%']
            
        if variance <= 0:
            return 'bad', ['No variance (constant/missing data)']
        
        # Check for "weak" conditions
        if snr < self.snr_weak_threshold:
            reasons.append(f'S/N < {self.snr_weak_threshold}')
            
        if missing_frac > self.missing_weak_frac:
            reasons.append(f'Missing > {self.missing_weak_frac*100:.0f}%')
            
        if bdl_frac > self.bdl_weak_frac:
            reasons.append(f'BDL > {self.bdl_weak_frac*100:.0f}%')
        
        # Determine final category
        if len(reasons) > 0:
            return 'weak', reasons
        else:
            return 'strong', []
    
    def apply_esat_categories(self, data_handler, exclude_bad: bool = True) -> Dict[str, str]:
        """
        Apply categorizations to ESAT DataHandler using set_category API.
        
        Args:
            data_handler: ESAT DataHandler instance
            exclude_bad: If True, exclude bad features from analysis
            
        Returns:
            Dictionary of applied categories
        """
        if not hasattr(data_handler, 'set_category'):
            print("⚠️ ESAT DataHandler does not support set_category - categories not applied")
            return self.categories
            
        print("🔧 Applying categories to ESAT DataHandler...")
        applied = {}
        
        for species, category in self.categories.items():
            try:
                if category == 'bad' and exclude_bad:
                    # Exclude bad features entirely
                    data_handler.set_category(species, 'bad')
                    applied[species] = 'excluded'
                    print(f"   ❌ {species}: excluded (bad)")
                elif category == 'weak':
                    # Mark as weak (ESAT will triple uncertainty)
                    data_handler.set_category(species, 'weak')
                    applied[species] = 'weak'
                    print(f"   ⚠️ {species}: marked weak (uncertainty tripled)")
                else:
                    # Keep as strong (no changes)
                    applied[species] = 'strong'
                    print(f"   ✅ {species}: kept strong")
                    
            except Exception as e:
                print(f"   ⚠️ {species}: failed to apply category - {e}")
                applied[species] = 'failed'
        
        return applied
    
    def save_diagnostics(self, output_dir: Path, filename_prefix: str):
        """
        Save S/N metrics and categorization results to CSV files.
        
        Args:
            output_dir: Directory to save files
            filename_prefix: Prefix for filenames
        """
        if not self.snr_metrics:
            print("⚠️ No S/N metrics to save")
            return
            
        try:
            # Save S/N metrics
            metrics_file = output_dir / f"{filename_prefix}_snr_metrics.csv"
            
            metrics_data = []
            for species, metrics in self.snr_metrics.items():
                row = {
                    'species': species,
                    'snr': metrics['snr'],
                    'missing_frac': metrics['missing_frac'],
                    'bdl_frac': metrics['bdl_frac'],
                    'valid_frac': metrics['valid_frac'],
                    'variance': metrics['variance'],
                    'total_count': metrics['total_count'],
                    'mdl': metrics['mdl']
                }
                metrics_data.append(row)
                
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_csv(metrics_file, index=False)
            print(f"💾 Saved S/N metrics: {metrics_file}")
            
            # Save categories and reasoning
            categories_file = output_dir / f"{filename_prefix}_species_categories.csv"
            
            categories_data = []
            for species in self.categories.keys():
                row = {
                    'species': species,
                    'category': self.categories[species],
                    'reasons': '; '.join(self.reasoning.get(species, [])),
                    'snr': self.snr_metrics[species]['snr'],
                    'missing_frac': self.snr_metrics[species]['missing_frac'],
                    'bdl_frac': self.snr_metrics[species]['bdl_frac']
                }
                categories_data.append(row)
                
            categories_df = pd.DataFrame(categories_data)
            categories_df.to_csv(categories_file, index=False)
            print(f"💾 Saved species categories: {categories_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save diagnostics: {e}")
    
    def get_summary(self) -> Dict:
        """Get summary of S/N categorization results."""
        if not self.categories:
            return {}
            
        category_counts = {'strong': 0, 'weak': 0, 'bad': 0}
        for cat in self.categories.values():
            category_counts[cat] += 1
            
        avg_snr = np.mean([m['snr'] for m in self.snr_metrics.values()])
        
        return {
            'total_species': len(self.categories),
            'strong_count': category_counts['strong'],
            'weak_count': category_counts['weak'],
            'bad_count': category_counts['bad'],
            'average_snr': avg_snr,
            'thresholds': {
                'snr_weak': self.snr_weak_threshold,
                'snr_bad': self.snr_bad_threshold,
                'bdl_weak': self.bdl_weak_frac,
                'bdl_bad': self.bdl_bad_frac,
                'missing_weak': self.missing_weak_frac,
                'missing_bad': self.missing_bad_frac
            }
        }


def create_snr_categorizer(snr_weak_threshold: float = 2.0, snr_bad_threshold: float = 0.2,
                          bdl_weak_frac: float = 0.6, bdl_bad_frac: float = 0.8,
                          missing_weak_frac: float = 0.2, missing_bad_frac: float = 0.4) -> SNRCategorizer:
    """
    Factory function to create and configure S/N categorizer.
    
    Args:
        snr_weak_threshold: S/N threshold for weak categorization
        snr_bad_threshold: S/N threshold for bad categorization  
        bdl_weak_frac: BDL fraction threshold for weak
        bdl_bad_frac: BDL fraction threshold for bad
        missing_weak_frac: Missing fraction threshold for weak
        missing_bad_frac: Missing fraction threshold for bad
        
    Returns:
        Configured SNRCategorizer instance
    """
    return SNRCategorizer(
        snr_weak_threshold=snr_weak_threshold,
        snr_bad_threshold=snr_bad_threshold,
        bdl_weak_frac=bdl_weak_frac,
        bdl_bad_frac=bdl_bad_frac,
        missing_weak_frac=missing_weak_frac,
        missing_bad_frac=missing_bad_frac
    )