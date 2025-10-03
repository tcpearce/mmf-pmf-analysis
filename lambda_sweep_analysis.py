#!/usr/bin/env python3
"""
Lambda Sweep Analysis: CH4 Regularization vs Exclusion Comparison

Research Question: Can CH4 regularization with increasing λ values achieve 
similar factor correlation diversity as excluding CH4 entirely?

This script runs PMF analysis with multiple lambda values and compares:
1. Factor profile correlations at different λ values
2. Closure metrics across regularization strengths  
3. Comparison with CH4 exclusion baseline
4. Optimal λ value recommendations

Author: PMF Analysis Pipeline
Date: 2025-09-28
"""

import os
import sys
import json
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class LambdaSweepAnalyzer:
    """
    Comprehensive lambda sweep analysis for CH4 regularization vs exclusion comparison
    """
    
    def __init__(self, base_output_dir="lambda_sweep_30day", 
                 station="MMF9", start_date="2023-10-01", end_date="2023-10-31"):
        self.base_output_dir = Path(base_output_dir)
        self.station = station
        self.start_date = start_date
        self.end_date = end_date
        
        # Lambda values to test (including λ=0 as baseline)
        # Focused on meaningful range based on 2-day analysis
        self.lambda_values = [0, 5, 20, 50, 100]
        
        # Results storage
        self.results = {}
        self.correlation_data = []
        self.closure_data = []
        
        # Ensure output directory exists
        self.base_output_dir.mkdir(exist_ok=True)
        
    def run_pmf_scenario(self, scenario_name, pmf_args):
        """Run PMF analysis for a specific scenario"""
        print(f"Running scenario: {scenario_name}")
        
        output_dir = self.base_output_dir / f"scenario_{scenario_name}"
        
        # Base command
        cmd = [
            "python", "pmf_source_app.py", self.station,
            "--start-date", self.start_date,
            "--end-date", self.end_date,
            "--output-dir", str(output_dir),
            "--uncertainty-mode", "epa",
            "--factors", "3",
            "--models", "3"
        ]
        
        # Add scenario-specific arguments
        cmd.extend(pmf_args)
        
        try:
            # Run PMF analysis
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"SUCCESS: {scenario_name} completed")
                # Check if PMF actually ran successfully by looking for key output files
                profile_files = list(output_dir.glob("*_factor_profiles.csv"))
                if profile_files:
                    print(f"  ✅ PMF results generated: {len(profile_files)} profile file(s)")
                else:
                    print(f"  ⚠️ PMF subprocess completed but no factor profiles generated")
                    print(f"  📂 Available files: {[f.name for f in output_dir.glob('*.csv')]}")
                    # Print last 20 lines of stdout for debugging
                    print(f"  📋 Last part of PMF output:")
                    stdout_lines = result.stdout.split('\n')
                    for line in stdout_lines[-20:]:
                        if line.strip():
                            print(f"    {line}")
                return output_dir
            else:
                print(f"FAILED: {scenario_name}")
                print(f"STDERR: {result.stderr}")
                print(f"STDOUT: {result.stdout[-1000:] if result.stdout else 'No stdout'}")  # Last 1000 chars
                return None
                
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT: {scenario_name} (10 minutes)")
            return None
        except Exception as e:
            print(f"ERROR: {scenario_name} - {e}")
            return None
    
    def extract_factor_profiles(self, output_dir):
        """Extract factor profiles (H matrix) from PMF results"""
        try:
            # Look for factor profiles CSV file - more specific pattern
            profile_files = list(output_dir.glob("*_factor_profiles.csv"))
            if not profile_files:
                print(f"No factor profiles found in {output_dir}")
                # List all CSV files for debugging
                all_csvs = list(output_dir.glob("*.csv"))
                print(f"Available CSV files: {[f.name for f in all_csvs]}")
                return None
                
            profiles_df = pd.read_csv(profile_files[0], index_col=0)
            print(f"Loaded factor profiles: {profiles_df.shape} from {profile_files[0].name}")
            return profiles_df
            
        except Exception as e:
            print(f"Error extracting profiles from {output_dir}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def compute_factor_correlations(self, profiles_df):
        """Compute correlation matrix between factor profiles"""
        if profiles_df is None:
            return None, None
            
        try:
            print(f"Computing correlations for profiles shape: {profiles_df.shape}")
            print(f"Columns: {list(profiles_df.columns)}")
            print(f"Index: {list(profiles_df.index)}")
            
            # The CSV structure has factors as rows (Factor_1, Factor_2, etc) and species as columns
            # We need to transpose to get factors as columns
            factor_matrix = profiles_df.values  # (factors x species)
            
            print(f"Factor matrix shape: {factor_matrix.shape}")
            
            if factor_matrix.shape[0] < 2:
                print(f"Need at least 2 factors, got {factor_matrix.shape[0]}")
                return None, None
                
            # Compute correlation between factors (rows in this case)
            factor_corr = np.corrcoef(factor_matrix)
            
            print(f"Factor correlation matrix shape: {factor_corr.shape}")
            
            # Compute diversity metrics
            # Mean absolute off-diagonal correlation (lower = more diverse)
            mask = ~np.eye(factor_corr.shape[0], dtype=bool)
            mean_abs_corr = np.mean(np.abs(factor_corr[mask]))
            
            # Maximum absolute correlation (lower = more distinct)
            max_abs_corr = np.max(np.abs(factor_corr[mask]))
            
            diversity_metrics = {
                'mean_abs_correlation': mean_abs_corr,
                'max_abs_correlation': max_abs_corr,
                'diversity_score': 1.0 - mean_abs_corr  # Higher = more diverse
            }
            
            print(f"Diversity metrics: {diversity_metrics}")
            
            return factor_corr, diversity_metrics
            
        except Exception as e:
            print(f"Error computing correlations: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def extract_closure_metrics(self, output_dir):
        """Extract closure metrics from PMF results"""
        try:
            # Look for closure summary CSV file
            closure_files = list(output_dir.glob("*_closure_summary.csv"))
            if not closure_files:
                print(f"No closure metrics found in {output_dir}")
                return None
                
            closure_df = pd.read_csv(closure_files[0])
            
            # Extract key metrics
            ch4_row = closure_df[closure_df['species'].str.upper() == 'CH4']
            if len(ch4_row) == 0:
                ch4_closure = None
                ch4_q_share = None
            else:
                ch4_closure = ch4_row['closure_pct'].iloc[0]
                ch4_q_share = ch4_row['q_share_pct'].iloc[0]
            
            # Overall metrics
            mean_closure = closure_df['closure_pct'].mean()
            std_closure = closure_df['closure_pct'].std()
            
            metrics = {
                'ch4_closure_pct': ch4_closure,
                'ch4_q_share_pct': ch4_q_share,
                'mean_closure_pct': mean_closure,
                'std_closure_pct': std_closure,
                'total_species': len(closure_df)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error extracting closure metrics from {output_dir}: {e}")
            return None
    
    def run_lambda_sweep(self):
        """Run PMF analysis across all lambda values"""
        print(f"Starting Lambda Sweep Analysis")
        print(f"Testing lambda values: {self.lambda_values}")
        print(f"Date range: {self.start_date} to {self.end_date}")
        print(f"Station: {self.station}")
        
        # 1. Run CH4 exclusion baseline
        print(f"\n=== BASELINE: CH4 Exclusion ===")
        
        exclusion_dir = self.run_pmf_scenario(
            "ch4_excluded", 
            ["--exclude-species", "CH4"]
        )
        
        if exclusion_dir:
            profiles = self.extract_factor_profiles(exclusion_dir)
            corr_matrix, diversity = self.compute_factor_correlations(profiles)
            closure = self.extract_closure_metrics(exclusion_dir)
            
            self.results['ch4_excluded'] = {
                'lambda': None,
                'factor_profiles': profiles,
                'correlation_matrix': corr_matrix,
                'diversity_metrics': diversity,
                'closure_metrics': closure,
                'output_dir': exclusion_dir
            }
            
            if diversity:
                print(f"Exclusion diversity score: {diversity['diversity_score']:.3f}")
        
        # 2. Run lambda sweep
        print(f"\n=== LAMBDA SWEEP ===")
        
        for lambda_val in self.lambda_values:
            scenario_name = f"lambda_{lambda_val}"
            
            if lambda_val == 0:
                # Baseline without regularization
                pmf_args = []
                print(f"\nLambda = {lambda_val} (Baseline)")
            else:
                # CH4 regularization with current lambda
                pmf_args = [
                    "--reg-species", "CH4",
                    "--reg-lambda", str(lambda_val),
                    "--reg-template", "zero"
                ]
                print(f"\nLambda = {lambda_val} (CH4 Regularization)")
            
            output_dir = self.run_pmf_scenario(scenario_name, pmf_args)
            
            if output_dir:
                profiles = self.extract_factor_profiles(output_dir)
                corr_matrix, diversity = self.compute_factor_correlations(profiles)
                closure = self.extract_closure_metrics(output_dir)
                
                self.results[scenario_name] = {
                    'lambda': lambda_val,
                    'factor_profiles': profiles,
                    'correlation_matrix': corr_matrix,
                    'diversity_metrics': diversity,
                    'closure_metrics': closure,
                    'output_dir': output_dir
                }
                
                if diversity:
                    print(f"Lambda={lambda_val} diversity score: {diversity['diversity_score']:.3f}")
                    
                # Store data for plotting
                if diversity:
                    self.correlation_data.append({
                        'lambda': lambda_val,
                        'scenario': scenario_name,
                        **diversity
                    })
                
                if closure:
                    closure_row = {'lambda': lambda_val, 'scenario': scenario_name, **closure}
                    self.closure_data.append(closure_row)
    
    def create_comparison_plots(self):
        """Create comprehensive comparison visualizations"""
        print(f"Creating comparison visualizations...")
        
        if not self.correlation_data:
            print("No correlation data available for plotting")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Lambda Sweep Analysis: CH4 Regularization vs Exclusion', 
                    fontsize=16, fontweight='bold')
        
        # Convert to DataFrames
        corr_df = pd.DataFrame(self.correlation_data)
        closure_df = pd.DataFrame(self.closure_data) if self.closure_data else pd.DataFrame()
        
        # Plot 1: Diversity Score vs Lambda
        ax1 = axes[0, 0]
        ax1.plot(corr_df['lambda'], corr_df['diversity_score'], 'bo-', linewidth=2, markersize=8)
        
        # Add CH4 exclusion baseline
        if 'ch4_excluded' in self.results and self.results['ch4_excluded']['diversity_metrics']:
            exclusion_diversity = self.results['ch4_excluded']['diversity_metrics']['diversity_score']
            ax1.axhline(y=exclusion_diversity, color='red', linestyle='--', linewidth=2, 
                       label=f'CH4 Exclusion ({exclusion_diversity:.3f})')
            ax1.legend()
        
        ax1.set_xlabel('Lambda')
        ax1.set_ylabel('Diversity Score')
        ax1.set_title('Factor Profile Diversity vs Lambda')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Mean Absolute Correlation vs Lambda
        ax2 = axes[0, 1]
        ax2.plot(corr_df['lambda'], corr_df['mean_abs_correlation'], 'go-', linewidth=2, markersize=8)
        
        if 'ch4_excluded' in self.results and self.results['ch4_excluded']['diversity_metrics']:
            exclusion_corr = self.results['ch4_excluded']['diversity_metrics']['mean_abs_correlation']
            ax2.axhline(y=exclusion_corr, color='red', linestyle='--', linewidth=2,
                       label=f'CH4 Exclusion ({exclusion_corr:.3f})')
            ax2.legend()
        
        ax2.set_xlabel('Lambda')
        ax2.set_ylabel('Mean Absolute Correlation')
        ax2.set_title('Factor Correlation vs Lambda')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: CH4 Closure vs Lambda (if available)
        ax3 = axes[1, 0]
        if not closure_df.empty and 'ch4_closure_pct' in closure_df.columns:
            valid_closure = closure_df.dropna(subset=['ch4_closure_pct'])
            if not valid_closure.empty:
                ax3.plot(valid_closure['lambda'], valid_closure['ch4_closure_pct'], 
                        'ro-', linewidth=2, markersize=8)
                ax3.set_xlabel('Lambda')
                ax3.set_ylabel('CH4 Closure %')
                ax3.set_title('CH4 Closure vs Lambda')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: CH4 Q Share vs Lambda (if available)
        ax4 = axes[1, 1]
        if not closure_df.empty and 'ch4_q_share_pct' in closure_df.columns:
            valid_q_share = closure_df.dropna(subset=['ch4_q_share_pct'])
            if not valid_q_share.empty:
                ax4.plot(valid_q_share['lambda'], valid_q_share['ch4_q_share_pct'], 
                        'mo-', linewidth=2, markersize=8)
                ax4.set_xlabel('Lambda')
                ax4.set_ylabel('CH4 Q Share %')
                ax4.set_title('CH4 Q Share vs Lambda')
                ax4.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        plot_path = self.base_output_dir / 'lambda_sweep_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved: {plot_path}")
        
        plt.show()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print(f"Generating summary report...")
        
        report_path = self.base_output_dir / 'lambda_sweep_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# Lambda Sweep Analysis Report\n\n")
            f.write(f"**Station:** {self.station}\n")
            f.write(f"**Date Range:** {self.start_date} to {self.end_date}\n")
            f.write(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
            
            f.write(f"## Research Question\n\n")
            f.write(f"Can CH4 regularization with increasing lambda values achieve similar factor ")
            f.write(f"correlation diversity as excluding CH4 entirely?\n\n")
            
            f.write(f"## Lambda Values Tested\n\n")
            f.write(f"Lambda = {', '.join(map(str, self.lambda_values))} (plus CH4 exclusion baseline)\n\n")
            
            # Results summary
            f.write(f"## Results Summary\n\n")
            
            # Exclusion baseline
            if 'ch4_excluded' in self.results:
                excl_result = self.results['ch4_excluded']
                if excl_result['diversity_metrics']:
                    div = excl_result['diversity_metrics']
                    f.write(f"### CH4 Exclusion Baseline\n")
                    f.write(f"- **Diversity Score:** {div['diversity_score']:.3f}\n")
                    f.write(f"- **Mean Abs Correlation:** {div['mean_abs_correlation']:.3f}\n")
                    f.write(f"- **Max Abs Correlation:** {div['max_abs_correlation']:.3f}\n\n")
            
            # Lambda results
            f.write(f"### Lambda Sweep Results\n\n")
            f.write(f"| Lambda | Diversity Score | Mean Correlation | CH4 Closure % |\n")
            f.write(f"|--------|----------------|------------------|---------------|\n")
            
            for lambda_val in self.lambda_values:
                scenario_name = f"lambda_{lambda_val}"
                if scenario_name in self.results:
                    result = self.results[scenario_name]
                    
                    # Diversity metrics
                    if result['diversity_metrics']:
                        div_score = f"{result['diversity_metrics']['diversity_score']:.3f}"
                        mean_corr = f"{result['diversity_metrics']['mean_abs_correlation']:.3f}"
                    else:
                        div_score = mean_corr = "N/A"
                    
                    # Closure metrics
                    if result['closure_metrics'] and result['closure_metrics']['ch4_closure_pct']:
                        ch4_closure = f"{result['closure_metrics']['ch4_closure_pct']:.1f}%"
                    else:
                        ch4_closure = "N/A"
                    
                    f.write(f"| {lambda_val} | {div_score} | {mean_corr} | {ch4_closure} |\n")
            
            # Analysis
            f.write(f"\n## Analysis\n\n")
            f.write(f"Generated files:\n")
            f.write(f"- **Summary Plot:** `lambda_sweep_comparison.png`\n")
            f.write(f"- **Individual Results:** `scenario_*/` directories\n\n")
        
        print(f"Summary report saved: {report_path}")
    
    def run_full_analysis(self):
        """Run complete lambda sweep analysis"""
        try:
            print(f"Starting Lambda Sweep Analysis")
            print(f"Output directory: {self.base_output_dir}")
            
            # Run sweep
            self.run_lambda_sweep()
            
            # Create visualizations
            self.create_comparison_plots()
            
            # Generate report
            self.generate_summary_report()
            
            print(f"Lambda Sweep Analysis Complete!")
            print(f"Results saved in: {self.base_output_dir}")
            
        except KeyboardInterrupt:
            print(f"Analysis interrupted by user")
        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main function for command-line usage"""
    analyzer = LambdaSweepAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()