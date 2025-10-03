#!/usr/bin/env python3
"""
Stage 9: Species Regularization Diagnostics and Validation Framework

This module provides comprehensive diagnostic capabilities for species regularization,
including convergence analysis, lambda sensitivity testing, and species push-out validation.

Key Features:
- Convergence tracking during staged training
- Lambda sweep analysis for sensitivity testing  
- Species push-out effectiveness metrics
- Before/after factor comparison analysis
- Comprehensive dashboard integration
- Mathematical validation of regularization behavior

Mathematical Foundation:
- Ridge regularization: min_{W,H ≥ 0} 1/2 || (V - WH) ⊙ We^{1/2} ||_F^2 + (λ/2) ||H[:, j*] - h0||_2^2
- Proximal update: (W^T D W + λ I) h = W^T D v + λ h0, h ← max(h, 0)
- Convergence metric: rel_change = ||h_new - h_old||_2 / ||h_old||_2
- Push-out metric: reduction in target species factor loadings vs baseline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

@dataclass
class RegularizationConvergenceMetrics:
    """Metrics tracking convergence during staged regularization training."""
    burst_number: int
    q_true_start: float
    q_true_end: float
    q_robust_start: float  
    q_robust_end: float
    objective_reduction: float
    relative_change: float
    species_name: str
    lambda_value: float
    converged: bool
    iterations_used: int

@dataclass  
class SpeciesPushOutMetrics:
    """Metrics measuring species push-out effectiveness."""
    species_name: str
    lambda_value: float
    baseline_max_loading: float
    regularized_max_loading: float
    loading_reduction_pct: float
    baseline_factor_correlation: float
    regularized_factor_correlation: float
    correlation_reduction: float
    factor_entropy_change: float
    push_out_score: float  # Combined effectiveness metric

@dataclass
class LambdaSensitivityPoint:
    """Single point in lambda sensitivity analysis."""
    lambda_value: float
    species_name: str
    max_factor_loading: float
    factor_correlation: float
    q_robust: float
    convergence_bursts: int
    push_out_score: float

class RegularizationDiagnostics:
    """Comprehensive diagnostics system for species regularization validation."""
    
    def __init__(self):
        self.convergence_history = []
        self.push_out_metrics = []
        self.lambda_sensitivity_data = []
        self.baseline_results = None
        self.regularized_results = None
        
    def track_convergence(self, burst_num: int, species_name: str, lambda_val: float,
                         q_start: Tuple[float, float], q_end: Tuple[float, float],
                         obj_reduction: float, rel_change: float, converged: bool,
                         iterations: int) -> None:
        """Track convergence metrics for a single burst."""
        
        metrics = RegularizationConvergenceMetrics(
            burst_number=burst_num,
            q_true_start=q_start[0],
            q_true_end=q_end[0], 
            q_robust_start=q_start[1],
            q_robust_end=q_end[1],
            objective_reduction=obj_reduction,
            relative_change=rel_change,
            species_name=species_name,
            lambda_value=lambda_val,
            converged=converged,
            iterations_used=iterations
        )
        
        self.convergence_history.append(metrics)
        
    def compute_push_out_metrics(self, species_name: str, lambda_val: float,
                                baseline_H: np.ndarray, regularized_H: np.ndarray,
                                species_idx: int) -> SpeciesPushOutMetrics:
        """Compute comprehensive push-out effectiveness metrics."""
        
        # Extract species profiles (columns j* for target species)
        baseline_profile = baseline_H[:, species_idx]
        regularized_profile = regularized_H[:, species_idx]
        
        # Maximum factor loading reduction
        baseline_max = np.max(baseline_profile)
        regularized_max = np.max(regularized_profile)
        loading_reduction = 100 * (1 - regularized_max / baseline_max) if baseline_max > 0 else 0
        
        # Factor correlation with other species (measure of separation)
        baseline_corr = self._compute_factor_correlation(baseline_H, species_idx)
        regularized_corr = self._compute_factor_correlation(regularized_H, species_idx)
        correlation_reduction = baseline_corr - regularized_corr
        
        # Factor entropy change (measure of concentration vs spread)
        baseline_entropy = self._compute_factor_entropy(baseline_profile)
        regularized_entropy = self._compute_factor_entropy(regularized_profile) 
        entropy_change = regularized_entropy - baseline_entropy
        
        # Combined push-out effectiveness score
        push_out_score = loading_reduction * 0.4 + correlation_reduction * 100 * 0.4 + entropy_change * 0.2
        
        return SpeciesPushOutMetrics(
            species_name=species_name,
            lambda_value=lambda_val,
            baseline_max_loading=baseline_max,
            regularized_max_loading=regularized_max,
            loading_reduction_pct=loading_reduction,
            baseline_factor_correlation=baseline_corr,
            regularized_factor_correlation=regularized_corr,
            correlation_reduction=correlation_reduction,
            factor_entropy_change=entropy_change,
            push_out_score=push_out_score
        )
        
    def run_lambda_sensitivity_analysis(self, pmf_analyzer: 'MMFPMFAnalyzer',
                                       species_name: str, lambda_range: List[float],
                                       output_dir: Path) -> List[LambdaSensitivityPoint]:
        """Run comprehensive lambda sensitivity analysis."""
        
        print(f"[SWEEP] Running lambda sensitivity analysis for {species_name}")
        print(f"   Lambda range: {lambda_range}")
        
        sensitivity_points = []
        
        for i, lambda_val in enumerate(lambda_range):
            print(f"   [STEP] {i+1}/{len(lambda_range)}: λ = {lambda_val}")
            
            # Configure regularization for this lambda
            pmf_analyzer.reg_species = [species_name]
            pmf_analyzer.reg_lambda = [lambda_val]
            pmf_analyzer.reg_template = ['zero']
            
            try:
                # Run PMF analysis with this lambda
                pmf_analyzer.output_dir = output_dir / f"lambda_sweep_{species_name}_{lambda_val}"
                pmf_analyzer.output_dir.mkdir(exist_ok=True)
                
                results = pmf_analyzer.run_pmf_analysis()
                
                if results and hasattr(results, 'H'):
                    # Extract metrics for this lambda point
                    species_idx = pmf_analyzer._regularization_plan['species_indices'][species_name]
                    max_loading = np.max(results.H[:, species_idx])
                    factor_corr = self._compute_factor_correlation(results.H, species_idx)
                    
                    point = LambdaSensitivityPoint(
                        lambda_value=lambda_val,
                        species_name=species_name,
                        max_factor_loading=max_loading,
                        factor_correlation=factor_corr,
                        q_robust=results.Qrobust,
                        convergence_bursts=len(self.convergence_history),
                        push_out_score=max_loading * factor_corr  # Simple combined metric
                    )
                    
                    sensitivity_points.append(point)
                    
            except Exception as e:
                print(f"   [ERROR] Lambda {lambda_val} failed: {e}")
                continue
                
        self.lambda_sensitivity_data.extend(sensitivity_points)
        return sensitivity_points
        
    def _compute_factor_correlation(self, H: np.ndarray, species_idx: int) -> float:
        """Compute average correlation of target species with other factors."""
        
        target_profile = H[:, species_idx]
        other_profiles = np.delete(H, species_idx, axis=1)
        
        if other_profiles.shape[1] == 0:
            return 0.0
            
        correlations = []
        for i in range(other_profiles.shape[1]):
            corr = np.corrcoef(target_profile, other_profiles[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
                
        return np.mean(correlations) if correlations else 0.0
        
    def _compute_factor_entropy(self, profile: np.ndarray) -> float:
        """Compute Shannon entropy of factor profile (normalized)."""
        
        # Normalize to probability distribution
        profile_norm = profile / np.sum(profile) if np.sum(profile) > 0 else profile
        profile_norm = profile_norm[profile_norm > 0]  # Remove zeros for log
        
        if len(profile_norm) == 0:
            return 0.0
            
        return -np.sum(profile_norm * np.log2(profile_norm))
        
    def generate_convergence_plots(self, output_dir: Path, prefix: str) -> Path:
        """Generate comprehensive convergence analysis plots."""
        
        if not self.convergence_history:
            print("[WARN] No convergence data available for plotting")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regularization Convergence Analysis', fontsize=16)
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame([asdict(m) for m in self.convergence_history])
        
        # Plot 1: Objective reduction by burst
        axes[0, 0].bar(df['burst_number'], df['objective_reduction'])
        axes[0, 0].set_title('Objective Reduction per Burst')
        axes[0, 0].set_xlabel('Burst Number')
        axes[0, 0].set_ylabel('Objective Reduction')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Relative change convergence
        axes[0, 1].plot(df['burst_number'], df['relative_change'], 'o-')
        axes[0, 1].axhline(y=0.0001, color='r', linestyle='--', label='Convergence Threshold')
        axes[0, 1].set_title('Relative Change Convergence')
        axes[0, 1].set_xlabel('Burst Number')  
        axes[0, 1].set_ylabel('Relative Change')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Plot 3: Q-value progression
        axes[1, 0].plot(df['burst_number'], df['q_robust_start'], 'o-', label='Q(robust) Start')
        axes[1, 0].plot(df['burst_number'], df['q_robust_end'], 's-', label='Q(robust) End')
        axes[1, 0].set_title('Q-Value Progression')
        axes[1, 0].set_xlabel('Burst Number')
        axes[1, 0].set_ylabel('Q(robust)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Plot 4: Convergence efficiency
        df['efficiency'] = df['objective_reduction'] / df['iterations_used']
        axes[1, 1].bar(df['burst_number'], df['efficiency'])
        axes[1, 1].set_title('Convergence Efficiency (Reduction/Iteration)')
        axes[1, 1].set_xlabel('Burst Number')
        axes[1, 1].set_ylabel('Objective Reduction per Iteration')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        convergence_file = output_dir / f"{prefix}_regularization_convergence.png"
        plt.savefig(convergence_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return convergence_file
        
    def generate_lambda_sensitivity_plots(self, output_dir: Path, prefix: str) -> Path:
        """Generate lambda sensitivity analysis plots."""
        
        if not self.lambda_sensitivity_data:
            print("[WARN] No lambda sensitivity data available for plotting")
            return None
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Lambda Sensitivity Analysis', fontsize=16)
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(p) for p in self.lambda_sensitivity_data])
        
        # Plot 1: Max factor loading vs lambda
        axes[0, 0].semilogx(df['lambda_value'], df['max_factor_loading'], 'o-')
        axes[0, 0].set_title('Maximum Factor Loading vs Lambda')
        axes[0, 0].set_xlabel('Lambda (log scale)')
        axes[0, 0].set_ylabel('Maximum Factor Loading')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Factor correlation vs lambda
        axes[0, 1].semilogx(df['lambda_value'], df['factor_correlation'], 'o-')
        axes[0, 1].set_title('Factor Correlation vs Lambda')
        axes[0, 1].set_xlabel('Lambda (log scale)')
        axes[0, 1].set_ylabel('Average Factor Correlation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Q(robust) vs lambda (model fit quality)
        axes[1, 0].semilogx(df['lambda_value'], df['q_robust'], 'o-')
        axes[1, 0].set_title('Model Fit Quality vs Lambda')
        axes[1, 0].set_xlabel('Lambda (log scale)')
        axes[1, 0].set_ylabel('Q(robust)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Combined push-out score vs lambda
        axes[1, 1].semilogx(df['lambda_value'], df['push_out_score'], 'o-')
        axes[1, 1].set_title('Push-Out Effectiveness vs Lambda')
        axes[1, 1].set_xlabel('Lambda (log scale)')
        axes[1, 1].set_ylabel('Push-Out Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        sensitivity_file = output_dir / f"{prefix}_lambda_sensitivity.png"
        plt.savefig(sensitivity_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return sensitivity_file
        
    def save_diagnostics_csv(self, output_dir: Path, prefix: str) -> None:
        """Save all diagnostic data to CSV files."""
        
        # Save convergence metrics
        if self.convergence_history:
            conv_df = pd.DataFrame([asdict(m) for m in self.convergence_history])
            conv_file = output_dir / f"{prefix}_regularization_convergence.csv"
            conv_df.to_csv(conv_file, index=False)
            print(f"[SAVE] Convergence metrics: {conv_file}")
            
        # Save push-out metrics
        if self.push_out_metrics:
            pushout_df = pd.DataFrame([asdict(m) for m in self.push_out_metrics])
            pushout_file = output_dir / f"{prefix}_species_pushout_metrics.csv"
            pushout_df.to_csv(pushout_file, index=False)
            print(f"[SAVE] Push-out metrics: {pushout_file}")
            
        # Save lambda sensitivity data
        if self.lambda_sensitivity_data:
            sensitivity_df = pd.DataFrame([asdict(p) for p in self.lambda_sensitivity_data])
            sensitivity_file = output_dir / f"{prefix}_lambda_sensitivity.csv"
            sensitivity_df.to_csv(sensitivity_file, index=False)
            print(f"[SAVE] Lambda sensitivity: {sensitivity_file}")
            
    def generate_diagnostic_summary_report(self, output_dir: Path, prefix: str) -> Path:
        """Generate comprehensive diagnostic summary report."""
        
        report_file = output_dir / f"{prefix}_regularization_diagnostics_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("SPECIES REGULARIZATION DIAGNOSTIC REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Convergence Analysis
            if self.convergence_history:
                f.write("CONVERGENCE ANALYSIS\n")
                f.write("-" * 20 + "\n")
                
                total_bursts = len(self.convergence_history)
                total_obj_reduction = sum(m.objective_reduction for m in self.convergence_history)
                avg_rel_change = np.mean([m.relative_change for m in self.convergence_history])
                
                f.write(f"Total Training Bursts: {total_bursts}\n")
                f.write(f"Total Objective Reduction: {total_obj_reduction:.2f}\n") 
                f.write(f"Average Relative Change: {avg_rel_change:.6f}\n")
                
                # Check if converged
                final_rel_change = self.convergence_history[-1].relative_change
                converged = final_rel_change < 0.0001
                f.write(f"Final Relative Change: {final_rel_change:.6f}\n")
                f.write(f"Converged (< 1e-4): {'Yes' if converged else 'No'}\n\n")
                
            # Push-out effectiveness  
            if self.push_out_metrics:
                f.write("SPECIES PUSH-OUT EFFECTIVENESS\n")
                f.write("-" * 30 + "\n")
                
                for metric in self.push_out_metrics:
                    f.write(f"Species: {metric.species_name} (λ = {metric.lambda_value})\n")
                    f.write(f"  Max Loading Reduction: {metric.loading_reduction_pct:.1f}%\n")
                    f.write(f"  Correlation Reduction: {metric.correlation_reduction:.3f}\n")
                    f.write(f"  Overall Push-Out Score: {metric.push_out_score:.2f}\n\n")
                    
            # Lambda sensitivity summary
            if self.lambda_sensitivity_data:
                f.write("LAMBDA SENSITIVITY SUMMARY\n")
                f.write("-" * 25 + "\n")
                
                df = pd.DataFrame([asdict(p) for p in self.lambda_sensitivity_data])
                
                f.write(f"Lambda Range: {df['lambda_value'].min():.3f} - {df['lambda_value'].max():.3f}\n")
                f.write(f"Optimal Lambda (max push-out): {df.loc[df['push_out_score'].idxmax(), 'lambda_value']:.3f}\n")
                f.write(f"Best Q(robust): {df['q_robust'].min():.2f}\n")
                f.write(f"Max Loading Reduction: {(1 - df['max_factor_loading'].min() / df['max_factor_loading'].max()) * 100:.1f}%\n\n")
                
        print(f"[SAVE] Diagnostic report: {report_file}")
        return report_file


def create_regularization_diagnostics() -> RegularizationDiagnostics:
    """Factory function to create regularization diagnostics system."""
    return RegularizationDiagnostics()