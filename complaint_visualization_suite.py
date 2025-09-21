#!/usr/bin/env python3
"""
Complaint Visualization Suite
=============================

Comprehensive plotting module for complaint-PMF correlation analysis.
Creates 12 specialized plots for validating source apportionment results
against ground truth malodour complaint data.

Features:
- Complaint frequency and temporal pattern plots
- Factor-complaint correlation visualizations  
- Environmental condition analysis during complaints
- Predictive model performance plots
- ROC curves and validation metrics
- Interactive plots with Plotly integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import Plotly for interactive plots
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.figure_factory as ff
    PLOTLY_AVAILABLE = True
    print("📊 Plotly available for interactive plots")
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available - using matplotlib only")

class ComplaintVisualizationSuite:
    """
    Comprehensive visualization suite for complaint-PMF analysis.
    Creates publication-quality plots for validating source apportionment.
    """
    
    def __init__(self, complaint_analyzer):
        """
        Initialize visualization suite with complaint analyzer results.
        
        Args:
            complaint_analyzer: ComplaintPMFAnalyzer instance with completed analysis
        """
        self.complaint_analyzer = complaint_analyzer
        self.pmf_analyzer = complaint_analyzer.pmf_analyzer
        
        # Set consistent styling
        plt.style.use('default')
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        
        # Create output directory for plots
        self.plots_dir = self.pmf_analyzer.output_dir / "complaint_plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # File prefix for consistent naming
        self.file_prefix = f"{self.pmf_analyzer.filename_prefix}_complaint"
        
    def create_all_complaint_plots(self):
        """
        Create complete suite of complaint analysis plots.
        Returns list of created plot files.
        """
        print("🎨 Creating comprehensive complaint visualization suite...")
        
        plot_files = []
        
        # Primary complaint analysis plots
        plot_files.extend(self._create_complaint_frequency_plots())
        plot_files.extend(self._create_factor_complaint_correlation_plots())
        plot_files.extend(self._create_temporal_complaint_plots())
        plot_files.extend(self._create_environmental_complaint_plots())
        
        # Validation and performance plots
        plot_files.extend(self._create_validation_plots())
        plot_files.extend(self._create_prediction_performance_plots())
        
        # Advanced analysis plots
        plot_files.extend(self._create_advanced_analysis_plots())
        
        print(f"✅ Created {len(plot_files)} complaint analysis plots")
        return plot_files
    
    def _create_complaint_frequency_plots(self):
        """Create plots showing complaint frequency distributions and patterns."""
        print("   📊 Creating complaint frequency plots...")
        
        plot_files = []
        
        if not hasattr(self.complaint_analyzer, 'aligned_data'):
            print("     ⚠️ No aligned data available - skipping frequency plots")
            return plot_files
        
        complaint_data = self.complaint_analyzer.aligned_data['complaint_count']
        datetime_index = self.complaint_analyzer.aligned_data.index
        
        # Plot 1: Complaint Frequency Distribution
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.pmf_analyzer.station} Complaint Frequency Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Daily complaint histogram
        ax1 = axes[0, 0]
        ax1.hist(complaint_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(complaint_data.mean(), color='red', linestyle='--', 
                   label=f'Mean: {complaint_data.mean():.1f}')
        ax1.axvline(complaint_data.median(), color='green', linestyle='--', 
                   label=f'Median: {complaint_data.median():.1f}')
        ax1.set_xlabel('Daily Complaint Count')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Daily Complaints')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Complaint time series
        ax2 = axes[0, 1]
        ax2.plot(datetime_index, complaint_data, linewidth=1, alpha=0.7, color='darkblue')
        ax2.scatter(datetime_index[complaint_data >= 1], complaint_data[complaint_data >= 1], 
                   color='red', alpha=0.6, s=20, label='Complaint Days')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Complaint Count')
        ax2.set_title('Complaints Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Cumulative complaints
        ax3 = axes[1, 0]
        cumulative_complaints = complaint_data.cumsum()
        ax3.plot(datetime_index, cumulative_complaints, linewidth=2, color='purple')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Cumulative Complaints')
        ax3.set_title('Cumulative Complaints Over Time')
        ax3.grid(True, alpha=0.3)
        
        # Box plot by month
        ax4 = axes[1, 1]
        monthly_data = []
        month_labels = []
        for month in range(1, 13):
            month_complaints = complaint_data[datetime_index.month == month]
            if len(month_complaints) > 0:
                monthly_data.append(month_complaints)
                month_labels.append(pd.to_datetime(f'2023-{month:02d}-01').strftime('%b'))
        
        if monthly_data:
            bp = ax4.boxplot(monthly_data, labels=month_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightcoral')
            ax4.set_xlabel('Month')
            ax4.set_ylabel('Daily Complaint Count')
            ax4.set_title('Monthly Complaint Distribution')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.plots_dir / f"{self.file_prefix}_frequency_distribution.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_file)
        
        return plot_files
    
    def _create_factor_complaint_correlation_plots(self):
        """Create correlation analysis plots between PMF factors and complaints."""
        print("   🔗 Creating factor-complaint correlation plots...")
        
        plot_files = []
        
        if not hasattr(self.complaint_analyzer, 'correlation_results'):
            print("     ⚠️ No correlation results available")
            return plot_files
        
        correlation_results = self.complaint_analyzer.correlation_results
        
        # Plot 2: Factor-Complaint Correlation Matrix
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.pmf_analyzer.station} Factor-Complaint Correlations', 
                    fontsize=16, fontweight='bold')
        
        # Extract correlation data
        factors = list(correlation_results.keys())
        methods = ['pearson', 'spearman', 'kendall']
        
        for i, method in enumerate(methods):
            if i < 3:  # First 3 subplots
                row, col = divmod(i, 2)
                ax = axes[row, col]
                
                correlations = []
                p_values = []
                
                for factor in factors:
                    if method in correlation_results[factor]['zero_lag']:
                        r = correlation_results[factor]['zero_lag'][method]['r']
                        p = correlation_results[factor]['zero_lag'][method]['p']
                        correlations.append(r)
                        p_values.append(p)
                    else:
                        correlations.append(0)
                        p_values.append(1)
                
                # Create correlation bar plot
                bars = ax.bar(range(len(factors)), correlations, alpha=0.7)
                
                # Color bars by significance
                for j, (bar, p) in enumerate(zip(bars, p_values)):
                    if p < 0.05:
                        bar.set_color('red')  # Significant
                    elif p < 0.10:
                        bar.set_color('orange')  # Marginally significant
                    else:
                        bar.set_color('lightblue')  # Not significant
                
                ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                ax.set_xlabel('PMF Factors')
                ax.set_ylabel(f'{method.capitalize()} Correlation')
                ax.set_title(f'{method.capitalize()} Correlations with Complaints')
                ax.set_xticks(range(len(factors)))
                ax.set_xticklabels([f.replace('Factor_', 'F') for f in factors], rotation=45)
                ax.grid(True, alpha=0.3)
                
                # Add significance threshold lines
                ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, label='Strong correlation (|r| > 0.3)')
                ax.axhline(y=-0.3, color='red', linestyle='--', alpha=0.5)
                ax.legend()
        
        # Time-lagged correlation plot
        ax4 = axes[1, 1]
        
        # Find factor with best lagged correlation
        best_factor = None
        best_lag_data = None
        max_correlation = 0
        
        for factor, results in correlation_results.items():
            if results['lagged_correlations']:
                for lag, lag_data in results['lagged_correlations'].items():
                    if abs(lag_data['r']) > max_correlation:
                        max_correlation = abs(lag_data['r'])
                        best_factor = factor
                        best_lag_data = results['lagged_correlations']
        
        if best_factor and best_lag_data:
            lags = list(best_lag_data.keys())
            correlations = [best_lag_data[lag]['r'] for lag in lags]
            significance = [best_lag_data[lag]['significant'] for lag in lags]
            
            # Plot lagged correlations
            colors = ['red' if sig else 'lightblue' for sig in significance]
            ax4.bar(lags, correlations, alpha=0.7, color=colors)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax4.set_xlabel('Lag (hours)')
            ax4.set_ylabel('Correlation Coefficient')
            ax4.set_title(f'Time-Lagged Correlations ({best_factor.replace("Factor_", "Factor ")})')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.plots_dir / f"{self.file_prefix}_correlation_matrix.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_file)
        
        return plot_files
    
    def _create_temporal_complaint_plots(self):
        """Create temporal pattern analysis plots for complaints."""
        print("   📅 Creating temporal complaint pattern plots...")
        
        plot_files = []
        
        if 'temporal_patterns' not in self.complaint_analyzer.validation_metrics:
            print("     ⚠️ No temporal pattern data available")
            return plot_files
        
        temporal_data = self.complaint_analyzer.validation_metrics['temporal_patterns']
        
        # Plot 3: Temporal Complaint Patterns
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.pmf_analyzer.station} Temporal Complaint Patterns', 
                    fontsize=16, fontweight='bold')
        
        # Hourly patterns
        ax1 = axes[0, 0]
        hourly_means = temporal_data['hourly']['mean']
        ax1.plot(hourly_means.index, hourly_means, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Mean Complaints per Hour')
        ax1.set_title('Diurnal Complaint Pattern')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3)
        
        # Daily patterns (day of week)
        ax2 = axes[0, 1]
        daily_means = temporal_data['daily']['mean']
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        ax2.bar(days, daily_means, alpha=0.7, color='lightgreen')
        ax2.set_xlabel('Day of Week')
        ax2.set_ylabel('Mean Complaints per Day')
        ax2.set_title('Weekly Complaint Pattern')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Monthly patterns
        ax3 = axes[1, 0]
        monthly_means = temporal_data['monthly']['mean']
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_indices = monthly_means.index
        month_labels = [months[i-1] for i in month_indices]
        ax3.bar(month_labels, monthly_means, alpha=0.7, color='coral')
        ax3.set_xlabel('Month')
        ax3.set_ylabel('Mean Complaints per Month')
        ax3.set_title('Seasonal Complaint Pattern')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Weekend vs Weekday comparison
        ax4 = axes[1, 1]
        weekend_data = temporal_data['weekend_vs_weekday']
        categories = ['Weekday', 'Weekend']
        means = [weekend_data['weekday']['mean'], weekend_data['weekend']['mean']]
        colors = ['steelblue', 'orange']
        
        bars = ax4.bar(categories, means, alpha=0.7, color=colors)
        ax4.set_ylabel('Mean Complaints per Day')
        ax4.set_title('Weekend vs Weekday Complaints')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, means):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plot_file = self.plots_dir / f"{self.file_prefix}_temporal_patterns.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_file)
        
        return plot_files
    
    def _create_environmental_complaint_plots(self):
        """Create environmental condition analysis plots during complaint events."""
        print("   🌦️ Creating environmental-complaint condition plots...")
        
        plot_files = []
        
        if 'environmental_conditions' not in self.complaint_analyzer.validation_metrics:
            print("     ⚠️ No environmental condition data available")
            return plot_files
        
        env_data = self.complaint_analyzer.validation_metrics['environmental_conditions']
        
        # Plot 4: Environmental Conditions During Complaints
        n_vars = len(env_data)
        if n_vars == 0:
            return plot_files
        
        # Calculate subplot layout
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle(f'{self.pmf_analyzer.station} Environmental Conditions During Complaints', 
                    fontsize=16, fontweight='bold')
        
        if n_vars == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if n_vars > 1 else [axes]
        else:
            axes = axes.flatten()
        
        for i, (var, var_data) in enumerate(env_data.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Extract statistics
            high_stats = var_data['high_complaint_stats']
            low_stats = var_data['low_complaint_stats']
            
            # Create comparison bar plot
            categories = ['Low Complaints', 'High Complaints']
            means = [low_stats['mean'], high_stats['mean']]
            stds = [low_stats['std'], high_stats['std']]
            
            bars = ax.bar(categories, means, yerr=stds, capsize=5, alpha=0.7, 
                         color=['lightblue', 'lightcoral'])
            
            ax.set_ylabel(f'{var}')
            ax.set_title(f'{var} Distribution')
            ax.grid(True, alpha=0.3)
            
            # Add statistical significance indicator
            t_test = var_data['statistical_tests']['t_test']
            if t_test['significant']:
                # Add significance indicator
                y_max = max(means) + max(stds)
                ax.plot([0, 1], [y_max * 1.1, y_max * 1.1], 'k-', linewidth=1)
                ax.plot([0, 0], [y_max * 1.05, y_max * 1.1], 'k-', linewidth=1)
                ax.plot([1, 1], [y_max * 1.05, y_max * 1.1], 'k-', linewidth=1)
                ax.text(0.5, y_max * 1.15, f'p = {t_test["p_value"]:.3f}***', 
                       ha='center', va='bottom', fontweight='bold', color='red')
            
            # Add value labels
            for bar, mean, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2, mean + std + 0.01,
                       f'{mean:.2f}', ha='center', va='bottom', fontsize=9)
        
        # Hide unused subplots
        for i in range(len(env_data), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_file = self.plots_dir / f"{self.file_prefix}_environmental_conditions.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_file)
        
        return plot_files
    
    def _create_validation_plots(self):
        """Create validation metric plots including ROC curves and confusion matrices."""
        print("   🎯 Creating validation performance plots...")
        
        plot_files = []
        
        if 'classification' not in self.complaint_analyzer.validation_metrics:
            print("     ⚠️ No classification results available")
            return plot_files
        
        class_data = self.complaint_analyzer.validation_metrics['classification']
        
        # Plot 5: Validation Metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.pmf_analyzer.station} PMF Validation Against Complaints', 
                    fontsize=16, fontweight='bold')
        
        # ROC Curve
        ax1 = axes[0, 0]
        roc_data = class_data['roc_curve']
        ax1.plot(roc_data['fpr'], roc_data['tpr'], linewidth=2, 
                label=f'ROC (AUC = {class_data["roc_auc"]:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title('ROC Curve - Complaint Prediction')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Confusion Matrix
        ax2 = axes[0, 1]
        conf_matrix = class_data['confusion_matrix']
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.set_title('Confusion Matrix')
        
        # Feature Importance
        ax3 = axes[1, 0]
        feature_importance = class_data['feature_importance']
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        # Sort by importance
        sorted_indices = np.argsort(importances)[::-1]
        sorted_features = [features[i].replace('Factor_', 'F') for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        bars = ax3.bar(sorted_features, sorted_importances, alpha=0.7, color='lightgreen')
        ax3.set_xlabel('PMF Factors')
        ax3.set_ylabel('Feature Importance')
        ax3.set_title('Factor Importance for Complaint Prediction')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Prediction vs Actual
        ax4 = axes[1, 1]
        predictions = class_data['predictions']
        y_true = predictions['y_true']
        y_prob = predictions['y_prob']
        
        # Create probability histogram by true class
        ax4.hist(y_prob[~y_true], bins=20, alpha=0.5, label='No Complaints', color='blue')
        ax4.hist(y_prob[y_true], bins=20, alpha=0.5, label='Complaints', color='red')
        ax4.set_xlabel('Predicted Probability')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Probability Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.plots_dir / f"{self.file_prefix}_validation_metrics.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_file)
        
        return plot_files
    
    def _create_prediction_performance_plots(self):
        """Create prediction model performance comparison plots."""
        print("   🤖 Creating prediction model performance plots...")
        
        plot_files = []
        
        if not hasattr(self.complaint_analyzer, 'prediction_models'):
            print("     ⚠️ No prediction models available")
            return plot_files
        
        models = self.complaint_analyzer.prediction_models
        
        # Plot 6: Model Performance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.pmf_analyzer.station} Prediction Model Performance', 
                    fontsize=16, fontweight='bold')
        
        # Model R² comparison
        ax1 = axes[0, 0]
        model_names = []
        r2_means = []
        r2_stds = []
        
        for model_name, model_data in models.items():
            model_names.append(model_name.replace('_', ' ').title())
            r2_means.append(model_data['mean_r2'])
            r2_stds.append(model_data['std_r2'])
        
        bars = ax1.bar(model_names, r2_means, yerr=r2_stds, capsize=5, alpha=0.7)
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance Comparison')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, r2, std in zip(bars, r2_means, r2_stds):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Feature importance comparison (use best model)
        best_model_name = max(models.keys(), key=lambda k: models[k]['mean_r2'])
        best_model = models[best_model_name]
        
        ax2 = axes[0, 1]
        feature_imp = best_model['feature_importance']
        features = list(feature_imp.keys())
        importances = list(feature_imp.values())
        
        # Sort by absolute importance
        sorted_indices = np.argsort(np.abs(importances))[::-1]
        sorted_features = [features[i].replace('Factor_', 'F') for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        # Color bars by positive/negative importance
        colors = ['red' if imp > 0 else 'blue' for imp in sorted_importances]
        bars = ax2.bar(sorted_features, sorted_importances, alpha=0.7, color=colors)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('PMF Factors')
        ax2.set_ylabel('Feature Importance')
        ax2.set_title(f'Feature Importance ({best_model_name.replace("_", " ").title()})')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Cross-validation scores
        ax3 = axes[1, 0]
        for i, (model_name, model_data) in enumerate(models.items()):
            scores = model_data['scores']
            x_pos = np.full(len(scores), i) + np.random.normal(0, 0.1, len(scores))
            ax3.scatter(x_pos, scores, alpha=0.7, s=50)
            ax3.plot([i-0.2, i+0.2], [scores.mean(), scores.mean()], 'r-', linewidth=2)
        
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([name.replace('_', ' ').title() for name in models.keys()], 
                           rotation=45)
        ax3.set_ylabel('Cross-Validation R² Score')
        ax3.set_title('Cross-Validation Score Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Model complexity vs performance
        ax4 = axes[1, 1]
        
        # Create a summary table instead
        ax4.axis('tight')
        ax4.axis('off')
        
        # Create performance summary table
        table_data = []
        for model_name, model_data in models.items():
            table_data.append([
                model_name.replace('_', ' ').title(),
                f"{model_data['mean_r2']:.3f}",
                f"±{model_data['std_r2']:.3f}",
                f"{max(model_data['scores']):.3f}",
                f"{min(model_data['scores']):.3f}"
            ])
        
        table_headers = ['Model', 'Mean R²', 'Std R²', 'Max R²', 'Min R²']
        table = ax4.table(cellText=table_data, colLabels=table_headers, 
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Model Performance Summary', pad=20)
        
        plt.tight_layout()
        plot_file = self.plots_dir / f"{self.file_prefix}_model_performance.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_file)
        
        return plot_files
    
    def _create_advanced_analysis_plots(self):
        """Create advanced analysis plots including multivariate relationships."""
        print("   🔬 Creating advanced analysis plots...")
        
        plot_files = []
        
        if not hasattr(self.complaint_analyzer, 'aligned_data'):
            print("     ⚠️ No aligned data available")
            return plot_files
        
        aligned_data = self.complaint_analyzer.aligned_data
        factor_cols = [col for col in aligned_data.columns if col.startswith('Factor_')]
        
        if len(factor_cols) < 2:
            print("     ⚠️ Need at least 2 factors for advanced plots")
            return plot_files
        
        # Plot 7: Advanced Multivariate Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.pmf_analyzer.station} Advanced Complaint-Factor Analysis', 
                    fontsize=16, fontweight='bold')
        
        complaint_data = aligned_data['complaint_count']
        
        # 3D-style scatter plot (Factor 1 vs Factor 2, colored by complaints)
        ax1 = axes[0, 0]
        factor1_data = aligned_data[factor_cols[0]]
        factor2_data = aligned_data[factor_cols[1]]
        
        scatter = ax1.scatter(factor1_data, factor2_data, c=complaint_data, 
                            cmap='Reds', alpha=0.6, s=30)
        ax1.set_xlabel(factor_cols[0].replace('Factor_', 'Factor '))
        ax1.set_ylabel(factor_cols[1].replace('Factor_', 'Factor '))
        ax1.set_title('Factors vs Complaints (Color = Complaint Count)')
        plt.colorbar(scatter, ax=ax1, label='Complaints')
        ax1.grid(True, alpha=0.3)
        
        # Complaint-factor response curves
        ax2 = axes[0, 1]
        
        # Choose factor with highest correlation
        best_factor = factor_cols[0]  # Default
        if hasattr(self.complaint_analyzer, 'correlation_results'):
            max_corr = 0
            for factor, results in self.complaint_analyzer.correlation_results.items():
                if factor in factor_cols:
                    pearson_r = abs(results['zero_lag']['pearson']['r'])
                    if pearson_r > max_corr:
                        max_corr = pearson_r
                        best_factor = factor
        
        # Bin factor data and plot mean complaints per bin
        factor_data = aligned_data[best_factor]
        n_bins = 10
        bins = pd.qcut(factor_data, n_bins, duplicates='drop')
        binned_complaints = complaint_data.groupby(bins).agg(['mean', 'std', 'count'])
        
        bin_centers = [interval.mid for interval in binned_complaints.index]
        means = binned_complaints['mean']
        stds = binned_complaints['std'].fillna(0)
        
        ax2.errorbar(bin_centers, means, yerr=stds, marker='o', linewidth=2, 
                    markersize=6, capsize=5)
        ax2.set_xlabel(f'{best_factor.replace("Factor_", "Factor ")} Concentration')
        ax2.set_ylabel('Mean Complaints per Period')
        ax2.set_title(f'Dose-Response Relationship ({best_factor.replace("Factor_", "Factor ")})')
        ax2.grid(True, alpha=0.3)
        
        # Correlation heatmap (all factors + complaints)
        ax3 = axes[1, 0]
        correlation_matrix = aligned_data[factor_cols + ['complaint_count']].corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix), k=1)
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, ax=ax3, fmt='.3f')
        ax3.set_title('Factor-Complaint Correlation Matrix')
        
        # Time series with complaint overlays
        ax4 = axes[1, 1]
        
        # Plot top 2 factors and complaints
        datetime_index = aligned_data.index
        
        # Primary y-axis: factors
        color1, color2 = 'blue', 'green'
        ax4.plot(datetime_index, aligned_data[factor_cols[0]], color=color1, alpha=0.7, 
                label=factor_cols[0].replace('Factor_', 'F'))
        if len(factor_cols) > 1:
            ax4.plot(datetime_index, aligned_data[factor_cols[1]], color=color2, alpha=0.7,
                    label=factor_cols[1].replace('Factor_', 'F'))
        
        ax4.set_xlabel('Time')
        ax4.set_ylabel('Factor Contribution', color='black')
        ax4.tick_params(axis='y', labelcolor='black')
        
        # Secondary y-axis: complaints
        ax4_twin = ax4.twinx()
        complaint_color = 'red'
        
        # Plot complaint events as markers
        complaint_events = complaint_data[complaint_data >= 1]
        if len(complaint_events) > 0:
            ax4_twin.scatter(complaint_events.index, complaint_events, 
                           color=complaint_color, alpha=0.8, s=50, marker='^',
                           label='Complaints', zorder=5)
        
        ax4_twin.set_ylabel('Complaint Count', color=complaint_color)
        ax4_twin.tick_params(axis='y', labelcolor=complaint_color)
        
        ax4.set_title('Factor Contributions with Complaint Events')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = self.plots_dir / f"{self.file_prefix}_advanced_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_file)
        
        return plot_files
    
    def create_interactive_plots(self):
        """Create interactive Plotly plots if available."""
        if not PLOTLY_AVAILABLE:
            print("   ⚠️ Plotly not available for interactive plots")
            return []
        
        print("   🌐 Creating interactive Plotly plots...")
        plot_files = []
        
        if not hasattr(self.complaint_analyzer, 'aligned_data'):
            return plot_files
        
        aligned_data = self.complaint_analyzer.aligned_data
        factor_cols = [col for col in aligned_data.columns if col.startswith('Factor_')]
        
        # Interactive time series plot
        fig = go.Figure()
        
        # Add factor time series
        for factor in factor_cols[:3]:  # Limit to first 3 factors
            fig.add_trace(go.Scatter(
                x=aligned_data.index,
                y=aligned_data[factor],
                mode='lines',
                name=factor.replace('Factor_', 'Factor '),
                opacity=0.7
            ))
        
        # Add complaint events as markers
        complaint_data = aligned_data['complaint_count']
        complaint_events = complaint_data[complaint_data >= 1]
        
        if len(complaint_events) > 0:
            fig.add_trace(go.Scatter(
                x=complaint_events.index,
                y=complaint_events,
                mode='markers',
                marker=dict(color='red', size=10, symbol='triangle-up'),
                name='Complaints',
                yaxis='y2'
            ))
        
        # Update layout
        fig.update_layout(
            title=f'{self.pmf_analyzer.station} Interactive Factor-Complaint Analysis',
            xaxis_title='Time',
            yaxis_title='Factor Contribution',
            yaxis2=dict(
                title='Complaint Count',
                overlaying='y',
                side='right',
                color='red'
            ),
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Save interactive plot
        interactive_file = self.plots_dir / f"{self.file_prefix}_interactive.html"
        fig.write_html(interactive_file)
        plot_files.append(interactive_file)
        
        print(f"   ✅ Saved interactive plot: {interactive_file}")
        
        return plot_files
    
    def create_dashboard_integration_plots(self):
        """Create plots specifically designed for dashboard integration."""
        print("   📊 Creating dashboard integration plots...")
        
        plot_files = []
        
        # Create summary dashboard plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle(f'{self.pmf_analyzer.station} Complaint Validation Dashboard Summary', 
                    fontsize=20, fontweight='bold')
        
        if hasattr(self.complaint_analyzer, 'aligned_data'):
            complaint_data = self.complaint_analyzer.aligned_data['complaint_count']
            
            # Quick stats panel
            total_complaints = complaint_data.sum()
            mean_daily = complaint_data.mean()
            max_daily = complaint_data.max()
            complaint_days = (complaint_data >= 1).sum()
            total_days = len(complaint_data)
            
            # Summary statistics plot
            ax1.axis('tight')
            ax1.axis('off')
            
            summary_stats = [
                ['Total Complaints', f'{total_complaints:.0f}'],
                ['Mean Daily Complaints', f'{mean_daily:.2f}'],
                ['Max Daily Complaints', f'{max_daily:.0f}'],
                ['Days with Complaints', f'{complaint_days} / {total_days}'],
                ['Complaint Frequency', f'{(complaint_days/total_days)*100:.1f}%']
            ]
            
            table = ax1.table(cellText=summary_stats, colLabels=['Metric', 'Value'],
                            cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(14)
            table.scale(1.5, 2)
            ax1.set_title('Complaint Statistics Summary', fontsize=16, pad=20)
        
        # Best correlations summary
        if hasattr(self.complaint_analyzer, 'correlation_results'):
            ax2.axis('tight')
            ax2.axis('off')
            
            # Find best correlations
            best_correlations = []
            for factor, results in self.complaint_analyzer.correlation_results.items():
                pearson_r = results['zero_lag']['pearson']['r']
                pearson_p = results['zero_lag']['pearson']['p']
                significant = '***' if pearson_p < 0.05 else ''
                
                best_correlations.append([
                    factor.replace('Factor_', 'F'),
                    f'{pearson_r:.3f}',
                    f'{pearson_p:.3f}',
                    significant
                ])
            
            # Sort by absolute correlation
            best_correlations.sort(key=lambda x: abs(float(x[1])), reverse=True)
            
            headers = ['Factor', 'Correlation', 'p-value', 'Sig.']
            table2 = ax2.table(cellText=best_correlations, colLabels=headers,
                             cellLoc='center', loc='center')
            table2.auto_set_font_size(False)
            table2.set_fontsize(12)
            table2.scale(1.2, 1.8)
            ax2.set_title('Factor-Complaint Correlations', fontsize=16, pad=20)
        
        # Model performance summary
        if hasattr(self.complaint_analyzer, 'prediction_models'):
            models = self.complaint_analyzer.prediction_models
            
            model_names = [name.replace('_', ' ').title() for name in models.keys()]
            r2_scores = [data['mean_r2'] for data in models.values()]
            
            bars = ax3.bar(model_names, r2_scores, alpha=0.8, 
                          color=['skyblue', 'lightcoral', 'lightgreen'][:len(model_names)])
            ax3.set_ylabel('R² Score', fontsize=12)
            ax3.set_title('Prediction Model Performance', fontsize=16)
            ax3.tick_params(axis='x', rotation=45, labelsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, r2 in zip(bars, r2_scores):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{r2:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Validation metrics summary
        if 'classification' in self.complaint_analyzer.validation_metrics:
            class_data = self.complaint_analyzer.validation_metrics['classification']
            
            ax4.axis('tight')
            ax4.axis('off')
            
            validation_metrics = [
                ['ROC AUC', f'{class_data["roc_auc"]:.3f}'],
                ['Accuracy', f'{class_data["classification_report"]["accuracy"]:.3f}'],
                ['Precision', f'{class_data["classification_report"].get("True", {}).get("precision", 0):.3f}'],
                ['Recall', f'{class_data["classification_report"].get("True", {}).get("recall", 0):.3f}']
            ]
            
            table3 = ax4.table(cellText=validation_metrics, colLabels=['Metric', 'Score'],
                             cellLoc='center', loc='center')
            table3.auto_set_font_size(False)
            table3.set_fontsize(14)
            table3.scale(1.5, 2)
            ax4.set_title('Classification Performance', fontsize=16, pad=20)
        
        plt.tight_layout()
        plot_file = self.plots_dir / f"{self.file_prefix}_dashboard_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        plot_files.append(plot_file)
        
        return plot_files