#!/usr/bin/env python3
"""
Complaint-PMF Correlation Analysis Methods
==========================================

This module provides comprehensive analytical methods for correlating malodour
complaint data with PMF factors and environmental conditions. It serves as the
core analysis engine for validating source apportionment results against 
ground truth complaint events.

Features:
- Statistical correlation analysis (Pearson, Spearman, Kendall)
- Time-lagged correlation analysis for transport delays
- Predictive modeling of complaints from environmental factors
- Environmental condition analysis during complaint events
- Validation metrics and significance testing
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ComplaintPMFAnalyzer:
    """
    Comprehensive analysis class for correlating complaint data with PMF factors
    and environmental conditions.
    """
    
    def __init__(self, pmf_analyzer, complaint_data=None):
        """
        Initialize complaint analysis with existing PMF analyzer.
        
        Args:
            pmf_analyzer: Existing MMFPMFAnalyzer instance with completed PMF analysis
            complaint_data: DataFrame with complaint data (optional, can be loaded later)
        """
        self.pmf_analyzer = pmf_analyzer
        self.complaint_data = complaint_data
        self.correlation_results = {}
        self.prediction_models = {}
        self.validation_metrics = {}
        
        # Analysis configuration
        self.significance_level = 0.05
        self.max_lag_hours = 48  # Maximum lag for correlation analysis
        self.min_complaint_threshold = 1  # Minimum complaints for "high complaint" classification
        
    def load_complaint_data(self, complaint_file_path, date_column='date', 
                          complaint_column='complaint_count'):
        """
        Load complaint data from file and align with PMF analysis timeframe.
        
        Args:
            complaint_file_path: Path to complaint data file (CSV/Excel)
            date_column: Name of date column
            complaint_column: Name of complaint count column
        """
        print(f"📊 Loading complaint data from {complaint_file_path}...")
        
        # Load complaint data
        if str(complaint_file_path).endswith('.xlsx'):
            self.complaint_data = pd.read_excel(complaint_file_path)
        else:
            self.complaint_data = pd.read_csv(complaint_file_path)
        
        # Parse dates and set as index
        self.complaint_data[date_column] = pd.to_datetime(self.complaint_data[date_column])
        self.complaint_data = self.complaint_data.set_index(date_column)
        
        # Align with PMF analysis timeframe
        pmf_start = pd.to_datetime(self.pmf_analyzer.start_date) if self.pmf_analyzer.start_date else None
        pmf_end = pd.to_datetime(self.pmf_analyzer.end_date) if self.pmf_analyzer.end_date else None
        
        if pmf_start:
            self.complaint_data = self.complaint_data[self.complaint_data.index >= pmf_start]
        if pmf_end:
            self.complaint_data = self.complaint_data[self.complaint_data.index <= pmf_end]
        
        print(f"✅ Loaded {len(self.complaint_data)} complaint records")
        print(f"   Date range: {self.complaint_data.index.min()} to {self.complaint_data.index.max()}")
        print(f"   Total complaints: {self.complaint_data[complaint_column].sum()}")
        
        return True
    
    def align_temporal_data(self):
        """
        Temporally align complaint data with PMF factor time series.
        Handles different temporal resolutions and missing data.
        """
        print("🔄 Aligning complaint data with PMF time series...")
        
        # Load PMF time series data
        conc_file = self.pmf_analyzer.output_dir / f"{self.pmf_analyzer.filename_prefix}_concentrations.csv"
        pmf_data = pd.read_csv(conc_file, index_col=0, parse_dates=True)
        
        # Get PMF factor contributions
        if hasattr(self.pmf_analyzer, 'best_model') and self.pmf_analyzer.best_model:
            factor_contributions = self.pmf_analyzer.best_model.W  # Time x Factors
            factor_df = pd.DataFrame(
                factor_contributions,
                index=pmf_data.index,
                columns=[f'Factor_{i+1}' for i in range(factor_contributions.shape[1])]
            )
        else:
            print("❌ No PMF results available - run PMF analysis first")
            return False
        
        # Resample complaint data to match PMF temporal resolution
        pmf_freq = pd.infer_freq(pmf_data.index[:100])  # Infer from first 100 points
        print(f"   PMF frequency: {pmf_freq}")
        
        # Resample complaints to match PMF frequency
        if pmf_freq:
            # Forward fill complaint data to match PMF resolution
            complaint_resampled = self.complaint_data.resample(pmf_freq).first().fillna(0)
        else:
            # Fallback: merge on nearest timestamp
            complaint_resampled = self.complaint_data.reindex(pmf_data.index, method='nearest').fillna(0)
        
        # Combine aligned data
        self.aligned_data = pd.concat([
            pmf_data,
            factor_df,
            complaint_resampled
        ], axis=1).dropna()
        
        print(f"✅ Aligned dataset: {len(self.aligned_data)} time points")
        print(f"   Columns: {list(self.aligned_data.columns)}")
        
        return True
    
    def correlate_factors_with_complaints(self, lag_hours=None):
        """
        Calculate correlations between PMF factors and complaint events.
        Includes time-lagged correlations to account for transport delays.
        
        Args:
            lag_hours: List of lag hours to test, or None for automatic selection
        """
        print("📈 Analyzing factor-complaint correlations...")
        
        if not hasattr(self, 'aligned_data'):
            print("❌ No aligned data - run align_temporal_data() first")
            return False
        
        # Get factor and complaint columns
        factor_cols = [col for col in self.aligned_data.columns if col.startswith('Factor_')]
        complaint_col = 'complaint_count'
        
        if complaint_col not in self.aligned_data.columns:
            print("❌ No complaint data found in aligned dataset")
            return False
        
        # Set up lag analysis
        if lag_hours is None:
            lag_hours = list(range(0, self.max_lag_hours + 1, 2))  # Every 2 hours
        
        correlation_results = {}
        
        for factor in factor_cols:
            print(f"   Analyzing {factor}...")
            factor_results = {
                'zero_lag': {},
                'lagged_correlations': {},
                'best_lag': None,
                'best_correlation': None
            }
            
            factor_data = self.aligned_data[factor]
            complaint_data = self.aligned_data[complaint_col]
            
            # Zero-lag correlations
            pearson_r, pearson_p = pearsonr(factor_data, complaint_data)
            spearman_r, spearman_p = spearmanr(factor_data, complaint_data)
            kendall_r, kendall_p = kendalltau(factor_data, complaint_data)
            
            factor_results['zero_lag'] = {
                'pearson': {'r': pearson_r, 'p': pearson_p, 'significant': pearson_p < self.significance_level},
                'spearman': {'r': spearman_r, 'p': spearman_p, 'significant': spearman_p < self.significance_level},
                'kendall': {'r': kendall_r, 'p': kendall_p, 'significant': kendall_p < self.significance_level}
            }
            
            # Time-lagged correlations
            best_r = abs(pearson_r)
            best_lag = 0
            
            for lag in lag_hours:
                if lag == 0:
                    continue
                
                # Shift complaint data by lag hours (positive lag = complaints follow factors)
                try:
                    if len(factor_data) > lag:
                        lagged_complaints = complaint_data.shift(-lag)  # Negative shift for future complaints
                        valid_data = ~(factor_data.isna() | lagged_complaints.isna())
                        
                        if valid_data.sum() > 10:  # Minimum data points for correlation
                            r_lag, p_lag = pearsonr(factor_data[valid_data], lagged_complaints[valid_data])
                            
                            factor_results['lagged_correlations'][lag] = {
                                'r': r_lag, 'p': p_lag, 'n_points': valid_data.sum(),
                                'significant': p_lag < self.significance_level
                            }
                            
                            # Track best correlation
                            if abs(r_lag) > best_r:
                                best_r = abs(r_lag)
                                best_lag = lag
                                factor_results['best_correlation'] = r_lag
                                factor_results['best_lag'] = lag
                except Exception as e:
                    print(f"     Warning: Could not calculate lag {lag}h: {e}")
            
            correlation_results[factor] = factor_results
        
        self.correlation_results = correlation_results
        
        # Display summary
        self._display_correlation_summary()
        
        return True
    
    def _display_correlation_summary(self):
        """Display summary of correlation analysis results."""
        print("\n📊 CORRELATION ANALYSIS SUMMARY")
        print("=" * 60)
        
        for factor, results in self.correlation_results.items():
            print(f"\n{factor}:")
            
            # Zero-lag results
            zero_lag = results['zero_lag']
            print(f"  Zero-lag correlations:")
            for method, stats_data in zero_lag.items():
                sig_marker = "***" if stats_data['significant'] else ""
                print(f"    {method.capitalize()}: r={stats_data['r']:.3f}, p={stats_data['p']:.3f} {sig_marker}")
            
            # Best lagged correlation
            if results['best_lag'] is not None:
                print(f"  Best lagged correlation: r={results['best_correlation']:.3f} at {results['best_lag']}h lag")
            else:
                print("  No significant lagged correlations found")
    
    def complaint_factor_regression(self, model_types=['linear', 'poisson', 'random_forest']):
        """
        Build predictive models for complaints based on PMF factors.
        
        Args:
            model_types: List of model types to try ['linear', 'poisson', 'random_forest']
        """
        print("🤖 Building complaint prediction models...")
        
        if not hasattr(self, 'aligned_data'):
            print("❌ No aligned data available")
            return False
        
        # Prepare data
        factor_cols = [col for col in self.aligned_data.columns if col.startswith('Factor_')]
        X = self.aligned_data[factor_cols]
        y = self.aligned_data['complaint_count']
        
        # Remove any remaining NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        print(f"   Training data: {len(X_clean)} samples, {len(factor_cols)} features")
        print(f"   Complaint statistics: mean={y_clean.mean():.2f}, max={y_clean.max()}")
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {}
        
        # Linear regression
        if 'linear' in model_types:
            print("   Training linear regression...")
            linear_model = LinearRegression()
            linear_scores = cross_val_score(linear_model, X_clean, y_clean, cv=tscv, scoring='r2')
            linear_model.fit(X_clean, y_clean)
            
            models['linear'] = {
                'model': linear_model,
                'scores': linear_scores,
                'mean_r2': linear_scores.mean(),
                'std_r2': linear_scores.std(),
                'feature_importance': dict(zip(factor_cols, linear_model.coef_))
            }
            print(f"     Linear R² = {linear_scores.mean():.3f} ± {linear_scores.std():.3f}")
        
        # Poisson regression (appropriate for count data)
        if 'poisson' in model_types:
            print("   Training Poisson regression...")
            # Ensure non-negative predictions
            X_scaled = StandardScaler().fit_transform(X_clean)
            poisson_model = PoissonRegressor(alpha=1e-6, max_iter=1000)
            
            try:
                poisson_scores = cross_val_score(poisson_model, X_scaled, y_clean, cv=tscv, scoring='r2')
                poisson_model.fit(X_scaled, y_clean)
                
                models['poisson'] = {
                    'model': poisson_model,
                    'scaler': StandardScaler().fit(X_clean),
                    'scores': poisson_scores,
                    'mean_r2': poisson_scores.mean(),
                    'std_r2': poisson_scores.std(),
                    'feature_importance': dict(zip(factor_cols, poisson_model.coef_))
                }
                print(f"     Poisson R² = {poisson_scores.mean():.3f} ± {poisson_scores.std():.3f}")
            except Exception as e:
                print(f"     Poisson regression failed: {e}")
        
        # Random Forest
        if 'random_forest' in model_types:
            print("   Training Random Forest...")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            rf_scores = cross_val_score(rf_model, X_clean, y_clean, cv=tscv, scoring='r2')
            rf_model.fit(X_clean, y_clean)
            
            models['random_forest'] = {
                'model': rf_model,
                'scores': rf_scores,
                'mean_r2': rf_scores.mean(),
                'std_r2': rf_scores.std(),
                'feature_importance': dict(zip(factor_cols, rf_model.feature_importances_))
            }
            print(f"     Random Forest R² = {rf_scores.mean():.3f} ± {rf_scores.std():.3f}")
        
        self.prediction_models = models
        
        # Find best model
        best_model_name = max(models.keys(), key=lambda k: models[k]['mean_r2'])
        print(f"\n🏆 Best model: {best_model_name} (R² = {models[best_model_name]['mean_r2']:.3f})")
        
        return True
    
    def complaint_classification_analysis(self):
        """
        Classify high vs low complaint days and analyze predictive performance.
        """
        print("🎯 Analyzing complaint classification performance...")
        
        if not hasattr(self, 'aligned_data'):
            print("❌ No aligned data available")
            return False
        
        # Create binary classification target
        complaint_data = self.aligned_data['complaint_count']
        high_complaint_days = complaint_data >= self.min_complaint_threshold
        
        print(f"   Classification threshold: {self.min_complaint_threshold} complaints")
        print(f"   High complaint days: {high_complaint_days.sum()} / {len(high_complaint_days)} ({high_complaint_days.mean()*100:.1f}%)")
        
        # Prepare features
        factor_cols = [col for col in self.aligned_data.columns if col.startswith('Factor_')]
        X = self.aligned_data[factor_cols]
        y = high_complaint_days
        
        # Remove NaN values
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        
        # Train Random Forest classifier
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_classifier.fit(X_clean, y_clean)
        
        # Predictions and probabilities
        y_pred = rf_classifier.predict(X_clean)
        y_prob = rf_classifier.predict_proba(X_clean)[:, 1]
        
        # Classification metrics
        classification_report_dict = classification_report(y_clean, y_pred, output_dict=True)
        confusion_mat = confusion_matrix(y_clean, y_pred)
        
        # ROC curve
        fpr, tpr, thresholds = roc_curve(y_clean, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Store results
        self.validation_metrics['classification'] = {
            'model': rf_classifier,
            'feature_importance': dict(zip(factor_cols, rf_classifier.feature_importances_)),
            'classification_report': classification_report_dict,
            'confusion_matrix': confusion_mat,
            'roc_curve': {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds},
            'roc_auc': roc_auc,
            'predictions': {'y_true': y_clean, 'y_pred': y_pred, 'y_prob': y_prob}
        }
        
        print(f"   ROC AUC: {roc_auc:.3f}")
        print(f"   Accuracy: {classification_report_dict['accuracy']:.3f}")
        print(f"   Precision: {classification_report_dict['True']['precision']:.3f}")
        print(f"   Recall: {classification_report_dict['True']['recall']:.3f}")
        
        return True
    
    def temporal_complaint_analysis(self):
        """
        Analyze temporal patterns in complaint data.
        """
        print("📅 Analyzing temporal complaint patterns...")
        
        if not hasattr(self, 'aligned_data'):
            print("❌ No aligned data available")
            return False
        
        complaint_data = self.aligned_data['complaint_count']
        datetime_index = self.aligned_data.index
        
        # Create temporal features
        temporal_analysis = {
            'hourly': complaint_data.groupby(datetime_index.hour).agg(['mean', 'sum', 'count']),
            'daily': complaint_data.groupby(datetime_index.dayofweek).agg(['mean', 'sum', 'count']),
            'monthly': complaint_data.groupby(datetime_index.month).agg(['mean', 'sum', 'count']),
            'seasonal': complaint_data.groupby(datetime_index.quarter).agg(['mean', 'sum', 'count'])
        }
        
        # Weekend vs weekday analysis
        is_weekend = datetime_index.dayofweek >= 5
        temporal_analysis['weekend_vs_weekday'] = {
            'weekday': complaint_data[~is_weekend].agg(['mean', 'sum', 'count']),
            'weekend': complaint_data[is_weekend].agg(['mean', 'sum', 'count'])
        }
        
        # Store results
        self.validation_metrics['temporal_patterns'] = temporal_analysis
        
        # Display summary
        print(f"   Peak complaint hour: {temporal_analysis['hourly']['mean'].idxmax()}")
        print(f"   Peak complaint day: {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][temporal_analysis['daily']['mean'].idxmax()]}")
        print(f"   Peak complaint month: {temporal_analysis['monthly']['mean'].idxmax()}")
        
        return True
    
    def environmental_complaint_analysis(self):
        """
        Analyze environmental conditions during high complaint periods.
        """
        print("🌦️ Analyzing environmental conditions during complaints...")
        
        if not hasattr(self, 'aligned_data'):
            print("❌ No aligned data available")
            return False
        
        # Identify high complaint periods
        complaint_data = self.aligned_data['complaint_count']
        high_complaint_mask = complaint_data >= self.min_complaint_threshold
        
        # Environmental variables to analyze
        env_vars = []
        for col in self.aligned_data.columns:
            if any(env_type in col.lower() for env_type in ['temp', 'pressure', 'wind', 'humid', 'rain']):
                env_vars.append(col)
        
        if not env_vars:
            print("   No environmental variables found in dataset")
            return False
        
        environmental_analysis = {}
        
        for var in env_vars:
            try:
                var_data = self.aligned_data[var].dropna()
                
                if len(var_data) > 10:
                    # High complaint vs low complaint conditions
                    high_complaint_conditions = var_data[high_complaint_mask]
                    low_complaint_conditions = var_data[~high_complaint_mask]
                    
                    # Statistical comparison
                    if len(high_complaint_conditions) > 0 and len(low_complaint_conditions) > 0:
                        # Two-sample t-test
                        t_stat, t_p = stats.ttest_ind(high_complaint_conditions, low_complaint_conditions)
                        
                        # Mann-Whitney U test (non-parametric)
                        u_stat, u_p = stats.mannwhitneyu(
                            high_complaint_conditions, low_complaint_conditions, 
                            alternative='two-sided'
                        )
                        
                        environmental_analysis[var] = {
                            'high_complaint_stats': {
                                'mean': high_complaint_conditions.mean(),
                                'std': high_complaint_conditions.std(),
                                'median': high_complaint_conditions.median(),
                                'count': len(high_complaint_conditions)
                            },
                            'low_complaint_stats': {
                                'mean': low_complaint_conditions.mean(),
                                'std': low_complaint_conditions.std(),
                                'median': low_complaint_conditions.median(),
                                'count': len(low_complaint_conditions)
                            },
                            'statistical_tests': {
                                't_test': {'statistic': t_stat, 'p_value': t_p, 'significant': t_p < self.significance_level},
                                'mann_whitney': {'statistic': u_stat, 'p_value': u_p, 'significant': u_p < self.significance_level}
                            }
                        }
                        
                        print(f"   {var}: High complaint mean = {high_complaint_conditions.mean():.2f}, Low complaint mean = {low_complaint_conditions.mean():.2f}")
                        if t_p < self.significance_level:
                            print(f"     *** Significant difference (p = {t_p:.3f})")
                
            except Exception as e:
                print(f"   Warning: Could not analyze {var}: {e}")
        
        self.validation_metrics['environmental_conditions'] = environmental_analysis
        
        return True
    
    def save_analysis_results(self, output_dir=None):
        """
        Save all analysis results to files.
        """
        if output_dir is None:
            output_dir = self.pmf_analyzer.output_dir
        
        output_dir = Path(output_dir)
        
        print(f"💾 Saving complaint analysis results to {output_dir}...")
        
        # Save correlation results
        if hasattr(self, 'correlation_results'):
            correlation_df = self._correlation_results_to_dataframe()
            correlation_file = output_dir / f"{self.pmf_analyzer.filename_prefix}_complaint_correlations.csv"
            correlation_df.to_csv(correlation_file)
            print(f"   Saved: {correlation_file}")
        
        # Save prediction model results
        if hasattr(self, 'prediction_models'):
            model_results = self._prediction_models_to_dataframe()
            model_file = output_dir / f"{self.pmf_analyzer.filename_prefix}_prediction_models.csv"
            model_results.to_csv(model_file)
            print(f"   Saved: {model_file}")
        
        # Save validation metrics
        if hasattr(self, 'validation_metrics'):
            # Create comprehensive analysis report
            self._create_analysis_report(output_dir)
        
        return True
    
    def _correlation_results_to_dataframe(self):
        """Convert correlation results to DataFrame for saving."""
        rows = []
        for factor, results in self.correlation_results.items():
            # Zero-lag correlations
            for method, stats_data in results['zero_lag'].items():
                rows.append({
                    'factor': factor,
                    'lag_hours': 0,
                    'method': method,
                    'correlation': stats_data['r'],
                    'p_value': stats_data['p'],
                    'significant': stats_data['significant']
                })
            
            # Lagged correlations
            for lag, lag_data in results['lagged_correlations'].items():
                rows.append({
                    'factor': factor,
                    'lag_hours': lag,
                    'method': 'pearson_lagged',
                    'correlation': lag_data['r'],
                    'p_value': lag_data['p'],
                    'significant': lag_data['significant']
                })
        
        return pd.DataFrame(rows)
    
    def _prediction_models_to_dataframe(self):
        """Convert prediction model results to DataFrame."""
        rows = []
        for model_name, model_data in self.prediction_models.items():
            for feature, importance in model_data['feature_importance'].items():
                rows.append({
                    'model': model_name,
                    'feature': feature,
                    'importance': importance,
                    'mean_r2': model_data['mean_r2'],
                    'std_r2': model_data['std_r2']
                })
        
        return pd.DataFrame(rows)
    
    def _create_analysis_report(self, output_dir):
        """Create comprehensive analysis report."""
        report_file = output_dir / f"{self.pmf_analyzer.filename_prefix}_complaint_analysis_report.md"
        
        with open(report_file, 'w') as f:
            f.write(f"# Complaint Analysis Report\n")
            f.write(f"## {self.pmf_analyzer.station} - PMF Source Apportionment Validation\n\n")
            
            f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"**Station**: {self.pmf_analyzer.station}\n")
            f.write(f"**Analysis Period**: {self.pmf_analyzer.start_date} to {self.pmf_analyzer.end_date}\n\n")
            
            # Correlation summary
            if hasattr(self, 'correlation_results'):
                f.write("## Factor-Complaint Correlations\n\n")
                for factor, results in self.correlation_results.items():
                    f.write(f"### {factor}\n")
                    
                    zero_lag = results['zero_lag']
                    for method, stats_data in zero_lag.items():
                        sig = "***" if stats_data['significant'] else ""
                        f.write(f"- **{method.capitalize()}**: r = {stats_data['r']:.3f}, p = {stats_data['p']:.3f} {sig}\n")
                    
                    if results['best_lag'] is not None:
                        f.write(f"- **Best lag**: {results['best_lag']} hours (r = {results['best_correlation']:.3f})\n")
                    f.write("\n")
            
            # Prediction models
            if hasattr(self, 'prediction_models'):
                f.write("## Prediction Model Performance\n\n")
                for model_name, model_data in self.prediction_models.items():
                    f.write(f"### {model_name.replace('_', ' ').title()}\n")
                    f.write(f"- **R² Score**: {model_data['mean_r2']:.3f} ± {model_data['std_r2']:.3f}\n")
                    f.write("- **Feature Importance**:\n")
                    
                    sorted_features = sorted(model_data['feature_importance'].items(), 
                                           key=lambda x: abs(x[1]), reverse=True)
                    for feature, importance in sorted_features[:5]:  # Top 5
                        f.write(f"  - {feature}: {importance:.3f}\n")
                    f.write("\n")
            
            # Classification results
            if 'classification' in self.validation_metrics:
                class_metrics = self.validation_metrics['classification']
                f.write("## Classification Performance\n\n")
                f.write(f"- **ROC AUC**: {class_metrics['roc_auc']:.3f}\n")
                f.write(f"- **Accuracy**: {class_metrics['classification_report']['accuracy']:.3f}\n")
                
                if 'True' in class_metrics['classification_report']:
                    f.write(f"- **Precision**: {class_metrics['classification_report']['True']['precision']:.3f}\n")
                    f.write(f"- **Recall**: {class_metrics['classification_report']['True']['recall']:.3f}\n")
                f.write("\n")
        
        print(f"   Saved: {report_file}")