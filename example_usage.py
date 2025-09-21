#!/usr/bin/env python3
"""
PMF-Complaint Integration Example Usage
=======================================

This script demonstrates how to use the integrated PMF-complaint analysis system
with sample data and provides a complete walkthrough of the workflow.

"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

def create_sample_complaint_data():
    """
    Create realistic sample complaint data for demonstration.
    
    This simulates malodour complaints with realistic patterns:
    - Higher complaints during certain weather conditions
    - Weekend vs weekday variations
    - Seasonal patterns
    - Random variation around baseline
    """
    print("📋 Creating sample complaint data...")
    
    # Date range: 6 months for realistic analysis
    start_date = pd.to_datetime('2024-03-01')
    end_date = pd.to_datetime('2024-08-31')
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    # Generate realistic complaint patterns
    complaints = []
    
    for date in date_range:
        # Base complaint rate (low)
        base_rate = 0.1
        
        # Weekly patterns (higher on weekdays when more people are around)
        if date.weekday() < 5:  # Monday-Friday
            weekly_multiplier = 1.2
        else:  # Weekend
            weekly_multiplier = 0.8
        
        # Seasonal patterns (higher in summer due to open windows)
        month = date.month
        if month in [6, 7, 8]:  # Summer months
            seasonal_multiplier = 1.5
        elif month in [3, 4, 5]:  # Spring
            seasonal_multiplier = 1.2
        else:
            seasonal_multiplier = 1.0
        
        # Weather-related patterns (higher during temperature inversions)
        # Simulate some high complaint days
        if np.random.random() < 0.05:  # 5% chance of high complaint day
            weather_multiplier = 3.0
        elif np.random.random() < 0.15:  # 15% chance of moderate day
            weather_multiplier = 2.0
        else:
            weather_multiplier = 1.0
        
        # Calculate expected complaints
        expected_complaints = base_rate * weekly_multiplier * seasonal_multiplier * weather_multiplier
        
        # Generate actual complaints (Poisson distribution)
        actual_complaints = np.random.poisson(expected_complaints)
        
        # Add some zero-inflation (many days with no complaints)
        if np.random.random() < 0.7:  # 70% chance of zero complaints
            actual_complaints = 0
        
        complaints.append(actual_complaints)
    
    # Create DataFrame
    complaint_df = pd.DataFrame({
        'date': date_range,
        'complaint_count': complaints,
        'severity_avg': np.random.uniform(1, 5, len(complaints)),  # 1-5 scale
        'weather_conditions': np.random.choice(['clear', 'overcast', 'windy', 'calm'], len(complaints)),
        'time_of_day_peak': np.random.choice(['morning', 'afternoon', 'evening', 'night'], len(complaints))
    })
    
    # Save sample data
    sample_file = Path('sample_complaint_data.csv')
    complaint_df.to_csv(sample_file, index=False)
    
    print(f"✅ Created sample complaint data: {sample_file}")
    print(f"   Total complaints: {complaint_df['complaint_count'].sum()}")
    print(f"   Days with complaints: {(complaint_df['complaint_count'] > 0).sum()}")
    print(f"   Date range: {complaint_df['date'].min()} to {complaint_df['date'].max()}")
    
    return sample_file

def run_example_analysis():
    """
    Run complete example analysis workflow.
    """
    print("🚀 STARTING PMF-COMPLAINT INTEGRATION EXAMPLE")
    print("=" * 60)
    
    # Step 1: Create sample complaint data
    complaint_file = create_sample_complaint_data()
    
    # Step 2: Set analysis parameters
    station = 'MMF1'  # Example station
    start_date = '2024-03-01'
    end_date = '2024-08-31'
    output_dir = 'example_results'
    
    print(f"\n📊 Analysis Configuration:")
    print(f"   Station: {station}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Complaint data: {complaint_file}")
    print(f"   Output directory: {output_dir}")
    
    # Step 3: Import and run integrated analysis
    try:
        from pmf_complaint_integration import PMFComplaintIntegration
        
        print(f"\n🔬 Initializing integrated analysis...")
        analysis = PMFComplaintIntegration(
            station=station,
            start_date=start_date,
            end_date=end_date,
            complaint_file=str(complaint_file),
            output_dir=output_dir
        )
        
        print(f"\n⚡ Running complete analysis workflow...")
        success = analysis.run_complete_analysis()
        
        if success:
            print(f"\n🎉 EXAMPLE ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"\nResults available in:")
            print(f"📂 Main directory: {output_dir}/")
            print(f"🌐 HTML Dashboard: {output_dir}/{station}_complaint_validation_dashboard.html")
            print(f"📊 Plots directory: {output_dir}/complaint_plots/")
            print(f"📄 Final report: {output_dir}/{station}_final_analysis_report.md")
            
            # Display key results
            if hasattr(analysis.complaint_analyzer, 'correlation_results'):
                print(f"\n📈 Key Correlation Results:")
                for factor, results in analysis.complaint_analyzer.correlation_results.items():
                    r = results['zero_lag']['pearson']['r']
                    p = results['zero_lag']['pearson']['p']
                    sig = "***" if p < 0.05 else "**" if p < 0.10 else ""
                    print(f"   {factor}: r = {r:.3f}, p = {p:.3f} {sig}")
            
            return True
            
        else:
            print(f"\n💥 EXAMPLE ANALYSIS FAILED!")
            return False
            
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print(f"   Make sure all required modules are available")
        return False
    except Exception as e:
        print(f"\n❌ Analysis error: {e}")
        return False

def demonstrate_individual_components():
    """
    Demonstrate individual components of the analysis system.
    """
    print("\n🔧 DEMONSTRATING INDIVIDUAL COMPONENTS")
    print("=" * 50)
    
    # Create sample data
    complaint_file = create_sample_complaint_data()
    
    try:
        # Demonstrate complaint analysis methods
        print("\n📊 1. Complaint Analysis Methods Demo:")
        from complaint_analysis_methods import ComplaintPMFAnalyzer
        
        # This would normally be initialized with a PMF analyzer
        print("   ✅ ComplaintPMFAnalyzer class available")
        print("   Features: correlation analysis, predictive modeling, validation metrics")
        
        # Demonstrate visualization suite
        print("\n🎨 2. Visualization Suite Demo:")
        from complaint_visualization_suite import ComplaintVisualizationSuite
        print("   ✅ ComplaintVisualizationSuite class available")
        print("   Features: 12+ specialized plots, interactive charts, dashboard integration")
        
        # Demonstrate integration script
        print("\n🔗 3. Integration Script Demo:")
        from pmf_complaint_integration import PMFComplaintIntegration
        print("   ✅ PMFComplaintIntegration class available")
        print("   Features: end-to-end workflow, automated reporting, quality assurance")
        
        print(f"\n✅ All components successfully imported and ready for use!")
        
    except ImportError as e:
        print(f"❌ Component import error: {e}")
        return False
    
    return True

def create_usage_documentation():
    """
    Create comprehensive usage documentation.
    """
    doc_file = Path('PMF_COMPLAINT_INTEGRATION_GUIDE.md')
    
    documentation = """# PMF-Complaint Integration Analysis Guide

## Overview

This system integrates malodour complaint data with PMF (Positive Matrix Factorization) source apportionment analysis to provide ground truth validation of environmental source identification.

## Quick Start

### 1. Prepare Your Complaint Data

Create a CSV file with the following columns:
```csv
date,complaint_count,severity_avg,weather_conditions,time_of_day_peak
2024-01-01,0,0,clear,morning
2024-01-02,2,3.5,overcast,afternoon
2024-01-03,1,2.0,windy,evening
```

Required columns:
- `date`: Date in YYYY-MM-DD format
- `complaint_count`: Number of complaints per day (integer)

Optional columns (enhance analysis):
- `severity_avg`: Average severity (1-5 scale)
- `weather_conditions`: Weather description
- `time_of_day_peak`: Peak complaint time

### 2. Run Complete Analysis

```bash
python pmf_complaint_integration.py \\
    --station MMF1 \\
    --start-date 2024-01-01 \\
    --end-date 2024-12-31 \\
    --complaint-file complaints.csv \\
    --output-dir results
```

### 3. View Results

- **HTML Dashboard**: `results/MMF1_complaint_validation_dashboard.html`
- **Analysis Report**: `results/MMF1_final_analysis_report.md`
- **Plots Directory**: `results/complaint_plots/`

## Analysis Components

### 1. PMF Source Apportionment
- Identifies emission sources from environmental monitoring data
- Follows EPA PMF 5.0 best practices
- Generates factor profiles and contribution time series

### 2. Complaint Data Integration
- Temporally aligns complaint data with environmental measurements
- Handles missing data and different temporal resolutions
- Validates data quality and completeness

### 3. Statistical Correlation Analysis
- Pearson, Spearman, and Kendall correlations
- Time-lagged correlation analysis for transport delays
- Statistical significance testing (p-values, confidence intervals)

### 4. Predictive Modeling
- Linear regression, Poisson regression, Random Forest
- Cross-validation for model robustness
- Feature importance analysis

### 5. Validation Metrics
- ROC curves and AUC scores
- Confusion matrices
- Classification performance metrics

### 6. Comprehensive Visualization
- 12+ specialized plots for complaint analysis
- Interactive Plotly charts
- Dashboard integration plots

## Output Files

### Analysis Data
- `*_complaint_correlations.csv`: Factor-complaint correlation results
- `*_prediction_models.csv`: Model performance and feature importance
- `*_concentrations.csv`: PMF input concentration data
- `*_uncertainties.csv`: PMF uncertainty estimates

### Visualizations
- `*_complaint_frequency_distribution.png`: Complaint patterns over time
- `*_complaint_correlation_matrix.png`: Factor-complaint correlations
- `*_complaint_temporal_patterns.png`: Diurnal and seasonal patterns
- `*_complaint_environmental_conditions.png`: Weather during complaints
- `*_complaint_validation_metrics.png`: ROC curves and performance
- `*_complaint_model_performance.png`: Prediction model comparison
- `*_complaint_advanced_analysis.png`: Multivariate relationships
- `*_complaint_dashboard_summary.png`: Executive summary

### Reports
- `*_complaint_validation_dashboard.html`: Interactive dashboard
- `*_final_analysis_report.md`: Comprehensive markdown report
- `*_complaint_analysis_report.md`: Detailed analysis report

## Interpretation Guidelines

### Correlation Strength
- **|r| > 0.3**: Strong correlation, good validation
- **|r| > 0.1**: Moderate correlation, some validation
- **|r| < 0.1**: Weak correlation, limited validation

### Model Performance
- **R² > 0.5**: Good predictive power
- **R² > 0.3**: Moderate predictive power
- **R² < 0.3**: Limited predictive power

### Classification Performance
- **AUC > 0.8**: Excellent discrimination
- **AUC > 0.7**: Good discrimination  
- **AUC > 0.6**: Fair discrimination
- **AUC < 0.6**: Poor discrimination

## Regulatory Applications

### Environmental Impact Assessment
- Quantified validation of emission source identification
- Evidence-based metrics for regulatory reporting
- Statistical significance testing for robust conclusions

### Community Engagement
- Visual proof of scientific analysis rigor
- Ground truth validation with community experience
- Transparent methodology and quality assurance

### Risk Management
- Early warning system for complaint events
- Source control prioritization based on impact
- Predictive capability for operational planning

## Quality Assurance

### Data Quality
- Temporal alignment validation
- Missing data handling with EPA methods
- Uncertainty quantification throughout

### Statistical Rigor
- Cross-validation for model robustness
- Multiple correlation methods for consistency
- Significance testing with multiple comparisons correction

### Methodological Standards
- EPA PMF 5.0 guidelines followed
- Peer-reviewed statistical methods
- Comprehensive uncertainty analysis

## Troubleshooting

### Common Issues

1. **No PMF results available**
   - Check that environmental data exists for the date range
   - Verify parquet files are properly formatted
   - Ensure sufficient data completeness (>50% valid)

2. **Low correlation with complaints**
   - Check temporal alignment between data sources
   - Consider time lags for pollutant transport
   - Verify complaint data quality and completeness

3. **Poor model performance**
   - Increase analysis time period for more data
   - Check for seasonal patterns in relationships
   - Consider additional environmental variables

### Support

For technical support or questions:
- Review the analysis logs for detailed error messages
- Check data format requirements
- Verify all dependencies are properly installed

## Advanced Usage

### Custom Analysis Parameters

Modify analysis parameters in the script:

```python
# Correlation analysis settings
complaint_analyzer.significance_level = 0.05
complaint_analyzer.max_lag_hours = 48

# Classification threshold
complaint_analyzer.min_complaint_threshold = 1

# Model selection
model_types = ['linear', 'poisson', 'random_forest']
```

### Integration with Existing Workflows

The system can be integrated with existing PMF analyses:

```python
from pmf_complaint_integration import PMFComplaintIntegration

# Use with existing PMF analyzer
integration = PMFComplaintIntegration(
    station='MMF1',
    start_date='2024-01-01', 
    end_date='2024-12-31',
    complaint_file='complaints.csv'
)

# Run only specific components
integration._integrate_complaint_data()
integration._run_correlation_analysis()
integration._create_visualizations()
```

## Citation

When using this system in publications:

> PMF source apportionment analysis was validated using ground truth malodour complaint data. 
> Statistical correlations between PMF factors and complaint events were assessed using multiple 
> correlation methods with significance testing. Predictive models were developed and validated 
> using cross-validation techniques following EPA PMF 5.0 best practices.

## Version History

- **v1.0**: Initial release with basic correlation analysis
- **v2.0**: Added predictive modeling and validation metrics  
- **v3.0**: Comprehensive visualization suite and dashboard integration
- **v4.0**: Interactive plots and advanced analysis capabilities

---

*Generated by PMF-Complaint Integration Analysis System*  
*University of Leicester | Environmental Health Sciences*
"""
    
    with open(doc_file, 'w', encoding='utf-8') as f:
        f.write(documentation)
    
    print(f"📚 Created comprehensive documentation: {doc_file}")
    return doc_file

def main():
    """Main demonstration function."""
    print("🎯 PMF-COMPLAINT INTEGRATION DEMONSTRATION")
    print("=" * 60)
    print("This script demonstrates the complete integrated analysis system")
    print("for validating PMF source apportionment with complaint data.\n")
    
    # Demonstrate individual components
    if not demonstrate_individual_components():
        print("⚠️ Some components may not be available")
    
    # Create documentation
    create_usage_documentation()
    
    # Create sample data for testing
    complaint_file = create_sample_complaint_data()
    
    print(f"\n📋 NEXT STEPS:")
    print(f"1. Review the created documentation: PMF_COMPLAINT_INTEGRATION_GUIDE.md")
    print(f"2. Use sample data for testing: {complaint_file}")
    print(f"3. Run analysis with your own complaint data:")
    print(f"   python pmf_complaint_integration.py --station MMF1 --start-date 2024-03-01 --end-date 2024-08-31 --complaint-file {complaint_file}")
    print(f"\n🎉 DEMONSTRATION COMPLETE!")

if __name__ == "__main__":
    main()