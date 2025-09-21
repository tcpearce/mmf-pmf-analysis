# Malodour Complaint Data Integration Plan
## PMF Source Apportionment Ground Truth Validation

### Executive Summary
This document outlines the comprehensive integration of malodour complaint data as ground truth validation for the PMF source apportionment analysis system. The integration will transform the analysis from purely statistical to validated environmental impact assessment.

---

## 1. Data Structure Design

### 1.1 Complaint Data Format
```python
# Expected complaint data structure
complaint_data = {
    'date': pd.Timestamp,           # Daily complaint aggregation
    'complaint_count': int,         # Number of complaints per day  
    'severity_avg': float,          # Average severity (1-5 scale)
    'complaint_types': list,        # Categories: odour, dust, noise, etc.
    'time_of_day_dist': dict,       # Hourly distribution of complaints
    'weather_conditions': str,      # Associated weather at time of complaints
    'source_attribution': str       # If available from complaint records
}
```

### 1.2 Integration with Existing Parquet Structure
- **New column**: `complaint_count` added to MMF parquet files
- **Temporal alignment**: Hourly complaint data interpolated to match MMF sampling
- **Missing data handling**: Zero complaints = 0, not NaN (important distinction)

---

## 2. Enhanced Analysis Methods

### 2.1 Complaint-Factor Correlation Analysis
```python
class ComplaintPMFAnalyzer:
    def correlate_factors_with_complaints(self):
        """
        Statistical correlation between PMF factors and complaint events
        - Pearson correlation coefficients
        - Spearman rank correlations (for non-linear relationships)
        - Time-lagged correlations (delayed odour transport)
        - Statistical significance testing (p-values, confidence intervals)
        """
        
    def complaint_factor_regression(self):
        """
        Predictive modeling of complaints based on PMF factors
        - Multiple linear regression
        - Poisson regression for count data
        - Random forest for non-linear relationships
        - Cross-validation for model robustness
        """
        
    def temporal_complaint_analysis(self):
        """
        Time-based complaint pattern analysis
        - Diurnal variation in complaints vs factors
        - Seasonal patterns and correlations
        - Weekend/weekday effects
        - Holiday period analysis
        """
```

### 2.2 Environmental Conditions During Complaints
```python
    def complaint_environmental_analysis(self):
        """
        Environmental conditions analysis during high complaint periods
        - Wind direction/speed during complaints
        - Temperature/pressure correlations
        - Meteorological stability conditions
        - Atmospheric dispersion modeling validation
        """
```

---

## 3. New Visualization Suite (12 Additional Plots)

### 3.1 Primary Complaint Analysis Plots
1. **Complaint Frequency Distribution**
   - Daily/monthly complaint counts
   - Seasonal variation patterns
   - Long-term trends analysis

2. **Factor-Complaint Correlation Matrix**
   - Heatmap of all PMF factors vs complaints
   - Statistical significance indicators
   - Time-lag analysis results

3. **Complaint Time Series with Factor Overlays**
   - Primary axis: complaint counts
   - Secondary axes: top correlated PMF factors
   - Identified high-complaint events highlighted

4. **Environmental Conditions During Complaints**
   - Wind roses for high complaint days
   - Temperature/pressure distributions
   - Atmospheric stability analysis

### 3.2 Validation and Accuracy Plots
5. **PMF Factor Validation Metrics**
   - ROC curves for complaint prediction
   - Precision-recall curves
   - Confusion matrices for complaint/no-complaint classification

6. **Complaint Prediction Model Performance**
   - Actual vs predicted complaints
   - Residual analysis of prediction models
   - Feature importance rankings

### 3.3 Advanced Diagnostic Plots
7. **Complaint-Factor Response Curves**
   - Non-linear relationship visualization
   - Threshold analysis for complaint triggers
   - Dose-response relationships

8. **Spatial Analysis (if location data available)**
   - Complaint density maps
   - Distance-decay relationships from source
   - Wind-corrected exposure zones

### 3.4 Integrated Analysis Plots
9. **Multi-variate Complaint Analysis**
   - 3D plots: Factor 1 vs Factor 2 vs Complaints
   - Principal component analysis including complaints
   - Cluster analysis of complaint events

10. **Complaint-Based Source Attribution**
    - Source contribution during high complaint periods
    - Factor profiles for complaint vs non-complaint days
    - Source-specific complaint correlation analysis

11. **Regulatory Compliance Dashboard**
    - Complaint frequency vs regulatory thresholds
    - Exceedance probability analysis
    - Risk assessment matrices

12. **Temporal Lag Analysis**
    - Cross-correlation functions (factor → complaint delays)
    - Atmospheric transport time analysis
    - Optimal prediction lead times

---

## 4. Enhanced Existing Plots (17+ Plots Modified)

### 4.1 All Existing Plots Enhanced With:
- **Complaint event markers**: Red dots/lines on time series
- **High complaint period shading**: Background coloring for complaint days
- **Dual-axis displays**: Primary data + complaint overlay
- **Statistical annotations**: Correlation coefficients with complaints
- **Color coding**: Factor/environmental data colored by complaint correlation

### 4.2 Specific Plot Enhancements:

#### Factor Profiles
- Additional panels showing profiles during high vs low complaint periods
- Complaint-weighted factor profiles (emphasizing pollution during complaints)

#### Time Series Plots
- Complaint count as secondary y-axis
- Highlighted periods of simultaneous high factors + high complaints

#### Wind Analysis
- Wind roses split by complaint/no-complaint conditions
- Transport pathway analysis for complaint events

#### Temperature/Pressure Analysis  
- Complaint frequency by environmental condition bins
- Atmospheric stability effects on complaint generation

---

## 5. Validation Metrics Implementation

### 5.1 Statistical Validation Metrics
```python
class ComplaintValidationMetrics:
    def calculate_correlation_metrics(self):
        """
        - Pearson correlation coefficients
        - Spearman rank correlations
        - Kendall tau correlations
        - Partial correlations (controlling for meteorology)
        """
        
    def prediction_accuracy_metrics(self):
        """
        - True positive rate (complaints predicted correctly)
        - False positive rate (false alarms)
        - True negative rate (no complaints predicted correctly)
        - F1 scores for complaint prediction
        - Area under ROC curve (AUC)
        """
        
    def temporal_validation(self):
        """
        - Lead/lag correlation analysis
        - Optimal prediction time windows
        - Seasonal validation consistency
        - Long-term trend validation
        """
```

### 5.2 Environmental Validation
```python
    def environmental_validation_metrics(self):
        """
        - Wind direction consistency (complaints from expected directions)
        - Meteorological condition validation
        - Atmospheric dispersion model validation
        - Distance-decay relationship validation
        """
```

---

## 6. Implementation Phases

### Phase 1: Data Pipeline (Week 1-2)
- Create complaint data ingestion module
- Implement temporal alignment with MMF data
- Data validation and quality control
- Integration testing

### Phase 2: Core Analysis Methods (Week 3-4)
- Implement correlation analysis methods
- Develop prediction models
- Statistical significance testing
- Validation metric calculations

### Phase 3: Visualization Suite (Week 5-6)
- Create 12 new complaint-specific plots
- Enhance all 17+ existing plots with complaint overlays
- Dashboard integration and layout optimization
- Interactive plot enhancements

### Phase 4: Integration Testing (Week 7)
- End-to-end testing with real complaint data
- Performance optimization
- Documentation and user guides
- Quality assurance testing

---

## 7. Expected Outcomes

### 7.1 Scientific Validation
- **Quantified PMF Performance**: Statistical measures of how well PMF factors predict real-world impacts
- **Source Attribution Validation**: Confirmation that identified sources actually cause complaints
- **Environmental Model Validation**: Verification that meteorological correlations match reality

### 7.2 Regulatory Applications
- **Compliance Monitoring**: Clear metrics for regulatory reporting
- **Impact Assessment**: Quantified health/nuisance impact predictions
- **Risk Management**: Predictive capability for complaint mitigation

### 7.3 Operational Benefits
- **Early Warning System**: Predict high complaint days from environmental data
- **Source Control Prioritization**: Focus efforts on factors most correlated with complaints
- **Communication Tool**: Evidence-based community engagement with visual proof

---

## 8. Technical Requirements

### 8.1 Additional Python Dependencies
```python
# Statistical analysis
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Advanced plotting
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

### 8.2 Data Requirements
- **Minimum**: Daily complaint counts for each MMF station coverage area
- **Optimal**: Hourly complaint data with severity ratings and categorization
- **Enhanced**: Geographic coordinates of complaints for spatial analysis

### 8.3 Computational Resources
- Additional processing time: ~30% increase for complaint correlation analysis
- Memory requirements: +~500MB for complaint data and analysis matrices
- Storage: +~50MB per station for enhanced output files

---

## 9. Success Criteria

### 9.1 Quantitative Metrics
- **Correlation Strength**: |r| > 0.3 between at least one PMF factor and complaints
- **Prediction Accuracy**: AUC > 0.7 for complaint day prediction
- **Statistical Significance**: p < 0.05 for factor-complaint correlations

### 9.2 Qualitative Outcomes  
- **Regulatory Acceptance**: Local authority approval of validation approach
- **Community Engagement**: Public understanding of source apportionment results
- **Scientific Publication**: Peer-reviewed validation of PMF results with complaint data

---

## 10. Risk Mitigation

### 10.1 Data Quality Issues
- **Incomplete complaint records**: Implement interpolation and uncertainty quantification
- **Reporting bias**: Account for seasonal/temporal reporting variations
- **Geographic mismatch**: Develop distance-weighted attribution methods

### 10.2 Statistical Challenges
- **Low correlation scenarios**: Implement sensitivity analysis and alternative metrics
- **Temporal misalignment**: Develop robust lag analysis and time-averaging methods
- **Confounding factors**: Control for meteorology, seasonal effects, and reporting patterns

---

This integration represents a **paradigm shift** from statistical source apportionment to **validated environmental impact assessment**. The complaint data will provide the crucial ground truth needed to demonstrate that your PMF analysis has real-world relevance and regulatory value.