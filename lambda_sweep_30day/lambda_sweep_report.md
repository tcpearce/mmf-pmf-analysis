# Lambda Sweep Analysis Report

**Station:** MMF9
**Date Range:** 2023-10-01 to 2023-10-31
**Analysis Date:** 2025-09-28 23:52

## Research Question

Can CH4 regularization with increasing lambda values achieve similar factor correlation diversity as excluding CH4 entirely?

## Lambda Values Tested

Lambda = 0, 5, 20, 50, 100 (plus CH4 exclusion baseline)

## Results Summary

### CH4 Exclusion Baseline
- **Diversity Score:** 0.676
- **Mean Abs Correlation:** 0.324
- **Max Abs Correlation:** 0.661

### Lambda Sweep Results

| Lambda | Diversity Score | Mean Correlation | CH4 Closure % |
|--------|----------------|------------------|---------------|
| 0 | 0.569 | 0.431 | 101.1% |
| 5 | 0.000 | 1.000 | 55.6% |
| 20 | 0.000 | 1.000 | 37.5% |
| 50 | 0.000 | 1.000 | 23.3% |
| 100 | 0.001 | 0.999 | 14.3% |

## Analysis

Generated files:
- **Summary Plot:** `lambda_sweep_comparison.png`
- **Individual Results:** `scenario_*/` directories

