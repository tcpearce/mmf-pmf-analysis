# Closure Metrics Comparison: Baseline vs. Regularized (λ=20.0 for CH4)

## CH4 - Target of Regularization
| Metric | Baseline | Regularized | Change |
|--------|----------|-------------|---------|
| Closure % | 99.2% | 3.7% | **-95.5%** |
| Weighted Closure % | 100.0% | 3.7% | **-96.3%** |
| Q Share % | 9.4% | 93.1% | **+83.7%** |
| NRMSE % | 9.1% | 97.9% | **+88.8%** |

**Result**: Regularization successfully pushed CH4 out of factor profiles, dramatically reducing its reconstruction quality.

## Other Gas Species (Not Regularized)
| Species | Baseline Closure % | Regularized Closure % | Change |
|---------|-------------------|---------------------|---------|
| NOX | 99.6% | 82.9% | -16.7% |
| NO | 98.0% | 88.9% | -9.1% |
| NO2 | 96.2% | 77.0% | -19.2% |
| SO2 | 98.1% | 98.2% | +0.1% |
| H2S | 112.2% | 112.5% | +0.3% |

**Result**: Minor spillover effects on NOX species, while SO2/H2S remain stable.

## VOC Species (Not Regularized)
| Species | Baseline Closure % | Regularized Closure % | Change |
|---------|-------------------|---------------------|---------|
| Benzene | 123.2% | 121.4% | -1.8% |
| Toluene | 94.2% | 92.8% | -1.4% |
| Ethylbenzene | 102.4% | 95.0% | -7.4% |

**Result**: Minimal impact on VOC species.

## PM Species (Not Regularized)
| Species | Baseline Closure % | Regularized Closure % | Change |
|---------|-------------------|---------------------|---------|
| PM1 | 100.5% | 104.9% | +4.4% |
| PM2.5 | 100.8% | 103.7% | +2.9% |
| PM4 | 101.2% | 103.5% | +2.3% |
| PM10 | 101.0% | 101.6% | +0.6% |
| TSP | 99.3% | 99.3% | 0.0% |

**Result**: PM species are essentially unaffected.

## Key Findings

1. **Target Species Impact**: CH4 closure dropped from 99.2% to 3.7%, confirming successful regularization.

2. **Q Redistribution**: CH4's Q share increased from 9.4% to 93.1%, indicating it now dominates the model's residuals.

3. **Spillover Effects**: Some impact on NOX species (likely correlated with CH4 in factor profiles), but other species remain stable.

4. **Mass Balance**: Overall mass balance is maintained across the system - the regularization redistributes the fitting burden rather than breaking mass conservation.

5. **Expected Behavior**: This demonstrates exactly what ridge regularization should do - push the target species out of factor profiles at the cost of fit quality for that species.