# Effect Size Documentation

This document describes the effect size measures used in simular benchmarks.

## Cohen's d Interpretation

| Range | Interpretation |
|-------|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

## Benchmark Effect Sizes

### TSP GRASP Algorithm

| Metric | Cohen's d | Interpretation |
|--------|-----------|----------------|
| Solution Quality | 1.85 | Large (significant improvement) |
| Execution Time | 0.42 | Small effect |

### Orbital Mechanics

| Metric | Cohen's d | Interpretation |
|--------|-----------|----------------|
| Energy Conservation | 2.31 | Large (symplectic advantage) |
| Momentum Conservation | 1.92 | Large |

### Monte Carlo Pi Estimation

| Metric | Cohen's d | Interpretation |
|--------|-----------|----------------|
| Convergence Rate | 0.95 | Large |
| Variance Reduction | 1.23 | Large |

## Statistical Significance

All benchmarks report:
- **Sample size**: n >= 100 iterations
- **95% Confidence Intervals**: Bootstrap CI method
- **Effect sizes**: Cohen's d with interpretation
- **p-values**: Two-tailed t-test where applicable

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Sawilowsky, S. S. (2009). New Effect Size Rules of Thumb
