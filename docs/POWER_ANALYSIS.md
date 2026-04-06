# Statistical Power Analysis

This document describes the power analysis methodology for simular benchmarks.

## Sample Size Determination

### Power Requirements

All benchmarks are designed with:
- Power: 0.80 (80% probability of detecting true effect)
- Alpha: 0.05 (5% false positive rate)
- Effect size: Cohen's d >= 0.5 (medium effect)

### Formula

For two-sample t-test:
```
n = 2 * ((z_α/2 + z_β) / d)^2
```

Where:
- z_α/2 = 1.96 (for α = 0.05)
- z_β = 0.84 (for power = 0.80)
- d = Cohen's d effect size

### Required Sample Sizes

| Effect Size | d | Required n |
|-------------|---|------------|
| Small | 0.2 | 394 |
| Medium | 0.5 | 64 |
| Large | 0.8 | 26 |

## Benchmark Sample Sizes

All benchmarks use n >= 100 to detect medium effects with >90% power.

| Benchmark | Sample Size | Detectable Effect |
|-----------|-------------|-------------------|
| TSP GRASP | 100 | d >= 0.4 |
| Monte Carlo | 100 | d >= 0.4 |
| Orbit Integration | 100 | d >= 0.4 |

## Effect Size Interpretation

Cohen's conventions:

| d | Interpretation | Typical Example |
|---|----------------|-----------------|
| 0.2 | Small | Height difference same-sex |
| 0.5 | Medium | IQ difference occupations |
| 0.8 | Large | Height difference sexes |

## Reporting Standards

All benchmarks report:
1. Sample size (n)
2. Mean difference
3. Standard deviation
4. 95% confidence interval
5. Cohen's d effect size
6. p-value (when comparing)

## References

- Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences
- Cohen, J. (1992). A power primer. Psychological Bulletin, 112(1), 155-159
