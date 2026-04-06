# Benchmark Configuration

## Sample Sizes

All benchmarks use the following sample sizes based on power analysis:

| Benchmark Suite | Sample Size | Warm-up | Confidence Level |
|-----------------|-------------|---------|------------------|
| tensor_ops | 100 | 10 | 95% |
| inference | 100 | 50 | 95% |
| cache | 100 | 10 | 95% |
| tokenizer | 100 | 10 | 95% |
| quantize | 100 | 10 | 95% |
| comparative | 1000 | 50 | 95% |
| lambda | 100 | 0 | 95% |

### Power Analysis

For detecting 10% difference with α=0.05, β=0.20, CV=5%:

```
n = 2 × (Z_α/2 + Z_β)² × (CV/δ)²
n = 2 × (1.96 + 0.84)² × (0.05/0.10)²
n ≈ 4 minimum
```

We use 100 samples (25x minimum) for robust outlier detection.

## Confidence Intervals

All results reported as `[lower, mean, upper]` at 95% confidence.

Method: Bootstrap percentile (10,000 resamples)

Example output:
```
tensor_creation/10      time:   [17.887 ns 17.966 ns 18.043 ns]
                                 ^         ^         ^
                              2.5%      mean      97.5%
                            percentile estimate  percentile
```

## Statistical Tests

### For Comparative Benchmarks

| Test | Purpose | Threshold |
|------|---------|-----------|
| Welch's t-test | Significance | p < 0.001 |
| Cohen's d | Effect size | Report value |
| Mann-Whitney U | Non-parametric | p < 0.001 |

## Outlier Handling

Method: Modified Z-score with MAD
```
modified_z = 0.6745 × (x - median) / MAD
Outlier if |modified_z| > 3.5
```

Policy: Report but don't remove automatically.

## Environment Requirements

- CPU governor: `performance`
- Background processes: minimized
- System load: <10%
- Thermal state: steady (wait 5 min after boot)

## Reproducibility Checklist

- [ ] Set `RUST_BACKTRACE=1`
- [ ] Set CPU governor to performance
- [ ] Close unnecessary applications
- [ ] Run 3 independent sessions
- [ ] Verify CV < 10%
