# Benchmark Report: certeza-test-suite

## Executive Summary

- **Total Benchmarks**: 3
- **Successful**: 3
- **Failed**: 0
- **Total Runtime**: 45.20 seconds

### Run Information

- **Timestamp**: 2025-11-18T11:07:45.378Z
- **Git Commit**: `abc123def`
- **Git Branch**: `main`
- **Operator**: test-runner

## Benchmark Results

| Benchmark | Type | Opt | Mean (ms) | Median (ms) | Std Dev | CV | CI 95% | Speedup | Status |
|-----------|------|-----|-----------|-------------|---------|-------|--------|---------|--------|
| trueno_vec_push | microbenchmark | release | 21.613 | 21.555 | 0.891 | 4.1% | [21.10, 22.10] | — |  |
| trueno_vec_pop | microbenchmark | release | 18.392 | 18.410 | 0.185 | 1.0% | [18.25, 18.54] | — |  |
| trueno_vec_get | microbenchmark | release | 12.398 | 12.395 | 0.095 | 0.8% | [12.32, 12.48] | — |  |

**Status Legend:**
- 🚀 Major improvement (>20% faster)
- ✅ Good improvement (>10% faster)
- ↗ Minor improvement
- ○ No significant change
- ↘ Minor regression
- ⚠️ Significant regression (>10% slower)
- 🔴 Major regression (>20% slower)

## Test Environment

### Hardware

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i7-9750H |
| Physical Cores | 6 |
| Logical Cores | 12 |
| Base Frequency | 2600 MHz |
| Total Memory | 16.00 GB |
| Available Memory | 8.00 GB |

### Software

| Component | Version |
|-----------|---------|
| Operating System | Linux 5.15.0 |
| Kernel | 5.15.0-91-generic |
| Rust Compiler | rustc 1.82.0 |
| Cargo | cargo 1.82.0 |
| LLVM | 19.1.3 |

### Environment Configuration

| Setting | Value |
|---------|-------|
| CPU Governor | performance |
| Swap Enabled | No |

## Statistical Methodology

### Measurement Protocol

- **Warmup Iterations**: 3
- **Measured Iterations**: 10

### Statistical Analysis

**Descriptive Statistics:**
- Mean, median, standard deviation, min, max
- Coefficient of Variation (CV) for measurement stability
- 95% confidence intervals using bootstrap method (1000 samples)

**Comparative Analysis (when baseline available):**
- Speedup ratio: baseline_mean / current_mean
- Effect size: Cohen's d (standardized mean difference)
- Statistical significance: Welch's t-test (α = 0.05)

**Interpretation Guidelines:**

| Cohen's d | Interpretation |
|-----------|----------------|
| < 0.2 | Negligible effect |
| 0.2 - 0.5 | Small effect |
| 0.5 - 0.8 | Medium effect |
| > 0.8 | Large effect |

**Significance Threshold:**
- P-value < 0.05 considered statistically significant
- Changes marked significant only if p < 0.05 AND |Cohen's d| > 0.2

## Reproducibility

This benchmark report includes complete environment metadata for reproducibility.

**To reproduce these results:**

1. Check out git commit: `abc123def`
2. Install Rust toolchain: `rustc 1.82.0`
3. Build with: `cargo build --release --locked`
4. Run benchmarks with the same configuration

**Note**: Minor variations in absolute timings are expected due to hardware differences and system load.
Statistical significance testing accounts for measurement variability.

---
*Report generated: 2025-11-18T11:07:45.552Z*
*Schema version: 1.0*