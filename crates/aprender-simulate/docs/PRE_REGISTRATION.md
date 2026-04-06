# Pre-Registration of Experimental Hypotheses

This document pre-registers the falsifiable hypotheses tested by simular benchmarks, following scientific best practices for reproducibility.

## Registration Date

2024-12-12

## Hypotheses

### H1: Symplectic Integration Energy Conservation

**Claim**: Yoshida 4th-order integrator conserves total system energy to < 1e-9 relative drift over 100 orbits.

**Measurement**: Energy drift = |E_final - E_initial| / |E_initial|

**Success Criterion**: drift < 1e-9

**Falsification**: If drift >= 1e-9, hypothesis is falsified.

### H2: TSP GRASP Optimality Gap

**Claim**: GRASP algorithm achieves solutions within 25% of Held-Karp lower bound for n <= 50 cities.

**Measurement**: Gap = (solution_cost - lower_bound) / lower_bound * 100%

**Success Criterion**: gap <= 25%

**Falsification**: If gap > 25%, hypothesis is falsified.

### H3: Monte Carlo Convergence Rate

**Claim**: Monte Carlo pi estimation error decreases at O(1/√n) rate.

**Measurement**: Regression slope of log(error) vs log(n)

**Success Criterion**: slope in [-0.6, -0.4] (theoretical: -0.5)

**Falsification**: If slope outside range, hypothesis is falsified.

### H4: Deterministic Reproducibility

**Claim**: Same seed produces bit-identical results across platforms.

**Measurement**: Hash of output state vector

**Success Criterion**: hash_linux == hash_macos == hash_windows

**Falsification**: If hashes differ, hypothesis is falsified.

## Statistical Analysis

All tests use:
- Sample size: n >= 100
- Confidence level: 95%
- Effect size: Cohen's d reported
- Multiple comparison correction: Bonferroni where applicable

## Data Availability

Raw benchmark data stored in `metrics/` directory.
Tracked with DVC for version control.

## Protocol

1. Run benchmarks with `cargo bench`
2. Export results to JSON
3. Compare against pre-registered thresholds
4. Report pass/fail with effect sizes
