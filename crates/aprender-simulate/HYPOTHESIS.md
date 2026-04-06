# Falsifiable Hypotheses - simular

This document specifies the falsifiable hypotheses for the simular unified simulation engine.
Each hypothesis has explicit, measurable thresholds that can be tested and potentially falsified.

## H1: Symplectic Integration Energy Conservation

**Claim**: The Störmer-Verlet integrator conserves total mechanical energy to within a bounded error.

**Falsification Criteria**:
- Energy drift must be < 1e-9 relative error per orbit for the Sun-Earth system
- Angular momentum conservation error < 1e-12 per orbit

**Test**: `cargo test test_symplectic_energy_conservation`

**Threshold**: If energy drift exceeds 1e-9 per orbit, the hypothesis is falsified.

## H2: GRASP Algorithm Optimality Gap

**Claim**: The GRASP algorithm for TSP achieves solutions within 25% of the 1-tree lower bound for random instances.

**Falsification Criteria**:
- Optimality gap < 25% for n=25 cities with randomized greedy construction
- 2-opt local search improves initial construction by at least 10%

**Test**: `cargo test test_grasp_optimality_gap`

**Threshold**: If gap exceeds 25% consistently over 10 trials, the hypothesis is falsified.

## H3: Monte Carlo Pi Estimation Convergence

**Claim**: Monte Carlo π estimation converges at rate O(1/√n).

**Falsification Criteria**:
- Error at n samples ≈ C/√n where C < 2
- 95% confidence interval narrows proportionally to sample size

**Test**: `cargo test test_monte_carlo_convergence`

**Threshold**: If error does not decrease with increased samples, the hypothesis is falsified.

## H4: Harmonic Oscillator Period Accuracy

**Claim**: Simple harmonic oscillator simulation matches analytical period T = 2π√(m/k).

**Falsification Criteria**:
- Simulated period within 0.1% of analytical value
- Energy conservation within 1e-10 over 100 periods

**Test**: `cargo test test_harmonic_oscillator_period`

**Threshold**: If period deviates by more than 0.1%, the hypothesis is falsified.

## H5: Little's Law Verification

**Claim**: Queueing simulations satisfy Little's Law: L = λW.

**Falsification Criteria**:
- |L - λW| / L < 5% for stable queues
- Law holds across utilization range ρ ∈ [0.1, 0.9]

**Test**: `cargo test test_littles_law`

**Threshold**: If Little's Law is violated by more than 5%, the hypothesis is falsified.

## H6: Kingman's Formula Accuracy

**Claim**: M/M/1 queue wait times follow Kingman's approximation.

**Falsification Criteria**:
- Simulated mean wait within 10% of Kingman's formula
- Variance matches G/G/1 approximation

**Test**: `cargo test test_kingmans_formula`

**Threshold**: If mean wait deviates by more than 10%, the hypothesis is falsified.

## H7: Reproducibility Guarantee

**Claim**: All simulations are reproducible given the same seed.

**Falsification Criteria**:
- Identical seeds produce bit-identical results across runs
- Results reproducible across x86_64 architecture

**Test**: `cargo test test_reproducibility`

**Threshold**: If any run with identical seed produces different results, the hypothesis is falsified.

## H8: Jidoka Stop-on-Error

**Claim**: Simulation halts immediately when invariants are violated.

**Falsification Criteria**:
- NaN/Inf detection triggers immediate stop
- Energy drift beyond threshold triggers Jidoka

**Test**: `cargo test test_jidoka_halt`

**Threshold**: If simulation continues after invariant violation, the hypothesis is falsified.

## Benchmark Confidence Intervals

All performance benchmarks report 95% confidence intervals:

| Benchmark | Mean | 95% CI | Sample Size |
|-----------|------|--------|-------------|
| TSP 25 cities | 1.2ms | ±0.1ms | n=100 |
| Monte Carlo 10k | 0.5ms | ±0.05ms | n=100 |
| Orbit step | 50μs | ±5μs | n=1000 |

Benchmark methodology:
- Run on reference hardware: AMD Ryzen 9 5950X, 64GB RAM
- 100+ iterations for statistical significance
- Report mean ± 1.96×(std/√n) for 95% CI
