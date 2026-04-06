# Quality Gates

Trueno enforces rigorous quality gates following Toyota Production System principles (Jidoka, Genchi Genbutsu). This chapter documents the quality enforcement mechanisms implemented in TRUENO-SPEC-013.

## Overview

Quality gates are automated checks that must pass before code can be committed or merged:

| Gate | Threshold | Enforcement |
|------|-----------|-------------|
| Test Coverage | ≥90% (95% for releases) | Pre-commit hook |
| Mutation Score | ≥80% | Tier 3 / Nightly |
| PMAT TDG Grade | B+ (85/100) | Pre-commit hook |
| Bashrs Linting | 0 errors | Pre-commit hook |
| Smoke Tests | All pass | Pre-merge |

## Coverage Requirements

### Minimum Thresholds

```
Daily Commits:  ≥90% line coverage
Releases:       ≥95% line coverage (TRUENO-SPEC-013)
```

### Running Coverage

```bash
# Generate coverage report (<5 minutes)
make coverage

# View HTML report
open target/coverage/html/index.html
```

### Coverage Breakdown

The coverage report shows per-crate metrics:

```
trueno:     ~96%  (core library)
trueno-gpu: ~96%  (GPU/CUDA backend)
TOTAL:      95.7% combined
```

### Technical Notes

Coverage instrumentation uses inline environment variables to override the linker:

```bash
# The Makefile handles this automatically via
# CARGO_PROFILE_DEV_LINKER and CARGO_PROFILE_TEST_LINKER
# env vars — no config.toml backup/restore needed.
```

## Smoke Tests (TRUENO-SPEC-013)

Smoke tests verify backend equivalence across SIMD, WGPU, and CUDA:

```bash
# Run all smoke tests
make smoke

# Individual backend tests
cargo test --test smoke_e2e smoke_simd -- --nocapture
cargo test --test smoke_e2e smoke_wgpu --features gpu -- --nocapture
```

### Smoke Test Categories

1. **SIMD Backend Tests**
   - Vector add, mul, dot product
   - ReLU, Softmax activations
   - L2 norm computation

2. **WGPU Backend Tests** (requires GPU)
   - Vector operations (100K+ elements)
   - Matrix multiplication (256x256+)

3. **Backend Equivalence Tests**
   - Scalar vs Auto backend comparison
   - Floating-point tolerance: 1e-5

4. **Edge Case Tests (Poka-Yoke)**
   - Empty inputs
   - Single element
   - Non-aligned sizes (17 elements)
   - NaN/Infinity propagation

## Pixel FKR Tests (Falsification Kernel Regression)

Pixel FKR tests catch GPU kernel bugs using Popperian falsification methodology:

```bash
# Run all pixel FKR tests
make pixel-fkr-all

# Individual suites
make pixel-scalar-fkr   # Baseline truth
make pixel-simd-fkr     # SIMD vs scalar
make pixel-wgpu-fkr     # WGPU vs scalar
make pixel-ptx-fkr      # PTX validation (CUDA)
```

### FKR Test Suites

| Suite | Purpose | Tolerance |
|-------|---------|-----------|
| scalar-pixel-fkr | Golden baseline | Exact |
| simd-pixel-fkr | SIMD correctness | ±1 ULP |
| wgpu-pixel-fkr | GPU correctness | ±2 ULP |
| ptx-pixel-fkr | PTX validation | Static analysis |

### Realizer Operations Tested

- RMS Normalization
- SiLU Activation
- Softmax
- RoPE (Rotary Position Embedding)
- Causal Mask
- Q4_K Dequantization

## Pre-Commit Hook

The pre-commit hook (`.git/hooks/pre-commit`) enforces all quality gates:

```bash
# Gates checked on every commit:
1. PMAT TDG regression check
2. PMAT TDG quality check (B+ minimum)
3. Bashrs linting (Makefile, shell scripts)
4. Coverage threshold (≥90%)
```

### Bypassing (Not Recommended)

```bash
# Only for emergencies - document why in commit message
git commit --no-verify
```

## Tiered Quality Workflow

Trueno uses a tiered approach inspired by certeza (97.7% mutation score):

### Tier 1: On-Save (Sub-second)

```bash
make tier1
# Checks: cargo check, clippy (lib), unit tests, property tests (10 cases)
```

### Tier 2: On-Commit (1-5 minutes)

```bash
make tier2
# Checks: fmt, full clippy, all tests, property tests (256 cases), coverage, TDG
```

### Tier 3: On-Merge/Nightly (Hours)

```bash
make tier3
# Checks: tier2 + mutation testing (80%), security audit, full benchmarks
```

## PMAT Integration

PMAT (Pragmatic AI Labs Multi-Agent Toolkit) provides Technical Debt Grading:

```bash
# Check TDG grade
pmat analyze tdg --min-grade B+

# Repository health score
pmat repo-score . --min-score 90
```

### TDG Grading Scale

| Grade | Score | Status |
|-------|-------|--------|
| A | 93-100 | Excellent |
| A- | 90-92 | Very Good |
| B+ | 85-89 | Good (minimum) |
| B | 80-84 | Acceptable |
| C | <80 | Needs Work |

## Toyota Way Principles

### Jidoka (Built-in Quality)

Quality is built in through:
- Pre-commit hooks that stop defects immediately
- Automated testing at every tier
- No bypass without explicit override

### Genchi Genbutsu (Go and See)

- Smoke tests run actual code on real hardware
- Pixel FKR tests verify visual output
- No simulation - real GPU execution

### Poka-Yoke (Error Prevention)

- Edge case tests prevent common bugs
- Type system enforces API contracts
- Clippy warnings are errors

## Quick Reference

```bash
# Full quality check
make quality-spec-013

# Coverage only
make coverage

# Smoke tests
make smoke

# Pixel FKR
make pixel-fkr-all

# Tier 2 (pre-commit)
make tier2
```

## See Also

- [Testing](./testing.md) - Test infrastructure details
- [Extreme TDD](./extreme-tdd.md) - TDD methodology
- [TRUENO-SPEC-013](../specifications/solidify-quality-spec.md) - Full specification
