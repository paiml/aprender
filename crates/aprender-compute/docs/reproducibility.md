# Reproducibility Guide

## 1. Build Reproducibility

### Version Control
- All dependencies are strictly versioned via `Cargo.toml`.
- `Cargo.lock` is committed to ensure deterministic dependency resolution.
- `rust-toolchain.toml` enforces the exact Rust compiler version/channel (stable).

### CI/CD
- We use GitHub Actions for continuous integration.
- Builds are verified on Ubuntu-latest runners.
- Miri is used to verify unsafe code correctness (see `TRUENO-SPEC-014`).

## 2. Numerical Reproducibility

### Randomness
- All benchmarks and property tests MUST use fixed seeds for pseudo-random number generation.
- We use a custom `SimpleRng` (see `tests/pixel_fkr.rs`) or `rand_pcg` with fixed seeds to ensure deterministic test inputs.
- Property-based tests (Proptest) are configured with explicit seeds for regression testing.

### Floating Point Determinism
- **SIMD vs Scalar**: We expect exact matches or within 1 ULP for most operations.
- **GPU vs CPU**: We acknowledge IEEE 754 deviations due to FMA (Fused Multiply-Add) differences and parallelism order.
- **Tolerance**:
  - Scalar/SIMD: `1e-6` (approx. 1 ULP)
  - GPU: `1e-5` (approx. 2 ULP)
  - Documented in `TRUENO-SPEC-013`.

## 3. ML/AI Reproducibility

- Model initialization uses deterministic RNG seeding.
- Training loops (if added) must log hyperparameters and seeds.
- "Golden Traces" (`golden_traces/`) are used to lock in behavior of complex numerical pipelines.
