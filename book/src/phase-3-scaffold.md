# Phase 3: Scaffold -- Contract to Rust Trait + Failing Tests

## Trait Generation

The YAML contract generates a Rust trait. Each equation becomes a method.
Each proof obligation becomes a doc-comment with `INVARIANT:` or `REQUIRES:`
prefix.

```rust
/// Kernel contract: softmax-kernel-v1.yaml
/// Paper: Goodfellow et al. (2016), Deep Learning, Ch. 6.2.2
///
/// Governing equation:
///   softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
pub trait SoftmaxKernel {
    /// Compute softmax over a single row.
    ///
    /// REQUIRES: input.len() >= 1
    /// REQUIRES: all values in input are finite
    ///
    /// INVARIANT (SM-INV-001): |sum(output) - 1.0| < 1e-6
    /// INVARIANT (SM-INV-002): output[i] > 0 for all i
    /// INVARIANT (SM-INV-003): 0 < output[i] < 1 for all i
    /// EQUIVALENCE (SM-EQV-001): softmax(x) == softmax(x + c) for all c
    /// MONOTONICITY (SM-MON-001): x[i] > x[j] implies output[i] > output[j]
    fn softmax(&self, input: &[f32], output: &mut [f32]);
}
```

## Test Generation

Each proof obligation generates a failing test:

```rust
#[cfg(test)]
mod contract_tests {
    use super::*;

    /// FALSIFY-SM-001: Softmax rows sum to 1.0
    /// Prediction: |sum(softmax(x)) - 1.0| < 1e-6 for all finite x
    /// If fails: Missing max-subtraction trick or accumulation error
    #[test]
    fn falsify_sm_001_normalization() {
        todo!("Implementation not yet written — test MUST fail")
    }

    /// FALSIFY-SM-002: Shift invariance
    /// Prediction: softmax(x) == softmax(x + c) within tolerance
    /// If fails: Not using numerically stable variant
    #[test]
    fn falsify_sm_002_shift_invariance() {
        todo!("Implementation not yet written — test MUST fail")
    }

    /// FALSIFY-SM-003: SIMD matches scalar reference
    /// Prediction: |simd_result - scalar_result| < ULP_TOLERANCE * epsilon
    /// If fails: SIMD reassociation or wrong lane extraction
    #[test]
    fn falsify_sm_003_simd_parity() {
        todo!("Implementation not yet written — test MUST fail")
    }
}
```

## The Rule

> **All scaffold tests MUST fail.** If a test passes before implementation,
> either the test is wrong (vacuously true) or the implementation already
> exists (and the contract is redundant).
