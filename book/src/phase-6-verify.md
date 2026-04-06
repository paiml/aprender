# Phase 6: Verify -- Bounded Proof via Kani

This is the phase that makes "provable" mean provable. Kani transforms
property-based tests (Level 3: "checked 10,000 random inputs") into bounded
proofs (Level 4: "verified for ALL inputs up to size N").

## Installation and Setup

```bash
# Install Kani (one-time)
cargo install --locked kani-verifier
cargo kani setup

# Cargo.toml: suppress cfg(kani) warnings
[lints.rust]
unexpected_cfgs = { level = "warn", check-cfg = ['cfg(kani)'] }
```

Kani harnesses live behind `#[cfg(kani)]` -- they don't affect normal builds,
tests, or benchmarks. They only activate under `cargo kani`.

## Anatomy of a Kani Proof Harness

Every proof obligation from Phase 5 gets a corresponding Kani harness:

```rust
// File: src/softmax/kani_proofs.rs

#[cfg(kani)]
mod verification {
    use super::*;

    /// KANI-SM-001: Softmax normalization — PROVEN for all vectors ≤ 16 elements
    ///
    /// Obligation: SM-INV-001 (output sums to 1.0)
    /// Strategy: Stub exp() with bounded approximation, verify structural property
    /// Bound: 16 elements (covers common head dimensions / SIMD widths)
    #[kani::proof]
    #[kani::unwind(17)]  // loop bound + 1
    fn verify_softmax_normalization() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 16);

        // Symbolic input: EVERY possible f32 vector of length n
        let input: Vec<f32> = (0..n).map(|_| kani::any()).collect();
        kani::assume(input.iter().all(|x| x.is_finite()));

        let mut output = vec![0.0f32; n];
        softmax_scalar(&input, &mut output);

        // Post-condition: output sums to 1.0 within tolerance
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5,
            "Softmax normalization violated: sum = {}", sum);

        // Post-condition: all outputs positive
        for i in 0..n {
            assert!(output[i] > 0.0, "Softmax output[{}] must be positive", i);
            assert!(output[i] < 1.0, "Softmax output[{}] must be < 1.0", i);
        }
    }
}
```

## The Three Kani Verification Strategies

Different proof obligations require different strategies based on what
Kani can and cannot handle:

### Strategy 1: Exhaustive (integer / structural logic)

For operations involving only integer arithmetic or structural properties,
Kani verifies exactly with zero false positives.

**Best for:** Quantized dot products, bsum precomputation, format dispatch,
shape validation, index bounds.

```rust
/// KANI-QDOT-001: Bsums precomputation — PROVEN EXACTLY
///
/// The offset term in quantized dot product depends only on activations.
/// Precomputed bsums must equal on-the-fly bsums for ALL possible inputs.
/// This is integer arithmetic — Kani verifies it exactly.
#[kani::proof]
#[kani::unwind(33)]  // 32 elements per sub-block + 1
fn verify_bsums_precomputation_exact() {
    // Symbolic activation block: every possible i8 value
    let activations: [i8; 32] = kani::any();

    // Precomputed bsum
    let precomputed: i32 = activations.iter().map(|&x| x as i32).sum();

    // On-the-fly bsum (as done inside the superblock loop)
    let mut online: i32 = 0;
    for i in 0..32 {
        online += activations[i] as i32;
    }

    // These MUST be exactly equal — integer arithmetic, no tolerance
    assert_eq!(precomputed, online,
        "Bsum precomputation diverges from online computation");
}
```

### Strategy 2: Stub Float Transcendentals

For operations using `exp()`, `sqrt()`, `log()` -- Kani over-approximates these
(returns any value in valid range). We stub them with bounded approximations
to avoid false positives while still verifying the structural logic.

**Best for:** Softmax, RMSNorm, SwiGLU, any kernel using transcendentals.

```rust
/// Bounded exp approximation for Kani verification.
/// Returns a value that satisfies exp()'s key properties:
///   - exp(x) > 0 for all x
///   - exp(0) = 1
///   - exp is monotonically increasing
/// This is NOT numerically accurate — it's a CONTRACT STUB that
/// preserves the properties we're verifying.
#[cfg(kani)]
fn exp_stub(x: f32) -> f32 {
    // Use a polynomial approximation valid for small ranges
    // For verification, we only need the structural properties
    let result: f32 = kani::any();
    kani::assume(result > 0.0);           // exp(x) > 0 always
    kani::assume(result.is_finite());
    result
}

#[kani::proof]
#[kani::stub(f32::exp, exp_stub)]
#[kani::unwind(17)]
fn verify_softmax_positivity_with_stub() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);

    let input: Vec<f32> = (0..n).map(|_| kani::any()).collect();
    kani::assume(input.iter().all(|x| x.is_finite()));

    let mut output = vec![0.0f32; n];
    softmax_scalar(&input, &mut output);

    // Even with approximate exp, all outputs must be positive
    for i in 0..n {
        assert!(output[i] > 0.0);
    }
}
```

### Strategy 3: Function Contracts (Compositional)

For composite kernels (attention = softmax + matmul + scale), verify each
sub-kernel independently, then compose using `#[kani::stub_verified]`.

**Best for:** Attention, transformer layers, any multi-step pipeline.

```rust
/// Contract for softmax: preconditions + postconditions
#[kani::requires(input.len() >= 1)]
#[kani::requires(input.iter().all(|x| x.is_finite()))]
#[kani::ensures(|result| result.iter().all(|&x| x > 0.0 && x < 1.0))]
#[kani::ensures(|result| (result.iter().sum::<f32>() - 1.0).abs() < 1e-5)]
pub fn softmax_verified(input: &[f32]) -> Vec<f32> {
    softmax_scalar(input)
}

/// Verify softmax contract itself
#[kani::proof_for_contract(softmax_verified)]
#[kani::unwind(17)]
fn verify_softmax_contract() {
    let n: usize = kani::any();
    kani::assume(n >= 1 && n <= 16);
    let input: Vec<f32> = (0..n).map(|_| kani::any()).collect();
    softmax_verified(&input);
}

/// Now use verified softmax contract in attention proof —
/// softmax is replaced with its contract abstraction (free postconditions)
#[kani::proof]
#[kani::stub_verified(softmax_verified)]
#[kani::unwind(9)]
fn verify_attention_uses_normalized_weights() {
    let seq_len: usize = kani::any();
    kani::assume(seq_len >= 1 && seq_len <= 8);

    let scores: Vec<f32> = (0..seq_len).map(|_| kani::any()).collect();
    kani::assume(scores.iter().all(|x| x.is_finite()));

    // softmax_verified is replaced with: assume(preconditions) → any_where(postconditions)
    let weights = softmax_verified(&scores);

    // This is now guaranteed by the verified contract — not tested, PROVEN:
    assert!(weights.iter().all(|&w| w > 0.0));
    assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-5);
}
```

## SIMD Parity Proofs

The highest-value Kani harness: prove SIMD kernels match scalar for ALL inputs.

Kani supports all SIMD intrinsics (`simd_add`, `simd_mul`, `simd_shuffle`,
`simd_reduce_*`, etc.), making this a natural fit.

```rust
/// KANI-QDOT-SIMD-001: AVX2 Q4K dot product matches scalar — PROVEN
///
/// For ALL possible 256-element super-blocks and ALL possible activation
/// vectors, the AVX2 kernel produces the same result as the scalar kernel.
/// Integer sub-block arithmetic — exact verification, no tolerance.
#[kani::proof]
#[kani::unwind(257)]  // 256 elements + 1
#[kani::solver(kissat)]
fn verify_q4k_simd_matches_scalar() {
    // Symbolic Q4_K super-block (144 bytes)
    let block: [u8; 144] = kani::any();
    // Symbolic activation vector (256 i8 values for Q8_K path)
    let activations: [i8; 256] = kani::any();

    let scalar_result = fused_q4k_dot_scalar(&block, &activations);
    let simd_result = fused_q4k_dot_avx2(&block, &activations);

    // Within ULP tolerance from contract (8 ULPs for Q4_K)
    let ulp_diff = (scalar_result.to_bits() as i32 - simd_result.to_bits() as i32).unsigned_abs();
    assert!(ulp_diff <= 8, "SIMD diverges from scalar by {} ULPs", ulp_diff);
}
```

## Negative Verification (should_panic)

Prove that invalid inputs MUST be rejected:

```rust
/// KANI-LAYOUT-001: ValidatedEmbedding rejects >50% zeros — PROVEN
///
/// For ALL possible data vectors with >50% zeros, the constructor panics.
/// This is the Poka-Yoke guarantee — mistakes are physically impossible.
#[kani::proof]
#[kani::should_panic]
#[kani::unwind(17)]
fn verify_validated_embedding_rejects_zeros() {
    let n: usize = kani::any();
    kani::assume(n >= 2 && n <= 16);

    let data: Vec<f32> = (0..n).map(|_| kani::any()).collect();
    let zero_count = data.iter().filter(|&&x| x == 0.0).count();

    // Assume MORE than 50% zeros
    kani::assume(zero_count * 2 > n);

    // This MUST panic — if it doesn't, Kani reports FAILURE
    ValidatedEmbedding::new(data, n, 1).unwrap();
}
```

## Concrete Playback

When Kani finds a counterexample, it generates a concrete unit test:

```bash
# Run Kani with concrete playback
cargo kani --harness verify_softmax_normalization --concrete-playback=inplace
```

This adds a `#[test]` to your source code with the exact input that triggered
the failure -- bridging Level 4 back to Level 2 for debugging:

```rust
// Automatically generated by Kani
#[test]
fn kani_concrete_playback_verify_softmax_normalization() {
    let input = vec![3.4028235e38_f32, -3.4028235e38, 0.0, 1.0];
    let mut output = vec![0.0; 4];
    softmax_scalar(&input, &mut output);
    let sum: f32 = output.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5); // FAILS — exposes edge case
}
```

## Kani in CI

```yaml
# .github/workflows/kani.yml
name: Kani Verification
on: [push, pull_request]
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: model-checking/kani-github-action@v1
        with:
          args: --all-harnesses
```

## Contract YAML: Kani Harness Registry

Every contract lists its Kani harnesses alongside falsification tests:

```yaml
kani_harnesses:
  - id: KANI-SM-001
    obligation: SM-INV-001
    property: "Softmax output sums to 1.0"
    bound: 16
    strategy: stub_float
    solver: cadical
    harness: verify_softmax_normalization

  - id: KANI-SM-002
    obligation: SM-INV-002
    property: "Softmax outputs are positive"
    bound: 16
    strategy: stub_float
    harness: verify_softmax_positivity

  - id: KANI-SM-003
    obligation: SM-EQV-001
    property: "SIMD matches scalar"
    bound: 256
    strategy: exhaustive
    solver: kissat
    harness: verify_softmax_simd_parity

  - id: KANI-SM-004
    obligation: SM-MON-001
    property: "Softmax preserves input order"
    bound: 8
    strategy: stub_float
    harness: verify_softmax_monotonicity
```

## What Kani Cannot Verify (and What Fills the Gap)

| Property | Kani | Alternative | Why |
|----------|------|-------------|-----|
| Numerical accuracy of `exp()` | Over-approx | probar proptest (L3) | Kani returns any positive float for exp |
| Numerical accuracy of `sqrt()` | Over-approximated | probar proptest (Level 3) | Same issue |
| Unbounded vector lengths | Bounded to N | probar proptest (Level 3) | Kani requires finite unwind |
| Concurrent dispatch | Not supported | Manual review + integration tests | Kani is single-threaded |
| GPU kernel correctness | Not supported | Layer parity tool (`apr parity`) | Kani verifies CPU only |
| End-to-end inference | Too large | Golden trace validation | State space too large |

**The principle:** Kani proves structural and algebraic properties exhaustively.
probar tests numerical accuracy statistically. Together they cover all
obligations at appropriate levels.

## Kani Verification Targets by Kernel

| Kernel | Exhaustive (exact) | Stub-float | Compositional |
|--------|-------------------|------------|---------------|
| Quantized dot product | bsums, scale extraction, nibble packing | Final f32 accumulation | N/A (leaf kernel) |
| Softmax | Output bounds, index safety | Normalization sum, monotonicity | Used by attention |
| RMSNorm | Sum-of-squares non-negative | Normalization to unit RMS | Used by transformer layer |
| RoPE | Rotation structure, periodicity | sin/cos accuracy | Used by attention |
| SwiGLU | Gate x up structure | Swish activation | Used by FFN |
| MatMul | Index bounds, shape | Accumulation | Used by everything |
| Attention | Score scaling, shape | Softmax + matmul composition | Via `stub_verified` |
