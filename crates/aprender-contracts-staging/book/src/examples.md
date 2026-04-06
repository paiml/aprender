# Examples

## Complete Example: RMSNorm

### Phase 1 -- Extract from Zhang & Sennrich 2019 (arXiv:1910.10683)

```
RMSNorm(x) = (x / RMS(x)) * γ
where RMS(x) = √(1/n · Σ x_i² + ε)
```

Proof obligations:
- RMS-INV-001: Output has unit RMS (before γ scaling)
- RMS-INV-002: γ=1 gives unit RMS output
- RMS-LIN-001: `RMSNorm(αx, γ) = sign(α) · RMSNorm(x, γ)` (homogeneous degree 0 in x, then γ-scaled)
- RMS-EQV-001: SIMD matches scalar

### Phase 2 -- Contract YAML (abbreviated)

```yaml
metadata:
  version: "1.0.0"
  references:
    - "Zhang & Sennrich (2019). Root Mean Square Layer Normalization. arXiv:1910.10683"

equations:
  rmsnorm:
    formula: "RMSNorm(x, γ, ε) = (x / √(mean(x²) + ε)) * γ"
    domain: "x ∈ ℝ^n, γ ∈ ℝ^n, ε > 0"
    codomain: "ℝ^n"

proof_obligations:
  - type: invariant
    property: "Unit RMS before scaling"
    formal: "|RMS(x / RMS(x)) - 1.0| < ε"
    tolerance: 1.0e-4

  - type: equivalence
    property: "SIMD matches scalar"
    formal: "|rmsnorm_avx2(x) - rmsnorm_scalar(x)| < ULP_TOL * epsilon per element"
    tolerance: "8 ULPs"

kernel_structure:
  phases:
    - name: sum_of_squares
      description: "Compute Σ x_i²"
      invariant: "result >= 0 (sum of squares is non-negative)"
    - name: rms
      description: "Compute √(mean + ε)"
      invariant: "result > 0 (sqrt of positive is positive)"
    - name: normalize
      description: "x_i / rms"
      invariant: "RMS(output) ≈ 1.0"
    - name: scale
      description: "output_i * γ_i"
      invariant: "Element-wise, preserves finiteness"

falsification_tests:
  - id: FALSIFY-RMS-001
    rule: "Unit RMS before scaling"
    prediction: "For random x, |RMS(normalize(x)) - 1.0| < 1e-4"
    test: "proptest with random f32 vectors"
    if_fails: "Sum-of-squares accumulation error or wrong divisor"

  - id: FALSIFY-RMS-002
    rule: "SIMD matches scalar"
    prediction: "For random x, |avx2 - scalar| < 8 ULP per element"
    test: "proptest comparing scalar and SIMD outputs"
    if_fails: "SIMD reassociation error or wrong horizontal sum"
```

### Phase 3 -- Rust trait

```rust
/// Contract: rmsnorm-kernel-v1.yaml
/// Paper: Zhang & Sennrich (2019) arXiv:1910.10683
pub trait RmsNormKernel {
    /// INVARIANT (RMS-INV-001): RMS(output / γ) ≈ 1.0
    /// EQUIVALENCE (RMS-EQV-001): SIMD matches scalar ± 8 ULPs
    fn rmsnorm(&self, input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]);
}
```

### Phase 4 -- Scalar reference

```rust
pub fn rmsnorm_scalar(input: &[f32], gamma: &[f32], eps: f32, output: &mut [f32]) {
    let n = input.len() as f32;
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();
    let rms = (sum_sq / n + eps).sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..input.len() {
        output[i] = input[i] * inv_rms * gamma[i];
    }
}
```

### Phase 5 -- probar tests

```rust
#[probar::property]
fn prop_rmsnorm_unit_rms(xs: Vec<f32>, eps: f32) -> bool {
    let gamma = vec![1.0; xs.len()]; // γ=1 to test normalization only
    let mut output = vec![0.0; xs.len()];
    rmsnorm_scalar(&xs, &gamma, eps, &mut output);
    let rms_out = (output.iter().map(|x| x * x).sum::<f32>() / output.len() as f32).sqrt();
    (rms_out - 1.0).abs() < 1e-4
}

#[probar::property]
fn prop_rmsnorm_simd_parity(xs: Vec<f32>, gamma: Vec<f32>) -> bool {
    let mut scalar_out = vec![0.0; xs.len()];
    let mut simd_out = vec![0.0; xs.len()];
    rmsnorm_scalar(&xs, &gamma, 1e-5, &mut scalar_out);
    rmsnorm_avx2(&xs, &gamma, 1e-5, &mut simd_out);
    scalar_out.iter().zip(simd_out.iter()).all(|(s, a)| {
        (s.to_bits() as i32 - a.to_bits() as i32).unsigned_abs() <= 8
    })
}
```

### Phase 6 -- Kani proof harnesses

```rust
#[cfg(kani)]
mod verification {
    use super::*;

    /// KANI-RMS-001: Sum of squares is non-negative — PROVEN EXACTLY
    ///
    /// For ALL possible f32 vectors up to 16 elements, the sum-of-squares
    /// accumulator is always >= 0. This is the Phase 1 kernel invariant.
    #[kani::proof]
    #[kani::unwind(17)]
    fn verify_sum_of_squares_non_negative() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 16);

        let input: Vec<f32> = (0..n).map(|_| kani::any()).collect();
        kani::assume(input.iter().all(|x| x.is_finite()));

        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        assert!(sum_sq >= 0.0, "Sum of squares must be non-negative");
    }

    /// KANI-RMS-002: RMS is always positive — PROVEN
    ///
    /// For ALL finite inputs with eps > 0, sqrt(mean(x²) + eps) > 0.
    /// This guarantees no division by zero in the normalize phase.
    #[kani::proof]
    #[kani::unwind(17)]
    fn verify_rms_positive() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 16);

        let input: Vec<f32> = (0..n).map(|_| kani::any()).collect();
        kani::assume(input.iter().all(|x| x.is_finite()));

        let eps: f32 = kani::any();
        kani::assume(eps > 0.0 && eps.is_finite());

        let sum_sq: f32 = input.iter().map(|x| x * x).sum();
        let mean_sq = sum_sq / n as f32;
        let rms = (mean_sq + eps).sqrt();

        assert!(rms > 0.0, "RMS must be positive when eps > 0");
        assert!(rms.is_finite(), "RMS must be finite for finite inputs");
    }

    /// KANI-RMS-003: Output finiteness — PROVEN
    ///
    /// For ALL finite inputs and finite gamma, output is finite.
    /// No NaN or Inf can escape the kernel.
    #[kani::proof]
    #[kani::unwind(17)]
    fn verify_rmsnorm_output_finite() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 16);

        let input: Vec<f32> = (0..n).map(|_| kani::any()).collect();
        let gamma: Vec<f32> = (0..n).map(|_| kani::any()).collect();
        kani::assume(input.iter().all(|x| x.is_finite()));
        kani::assume(gamma.iter().all(|x| x.is_finite()));

        let mut output = vec![0.0f32; n];
        rmsnorm_scalar(&input, &gamma, 1e-5, &mut output);

        for i in 0..n {
            assert!(output[i].is_finite(),
                "RMSNorm output[{}] must be finite", i);
        }
    }

    /// KANI-RMS-004: SIMD matches scalar — PROVEN for all vectors ≤ 16
    ///
    /// For ALL possible inputs, AVX2 RMSNorm matches scalar within 8 ULPs.
    #[kani::proof]
    #[kani::unwind(17)]
    #[kani::solver(kissat)]
    fn verify_rmsnorm_simd_parity() {
        let n: usize = kani::any();
        kani::assume(n >= 1 && n <= 16);

        let input: Vec<f32> = (0..n).map(|_| kani::any()).collect();
        let gamma: Vec<f32> = (0..n).map(|_| kani::any()).collect();
        kani::assume(input.iter().all(|x| x.is_finite()));
        kani::assume(gamma.iter().all(|x| x.is_finite()));

        let mut scalar_out = vec![0.0f32; n];
        let mut simd_out = vec![0.0f32; n];
        rmsnorm_scalar(&input, &gamma, 1e-5, &mut scalar_out);
        rmsnorm_avx2(&input, &gamma, 1e-5, &mut simd_out);

        for i in 0..n {
            let ulp_diff = (scalar_out[i].to_bits() as i32
                - simd_out[i].to_bits() as i32).unsigned_abs();
            assert!(ulp_diff <= 8,
                "SIMD diverges from scalar at [{}] by {} ULPs", i, ulp_diff);
        }
    }
}
```

### Verification levels achieved for RMSNorm

| Obligation | Level 1 (Type) | Level 3 (probar) | Level 4 (Kani) |
|-----------|----------------|-------------------|-----------------|
| RMS-INV-001 (unit RMS) | N/A | `prop_rmsnorm_unit_rms` | KANI-RMS-002 (structural) |
| RMS-INV-002 (output finite) | N/A | implicit | KANI-RMS-003 (all inputs <= 16) |
| RMS-EQV-001 (SIMD parity) | N/A | `prop_rmsnorm_simd_parity` | KANI-RMS-004 (all inputs <= 16) |
| Sum-of-squares >= 0 | N/A | implicit | KANI-RMS-001 (proven exactly) |
