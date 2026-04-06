# Phase 5: Falsify -- Property Testing via probar + certeza

## Property-Based Testing (probar)

Each proof obligation maps to a probar property test:

```rust
/// SM-INV-001: Normalization invariant
#[probar::property]
fn prop_softmax_sums_to_one(xs: Vec<f32>) -> bool {
    let result = softmax_scalar(&xs);
    (result.iter().sum::<f32>() - 1.0).abs() < 1e-6
}

/// SM-EQV-001: Shift invariance
#[probar::property]
fn prop_softmax_shift_invariant(xs: Vec<f32>, c: f32) -> bool {
    let shifted: Vec<f32> = xs.iter().map(|x| x + c).collect();
    let r1 = softmax_scalar(&xs);
    let r2 = softmax_scalar(&shifted);
    r1.iter().zip(r2.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
}
```

## Metamorphic Relations (probar)

Metamorphic relations test properties that relate different inputs/outputs
without knowing the exact expected output:

```rust
/// SM-MON-001: Monotonicity — order is preserved
#[probar::metamorphic]
fn mr_softmax_preserves_order(xs: Vec<f32>) {
    let result = softmax_scalar(&xs);
    for i in 0..xs.len() {
        for j in 0..xs.len() {
            if xs[i] > xs[j] {
                assert!(result[i] > result[j],
                    "softmax must preserve order: x[{}]={} > x[{}]={} but σ[{}]={} <= σ[{}]={}",
                    i, xs[i], j, xs[j], i, result[i], j, result[j]);
            }
        }
    }
}
```

## SIMD Parity (probar)

The universal SIMD parity test -- every contract gets one:

```rust
/// FALSIFY-<PREFIX>-SIMD: SIMD matches scalar within ULP tolerance
#[probar::property]
fn prop_simd_matches_scalar(data: Vec<f32>) -> bool {
    let scalar_result = softmax_scalar(&data);
    let simd_result = softmax_avx2(&data);
    scalar_result.iter().zip(simd_result.iter()).all(|(s, a)| {
        let ulp_diff = (s.to_bits() as i32 - a.to_bits() as i32).unsigned_abs();
        ulp_diff <= ULP_TOLERANCE
    })
}
```

## Cross-Kernel Isolation

Adapted from `FALSIFY-QDOT-002` -- verify that using the wrong kernel produces
garbage, not accidentally correct results:

```rust
/// FALSIFY-<PREFIX>-ISOLATION: Wrong kernel produces garbage
#[test]
fn falsify_cross_kernel_isolation() {
    let input = generate_test_vector(256);
    let correct = rmsnorm_scalar(&input, &weights, eps);
    let wrong = layernorm_scalar(&input, &weights, eps); // wrong kernel!
    let diff = l2_distance(&correct, &wrong);
    assert!(diff > 1.0, "RMSNorm and LayerNorm must differ — if not, kernels are not isolated");
}
```

## Quality Gates (certeza)

Every contract defines a QA gate that certeza enforces:

```yaml
qa_gate:
  id: "F-SOFTMAX-001"
  name: "Softmax Kernel Contract"
  checks:
    - "All normalization tests pass (SM-INV-001)"
    - "Shift invariance holds (SM-EQV-001)"
    - "SIMD matches scalar (FALSIFY-SM-003)"
    - "Monotonicity holds (SM-MON-001)"
  pass_criteria: "All falsification tests pass"
  falsification: "Introduce off-by-one in max reduction — gate must catch"
```
