// =========================================================================
// FALSIFY-LN: layernorm-kernel-v1.yaml contract (aprender LayerNorm)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 20+ LayerNorm tests but zero FALSIFY-LN-* tests
//   Why 2: unit tests verify shapes/params, not mathematical invariants
//   Why 3: no mapping from layernorm-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: LayerNorm was "obviously correct" (y = (x - E[x])/σ * γ + β)
//
// References:
//   - provable-contracts/contracts/layernorm-kernel-v1.yaml
//   - Ba et al. (2016) "Layer Normalization"
// =========================================================================

use super::*;

/// FALSIFY-LN-001: Centering — mean of normalized output ≈ beta
///
/// With gamma=1 and beta=0 (default), E[LN(x)] ≈ 0
#[test]
fn falsify_ln_001_centering() {
    let norm = LayerNorm::new(&[8]);
    let x = Tensor::new(
        &[
            1.0, -2.0, 3.0, 0.5, -1.5, 2.5, -0.5, 1.5, 4.0, -1.0, 2.0, -3.0, 0.0, 1.0, -2.0, 3.0,
        ],
        &[2, 8],
    );
    let y = norm.forward(&x);
    let y_data = y.data();

    // Each row's mean should be ≈ 0 (since beta=0)
    for row in 0..2 {
        let row_mean: f32 = (0..8).map(|c| y_data[row * 8 + c]).sum::<f32>() / 8.0;
        assert!(
            row_mean.abs() < 1e-5,
            "FALSIFIED LN-001: mean(LN(x))[row={row}] = {row_mean}, expected ≈ 0"
        );
    }
}

/// FALSIFY-LN-002: Standardization — variance of normalized output ≈ 1
///
/// With gamma=1, beta=0: Var[LN(x)] ≈ 1
#[test]
fn falsify_ln_002_standardization() {
    let norm = LayerNorm::new(&[8]);
    let x = Tensor::new(
        &[
            1.0, -2.0, 3.0, 0.5, -1.5, 2.5, -0.5, 1.5, 4.0, -1.0, 2.0, -3.0, 0.0, 1.0, -2.0, 3.0,
        ],
        &[2, 8],
    );
    let y = norm.forward(&x);
    let y_data = y.data();

    for row in 0..2 {
        let mean: f32 = (0..8).map(|c| y_data[row * 8 + c]).sum::<f32>() / 8.0;
        let var: f32 = (0..8)
            .map(|c| (y_data[row * 8 + c] - mean).powi(2))
            .sum::<f32>()
            / 8.0;
        // Variance should be close to 1 (gamma=1)
        assert!(
            (var - 1.0).abs() < 0.05,
            "FALSIFIED LN-002: var(LN(x))[row={row}] = {var}, expected ≈ 1.0"
        );
    }
}

/// FALSIFY-LN-005: Idempotency — LN(LN(x)) ≈ LN(x) with gamma=1, beta=0
///
/// Already-normalized data should be unchanged by a second pass.
#[test]
fn falsify_ln_005_idempotency() {
    let norm = LayerNorm::new(&[6]);
    let x = Tensor::new(&[10.0, -5.0, 3.0, 7.0, -2.0, 0.5], &[1, 6]);
    let y1 = norm.forward(&x);
    let y2 = norm.forward(&y1);

    for (i, (&a, &b)) in y1.data().iter().zip(y2.data().iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4,
            "FALSIFIED LN-005: LN(LN(x))[{i}] = {b}, LN(x)[{i}] = {a}, diff = {diff}"
        );
    }
}

/// FALSIFY-LN-006: Shift invariance — LN(x + c) = LN(x) for scalar c
///
/// Adding a constant to all elements doesn't change the normalized output
/// (because mean subtraction cancels the shift).
#[test]
fn falsify_ln_006_shift_invariance() {
    let norm = LayerNorm::new(&[5]);
    let x = Tensor::new(&[1.0, -2.0, 3.0, 0.5, -1.5], &[1, 5]);
    let y_base = norm.forward(&x);

    // Note: very large shifts (1e6+) lose precision in f32 mean subtraction
    for &c in &[10.0_f32, -100.0, 0.001, 1000.0] {
        let shifted_data: Vec<f32> = x.data().iter().map(|&v| v + c).collect();
        let x_shifted = Tensor::new(&shifted_data, x.shape());
        let y_shifted = norm.forward(&x_shifted);

        for (i, (&a, &b)) in y_base
            .data()
            .iter()
            .zip(y_shifted.data().iter())
            .enumerate()
        {
            let diff = (a - b).abs();
            let tol = 1e-3 * a.abs().max(1.0);
            assert!(
                diff < tol,
                "FALSIFIED LN-006: LN(x+{c})[{i}] = {b}, LN(x)[{i}] = {a}, diff = {diff}"
            );
        }
    }
}

/// FALSIFY-LN-007: Boundary — constant input yields beta
///
/// If all elements are the same, output = beta (0 by default).
#[test]
fn falsify_ln_007_constant_input() {
    let norm = LayerNorm::new(&[4]);
    for &c in &[0.0_f32, 1.0, -5.0, 1e6, 1e-6] {
        let x = Tensor::new(&[c, c, c, c], &[1, 4]);
        let y = norm.forward(&x);

        for (i, &val) in y.data().iter().enumerate() {
            // With gamma=1, beta=0: output should be 0 (since (c-c)/σ * 1 + 0 = 0)
            assert!(
                val.abs() < 1e-3,
                "FALSIFIED LN-007: LN([{c},{c},{c},{c}])[{i}] = {val}, expected ≈ 0"
            );
            assert!(
                val.is_finite(),
                "FALSIFIED LN-003 (via LN-007): NaN/Inf for constant input {c}"
            );
        }
    }
}

/// FALSIFY-LN-003: Denominator safety — output finite for all finite input when eps > 0
///
/// Contract: |LN(x)_i| < ∞ for all i when ε > 0
#[test]
fn falsify_ln_003_denominator_safety() {
    let norm = LayerNorm::new(&[4]);

    let test_cases: Vec<(&str, Vec<f32>)> = vec![
        ("normal", vec![1.0, 2.0, 3.0, 4.0]),
        ("small", vec![1e-7, 1e-7, 1e-7, 1e-7]),
        ("large", vec![1e6, 1e6, 1e6, 1e6]),
        ("mixed_sign", vec![-3.0, 2.0, -1.0, 4.0]),
        ("near_zero", vec![1e-20, 0.0, 1e-20, 0.0]),
        ("all_zero", vec![0.0, 0.0, 0.0, 0.0]),
        ("extreme_range", vec![1e-7, 1e7, -1e-7, -1e7]),
    ];

    for (name, data) in &test_cases {
        let x = Tensor::new(data, &[1, 4]);
        let y = norm.forward(&x);

        for (i, &val) in y.data().iter().enumerate() {
            assert!(
                val.is_finite(),
                "FALSIFIED LN-003: output[{i}] = {val} is not finite for case '{name}'"
            );
        }
    }
}

// =========================================================================
// PROPTEST FALSIFY: LayerNorm property-based falsification
//
// Five-Whys (PMAT-354, Phase 10):
//   Why 1: LN-001..007 used fixed d=4, d=5, d=6, d=8 dimensions
//   Why 2: Shift invariance (LN-006) could break at f32 precision limits
//   Why 3: proptest explores dimension/value combos humans miss
//   Why 4: Constant-input regime (LN-007) untested at varied dimensions
//   Why 5: YAML layernorm-kernel-v1 calls for proptest on all claims
// =========================================================================

mod ln_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    // LN-001-prop: centering — mean of LN output ≈ 0 (beta=0)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]
        #[test]
        fn falsify_ln_001_prop_centering(
            dim in prop::sample::select(vec![4_usize, 8, 16, 32, 64]),
            scale in 0.01_f32..100.0,
        ) {
            let norm = LayerNorm::new(&[dim]);
            let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37 * scale).sin() * scale).collect();
            let x = Tensor::new(&data, &[1, dim]);
            let y = norm.forward(&x);
            let y_data = y.data();

            let mean: f32 = y_data.iter().sum::<f32>() / dim as f32;
            prop_assert!(
                mean.abs() < 1e-4,
                "FALSIFIED LN-001-prop: mean(LN(x)) = {} (d={}, scale={})",
                mean, dim, scale
            );
        }
    }

    // LN-002-prop: standardization — variance of LN output ≈ 1 (gamma=1)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]
        #[test]
        fn falsify_ln_002_prop_standardization(
            dim in prop::sample::select(vec![8_usize, 16, 32, 64]),
            scale in 0.1_f32..100.0,
        ) {
            let norm = LayerNorm::new(&[dim]);
            // Use non-constant data with enough spread
            let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).sin() * scale).collect();
            let x = Tensor::new(&data, &[1, dim]);
            let y = norm.forward(&x);
            let y_data = y.data();

            let mean: f32 = y_data.iter().sum::<f32>() / dim as f32;
            let var: f32 = y_data.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
            prop_assert!(
                (var - 1.0).abs() < 0.1,
                "FALSIFIED LN-002-prop: var(LN(x)) = {} (d={}, scale={})",
                var, dim, scale
            );
        }
    }

    // LN-006-prop: shift invariance — LN(x + c) = LN(x)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn falsify_ln_006_prop_shift_invariance(
            dim in prop::sample::select(vec![4_usize, 8, 16, 32]),
            shift in prop::sample::select(vec![-100.0_f32, -1.0, 0.5, 10.0, 1000.0]),
        ) {
            let norm = LayerNorm::new(&[dim]);
            let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
            let x = Tensor::new(&data, &[1, dim]);
            let y_base = norm.forward(&x);

            let shifted: Vec<f32> = data.iter().map(|&v| v + shift).collect();
            let x_shifted = Tensor::new(&shifted, &[1, dim]);
            let y_shifted = norm.forward(&x_shifted);

            for (i, (&a, &b)) in y_base.data().iter().zip(y_shifted.data().iter()).enumerate() {
                let tol = 1e-3 * a.abs().max(1.0);
                prop_assert!(
                    (a - b).abs() < tol,
                    "FALSIFIED LN-006-prop: LN(x)[{i}]={a}, LN(x+{shift})[{i}]={b} (d={dim})"
                );
            }
        }
    }

    // LN-007-prop: constant input → output ≈ 0 (beta=0)
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn falsify_ln_007_prop_constant_input(
            dim in prop::sample::select(vec![4_usize, 8, 16, 32]),
            c in prop::sample::select(vec![-1e6_f32, -1.0, 0.0, 1.0, 1e6]),
        ) {
            let norm = LayerNorm::new(&[dim]);
            let data = vec![c; dim];
            let x = Tensor::new(&data, &[1, dim]);
            let y = norm.forward(&x);

            for (i, &val) in y.data().iter().enumerate() {
                prop_assert!(
                    val.is_finite(),
                    "FALSIFIED LN-003-prop: NaN/Inf at [{i}] for constant {c} (d={dim})"
                );
                prop_assert!(
                    val.abs() < 1e-3,
                    "FALSIFIED LN-007-prop: LN([{c};{dim}])[{i}] = {val} (expected ≈ 0)"
                );
            }
        }
    }
}
