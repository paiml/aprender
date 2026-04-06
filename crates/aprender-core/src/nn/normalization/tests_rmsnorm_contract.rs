// =========================================================================
// FALSIFY-RN: rmsnorm-kernel-v1.yaml contract (aprender RMSNorm)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had 30+ RMSNorm unit tests but zero FALSIFY-RN-* tests
//   Why 2: unit tests verify API surface, not mathematical invariants
//   Why 3: no mapping from rmsnorm-kernel-v1.yaml to aprender test names
//   Why 4: aprender predates the provable-contracts YAML convention
//   Why 5: RMSNorm was "obviously correct" (3 lines of math)
//
// References:
//   - provable-contracts/contracts/rmsnorm-kernel-v1.yaml
//   - Zhang & Sennrich (2019) "Root Mean Square Layer Normalization"
// =========================================================================

use super::*;

/// FALSIFY-RN-001: Finiteness — output must be finite for all finite input when eps > 0
///
/// Contract: |RMSNorm(x)_i| < ∞ for all i when ε > 0
#[test]
fn falsify_rn_001_finiteness() {
    let norm = RMSNorm::without_affine(&[4]);

    let test_cases: Vec<(&str, Vec<f32>)> = vec![
        ("normal", vec![1.0, 2.0, 3.0, 4.0]),
        ("small", vec![1e-7, 1e-7, 1e-7, 1e-7]),
        ("large", vec![1e6, 1e6, 1e6, 1e6]),
        ("mixed_sign", vec![-3.0, 2.0, -1.0, 4.0]),
        ("near_zero", vec![1e-20, 0.0, 1e-20, 0.0]),
    ];

    for (name, data) in &test_cases {
        let x = Tensor::new(data, &[1, 4]);
        let y = norm.forward(&x);

        for (i, &val) in y.data().iter().enumerate() {
            assert!(
                val.is_finite(),
                "FALSIFIED RN-001: output[{i}] = {val} is not finite for case '{name}'"
            );
        }
    }
}

/// FALSIFY-RN-002: Scale invariance — RMSNorm(α·x) = sign(α)·RMSNorm(x) for α ≠ 0
///
/// Contract: RMSNorm(α·x) ≈ sign(α) · RMSNorm(x) (with unit gamma)
#[test]
fn falsify_rn_002_scale_invariance() {
    let norm = RMSNorm::without_affine(&[4]);

    let x = Tensor::new(&[1.0, -2.0, 3.0, -0.5], &[1, 4]);
    let y_base = norm.forward(&x);

    for &alpha in &[2.0_f32, 0.5, 100.0, -1.0, -3.0] {
        let x_scaled = Tensor::new(
            &x.data().iter().map(|&v| v * alpha).collect::<Vec<_>>(),
            &[1, 4],
        );
        let y_scaled = norm.forward(&x_scaled);

        let sign = alpha.signum();
        for (i, (&ys, &yb)) in y_scaled.data().iter().zip(y_base.data().iter()).enumerate() {
            let expected = sign * yb;
            let diff = (ys - expected).abs();
            assert!(
                diff < 1e-4,
                "FALSIFIED RN-002: RMSNorm({alpha}·x)[{i}] = {ys}, expected sign({alpha})·RMSNorm(x)[{i}] = {expected}, diff = {diff}"
            );
        }
    }
}

/// FALSIFY-RN-004: Zero vector — RMSNorm(0) should not produce NaN
///
/// Contract: RMSNorm(0) = 0 (output is zero vector, not NaN)
#[test]
fn falsify_rn_004_zero_vector() {
    let norm = RMSNorm::without_affine(&[4]);

    let x = Tensor::new(&[0.0, 0.0, 0.0, 0.0], &[1, 4]);
    let y = norm.forward(&x);

    for (i, &val) in y.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED RN-004: RMSNorm(0)[{i}] = {val} (expected finite, not NaN)"
        );
        // With eps > 0, RMS = sqrt(eps), so output = 0 / sqrt(eps) = 0
        assert!(
            val.abs() < 1e-3,
            "FALSIFIED RN-004: RMSNorm(0)[{i}] = {val} (expected ≈ 0)"
        );
    }
}

/// FALSIFY-RN-005: Unit γ normalized RMS — RMS(RMSNorm(x)/1) ≈ 1 for γ = [1,1,...,1]
///
/// Contract: After RMSNorm with unit weights, the RMS of the output ≈ 1
#[test]
fn falsify_rn_005_unit_gamma_normalized_rms() {
    let norm = RMSNorm::new(&[8]); // default weight = 1.0

    // Note: RN-005 holds when ||x||² >> n·ε. Very small inputs
    // where eps dominates the denominator will have RMS(output) < 1.
    let test_vectors: Vec<Vec<f32>> = vec![
        vec![1.0, -2.0, 3.0, -0.5, 4.0, -1.0, 2.5, -3.0],
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    ];

    for (idx, data) in test_vectors.iter().enumerate() {
        let x = Tensor::new(data, &[1, 8]);
        let y = norm.forward(&x);
        let y_data = y.data();

        // Compute RMS of output
        let rms_out: f32 =
            (y_data.iter().map(|&v| v * v).sum::<f32>() / y_data.len() as f32).sqrt();

        assert!(
            (rms_out - 1.0).abs() < 0.01,
            "FALSIFIED RN-005: RMS(RMSNorm(x)) = {rms_out}, expected ≈ 1.0 (test case {idx})"
        );
    }
}

/// FALSIFY-RN-001b: Functional rms_norm also produces finite output
///
/// Tests the canonical functional::rms_norm directly
#[test]
fn falsify_rn_001_functional_finiteness() {
    let x = Tensor::new(&[1e-7, 1e7, -1e-7, -1e7], &[1, 4]);
    let weight = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let y = crate::nn::functional::rms_norm(&x, &weight, 1e-6);

    for (i, &val) in y.data().iter().enumerate() {
        assert!(
            val.is_finite(),
            "FALSIFIED RN-001: functional::rms_norm output[{i}] = {val} is not finite"
        );
    }
}

/// FALSIFY-RN-002b: Functional rms_norm scale invariance
#[test]
fn falsify_rn_002_functional_scale_invariance() {
    let x = Tensor::new(&[3.0, -1.0, 2.0, -4.0], &[1, 4]);
    let weight = Tensor::new(&[1.0, 1.0, 1.0, 1.0], &[4]);
    let eps = 1e-6;

    let y_base = crate::nn::functional::rms_norm(&x, &weight, eps);

    let alpha = 5.0_f32;
    let x_scaled = Tensor::new(
        &x.data().iter().map(|&v| v * alpha).collect::<Vec<_>>(),
        &[1, 4],
    );
    let y_scaled = crate::nn::functional::rms_norm(&x_scaled, &weight, eps);

    for (i, (&ys, &yb)) in y_scaled.data().iter().zip(y_base.data().iter()).enumerate() {
        let expected = alpha.signum() * yb;
        let diff = (ys - expected).abs();
        assert!(
            diff < 1e-4,
            "FALSIFIED RN-002: functional rms_norm scale invariance violated at [{i}]: got {ys}, expected {expected}"
        );
    }
}

// =========================================================================
// PROPTEST FALSIFY: RMSNorm property-based falsification
//
// Five-Whys (PMAT-354, Phase 10):
//   Why 1: RN-001..005 used fixed d=4 or d=8 dimensions
//   Why 2: Scale invariance (RN-002) could break at edge float ranges
//   Why 3: proptest explores dimension/value combos humans miss
//   Why 4: Epsilon-dominated regime (tiny inputs) untested at scale
//   Why 5: YAML rmsnorm-kernel-v1 calls for proptest on all claims
// =========================================================================

mod rn_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    // RN-001-prop: finiteness for random vectors and dimensions
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]
        #[test]
        fn falsify_rn_001_prop_finiteness(
            dim in prop::sample::select(vec![4_usize, 8, 16, 32, 64]),
            scale in 0.001_f32..1000.0,
        ) {
            let norm = RMSNorm::without_affine(&[dim]);
            let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.13 * scale).sin()).collect();
            let x = Tensor::new(&data, &[1, dim]);
            let y = norm.forward(&x);
            for (i, &val) in y.data().iter().enumerate() {
                prop_assert!(
                    val.is_finite(),
                    "FALSIFIED RN-001-prop: output[{}]={} not finite (d={}, scale={})",
                    i, val, dim, scale
                );
            }
        }
    }

    // RN-002-prop: scale invariance for random vectors
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn falsify_rn_002_prop_scale_invariance(
            dim in prop::sample::select(vec![4_usize, 8, 16, 32]),
            alpha in prop::sample::select(vec![-10.0_f32, -1.0, 0.5, 2.0, 100.0]),
        ) {
            let norm = RMSNorm::without_affine(&[dim]);
            let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
            let x = Tensor::new(&data, &[1, dim]);
            let y_base = norm.forward(&x);

            let x_scaled_data: Vec<f32> = data.iter().map(|&v| v * alpha).collect();
            let x_scaled = Tensor::new(&x_scaled_data, &[1, dim]);
            let y_scaled = norm.forward(&x_scaled);

            let sign = alpha.signum();
            for (i, (&ys, &yb)) in y_scaled.data().iter().zip(y_base.data().iter()).enumerate() {
                let expected = sign * yb;
                prop_assert!(
                    (ys - expected).abs() < 1e-3,
                    "FALSIFIED RN-002-prop: [{i}] got {ys}, expected {expected} (alpha={alpha}, d={dim})"
                );
            }
        }
    }

    // RN-005-prop: unit gamma normalized RMS for random vectors
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        #[test]
        fn falsify_rn_005_prop_unit_gamma_rms(
            dim in prop::sample::select(vec![8_usize, 16, 32, 64]),
        ) {
            let norm = RMSNorm::new(&[dim]);
            let data: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.23).sin() * 10.0).collect();
            let x = Tensor::new(&data, &[1, dim]);
            let y = norm.forward(&x);
            let y_data = y.data();

            let rms_out: f32 = (y_data.iter().map(|&v| v * v).sum::<f32>() / y_data.len() as f32).sqrt();
            prop_assert!(
                (rms_out - 1.0).abs() < 0.05,
                "FALSIFIED RN-005-prop: RMS(output)={} != 1.0 (d={})",
                rms_out, dim
            );
        }
    }
}
