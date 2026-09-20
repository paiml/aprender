// CONTRACT: rmsnorm-kernel-v1.yaml
// HAND-MAINTAINED, contract-bound. Every test here is named by an obligation or a
// FALSIFY-NORM row in contracts/rmsnorm-kernel-v1.yaml; edit both together.
//
// This file once said "DO NOT EDIT — regenerate with `pv probar --binding`". That was
// false (PMAT-3627, measured 2026-09-20): the in-tree `pv probar` emits seven
// `#[ignore = "no binding available"]` stubs for this contract and cannot produce these
// bodies, which came from the pre-monorepo generator on 2026-02-18. Obeying the old
// header would have replaced five real property tests with seven ignored stubs. The
// same false header stands on the rest of tests/contracts/ -- that is #3630.

use aprender::autograd::Tensor;
use aprender::nn::Module;
use aprender::nn::RMSNorm;
use proptest::prelude::*;

proptest! {
    /// Obligation: Output is finite (invariant)
    /// Formal: |RMSNorm(x)_i| < infinity for all i when eps > 0
    #[test]
    fn prop_output_is_finite(
        data in proptest::collection::vec(-100.0f32..100.0, 1..64usize)
    ) {
        let n = data.len();
        let norm = RMSNorm::new(&[n]);
        let x = Tensor::new(&data, &[1, n]);
        let y = norm.forward(&x);
        for (i, &val) in y.data().iter().enumerate() {
            prop_assert!(
                val.is_finite(),
                "output[{i}]={val} is not finite"
            );
        }
    }

    /// Obligation: Scale invariance (invariant)
    /// Formal: RMSNorm(alpha*x) = sign(alpha)*RMSNorm(x) for alpha != 0
    #[test]
    fn prop_scale_invariance(
        data in proptest::collection::vec(-10.0f32..10.0, 2..32usize),
        alpha in prop::num::f32::NORMAL.prop_filter(
            "nonzero alpha",
            |a| a.abs() > 0.01 && a.abs() < 100.0
        )
    ) {
        let n = data.len();
        let norm = RMSNorm::without_affine(&[n]);
        let x = Tensor::new(&data, &[1, n]);
        let scaled: Vec<f32> = data.iter().map(|&v| v * alpha).collect();
        let x_scaled = Tensor::new(&scaled, &[1, n]);

        let y_orig = norm.forward(&x);
        let y_scaled = norm.forward(&x_scaled);

        // The eps>0 property (PMAT-3627). With eps = 1e-6 in the denominator,
        //     RMSNorm(alpha*x) = sign(alpha) * RMSNorm(x) * sqrt((m+eps) / (m+eps/alpha^2)),
        //     m = mean(x^2),
        // exactly. The eps=0 theorem (RMSNorm.rms_scale_zero_eps) is the k -> 1 limit
        // and is FALSE whenever alpha^2*m is comparable to eps: the input that killed
        // #3621 had alpha^2*m = 1.91e-6 and a 19% discrepancy. Asserting the eps>0 form
        // tests what the implementation computes, in every regime the generator
        // reaches, with a relative tolerance instead of the guessed `0.05 if |alpha|<0.1`.
        let eps = 1e-6f32;
        let m: f32 = data.iter().map(|&v| v * v).sum::<f32>() / n as f32;
        let k = ((m + eps) / (m + eps / (alpha * alpha))).sqrt();
        let sign = alpha.signum();
        let orig_data = y_orig.data();
        let scaled_data = y_scaled.data();
        for i in 0..n {
            let expected = sign * orig_data[i] * k;
            let denom = expected.abs().max(1e-3);
            let rel = (expected - scaled_data[i]).abs() / denom;
            prop_assert!(
                rel < 1e-4,
                "scale invariance (eps>0 form): sign*y[{i}]*k={expected} vs y_scaled[{i}]={}, rel={rel}, k={k}, alpha={alpha}, m={m}",
                scaled_data[i]
            );
        }
    }

    /// Obligation: RMS denominator is positive (bound)
    /// Formal: RMS(x) > 0 when eps > 0
    #[test]
    fn prop_rms_denominator_positive(
        data in proptest::collection::vec(-100.0f32..100.0, 1..64usize)
    ) {
        let n = data.len();
        let norm = RMSNorm::new(&[n]);
        let x = Tensor::new(&data, &[1, n]);
        let y = norm.forward(&x);
        // If RMS denominator were zero/negative, output would be NaN/Inf
        for &val in y.data() {
            prop_assert!(
                val.is_finite(),
                "non-finite output implies RMS denominator issue"
            );
        }
    }

    /// Obligation: SIMD matches scalar within ULP (equivalence)
    #[test]
    #[ignore = "SIMD equivalence — trueno domain"]
    fn prop_simd_matches_scalar_within_ulp(
        _x in proptest::collection::vec(-100.0f32..100.0, 1..32usize)
    ) {
        // SIMD equivalence testing is trueno's responsibility
    }

    /// Obligation: Normalized RMS approximately 1 (idempotency)
    /// Formal: RMS(RMSNorm(x)/gamma) approximately 1 when gamma = 1
    #[test]
    fn prop_normalized_rms_approx_1(
        data in proptest::collection::vec(-10.0f32..10.0, 2..32usize)
            .prop_filter("non-zero", |d| d.iter().any(|v| v.abs() > 0.01))
    ) {
        let n = data.len();
        let norm = RMSNorm::without_affine(&[n]);
        let x = Tensor::new(&data, &[1, n]);
        let y = norm.forward(&x);
        let y_data = y.data();

        // Compute RMS of normalized output
        let sum_sq: f32 = y_data.iter().map(|v| v * v).sum();
        let rms = (sum_sq / n as f32).sqrt();

        prop_assert!(
            (rms - 1.0).abs() < 0.1,
            "RMS of normalized output = {rms}, expected ~1.0"
        );
    }
}

// =========================================================================
// FALSIFY-NORM-001..003: normalization-kernel-v1.yaml falsification tests
// =========================================================================

/// FALSIFY-NORM-001: RMSNorm output has unit RMS (variance ≈ 1).
#[test]
fn falsify_norm_001_rmsnorm_unit_rms() {
    let data: Vec<f32> = (0..128).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let n = data.len();
    let norm = RMSNorm::new(&[n]);
    let x = Tensor::new(&data, &[1, n]);
    let y = norm.forward(&x);
    let sum_sq: f32 = y.data().iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32).sqrt();
    assert!(
        (rms - 1.0).abs() < 0.15,
        "FALSIFY-NORM-001: RMS={rms}, expected ~1.0"
    );
}

/// FALSIFY-NORM-002: RMSNorm zero input produces finite output.
#[test]
fn falsify_norm_002_rmsnorm_zero_input() {
    let n = 64;
    let norm = RMSNorm::new(&[n]);
    let x = Tensor::new(&vec![0.0f32; n], &[1, n]);
    let y = norm.forward(&x);
    for &v in y.data() {
        assert!(v.is_finite(), "FALSIFY-NORM-002: NaN/Inf from zero input");
    }
}

/// FALSIFY-NORM-003: RMSNorm preserves sign structure.
#[test]
fn falsify_norm_003_rmsnorm_sign_preservation() {
    let data = vec![1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5];
    let n = data.len();
    let norm = RMSNorm::new(&[n]);
    let x = Tensor::new(&data, &[1, n]);
    let y = norm.forward(&x);
    // With gamma=1 (default), signs should be preserved
    for i in 0..n {
        let input_sign = data[i].signum();
        let output_sign = y.data()[i].signum();
        assert_eq!(
            input_sign,
            output_sign,
            "FALSIFY-NORM-003: sign flipped at [{i}]: input={}, output={}",
            data[i],
            y.data()[i]
        );
    }
}

/// Regression for #3627 — the exact input that killed #3621's shard 1/3.
///
/// α²·mean(x²) = 1.91e-6 ≈ ε = 1e-6, so the ε term is not negligible and
/// RMSNorm(αx) ≠ sign(α)·RMSNorm(x). The ε=0 theorem (`RMSNorm.rms_scale_zero_eps`,
/// "Proved for ε=0") does not apply in this regime; the implementation is correct and
/// the old assertion was wrong. Numbers reproduced to 6 digits independently of the
/// CI log: y[1] = -1.414158, y_scaled[1] = -1.146122, diff 0.268 > tol 0.05.
///
/// This case must PASS under the ε>0 assertion and turn RED when the ε=0 one is
/// restored — that is the mutation #3627's done_when asks for.
#[test]
fn regression_3627_scale_invariance_in_the_eps_dominated_regime() {
    let data = [0.0f32, -0.159_643_08];
    let alpha = 0.012_254_778f32;
    let n = data.len();
    let norm = RMSNorm::without_affine(&[n]);
    let x = Tensor::new(&data, &[1, n]);
    let scaled: Vec<f32> = data.iter().map(|&v| v * alpha).collect();
    let x_scaled = Tensor::new(&scaled, &[1, n]);
    let y = norm.forward(&x);
    let y_scaled = norm.forward(&x_scaled);
    let eps = 1e-6f32;
    let m: f32 = data.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let k = ((m + eps) / (m + eps / (alpha * alpha))).sqrt();
    let sign = alpha.signum();
    for i in 0..n {
        let expected = sign * y.data()[i] * k;
        let denom = expected.abs().max(1e-3);
        let rel = (expected - y_scaled.data()[i]).abs() / denom;
        assert!(
            rel < 1e-4,
            "3627 regression (eps>0 form): sign*y[{i}]*k={expected} vs y_scaled[{i}]={}, rel={rel}, k={k}",
            y_scaled.data()[i]
        );
    }
    // and the number the CI log reported, so the case is anchored to the incident
    assert!(
        (y_scaled.data()[1] - (-1.146_122f32)).abs() < 1e-5,
        "y_scaled[1]={}",
        y_scaled.data()[1]
    );
    assert!((k - 0.810_462f32).abs() < 1e-5, "k={k}");
}
