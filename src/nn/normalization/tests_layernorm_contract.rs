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
