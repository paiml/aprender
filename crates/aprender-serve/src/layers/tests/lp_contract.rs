// =========================================================================
// FALSIFY-LP: linear-projection-v1.yaml contract (realizar Linear)
//
// Five-Whys (PMAT-354):
//   Why 1: realizar Linear had 6 basic tests but zero FALSIFY-LP-* contract tests
//   Why 2: existing tests verify accessors and shapes, not mathematical invariants
//   Why 3: no mapping from linear-projection-v1.yaml claims to test names
//   Why 4: realizar Linear predates the provable-contracts YAML convention
//   Why 5: Linear projection was "obviously correct" (matmul + bias)
//
// References:
//   - provable-contracts/contracts/linear-projection-v1.yaml
//   - Bishop (2006) "Pattern Recognition and Machine Learning"
// =========================================================================

use crate::layers::Linear;
use crate::tensor::Tensor;

/// Helper: create a Linear layer with known weights
fn make_linear_with_weights(
    d_in: usize,
    d_out: usize,
    weights: &[f32],
    bias: &[f32],
) -> Linear {
    let mut layer = Linear::new(d_in, d_out).expect("create layer");
    assert_eq!(weights.len(), d_in * d_out);
    assert_eq!(bias.len(), d_out);
    layer.weight_mut().copy_from_slice(weights);
    layer.bias_mut().copy_from_slice(bias);
    layer
}

/// Helper: create a Linear with zero bias (simulates no-bias)
fn make_linear_no_bias(d_in: usize, d_out: usize, weights: &[f32]) -> Linear {
    make_linear_with_weights(d_in, d_out, weights, &vec![0.0; d_out])
}

/// FALSIFY-LP-001: Output shape — y.shape = (batch, d_out)
///
/// Contract: For x.shape = (batch, d_in), output must have shape (batch, d_out).
#[test]
fn falsify_lp_001_output_shape() {
    for &(batch, d_in, d_out) in &[
        (1, 4, 8),
        (16, 32, 16),
        (3, 1, 1),
        (1, 1, 1),
        (5, 10, 20),
    ] {
        let layer = Linear::new(d_in, d_out).expect("create layer");
        let input = Tensor::from_vec(vec![batch, d_in], vec![0.1; batch * d_in])
            .expect("input");
        let output = layer.forward(&input).expect("forward");
        assert_eq!(
            output.shape(),
            &[batch, d_out],
            "FALSIFIED LP-001: output shape {:?}, expected [{batch}, {d_out}]",
            output.shape()
        );
    }
}

/// FALSIFY-LP-002: Homogeneity without bias — f(alpha*x, W) = alpha*f(x, W)
///
/// Contract: Linear layer without bias satisfies f(alpha*x) = alpha*f(x).
#[test]
fn falsify_lp_002_homogeneity_no_bias() {
    let d_in = 4;
    let d_out = 3;
    // Known weights
    let weights: Vec<f32> = (0..d_in * d_out)
        .map(|i| (i as f32 * 0.37).sin())
        .collect();
    let layer = make_linear_no_bias(d_in, d_out, &weights);

    let x_data: Vec<f32> = (0..2 * d_in)
        .map(|i| (i as f32 * 0.73).cos())
        .collect();
    let x = Tensor::from_vec(vec![2, d_in], x_data.clone()).expect("input");
    let y_base = layer.forward(&x).expect("forward base");

    for &alpha in &[2.0_f32, 0.5, -1.0, -3.0, 0.1] {
        let scaled: Vec<f32> = x_data.iter().map(|&v| v * alpha).collect();
        let x_scaled = Tensor::from_vec(vec![2, d_in], scaled).expect("input scaled");
        let y_scaled = layer.forward(&x_scaled).expect("forward scaled");

        for (i, (&ys, &yb)) in y_scaled.data().iter().zip(y_base.data().iter()).enumerate() {
            let expected = alpha * yb;
            let diff = (ys - expected).abs();
            let tol = 1e-3 * expected.abs().max(1.0);
            assert!(
                diff < tol,
                "FALSIFIED LP-002: f({alpha}*x)[{i}] = {ys}, expected {expected}, diff = {diff}"
            );
        }
    }
}

/// FALSIFY-LP-003: Bias additivity — f(x,W,b) = f(x,W,0) + b
///
/// Contract: Adding bias is independent of matmul.
#[test]
fn falsify_lp_003_bias_additivity() {
    let d_in = 4;
    let d_out = 3;
    let weights: Vec<f32> = (0..d_in * d_out)
        .map(|i| (i as f32 * 0.37).sin())
        .collect();
    let bias = vec![1.5, -2.0, 0.7];

    let layer_with_bias = make_linear_with_weights(d_in, d_out, &weights, &bias);
    let layer_no_bias = make_linear_no_bias(d_in, d_out, &weights);

    let x_data: Vec<f32> = (0..2 * d_in)
        .map(|i| (i as f32 * 0.73).cos())
        .collect();
    let x = Tensor::from_vec(vec![2, d_in], x_data).expect("input");

    let y_bias = layer_with_bias.forward(&x).expect("forward with bias");
    let y_no_bias = layer_no_bias.forward(&x).expect("forward no bias");

    // For each row, the difference should be exactly the bias vector
    for row in 0..2 {
        for col in 0..d_out {
            let idx = row * d_out + col;
            let diff = y_bias.data()[idx] - y_no_bias.data()[idx];
            let expected = bias[col];
            let err = (diff - expected).abs();
            assert!(
                err < 1e-5,
                "FALSIFIED LP-003: row={row}, col={col}: f(x,W,b)-f(x,W,0)={diff}, expected b={expected}"
            );
        }
    }
}

/// FALSIFY-LP-004: Zero input produces bias — f(0, W, b) = b
///
/// Contract: With zero input, output equals the bias vector (broadcast to batch).
#[test]
fn falsify_lp_004_zero_input_produces_bias() {
    let d_in = 4;
    let d_out = 3;
    let weights: Vec<f32> = (0..d_in * d_out)
        .map(|i| (i as f32 * 0.37).sin())
        .collect();
    let bias = vec![10.0, -5.0, 3.14];

    let layer = make_linear_with_weights(d_in, d_out, &weights, &bias);
    let x = Tensor::from_vec(vec![3, d_in], vec![0.0; 3 * d_in]).expect("zero input");
    let y = layer.forward(&x).expect("forward");

    for row in 0..3 {
        for col in 0..d_out {
            let val = y.data()[row * d_out + col];
            let diff = (val - bias[col]).abs();
            assert!(
                diff < 1e-5,
                "FALSIFIED LP-004: f(0)[{row}][{col}] = {val}, expected bias = {}",
                bias[col]
            );
        }
    }
}

/// FALSIFY-LP-002b: Zero preservation without bias — f(0, W) = 0
///
/// Contract: Without bias, zero input produces zero output.
#[test]
fn falsify_lp_002b_zero_preservation_no_bias() {
    let d_in = 4;
    let d_out = 3;
    let weights: Vec<f32> = (0..d_in * d_out)
        .map(|i| (i as f32 * 0.37).sin())
        .collect();
    let layer = make_linear_no_bias(d_in, d_out, &weights);

    let x = Tensor::from_vec(vec![2, d_in], vec![0.0; 2 * d_in]).expect("zero input");
    let y = layer.forward(&x).expect("forward");

    for (i, &val) in y.data().iter().enumerate() {
        assert!(
            val.abs() < 1e-6,
            "FALSIFIED LP-002b: f_no_bias(0)[{i}] = {val}, expected 0"
        );
    }
}

mod lp_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-LP-001-prop: Output shape for random dimensions
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_lp_001_prop_output_shape(
            batch in 1..=8usize,
            d_in in 1..=16usize,
            d_out in 1..=16usize,
        ) {
            let layer = Linear::new(d_in, d_out).expect("create layer");
            let input = Tensor::from_vec(
                vec![batch, d_in],
                vec![0.1; batch * d_in],
            ).expect("input");
            let output = layer.forward(&input).expect("forward");
            prop_assert_eq!(
                output.shape(),
                &[batch, d_out],
                "FALSIFIED LP-001-prop: shape {:?}, expected [{}, {}]",
                output.shape(), batch, d_out
            );
        }
    }

    /// FALSIFY-LP-002-prop: Homogeneity for random alpha, x, W
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn falsify_lp_002_prop_homogeneity(
            alpha in -10.0f32..10.0,
            seed in 0..1000u32,
        ) {
            let d_in = 4;
            let d_out = 3;
            let weights: Vec<f32> = (0..d_in * d_out)
                .map(|i| ((i as f32 + seed as f32) * 0.37).sin())
                .collect();
            let layer = make_linear_no_bias(d_in, d_out, &weights);

            let x_data: Vec<f32> = (0..d_in)
                .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                .collect();
            let x = Tensor::from_vec(vec![1, d_in], x_data.clone()).expect("input");
            let y = layer.forward(&x).expect("forward");

            let scaled: Vec<f32> = x_data.iter().map(|&v| v * alpha).collect();
            let x_s = Tensor::from_vec(vec![1, d_in], scaled).expect("scaled input");
            let y_s = layer.forward(&x_s).expect("forward scaled");

            for (i, (&ys, &yb)) in y_s.data().iter().zip(y.data().iter()).enumerate() {
                let expected = alpha * yb;
                let diff = (ys - expected).abs();
                let tol = 1e-3 * expected.abs().max(1e-6);
                prop_assert!(
                    diff < tol,
                    "FALSIFIED LP-002-prop: f({}*x)[{}] = {}, expected {}, diff = {}",
                    alpha, i, ys, expected, diff
                );
            }
        }
    }

    /// FALSIFY-LP-004-prop: Zero input produces bias for random bias vectors
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_lp_004_prop_zero_input_bias(
            b0 in -100.0f32..100.0,
            b1 in -100.0f32..100.0,
            b2 in -100.0f32..100.0,
        ) {
            let d_in = 4;
            let d_out = 3;
            let weights: Vec<f32> = (0..d_in * d_out)
                .map(|i| (i as f32 * 0.37).sin())
                .collect();
            let bias = vec![b0, b1, b2];
            let layer = make_linear_with_weights(d_in, d_out, &weights, &bias);

            let x = Tensor::from_vec(vec![1, d_in], vec![0.0; d_in]).expect("zero input");
            let y = layer.forward(&x).expect("forward");

            for (col, &expected) in bias.iter().enumerate() {
                let val = y.data()[col];
                let diff = (val - expected).abs();
                prop_assert!(
                    diff < 1e-5,
                    "FALSIFIED LP-004-prop: f(0)[{}] = {}, expected bias = {}",
                    col, val, expected
                );
            }
        }
    }
}
