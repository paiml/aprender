// =========================================================================
// FALSIFY-LP: linear-projection-v1.yaml contract (aprender Linear layer)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-LP-* tests for linear layers
//   Why 2: Linear tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no mapping from linear-projection-v1.yaml to inline test names
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Linear layer was "obviously correct" (y = xW^T + b)
//
// References:
//   - provable-contracts/contracts/linear-projection-v1.yaml
//   - Glorot & Bengio (2010) "Understanding the difficulty of training
//     deep feedforward neural networks"
// =========================================================================

use super::*;
use crate::autograd::Tensor;

/// FALSIFY-LP-001: Output shape is [batch, out_features]
#[test]
fn falsify_lp_001_output_shape() {
    let layer = Linear::new(8, 16);
    let input = Tensor::new(&vec![0.1; 4 * 8], &[4, 8]);

    let output = layer.forward(&input);
    assert_eq!(
        output.shape(),
        &[4, 16],
        "FALSIFIED LP-001: output shape {:?} != [4, 16]",
        output.shape()
    );
}

/// FALSIFY-LP-002: Output values are finite
#[test]
fn falsify_lp_002_finite_output() {
    let layer = Linear::new(4, 8);
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);

    let output = layer.forward(&input);
    for (i, &v) in output.data().iter().enumerate() {
        assert!(
            v.is_finite(),
            "FALSIFIED LP-002: output[{i}] = {v} is not finite"
        );
    }
}

/// FALSIFY-LP-003: Without-bias layer has no bias
#[test]
fn falsify_lp_003_without_bias() {
    let layer = Linear::without_bias(4, 8);
    assert!(
        !layer.has_bias(),
        "FALSIFIED LP-003: without_bias layer reports has_bias=true"
    );
}

/// FALSIFY-LP-004: Parameter count matches expectation
#[test]
fn falsify_lp_004_parameter_count() {
    let with_bias = Linear::new(4, 8);
    let without_bias = Linear::without_bias(4, 8);

    assert_eq!(
        with_bias.parameters().len(),
        2,
        "FALSIFIED LP-004: expected 2 params (weight+bias), got {}",
        with_bias.parameters().len()
    );
    assert_eq!(
        without_bias.parameters().len(),
        1,
        "FALSIFIED LP-004: expected 1 param (weight only), got {}",
        without_bias.parameters().len()
    );
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
            let layer = Linear::new(d_in, d_out);
            let input = Tensor::new(&vec![0.1; batch * d_in], &[batch, d_in]);
            let output = layer.forward(&input);
            prop_assert_eq!(
                output.shape(),
                &[batch, d_out],
                "FALSIFIED LP-001-prop: shape {:?}, expected [{}, {}]",
                output.shape(), batch, d_out
            );
        }
    }

    /// FALSIFY-LP-002-prop: Homogeneity for random alpha values
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn falsify_lp_002_prop_homogeneity(
            alpha in -10.0f32..10.0,
            seed in 0..1000u32,
        ) {
            let d_in = 4;
            let d_out = 3;
            let layer = Linear::without_bias_with_seed(d_in, d_out, Some(seed as u64));

            let x_data: Vec<f32> = (0..d_in)
                .map(|i| ((i as f32 + seed as f32) * 0.73).cos())
                .collect();
            let x = Tensor::new(&x_data, &[1, d_in]);
            let y = layer.forward(&x);

            let scaled: Vec<f32> = x_data.iter().map(|&v| v * alpha).collect();
            let x_s = Tensor::new(&scaled, &[1, d_in]);
            let y_s = layer.forward(&x_s);

            for (i, (&ys, &yb)) in y_s.data().iter().zip(y.data().iter()).enumerate() {
                let expected = alpha * yb;
                let diff = (ys - expected).abs();
                // Use absolute + relative tolerance for f32 accumulation error
                let tol = 1e-4 + 1e-3 * expected.abs();
                prop_assert!(
                    diff < tol,
                    "FALSIFIED LP-002-prop: f({}*x)[{}] = {}, expected {}",
                    alpha, i, ys, expected
                );
            }
        }
    }

    /// FALSIFY-LP-004-prop: Zero input produces bias for random bias vectors
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_lp_004_prop_zero_input_bias(
            seed in 0..1000u32,
        ) {
            let d_in = 4;
            let d_out = 3;
            let layer = Linear::with_seed(d_in, d_out, Some(seed as u64));
            let x = Tensor::zeros(&[1, d_in]);
            let y = layer.forward(&x);

            let bias = layer.bias().expect("layer has bias");
            let bias_data = bias.data();

            for (col, &expected) in bias_data.iter().enumerate() {
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
