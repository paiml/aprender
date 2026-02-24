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
