// =========================================================================
// FALSIFY-DO: Dropout contract (aprender nn)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-DO-* tests for dropout
//   Why 2: dropout tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for dropout yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Dropout was "obviously correct" (random zeroing)
//
// References:
//   - Srivastava et al. (2014) "Dropout: A Simple Way to Prevent NNs from Overfitting"
// =========================================================================

use super::*;
use crate::autograd::Tensor;
use crate::nn::module::Module;

/// FALSIFY-DO-001: Eval mode returns input unchanged
#[test]
fn falsify_do_001_eval_identity() {
    let mut dropout = Dropout::with_seed(0.5, 42);
    dropout.eval();

    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let output = dropout.forward(&input);

    for (i, (&inp, &out)) in input.data().iter().zip(output.data().iter()).enumerate() {
        assert!(
            (inp - out).abs() < 1e-6,
            "FALSIFIED DO-001: eval output[{i}]={out} != input[{i}]={inp}"
        );
    }
}

/// FALSIFY-DO-002: Output shape matches input shape
#[test]
fn falsify_do_002_shape_preserved() {
    let dropout = Dropout::with_seed(0.5, 42);
    let input = Tensor::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let output = dropout.forward(&input);

    assert_eq!(
        output.shape(),
        input.shape(),
        "FALSIFIED DO-002: output shape={:?} != input shape={:?}",
        output.shape(),
        input.shape()
    );
}

/// FALSIFY-DO-003: p=0.0 dropout returns input unchanged (training mode)
#[test]
fn falsify_do_003_zero_p_identity() {
    let dropout = Dropout::with_seed(0.0, 42);
    let input = Tensor::new(&[1.0, 2.0, 3.0], &[3]);
    let output = dropout.forward(&input);

    for (i, (&inp, &out)) in input.data().iter().zip(output.data().iter()).enumerate() {
        assert!(
            (inp - out).abs() < 1e-6,
            "FALSIFIED DO-003: p=0 output[{i}]={out} != input[{i}]={inp}"
        );
    }
}

/// FALSIFY-DO-004: Training dropout produces some zeros
#[test]
fn falsify_do_004_training_produces_zeros() {
    let dropout = Dropout::with_seed(0.5, 42);
    let input = Tensor::ones(&[100]);
    let output = dropout.forward(&input);

    let n_zeros = output.data().iter().filter(|&&x| x == 0.0).count();
    // With p=0.5 and 100 elements, expect ~50 zeros (allow 20-80)
    assert!(
        (20..=80).contains(&n_zeros),
        "FALSIFIED DO-004: {n_zeros} zeros out of 100 (expected ~50 for p=0.5)"
    );
}
