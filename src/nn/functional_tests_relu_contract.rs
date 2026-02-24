// =========================================================================
// FALSIFY-AK (ReLU): activation-kernel-v1.yaml contract (aprender ReLU)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-AK-* tests for ReLU
//   Why 2: GELU/SiLU have FALSIFY tests but ReLU was omitted
//   Why 3: ReLU was "too simple" to need contract tests
//   Why 4: activation-kernel-v1.yaml covers GELU, SiLU, AND ReLU
//   Why 5: ReLU(x)=max(0,x) seemed un-falsifiable
//
// References:
//   - provable-contracts/contracts/activation-kernel-v1.yaml
//   - Nair & Hinton (2010) "Rectified Linear Units"
// =========================================================================

use super::*;
use crate::autograd::Tensor;

/// FALSIFY-AK-001: ReLU(x) >= 0 (non-negativity)
#[test]
fn falsify_ak_001_relu_non_negative() {
    let input = Tensor::new(&[-3.0, -1.0, 0.0, 1.0, 5.0], &[5]);
    let output = relu(&input);

    for (i, &v) in output.data().iter().enumerate() {
        assert!(
            v >= 0.0,
            "FALSIFIED AK-001: ReLU output[{i}] = {v} < 0"
        );
    }
}

/// FALSIFY-AK-002: ReLU(x) = x for x > 0
#[test]
fn falsify_ak_002_relu_positive_identity() {
    let input = Tensor::new(&[0.5, 1.0, 2.5, 10.0], &[4]);
    let output = relu(&input);

    for (i, (&out, &inp)) in output.data().iter().zip(input.data().iter()).enumerate() {
        assert!(
            (out - inp).abs() < 1e-6,
            "FALSIFIED AK-002: ReLU({inp}) = {out}, expected {inp} (positive identity) at [{i}]"
        );
    }
}

/// FALSIFY-AK-003: ReLU(x) = 0 for x <= 0
#[test]
fn falsify_ak_003_relu_zero_for_negative() {
    let input = Tensor::new(&[-5.0, -1.0, -0.01, 0.0], &[4]);
    let output = relu(&input);

    for (i, &v) in output.data().iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "FALSIFIED AK-003: ReLU({}) = {v}, expected 0",
            input.data()[i]
        );
    }
}

/// FALSIFY-AK-004: ReLU preserves output length
#[test]
fn falsify_ak_004_relu_output_length() {
    let input = Tensor::new(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[6]);
    let output = relu(&input);

    assert_eq!(
        output.data().len(),
        6,
        "FALSIFIED AK-004: ReLU output len {} != input len 6",
        output.data().len()
    );
}
