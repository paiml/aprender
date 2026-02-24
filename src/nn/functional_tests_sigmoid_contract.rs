// =========================================================================
// FALSIFY-SG: Sigmoid activation contract (aprender nn)
//
// Five-Whys (PMAT-354):
//   Why 1: aprender had no inline FALSIFY-SG-* tests for sigmoid
//   Why 2: sigmoid tests exist but lack contract-mapped FALSIFY naming
//   Why 3: no YAML contract for sigmoid activation yet
//   Why 4: aprender predates the inline FALSIFY convention
//   Why 5: Sigmoid was "obviously correct" (1/(1+exp(-x)))
//
// References:
//   - Nair & Hinton (2010) "Rectified Linear Units Improve RBMs"
// =========================================================================

use crate::autograd::Tensor;
use crate::nn::functional::{sigmoid, sigmoid_scalar};

/// FALSIFY-SG-001: Sigmoid output is in [0, 1] (f32 saturates at extremes)
#[test]
fn falsify_sg_001_output_bounded() {
    let x = Tensor::new(&[-10.0, -1.0, 0.0, 1.0, 10.0], &[5]);
    let y = sigmoid(&x);

    for (i, &val) in y.data().iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&val),
            "FALSIFIED SG-001: sigmoid[{i}]={val}, expected in [0,1]"
        );
    }
}

/// FALSIFY-SG-002: Sigmoid(0) = 0.5
#[test]
fn falsify_sg_002_zero_maps_to_half() {
    let val = sigmoid_scalar(0.0);
    assert!(
        (val - 0.5).abs() < 1e-6,
        "FALSIFIED SG-002: sigmoid(0)={val}, expected 0.5"
    );
}

/// FALSIFY-SG-003: Sigmoid is monotonically increasing
#[test]
fn falsify_sg_003_monotone_increasing() {
    let values: Vec<f32> = (-50..=50).map(|i| i as f32 * 0.1).collect();
    let mut prev = 0.0_f32;

    for (i, &x) in values.iter().enumerate() {
        let y = sigmoid_scalar(x);
        if i > 0 {
            assert!(
                y >= prev - 1e-7,
                "FALSIFIED SG-003: sigmoid({x})={y} < sigmoid({})={prev}",
                values[i - 1]
            );
        }
        prev = y;
    }
}

mod sigmoid_proptest_falsify {
    use super::*;
    use proptest::prelude::*;

    /// FALSIFY-SG-001-prop: Sigmoid output in [0, 1] for random inputs
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn falsify_sg_001_prop_output_bounded(
            x in -100.0f32..100.0,
        ) {
            let val = sigmoid_scalar(x);
            prop_assert!(
                (0.0..=1.0).contains(&val),
                "FALSIFIED SG-001-prop: sigmoid({})={} not in [0,1]",
                x, val
            );
        }
    }

    /// FALSIFY-SG-003-prop: Sigmoid monotone — sigmoid(a) <= sigmoid(b) for a < b
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn falsify_sg_003_prop_monotone(
            a in -50.0f32..50.0,
            delta in 0.01f32..10.0,
        ) {
            let b = a + delta;
            let ya = sigmoid_scalar(a);
            let yb = sigmoid_scalar(b);
            prop_assert!(
                yb >= ya - 1e-7,
                "FALSIFIED SG-003-prop: sigmoid({})={} > sigmoid({})={}",
                a, ya, b, yb
            );
        }
    }
}
