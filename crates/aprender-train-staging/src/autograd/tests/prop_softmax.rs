//! Property-based tests for softmax operations

use super::test_utils::finite_difference;
use crate::autograd::{backward, softmax, Tensor};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_softmax_backward_gradient_check(
        x in prop::collection::vec(-10.0f32..10.0, 2..30)
    ) {
        let a = Tensor::from_vec(x.clone(), true);
        let mut y = softmax(&a);

        let y_len = y.len();
        backward(&mut y, Some(ndarray::Array1::ones(y_len)));

        let analytical = a.grad().expect("gradient should be available");
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let s = softmax(&t);
                s.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.01, "Gradient mismatch at index {}: analytical={}, numerical={}, diff={}",
                        i, analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_softmax_outputs_sum_to_one(
        x in prop::collection::vec(-20.0f32..20.0, 1..100)
    ) {
        let a = Tensor::from_vec(x, false);
        let y = softmax(&a);

        let sum: f32 = y.data().iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5);
    }
}
