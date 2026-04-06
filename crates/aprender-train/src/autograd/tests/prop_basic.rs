//! Property-based tests for basic operations (add, mul, relu, scale, gelu, swish)

use super::test_utils::finite_difference;
use crate::autograd::{add, backward, gelu, mul, relu, swish, Tensor};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_add_backward_gradient_check(
        xy in prop::collection::vec((-10.0f32..10.0, -10.0f32..10.0), 2..20)
    ) {
        let (x, y): (Vec<f32>, Vec<f32>) = xy.into_iter().unzip();

        let a = Tensor::from_vec(x.clone(), true);
        let b = Tensor::from_vec(y.clone(), true);
        let mut c = add(&a, &b);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical_a = a.grad().expect("gradient should be available");

        // Numerical gradient for a
        let numerical_a = finite_difference(
            |x_val| {
                let t_a = Tensor::from_vec(x_val.to_vec(), false);
                let t_b = Tensor::from_vec(y.clone(), false);
                let t_c = add(&t_a, &t_b);
                t_c.data().sum()
            },
            &x,
            1e-3,  // Larger epsilon for f32 precision
        );

        // Check gradient
        for i in 0..x.len() {
            let diff = (analytical_a[i] - numerical_a[i]).abs();
            prop_assert!(diff < 0.1, "Gradient mismatch at index {}: analytical={}, numerical={}, diff={}",
                        i, analytical_a[i], numerical_a[i], diff);
        }
    }

    #[test]
    fn prop_mul_backward_gradient_check(
        xy in prop::collection::vec((-5.0f32..5.0, -5.0f32..5.0), 2..20)
    ) {
        let (x, y): (Vec<f32>, Vec<f32>) = xy.into_iter().unzip();

        let a = Tensor::from_vec(x.clone(), true);
        let b = Tensor::from_vec(y.clone(), true);
        let mut c = mul(&a, &b);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical_a = a.grad().expect("gradient should be available");

        let numerical_a = finite_difference(
            |x_val| {
                let t_a = Tensor::from_vec(x_val.to_vec(), false);
                let t_b = Tensor::from_vec(y.clone(), false);
                let t_c = mul(&t_a, &t_b);
                t_c.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical_a[i] - numerical_a[i]).abs();
            prop_assert!(diff < 0.1, "Gradient mismatch at index {}: analytical={}, numerical={}, diff={}",
                        i, analytical_a[i], numerical_a[i], diff);
        }
    }

    #[test]
    fn prop_relu_backward_gradient_check(
        x_raw in prop::collection::vec(-10.0f32..10.0, 1..50)
    ) {
        // Filter out values too close to 0 (ReLU discontinuity)
        let x: Vec<f32> = x_raw.into_iter()
            .map(|v| if v.abs() < 0.1 { if v >= 0.0 { 0.2 } else { -0.2 } } else { v })
            .collect();
        let a = Tensor::from_vec(x.clone(), true);
        let mut c = relu(&a);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical = a.grad().expect("gradient should be available");
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let r = relu(&t);
                r.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            // For ReLU at exactly 0, numerical derivative is undefined, so allow larger error
            let tolerance = if x[i].abs() < 0.01 { 0.2 } else { 0.1 };
            prop_assert!(diff < tolerance, "Gradient mismatch at index {}: x={}, analytical={}, numerical={}, diff={}",
                        i, x[i], analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_gelu_backward_gradient_check(
        x in prop::collection::vec(-5.0f32..5.0, 2..20)
    ) {
        let a = Tensor::from_vec(x.clone(), true);
        let mut c = gelu(&a);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical = a.grad().expect("gradient should be available");
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let g = gelu(&t);
                g.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.1,
                "GELU gradient mismatch at index {}: x={}, analytical={}, numerical={}, diff={}",
                i, x[i], analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_swish_backward_gradient_check(
        x in prop::collection::vec(-5.0f32..5.0, 2..20)
    ) {
        let a = Tensor::from_vec(x.clone(), true);
        let mut c = swish(&a);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical = a.grad().expect("gradient should be available");
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let s = swish(&t);
                s.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.1,
                "Swish gradient mismatch at index {}: x={}, analytical={}, numerical={}, diff={}",
                i, x[i], analytical[i], numerical[i], diff);
        }
    }
}
