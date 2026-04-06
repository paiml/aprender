//! Property-based tests for layer normalization

use super::test_utils::finite_difference;
use crate::autograd::{backward, layer_norm, Tensor};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn prop_layer_norm_backward_gradient_check_x(
        x in prop::collection::vec(-5.0f32..5.0, 3..15)
    ) {
        let n = x.len();
        let a = Tensor::from_vec(x.clone(), true);
        let gamma = Tensor::from_vec(vec![1.0; n], false);
        let beta = Tensor::from_vec(vec![0.0; n], false);
        let mut c = layer_norm(&a, &gamma, &beta, 1e-5);

        let c_len = c.len();
        backward(&mut c, Some(ndarray::Array1::ones(c_len)));

        let analytical = a.grad().expect("gradient should be available");
        let numerical = finite_difference(
            |x_val| {
                let t = Tensor::from_vec(x_val.to_vec(), false);
                let g = Tensor::from_vec(vec![1.0; n], false);
                let b = Tensor::from_vec(vec![0.0; n], false);
                let ln = layer_norm(&t, &g, &b, 1e-5);
                ln.data().sum()
            },
            &x,
            1e-3,
        );

        for i in 0..x.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.15,
                "LayerNorm gradient (x) mismatch at index {}: x={}, analytical={}, numerical={}, diff={}",
                i, x[i], analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_layer_norm_backward_gradient_check_gamma(
        x in prop::collection::vec(-5.0f32..5.0, 3..15),
        gamma in prop::collection::vec(0.5f32..2.0, 3..15)
    ) {
        // Ensure x and gamma have the same length
        let n = x.len().min(gamma.len());
        let x_vec: Vec<f32> = x.into_iter().take(n).collect();
        let gamma_vec: Vec<f32> = gamma.into_iter().take(n).collect();

        let a = Tensor::from_vec(x_vec.clone(), false);
        let g = Tensor::from_vec(gamma_vec.clone(), true);
        let b = Tensor::from_vec(vec![0.0; n], false);
        let mut c = layer_norm(&a, &g, &b, 1e-5);

        backward(&mut c, Some(ndarray::Array1::ones(n)));

        let analytical = g.grad().expect("gradient should be available");
        let numerical = finite_difference(
            |gamma_val| {
                let t = Tensor::from_vec(x_vec.clone(), false);
                let gam = Tensor::from_vec(gamma_val.to_vec(), false);
                let bet = Tensor::from_vec(vec![0.0; n], false);
                let ln = layer_norm(&t, &gam, &bet, 1e-5);
                ln.data().sum()
            },
            &gamma_vec,
            1e-3,
        );

        for i in 0..n {
            let diff = (analytical[i] - numerical[i]).abs();
            prop_assert!(diff < 0.15,
                "LayerNorm gradient (gamma) mismatch at index {}: gamma={}, analytical={}, numerical={}, diff={}",
                i, gamma_vec[i], analytical[i], numerical[i], diff);
        }
    }

    #[test]
    fn prop_layer_norm_backward_gradient_check_beta(
        x in prop::collection::vec(-5.0f32..5.0, 3..15),
        beta in prop::collection::vec(-2.0f32..2.0, 3..15)
    ) {
        // Ensure x and beta have the same length
        let n = x.len().min(beta.len());
        let x_vec: Vec<f32> = x.into_iter().take(n).collect();
        let beta_vec: Vec<f32> = beta.into_iter().take(n).collect();

        let a = Tensor::from_vec(x_vec.clone(), false);
        let g = Tensor::from_vec(vec![1.0; n], false);
        let b = Tensor::from_vec(beta_vec.clone(), true);
        let mut c = layer_norm(&a, &g, &b, 1e-5);

        backward(&mut c, Some(ndarray::Array1::ones(n)));

        let analytical = b.grad().expect("gradient should be available");

        // Beta gradient should be exactly the upstream gradient (1.0 for all elements)
        for i in 0..n {
            prop_assert_eq!(analytical[i], 1.0);
        }
    }
}
