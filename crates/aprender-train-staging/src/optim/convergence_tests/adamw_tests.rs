//! AdamW optimizer convergence tests

#[cfg(test)]
mod tests {
    use super::super::helpers::*;
    use crate::optim::*;
    use crate::Tensor;
    use proptest::prelude::*;
    use proptest::test_runner::Config;

    proptest! {
        #[test]
        fn prop_adamw_converges_quadratic(
            lr in 0.05f32..0.5
        ) {
            let optimizer = AdamW::default_params(lr);
            prop_assert!(test_quadratic_convergence(optimizer, 100, 1.5));
        }

        #[test]
        fn prop_adamw_loss_decreases(
            lr in 0.01f32..0.3
        ) {
            let optimizer = AdamW::default_params(lr);
            prop_assert!(test_loss_decreases(optimizer, 30));
        }
    }

    #[test]
    fn test_adamw_weight_decay_effect() {
        // AdamW with weight decay should have smaller final weights than Adam
        let mut params_adamw = vec![Tensor::from_vec(vec![2.0, 2.0], true)];
        let mut params_adam = vec![Tensor::from_vec(vec![2.0, 2.0], true)];

        let mut adamw = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.01);
        let mut adam = Adam::new(0.1, 0.9, 0.999, 1e-8);

        for _ in 0..50 {
            // Same small gradient for both
            let grad = ndarray::arr1(&[0.1, 0.1]);
            params_adamw[0].set_grad(grad.clone());
            params_adam[0].set_grad(grad);

            adamw.step(&mut params_adamw);
            adam.step(&mut params_adam);
        }

        // AdamW should have smaller weights due to weight decay
        let adamw_norm: f32 = params_adamw[0].data().iter().map(|&x| x * x).sum::<f32>().sqrt();
        let adam_norm: f32 = params_adam[0].data().iter().map(|&x| x * x).sum::<f32>().sqrt();

        assert!(adamw_norm < adam_norm);
    }

    // ========================================================================
    // EXTENDED PROPERTY TESTS - High iteration counts for quality validation
    // ========================================================================

    proptest! {
        #![proptest_config(Config::with_cases(1000))]

        #[test]
        fn prop_adamw_ill_conditioned(
            lr in 0.05f32..0.2,
            weight_decay in 0.0f32..0.05
        ) {
            let optimizer = AdamW::new(lr, 0.9, 0.999, 1e-8, weight_decay);
            prop_assert!(test_ill_conditioned_convergence(optimizer, 300, 10.0));
        }

        #[test]
        fn prop_numerical_stability_adamw(
            lr in 0.001f32..0.5,
            weight_decay in 0.0f32..0.5
        ) {
            let optimizer = AdamW::new(lr, 0.9, 0.999, 1e-8, weight_decay);
            prop_assert!(test_large_gradient_stability(optimizer));
        }
    }

    // ========================================================================
    // DETERMINISTIC CONVERGENCE TESTS
    // ========================================================================

    #[test]
    fn test_adamw_regularization_strength() {
        // Higher weight decay = smaller final weights
        let mut params_high = vec![Tensor::from_vec(vec![5.0, 5.0], true)];
        let mut params_low = vec![Tensor::from_vec(vec![5.0, 5.0], true)];

        let mut opt_high = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.1);
        let mut opt_low = AdamW::new(0.1, 0.9, 0.999, 1e-8, 0.001);

        for _ in 0..100 {
            // Constant small gradient
            let grad = ndarray::arr1(&[0.01, 0.01]);
            params_high[0].set_grad(grad.clone());
            params_low[0].set_grad(grad);
            opt_high.step(&mut params_high);
            opt_low.step(&mut params_low);
        }

        let norm_high: f32 = params_high[0].data().iter().map(|x| x * x).sum();
        let norm_low: f32 = params_low[0].data().iter().map(|x| x * x).sum();

        assert!(norm_high < norm_low);
    }
}
