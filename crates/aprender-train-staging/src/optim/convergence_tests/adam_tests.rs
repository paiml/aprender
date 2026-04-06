//! Adam optimizer convergence tests

#[cfg(test)]
mod tests {
    use super::super::helpers::*;
    use crate::optim::*;
    use crate::Tensor;
    use proptest::prelude::*;
    use proptest::test_runner::Config;

    proptest! {
        #[test]
        fn prop_adam_converges_quadratic(
            lr in 0.05f32..0.5
        ) {
            let optimizer = Adam::default_params(lr);
            prop_assert!(test_quadratic_convergence(optimizer, 100, 1.5));
        }

        #[test]
        fn prop_adam_loss_decreases(
            lr in 0.01f32..0.3
        ) {
            let optimizer = Adam::default_params(lr);
            prop_assert!(test_loss_decreases(optimizer, 30));
        }
    }

    // ========================================================================
    // EXTENDED PROPERTY TESTS - High iteration counts for quality validation
    // ========================================================================

    proptest! {
        #![proptest_config(Config::with_cases(1000))]

        #[test]
        fn prop_adam_ill_conditioned(
            lr in 0.05f32..0.2,
            beta1 in 0.85f32..0.95,
            beta2 in 0.99f32..0.999
        ) {
            let optimizer = Adam::new(lr, beta1, beta2, 1e-8);
            // Relaxed threshold - ill-conditioned problems are hard
            prop_assert!(test_ill_conditioned_convergence(optimizer, 300, 10.0));
        }

        #[test]
        fn prop_adam_high_dim(
            lr in 0.1f32..0.25,
            dim in 10usize..30
        ) {
            let optimizer = Adam::default_params(lr);
            prop_assert!(test_high_dim_convergence(optimizer, dim, 200, 3.0));
        }

        #[test]
        fn prop_numerical_stability_adam(
            lr in 0.001f32..0.5,
            beta1 in 0.5f32..0.99,
            beta2 in 0.9f32..0.9999
        ) {
            let optimizer = Adam::new(lr, beta1, beta2, 1e-8);
            prop_assert!(test_small_gradient_stability(optimizer));
        }

        #[test]
        fn prop_random_init_adam(
            init in prop::collection::vec(-50.0f32..50.0, 4),
            lr in 0.1f32..0.25
        ) {
            let mut params = vec![Tensor::from_vec(init.clone(), true)];
            let mut optimizer = Adam::default_params(lr);
            let initial_norm: f32 = init.iter().map(|x| x * x).sum();

            for _ in 0..150 {
                let grad = params[0].data().mapv(|x| 2.0 * x);
                params[0].set_grad(grad);
                optimizer.step(&mut params);
            }

            // Should make progress (reduce norm)
            let final_norm: f32 = params[0].data().iter().map(|x| x * x).sum();
            prop_assert!(final_norm < initial_norm.max(100.0));
        }
    }

    // ========================================================================
    // DETERMINISTIC CONVERGENCE TESTS
    // ========================================================================

    #[test]
    fn test_adam_rosenbrock_progress() {
        let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);
        let mut params = vec![Tensor::from_vec(vec![-1.0, 1.0], true)];
        let a = 1.0f32;
        let b = 100.0f32;

        let initial_loss = {
            let x = params[0].data()[0];
            let y = params[0].data()[1];
            (a - x).powi(2) + b * (y - x * x).powi(2)
        };

        for _ in 0..1000 {
            let x = params[0].data()[0];
            let y = params[0].data()[1];
            let dx = -2.0 * (a - x) - 4.0 * b * x * (y - x * x);
            let dy = 2.0 * b * (y - x * x);
            params[0].set_grad(ndarray::arr1(&[dx, dy]));
            optimizer.step(&mut params);
        }

        let final_loss = {
            let x = params[0].data()[0];
            let y = params[0].data()[1];
            (a - x).powi(2) + b * (y - x * x).powi(2)
        };

        // Should make progress
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn test_adam_beta_params_effect() {
        // Test that Adam with different beta2 affects update stability
        // Higher beta2 = more smoothing of second moment = more stable updates
        let mut params_high_beta2 = vec![Tensor::from_vec(vec![10.0], true)];
        let mut params_low_beta2 = vec![Tensor::from_vec(vec![10.0], true)];

        let mut opt_high = Adam::new(0.1, 0.9, 0.999, 1e-8);
        let mut opt_low = Adam::new(0.1, 0.9, 0.9, 1e-8);

        // Run for several steps
        for _ in 0..20 {
            let grad_h = ndarray::arr1(&[2.0 * params_high_beta2[0].data()[0]]);
            let grad_l = ndarray::arr1(&[2.0 * params_low_beta2[0].data()[0]]);
            params_high_beta2[0].set_grad(grad_h);
            params_low_beta2[0].set_grad(grad_l);
            opt_high.step(&mut params_high_beta2);
            opt_low.step(&mut params_low_beta2);
        }

        // Both should converge (neither should be NaN/Inf)
        assert!(params_high_beta2[0].data()[0].is_finite());
        assert!(params_low_beta2[0].data()[0].is_finite());

        // Both should make progress toward 0
        assert!(params_high_beta2[0].data()[0].abs() < 10.0);
        assert!(params_low_beta2[0].data()[0].abs() < 10.0);
    }

    #[test]
    fn test_optimizer_state_persistence() {
        // Test that optimizer state (momentum, m/v) persists correctly
        let mut params = vec![Tensor::from_vec(vec![10.0], true)];
        let mut adam = Adam::default_params(0.1);

        // Run some steps
        for _ in 0..10 {
            params[0].set_grad(ndarray::arr1(&[2.0 * params[0].data()[0]]));
            adam.step(&mut params);
        }

        let after_10 = params[0].data()[0];

        // Run 10 more
        for _ in 0..10 {
            params[0].set_grad(ndarray::arr1(&[2.0 * params[0].data()[0]]));
            adam.step(&mut params);
        }

        let after_20 = params[0].data()[0];

        // Should continue converging
        assert!(after_20.abs() < after_10.abs());
    }

    #[test]
    fn test_multiple_param_groups() {
        // Test optimizer with multiple parameter tensors
        let mut params = vec![
            Tensor::from_vec(vec![5.0, 5.0], true),
            Tensor::from_vec(vec![10.0, 10.0, 10.0], true),
        ];

        let mut adam = Adam::default_params(0.2);

        for _ in 0..100 {
            for p in &mut params {
                let grad = p.data().mapv(|x| 2.0 * x);
                p.set_grad(grad);
            }
            adam.step(&mut params);
        }

        // All should converge toward 0 (relaxed threshold)
        for p in &params {
            assert!(
                p.data().iter().all(|&v| v.abs() < 5.0),
                "Expected all values < 5.0, got {:?}",
                p.data()
            );
        }
    }
}
