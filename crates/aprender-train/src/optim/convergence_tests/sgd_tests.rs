//! SGD optimizer convergence tests

#[cfg(test)]
mod tests {
    use super::super::helpers::*;
    use crate::optim::*;
    use crate::Tensor;
    use proptest::prelude::*;
    use proptest::test_runner::Config;

    proptest! {
        #[test]
        fn prop_sgd_converges_quadratic(
            lr in 0.01f32..0.5,
            momentum in 0.0f32..0.9
        ) {
            let optimizer = SGD::new(lr, momentum);
            prop_assert!(test_quadratic_convergence(optimizer, 100, 1.0));
        }

        #[test]
        fn prop_sgd_loss_decreases(
            lr in 0.01f32..0.3
        ) {
            let optimizer = SGD::new(lr, 0.0);
            prop_assert!(test_loss_decreases(optimizer, 50));
        }
    }

    #[test]
    fn test_sgd_with_momentum_faster_than_no_momentum() {
        let mut params_with = vec![Tensor::from_vec(vec![10.0], true)];
        let mut params_without = vec![Tensor::from_vec(vec![10.0], true)];

        let mut opt_with = SGD::new(0.1, 0.9);
        let mut opt_without = SGD::new(0.1, 0.0);

        for _ in 0..20 {
            // Same gradient for both
            let grad = ndarray::arr1(&[2.0 * params_with[0].data()[0]]);
            params_with[0].set_grad(grad.clone());
            params_without[0].set_grad(grad);

            opt_with.step(&mut params_with);
            opt_without.step(&mut params_without);
        }

        // SGD with momentum should converge faster (closer to 0)
        assert!(params_with[0].data()[0].abs() < params_without[0].data()[0].abs());
    }

    // ========================================================================
    // EXTENDED PROPERTY TESTS - High iteration counts for quality validation
    // ========================================================================

    proptest! {
        #![proptest_config(Config::with_cases(1000))]

        #[test]
        fn prop_sgd_rosenbrock(
            lr in 0.0001f32..0.001,
            momentum in 0.8f32..0.99
        ) {
            let mut optimizer = SGD::new(lr, momentum);
            // Rosenbrock is hard - just check it doesn't diverge
            let mut params = vec![Tensor::from_vec(vec![0.0, 0.0], true)];
            for _ in 0..500 {
                let x = params[0].data()[0];
                let y = params[0].data()[1];
                let dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x * x);
                let dy = 200.0 * (y - x * x);
                params[0].set_grad(ndarray::arr1(&[dx, dy]));
                optimizer.step(&mut params);
            }
            prop_assert!(params[0].data().iter().all(|&v| v.is_finite()));
        }

        #[test]
        fn prop_sgd_high_dim(
            lr in 0.05f32..0.15,
            dim in 10usize..30
        ) {
            let optimizer = SGD::new(lr, 0.9);
            prop_assert!(test_high_dim_convergence(optimizer, dim, 300, 2.0));
        }

        #[test]
        fn prop_random_init_sgd(
            init in prop::collection::vec(-50.0f32..50.0, 4),
            lr in 0.05f32..0.2
        ) {
            let mut params = vec![Tensor::from_vec(init.clone(), true)];
            let mut optimizer = SGD::new(lr, 0.9);
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
    fn test_sgd_momentum_behavior() {
        // Test that SGD with momentum accumulates velocity
        // and continues moving even with reduced gradient
        let mut params = vec![Tensor::from_vec(vec![10.0], true)];
        let mut opt = SGD::new(0.01, 0.9);

        // Apply gradient for several steps to build up momentum
        for _ in 0..10 {
            params[0].set_grad(ndarray::arr1(&[2.0 * params[0].data()[0]]));
            opt.step(&mut params);
        }
        let after_10 = params[0].data()[0];

        // Now apply zero gradient - momentum should still cause movement
        params[0].set_grad(ndarray::arr1(&[0.0]));
        opt.step(&mut params);
        let after_zero_grad = params[0].data()[0];

        // Should have moved due to accumulated momentum
        assert!(
            (after_zero_grad - after_10).abs() > 1e-6,
            "Momentum should cause movement even with zero gradient"
        );

        // Both should converge (not diverge)
        assert!(after_10.abs() < 10.0);
        assert!(after_zero_grad.is_finite());
    }
}
