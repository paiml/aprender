//! Tests for the Differential Privacy module.

#![allow(clippy::module_inception)]
#[cfg(test)]
mod tests {
    use crate::optim::dp::{
        accountant::RdpAccountant,
        budget::PrivacyBudget,
        config::DpSgdConfig,
        dp_sgd::DpSgd,
        error::DpError,
        gradient::{add_gaussian_noise, clip_gradient, grad_norm},
        utils::{estimate_noise_multiplier, privacy_cost_per_step},
    };

    // -------------------------------------------------------------------------
    // PrivacyBudget Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_privacy_budget_new() {
        let budget = PrivacyBudget::new(8.0, 1e-5);
        assert_eq!(budget.epsilon, 8.0);
        assert_eq!(budget.delta, 1e-5);
    }

    #[test]
    fn test_privacy_budget_default() {
        let budget = PrivacyBudget::default();
        assert_eq!(budget.epsilon, 8.0);
        assert_eq!(budget.delta, 1e-5);
    }

    #[test]
    fn test_privacy_budget_allows() {
        let budget = PrivacyBudget::new(8.0, 1e-5);
        assert!(budget.allows(5.0));
        assert!(budget.allows(8.0));
        assert!(!budget.allows(10.0));
    }

    #[test]
    fn test_privacy_budget_remaining() {
        let budget = PrivacyBudget::new(8.0, 1e-5);
        assert!((budget.remaining(3.0) - 5.0).abs() < 1e-10);
        assert!((budget.remaining(10.0) - 0.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // RdpAccountant Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rdp_accountant_new() {
        let accountant = RdpAccountant::new();
        assert_eq!(accountant.n_steps(), 0);
        // Initially no privacy has been spent (RDP values are 0)
        assert_eq!(accountant.rdp.iter().sum::<f64>(), 0.0);
    }

    #[test]
    fn test_rdp_accountant_step() {
        let mut accountant = RdpAccountant::new();
        accountant.step(1.0, 0.01);
        assert_eq!(accountant.n_steps(), 1);

        let (epsilon, delta) = accountant.get_privacy_spent(1e-5);
        assert!(epsilon > 0.0);
        assert_eq!(delta, 1e-5);
    }

    #[test]
    fn test_rdp_accountant_accumulates() {
        let mut accountant = RdpAccountant::new();
        accountant.step(1.0, 0.01);
        let (e1, _) = accountant.get_privacy_spent(1e-5);

        accountant.step(1.0, 0.01);
        let (e2, _) = accountant.get_privacy_spent(1e-5);

        // Privacy loss should increase
        assert!(e2 > e1);
    }

    #[test]
    fn test_rdp_accountant_reset() {
        let mut accountant = RdpAccountant::new();
        accountant.step(1.0, 0.01);
        accountant.step(1.0, 0.01);
        assert_eq!(accountant.n_steps(), 2);

        accountant.reset();
        assert_eq!(accountant.n_steps(), 0);
    }

    #[test]
    fn test_rdp_higher_noise_lower_epsilon() {
        let mut acc1 = RdpAccountant::new();
        let mut acc2 = RdpAccountant::new();

        // Same number of steps
        for _ in 0..100 {
            acc1.step(1.0, 0.01); // Lower noise
            acc2.step(2.0, 0.01); // Higher noise
        }

        let (e1, _) = acc1.get_privacy_spent(1e-5);
        let (e2, _) = acc2.get_privacy_spent(1e-5);

        // Higher noise should give lower epsilon
        assert!(e2 < e1);
    }

    // -------------------------------------------------------------------------
    // Gradient Operations Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_clip_gradient_within_norm() {
        let grad = vec![0.3, 0.4, 0.0];
        let clipped = clip_gradient(&grad, 1.0);
        // Norm is 0.5, within limit
        assert!((clipped[0] - 0.3).abs() < 1e-10);
        assert!((clipped[1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_clip_gradient_exceeds_norm() {
        let grad = vec![3.0, 4.0, 0.0];
        let clipped = clip_gradient(&grad, 1.0);
        // Norm is 5.0, should be clipped to 1.0
        let norm = grad_norm(&clipped);
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clip_gradient_preserves_direction() {
        let grad = vec![3.0, 4.0, 0.0];
        let clipped = clip_gradient(&grad, 1.0);
        // Direction should be preserved
        let ratio = clipped[0] / clipped[1];
        assert!((ratio - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_grad_norm() {
        let grad = vec![3.0, 4.0];
        assert!((grad_norm(&grad) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_add_gaussian_noise() {
        let grad = vec![1.0, 2.0, 3.0];
        let mut rng = rand::rng();
        let noised = add_gaussian_noise(&grad, 0.1, &mut rng);

        // Should have same length
        assert_eq!(noised.len(), 3);

        // Should be different from original (with high probability)
        let diff: f64 = grad.iter().zip(noised.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0);
    }

    // -------------------------------------------------------------------------
    // DpSgdConfig Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dp_config_new() {
        let config = DpSgdConfig::new();
        assert!((config.max_grad_norm - 1.0).abs() < 1e-10);
        assert!((config.noise_multiplier - 1.1).abs() < 1e-10);
    }

    #[test]
    fn test_dp_config_builder() {
        let config = DpSgdConfig::new()
            .with_max_grad_norm(2.0)
            .with_noise_multiplier(1.5)
            .with_budget(PrivacyBudget::new(4.0, 1e-6))
            .with_sample_rate(0.05);

        assert!((config.max_grad_norm - 2.0).abs() < 1e-10);
        assert!((config.noise_multiplier - 1.5).abs() < 1e-10);
        assert!((config.budget.epsilon - 4.0).abs() < 1e-10);
        assert!((config.sample_rate - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_dp_config_validate_valid() {
        let config = DpSgdConfig::new();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_dp_config_validate_invalid_norm() {
        let config = DpSgdConfig::new().with_max_grad_norm(-1.0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dp_config_validate_invalid_epsilon() {
        let config = DpSgdConfig::new().with_budget(PrivacyBudget::new(-1.0, 1e-5));
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_dp_config_noise_std() {
        let config = DpSgdConfig::new().with_max_grad_norm(2.0).with_noise_multiplier(1.5);
        assert!((config.noise_std() - 3.0).abs() < 1e-10);
    }

    // -------------------------------------------------------------------------
    // DpSgd Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_dp_sgd_new() {
        let config = DpSgdConfig::new();
        let dp_sgd = DpSgd::new(0.01, config).expect("config should be valid");
        assert_eq!(dp_sgd.n_steps(), 0);
        assert!(!dp_sgd.is_budget_exhausted());
    }

    #[test]
    fn test_dp_sgd_privatize_gradients() {
        let config = DpSgdConfig::new().with_max_grad_norm(1.0).with_noise_multiplier(0.1);
        let mut dp_sgd = DpSgd::new(0.01, config).expect("config should be valid");

        let grads = vec![vec![0.1, 0.2, 0.3], vec![0.2, 0.3, 0.1], vec![0.3, 0.1, 0.2]];

        let result = dp_sgd.privatize_gradients(&grads);
        assert!(result.is_ok());

        let private_grad = result.expect("operation should succeed");
        assert_eq!(private_grad.len(), 3);
        assert_eq!(dp_sgd.n_steps(), 1);
    }

    #[test]
    fn test_dp_sgd_step() {
        let config = DpSgdConfig::new().with_max_grad_norm(1.0).with_noise_multiplier(0.1);
        let mut dp_sgd = DpSgd::new(0.1, config).expect("config should be valid");

        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![vec![0.1, 0.1, 0.1], vec![0.1, 0.1, 0.1]];

        let result = dp_sgd.step(&mut params, &grads);
        assert!(result.is_ok());

        // Params should have changed
        let diff: f64 = params.iter().zip(&[1.0, 2.0, 3.0]).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_dp_sgd_privacy_accumulates() {
        let config = DpSgdConfig::new();
        let mut dp_sgd = DpSgd::new(0.01, config).expect("config should be valid");

        let grads = vec![vec![0.1, 0.2, 0.3]];

        let (e1, _) = dp_sgd.privacy_spent();

        dp_sgd.privatize_gradients(&grads).expect("operation should succeed");
        let (e2, _) = dp_sgd.privacy_spent();

        dp_sgd.privatize_gradients(&grads).expect("operation should succeed");
        let (e3, _) = dp_sgd.privacy_spent();

        // Privacy loss should increase monotonically
        assert!(e3 > e2);
        assert!(e2 > e1);
    }

    #[test]
    fn test_dp_sgd_budget_exhaustion() {
        // Use very tight budget
        let config = DpSgdConfig::new()
            .with_budget(PrivacyBudget::new(0.1, 1e-5))
            .with_noise_multiplier(0.1)
            .with_strict_budget(true);
        let mut dp_sgd = DpSgd::new(0.01, config).expect("config should be valid");

        let grads = vec![vec![0.1, 0.2, 0.3]];

        // Run until budget exhausted
        let mut exhausted = false;
        for _ in 0..1000 {
            match dp_sgd.privatize_gradients(&grads) {
                Err(DpError::BudgetExhausted { .. }) => {
                    exhausted = true;
                    break;
                }
                Ok(_) => {}
                Err(e) => panic!("Unexpected error: {e:?}"),
            }
        }

        assert!(exhausted);
    }

    #[test]
    fn test_dp_sgd_reset() {
        let config = DpSgdConfig::new();
        let mut dp_sgd = DpSgd::new(0.01, config).expect("config should be valid");

        let grads = vec![vec![0.1, 0.2, 0.3]];
        dp_sgd.privatize_gradients(&grads).expect("operation should succeed");
        dp_sgd.privatize_gradients(&grads).expect("operation should succeed");

        assert!(dp_sgd.n_steps() > 0);

        dp_sgd.reset();
        assert_eq!(dp_sgd.n_steps(), 0);
    }

    // -------------------------------------------------------------------------
    // Utility Function Tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_estimate_noise_multiplier() {
        let noise = estimate_noise_multiplier(
            8.0,   // target epsilon
            1e-5,  // delta
            60000, // dataset size (e.g., MNIST)
            256,   // batch size
            10,    // epochs
        );

        // Should return a reasonable value
        assert!(noise > 0.0);
        assert!(noise < 100.0);
    }

    #[test]
    fn test_privacy_cost_per_step() {
        let cost = privacy_cost_per_step(1.0, 0.01, 1e-5);
        assert!(cost > 0.0);
        assert!(cost < 1.0); // Single step shouldn't use much budget
    }
}

// =============================================================================
// Property Tests
// =============================================================================

#[cfg(test)]
mod property_tests {
    use crate::optim::dp::{
        accountant::RdpAccountant,
        budget::PrivacyBudget,
        config::DpSgdConfig,
        dp_sgd::DpSgd,
        gradient::{clip_gradient, grad_norm},
        utils::privacy_cost_per_step,
    };
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn prop_clip_gradient_norm_bounded(
            grad in prop::collection::vec(-100.0f64..100.0, 1..50),
            max_norm in 0.1f64..10.0
        ) {
            let clipped = clip_gradient(&grad, max_norm);
            let norm = grad_norm(&clipped);
            prop_assert!(norm <= max_norm + 1e-10);
        }

        #[test]
        fn prop_privacy_budget_remaining_non_negative(
            epsilon in 0.1f64..10.0,
            delta in 1e-8f64..1e-3,
            spent in 0.0f64..20.0
        ) {
            let budget = PrivacyBudget::new(epsilon, delta);
            prop_assert!(budget.remaining(spent) >= 0.0);
        }

        #[test]
        fn prop_rdp_accountant_monotonic(
            noise_mult in 0.1f64..10.0,
            sample_rate in 0.001f64..0.1,
            steps in 1usize..50
        ) {
            let mut accountant = RdpAccountant::new();
            let mut prev_epsilon = 0.0f64;

            for _ in 0..steps {
                accountant.step(noise_mult, sample_rate);
                let (epsilon, _) = accountant.get_privacy_spent(1e-5);
                prop_assert!(epsilon >= prev_epsilon);
                prev_epsilon = epsilon;
            }
        }

        #[test]
        fn prop_higher_noise_lower_privacy_cost(
            noise1 in 0.1f64..5.0,
            sample_rate in 0.001f64..0.1
        ) {
            let noise2 = noise1 * 2.0; // Higher noise

            let cost1 = privacy_cost_per_step(noise1, sample_rate, 1e-5);
            let cost2 = privacy_cost_per_step(noise2, sample_rate, 1e-5);

            // Higher noise should have lower or equal privacy cost
            prop_assert!(cost2 <= cost1 + 0.01);
        }

        #[test]
        fn prop_gradient_privatization_preserves_dimension(
            n_samples in 1usize..10,
            dim in 1usize..100
        ) {
            let config = DpSgdConfig::new()
                .with_noise_multiplier(0.1)
                .with_strict_budget(false);
            let mut dp_sgd = DpSgd::new(0.01, config).expect("config should be valid");

            let grads: Vec<Vec<f64>> = (0..n_samples)
                .map(|_| vec![0.1; dim])
                .collect();

            let result = dp_sgd.privatize_gradients(&grads).expect("operation should succeed");
            prop_assert_eq!(result.len(), dim);
        }
    }
}
