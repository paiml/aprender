#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::optim::Optimizer;
    use crate::primitives::Vector;

    #[test]
    fn test_sgd_basic_update() {
        let mut optimizer = SGD::new(0.1);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.5, 1.0]);

        optimizer.step(&mut params, &gradients);

        assert!((params[0] - 0.95).abs() < 1e-6);
        assert!((params[1] - 1.9).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let mut optimizer = SGD::new(0.1).with_momentum(0.9);
        let mut params = Vector::from_slice(&[0.0]);
        let gradients = Vector::from_slice(&[1.0]);

        // First step: v = 0.9*0 + 0.1*1 = 0.1, params = 0 - 0.1 = -0.1
        optimizer.step(&mut params, &gradients);
        assert!((params[0] - (-0.1)).abs() < 1e-6);

        // Second step: v = 0.9*0.1 + 0.1*1 = 0.19, params = -0.1 - 0.19 = -0.29
        optimizer.step(&mut params, &gradients);
        assert!((params[0] - (-0.29)).abs() < 1e-6);
    }

    #[test]
    fn test_adam_basic_update() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.1, 0.2]);

        let original = params.clone();
        optimizer.step(&mut params, &gradients);

        // Parameters should have decreased (gradients are positive)
        assert!(params[0] < original[0]);
        assert!(params[1] < original[1]);
        assert_eq!(optimizer.steps(), 1);
    }

    #[test]
    fn test_adam_reset() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[1.0]);

        optimizer.step(&mut params, &gradients);
        assert_eq!(optimizer.steps(), 1);

        optimizer.reset();
        assert_eq!(optimizer.steps(), 0);
    }

    // ================================================================
    // Additional coverage tests for missed branches
    // ================================================================

    #[test]
    fn test_sgd_reset() {
        let mut optimizer = SGD::new(0.1).with_momentum(0.9);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.5, 0.5]);

        optimizer.step(&mut params, &gradients);
        // Velocity is now initialized
        assert!(optimizer.velocity.is_some());

        optimizer.reset();
        assert!(optimizer.velocity.is_none());
    }

    #[test]
    fn test_sgd_has_momentum() {
        let sgd_no_momentum = SGD::new(0.1);
        assert!(!sgd_no_momentum.has_momentum());

        let sgd_with_momentum = SGD::new(0.1).with_momentum(0.9);
        assert!(sgd_with_momentum.has_momentum());

        let sgd_zero_momentum = SGD::new(0.1).with_momentum(0.0);
        assert!(!sgd_zero_momentum.has_momentum());
    }

    #[test]
    fn test_sgd_learning_rate_accessor() {
        let sgd = SGD::new(0.05);
        assert!((sgd.learning_rate() - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum_accessor() {
        let sgd = SGD::new(0.1).with_momentum(0.99);
        assert!((sgd.momentum() - 0.99).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_no_momentum_multi_step() {
        let mut optimizer = SGD::new(0.1);
        let mut params = Vector::from_slice(&[5.0]);
        let gradients = Vector::from_slice(&[1.0]);

        // Step 1: 5.0 - 0.1*1.0 = 4.9
        optimizer.step(&mut params, &gradients);
        assert!((params[0] - 4.9).abs() < 1e-6);

        // Step 2: 4.9 - 0.1*1.0 = 4.8
        optimizer.step(&mut params, &gradients);
        assert!((params[0] - 4.8).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_momentum_velocity_reinit_on_size_change() {
        let mut optimizer = SGD::new(0.1).with_momentum(0.9);

        // Step with 2 params
        let mut params2 = Vector::from_slice(&[1.0, 2.0]);
        let grads2 = Vector::from_slice(&[0.1, 0.2]);
        optimizer.step(&mut params2, &grads2);

        // Step with 3 params (different size triggers velocity reinitialization)
        let mut params3 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let grads3 = Vector::from_slice(&[0.1, 0.2, 0.3]);
        optimizer.step(&mut params3, &grads3);

        // Should not panic; velocity was reinitialized
        assert!(optimizer.velocity.is_some());
    }

    #[test]
    fn test_adam_custom_hyperparameters() {
        let optimizer = Adam::new(0.0001)
            .with_beta1(0.95)
            .with_beta2(0.9999)
            .with_epsilon(1e-7);

        assert!((optimizer.learning_rate() - 0.0001).abs() < 1e-9);
        assert!((optimizer.beta1() - 0.95).abs() < 1e-9);
        assert!((optimizer.beta2() - 0.9999).abs() < 1e-9);
        assert!((optimizer.epsilon() - 1e-7).abs() < 1e-15);
    }

    #[test]
    fn test_adam_multiple_steps() {
        let mut optimizer = Adam::new(0.01);
        let mut params = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let gradients = Vector::from_slice(&[0.1, 0.2, 0.3]);

        for _ in 0..10 {
            optimizer.step(&mut params, &gradients);
        }

        assert_eq!(optimizer.steps(), 10);
        // Params should have decreased
        assert!(params[0] < 1.0);
        assert!(params[1] < 2.0);
        assert!(params[2] < 3.0);
    }

    #[test]
    fn test_adam_reset_clears_moments() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0]);
        let gradients = Vector::from_slice(&[1.0]);

        optimizer.step(&mut params, &gradients);
        assert!(optimizer.m.is_some());
        assert!(optimizer.v.is_some());
        assert_eq!(optimizer.t, 1);

        optimizer.reset();
        assert!(optimizer.m.is_none());
        assert!(optimizer.v.is_none());
        assert_eq!(optimizer.t, 0);
    }

    #[test]
    fn test_adam_size_change_reinit() {
        let mut optimizer = Adam::new(0.001);

        // Step with 2 params
        let mut params2 = Vector::from_slice(&[1.0, 2.0]);
        let grads2 = Vector::from_slice(&[0.1, 0.2]);
        optimizer.step(&mut params2, &grads2);
        assert_eq!(optimizer.steps(), 1);

        // Step with 3 params (triggers moment reinitialization)
        let mut params3 = Vector::from_slice(&[1.0, 2.0, 3.0]);
        let grads3 = Vector::from_slice(&[0.1, 0.2, 0.3]);
        optimizer.step(&mut params3, &grads3);

        // Step counter resets to 1 after reinit
        assert_eq!(optimizer.steps(), 1);
    }

    #[test]
    fn test_sgd_optimizer_trait() {
        let mut optimizer = SGD::new(0.1);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.5, 1.0]);

        // Use Optimizer trait methods
        Optimizer::step(&mut optimizer, &mut params, &gradients);
        assert!((params[0] - 0.95).abs() < 1e-6);

        Optimizer::reset(&mut optimizer);
        // After reset, velocity is None
        assert!(optimizer.velocity.is_none());
    }

    #[test]
    fn test_adam_optimizer_trait() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.1, 0.2]);

        // Use Optimizer trait methods
        Optimizer::step(&mut optimizer, &mut params, &gradients);
        assert_eq!(optimizer.steps(), 1);

        Optimizer::reset(&mut optimizer);
        assert_eq!(optimizer.steps(), 0);
    }

    #[test]
    fn test_sgd_clone() {
        let optimizer = SGD::new(0.05).with_momentum(0.9);
        let cloned = optimizer.clone();

        assert!((cloned.learning_rate() - 0.05).abs() < 1e-6);
        assert!((cloned.momentum() - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_adam_clone() {
        let optimizer = Adam::new(0.001).with_beta1(0.95).with_beta2(0.9999);
        let cloned = optimizer.clone();

        assert!((cloned.learning_rate() - 0.001).abs() < 1e-9);
        assert!((cloned.beta1() - 0.95).abs() < 1e-9);
        assert!((cloned.beta2() - 0.9999).abs() < 1e-9);
    }

    #[test]
    fn test_sgd_debug() {
        let optimizer = SGD::new(0.1);
        let debug_str = format!("{:?}", optimizer);
        assert!(debug_str.contains("SGD"));
        assert!(debug_str.contains("learning_rate"));
    }

    #[test]
    fn test_adam_debug() {
        let optimizer = Adam::new(0.001);
        let debug_str = format!("{:?}", optimizer);
        assert!(debug_str.contains("Adam"));
        assert!(debug_str.contains("learning_rate"));
    }

    #[test]
    fn test_adam_zero_gradient() {
        let mut optimizer = Adam::new(0.001);
        let mut params = Vector::from_slice(&[1.0, 2.0]);
        let gradients = Vector::from_slice(&[0.0, 0.0]);

        let original = params.clone();
        optimizer.step(&mut params, &gradients);

        // With zero gradients, params should barely change (only due to bias correction of zeros)
        assert!((params[0] - original[0]).abs() < 1e-4);
        assert!((params[1] - original[1]).abs() < 1e-4);
    }
}
