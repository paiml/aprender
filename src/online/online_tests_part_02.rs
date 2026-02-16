use super::*;

#[test]
fn test_config_clone() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.5,
        decay: LearningRateDecay::Step { decay_rate: 0.2 },
        l2_reg: 0.01,
        momentum: 0.9,
        gradient_clip: Some(1.0),
    };
    let cloned = config.clone();
    assert_eq!(cloned.learning_rate, 0.5);
    assert_eq!(cloned.gradient_clip, Some(1.0));
}

#[test]
fn test_learning_rate_decay_debug() {
    let decay = LearningRateDecay::AdaGrad { epsilon: 1e-8 };
    let debug_str = format!("{:?}", decay);
    assert!(debug_str.contains("AdaGrad"));
}

#[test]
fn test_learning_rate_decay_clone() {
    let decay = LearningRateDecay::Step { decay_rate: 0.1 };
    let cloned = decay;
    match cloned {
        LearningRateDecay::Step { decay_rate } => {
            assert!((decay_rate - 0.1).abs() < f64::EPSILON);
        }
        _ => panic!("Expected Step variant"),
    }
}

#[test]
fn test_online_linear_debug() {
    let model = OnlineLinearRegression::new(3);
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("OnlineLinearRegression"));
}

#[test]
fn test_online_linear_clone() {
    let mut model = OnlineLinearRegression::new(2);
    model.partial_fit(&[1.0, 2.0], &[3.0], Some(0.1)).unwrap();
    let cloned = model.clone();
    assert_eq!(cloned.weights().len(), 2);
    assert_eq!(cloned.n_samples_seen(), 1);
}

#[test]
fn test_online_logistic_debug() {
    let model = OnlineLogisticRegression::new(3);
    let debug_str = format!("{:?}", model);
    assert!(debug_str.contains("OnlineLogisticRegression"));
}

#[test]
fn test_online_logistic_clone() {
    let mut model = OnlineLogisticRegression::new(2);
    model.partial_fit(&[1.0, 2.0], &[1.0], Some(0.1)).unwrap();
    let cloned = model.clone();
    assert_eq!(cloned.weights().len(), 2);
    assert_eq!(cloned.n_samples_seen(), 1);
}

#[test]
fn test_learning_rate_decay_eq() {
    assert_eq!(LearningRateDecay::Constant, LearningRateDecay::Constant);
    assert_ne!(LearningRateDecay::Constant, LearningRateDecay::Inverse);
    assert_eq!(
        LearningRateDecay::Step { decay_rate: 0.1 },
        LearningRateDecay::Step { decay_rate: 0.1 }
    );
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_online_linear_zero_features() {
    // Edge case: 0 features model
    let model = OnlineLinearRegression::new(0);
    assert_eq!(model.weights().len(), 0);
}

#[test]
fn test_online_logistic_zero_features() {
    // Edge case: 0 features model
    let model = OnlineLogisticRegression::new(0);
    assert_eq!(model.weights().len(), 0);
}

#[test]
fn test_online_linear_large_features() {
    let model = OnlineLinearRegression::new(1000);
    assert_eq!(model.weights().len(), 1000);
}

#[test]
fn test_online_logistic_large_features() {
    let model = OnlineLogisticRegression::new(1000);
    assert_eq!(model.weights().len(), 1000);
}

#[test]
fn test_online_linear_explicit_learning_rate() {
    let mut model = OnlineLinearRegression::new(1);
    // Use explicit learning rate instead of computed
    let loss = model.partial_fit(&[1.0], &[2.0], Some(0.5)).unwrap();
    assert!(loss.is_finite());
}

#[test]
fn test_online_logistic_explicit_learning_rate() {
    let mut model = OnlineLogisticRegression::new(1);
    let loss = model.partial_fit(&[1.0], &[1.0], Some(0.5)).unwrap();
    assert!(loss.is_finite());
}

#[test]
fn test_online_linear_multiple_mini_batches() {
    let mut model = OnlineLinearRegression::new(2);

    // First batch
    let x1 = vec![1.0, 0.0, 0.0, 1.0];
    let y1 = vec![2.0, 3.0];
    model.partial_fit(&x1, &y1, Some(0.1)).unwrap();
    assert_eq!(model.n_samples_seen(), 2);

    // Second batch
    let x2 = vec![1.0, 1.0, 2.0, 1.0];
    let y2 = vec![5.0, 7.0];
    model.partial_fit(&x2, &y2, Some(0.1)).unwrap();
    assert_eq!(model.n_samples_seen(), 4);
}

#[test]
fn test_online_logistic_multiple_mini_batches() {
    let mut model = OnlineLogisticRegression::new(2);

    // First batch
    let x1 = vec![0.0, 0.0, 1.0, 1.0];
    let y1 = vec![0.0, 1.0];
    model.partial_fit(&x1, &y1, Some(0.5)).unwrap();
    assert_eq!(model.n_samples_seen(), 2);

    // Second batch
    let x2 = vec![0.5, 0.5, 0.2, 0.8];
    let y2 = vec![0.0, 1.0];
    model.partial_fit(&x2, &y2, Some(0.5)).unwrap();
    assert_eq!(model.n_samples_seen(), 4);
}

#[test]
fn test_config_with_all_fields() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.001,
        decay: LearningRateDecay::AdaGrad { epsilon: 1e-6 },
        l2_reg: 0.01,
        momentum: 0.9,
        gradient_clip: Some(5.0),
    };

    assert_eq!(config.learning_rate, 0.001);
    assert_eq!(config.l2_reg, 0.01);
    assert_eq!(config.momentum, 0.9);
    assert_eq!(config.gradient_clip, Some(5.0));
}

#[test]
fn test_online_linear_initial_lr() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.5,
        decay: LearningRateDecay::InverseSqrt,
        ..Default::default()
    };
    let model = OnlineLinearRegression::with_config(1, config);

    // Before any samples, lr should be base / sqrt(1) = base
    let lr = model.current_learning_rate();
    assert!((lr - 0.5).abs() < 0.01);
}

#[test]
fn test_online_logistic_initial_lr() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.5,
        decay: LearningRateDecay::InverseSqrt,
        ..Default::default()
    };
    let model = OnlineLogisticRegression::with_config(1, config);

    let lr = model.current_learning_rate();
    assert!((lr - 0.5).abs() < 0.01);
}

#[test]
fn test_online_linear_bias_update() {
    let mut model = OnlineLinearRegression::new(1);

    // Fit some data - bias should change
    let initial_bias = model.bias();
    model.partial_fit(&[1.0], &[10.0], Some(0.5)).unwrap();
    let new_bias = model.bias();

    // Bias should have changed (moved toward target)
    assert_ne!(initial_bias, new_bias);
}

#[test]
fn test_online_logistic_bias_update() {
    let mut model = OnlineLogisticRegression::new(1);

    let initial_bias = model.bias();
    model.partial_fit(&[1.0], &[1.0], Some(0.5)).unwrap();
    let new_bias = model.bias();

    // Bias should have changed
    assert_ne!(initial_bias, new_bias);
}

#[test]
fn test_online_linear_predict_correct_calculation() {
    let mut model = OnlineLinearRegression::new(2);

    // Manually set weights for testing
    model.weights[0] = 2.0;
    model.weights[1] = 3.0;
    model.bias = 1.0;

    // Prediction: 2*1 + 3*2 + 1 = 2 + 6 + 1 = 9
    let pred = model.predict_one(&[1.0, 2.0]).unwrap();
    assert!((pred - 9.0).abs() < 1e-10);
}

#[test]
fn test_online_logistic_sigmoid_bounds() {
    // All predictions should be between 0 and 1
    for x in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        let p = OnlineLogisticRegression::sigmoid(x);
        assert!(p >= 0.0 && p <= 1.0, "sigmoid({}) = {} out of bounds", x, p);
    }
}

#[test]
fn test_online_linear_after_reset_can_train() {
    let mut model = OnlineLinearRegression::new(2);

    model.partial_fit(&[1.0, 1.0], &[5.0], Some(0.1)).unwrap();
    model.reset();

    // Should be able to train again after reset
    let loss = model.partial_fit(&[1.0, 1.0], &[5.0], Some(0.1)).unwrap();
    assert!(loss.is_finite());
    assert_eq!(model.n_samples_seen(), 1);
}

#[test]
fn test_online_logistic_after_reset_can_train() {
    let mut model = OnlineLogisticRegression::new(2);

    model.partial_fit(&[1.0, 1.0], &[1.0], Some(0.5)).unwrap();
    model.reset();

    let loss = model.partial_fit(&[1.0, 1.0], &[1.0], Some(0.5)).unwrap();
    assert!(loss.is_finite());
    assert_eq!(model.n_samples_seen(), 1);
}

#[test]
fn test_decay_adagrad_eq() {
    let d1 = LearningRateDecay::AdaGrad { epsilon: 1e-8 };
    let d2 = LearningRateDecay::AdaGrad { epsilon: 1e-8 };
    let d3 = LearningRateDecay::AdaGrad { epsilon: 1e-6 };
    assert_eq!(d1, d2);
    assert_ne!(d1, d3);
}

#[test]
fn test_learning_rate_decay_copy() {
    let decay = LearningRateDecay::Inverse;
    let copy = decay;
    assert_eq!(decay, copy);
}

#[test]
fn test_online_linear_with_constant_decay() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.1,
        decay: LearningRateDecay::Constant,
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(1, config);

    // Train many samples
    for _ in 0..100 {
        model.partial_fit(&[1.0], &[2.0], None).unwrap();
    }

    // Learning rate should remain constant
    let lr = model.current_learning_rate();
    assert!((lr - 0.1).abs() < 0.001);
}

#[test]
fn test_online_logistic_with_constant_decay() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.1,
        decay: LearningRateDecay::Constant,
        ..Default::default()
    };
    let mut model = OnlineLogisticRegression::with_config(1, config);

    for _ in 0..100 {
        model.partial_fit(&[1.0], &[1.0], None).unwrap();
    }

    let lr = model.current_learning_rate();
    assert!((lr - 0.1).abs() < 0.001);
}

#[test]
fn test_online_linear_negative_targets() {
    let mut model = OnlineLinearRegression::new(2);

    // Should handle negative targets
    let loss = model.partial_fit(&[1.0, 1.0], &[-5.0], Some(0.1)).unwrap();
    assert!(loss.is_finite());
}

#[test]
fn test_online_linear_large_gradients() {
    let mut model = OnlineLinearRegression::new(1);

    // Large input * large error = large gradient
    let loss = model
        .partial_fit(&[1000.0], &[1000000.0], Some(0.001))
        .unwrap();
    assert!(loss.is_finite());
    assert!(model.weights()[0].is_finite());
}

#[test]
fn test_online_logistic_boundary_labels() {
    let mut model = OnlineLogisticRegression::new(1);

    // Exact 0 and 1 labels
    model.partial_fit(&[1.0], &[0.0], Some(0.1)).unwrap();
    model.partial_fit(&[2.0], &[1.0], Some(0.1)).unwrap();

    // Probabilities should be bounded even with extreme labels
    let p1 = model.predict_proba_one(&[1.0]).unwrap();
    let p2 = model.predict_proba_one(&[2.0]).unwrap();
    assert!(p1 >= 0.0 && p1 <= 1.0);
    assert!(p2 >= 0.0 && p2 <= 1.0);
}

#[test]
fn test_online_linear_accum_grad_grows() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.1,
        decay: LearningRateDecay::AdaGrad { epsilon: 1e-8 },
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(1, config);

    let initial = model.accum_grad[0];

    for _ in 0..10 {
        model.partial_fit(&[1.0], &[10.0], None).unwrap();
    }

    // Accumulated gradient should grow
    assert!(model.accum_grad[0] > initial);
}

#[test]
fn test_online_logistic_accum_grad_grows() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.1,
        decay: LearningRateDecay::AdaGrad { epsilon: 1e-8 },
        ..Default::default()
    };
    let mut model = OnlineLogisticRegression::with_config(1, config);

    let initial = model.accum_grad[0];

    for _ in 0..10 {
        model.partial_fit(&[1.0], &[1.0], None).unwrap();
    }

    assert!(model.accum_grad[0] > initial);
}
