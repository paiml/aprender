use super::*;

#[test]
fn test_online_linear_regression_basic() {
    let mut model = OnlineLinearRegression::new(2);

    // Simple linear: y = 2*x1 + 3*x2
    let samples = vec![
        (vec![1.0, 0.0], 2.0),
        (vec![0.0, 1.0], 3.0),
        (vec![1.0, 1.0], 5.0),
        (vec![2.0, 1.0], 7.0),
    ];

    // Train incrementally
    for (x, y) in &samples {
        let loss = model.partial_fit(x, &[*y], Some(0.1)).unwrap();
        assert!(loss.is_finite());
    }

    assert!(model.n_samples_seen() == 4);
}

#[test]
fn test_online_linear_regression_convergence() {
    // y = 3*x + 1
    let config = OnlineLearnerConfig {
        learning_rate: 0.1,
        decay: LearningRateDecay::Constant,
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(1, config);

    // Multiple passes to converge
    for _ in 0..100 {
        model.partial_fit(&[1.0], &[4.0], None).unwrap();
        model.partial_fit(&[2.0], &[7.0], None).unwrap();
        model.partial_fit(&[3.0], &[10.0], None).unwrap();
    }

    // Check predictions
    let pred1 = model.predict_one(&[1.0]).unwrap();
    let pred2 = model.predict_one(&[4.0]).unwrap();

    assert!((pred1 - 4.0).abs() < 0.5, "pred1={}", pred1);
    assert!((pred2 - 13.0).abs() < 1.0, "pred2={}", pred2);
}

#[test]
fn test_online_linear_regression_mini_batch() {
    let mut model = OnlineLinearRegression::new(2);

    // Mini-batch of 3 samples
    let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let y = vec![2.0, 3.0, 5.0];

    let loss = model.partial_fit(&x, &y, Some(0.1)).unwrap();
    assert!(loss.is_finite());
    assert_eq!(model.n_samples_seen(), 3);
}

#[test]
fn test_online_linear_regression_dimension_mismatch() {
    let mut model = OnlineLinearRegression::new(2);

    // Wrong number of features
    let result = model.partial_fit(&[1.0, 2.0, 3.0], &[1.0], None);
    assert!(result.is_err());

    // Wrong y length
    let result = model.partial_fit(&[1.0, 2.0], &[1.0, 2.0], None);
    assert!(result.is_err());
}

#[test]
fn test_online_linear_regression_reset() {
    let mut model = OnlineLinearRegression::new(2);
    model.partial_fit(&[1.0, 1.0], &[5.0], Some(0.1)).unwrap();

    assert!(model.n_samples_seen() > 0);
    model.reset();
    assert_eq!(model.n_samples_seen(), 0);
    assert_eq!(model.weights(), &[0.0, 0.0]);
}

#[test]
fn test_online_logistic_regression_basic() {
    let mut model = OnlineLogisticRegression::new(2);

    // Binary classification
    let samples = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 0.0),
        (vec![1.0, 0.0], 0.0),
        (vec![1.0, 1.0], 1.0),
    ];

    for (x, y) in &samples {
        let loss = model.partial_fit(x, &[*y], Some(0.5)).unwrap();
        assert!(loss.is_finite());
    }

    assert_eq!(model.n_samples_seen(), 4);
}

#[test]
fn test_online_logistic_regression_convergence() {
    let config = OnlineLearnerConfig {
        learning_rate: 1.0,
        decay: LearningRateDecay::Constant,
        ..Default::default()
    };
    let mut model = OnlineLogisticRegression::with_config(2, config);

    // XOR-like data (won't fully converge but should show learning)
    for _ in 0..200 {
        model.partial_fit(&[0.0, 0.0], &[0.0], None).unwrap();
        model.partial_fit(&[1.0, 1.0], &[1.0], None).unwrap();
    }

    // Should be biased toward 1 for [1,1]
    let p00 = model.predict_proba_one(&[0.0, 0.0]).unwrap();
    let p11 = model.predict_proba_one(&[1.0, 1.0]).unwrap();

    assert!(p00 < 0.5, "p00={}", p00);
    assert!(p11 > 0.5, "p11={}", p11);
}

#[test]
fn test_learning_rate_decay() {
    let config = OnlineLearnerConfig {
        learning_rate: 1.0,
        decay: LearningRateDecay::InverseSqrt,
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(1, config);

    // Train some samples
    for _ in 0..100 {
        model.partial_fit(&[1.0], &[1.0], None).unwrap();
    }

    // Learning rate should have decayed
    let lr = model.current_learning_rate();
    assert!(lr < 1.0, "lr should decay, got {}", lr);
    assert!(lr > 0.05, "lr should not decay too much, got {}", lr);
}

#[test]
fn test_adagrad_decay() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.5,
        decay: LearningRateDecay::AdaGrad { epsilon: 1e-8 },
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(1, config);

    // Train with consistent gradients
    for _ in 0..50 {
        model.partial_fit(&[1.0], &[2.0], None).unwrap();
    }

    // With AdaGrad, accumulated gradients should grow
    assert!(model.accum_grad[0] > 1e-8);
}

#[test]
fn test_gradient_clipping() {
    let config = OnlineLearnerConfig {
        learning_rate: 1.0,
        decay: LearningRateDecay::Constant,
        gradient_clip: Some(0.1),
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(1, config);

    // Large target should produce large gradient, but clipping limits update
    model.partial_fit(&[1.0], &[1000.0], None).unwrap();

    // Weight should be bounded by clipping
    assert!(model.weights()[0].abs() < 1.0);
}

#[test]
fn test_l2_regularization() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.1,
        decay: LearningRateDecay::Constant,
        l2_reg: 0.1,
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(1, config);

    // Without regularization, weights would grow larger
    for _ in 0..100 {
        model.partial_fit(&[1.0], &[10.0], None).unwrap();
    }

    // With L2 reg, weights should be somewhat constrained
    let w_with_reg = model.weights()[0];

    let config_no_reg = OnlineLearnerConfig {
        learning_rate: 0.1,
        decay: LearningRateDecay::Constant,
        l2_reg: 0.0,
        ..Default::default()
    };
    let mut model_no_reg = OnlineLinearRegression::with_config(1, config_no_reg);

    for _ in 0..100 {
        model_no_reg.partial_fit(&[1.0], &[10.0], None).unwrap();
    }

    // Weight without reg should be at least as large
    assert!(model_no_reg.weights()[0].abs() >= w_with_reg.abs() * 0.9);
}

#[test]
fn test_empty_input_error() {
    let mut model = OnlineLinearRegression::new(2);

    let result = model.partial_fit(&[], &[1.0], None);
    assert!(result.is_err());

    let result = model.partial_fit(&[1.0, 2.0], &[], None);
    assert!(result.is_err());
}

#[test]
fn test_supports_warm_start() {
    let model = OnlineLinearRegression::new(2);
    assert!(model.supports_warm_start());

    let model = OnlineLogisticRegression::new(2);
    assert!(model.supports_warm_start());
}

#[test]
fn test_default_config() {
    let config = OnlineLearnerConfig::default();
    assert_eq!(config.learning_rate, 0.01);
    assert_eq!(config.decay, LearningRateDecay::InverseSqrt);
    assert_eq!(config.l2_reg, 0.0);
    assert_eq!(config.momentum, 0.0);
    assert!(config.gradient_clip.is_none());
}

// =========================================================================
// Extended coverage tests
// =========================================================================

#[test]
fn test_learning_rate_decay_inverse() {
    let config = OnlineLearnerConfig {
        learning_rate: 1.0,
        decay: LearningRateDecay::Inverse,
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(1, config);

    // Train to increase sample count
    for _ in 0..10 {
        model.partial_fit(&[1.0], &[1.0], None).unwrap();
    }

    // lr = 1.0 / t, at t=10, lr = 0.1
    let lr = model.current_learning_rate();
    assert!(
        lr < 0.2,
        "Inverse decay should reduce lr to ~0.1, got {}",
        lr
    );
}

#[test]
fn test_learning_rate_decay_step() {
    let config = OnlineLearnerConfig {
        learning_rate: 1.0,
        decay: LearningRateDecay::Step { decay_rate: 0.1 },
        ..Default::default()
    };
    let mut model = OnlineLinearRegression::with_config(1, config);

    // Train to increase sample count
    for _ in 0..10 {
        model.partial_fit(&[1.0], &[1.0], None).unwrap();
    }

    // lr = 1.0 / (1 + 0.1 * 10) = 1.0 / 2 = 0.5
    let lr = model.current_learning_rate();
    assert!(
        lr < 0.6 && lr > 0.4,
        "Step decay should reduce lr to ~0.5, got {}",
        lr
    );
}

#[test]
fn test_online_logistic_reset() {
    let mut model = OnlineLogisticRegression::new(2);
    model.partial_fit(&[1.0, 1.0], &[1.0], Some(0.5)).unwrap();

    assert!(model.n_samples_seen() > 0);
    assert!(model.weights().iter().any(|&w| w != 0.0) || model.bias() != 0.0);

    model.reset();
    assert_eq!(model.n_samples_seen(), 0);
    assert_eq!(model.weights(), &[0.0, 0.0]);
    assert_eq!(model.bias(), 0.0);
}

#[test]
fn test_predict_one_dimension_mismatch() {
    let model = OnlineLinearRegression::new(3);
    let result = model.predict_one(&[1.0, 2.0]); // Only 2 features, expected 3
    assert!(result.is_err());
}

#[test]
fn test_predict_proba_one_dimension_mismatch() {
    let model = OnlineLogisticRegression::new(3);
    let result = model.predict_proba_one(&[1.0, 2.0]); // Only 2 features, expected 3
    assert!(result.is_err());
}

#[test]
fn test_online_logistic_empty_input() {
    let mut model = OnlineLogisticRegression::new(2);

    let result = model.partial_fit(&[], &[1.0], None);
    assert!(result.is_err());

    let result = model.partial_fit(&[1.0, 2.0], &[], None);
    assert!(result.is_err());
}

#[test]
fn test_online_logistic_dimension_mismatch() {
    let mut model = OnlineLogisticRegression::new(2);

    // Wrong number of features
    let result = model.partial_fit(&[1.0, 2.0, 3.0], &[1.0], None);
    assert!(result.is_err());

    // Wrong y length
    let result = model.partial_fit(&[1.0, 2.0], &[1.0, 2.0], None);
    assert!(result.is_err());
}

#[test]
fn test_online_logistic_adagrad() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.5,
        decay: LearningRateDecay::AdaGrad { epsilon: 1e-8 },
        ..Default::default()
    };
    let mut model = OnlineLogisticRegression::with_config(1, config);

    // Train with consistent gradients
    for _ in 0..50 {
        model.partial_fit(&[1.0], &[1.0], None).unwrap();
    }

    // With AdaGrad, accumulated gradients should grow
    assert!(model.accum_grad[0] > 1e-8);
}

#[test]
fn test_online_logistic_gradient_clipping() {
    let config = OnlineLearnerConfig {
        learning_rate: 1.0,
        decay: LearningRateDecay::Constant,
        gradient_clip: Some(0.1),
        ..Default::default()
    };
    let mut model = OnlineLogisticRegression::with_config(1, config);

    // Large target should produce gradient, but clipping limits update
    model.partial_fit(&[10.0], &[1.0], None).unwrap();

    // Weight should be bounded
    assert!(model.weights()[0].abs() < 2.0);
}

#[test]
fn test_online_logistic_l2_reg() {
    let config = OnlineLearnerConfig {
        learning_rate: 0.5,
        decay: LearningRateDecay::Constant,
        l2_reg: 0.1,
        ..Default::default()
    };
    let mut model = OnlineLogisticRegression::with_config(1, config);

    // Train
    for _ in 0..100 {
        model.partial_fit(&[1.0], &[1.0], None).unwrap();
    }

    // L2 reg should constrain weights
    assert!(model.weights()[0].is_finite());
}

#[test]
fn test_online_logistic_inverse_decay() {
    let config = OnlineLearnerConfig {
        learning_rate: 1.0,
        decay: LearningRateDecay::Inverse,
        ..Default::default()
    };
    let mut model = OnlineLogisticRegression::with_config(1, config);

    for _ in 0..10 {
        model.partial_fit(&[1.0], &[1.0], None).unwrap();
    }

    let lr = model.current_learning_rate();
    assert!(lr < 0.2, "Inverse decay should reduce lr, got {}", lr);
}

#[test]
fn test_online_logistic_step_decay() {
    let config = OnlineLearnerConfig {
        learning_rate: 1.0,
        decay: LearningRateDecay::Step { decay_rate: 0.1 },
        ..Default::default()
    };
    let mut model = OnlineLogisticRegression::with_config(1, config);

    for _ in 0..10 {
        model.partial_fit(&[1.0], &[1.0], None).unwrap();
    }

    let lr = model.current_learning_rate();
    assert!(lr < 0.6, "Step decay should reduce lr, got {}", lr);
}

#[test]
fn test_learning_rate_decay_default() {
    let decay: LearningRateDecay = Default::default();
    assert_eq!(decay, LearningRateDecay::InverseSqrt);
}

#[test]
fn test_config_debug() {
    let config = OnlineLearnerConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("OnlineLearnerConfig"));
}

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
