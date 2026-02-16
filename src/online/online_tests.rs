pub(crate) use super::*;

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

#[path = "online_tests_part_02.rs"]

mod online_tests_part_02;
