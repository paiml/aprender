use super::*;
use crate::ensemble::gating::SoftmaxGating;
use crate::linear_model::LinearRegression;

#[test]
fn test_moe_config_default() {
    let config = MoeConfig::default();
    assert_eq!(config.top_k, 1);
    assert!((config.capacity_factor - 1.0).abs() < 1e-6);
    assert!((config.expert_dropout - 0.0).abs() < 1e-6);
    assert!((config.load_balance_weight - 0.01).abs() < 1e-6);
}

#[test]
fn test_moe_config_builders() {
    let config = MoeConfig::default()
        .with_top_k(3)
        .with_capacity_factor(1.5)
        .with_expert_dropout(0.1)
        .with_load_balance_weight(0.05);

    assert_eq!(config.top_k, 3);
    assert!((config.capacity_factor - 1.5).abs() < 1e-6);
    assert!((config.expert_dropout - 0.1).abs() < 1e-6);
    assert!((config.load_balance_weight - 0.05).abs() < 1e-6);
}

#[test]
fn test_moe_config_clone() {
    let config = MoeConfig::default().with_top_k(2);
    let cloned = config.clone();
    assert_eq!(cloned.top_k, config.top_k);
}

#[test]
fn test_moe_config_debug() {
    let config = MoeConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("MoeConfig"));
}

#[test]
fn test_moe_builder_without_gating() {
    let expert = LinearRegression::new();
    let result = MixtureOfExperts::<LinearRegression, SoftmaxGating>::builder()
        .expert(expert)
        .build();
    assert!(result.is_err());
}

#[test]
fn test_moe_builder_basic() {
    let mut expert1 = LinearRegression::new();
    let mut expert2 = LinearRegression::new();

    // Use more samples than features for valid OLS
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert1.fit(&x, &y).expect("fit expert1");
    expert2.fit(&x, &y).expect("fit expert2");

    let gating = SoftmaxGating::new(1, 2);
    let moe = MixtureOfExperts::builder()
        .expert(expert1)
        .expert(expert2)
        .gating(gating)
        .build()
        .expect("build moe");

    assert_eq!(moe.n_experts(), 2);
}

#[test]
fn test_moe_predict() {
    let mut expert1 = LinearRegression::new();
    let mut expert2 = LinearRegression::new();

    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert1.fit(&x, &y).expect("fit expert1");
    expert2.fit(&x, &y).expect("fit expert2");

    let gating = SoftmaxGating::new(1, 2);
    let moe = MixtureOfExperts::builder()
        .expert(expert1)
        .expert(expert2)
        .gating(gating)
        .build()
        .expect("build moe");

    let input = vec![3.0];
    let prediction = moe.predict(&input);
    // Should produce some output
    assert!(prediction.is_finite());
}

#[test]
fn test_moe_predict_batch() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit expert");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    let inputs = Matrix::from_vec(2, 1, vec![1.0, 2.0]).expect("valid inputs");
    let predictions = moe.predict_batch(&inputs);
    assert_eq!(predictions.len(), 2);
}

#[test]
fn test_moe_config_getter() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let config = MoeConfig::default().with_top_k(2);
    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .config(config)
        .build()
        .expect("build moe");

    assert_eq!(moe.config().top_k, 2);
}

#[test]
fn test_moe_get_routing_weights() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    let weights = moe.get_routing_weights(&[1.0]);
    assert_eq!(weights.len(), 1); // 1 expert
}

#[test]
fn test_moe_load_balance_loss_empty() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    // Empty inputs
    let inputs = Matrix::from_vec(0, 1, vec![]).expect("empty matrix");
    let loss = moe.compute_load_balance_loss(&inputs);
    assert!((loss - 0.0).abs() < 1e-6);
}

#[test]
fn test_moe_expert_usage() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    let usage = moe.expert_usage(&x);
    assert_eq!(usage.len(), 1); // 1 expert
}

#[test]
fn test_moe_expert_usage_empty() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    // Empty inputs
    let empty = Matrix::from_vec(0, 1, vec![]).expect("empty matrix");
    let usage = moe.expert_usage(&empty);
    assert_eq!(usage.len(), 1);
    assert!((usage[0] - 0.0).abs() < 1e-6);
}

#[test]
fn test_moe_fit() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let mut moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    // Fit does nothing in simple implementation
    assert!(moe.fit(&x, &y).is_ok());
}

#[test]
fn test_moe_debug() {
    let mut expert = LinearRegression::new();
    let x = Matrix::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("valid matrix");
    let y = Vector::from_slice(&[2.0, 4.0, 6.0, 8.0, 10.0]);
    expert.fit(&x, &y).expect("fit");

    let gating = SoftmaxGating::new(1, 1);
    let moe = MixtureOfExperts::builder()
        .expert(expert)
        .gating(gating)
        .build()
        .expect("build moe");

    let debug_str = format!("{:?}", moe);
    assert!(debug_str.contains("MixtureOfExperts"));
}

#[test]
fn test_moe_builder_debug() {
    let builder = MixtureOfExperts::<LinearRegression, SoftmaxGating>::builder();
    let debug_str = format!("{:?}", builder);
    assert!(debug_str.contains("MoeBuilder"));
}
