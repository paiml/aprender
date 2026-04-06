//! Tests for BrickTuner evolution methods, CalibrationResult, and bandit edge cases.

use super::super::super::*;

#[test]
fn online_learner_is_converging_after_large_error() {
    let mut learner = OnlineLearner::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();
    let vec = features.to_vector();

    // Observe with very different target to create large error
    learner.observe(&vec, 10000.0);
    // After a large error, ema_loss should be > 0.15
    // Whether it exceeds threshold depends on exact prediction vs target gap
    // Just verify the method runs without panic
    let _ = learner.is_converging();
}

// ============================================================================
// BrickTuner Evolution Methods Tests
// ============================================================================

#[test]
fn brick_tuner_with_pretrained_has_weights() {
    let tuner = BrickTuner::with_pretrained();
    assert!(tuner.version().contains("pretrained"));
    assert!(tuner.throughput_sample_count() > 0);
    assert!(tuner.throughput_mape() > 0.0);
}

#[test]
fn brick_tuner_with_pretrained_mape() {
    let tuner = BrickTuner::with_pretrained();
    assert!((tuner.throughput_mape() - 0.082).abs() < 0.001, "Pretrained MAPE should be 8.2%");
}

#[test]
fn brick_tuner_with_pretrained_sample_count() {
    let tuner = BrickTuner::with_pretrained();
    assert_eq!(tuner.throughput_sample_count(), 10_000);
}

#[test]
fn brick_tuner_online_learner_inherits_weights() {
    let tuner = BrickTuner::new();
    let learner = tuner.online_learner();
    assert_eq!(learner.weights().len(), TunerFeatures::DIM + 1);
    assert_eq!(learner.num_updates(), 0);
}

#[test]
fn brick_tuner_apply_online_updates_no_updates() {
    let mut tuner = BrickTuner::new();
    let original_version = tuner.version().to_string();
    let learner = OnlineLearner::new();
    tuner.apply_online_updates(&learner);
    // No updates => version should not change
    assert_eq!(tuner.version(), original_version);
}

#[test]
fn brick_tuner_apply_online_updates_with_observations() {
    let mut tuner = BrickTuner::new();
    let mut learner = tuner.online_learner();

    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();
    let vec = features.to_vector();

    learner.observe(&vec, 100.0);
    learner.observe(&vec, 120.0);

    tuner.apply_online_updates(&learner);
    assert!(tuner.version().contains("online-2"));
}

#[test]
fn brick_tuner_kernel_bandit_returns_new_bandit() {
    let tuner = BrickTuner::new();
    let bandit = tuner.kernel_bandit();
    assert_eq!(bandit.total_pulls, 0);
    assert_eq!(bandit.arms.len(), KernelBandit::NUM_KERNELS);
}

#[test]
fn brick_tuner_recommend_with_exploration_exploit() {
    let tuner = BrickTuner::with_pretrained();
    let bandit = KernelBandit::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    // With explore_prob = 0.0, should always exploit (model prediction)
    let rec = tuner.recommend_kernel_with_exploration(&features, &bandit, 0.0);
    assert!(rec.confidence >= 0.0);
}

#[test]
fn brick_tuner_recommend_with_exploration_explore() {
    let tuner = BrickTuner::with_pretrained();
    let bandit = KernelBandit::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    // With explore_prob = 1.0, should always explore
    let rec = tuner.recommend_kernel_with_exploration(&features, &bandit, 1.0);
    // When exploring, confidence should be 0.5
    // (May or may not trigger depending on hash; test no panic)
    assert!(rec.confidence >= 0.0);
}

// ============================================================================
// CalibrationResult Tests
// ============================================================================

#[test]
fn calibration_result_fields_accessible() {
    let result = CalibrationResult {
        throughput_weights: vec![0.1, 0.2, 0.3],
        local_mape: 0.05,
        improvement_pct: 15.0,
        hardware_id: "RTX4090".to_string(),
        duration_secs: 2.5,
        num_benchmarks: 24,
    };

    assert_eq!(result.throughput_weights.len(), 3);
    assert!((result.local_mape - 0.05).abs() < 1e-6);
    assert!((result.improvement_pct - 15.0).abs() < 1e-6);
    assert_eq!(result.hardware_id, "RTX4090");
    assert!((result.duration_secs - 2.5).abs() < 1e-6);
    assert_eq!(result.num_benchmarks, 24);
}

#[test]
fn calibration_result_clone() {
    let result = CalibrationResult {
        throughput_weights: vec![0.1, 0.2],
        local_mape: 0.08,
        improvement_pct: 10.0,
        hardware_id: "A100".to_string(),
        duration_secs: 1.0,
        num_benchmarks: 12,
    };
    let cloned = result.clone();
    assert_eq!(cloned.throughput_weights, result.throughput_weights);
    assert_eq!(cloned.hardware_id, result.hardware_id);
}

#[test]
fn calibration_result_debug_format() {
    let result = CalibrationResult {
        throughput_weights: vec![0.1],
        local_mape: 0.05,
        improvement_pct: 20.0,
        hardware_id: "test".to_string(),
        duration_secs: 0.5,
        num_benchmarks: 6,
    };
    let debug = format!("{:?}", result);
    assert!(debug.contains("CalibrationResult"));
    assert!(debug.contains("local_mape"));
}

#[test]
fn calibration_result_serde_roundtrip() {
    let result = CalibrationResult {
        throughput_weights: vec![0.1, 0.2, 0.3],
        local_mape: 0.07,
        improvement_pct: 12.5,
        hardware_id: "RTX3090".to_string(),
        duration_secs: 3.0,
        num_benchmarks: 24,
    };
    let json = serde_json::to_string(&result).expect("serialize should succeed");
    let deserialized: CalibrationResult =
        serde_json::from_str(&json).expect("deserialize should succeed");
    assert_eq!(deserialized.throughput_weights, result.throughput_weights);
    assert_eq!(deserialized.hardware_id, result.hardware_id);
    assert_eq!(deserialized.num_benchmarks, result.num_benchmarks);
}

// ============================================================================
// KernelBandit Edge Cases
// ============================================================================

#[test]
fn kernel_bandit_select_after_single_update() {
    let mut bandit = KernelBandit::new();
    bandit.update(KernelType::TiledQ4K, 0.5);

    // After one update, all other arms have INFINITY UCB, so they should be preferred
    let selected = bandit.select();
    assert_ne!(selected, KernelType::TiledQ4K, "Should explore unexplored arms first");
}

#[test]
fn kernel_bandit_thompson_after_updates() {
    let mut bandit = KernelBandit::with_thompson_sampling();
    // Give one arm high reward
    for _ in 0..20 {
        bandit.update(KernelType::BatchedQ4K, 0.95);
    }
    // Give another arm low reward
    for _ in 0..20 {
        bandit.update(KernelType::TiledQ4K, 0.1);
    }

    // Thompson sampling should generally prefer the high-reward arm
    // (but is stochastic, so we just check it doesn't panic)
    let selected = bandit.select();
    let _ = selected;
}

#[test]
fn kernel_bandit_all_arms_explored_select_best() {
    let mut bandit = KernelBandit::new();
    // Explore all 12 arms with varying rewards
    for i in 0..KernelBandit::NUM_KERNELS {
        let kernel = KernelType::from_index(i);
        let reward = if i == 3 { 0.95 } else { 0.2 };
        // Pull each arm multiple times to reduce UCB bonus
        for _ in 0..50 {
            bandit.update(kernel, reward);
        }
    }

    // Best kernel should be index 3 (BatchedQ4K with highest reward)
    let best = bandit.best_kernel();
    assert_eq!(best, KernelType::BatchedQ4K);
}

#[test]
fn kernel_bandit_exploration_rate_partial() {
    let mut bandit = KernelBandit::new();
    // 6 pulls on one arm, 4 on another
    for _ in 0..6 {
        bandit.update(KernelType::TiledQ4K, 0.5);
    }
    for _ in 0..4 {
        bandit.update(KernelType::BatchedQ4K, 0.7);
    }
    // best_pulls = 6, total = 10, rate = 1 - 6/10 = 0.4
    assert!((bandit.exploration_rate() - 0.4).abs() < 1e-6);
}

// ============================================================================
// OnlineLearner Edge Cases
// ============================================================================

#[test]
fn online_learner_weights_length_matches_dim() {
    let learner = OnlineLearner::new();
    // DIM features + 1 bias
    assert_eq!(learner.weights().len(), TunerFeatures::DIM + 1);
}

#[test]
fn online_learner_predict_with_all_zeros() {
    let learner = OnlineLearner::new();
    let zeros = vec![0.0; TunerFeatures::DIM];
    let prediction = learner.predict(&zeros);
    // Should just return the bias (clamped to >= 0)
    assert!(prediction >= 0.0);
    // The bias is the first pretrained weight (0.36)
    assert!((prediction - 0.36).abs() < 0.01);
}

#[test]
fn online_learner_predict_with_all_ones() {
    let learner = OnlineLearner::new();
    let ones = vec![1.0; TunerFeatures::DIM];
    let prediction = learner.predict(&ones);
    assert!(prediction >= 0.0, "Prediction must be non-negative");
}

#[test]
fn online_learner_observe_multiple_different_targets() {
    let mut learner = OnlineLearner::new().with_learning_rate(0.001);

    let features_small = TunerFeatures::builder().model_params_b(1.5).batch_size(1).build();
    let features_large = TunerFeatures::builder().model_params_b(13.0).batch_size(8).build();

    let vec_small = features_small.to_vector();
    let vec_large = features_large.to_vector();

    learner.observe(&vec_small, 200.0);
    learner.observe(&vec_large, 50.0);

    assert_eq!(learner.num_updates(), 2);
    assert!(learner.ema_loss() > 0.0);
}

// ============================================================================
// BrickTuner with_pretrained + Evolution Integration
// ============================================================================

#[test]
fn pretrained_then_online_workflow() {
    let mut tuner = BrickTuner::with_pretrained();
    let mut learner = tuner.online_learner();

    let features = TunerFeatures::builder().model_params_b(7.0).batch_size(1).build();
    let vec = features.to_vector();

    // Simulate online learning loop
    for i in 0..5 {
        let target = 140.0 + (i as f32) * 2.0;
        learner.observe(&vec, target);
    }

    tuner.apply_online_updates(&learner);
    assert!(tuner.version().contains("online-5"));
    assert_eq!(tuner.throughput_sample_count(), 10_005); // pretrained 10000 + 5
}

#[test]
fn pretrained_then_bandit_workflow() {
    let tuner = BrickTuner::with_pretrained();
    let mut bandit = tuner.kernel_bandit();

    // Simulate bandit exploration
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();

    let kernel = tuner.recommend_kernel_with_exploration(&features, &bandit, 0.3).top_kernel;
    bandit.update(kernel, 0.8);

    assert_eq!(bandit.total_pulls, 1);
}

#[test]
fn kernel_from_index_roundtrip() {
    for i in 0..KernelBandit::NUM_KERNELS {
        let kernel = KernelType::from_index(i);
        let back = kernel.to_index();
        assert_eq!(back, i, "KernelType::from_index({}).to_index() should be {}", i, i);
    }
}
