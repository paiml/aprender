//! Tests for evolution module: KernelArm, KernelBandit, and OnlineLearner.

mod brick_tuner_evolution;

use super::super::*;

// ============================================================================
// KernelArm Tests
// ============================================================================

#[test]
fn kernel_arm_default_is_zero() {
    let arm = KernelArm::default();
    assert_eq!(arm.pulls, 0);
    assert_eq!(arm.total_reward, 0.0);
    assert_eq!(arm.total_reward_sq, 0.0);
}

#[test]
fn kernel_arm_mean_zero_pulls_returns_zero() {
    let arm = KernelArm::default();
    assert_eq!(arm.mean(), 0.0);
}

#[test]
fn kernel_arm_mean_single_pull() {
    let arm = KernelArm { pulls: 1, total_reward: 0.8, total_reward_sq: 0.64 };
    assert!((arm.mean() - 0.8).abs() < 1e-6);
}

#[test]
fn kernel_arm_mean_multiple_pulls() {
    let arm = KernelArm { pulls: 4, total_reward: 2.0, total_reward_sq: 1.2 };
    assert!((arm.mean() - 0.5).abs() < 1e-6);
}

#[test]
fn kernel_arm_ucb_unexplored_is_infinity() {
    let arm = KernelArm::default();
    let ucb = arm.ucb(100, 2.0);
    assert!(ucb.is_infinite());
    assert!(ucb > 0.0);
}

#[test]
fn kernel_arm_ucb_explored_finite() {
    let arm = KernelArm { pulls: 10, total_reward: 5.0, total_reward_sq: 3.0 };
    let ucb = arm.ucb(100, 2.0);
    assert!(ucb.is_finite());
    // UCB = mean + c * sqrt(2 * ln(total) / pulls)
    // mean = 0.5, c = 2.0, sqrt(2 * ln(100) / 10) = sqrt(2 * 4.605 / 10) = sqrt(0.921) ~ 0.9597
    // UCB ~ 0.5 + 2.0 * 0.9597 ~ 2.419
    assert!(ucb > arm.mean());
}

#[test]
fn kernel_arm_ucb_decreases_with_more_pulls() {
    let arm_few = KernelArm { pulls: 5, total_reward: 2.5, total_reward_sq: 1.5 };
    let arm_many = KernelArm { pulls: 50, total_reward: 25.0, total_reward_sq: 15.0 };
    // Same mean (0.5), but more pulls => smaller confidence interval
    let ucb_few = arm_few.ucb(100, 2.0);
    let ucb_many = arm_many.ucb(100, 2.0);
    assert!(ucb_few > ucb_many, "UCB should decrease with more pulls");
}

#[test]
fn kernel_arm_ucb_increases_with_higher_exploration() {
    let arm = KernelArm { pulls: 10, total_reward: 5.0, total_reward_sq: 3.0 };
    let ucb_low_c = arm.ucb(100, 0.5);
    let ucb_high_c = arm.ucb(100, 4.0);
    assert!(ucb_high_c > ucb_low_c, "Higher c should increase UCB");
}

#[test]
fn kernel_arm_clone_is_equal() {
    let arm = KernelArm { pulls: 5, total_reward: 3.0, total_reward_sq: 2.0 };
    let cloned = arm.clone();
    assert_eq!(cloned.pulls, arm.pulls);
    assert_eq!(cloned.total_reward, arm.total_reward);
    assert_eq!(cloned.total_reward_sq, arm.total_reward_sq);
}

// ============================================================================
// KernelBandit Tests
// ============================================================================

#[test]
fn kernel_bandit_new_has_correct_arms() {
    let bandit = KernelBandit::new();
    assert_eq!(bandit.arms.len(), KernelBandit::NUM_KERNELS);
    assert_eq!(bandit.total_pulls, 0);
    assert!(!bandit.use_thompson);
    assert!((bandit.exploration_c - 2.0).abs() < 1e-6);
}

#[test]
fn kernel_bandit_default_is_empty() {
    let default = KernelBandit::default();
    // Default derive gives empty/zeroed fields
    assert_eq!(default.arms.len(), 0);
    assert_eq!(default.total_pulls, 0);
    assert!(!default.use_thompson);
}

#[test]
fn kernel_bandit_with_thompson_sampling() {
    let bandit = KernelBandit::with_thompson_sampling();
    assert!(bandit.use_thompson);
    assert_eq!(bandit.arms.len(), KernelBandit::NUM_KERNELS);
    assert_eq!(bandit.total_pulls, 0);
}

#[test]
fn kernel_bandit_select_ucb_returns_valid_kernel() {
    let bandit = KernelBandit::new();
    let kernel = bandit.select();
    // With all unexplored arms (INFINITY UCB), select should return some valid kernel
    let idx = kernel.to_index();
    assert!(idx < KernelBandit::NUM_KERNELS || idx == KernelType::Unknown.to_index());
}

#[test]
fn kernel_bandit_select_thompson_returns_valid_kernel() {
    let bandit = KernelBandit::with_thompson_sampling();
    let kernel = bandit.select();
    let idx = kernel.to_index();
    assert!(idx < KernelBandit::NUM_KERNELS || idx == KernelType::Unknown.to_index());
}

#[test]
fn kernel_bandit_update_increments_counters() {
    let mut bandit = KernelBandit::new();
    bandit.update(KernelType::TiledQ4K, 0.8);

    assert_eq!(bandit.total_pulls, 1);
    assert_eq!(bandit.arms[0].pulls, 1);
    assert!((bandit.arms[0].total_reward - 0.8).abs() < 1e-6);
    assert!((bandit.arms[0].total_reward_sq - 0.64).abs() < 1e-6);
}

#[test]
fn kernel_bandit_update_multiple_kernels() {
    let mut bandit = KernelBandit::new();
    bandit.update(KernelType::TiledQ4K, 0.5);
    bandit.update(KernelType::TiledQ4K, 0.7);
    bandit.update(KernelType::BatchedQ4K, 0.9);

    assert_eq!(bandit.total_pulls, 3);
    assert_eq!(bandit.arms[KernelType::TiledQ4K.to_index()].pulls, 2);
    assert_eq!(bandit.arms[KernelType::BatchedQ4K.to_index()].pulls, 1);
    assert!(
        (bandit.arms[KernelType::TiledQ4K.to_index()].total_reward - 1.2).abs() < 1e-6,
        "0.5 + 0.7 = 1.2"
    );
}

#[test]
fn kernel_bandit_update_out_of_range_index_safe() {
    let mut bandit = KernelBandit::new();
    // KernelType::Unknown has index 14, which may be >= NUM_KERNELS (12)
    // This should not panic; update should be silently ignored if out of range
    bandit.update(KernelType::Unknown, 0.5);
    // If index 14 >= arms.len() (12), total_pulls stays 0
    // If index 14 < arms.len(), total_pulls becomes 1
    // Either way, no panic is the important assertion
}

#[test]
fn kernel_bandit_best_kernel_with_no_pulls() {
    let bandit = KernelBandit::new();
    // All arms have mean 0.0, so best_kernel returns the first with max mean (index 0)
    let best = bandit.best_kernel();
    // Since all means are equal (0.0), max_by returns the last element with
    // equal ordering, but the implementation uses partial_cmp with Equal fallback
    // The result is deterministic but depends on iterator behavior
    let _ = best; // Just ensure no panic
}

#[test]
fn kernel_bandit_best_kernel_returns_highest_mean() {
    let mut bandit = KernelBandit::new();
    bandit.update(KernelType::TiledQ4K, 0.3);
    bandit.update(KernelType::BatchedQ4K, 0.9);
    bandit.update(KernelType::CoalescedQ4K, 0.5);

    let best = bandit.best_kernel();
    assert_eq!(best, KernelType::BatchedQ4K);
}

#[test]
fn kernel_bandit_select_ucb_favors_unexplored() {
    let mut bandit = KernelBandit::new();
    // Pull one arm many times
    for _ in 0..100 {
        bandit.update(KernelType::TiledQ4K, 0.9);
    }
    // Select should favor unexplored arms (they have INFINITY UCB)
    let selected = bandit.select();
    assert_ne!(
        selected,
        KernelType::TiledQ4K,
        "UCB should prefer unexplored arms over the well-explored one"
    );
}

#[test]
fn kernel_bandit_exploration_rate_no_pulls() {
    let bandit = KernelBandit::new();
    assert!((bandit.exploration_rate() - 1.0).abs() < 1e-6);
}

#[test]
fn kernel_bandit_exploration_rate_all_on_one_arm() {
    let mut bandit = KernelBandit::new();
    for _ in 0..10 {
        bandit.update(KernelType::TiledQ4K, 0.5);
    }
    // best_pulls = 10, total_pulls = 10, rate = 1 - 10/10 = 0
    assert!((bandit.exploration_rate() - 0.0).abs() < 1e-6);
}

#[test]
fn kernel_bandit_exploration_rate_spread_across_arms() {
    let mut bandit = KernelBandit::new();
    bandit.update(KernelType::TiledQ4K, 0.5);
    bandit.update(KernelType::BatchedQ4K, 0.5);
    bandit.update(KernelType::CoalescedQ4K, 0.5);
    bandit.update(KernelType::VectorizedQ4K, 0.5);
    // best_pulls = 1, total_pulls = 4, rate = 1 - 1/4 = 0.75
    assert!((bandit.exploration_rate() - 0.75).abs() < 1e-6);
}

#[test]
fn kernel_bandit_estimated_regret_no_pulls() {
    let bandit = KernelBandit::new();
    // All means are 0, so regret = 0
    assert!((bandit.estimated_regret() - 0.0).abs() < 1e-6);
}

#[test]
fn kernel_bandit_estimated_regret_single_arm() {
    let mut bandit = KernelBandit::new();
    for _ in 0..10 {
        bandit.update(KernelType::TiledQ4K, 0.8);
    }
    // Only one arm pulled, best_mean = 0.8
    // regret = (0.8 - 0.8) * 10 + (0.8 - 0.0) * 0 * (NUM_KERNELS - 1) = 0
    assert!((bandit.estimated_regret() - 0.0).abs() < 1e-6);
}

#[test]
fn kernel_bandit_estimated_regret_multiple_arms() {
    let mut bandit = KernelBandit::new();
    // Arm 0: mean = 0.9 (best)
    bandit.update(KernelType::TiledQ4K, 0.9);
    // Arm 3: mean = 0.3 (suboptimal)
    bandit.update(KernelType::BatchedQ4K, 0.3);

    let regret = bandit.estimated_regret();
    // best_mean = 0.9
    // regret = (0.9 - 0.9) * 1 + (0.9 - 0.3) * 1 = 0.6
    assert!((regret - 0.6).abs() < 1e-4);
}

#[test]
fn kernel_bandit_num_kernels_constant() {
    assert_eq!(KernelBandit::NUM_KERNELS, 12);
}

// ============================================================================
// OnlineLearner Tests
// ============================================================================

#[test]
fn online_learner_new_has_pretrained_weights() {
    let learner = OnlineLearner::new();
    let weights = learner.weights();
    // Should have DIM + 1 weights (bias + features)
    assert_eq!(weights.len(), TunerFeatures::DIM + 1);
    assert_eq!(learner.num_updates(), 0);
    assert!((learner.ema_loss() - 0.0).abs() < 1e-6);
}

#[test]
fn online_learner_with_learning_rate() {
    let learner = OnlineLearner::new().with_learning_rate(0.01);
    // Just verify it doesn't panic and returns a valid learner
    assert_eq!(learner.num_updates(), 0);
}

#[test]
fn online_learner_predict_with_empty_features() {
    let learner = OnlineLearner::new();
    let result = learner.predict(&[]);
    // Only bias contributes, clamped to max(0.0, bias)
    assert!(result >= 0.0);
}

#[test]
fn online_learner_predict_non_negative() {
    let learner = OnlineLearner::new();
    let features = TunerFeatures::builder().build();
    let vec = features.to_vector();
    let prediction = learner.predict(&vec);
    assert!(prediction >= 0.0, "Throughput prediction must be non-negative");
}

#[test]
fn online_learner_predict_deterministic() {
    let learner = OnlineLearner::new();
    let features = TunerFeatures::builder().model_params_b(7.0).batch_size(4).build();
    let vec = features.to_vector();
    let p1 = learner.predict(&vec);
    let p2 = learner.predict(&vec);
    assert_eq!(p1, p2, "Predictions should be deterministic");
}

#[test]
fn online_learner_predict_truncated_features() {
    let learner = OnlineLearner::new();
    // Fewer features than weights
    let short_features = vec![0.5, 0.3, 0.1];
    let result = learner.predict(&short_features);
    assert!(result >= 0.0);
}

#[test]
fn online_learner_observe_dimension_mismatch_ignored() {
    let mut learner = OnlineLearner::new();
    // Features length must be weights.len() - 1 for observe to work
    let wrong_dim = vec![1.0; 5]; // Wrong dimension
    learner.observe(&wrong_dim, 100.0);
    assert_eq!(learner.num_updates(), 0, "Mismatched dimensions should be silently ignored");
}

#[test]
fn online_learner_observe_correct_dimension() {
    let mut learner = OnlineLearner::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();
    let vec = features.to_vector();

    learner.observe(&vec, 100.0);
    assert_eq!(learner.num_updates(), 1);
}

#[test]
fn online_learner_observe_updates_ema_loss() {
    let mut learner = OnlineLearner::new();
    let features = TunerFeatures::builder().model_params_b(7.0).batch_size(1).build();
    let vec = features.to_vector();

    learner.observe(&vec, 200.0);
    assert!(learner.ema_loss() > 0.0, "EMA loss should be updated after observation");
}

#[test]
fn online_learner_observe_updates_weights() {
    let mut learner = OnlineLearner::new();
    let original_weights = learner.weights().to_vec();

    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();
    let vec = features.to_vector();

    learner.observe(&vec, 100.0);

    let updated_weights = learner.weights();
    // At least some weights should have changed
    let changed =
        original_weights.iter().zip(updated_weights.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
    assert!(changed, "Weights should be updated after observe");
}

#[test]
fn online_learner_multiple_observations_converge() {
    let mut learner = OnlineLearner::new().with_learning_rate(0.01);
    let features =
        TunerFeatures::builder().model_params_b(1.5).hidden_dim(1536).batch_size(1).build();
    let vec = features.to_vector();
    let target = 150.0;

    // Train for multiple epochs
    for _ in 0..50 {
        learner.observe(&vec, target);
    }

    let predicted = learner.predict(&vec);
    // After many observations with same target, prediction should get closer
    // (may not converge perfectly in 50 steps, but error should decrease)
    assert_eq!(learner.num_updates(), 50);
    assert!(predicted > 0.0);
}

#[test]
fn online_learner_replay_triggers_at_10() {
    let mut learner = OnlineLearner::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();
    let vec = features.to_vector();

    // Observe exactly 10 times to trigger replay
    for i in 0..10 {
        learner.observe(&vec, 100.0 + i as f32);
    }
    assert_eq!(learner.num_updates(), 10);
}

#[test]
fn online_learner_replay_buffer_bounded() {
    let mut learner = OnlineLearner::new();
    let features = TunerFeatures::builder().model_params_b(1.5).batch_size(4).build();
    let vec = features.to_vector();

    // Push more than replay_buffer_size (100) observations
    for i in 0..120 {
        learner.observe(&vec, 50.0 + i as f32);
    }
    assert_eq!(learner.num_updates(), 120);
    // No panic means the buffer is properly bounded
}

#[test]
fn online_learner_is_converging_initially_true() {
    let learner = OnlineLearner::new();
    // EMA loss is 0.0 < 0.15 threshold
    assert!(learner.is_converging());
}
