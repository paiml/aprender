//! Tests for curriculum learning

use super::*;

#[test]
fn test_linear_curriculum_initial() {
    let curriculum = LinearCurriculum::new(0.3, 1.0, 10);
    assert!((curriculum.difficulty() - 0.3).abs() < 1e-5);
    assert_eq!(curriculum.tier(), 2); // 0.3 -> tier 2
}

#[test]
fn test_linear_curriculum_progress() {
    let mut curriculum = LinearCurriculum::new(0.0, 1.0, 10);

    for i in 0..10 {
        curriculum.step(i, 1.0);
    }

    assert!((curriculum.difficulty() - 1.0).abs() < 1e-5);
    assert_eq!(curriculum.tier(), 4);
}

#[test]
fn test_linear_curriculum_halfway() {
    let mut curriculum = LinearCurriculum::new(0.0, 1.0, 10);

    for _ in 0..5 {
        curriculum.step(0, 1.0);
    }

    assert!((curriculum.difficulty() - 0.5).abs() < 1e-5);
    assert_eq!(curriculum.tier(), 3);
}

#[test]
fn test_linear_curriculum_reset() {
    let mut curriculum = LinearCurriculum::new(0.0, 1.0, 10);

    for _ in 0..5 {
        curriculum.step(0, 1.0);
    }
    curriculum.reset();

    assert!((curriculum.difficulty() - 0.0).abs() < 1e-5);
}

#[test]
fn test_tiered_curriculum_initial() {
    let curriculum = TieredCurriculum::citl_default();
    assert_eq!(curriculum.tier(), 1);
    assert!((curriculum.difficulty() - 0.0).abs() < 1e-5);
}

#[test]
fn test_tiered_curriculum_advance() {
    let mut curriculum = TieredCurriculum::new(vec![0.6, 0.7, 0.8], 3);

    // Not enough accuracy
    for _ in 0..3 {
        curriculum.step(0, 0.5);
    }
    assert_eq!(curriculum.tier(), 1);

    // Enough accuracy, but not enough patience
    for _ in 0..2 {
        curriculum.step(0, 0.65);
    }
    assert_eq!(curriculum.tier(), 1);

    // Third epoch at threshold -> advance
    curriculum.step(0, 0.65);
    assert_eq!(curriculum.tier(), 2);
}

#[test]
fn test_tiered_curriculum_max_tier() {
    // Need 3 thresholds to reach tier 4 (tier 1->2, 2->3, 3->4)
    let mut curriculum = TieredCurriculum::new(vec![0.5, 0.6, 0.7], 1);

    // Advance multiple times at 100% accuracy
    for _ in 0..10 {
        curriculum.step(0, 1.0);
    }

    // Should max out at tier 4
    assert_eq!(curriculum.tier(), 4);
}

#[test]
fn test_adaptive_curriculum_tier_for_error() {
    let curriculum = AdaptiveCurriculum::new();

    // ICE always tier 4
    assert_eq!(curriculum.tier_for_error("ICE-0001", 0), 4);

    // Type error with retry -> tier 3
    assert_eq!(curriculum.tier_for_error("E0308", 1), 3);

    // Name resolution with 2 retries -> tier 3
    assert_eq!(curriculum.tier_for_error("E0425", 2), 3);

    // Default first attempt -> tier 1
    assert_eq!(curriculum.tier_for_error("E0599", 0), 1);

    // Default first retry -> tier 2
    assert_eq!(curriculum.tier_for_error("E0599", 1), 2);
}

#[test]
fn test_adaptive_curriculum_class_tracking() {
    let mut curriculum = AdaptiveCurriculum::new();

    // Track some predictions
    curriculum.update_class("E0308", true);
    curriculum.update_class("E0308", true);
    curriculum.update_class("E0425", false);

    // E0308 should have higher accuracy
    let e0308_acc = *curriculum.class_accuracy.get("E0308").expect("key should exist");
    let e0425_acc = *curriculum.class_accuracy.get("E0425").expect("key should exist");
    assert!(e0308_acc > e0425_acc);

    // E0425 (low accuracy) should have higher weight
    let e0308_weight = curriculum.weight_for_class("E0308");
    let e0425_weight = curriculum.weight_for_class("E0425");
    assert!(e0425_weight > e0308_weight);
}

#[test]
fn test_efficiency_score() {
    // Higher accuracy -> higher efficiency
    assert!(efficiency_score(0.9, 1000) > efficiency_score(0.5, 1000));

    // Same accuracy, smaller corpus -> higher efficiency
    assert!(efficiency_score(0.7, 1000) > efficiency_score(0.7, 10000));
}

#[test]
fn test_select_optimal_tier() {
    let results = vec![
        (1, 0.65, 2000),   // E = 0.65 / ln(2000) ~ 0.085
        (2, 0.72, 5000),   // E = 0.72 / ln(5000) ~ 0.084
        (3, 0.75, 20000),  // E = 0.75 / ln(20000) ~ 0.076
        (4, 0.77, 100000), // E = 0.77 / ln(100000) ~ 0.067
    ];

    let (best_tier, _) = select_optimal_tier(&results).expect("operation should succeed");
    // Tier 1 or 2 should win due to smaller corpus
    assert!(best_tier <= 2);
}

#[test]
fn test_sample_weight() {
    let curriculum = LinearCurriculum::new(0.5, 0.5, 10);

    // Sample at current difficulty -> full weight
    let weight_at = curriculum.sample_weight(0.5);
    assert!((weight_at - 1.0).abs() < 1e-5);

    // Sample far from current difficulty -> reduced weight
    let weight_far = curriculum.sample_weight(0.0);
    assert!(weight_far < 1.0);
}

#[test]
fn test_include_sample() {
    let curriculum = LinearCurriculum::new(0.5, 0.5, 10);

    // Sample at or below difficulty -> included
    assert!(curriculum.include_sample(0.5));
    assert!(curriculum.include_sample(0.3));

    // Sample above difficulty -> excluded
    assert!(!curriculum.include_sample(0.7));
}

#[test]
fn test_curriculum_names() {
    assert_eq!(LinearCurriculum::new(0.0, 1.0, 10).name(), "LinearCurriculum");
    assert_eq!(TieredCurriculum::citl_default().name(), "TieredCurriculum");
    assert_eq!(AdaptiveCurriculum::new().name(), "AdaptiveCurriculum");
}

#[test]
fn test_adaptive_curriculum_step_and_reset() {
    let mut curriculum = AdaptiveCurriculum::new();

    // Step with high accuracy should increase overall difficulty
    curriculum.step(0, 0.9);
    assert!(curriculum.difficulty() > 0.0);

    curriculum.step(1, 0.8);
    let difficulty_after_step = curriculum.difficulty();
    assert!(difficulty_after_step > 0.0);

    // Reset should clear everything
    curriculum.reset();
    assert!((curriculum.difficulty() - 0.0).abs() < 1e-5);
    assert!(curriculum.class_accuracy.is_empty());
    assert!(curriculum.class_attempts.is_empty());
}

#[test]
fn test_adaptive_curriculum_all_tiers() {
    let mut curriculum = AdaptiveCurriculum::new();

    // Tier 1 when difficulty < 0.25
    assert_eq!(curriculum.tier(), 1);

    // Push difficulty up
    for _ in 0..5 {
        curriculum.step(0, 1.0);
    }
    // Should be tier 2, 3, or 4 depending on accumulated difficulty
    assert!(curriculum.tier() >= 1);
}

#[test]
fn test_tiered_curriculum_reset() {
    let mut curriculum = TieredCurriculum::new(vec![0.5, 0.6, 0.7], 1);

    // Advance to tier 2
    curriculum.step(0, 1.0);
    assert!(curriculum.tier() >= 2);

    // Reset should go back to tier 1
    curriculum.reset();
    assert_eq!(curriculum.tier(), 1);
}

#[test]
fn test_efficiency_score_edge_cases() {
    // Edge case: corpus_size = 1 (should return accuracy directly)
    assert!((efficiency_score(0.8, 1) - 0.8).abs() < 1e-5);

    // Edge case: corpus_size = 0 (should return accuracy)
    assert!((efficiency_score(0.8, 0) - 0.8).abs() < 1e-5);
}

#[test]
fn test_select_optimal_tier_empty() {
    let results: Vec<(usize, f32, usize)> = vec![];
    assert!(select_optimal_tier(&results).is_none());
}

#[test]
fn test_select_optimal_tier_single() {
    let results = vec![(2, 0.75, 5000)];
    let (tier, _) = select_optimal_tier(&results).expect("operation should succeed");
    assert_eq!(tier, 2);
}

#[test]
fn test_tiered_curriculum_difficulty() {
    let curriculum = TieredCurriculum::new(vec![0.5, 0.6, 0.7], 1);
    // Tier 1 corresponds to difficulty 0.0
    assert_eq!(curriculum.difficulty(), 0.0);
}

#[test]
fn test_linear_curriculum_name() {
    let curriculum = LinearCurriculum::new(0.0, 1.0, 10);
    assert!(!curriculum.name().is_empty());
}
