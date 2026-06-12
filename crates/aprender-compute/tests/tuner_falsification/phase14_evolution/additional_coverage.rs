//! Additional Coverage Tests (Phase 14)

use trueno::tuner::{KernelType, QuantType, TunerFeatures};

/// Test OnlineLearner with custom learning rate
#[test]
fn test_online_learner_custom_lr() {
    use trueno::tuner::OnlineLearner;

    let learner = OnlineLearner::new().with_learning_rate(0.01);
    let features = vec![0.5; TunerFeatures::DIM];

    // Higher learning rate should still work
    let mut learner = learner;
    learner.observe(&features, 100.0);
    assert_eq!(learner.num_updates(), 1);
}

/// Test OnlineLearner replay buffer overflow
#[test]
fn test_online_learner_replay_buffer() {
    use trueno::tuner::OnlineLearner;

    let mut learner = OnlineLearner::new().with_learning_rate(0.001);
    let features = vec![0.5; TunerFeatures::DIM];

    // Fill replay buffer (default size 100) and overflow
    for i in 0..150 {
        learner.observe(&features, 100.0 + i as f32);
    }

    // Should have triggered multiple replay steps (every 10 updates)
    assert_eq!(learner.num_updates(), 150);
}

/// Test OnlineLearner dimension mismatch handling
#[test]
fn test_online_learner_dimension_mismatch() {
    use trueno::tuner::OnlineLearner;

    let mut learner = OnlineLearner::new();

    // Wrong dimension - should be ignored
    let wrong_features = vec![0.5; 10]; // Too few
    learner.observe(&wrong_features, 100.0);
    assert_eq!(learner.num_updates(), 0, "Dimension mismatch should be ignored");

    // Empty features - should be ignored
    learner.observe(&[], 100.0);
    assert_eq!(learner.num_updates(), 0);
}

/// Test Thompson sampling convergence
#[test]
fn test_thompson_convergence() {
    use trueno::tuner::KernelBandit;

    let mut bandit = KernelBandit::with_thompson_sampling();

    // Heavily favor one arm
    for _ in 0..50 {
        bandit.update(KernelType::BatchedQ4K, 0.95);
        bandit.update(KernelType::TiledQ4K, 0.3);
    }

    // Best kernel should converge
    assert_eq!(bandit.best_kernel(), KernelType::BatchedQ4K);
}

/// Test bandit regret calculation
#[test]
fn test_bandit_regret_positive() {
    use trueno::tuner::KernelBandit;

    let mut bandit = KernelBandit::new();

    // Mix of good and bad choices
    bandit.update(KernelType::BatchedQ4K, 0.9);
    bandit.update(KernelType::TiledQ4K, 0.5);
    bandit.update(KernelType::CoalescedQ4K, 0.6);

    // Regret should be non-negative
    let regret = bandit.estimated_regret();
    assert!(regret >= 0.0, "Regret should be non-negative: {}", regret);
}

/// Test pretrained weights dimensions match TunerFeatures
#[test]
fn test_pretrained_dimensions_consistent() {
    use trueno::tuner::pretrained;

    // All kernel weight arrays should have same length
    for (i, weights) in pretrained::KERNEL_WEIGHTS.iter().enumerate() {
        assert_eq!(
            weights.len(),
            pretrained::THROUGHPUT_WEIGHTS.len(),
            "Kernel weights {} should match throughput weights length",
            i
        );
    }
}

/// Test feature importance indices are valid
#[test]
fn test_feature_importance_indices_valid() {
    use trueno::tuner::pretrained;

    for (idx, name, importance) in &pretrained::FEATURE_IMPORTANCE {
        assert!(
            *idx < TunerFeatures::DIM,
            "Feature index {} ({}) exceeds DIM {}",
            idx,
            name,
            TunerFeatures::DIM
        );
        assert!(*importance >= 0.0 && *importance <= 1.0);
    }
}

/// Test QuantType bytes_per_param
#[test]
fn test_quant_type_bytes() {
    assert!(QuantType::Q4K.bytes_per_param() < 1.0);
    assert!(QuantType::F32.bytes_per_param() == 4.0);
    assert!(QuantType::F16.bytes_per_param() == 2.0);
    assert!(QuantType::Q8_0.bytes_per_param() == 1.0);
}

/// Test BottleneckClass recommended_action
#[test]
fn test_bottleneck_actions() {
    use trueno::tuner::BottleneckClass;

    assert!(!BottleneckClass::MemoryBound.recommended_action().is_empty());
    assert!(!BottleneckClass::ComputeBound.recommended_action().is_empty());
    assert!(!BottleneckClass::LaunchBound.recommended_action().is_empty());
    assert!(!BottleneckClass::AttentionBound.recommended_action().is_empty());
    assert!(!BottleneckClass::Unknown.recommended_action().is_empty());
}

/// Test more TunerFeatures builder methods
#[test]
fn test_tuner_features_builder_extended() {
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .hidden_dim(4096)
        .num_layers(32)
        .num_heads(32)
        .head_dim(128)
        .vocab_size(32000)
        .batch_size(4)
        .seq_len(512)
        .cuda_graphs(true)
        .kv_caches(1)
        .is_prefill(false)
        .quant_type(QuantType::Q4K)
        .kernel_type(KernelType::BatchedQ4K)
        .gpu_mem_bw_gbs(1000.0)
        .gpu_compute_tflops(83.0)
        .gpu_sm_count(128)
        .gpu_l2_cache_mb(72.0)
        .is_zero_copy(false)
        .measured_tps(150.0)
        .build();

    assert!(features.validate().is_ok());
    let vec = features.to_vector();
    assert_eq!(vec.len(), TunerFeatures::DIM);
}

/// Test KernelType to_index covers all variants
#[test]
fn test_kernel_type_to_index_all() {
    let kernels = [
        KernelType::TiledQ4K,
        KernelType::CoalescedQ4K,
        KernelType::VectorizedQ4K,
        KernelType::BatchedQ4K,
        KernelType::Dp4aQ4K,
        KernelType::FusedRmsNormQ4K,
        KernelType::CoalescedQ6K,
        KernelType::IncrementalAttention,
        KernelType::MultiWarpAttention,
        KernelType::BatchedAttention,
        KernelType::RmsNorm,
        KernelType::VectorizedRmsNorm,
        KernelType::BatchedRmsNorm,
        KernelType::FusedQKVHwDp4aQ4KGemv,
        KernelType::Generic,
        KernelType::Unknown,
    ];

    for (expected_idx, kernel) in kernels.iter().enumerate() {
        assert_eq!(kernel.to_index(), expected_idx, "Index mismatch for {:?}", kernel);
    }
}

/// Test QuantType to_index covers all variants
#[test]
fn test_quant_type_to_index_all() {
    assert_eq!(QuantType::Q4_0.to_index(), 0);
    assert_eq!(QuantType::Q4_1.to_index(), 1);
    assert_eq!(QuantType::Q4K.to_index(), 2);
    assert_eq!(QuantType::Q5K.to_index(), 3);
    assert_eq!(QuantType::Q6K.to_index(), 4);
    assert_eq!(QuantType::Q8_0.to_index(), 5);
    assert_eq!(QuantType::F16.to_index(), 6);
    assert_eq!(QuantType::F32.to_index(), 7);
}

// ============================================================================
// Test Suite Summary
// ============================================================================

/// Generate test score report
#[test]
fn test_score_summary() {
    // This test always passes - it's for documentation
    println!("\n=== Popperian Falsification Test Suite ===");
    println!("Categories:");
    println!("  F001-F020: Model Accuracy (20 points)");
    println!("  F021-F040: Feature Engineering (20 points)");
    println!("  F041-F060: Training Data Quality (20 points)");
    println!("  F061-F080: Integration Correctness (20 points)");
    println!("  F081-F100: Generalization & Robustness (20 points)");
    println!("  F280-F295: Phase 14 ML-Tuner Evolution (16 points)");
    println!("\nTotal: 116 points");
    println!("Minimum passing score: 100 points");
    println!("\nRun with: cargo test tuner_falsification --release");
}
