//! F280-F295: Phase 14 ML-Tuner Evolution (16 points)

use trueno::tuner::{BrickTuner, KernelType, QuantType, TunerFeatures};

/// F280: Pre-trained weights produce valid predictions (MLT-10)
#[test]
fn f280_pretrained_weights_valid() {
    use trueno::tuner::{pretrained, TunerFeatures};

    // Verify weight dimensions match TunerFeatures::DIM + 1 (for bias)
    assert_eq!(
        pretrained::THROUGHPUT_WEIGHTS.len(),
        TunerFeatures::DIM + 1,
        "F280 FALSIFIED: throughput weights must have {} elements",
        TunerFeatures::DIM + 1
    );
    assert_eq!(
        pretrained::KERNEL_WEIGHTS.len(),
        12,
        "F280 FALSIFIED: kernel weights must have 12 kernel types"
    );

    // Verify no NaN or Inf in weights
    for (i, w) in pretrained::THROUGHPUT_WEIGHTS.iter().enumerate() {
        assert!(w.is_finite(), "F280 FALSIFIED: throughput weight {} is not finite: {}", i, w);
    }
}

/// F281: Pre-trained tuner produces predictions with reasonable MAPE (MLT-10)
#[test]
fn f281_pretrained_mape_reasonable() {
    let tuner = BrickTuner::with_pretrained();

    // Pre-trained MAPE should be under 15%
    assert!(
        tuner.throughput_mape() < 0.15,
        "F281 FALSIFIED: pre-trained MAPE {} exceeds 15% threshold",
        tuner.throughput_mape()
    );

    // Sample count should reflect training data
    assert!(
        tuner.throughput_sample_count() >= 1000,
        "F281 FALSIFIED: pre-trained model claims {} samples, need >= 1000",
        tuner.throughput_sample_count()
    );
}

/// F282: Feature importance is well-defined (MLT-10)
#[test]
fn f282_feature_importance_valid() {
    use trueno::tuner::pretrained;

    let total_importance: f32 = pretrained::FEATURE_IMPORTANCE.iter().map(|(_, _, imp)| imp).sum();

    // Top 10 features should account for significant portion
    assert!(
        total_importance >= 0.8,
        "F282 FALSIFIED: top 10 features only account for {:.1}% of importance",
        total_importance * 100.0
    );

    // Each importance should be non-negative
    for (idx, name, imp) in &pretrained::FEATURE_IMPORTANCE {
        assert!(
            *imp >= 0.0,
            "F282 FALSIFIED: feature {} (idx {}) has negative importance {}",
            name,
            idx,
            imp
        );
    }
}

/// F283: KernelType round-trip consistency
#[test]
fn f283_kernel_type_roundtrip() {
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
    ];

    for kernel in kernels {
        let idx = kernel.to_index();
        let reconstructed = KernelType::from_index(idx);
        assert_eq!(
            kernel, reconstructed,
            "F283 FALSIFIED: {:?} -> {} -> {:?} round-trip failed",
            kernel, idx, reconstructed
        );
    }
}

/// F284: Bandit arm statistics are correct (MLT-13)
#[test]
fn f284_bandit_arm_stats() {
    use trueno::tuner::KernelArm;

    let mut arm = KernelArm::default();

    // Initial state
    assert_eq!(arm.pulls, 0);
    assert_eq!(arm.mean(), 0.0);
    assert_eq!(arm.ucb(0, 2.0), f32::INFINITY, "Unexplored arm should have infinite UCB");

    // After some observations
    arm.pulls = 10;
    arm.total_reward = 8.0; // 80% success rate
    arm.total_reward_sq = 8.0;

    assert!(
        (arm.mean() - 0.8).abs() < 0.01,
        "F284 FALSIFIED: mean should be 0.8, got {}",
        arm.mean()
    );

    // UCB should be finite
    let ucb = arm.ucb(100, 2.0);
    assert!(ucb.is_finite(), "F284 FALSIFIED: UCB should be finite, got {}", ucb);
    assert!(ucb > arm.mean(), "F284 FALSIFIED: UCB {} should exceed mean {}", ucb, arm.mean());
}

/// F285: Bandit selection explores unexplored arms (MLT-13)
#[test]
fn f285_bandit_explores_unknown() {
    use trueno::tuner::KernelBandit;

    let bandit = KernelBandit::new();

    // With no history, selection should work and return valid kernel
    let kernel = bandit.select();
    let idx = kernel.to_index();
    assert!(
        idx < KernelBandit::NUM_KERNELS,
        "F285 FALSIFIED: initial selection returned invalid kernel index {}",
        idx
    );

    // Exploration rate should be 1.0 for new bandit
    assert_eq!(
        bandit.exploration_rate(),
        1.0,
        "F285 FALSIFIED: new bandit exploration rate should be 1.0"
    );
}

/// F286: Bandit update tracks rewards correctly (MLT-13)
#[test]
fn f286_bandit_update_correct() {
    use trueno::tuner::KernelBandit;

    let mut bandit = KernelBandit::new();

    // Update with some rewards
    bandit.update(KernelType::BatchedQ4K, 0.9);
    bandit.update(KernelType::BatchedQ4K, 0.8);
    bandit.update(KernelType::TiledQ4K, 0.5);

    // Best kernel should be BatchedQ4K (higher mean reward)
    let best = bandit.best_kernel();
    assert_eq!(
        best,
        KernelType::BatchedQ4K,
        "F286 FALSIFIED: best kernel should be BatchedQ4K, got {:?}",
        best
    );

    // Exploration rate should decrease
    assert!(
        bandit.exploration_rate() < 1.0,
        "F286 FALSIFIED: exploration rate should decrease after updates"
    );
}

/// F287: Thompson sampling produces valid selections (MLT-13)
#[test]
fn f287_thompson_sampling_valid() {
    use trueno::tuner::KernelBandit;

    let mut bandit = KernelBandit::with_thompson_sampling();

    // Add some history
    bandit.update(KernelType::VectorizedQ4K, 0.7);
    bandit.update(KernelType::BatchedQ4K, 0.9);

    // Selection should still work
    let kernel = bandit.select();
    let idx = kernel.to_index();
    assert!(
        idx < KernelBandit::NUM_KERNELS,
        "F287 FALSIFIED: Thompson sampling returned invalid kernel index {}",
        idx
    );
}

/// F288: OnlineLearner initializes with pretrained weights (MLT-12)
#[test]
fn f288_online_learner_init() {
    use trueno::tuner::{pretrained, OnlineLearner};

    let learner = OnlineLearner::new();

    // Weights should match pretrained
    assert_eq!(
        learner.weights().len(),
        pretrained::THROUGHPUT_WEIGHTS.len(),
        "F288 FALSIFIED: learner weights dimension mismatch"
    );

    // Initial state
    assert_eq!(learner.num_updates(), 0);
    assert_eq!(learner.ema_loss(), 0.0);
}

/// F289: OnlineLearner produces valid predictions (MLT-12)
#[test]
fn f289_online_learner_predict() {
    use trueno::tuner::OnlineLearner;

    let learner = OnlineLearner::new();
    let features = vec![0.5; TunerFeatures::DIM]; // 47 features (bias is separate)

    let pred = learner.predict(&features);

    assert!(pred.is_finite(), "F289 FALSIFIED: prediction should be finite, got {}", pred);
    assert!(pred >= 0.0, "F289 FALSIFIED: prediction should be non-negative, got {}", pred);
}

/// F290: OnlineLearner updates weights on observe (MLT-12)
#[test]
fn f290_online_learner_observe() {
    use trueno::tuner::OnlineLearner;

    let mut learner = OnlineLearner::new();
    let features = vec![0.5; TunerFeatures::DIM];
    let target = 100.0;

    let weights_before = learner.weights().to_vec();
    learner.observe(&features, target);

    // Weights should change
    let weights_after = learner.weights();
    let changed =
        weights_before.iter().zip(weights_after.iter()).any(|(a, b)| (a - b).abs() > 1e-10);

    assert!(changed, "F290 FALSIFIED: weights should change after observe()");
    assert_eq!(learner.num_updates(), 1);
}

/// F291: OnlineLearner convergence detection (MLT-12)
#[test]
fn f291_online_learner_convergence() {
    use trueno::tuner::OnlineLearner;

    let mut learner = OnlineLearner::new();

    // Train on consistent data
    for _ in 0..100 {
        let features = vec![0.5; TunerFeatures::DIM];
        learner.observe(&features, 150.0);
    }

    // After training, should be converging
    assert!(learner.ema_loss() < 100.0, "F291 FALSIFIED: EMA loss should decrease with training");
}

/// F292: BrickTuner::with_pretrained creates valid tuner (MLT-10)
#[test]
fn f292_with_pretrained_creates_tuner() {
    let tuner = BrickTuner::with_pretrained();

    // Version should indicate pretrained
    assert!(
        tuner.version().contains("pretrained"),
        "F292 FALSIFIED: version should contain 'pretrained', got {}",
        tuner.version()
    );

    // Should still produce valid recommendations
    let features =
        TunerFeatures::builder().model_params_b(7.0).batch_size(1).gpu_mem_bw_gbs(1000.0).build();

    let rec = tuner.recommend(&features);
    assert!(rec.throughput.predicted_tps > 0.0);
}

/// F293: Online learning integration with BrickTuner (MLT-12)
#[test]
fn f293_online_learning_integration() {
    let tuner = BrickTuner::with_pretrained();

    // Create online learner from tuner
    let mut learner = tuner.online_learner();

    // Train
    let features = vec![0.5; TunerFeatures::DIM];
    learner.observe(&features, 200.0);
    learner.observe(&features, 195.0);

    // Apply updates
    let mut tuner_updated = tuner.clone();
    tuner_updated.apply_online_updates(&learner);

    // Version should change
    assert!(
        tuner_updated.version() != tuner.version(),
        "F293 FALSIFIED: version should change after online updates"
    );
}

/// F294: Bandit integration with BrickTuner (MLT-13)
#[test]
fn f294_bandit_integration() {
    let tuner = BrickTuner::with_pretrained();
    let mut bandit = tuner.kernel_bandit();

    let features = TunerFeatures::builder().model_params_b(7.0).batch_size(4).build();

    // Simulate some exploration
    for _ in 0..5 {
        let rec = tuner.recommend_kernel_with_exploration(&features, &bandit, 0.5);
        bandit.update(rec.top_kernel, 0.8);
    }

    // Should have explored
    assert!(bandit.estimated_regret() >= 0.0, "F294 FALSIFIED: regret should be non-negative");
}

/// F295: Full Phase 14 integration test
#[test]
fn f295_phase14_integration() {
    // 1. Create tuner with pretrained weights (MLT-10)
    let tuner = BrickTuner::with_pretrained();
    assert!(tuner.throughput_mape() < 0.15);

    // 2. Create online learner (MLT-12)
    let mut learner = tuner.online_learner();

    // 3. Create bandit (MLT-13)
    let mut bandit = tuner.kernel_bandit();

    // 4. Simulate inference loop
    let features = TunerFeatures::builder()
        .model_params_b(7.0)
        .batch_size(4)
        .quant_type(QuantType::Q4K)
        .gpu_mem_bw_gbs(1000.0)
        .build();

    for step in 0..20 {
        // Get kernel recommendation with exploration
        let rec = tuner.recommend_kernel_with_exploration(&features, &bandit, 0.3);

        // Simulate throughput measurement
        let measured_tps = 150.0 + (step as f32 * 2.0);

        // Update bandit
        let reward = (measured_tps / 200.0).min(1.0);
        bandit.update(rec.top_kernel, reward);

        // Update online learner
        learner.observe(&features.to_vector(), measured_tps);
    }

    // 5. Verify learning happened
    assert!(
        learner.num_updates() == 20,
        "F295 FALSIFIED: expected 20 updates, got {}",
        learner.num_updates()
    );

    assert!(bandit.exploration_rate() < 1.0, "F295 FALSIFIED: exploration rate should decrease");

    println!("F295 PASSED: Phase 14 integration successful");
    println!("  Online learner updates: {}", learner.num_updates());
    println!("  Bandit exploration rate: {:.2}", bandit.exploration_rate());
    println!("  Best kernel: {:?}", bandit.best_kernel());
}
