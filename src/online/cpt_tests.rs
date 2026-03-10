use super::*;

// ============================================================================
// CptConfig tests
// ============================================================================

#[test]
fn test_config_default() {
    let cfg = CptConfig::default();
    assert!((cfg.learning_rate - 2e-5).abs() < 1e-10);
    assert_eq!(cfg.warmup_steps, 100);
    assert_eq!(cfg.total_steps, 1000);
    assert_eq!(cfg.seq_length, 512);
    assert!((cfg.domain_mix_ratio - 0.7).abs() < 1e-10);
}

#[test]
fn test_config_validate_ok() {
    let cfg = CptConfig::default();
    assert!(cfg.validate().is_ok());
}

#[test]
fn test_config_validate_bad_lr() {
    let cfg = CptConfig {
        learning_rate: -1.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_bad_mix() {
    let cfg = CptConfig {
        domain_mix_ratio: 1.5,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_zero_steps() {
    let cfg = CptConfig {
        total_steps: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_zero_seq() {
    let cfg = CptConfig {
        seq_length: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// ============================================================================
// CptSchedule tests
// ============================================================================

#[test]
fn test_schedule_warmup() {
    let cfg = CptConfig {
        learning_rate: 1e-4,
        warmup_steps: 10,
        total_steps: 100,
        ..Default::default()
    };
    let sched = CptSchedule::new(&cfg);

    // Step 0: lr = 0
    assert!((sched.lr_at_step(0) - 0.0).abs() < 1e-10);

    // Step 5 (halfway through warmup): lr ≈ 5e-5
    assert!((sched.lr_at_step(5) - 5e-5).abs() < 1e-8);

    // Step 10 (end of warmup): lr = 1e-4
    assert!((sched.lr_at_step(10) - 1e-4).abs() < 1e-10);
}

#[test]
fn test_schedule_cosine_decay() {
    let cfg = CptConfig {
        learning_rate: 1e-4,
        warmup_steps: 0,
        total_steps: 100,
        ..Default::default()
    };
    let sched = CptSchedule::new(&cfg);

    // Step 0: full lr
    assert!((sched.lr_at_step(0) - 1e-4).abs() < 1e-10);

    // Step 50: half cosine
    let mid_lr = sched.lr_at_step(50);
    assert!(mid_lr > 0.0 && mid_lr < 1e-4);

    // Step 100: near zero
    let end_lr = sched.lr_at_step(100);
    assert!(end_lr < 1e-8);
}

#[test]
fn test_schedule_monotonic_decay() {
    let cfg = CptConfig {
        learning_rate: 1e-4,
        warmup_steps: 10,
        total_steps: 100,
        ..Default::default()
    };
    let sched = CptSchedule::new(&cfg);

    // After warmup, LR should be monotonically decreasing
    let mut prev = sched.lr_at_step(10);
    for step in 11..=100 {
        let lr = sched.lr_at_step(step);
        assert!(
            lr <= prev + 1e-12,
            "LR should decay: step {}, {} > {}",
            step,
            lr,
            prev
        );
        prev = lr;
    }
}

// ============================================================================
// DataMixer tests
// ============================================================================

#[test]
fn test_mixer_ratio() {
    let mut mixer = DataMixer::new(0.8, 42);
    let mut domain_count = 0;
    let total = 1000;

    for _ in 0..total {
        if mixer.next_is_domain() {
            domain_count += 1;
        }
    }

    let ratio = domain_count as f64 / total as f64;
    assert!(
        (ratio - 0.8).abs() < 0.1,
        "Domain ratio should be ~0.8, got {}",
        ratio
    );
}

#[test]
fn test_mixer_all_domain() {
    let mut mixer = DataMixer::new(1.0, 42);
    for _ in 0..100 {
        assert!(mixer.next_is_domain());
    }
}

#[test]
fn test_mixer_no_domain() {
    let mut mixer = DataMixer::new(0.0, 42);
    for _ in 0..100 {
        assert!(!mixer.next_is_domain());
    }
}

#[test]
fn test_mix_batches() {
    let mut mixer = DataMixer::new(0.5, 42);
    let domain = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let general = vec![vec![7, 8, 9], vec![10, 11, 12]];

    let batch = mixer.mix_batches(&domain, &general, 3);
    assert_eq!(batch.len(), 3);
}

// ============================================================================
// ReplayBuffer tests
// ============================================================================

#[test]
fn test_replay_buffer_new() {
    let buf = ReplayBuffer::new(100);
    assert!(buf.is_empty());
    assert_eq!(buf.capacity(), 100);
}

#[test]
fn test_replay_buffer_add_and_sample() {
    let mut buf = ReplayBuffer::new(10);
    buf.add(vec![1, 2, 3]);
    buf.add(vec![4, 5, 6]);
    assert_eq!(buf.len(), 2);

    let samples = buf.sample(3, 42);
    assert_eq!(samples.len(), 3);
    // All samples should be one of the two added
    for s in &samples {
        assert!(s == &[1, 2, 3] || s == &[4, 5, 6]);
    }
}

#[test]
fn test_replay_buffer_ring() {
    let mut buf = ReplayBuffer::new(2);
    buf.add(vec![1]);
    buf.add(vec![2]);
    buf.add(vec![3]); // Should overwrite first
    assert_eq!(buf.len(), 2);
}

#[test]
fn test_replay_buffer_empty_sample() {
    let buf = ReplayBuffer::new(10);
    let samples = buf.sample(5, 42);
    assert!(samples.is_empty());
}

// ============================================================================
// CptProgress tests
// ============================================================================

#[test]
fn test_progress_new() {
    let p = CptProgress::new(100);
    assert_eq!(p.step, 0);
    assert_eq!(p.total_steps, 100);
    assert!(!p.is_done());
    assert!((p.fraction() - 0.0).abs() < 1e-10);
}

#[test]
fn test_progress_update() {
    let mut p = CptProgress::new(10);
    p.update(1e-4, 5.0, true);
    assert_eq!(p.step, 1);
    assert_eq!(p.domain_samples, 1);
    assert_eq!(p.general_samples, 0);
    assert!(p.avg_loss > 0.0);
}

#[test]
fn test_progress_done() {
    let mut p = CptProgress::new(2);
    p.update(1e-4, 5.0, true);
    p.update(1e-4, 4.0, false);
    assert!(p.is_done());
    assert!((p.fraction() - 1.0).abs() < 1e-10);
}

#[test]
fn test_progress_fraction() {
    let mut p = CptProgress::new(4);
    p.update(1e-4, 5.0, true);
    assert!((p.fraction() - 0.25).abs() < 1e-10);
}

// ============================================================================
// Falsification tests
// ============================================================================

/// FALSIFY-CPT-001: LR schedule is always non-negative.
#[test]
fn falsify_cpt_001_lr_nonnegative() {
    let cfg = CptConfig {
        learning_rate: 1e-4,
        warmup_steps: 50,
        total_steps: 200,
        ..Default::default()
    };
    let sched = CptSchedule::new(&cfg);

    for step in 0..=250 {
        let lr = sched.lr_at_step(step);
        assert!(lr >= 0.0, "LR must be >= 0 at step {}, got {}", step, lr);
    }
}

/// FALSIFY-CPT-002: Data mixer respects ratio bounds.
#[test]
fn falsify_cpt_002_mixer_ratio_bounds() {
    for ratio in [0.0, 0.1, 0.5, 0.9, 1.0] {
        let mut mixer = DataMixer::new(ratio, 42);
        let mut domain_count = 0;
        let n = 500;

        for _ in 0..n {
            if mixer.next_is_domain() {
                domain_count += 1;
            }
        }

        let actual = domain_count as f64 / n as f64;
        assert!(
            (actual - ratio).abs() < 0.15,
            "Ratio {} should produce ~{}, got {}",
            ratio,
            ratio,
            actual
        );
    }
}

/// FALSIFY-CPT-003: Replay buffer never exceeds capacity.
#[test]
fn falsify_cpt_003_replay_bounded() {
    let mut buf = ReplayBuffer::new(5);
    for i in 0..100 {
        buf.add(vec![i]);
    }
    assert!(buf.len() <= 5, "Buffer must not exceed capacity");
}
