use super::*;

fn make_pair(chosen: f64, rejected: f64) -> PreferencePair {
    PreferencePair {
        chosen_logprob: chosen,
        rejected_logprob: rejected,
        ref_chosen_logprob: chosen * 0.9,
        ref_rejected_logprob: rejected * 0.9,
    }
}

// ============================================================================
// Config tests
// ============================================================================

#[test]
fn test_config_default() {
    let cfg = DpoConfig::default();
    assert!((cfg.beta - 0.1).abs() < 1e-10);
    assert!(cfg.use_reference);
    assert!(!cfg.length_normalize);
}

#[test]
fn test_config_validate_ok() {
    assert!(DpoConfig::default().validate().is_ok());
}

#[test]
fn test_config_validate_bad_beta() {
    let cfg = DpoConfig {
        beta: -1.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_config_validate_bad_smoothing() {
    let cfg = DpoConfig {
        label_smoothing: 1.0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// ============================================================================
// DPO Loss tests
// ============================================================================

#[test]
fn test_dpo_loss_chosen_preferred() {
    let loss = DpoLoss::new(DpoConfig::default());
    // Policy strongly prefers chosen over rejected
    let pair = PreferencePair {
        chosen_logprob: -1.0,
        rejected_logprob: -5.0,
        ref_chosen_logprob: -2.0,
        ref_rejected_logprob: -3.0,
    };
    let l = loss.compute(&pair);
    assert!(l >= 0.0);
    assert!(
        l < 1.0,
        "Loss should be low when policy prefers chosen, got {}",
        l
    );
}

#[test]
fn test_dpo_loss_rejected_preferred() {
    let loss = DpoLoss::new(DpoConfig::default());
    // Policy incorrectly prefers rejected
    let pair = PreferencePair {
        chosen_logprob: -5.0,
        rejected_logprob: -1.0,
        ref_chosen_logprob: -3.0,
        ref_rejected_logprob: -2.0,
    };
    let l = loss.compute(&pair);
    assert!(
        l > 0.5,
        "Loss should be high when policy prefers rejected, got {}",
        l
    );
}

#[test]
fn test_dpo_loss_equal() {
    let loss = DpoLoss::new(DpoConfig::default());
    let pair = PreferencePair {
        chosen_logprob: -2.0,
        rejected_logprob: -2.0,
        ref_chosen_logprob: -2.0,
        ref_rejected_logprob: -2.0,
    };
    let l = loss.compute(&pair);
    // log σ(0) = -log(2) ≈ 0.693
    assert!(
        (l - 0.693).abs() < 0.01,
        "Equal should give log(2), got {}",
        l
    );
}

#[test]
fn test_dpo_loss_batch() {
    let loss = DpoLoss::new(DpoConfig::default());
    let pairs = vec![make_pair(-1.0, -3.0), make_pair(-2.0, -4.0)];
    let batch_loss = loss.compute_batch(&pairs);
    assert!(batch_loss >= 0.0);
    assert!(batch_loss.is_finite());
}

#[test]
fn test_dpo_loss_batch_empty() {
    let loss = DpoLoss::new(DpoConfig::default());
    assert!((loss.compute_batch(&[]) - 0.0).abs() < 1e-10);
}

#[test]
fn test_dpo_label_smoothing() {
    let loss_no_smooth = DpoLoss::new(DpoConfig::default());
    let loss_smooth = DpoLoss::new(DpoConfig {
        label_smoothing: 0.1,
        ..Default::default()
    });

    let pair = make_pair(-1.0, -3.0);
    let l_no = loss_no_smooth.compute(&pair);
    let l_yes = loss_smooth.compute(&pair);

    // Label smoothing should change the loss
    assert!((l_no - l_yes).abs() > 1e-6);
}

// ============================================================================
// Gradient tests
// ============================================================================

#[test]
fn test_dpo_gradient() {
    let loss = DpoLoss::new(DpoConfig::default());
    let pair = make_pair(-1.0, -3.0);
    let (grad_c, grad_r) = loss.gradient(&pair);

    // Chosen gradient should be negative (encourage chosen)
    assert!(
        grad_c < 0.0,
        "Chosen gradient should be negative, got {}",
        grad_c
    );
    // Rejected gradient should be positive (discourage rejected)
    assert!(
        grad_r > 0.0,
        "Rejected gradient should be positive, got {}",
        grad_r
    );
    // They should have equal magnitude
    assert!((grad_c.abs() - grad_r.abs()).abs() < 1e-10);
}

// ============================================================================
// Implicit reward & accuracy tests
// ============================================================================

#[test]
fn test_implicit_reward() {
    let loss = DpoLoss::new(DpoConfig {
        beta: 0.1,
        ..Default::default()
    });
    let r = loss.implicit_reward(-1.0, -2.0);
    // β * (policy - ref) = 0.1 * (-1.0 - (-2.0)) = 0.1
    assert!((r - 0.1).abs() < 1e-10);
}

#[test]
fn test_accuracy() {
    let loss = DpoLoss::new(DpoConfig::default());
    let pairs = vec![
        PreferencePair {
            chosen_logprob: -1.0,
            rejected_logprob: -3.0,
            ref_chosen_logprob: -2.0,
            ref_rejected_logprob: -2.0,
        },
        PreferencePair {
            chosen_logprob: -3.0,
            rejected_logprob: -1.0,
            ref_chosen_logprob: -2.0,
            ref_rejected_logprob: -2.0,
        },
    ];
    let acc = loss.accuracy(&pairs);
    assert!((acc - 0.5).abs() < 1e-10); // 1 correct out of 2
}

// ============================================================================
// Metrics tests
// ============================================================================

#[test]
fn test_dpo_metrics() {
    let loss = DpoLoss::new(DpoConfig::default());
    let pairs = vec![make_pair(-1.0, -3.0), make_pair(-1.5, -4.0)];
    let metrics = DpoMetrics::from_batch(&loss, &pairs);

    assert_eq!(metrics.num_pairs, 2);
    assert!(metrics.avg_loss >= 0.0);
    assert!(
        metrics.reward_margin > 0.0,
        "Chosen should have higher reward"
    );
}

#[test]
fn test_dpo_metrics_empty() {
    let loss = DpoLoss::new(DpoConfig::default());
    let metrics = DpoMetrics::from_batch(&loss, &[]);
    assert_eq!(metrics.num_pairs, 0);
}

// ============================================================================
// SimPO (reference-free) tests
// ============================================================================

#[test]
fn test_simpo_no_reference() {
    let loss = DpoLoss::new(DpoConfig {
        use_reference: false,
        ..Default::default()
    });
    let pair = PreferencePair {
        chosen_logprob: -1.0,
        rejected_logprob: -3.0,
        ref_chosen_logprob: 999.0, // Should be ignored
        ref_rejected_logprob: 999.0,
    };
    let l = loss.compute(&pair);
    assert!(l >= 0.0);
    assert!(l.is_finite());
}

// ============================================================================
// Falsification tests
// ============================================================================

/// FALSIFY-DPO-001: DPO loss is always non-negative.
#[test]
fn falsify_dpo_001_loss_nonnegative() {
    let loss = DpoLoss::new(DpoConfig::default());
    for chosen in [-10.0, -5.0, -1.0, 0.0] {
        for rejected in [-10.0, -5.0, -1.0, 0.0] {
            let pair = PreferencePair {
                chosen_logprob: chosen,
                rejected_logprob: rejected,
                ref_chosen_logprob: chosen * 0.5,
                ref_rejected_logprob: rejected * 0.5,
            };
            let l = loss.compute(&pair);
            assert!(
                l >= 0.0,
                "Loss must be >= 0, got {} for ({}, {})",
                l,
                chosen,
                rejected
            );
        }
    }
}

/// FALSIFY-DPO-002: Gradients are zero-sum (chosen + rejected).
#[test]
fn falsify_dpo_002_gradient_zero_sum() {
    let loss = DpoLoss::new(DpoConfig::default());
    for chosen in [-5.0, -2.0, -0.5] {
        for rejected in [-5.0, -2.0, -0.5] {
            let pair = PreferencePair {
                chosen_logprob: chosen,
                rejected_logprob: rejected,
                ref_chosen_logprob: chosen * 0.8,
                ref_rejected_logprob: rejected * 0.8,
            };
            let (gc, gr) = loss.gradient(&pair);
            assert!(
                (gc + gr).abs() < 1e-10,
                "Gradients must sum to 0: {} + {} = {}",
                gc,
                gr,
                gc + gr
            );
        }
    }
}

/// FALSIFY-DPO-003: Higher reward for chosen when policy is correct.
#[test]
fn falsify_dpo_003_reward_ordering() {
    let loss = DpoLoss::new(DpoConfig::default());
    // Policy assigns higher prob to chosen
    let r_chosen = loss.implicit_reward(-1.0, -2.0);
    let r_rejected = loss.implicit_reward(-4.0, -2.0);
    assert!(
        r_chosen > r_rejected,
        "Chosen reward {} should be > rejected {}",
        r_chosen,
        r_rejected
    );
}
