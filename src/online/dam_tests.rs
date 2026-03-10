use super::*;

// ============================================================================
// Config tests
// ============================================================================

#[test]
fn test_dam_config_default() {
    let cfg = DamConfig::default();
    assert!((cfg.learning_rate - 0.01).abs() < 1e-10);
    assert_eq!(cfg.num_iterations, 100);
    assert!((cfg.regularization - 0.01).abs() < 1e-10);
    assert_eq!(cfg.seed, 42);
}

#[test]
fn test_dam_config_validate_ok() {
    assert!(DamConfig::default().validate().is_ok());
}

#[test]
fn test_dam_config_validate_bad_lr() {
    let cfg = DamConfig {
        learning_rate: -0.01,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

#[test]
fn test_dam_config_validate_zero_iterations() {
    let cfg = DamConfig {
        num_iterations: 0,
        ..Default::default()
    };
    assert!(cfg.validate().is_err());
}

// ============================================================================
// Loss tests
// ============================================================================

#[test]
fn test_dam_merge_loss_zero() {
    let a = vec![1.0, 2.0, 3.0];
    let loss = DamLoss::compute_merge_loss(&a, &a);
    assert!((loss - 0.0).abs() < 1e-10);
}

#[test]
fn test_dam_merge_loss_nonzero() {
    let merged = vec![1.0, 2.0, 3.0];
    let target = vec![2.0, 3.0, 4.0];
    let loss = DamLoss::compute_merge_loss(&merged, &target);
    // MSE = ((1)^2 + (1)^2 + (1)^2) / 3 = 1.0
    assert!((loss - 1.0).abs() < 1e-10);
}

#[test]
fn test_dam_regularization() {
    let coeffs = vec![1.0, 2.0, 3.0];
    let reg = DamLoss::compute_regularization(&coeffs);
    // L2 = (1 + 4 + 9) / 3 = 14/3 ≈ 4.667
    assert!((reg - 14.0 / 3.0).abs() < 1e-10);
}

#[test]
fn test_dam_gradient_step() {
    let mut coeffs = vec![1.0, 2.0, 3.0];
    let gradients = vec![0.1, 0.2, 0.3];
    DamLoss::gradient_step(&mut coeffs, &gradients, 1.0);
    // coeffs = [1-0.1, 2-0.2, 3-0.3] = [0.9, 1.8, 2.7]
    assert!((coeffs[0] - 0.9).abs() < 1e-10);
    assert!((coeffs[1] - 1.8).abs() < 1e-10);
    assert!((coeffs[2] - 2.7).abs() < 1e-10);
}

// ============================================================================
// Softmax tests
// ============================================================================

#[test]
fn test_softmax_uniform() {
    let result = softmax(&[0.0, 0.0, 0.0]);
    for &v in &result {
        assert!((v - 1.0 / 3.0).abs() < 1e-10);
    }
}

#[test]
fn test_softmax_sum_to_one() {
    let result = softmax(&[1.0, 2.0, 3.0]);
    let sum: f64 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_softmax_monotonic() {
    let result = softmax(&[1.0, 2.0, 3.0]);
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
}

#[test]
fn test_softmax_single() {
    let result = softmax(&[5.0]);
    assert!((result[0] - 1.0).abs() < 1e-10);
}

// ============================================================================
// Optimization tests
// ============================================================================

#[test]
fn test_optimize_coefficients_basic() {
    let cfg = DamConfig {
        num_iterations: 50,
        ..Default::default()
    };
    // Simple quadratic loss: minimize (x - 0.5)^2
    let coeffs = optimize_coefficients(
        2,
        |c| {
            let s = softmax(c);
            (s[0] - 0.5) * (s[0] - 0.5) + (s[1] - 0.5) * (s[1] - 0.5)
        },
        &cfg,
    );
    assert_eq!(coeffs.len(), 2);
    assert!(coeffs.iter().all(|c| c.is_finite()));
}

#[test]
fn test_optimize_coefficients_converges() {
    let cfg = DamConfig {
        num_iterations: 200,
        learning_rate: 0.1,
        regularization: 0.0,
        ..Default::default()
    };
    let coeffs = optimize_coefficients(
        3,
        |c| {
            // Loss is minimized when coefficients are equal
            let s = softmax(c);
            let target = 1.0 / 3.0;
            s.iter().map(|&x| (x - target).powi(2)).sum()
        },
        &cfg,
    );
    let s = softmax(&coeffs);
    // After optimization, should be roughly equal
    for &v in &s {
        assert!(
            (v - 1.0 / 3.0).abs() < 0.2,
            "Coefficient should be near 1/3, got {}",
            v
        );
    }
}

// ============================================================================
// Report tests
// ============================================================================

#[test]
fn test_dam_report() {
    let report = DamReport {
        final_loss: 0.01,
        num_iterations: 100,
        coefficients: vec![0.3, 0.7],
        converged: true,
    };
    assert!(report.converged);
    assert_eq!(report.num_iterations, 100);
}

// ============================================================================
// Falsification tests
// ============================================================================

/// FALSIFY-DAM-001: Softmax output sums to 1.
#[test]
fn falsify_dam_001_softmax_sum() {
    for inputs in [
        vec![0.0],
        vec![1.0, -1.0],
        vec![10.0, 20.0, 30.0],
        vec![-100.0, 0.0, 100.0],
        vec![0.0; 10],
    ] {
        let result = softmax(&inputs);
        let sum: f64 = result.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-8,
            "Softmax sum should be 1.0 for {:?}, got {}",
            inputs,
            sum
        );
    }
}

/// FALSIFY-DAM-002: Optimization reduces loss.
#[test]
fn falsify_dam_002_optimization_improves() {
    let cfg = DamConfig {
        num_iterations: 100,
        learning_rate: 0.05,
        ..Default::default()
    };
    let loss_fn = |c: &[f64]| -> f64 {
        let s = softmax(c);
        s.iter().map(|&x| (x - 0.5).powi(2)).sum()
    };

    let initial = vec![0.0; 2];
    let initial_loss = loss_fn(&initial);

    let optimized = optimize_coefficients(2, loss_fn, &cfg);
    let final_loss = loss_fn(&optimized);

    assert!(
        final_loss <= initial_loss + 1e-6,
        "Optimization should not increase loss: {} -> {}",
        initial_loss,
        final_loss
    );
}

/// FALSIFY-DAM-003: MSE loss is non-negative.
#[test]
fn falsify_dam_003_loss_nonneg() {
    for (a, b) in [
        (vec![0.0], vec![0.0]),
        (vec![1.0, 2.0], vec![3.0, 4.0]),
        (vec![-1.0], vec![1.0]),
    ] {
        let loss = DamLoss::compute_merge_loss(&a, &b);
        assert!(loss >= 0.0, "MSE must be >= 0, got {}", loss);
    }
}
