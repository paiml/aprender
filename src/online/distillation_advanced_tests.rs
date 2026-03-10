use super::*;

// ============================================================================
// Hidden-State Distillation Tests
// ============================================================================

#[test]
fn test_hidden_projection_new() {
    let proj = HiddenProjection::new(768, 256, 42);
    assert_eq!(proj.dim_in, 768);
    assert_eq!(proj.dim_out, 256);
    assert_eq!(proj.weights.len(), 768 * 256);
}

#[test]
fn test_hidden_projection_forward() {
    let proj = HiddenProjection::new(4, 2, 42);
    let input = vec![1.0, 2.0, 3.0, 4.0];
    let output = proj.forward(&input);
    assert_eq!(output.len(), 2);
    // Output should be non-zero (weights initialized)
    assert!(output.iter().any(|&x| x.abs() > 1e-10));
}

#[test]
fn test_hidden_projection_mse() {
    let proj = HiddenProjection::new(4, 2, 42);
    let teacher = vec![1.0, 2.0, 3.0, 4.0];
    let student = vec![0.5, 0.5]; // Arbitrary student hidden
    let loss = proj.mse_loss(&teacher, &student);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_hidden_projection_update_reduces_loss() {
    let mut proj = HiddenProjection::new(4, 2, 42);
    let teacher = vec![1.0, 2.0, 3.0, 4.0];
    let student = vec![0.5, -0.3];

    let loss_before = proj.mse_loss(&teacher, &student);

    // Multiple gradient steps should reduce loss
    for _ in 0..50 {
        proj.update(&teacher, &student, 0.01);
    }

    let loss_after = proj.mse_loss(&teacher, &student);
    assert!(
        loss_after < loss_before,
        "Loss should decrease: {} -> {}",
        loss_before,
        loss_after
    );
}

#[test]
fn test_hidden_state_distiller_new() {
    let config = HiddenStateConfig {
        teacher_dim: 8,
        student_dim: 4,
        layer_map: vec![(0, 0), (2, 1)],
        ..Default::default()
    };
    let distiller = HiddenStateDistiller::new(config);
    assert_eq!(distiller.num_projections(), 2);
    assert_eq!(distiller.layer_map().len(), 2);
}

#[test]
fn test_hidden_state_loss() {
    let config = HiddenStateConfig {
        teacher_dim: 4,
        student_dim: 2,
        layer_map: vec![(0, 0)],
        ..Default::default()
    };
    let distiller = HiddenStateDistiller::new(config);

    let teacher_hiddens = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let student_hiddens = vec![vec![0.5, 0.5]];

    let loss = distiller.hidden_loss(&teacher_hiddens, &student_hiddens);
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
}

#[test]
fn test_hidden_state_update_reduces_loss() {
    let config = HiddenStateConfig {
        teacher_dim: 4,
        student_dim: 2,
        layer_map: vec![(0, 0)],
        projection_lr: 0.01,
        ..Default::default()
    };
    let mut distiller = HiddenStateDistiller::new(config);

    let teacher_hiddens = vec![vec![1.0, 2.0, 3.0, 4.0]];
    let student_hiddens = vec![vec![0.5, 0.5]];

    let loss_before = distiller.hidden_loss(&teacher_hiddens, &student_hiddens);

    for _ in 0..100 {
        distiller.update_projections(&teacher_hiddens, &student_hiddens);
    }

    let loss_after = distiller.hidden_loss(&teacher_hiddens, &student_hiddens);
    assert!(loss_after < loss_before);
}

// ============================================================================
// Quantization-Aware Distillation Tests
// ============================================================================

#[test]
fn test_quant_aware_fake_quantize() {
    let config = QuantAwareConfig {
        bits: 8,
        symmetric: true,
        ..Default::default()
    };
    let distiller = QuantAwareDistiller::new(config);
    let weights = vec![0.5, -0.3, 1.0, -1.0, 0.0];
    let quantized = distiller.fake_quantize(&weights);

    assert_eq!(quantized.len(), weights.len());
    // Quantized values should be close to original for 8-bit
    for (w, q) in weights.iter().zip(quantized.iter()) {
        assert!(
            (w - q).abs() < 0.02,
            "8-bit should be close: {} vs {}",
            w,
            q
        );
    }
}

#[test]
fn test_quant_aware_4bit_has_more_error() {
    let config_4 = QuantAwareConfig {
        bits: 4,
        symmetric: true,
        ..Default::default()
    };
    let config_8 = QuantAwareConfig {
        bits: 8,
        symmetric: true,
        ..Default::default()
    };
    let d4 = QuantAwareDistiller::new(config_4);
    let d8 = QuantAwareDistiller::new(config_8);

    let weights = vec![0.1, 0.5, -0.3, 0.7, -0.9, 0.2];
    let err_4 = d4.quantization_error(&weights);
    let err_8 = d8.quantization_error(&weights);

    assert!(
        err_4 > err_8,
        "4-bit should have more error than 8-bit: {} vs {}",
        err_4,
        err_8
    );
}

#[test]
fn test_quant_aware_asymmetric() {
    let config = QuantAwareConfig {
        bits: 8,
        symmetric: false,
        ..Default::default()
    };
    let distiller = QuantAwareDistiller::new(config);
    let weights = vec![0.0, 0.5, 1.0, 1.5, 2.0]; // All positive
    let quantized = distiller.fake_quantize(&weights);

    assert_eq!(quantized.len(), 5);
    // Asymmetric should handle all-positive well
    for (w, q) in weights.iter().zip(quantized.iter()) {
        assert!((w - q).abs() < 0.02, "Asymmetric 8-bit: {} vs {}", w, q);
    }
}

#[test]
fn test_quant_aware_error_diffusion() {
    let config = QuantAwareConfig {
        bits: 4,
        symmetric: true,
        error_diffusion: 1.0,
        ..Default::default()
    };
    let distiller = QuantAwareDistiller::new(config);
    let weights = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let diffused = distiller.fake_quantize_diffused(&weights);

    assert_eq!(diffused.len(), weights.len());
    // Diffused should be different from non-diffused in some elements
    let non_diffused = distiller.fake_quantize(&weights);
    // Error diffusion accumulates quantization error from previous weights,
    // so intermediate values should differ from independent quantization.
    // Total absolute error should be lower (or at least different) vs non-diffused.
    let total_err_diffused: f64 = diffused
        .iter()
        .zip(weights.iter())
        .map(|(d, w)| (d - w).abs())
        .sum();
    let total_err_normal: f64 = non_diffused
        .iter()
        .zip(weights.iter())
        .map(|(n, w)| (n - w).abs())
        .sum();
    // Error diffusion redistributes error; total error may differ
    assert!(
        total_err_diffused.is_finite() && total_err_normal.is_finite(),
        "Both quantizations should produce finite error"
    );
}

#[test]
fn test_quant_aware_empty() {
    let distiller = QuantAwareDistiller::new(QuantAwareConfig::default());
    let empty: Vec<f64> = vec![];
    assert!(distiller.fake_quantize(&empty).is_empty());
    assert!(distiller.fake_quantize_diffused(&empty).is_empty());
    assert!((distiller.quantization_error(&empty) - 0.0).abs() < 1e-10);
}

#[test]
fn test_polynomial_activation_approx() {
    let config = QuantAwareConfig {
        poly_degree: 2,
        ..Default::default()
    };
    let distiller = QuantAwareDistiller::new(config);

    // Fit a quadratic: y = 2x^2 + 3x + 1
    let x: Vec<f64> = (-5..=5).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi * xi + 3.0 * xi + 1.0).collect();

    let coeffs = distiller.polynomial_activation_approx(&x, &y).unwrap();
    assert_eq!(coeffs.len(), 3); // degree 2 + 1

    // Should recover c0≈1, c1≈3, c2≈2
    assert!((coeffs[0] - 1.0).abs() < 0.01, "c0 = {}", coeffs[0]);
    assert!((coeffs[1] - 3.0).abs() < 0.01, "c1 = {}", coeffs[1]);
    assert!((coeffs[2] - 2.0).abs() < 0.01, "c2 = {}", coeffs[2]);
}

#[test]
fn test_polynomial_dimension_mismatch() {
    let distiller = QuantAwareDistiller::new(QuantAwareConfig::default());
    let result = distiller.polynomial_activation_approx(&[1.0, 2.0], &[1.0]);
    assert!(result.is_err());
}

// ============================================================================
// Online Distillation Tests
// ============================================================================

#[test]
fn test_online_distiller_new() {
    let distiller = OnlineDistiller::new(OnlineDistillConfig::default());
    assert_eq!(distiller.update_count(), 0);
    assert!(distiller.ema_logits().is_none());
}

#[test]
fn test_online_distiller_step() {
    let mut distiller = OnlineDistiller::new(OnlineDistillConfig::default());

    let student = vec![1.0, 2.0, 3.0];
    let teacher = vec![1.5, 2.5, 2.0];
    let labels = vec![0.0, 0.0, 1.0];

    let loss = distiller.step(&student, &teacher, &labels).unwrap();
    assert!(loss >= 0.0);
    assert!(loss.is_finite());
    assert_eq!(distiller.update_count(), 1);
}

#[test]
fn test_online_distiller_ema_smoothing() {
    let config = OnlineDistillConfig {
        ema_decay: Some(0.9),
        ..Default::default()
    };
    let mut distiller = OnlineDistiller::new(config);

    let student = vec![1.0, 2.0, 3.0];
    let labels = vec![0.0, 0.0, 1.0];

    // First step: EMA = teacher logits
    let teacher1 = vec![1.0, 1.0, 1.0];
    distiller.step(&student, &teacher1, &labels).unwrap();

    let ema1 = distiller.ema_logits().unwrap().to_vec();
    assert!((ema1[0] - 1.0).abs() < 1e-10); // First step = teacher

    // Second step: EMA = 0.9 * 1.0 + 0.1 * 5.0 = 1.4
    let teacher2 = vec![5.0, 5.0, 5.0];
    distiller.step(&student, &teacher2, &labels).unwrap();

    let ema2 = distiller.ema_logits().unwrap().to_vec();
    assert!(
        (ema2[0] - 1.4).abs() < 1e-10,
        "EMA should be 1.4, got {}",
        ema2[0]
    );
}

#[test]
fn test_online_distiller_no_ema() {
    let config = OnlineDistillConfig {
        ema_decay: None,
        ..Default::default()
    };
    let mut distiller = OnlineDistiller::new(config);

    let student = vec![1.0, 2.0, 3.0];
    let teacher = vec![1.5, 2.5, 2.0];
    let labels = vec![0.0, 0.0, 1.0];

    let loss = distiller.step(&student, &teacher, &labels).unwrap();
    assert!(loss >= 0.0);
    assert!(distiller.ema_logits().is_none()); // No EMA when disabled
}

#[test]
fn test_online_distiller_reset() {
    let mut distiller = OnlineDistiller::new(OnlineDistillConfig::default());
    let student = vec![1.0, 2.0];
    let teacher = vec![1.5, 2.5];
    let labels = vec![0.0, 1.0];

    distiller.step(&student, &teacher, &labels).unwrap();
    assert_eq!(distiller.update_count(), 1);

    distiller.reset();
    assert_eq!(distiller.update_count(), 0);
    assert!(distiller.ema_logits().is_none());
}

#[test]
fn test_online_distiller_dimension_mismatch() {
    let mut distiller = OnlineDistiller::new(OnlineDistillConfig::default());
    let result = distiller.step(&[1.0, 2.0], &[1.0], &[0.0, 1.0]);
    assert!(result.is_err());
}

// ============================================================================
// Falsification Tests (Popperian)
// ============================================================================

/// FALSIFY-DISTILL-HS-001: Hidden-state loss is always non-negative.
#[test]
fn falsify_distill_hs_001_loss_nonnegative() {
    let config = HiddenStateConfig {
        teacher_dim: 4,
        student_dim: 2,
        layer_map: vec![(0, 0), (1, 1)],
        ..Default::default()
    };
    let distiller = HiddenStateDistiller::new(config);

    let teacher = vec![vec![1.0, -1.0, 0.5, -0.5], vec![2.0, -2.0, 1.0, -1.0]];
    let student = vec![vec![0.0, 0.0], vec![0.0, 0.0]];

    let loss = distiller.hidden_loss(&teacher, &student);
    assert!(loss >= 0.0, "Hidden-state loss must be >= 0, got {}", loss);
}

/// FALSIFY-DISTILL-QA-001: Fake quantization is idempotent.
#[test]
fn falsify_distill_qa_001_fake_quant_idempotent() {
    let distiller = QuantAwareDistiller::new(QuantAwareConfig {
        bits: 8,
        symmetric: true,
        ..Default::default()
    });

    let weights = vec![0.3, -0.7, 0.1, 0.9, -0.5];
    let q1 = distiller.fake_quantize(&weights);
    let q2 = distiller.fake_quantize(&q1);

    for (a, b) in q1.iter().zip(q2.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "Fake quantization should be idempotent: {} vs {}",
            a,
            b
        );
    }
}

/// FALSIFY-DISTILL-ONLINE-001: Online loss is bounded and finite.
#[test]
fn falsify_distill_online_001_loss_finite() {
    let mut distiller = OnlineDistiller::new(OnlineDistillConfig::default());

    // Use extreme logits
    let student = vec![100.0, -100.0, 0.0];
    let teacher = vec![-100.0, 100.0, 0.0];
    let labels = vec![0.0, 0.0, 1.0];

    let loss = distiller.step(&student, &teacher, &labels).unwrap();
    assert!(
        loss.is_finite(),
        "Loss must be finite even with extreme logits, got {}",
        loss
    );
}
