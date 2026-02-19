pub(crate) use super::*;
#[test]
fn test_mse_loss_perfect() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

    let loss = mse_loss(&y_pred, &y_true);
    assert!((loss - 0.0).abs() < 1e-6);
}

#[test]
fn test_mse_loss_basic() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y_pred = Vector::from_slice(&[2.0, 3.0, 4.0]);

    // Errors: [1, 1, 1], squared: [1, 1, 1], mean: 1.0
    let loss = mse_loss(&y_pred, &y_true);
    assert!((loss - 1.0).abs() < 1e-6);
}

#[test]
fn test_mse_loss_different_errors() {
    let y_true = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

    // Errors: [1, 2, 3], squared: [1, 4, 9], mean: 14/3 ≈ 4.667
    let loss = mse_loss(&y_pred, &y_true);
    assert!((loss - 14.0 / 3.0).abs() < 1e-5);
}

#[test]
#[should_panic(expected = "same length")]
fn test_mse_loss_mismatched_lengths() {
    let y_true = Vector::from_slice(&[1.0, 2.0]);
    let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

    let _ = mse_loss(&y_pred, &y_true);
}

#[test]
fn test_mae_loss_perfect() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

    let loss = mae_loss(&y_pred, &y_true);
    assert!((loss - 0.0).abs() < 1e-6);
}

#[test]
fn test_mae_loss_basic() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y_pred = Vector::from_slice(&[1.5, 2.5, 2.5]);

    // Errors: [0.5, 0.5, -0.5], abs: [0.5, 0.5, 0.5], mean: 0.5
    let loss = mae_loss(&y_pred, &y_true);
    assert!((loss - 0.5).abs() < 1e-6);
}

#[test]
fn test_mae_loss_outlier_robustness() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y_pred = Vector::from_slice(&[2.0, 3.0, 100.0]);

    // MAE: (1 + 1 + 97) / 3 = 33.0
    let mae = mae_loss(&y_pred, &y_true);

    // MSE: (1 + 1 + 9409) / 3 = 3137.0
    let mse = mse_loss(&y_pred, &y_true);

    // MAE should be much less affected by outlier
    assert!(mae < mse / 10.0);
}

#[test]
#[should_panic(expected = "same length")]
fn test_mae_loss_mismatched_lengths() {
    let y_true = Vector::from_slice(&[1.0, 2.0]);
    let y_pred = Vector::from_slice(&[1.0, 2.0, 3.0]);

    let _ = mae_loss(&y_pred, &y_true);
}

#[test]
fn test_huber_loss_small_errors() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y_pred = Vector::from_slice(&[1.5, 2.5, 3.5]);

    // All errors (0.5) <= delta (1.0), so use quadratic
    // 0.5 * (0.5)² * 3 / 3 = 0.125
    let loss = huber_loss(&y_pred, &y_true, 1.0);
    assert!((loss - 0.125).abs() < 1e-6);
}

#[test]
fn test_huber_loss_large_errors() {
    let y_true = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let y_pred = Vector::from_slice(&[5.0, 5.0, 5.0]);

    // All errors (5.0) > delta (1.0), so use linear
    // 1.0 * (5.0 - 0.5 * 1.0) * 3 / 3 = 4.5
    let loss = huber_loss(&y_pred, &y_true, 1.0);
    assert!((loss - 4.5).abs() < 1e-6);
}

#[test]
fn test_huber_loss_mixed_errors() {
    let y_true = Vector::from_slice(&[0.0, 0.0]);
    let y_pred = Vector::from_slice(&[0.5, 5.0]);

    // First: 0.5 <= 1.0, use 0.5 * 0.5² = 0.125
    // Second: 5.0 > 1.0, use 1.0 * (5.0 - 0.5) = 4.5
    // Mean: (0.125 + 4.5) / 2 = 2.3125
    let loss = huber_loss(&y_pred, &y_true, 1.0);
    assert!((loss - 2.3125).abs() < 1e-5);
}

#[test]
#[should_panic(expected = "Delta must be positive")]
fn test_huber_loss_zero_delta() {
    let y_true = Vector::from_slice(&[1.0]);
    let y_pred = Vector::from_slice(&[2.0]);

    let _ = huber_loss(&y_pred, &y_true, 0.0);
}

#[test]
#[should_panic(expected = "Delta must be positive")]
fn test_huber_loss_negative_delta() {
    let y_true = Vector::from_slice(&[1.0]);
    let y_pred = Vector::from_slice(&[2.0]);

    let _ = huber_loss(&y_pred, &y_true, -1.0);
}

#[test]
fn test_huber_vs_mse_small_errors() {
    let y_true = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let y_pred = Vector::from_slice(&[1.1, 2.1, 3.1]);

    let huber = huber_loss(&y_pred, &y_true, 1.0);
    let mse = mse_loss(&y_pred, &y_true);

    // For small errors, Huber should be close to MSE
    assert!((huber - mse).abs() < 0.01);
}

#[test]
fn test_huber_vs_mae_large_errors() {
    let y_true = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let y_pred = Vector::from_slice(&[10.0, 10.0, 10.0]);

    let huber = huber_loss(&y_pred, &y_true, 1.0);
    let mae = mae_loss(&y_pred, &y_true);

    // For large errors, Huber should approximate MAE (with offset)
    assert!(huber < mae);
    assert!((huber - (mae - 0.5)).abs() < 0.1);
}

#[test]
fn test_mse_loss_struct() {
    let loss_fn = MSELoss;
    let y_true = Vector::from_slice(&[1.0, 2.0]);
    let y_pred = Vector::from_slice(&[1.0, 2.0]);

    let loss = loss_fn.compute(&y_pred, &y_true);
    assert!((loss - 0.0).abs() < 1e-6);
    assert_eq!(loss_fn.name(), "MSE");
}

#[test]
fn test_mae_loss_struct() {
    let loss_fn = MAELoss;
    let y_true = Vector::from_slice(&[1.0, 2.0]);
    let y_pred = Vector::from_slice(&[1.5, 2.5]);

    let loss = loss_fn.compute(&y_pred, &y_true);
    assert!((loss - 0.5).abs() < 1e-6);
    assert_eq!(loss_fn.name(), "MAE");
}

#[test]
fn test_huber_loss_struct() {
    let loss_fn = HuberLoss::new(1.0);
    let y_true = Vector::from_slice(&[1.0, 2.0]);
    let y_pred = Vector::from_slice(&[1.5, 2.5]);

    let loss = loss_fn.compute(&y_pred, &y_true);
    assert!(loss > 0.0);
    assert_eq!(loss_fn.name(), "Huber");
    assert!((loss_fn.delta() - 1.0).abs() < 1e-6);
}

#[test]
fn test_loss_trait_polymorphism() {
    let loss_fns: Vec<Box<dyn Loss>> = vec![
        Box::new(MSELoss),
        Box::new(MAELoss),
        Box::new(HuberLoss::new(1.0)),
    ];

    let y_true = Vector::from_slice(&[1.0, 2.0]);
    let y_pred = Vector::from_slice(&[1.5, 2.5]);

    for loss_fn in loss_fns {
        let loss = loss_fn.compute(&y_pred, &y_true);
        assert!(loss > 0.0);
        assert!(!loss_fn.name().is_empty());
    }
}

#[test]
fn test_negative_values() {
    let y_true = Vector::from_slice(&[-1.0, -2.0, -3.0]);
    let y_pred = Vector::from_slice(&[-1.5, -2.5, -3.5]);

    let mse = mse_loss(&y_pred, &y_true);
    let mae = mae_loss(&y_pred, &y_true);
    let huber = huber_loss(&y_pred, &y_true, 1.0);

    assert!(mse > 0.0);
    assert!(mae > 0.0);
    assert!(huber > 0.0);
}

#[test]
fn test_zero_values() {
    let y_true = Vector::from_slice(&[0.0, 0.0, 0.0]);
    let y_pred = Vector::from_slice(&[0.0, 0.0, 0.0]);

    let mse = mse_loss(&y_pred, &y_true);
    let mae = mae_loss(&y_pred, &y_true);
    let huber = huber_loss(&y_pred, &y_true, 1.0);

    assert!((mse - 0.0).abs() < 1e-6);
    assert!((mae - 0.0).abs() < 1e-6);
    assert!((huber - 0.0).abs() < 1e-6);
}

#[test]
fn test_single_value() {
    let y_true = Vector::from_slice(&[5.0]);
    let y_pred = Vector::from_slice(&[3.0]);

    let mse = mse_loss(&y_pred, &y_true);
    let mae = mae_loss(&y_pred, &y_true);
    let huber = huber_loss(&y_pred, &y_true, 1.0);

    assert!((mse - 4.0).abs() < 1e-6);
    assert!((mae - 2.0).abs() < 1e-6);
    assert!(huber > 0.0);
}

// ==========================================================================
// Contrastive Loss Tests (EXTREME TDD)
// ==========================================================================

#[test]
fn test_triplet_loss_satisfied_margin() {
    // Positive is much closer than negative - loss should be 0
    let anchor = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let positive = Vector::from_slice(&[0.9, 0.1, 0.0]); // d ≈ 0.14
    let negative = Vector::from_slice(&[0.0, 1.0, 0.0]); // d ≈ 1.41

    let loss = triplet_loss(&anchor, &positive, &negative, 0.2);
    assert!(
        (loss - 0.0).abs() < 1e-6,
        "Loss should be 0 when margin satisfied"
    );
}

#[test]
fn test_triplet_loss_violated_margin() {
    // Positive and negative equidistant - should have loss = margin
    let anchor = Vector::from_slice(&[0.0, 0.0]);
    let positive = Vector::from_slice(&[1.0, 0.0]); // d = 1.0
    let negative = Vector::from_slice(&[0.0, 1.0]); // d = 1.0

    let loss = triplet_loss(&anchor, &positive, &negative, 0.5);
    // d_pos - d_neg + margin = 1.0 - 1.0 + 0.5 = 0.5
    assert!((loss - 0.5).abs() < 1e-5);
}

#[test]
fn test_triplet_loss_hard_negative() {
    // Negative closer than positive (hard case)
    let anchor = Vector::from_slice(&[0.0, 0.0]);
    let positive = Vector::from_slice(&[2.0, 0.0]); // d = 2.0
    let negative = Vector::from_slice(&[0.5, 0.0]); // d = 0.5

    let loss = triplet_loss(&anchor, &positive, &negative, 0.2);
    // d_pos - d_neg + margin = 2.0 - 0.5 + 0.2 = 1.7
    assert!((loss - 1.7).abs() < 1e-5);
}

#[test]
fn test_triplet_loss_zero_margin() {
    let anchor = Vector::from_slice(&[0.0, 0.0]);
    let positive = Vector::from_slice(&[1.0, 0.0]);
    let negative = Vector::from_slice(&[0.0, 2.0]);

    let loss = triplet_loss(&anchor, &positive, &negative, 0.0);
    // d_pos = 1.0, d_neg = 2.0, loss = max(0, 1.0 - 2.0 + 0) = 0
    assert!((loss - 0.0).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "same dimension")]
fn test_triplet_loss_dimension_mismatch() {
    let anchor = Vector::from_slice(&[1.0, 0.0]);
    let positive = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let negative = Vector::from_slice(&[0.0, 1.0]);

    let _ = triplet_loss(&anchor, &positive, &negative, 0.2);
}

#[test]
fn test_info_nce_loss_basic() {
    let anchor = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let positive = Vector::from_slice(&[0.95, 0.05, 0.0]); // high similarity
    let negatives = vec![
        Vector::from_slice(&[0.0, 1.0, 0.0]), // low similarity
        Vector::from_slice(&[0.0, 0.0, 1.0]), // low similarity
    ];

    let loss = info_nce_loss(&anchor, &positive, &negatives, 0.1);
    // Should be positive (it's a proper loss)
    assert!(loss >= 0.0);
    // Should be relatively low since positive is very similar
    assert!(loss < 2.0);
}

#[test]
fn test_info_nce_loss_perfect_alignment() {
    // Identical vectors should give low loss
    let anchor = Vector::from_slice(&[1.0, 0.0]);
    let positive = Vector::from_slice(&[1.0, 0.0]); // identical
    let negatives = vec![
        Vector::from_slice(&[-1.0, 0.0]), // opposite
    ];

    let loss = info_nce_loss(&anchor, &positive, &negatives, 0.5);
    // With perfect positive and opposite negative, loss should be very low
    assert!(loss < 0.5);
}

#[test]
fn test_info_nce_loss_temperature_effect() {
    let anchor = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let positive = Vector::from_slice(&[0.7, 0.7, 0.0]);
    let negatives = vec![Vector::from_slice(&[0.0, 1.0, 0.0])];

    let loss_low_temp = info_nce_loss(&anchor, &positive, &negatives, 0.1);
    let loss_high_temp = info_nce_loss(&anchor, &positive, &negatives, 1.0);

    // Lower temperature makes distribution sharper
    // The exact relationship depends on the similarities
    assert!(loss_low_temp != loss_high_temp);
}

#[test]
fn test_info_nce_loss_many_negatives() {
    let anchor = Vector::from_slice(&[1.0, 0.0]);
    let positive = Vector::from_slice(&[0.9, 0.1]);
    let negatives: Vec<Vector<f32>> = (0..10)
        .map(|i| {
            let angle = (i as f32) * 0.3 + 1.0; // various angles
            Vector::from_slice(&[angle.cos(), angle.sin()])
        })
        .collect();

    let loss = info_nce_loss(&anchor, &positive, &negatives, 0.2);
    // More negatives should increase difficulty
    assert!(loss > 0.0);
}

#[test]
#[should_panic(expected = "same dimension")]
fn test_info_nce_loss_dimension_mismatch() {
    let anchor = Vector::from_slice(&[1.0, 0.0]);
    let positive = Vector::from_slice(&[0.9, 0.1]);
    let negatives = vec![Vector::from_slice(&[0.0, 1.0, 0.0])]; // wrong dim

    let _ = info_nce_loss(&anchor, &positive, &negatives, 0.1);
}

#[test]
#[should_panic(expected = "Temperature must be positive")]
fn test_info_nce_loss_zero_temperature() {
    let anchor = Vector::from_slice(&[1.0, 0.0]);
    let positive = Vector::from_slice(&[0.9, 0.1]);
    let negatives = vec![Vector::from_slice(&[0.0, 1.0])];

    let _ = info_nce_loss(&anchor, &positive, &negatives, 0.0);
}

#[test]
fn test_focal_loss_confident_correct() {
    // Model is confident and correct - loss should be low
    let predictions = Vector::from_slice(&[0.99, 0.01]);
    let targets = Vector::from_slice(&[1.0, 0.0]);

    let loss = focal_loss(&predictions, &targets, 0.25, 2.0);
    // Focal loss down-weights easy examples
    assert!(loss < 0.01);
}

#[test]
fn test_focal_loss_confident_wrong() {
    // Model is confident but wrong - loss should be high
    let predictions = Vector::from_slice(&[0.01, 0.99]);
    let targets = Vector::from_slice(&[1.0, 0.0]);

    let loss = focal_loss(&predictions, &targets, 0.25, 2.0);
    // High loss for confident wrong predictions
    assert!(loss > 1.0);
}

#[test]
fn test_focal_loss_uncertain() {
    // Model is uncertain - moderate loss
    let predictions = Vector::from_slice(&[0.5, 0.5]);
    let targets = Vector::from_slice(&[1.0, 0.0]);

    let loss = focal_loss(&predictions, &targets, 0.25, 2.0);
    assert!(loss > 0.0);
    assert!(loss < 1.0);
}

#[test]
fn test_focal_loss_gamma_effect() {
    let predictions = Vector::from_slice(&[0.9]);
    let targets = Vector::from_slice(&[1.0]);

    let loss_gamma_0 = focal_loss(&predictions, &targets, 0.25, 0.0);
    let loss_gamma_2 = focal_loss(&predictions, &targets, 0.25, 2.0);

    // Higher gamma reduces loss for easy examples more
    assert!(loss_gamma_2 < loss_gamma_0);
}

// ==========================================================================
// Cross-Entropy Loss Tests (GH-280)
// ==========================================================================

#[test]
fn test_cross_entropy_loss_one_hot() {
    let logits = Vector::from_slice(&[2.0, 1.0, 0.5]);
    let targets = Vector::from_slice(&[1.0, 0.0, 0.0]); // one-hot class 0

    let loss = cross_entropy_loss(&logits, &targets);
    assert!(loss > 0.0);
    assert!(loss.is_finite());
}


include!("tests_include_01.rs");
