use super::*;
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

#[test]
fn test_focal_loss_alpha_balancing() {
    let predictions = Vector::from_slice(&[0.5]);
    let targets_pos = Vector::from_slice(&[1.0]);
    let targets_neg = Vector::from_slice(&[0.0]);

    let loss_pos = focal_loss(&predictions, &targets_pos, 0.25, 2.0);
    let loss_neg = focal_loss(&predictions, &targets_neg, 0.25, 2.0);

    // Alpha=0.25 means positive class gets 0.25 weight, negative gets 0.75
    // So loss for negative class should be higher at same confidence
    assert!(loss_neg > loss_pos);
}

#[test]
#[should_panic(expected = "same length")]
fn test_focal_loss_length_mismatch() {
    let predictions = Vector::from_slice(&[0.9, 0.1]);
    let targets = Vector::from_slice(&[1.0]);

    let _ = focal_loss(&predictions, &targets, 0.25, 2.0);
}

#[test]
fn test_kl_divergence_identical() {
    let p = Vector::from_slice(&[0.3, 0.4, 0.3]);
    let q = Vector::from_slice(&[0.3, 0.4, 0.3]);

    let kl = kl_divergence(&p, &q);
    assert!((kl - 0.0).abs() < 1e-6);
}

#[test]
fn test_kl_divergence_different() {
    let p = Vector::from_slice(&[0.9, 0.1]);
    let q = Vector::from_slice(&[0.1, 0.9]);

    let kl = kl_divergence(&p, &q);
    // KL divergence should be positive for different distributions
    assert!(kl > 0.0);
}

#[test]
fn test_kl_divergence_asymmetry() {
    let p = Vector::from_slice(&[0.9, 0.1]);
    let q = Vector::from_slice(&[0.5, 0.5]);

    let kl_pq = kl_divergence(&p, &q);
    let kl_qp = kl_divergence(&q, &p);

    // KL divergence is not symmetric
    assert!((kl_pq - kl_qp).abs() > 0.01);
}

#[test]
fn test_kl_divergence_zero_in_p() {
    // When p[i] = 0, that term contributes 0 to KL
    let p = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let q = Vector::from_slice(&[0.5, 0.3, 0.2]);

    let kl = kl_divergence(&p, &q);
    // KL = 1.0 * ln(1.0 / 0.5) = ln(2) ≈ 0.693
    assert!((kl - 2.0_f32.ln()).abs() < 1e-5);
}

#[test]
fn test_kl_divergence_handles_small_q() {
    // When q[i] is very small, we clamp to avoid infinity
    let p = Vector::from_slice(&[0.5, 0.5]);
    let q = Vector::from_slice(&[0.999, 0.001]);

    let kl = kl_divergence(&p, &q);
    assert!(kl.is_finite());
    assert!(kl > 0.0);
}

#[test]
#[should_panic(expected = "same length")]
fn test_kl_divergence_length_mismatch() {
    let p = Vector::from_slice(&[0.5, 0.5]);
    let q = Vector::from_slice(&[0.3, 0.3, 0.4]);

    let _ = kl_divergence(&p, &q);
}

// ==========================================================================
// Struct Wrapper Tests
// ==========================================================================

#[test]
fn test_triplet_loss_struct() {
    let loss_fn = TripletLoss::new(0.3);
    assert!((loss_fn.margin() - 0.3).abs() < 1e-6);

    let anchor = Vector::from_slice(&[1.0, 0.0]);
    let positive = Vector::from_slice(&[0.9, 0.1]);
    let negative = Vector::from_slice(&[0.0, 1.0]);

    let loss = loss_fn.compute_triplet(&anchor, &positive, &negative);
    assert!(loss >= 0.0);
}

#[test]
fn test_focal_loss_struct() {
    let loss_fn = FocalLoss::new(0.25, 2.0);
    assert!((loss_fn.alpha() - 0.25).abs() < 1e-6);
    assert!((loss_fn.gamma() - 2.0).abs() < 1e-6);

    let predictions = Vector::from_slice(&[0.9, 0.1]);
    let targets = Vector::from_slice(&[1.0, 0.0]);

    let loss = loss_fn.compute(&predictions, &targets);
    assert!(loss >= 0.0);
    assert_eq!(loss_fn.name(), "Focal");
}

#[test]
fn test_focal_loss_trait_polymorphism() {
    let loss_fns: Vec<Box<dyn Loss>> = vec![Box::new(MSELoss), Box::new(FocalLoss::new(0.25, 2.0))];

    let y_pred = Vector::from_slice(&[0.9, 0.1]);
    let y_true = Vector::from_slice(&[1.0, 0.0]);

    for loss_fn in loss_fns {
        let loss = loss_fn.compute(&y_pred, &y_true);
        assert!(loss >= 0.0);
    }
}

#[test]
fn test_info_nce_loss_struct() {
    let loss_fn = InfoNCELoss::new(0.1);
    assert!((loss_fn.temperature() - 0.1).abs() < 1e-6);

    let anchor = Vector::from_slice(&[1.0, 0.0, 0.0]);
    let positive = Vector::from_slice(&[0.9, 0.1, 0.0]);
    let negatives = vec![Vector::from_slice(&[0.0, 1.0, 0.0])];

    let loss = loss_fn.compute_contrastive(&anchor, &positive, &negatives);
    assert!(loss >= 0.0);
}

#[test]
fn test_euclidean_distance_via_triplet() {
    // Test euclidean distance indirectly through triplet loss
    // d(a, a) = 0, so triplet_loss with positive=anchor should give margin
    let anchor = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let positive = anchor.clone();
    let negative = Vector::from_slice(&[4.0, 5.0, 6.0]);

    let loss = triplet_loss(&anchor, &positive, &negative, 0.5);
    // d_pos = 0, d_neg = sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27)
    // loss = max(0, 0 - sqrt(27) + 0.5) = 0
    assert!((loss - 0.0).abs() < 1e-6);
}

#[test]
fn test_cosine_similarity_via_info_nce() {
    // Test cosine similarity indirectly through InfoNCE
    // Compare aligned vs orthogonal positive vectors
    let anchor = Vector::from_slice(&[1.0, 0.0]);
    let negatives = vec![Vector::from_slice(&[0.0, 1.0])];

    // Aligned positive (cosine sim = 1)
    let positive_aligned = Vector::from_slice(&[1.0, 0.0]);
    let loss_aligned = info_nce_loss(&anchor, &positive_aligned, &negatives, 0.5);

    // Orthogonal positive (cosine sim = 0)
    let positive_ortho = Vector::from_slice(&[0.0, 1.0]);
    let loss_ortho = info_nce_loss(&anchor, &positive_ortho, &negatives, 0.5);

    // Loss with orthogonal positive should be higher than aligned
    assert!(loss_ortho > loss_aligned);
    assert!(loss_aligned >= 0.0);
}

#[test]
fn test_contrastive_losses_numerical_stability() {
    // Test with extreme values
    let anchor = Vector::from_slice(&[1e6, 1e-6]);
    let positive = Vector::from_slice(&[1e6, 1e-6]);
    let negative = Vector::from_slice(&[-1e6, 1e-6]);

    // Triplet should handle large values
    let triplet = triplet_loss(&anchor, &positive, &negative, 0.1);
    assert!(triplet.is_finite());

    // InfoNCE with normalized cosine should be stable
    let info_nce = info_nce_loss(&anchor, &positive, &[negative.clone()], 0.1);
    assert!(info_nce.is_finite());
}

// Dice Loss Tests
#[test]
fn test_dice_loss_perfect() {
    let y_pred = Vector::from_slice(&[1.0, 0.0, 1.0, 0.0]);
    let y_true = Vector::from_slice(&[1.0, 0.0, 1.0, 0.0]);
    let loss = dice_loss(&y_pred, &y_true, 1e-6);
    assert!(loss < 0.01);
}

#[test]
fn test_dice_loss_no_overlap() {
    let y_pred = Vector::from_slice(&[1.0, 1.0, 0.0, 0.0]);
    let y_true = Vector::from_slice(&[0.0, 0.0, 1.0, 1.0]);
    let loss = dice_loss(&y_pred, &y_true, 1e-6);
    assert!(loss > 0.99);
}

#[test]
fn test_dice_loss_partial_overlap() {
    let y_pred = Vector::from_slice(&[1.0, 1.0, 0.0, 0.0]);
    let y_true = Vector::from_slice(&[1.0, 0.0, 1.0, 0.0]);
    let loss = dice_loss(&y_pred, &y_true, 1e-6);
    assert!(loss > 0.0 && loss < 1.0);
}

#[test]
fn test_dice_loss_struct() {
    let loss_fn = DiceLoss::new(1.0);
    assert_eq!(loss_fn.smooth(), 1.0);
    assert_eq!(loss_fn.name(), "Dice");
}

// Hinge Loss Tests
#[test]
fn test_hinge_loss_correct() {
    let y_pred = Vector::from_slice(&[2.0, -2.0]);
    let y_true = Vector::from_slice(&[1.0, -1.0]);
    let loss = hinge_loss(&y_pred, &y_true, 1.0);
    assert!((loss - 0.0).abs() < 1e-6);
}

#[test]
fn test_hinge_loss_incorrect() {
    let y_pred = Vector::from_slice(&[-1.0, 1.0]);
    let y_true = Vector::from_slice(&[1.0, -1.0]);
    let loss = hinge_loss(&y_pred, &y_true, 1.0);
    assert!(loss > 0.0);
}

#[test]
fn test_hinge_loss_margin_violation() {
    let y_pred = Vector::from_slice(&[0.5]);
    let y_true = Vector::from_slice(&[1.0]);
    let loss = hinge_loss(&y_pred, &y_true, 1.0);
    assert!((loss - 0.5).abs() < 1e-6);
}

#[test]
fn test_squared_hinge_loss() {
    let y_pred = Vector::from_slice(&[0.5]);
    let y_true = Vector::from_slice(&[1.0]);
    let loss = squared_hinge_loss(&y_pred, &y_true, 1.0);
    assert!((loss - 0.25).abs() < 1e-6);
}

#[test]
fn test_hinge_loss_struct() {
    let loss_fn = HingeLoss::new(1.0);
    assert_eq!(loss_fn.margin(), 1.0);
    assert_eq!(loss_fn.name(), "Hinge");
}

// CTC Loss Tests
#[test]
fn test_ctc_loss_creation() {
    let ctc = CTCLoss::new(0);
    assert_eq!(ctc.blank_idx(), 0);
}

#[test]
fn test_ctc_loss_simple() {
    let ctc = CTCLoss::new(0);
    // Log probs: 3 time steps, 3 classes (blank=0, a=1, b=2)
    // Using valid log probabilities (negative values)
    let log_probs = vec![
        vec![-1.0, -0.7, -0.7], // t=0
        vec![-0.7, -1.0, -0.7], // t=1
        vec![-0.7, -0.7, -1.0], // t=2
    ];
    let targets = vec![1, 2]; // "ab"
    let loss = ctc.forward(&log_probs, &targets, 3, 2);
    assert!(loss.is_finite());
}

#[test]
fn test_ctc_loss_empty_target() {
    let ctc = CTCLoss::new(0);
    let log_probs = vec![vec![0.0; 3]; 5];
    let loss = ctc.forward(&log_probs, &[], 5, 0);
    assert_eq!(loss, 0.0);
}

#[test]
fn test_ctc_loss_single_char() {
    let ctc = CTCLoss::new(0);
    // High prob for class 1 at all times (log prob ~0 means prob ~1)
    let log_probs = vec![
        vec![-10.0, -0.01, -10.0], // high prob for class 1
        vec![-10.0, -0.01, -10.0],
    ];
    let targets = vec![1];
    let loss = ctc.forward(&log_probs, &targets, 2, 1);
    assert!(loss.is_finite());
}

// Wasserstein Loss Tests
#[test]
fn test_wasserstein_loss_equal() {
    let real = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let fake = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let loss = wasserstein_loss(&real, &fake);
    assert!((loss - 0.0).abs() < 1e-6);
}

#[test]
fn test_wasserstein_loss_diff() {
    let real = Vector::from_slice(&[1.0, 1.0, 1.0]);
    let fake = Vector::from_slice(&[2.0, 2.0, 2.0]);
    let loss = wasserstein_loss(&real, &fake);
    assert!((loss - 1.0).abs() < 1e-6);
}

#[test]
fn test_wasserstein_discriminator_loss() {
    let real = Vector::from_slice(&[3.0, 3.0]);
    let fake = Vector::from_slice(&[1.0, 1.0]);
    let loss = wasserstein_discriminator_loss(&real, &fake);
    // Should be positive (wants real > fake)
    assert!((loss - 2.0).abs() < 1e-6);
}

#[test]
fn test_wasserstein_generator_loss() {
    let fake = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let loss = wasserstein_generator_loss(&fake);
    assert!((loss - (-2.0)).abs() < 1e-6);
}

#[test]
fn test_gradient_penalty() {
    let grads = vec![0.6, 0.8]; // norm = 1.0
    let penalty = gradient_penalty(&grads, 10.0);
    assert!((penalty - 0.0).abs() < 1e-6);

    let grads2 = vec![1.2, 1.6]; // norm = 2.0
    let penalty2 = gradient_penalty(&grads2, 10.0);
    assert!((penalty2 - 10.0).abs() < 1e-6);
}

#[test]
fn test_wasserstein_loss_struct() {
    let loss_fn = WassersteinLoss::new(10.0);
    assert_eq!(loss_fn.lambda_gp(), 10.0);
    assert_eq!(loss_fn.name(), "Wasserstein");

    let real = Vector::from_slice(&[2.0, 2.0]);
    let fake = Vector::from_slice(&[1.0, 1.0]);
    let d_loss = loss_fn.discriminator_loss(&real, &fake);
    assert!(d_loss > 0.0);
}
