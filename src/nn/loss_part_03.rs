#[cfg(test)]
mod tests {
    use super::super::ucbd::{abs, log_softmax, softmax_2d};
    use super::super::*;
    use crate::autograd::clear_graph;

    #[test]
    fn test_mse_loss_zero() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let target = Tensor::from_slice(&[1.0, 2.0, 3.0]);

        let criterion = MSELoss::new();
        let loss = criterion.forward(&pred, &target);

        assert!(loss.item() < 1e-6);
    }

    #[test]
    fn test_mse_loss_nonzero() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let target = Tensor::from_slice(&[2.0, 2.0, 2.0]);

        let criterion = MSELoss::new();
        let loss = criterion.forward(&pred, &target);

        // MSE = ((1-2)² + (2-2)² + (3-2)²) / 3 = (1 + 0 + 1) / 3 = 2/3
        assert!((loss.item() - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_mse_loss_gradient() {
        clear_graph();

        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]).requires_grad();
        let pred_id = pred.id();
        let target = Tensor::from_slice(&[2.0, 2.0, 2.0]);

        let criterion = MSELoss::new();
        let loss = criterion.forward(&pred, &target);
        loss.backward();

        let grad = crate::autograd::get_grad(pred_id).expect("Should have gradient");

        // Gradient of MSE: 2 * (pred - target) / n
        // = 2 * [-1, 0, 1] / 3 = [-2/3, 0, 2/3]
        let expected = [-2.0 / 3.0, 0.0, 2.0 / 3.0];
        for (g, e) in grad.data().iter().zip(expected.iter()) {
            assert!((g - e).abs() < 1e-5, "Expected {e}, got {g}");
        }
    }

    #[test]
    fn test_l1_loss() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0]);
        let target = Tensor::from_slice(&[2.0, 2.0, 2.0]);

        let criterion = L1Loss::new();
        let loss = criterion.forward(&pred, &target);

        // MAE = (|1-2| + |2-2| + |3-2|) / 3 = (1 + 0 + 1) / 3 = 2/3
        assert!((loss.item() - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_l1_loss() {
        let pred = Tensor::from_slice(&[0.0]);
        let target = Tensor::from_slice(&[0.5]);

        let criterion = SmoothL1Loss::new();
        let loss = criterion.forward(&pred, &target);

        // |x| = 0.5 < 1.0 (beta), so loss = 0.5 * 0.5² / 1.0 = 0.125
        assert!((loss.item() - 0.125).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Two samples, 3 classes
        // Sample 0: logits [1, 2, 0.5], target 1 (class with logit 2)
        // Sample 1: logits [0.1, 3, 0.2], target 1 (class with logit 3)
        let logits = Tensor::new(&[1.0, 2.0, 0.5, 0.1, 3.0, 0.2], &[2, 3]);
        let targets = Tensor::from_slice(&[1.0, 1.0]);

        let criterion = CrossEntropyLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Loss should be relatively small since we're predicting the correct class
        // (the class with highest logit matches target)
        assert!(loss.item() < 1.0);
    }

    #[test]
    fn test_cross_entropy_wrong_prediction() {
        // Logits favor class 0, but target is class 2
        let logits = Tensor::new(&[10.0, 0.0, 0.0], &[1, 3]);
        let targets = Tensor::from_slice(&[2.0]);

        let criterion = CrossEntropyLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Loss should be high since prediction is wrong
        assert!(loss.item() > 5.0);
    }

    #[test]
    fn test_bce_with_logits() {
        // Perfect prediction: logit = 10 (high sigmoid) for target = 1
        let logits = Tensor::from_slice(&[10.0]);
        let targets = Tensor::from_slice(&[1.0]);

        let criterion = BCEWithLogitsLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Loss should be very small
        assert!(loss.item() < 0.001);
    }

    #[test]
    fn test_bce_with_logits_wrong() {
        // Wrong prediction: logit = 10 (high sigmoid) for target = 0
        let logits = Tensor::from_slice(&[10.0]);
        let targets = Tensor::from_slice(&[0.0]);

        let criterion = BCEWithLogitsLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Loss should be high
        assert!(loss.item() > 5.0);
    }

    #[test]
    fn test_nll_loss() {
        // Log-probs where class 1 has highest probability
        let log_probs = Tensor::new(&[-2.0, -0.1, -3.0], &[1, 3]);
        let targets = Tensor::from_slice(&[1.0]);

        let criterion = NLLLoss::new();
        let loss = criterion.forward(&log_probs, &targets);

        // NLL = -log_probs[target] = -(-0.1) = 0.1
        assert!((loss.item() - 0.1).abs() < 1e-5);
    }

    #[test]
    fn test_reduction_modes() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let target = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0]);

        // None: returns per-element loss
        let criterion = MSELoss::with_reduction(Reduction::None);
        let loss = criterion.forward(&pred, &target);
        assert_eq!(loss.shape(), &[4]);
        assert_eq!(loss.data(), &[1.0, 4.0, 9.0, 16.0]);

        // Sum
        let criterion = MSELoss::with_reduction(Reduction::Sum);
        let loss = criterion.forward(&pred, &target);
        assert!((loss.item() - 30.0).abs() < 1e-5);

        // Mean
        let criterion = MSELoss::with_reduction(Reduction::Mean);
        let loss = criterion.forward(&pred, &target);
        assert!((loss.item() - 7.5).abs() < 1e-5);
    }

    #[test]
    fn test_cross_entropy_gradient() {
        clear_graph();

        // Simple classification: 2 samples, 3 classes
        // logits favor class 1 for both samples
        let logits = Tensor::new(&[0.0, 2.0, 0.0, 0.0, 2.0, 0.0], &[2, 3]).requires_grad();
        let logits_id = logits.id();
        let targets = Tensor::from_slice(&[1.0, 1.0]); // Both target class 1

        let criterion = CrossEntropyLoss::new();
        let loss = criterion.forward(&logits, &targets);

        // Verify loss is computed
        assert!(
            loss.item() < 1.0,
            "Loss should be small for correct predictions"
        );

        // Backward pass
        loss.backward();

        // Verify gradients exist
        let grad = crate::autograd::get_grad(logits_id).expect("Should have gradient");
        assert_eq!(grad.shape(), &[2, 3], "Gradient shape should match logits");

        // Gradient for cross-entropy: softmax(logits) - one_hot(targets)
        // For logits [0, 2, 0], softmax ≈ [0.106, 0.788, 0.106]
        // Target class 1, so gradient ≈ [0.106, -0.212, 0.106] (after mean reduction)
        // Check that gradient for target class is negative (should decrease)
        let grad_data = grad.data();
        // Sample 0, class 1 (target): gradient should be negative
        assert!(
            grad_data[1] < 0.0,
            "Gradient for target class should be negative"
        );
        // Sample 0, class 0 (non-target): gradient should be positive
        assert!(
            grad_data[0] > 0.0,
            "Gradient for non-target class should be positive"
        );
    }

    // =========================================================================
    // Additional Coverage Tests
    // =========================================================================

    #[test]
    fn test_reduction_enum_default() {
        let reduction = Reduction::default();
        assert_eq!(reduction, Reduction::Mean);
    }

    #[test]
    fn test_reduction_enum_clone() {
        let r1 = Reduction::Sum;
        let r2 = r1.clone();
        assert_eq!(r2, Reduction::Sum);
    }

    #[test]
    fn test_mse_loss_default() {
        let loss = MSELoss::default();
        let debug_str = format!("{:?}", loss);
        assert!(debug_str.contains("MSELoss"));
    }

    #[test]
    fn test_mse_loss_clone() {
        let loss1 = MSELoss::with_reduction(Reduction::Sum);
        let loss2 = loss1.clone();
        let debug_str = format!("{:?}", loss2);
        assert!(debug_str.contains("MSELoss"));
    }

    #[test]
    fn test_l1_loss_with_reduction() {
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let target = Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0]);

        // None
        let criterion = L1Loss::with_reduction(Reduction::None);
        let loss = criterion.forward(&pred, &target);
        assert_eq!(loss.shape(), &[4]);
        assert_eq!(loss.data(), &[1.0, 2.0, 3.0, 4.0]);

        // Sum
        let criterion = L1Loss::with_reduction(Reduction::Sum);
        let loss = criterion.forward(&pred, &target);
        assert!((loss.item() - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_l1_loss_default() {
        let loss = L1Loss::default();
        let debug_str = format!("{:?}", loss);
        assert!(debug_str.contains("L1Loss"));
    }

    #[test]
    fn test_smooth_l1_loss_linear_region() {
        // Test the linear region where |x| >= beta
        let pred = Tensor::from_slice(&[0.0]);
        let target = Tensor::from_slice(&[2.0]); // |diff| = 2 > 1 (beta)

        let criterion = SmoothL1Loss::new();
        let loss = criterion.forward(&pred, &target);

        // |x| = 2 >= 1 (beta), so loss = |x| - 0.5 * beta = 2 - 0.5 = 1.5
        assert!((loss.item() - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_l1_loss_with_beta() {
        let pred = Tensor::from_slice(&[0.0]);
        let target = Tensor::from_slice(&[0.1]); // Small diff

        let criterion = SmoothL1Loss::with_beta(0.5);
        let loss = criterion.forward(&pred, &target);

        // |x| = 0.1 < 0.5 (beta), so loss = 0.5 * 0.1² / 0.5 = 0.01
        assert!((loss.item() - 0.01).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_l1_loss_default() {
        let loss = SmoothL1Loss::default();
        assert!((loss.beta - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_with_reduction_none() {
        let logits = Tensor::new(&[1.0, 2.0, 0.5, 0.1, 3.0, 0.2], &[2, 3]);
        let targets = Tensor::from_slice(&[1.0, 1.0]);

        let criterion = CrossEntropyLoss::with_reduction(Reduction::None);
        let loss = criterion.forward(&logits, &targets);

        assert_eq!(loss.shape(), &[2]);
    }

    #[test]
    fn test_cross_entropy_with_reduction_sum() {
        let logits = Tensor::new(&[1.0, 2.0, 0.5, 0.1, 3.0, 0.2], &[2, 3]);
        let targets = Tensor::from_slice(&[1.0, 1.0]);

        let criterion = CrossEntropyLoss::with_reduction(Reduction::Sum);
        let loss = criterion.forward(&logits, &targets);

        // Should be a scalar
        assert!(loss.item() > 0.0);
    }

    #[test]
    fn test_cross_entropy_with_label_smoothing() {
        let logits = Tensor::new(&[1.0, 2.0, 0.5], &[1, 3]);
        let targets = Tensor::from_slice(&[1.0]);

        let criterion = CrossEntropyLoss::with_label_smoothing(0.1);
        let loss = criterion.forward(&logits, &targets);

        // Loss with label smoothing should be slightly higher than without
        assert!(loss.item() > 0.0);
    }

    #[test]
    fn test_cross_entropy_default() {
        let loss = CrossEntropyLoss::default();
        let debug_str = format!("{:?}", loss);
        assert!(debug_str.contains("CrossEntropyLoss"));
    }

    #[test]
    fn test_bce_with_logits_with_reduction() {
        let logits = Tensor::from_slice(&[1.0, 2.0, -1.0, 0.0]);
        let targets = Tensor::from_slice(&[1.0, 1.0, 0.0, 0.0]);

        // None
        let criterion = BCEWithLogitsLoss::with_reduction(Reduction::None);
        let loss = criterion.forward(&logits, &targets);
        assert_eq!(loss.shape(), &[4]);

        // Sum
        let criterion = BCEWithLogitsLoss::with_reduction(Reduction::Sum);
        let loss = criterion.forward(&logits, &targets);
        assert!(loss.item() > 0.0);
    }

    #[test]
    fn test_bce_with_logits_pos_weight() {
        let logits = Tensor::from_slice(&[1.0, -1.0]);
        let targets = Tensor::from_slice(&[1.0, 0.0]);

        let criterion = BCEWithLogitsLoss::with_pos_weight(2.0);
        let loss = criterion.forward(&logits, &targets);

        // Loss should be higher than without pos_weight due to weighting
        assert!(loss.item() > 0.0);
    }

    #[test]
    fn test_bce_with_logits_default() {
        let loss = BCEWithLogitsLoss::default();
        let debug_str = format!("{:?}", loss);
        assert!(debug_str.contains("BCEWithLogitsLoss"));
    }

    #[test]
    fn test_nll_loss_default() {
        let loss = NLLLoss::default();
        let debug_str = format!("{:?}", loss);
        assert!(debug_str.contains("NLLLoss"));
    }

    #[test]
    fn test_nll_loss_with_sum() {
        // Multiple samples
        let log_probs = Tensor::new(&[-2.0, -0.1, -3.0, -0.2, -1.0, -2.5], &[2, 3]);
        let targets = Tensor::from_slice(&[1.0, 0.0]);

        let criterion = NLLLoss::new();
        let loss = criterion.forward(&log_probs, &targets);

        // Mean of -log_probs[1] and -log_probs[0] = (0.1 + 0.2) / 2 = 0.15
        assert!((loss.item() - 0.15).abs() < 1e-5);
    }

    #[test]
    fn test_softmax_2d_sum_to_one() {
        let x = Tensor::new(&[1.0, 2.0, 3.0, 0.5, 1.5, 2.5], &[2, 3]);
        let softmax = softmax_2d(&x);

        // Each row should sum to 1
        let data = softmax.data();
        let row1_sum: f32 = data[0..3].iter().sum();
        let row2_sum: f32 = data[3..6].iter().sum();

        assert!((row1_sum - 1.0).abs() < 1e-5);
        assert!((row2_sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_property() {
        let x = Tensor::new(&[1.0, 2.0, 3.0], &[1, 3]);
        let log_sm = log_softmax(&x);

        // exp(log_softmax) should sum to 1
        let sum: f32 = log_sm.data().iter().map(|&v| v.exp()).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_abs_helper() {
        let x = Tensor::from_slice(&[-1.0, 2.0, -3.0, 0.0]);
        let result = abs(&x);
        assert_eq!(result.data(), &[1.0, 2.0, 3.0, 0.0]);
    }

    #[test]
    fn test_l1_loss_clone_copy() {
        let loss1 = L1Loss::new();
        let loss2 = loss1; // Copy
        let loss3 = loss2.clone();
        let debug_str = format!("{:?}", loss3);
        assert!(debug_str.contains("L1Loss"));
    }

    #[test]
    fn test_nll_loss_copy() {
        let loss1 = NLLLoss::new();
        let loss2 = loss1; // Copy
        let debug_str = format!("{:?}", loss2);
        assert!(debug_str.contains("NLLLoss"));
    }
}
