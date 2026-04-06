//! Property-based tests for distillation

#![allow(clippy::field_reassign_with_default)]

use super::*;
use ndarray::{array, Array2};
use proptest::prelude::*;

// Helper to generate random logits
fn logits_strategy(batch_size: usize, num_classes: usize) -> impl Strategy<Value = Array2<f32>> {
    prop::collection::vec(prop::collection::vec(-10.0f32..10.0, num_classes), batch_size).prop_map(
        move |data| {
            let flat: Vec<f32> = data.into_iter().flatten().collect();
            Array2::from_shape_vec((batch_size, num_classes), flat)
                .expect("operation should succeed")
        },
    )
}

// Helper to generate random hidden states
fn hiddens_strategy(batch_size: usize, hidden_dim: usize) -> impl Strategy<Value = Array2<f32>> {
    prop::collection::vec(prop::collection::vec(-5.0f32..5.0, hidden_dim), batch_size).prop_map(
        move |data| {
            let flat: Vec<f32> = data.into_iter().flatten().collect();
            Array2::from_shape_vec((batch_size, hidden_dim), flat)
                .expect("operation should succeed")
        },
    )
}

proptest! {
    /// Distillation loss should always be non-negative
    #[test]
    fn prop_distillation_loss_non_negative(
        temperature in 0.5f32..10.0,
        alpha in 0.0f32..=1.0,
    ) {
        let loss_fn = DistillationLoss::new(temperature, alpha);

        let student = array![[2.0, 1.0, 0.5], [1.5, 2.5, 0.8]];
        let teacher = array![[1.8, 1.1, 0.6], [1.6, 2.3, 0.9]];
        let labels = vec![0, 1];

        let loss = loss_fn.forward(&student, &teacher, &labels);

        prop_assert!(loss >= 0.0);
        prop_assert!(loss.is_finite());
    }

    /// Higher temperature should smooth distributions more
    #[test]
    fn prop_temperature_smooths_distributions(
        temp_low in 0.5f32..2.0,
        temp_high in 5.0f32..10.0,
    ) {
        let student = array![[10.0, 1.0, 0.1]]; // Sharp distribution
        let teacher = array![[5.0, 4.0, 3.0]];  // More balanced
        let labels = vec![0];

        let loss_low = DistillationLoss::new(temp_low, 1.0);
        let loss_high = DistillationLoss::new(temp_high, 1.0);

        let kl_low = loss_low.forward(&student, &teacher, &labels);
        let kl_high = loss_high.forward(&student, &teacher, &labels);

        // Both should be finite and non-negative
        prop_assert!(kl_low.is_finite() && kl_low >= 0.0);
        prop_assert!(kl_high.is_finite() && kl_high >= 0.0);
    }

    /// When alpha=0, should only use hard loss (independent of teacher)
    #[test]
    fn prop_alpha_zero_ignores_teacher(
        student in logits_strategy(2, 3),
    ) {
        let teacher1 = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let teacher2 = array![[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]];
        let labels = vec![0, 1];

        let loss_fn = DistillationLoss::new(2.0, 0.0); // alpha=0

        let loss1 = loss_fn.forward(&student, &teacher1, &labels);
        let loss2 = loss_fn.forward(&student, &teacher2, &labels);

        // Losses should be identical since teacher is ignored
        prop_assert!((loss1 - loss2).abs() < 1e-4);
    }

    /// When alpha=1, hard loss should not affect result
    #[test]
    fn prop_alpha_one_uses_only_soft_targets(
        student in logits_strategy(2, 3),
        teacher in logits_strategy(2, 3),
    ) {
        let labels1 = vec![0, 0];
        let labels2 = vec![1, 2];

        let loss_fn = DistillationLoss::new(2.0, 1.0); // alpha=1

        let loss1 = loss_fn.forward(&student, &teacher, &labels1);
        let loss2 = loss_fn.forward(&student, &teacher, &labels2);

        // Losses should be identical since labels are ignored
        prop_assert!((loss1 - loss2).abs() < 1e-4);
    }

    /// Ensemble with uniform weights should average teachers
    #[test]
    fn prop_ensemble_averages_uniformly(
        t1 in logits_strategy(2, 3),
        t2 in logits_strategy(2, 3),
    ) {
        let distiller = EnsembleDistiller::uniform(2, 2.0);
        let teachers = vec![t1.clone(), t2.clone()];

        let ensemble = distiller.combine_teachers(&teachers);

        // Check that ensemble is average of teachers
        for i in 0..t1.nrows() {
            for j in 0..t1.ncols() {
                let expected = f32::midpoint(t1[[i, j]], t2[[i, j]]);
                prop_assert!((ensemble[[i, j]] - expected).abs() < 1e-4);
            }
        }
    }

    /// MSE loss should be zero for identical inputs
    #[test]
    fn prop_mse_zero_for_identical(
        hidden in hiddens_strategy(2, 4),
    ) {
        use super::progressive::ProgressiveDistiller;

        let distiller = ProgressiveDistiller::uniform(1, 2.0);
        let hiddens_vec = vec![hidden.clone()];

        let loss = distiller.layer_wise_mse_loss(&hiddens_vec, &hiddens_vec);

        prop_assert!(loss.abs() < 1e-5);
    }

    /// Cosine similarity should be 1 for identical inputs
    #[test]
    fn prop_cosine_one_for_identical(
        hidden in hiddens_strategy(2, 4),
    ) {
        use super::progressive::ProgressiveDistiller;

        let distiller = ProgressiveDistiller::uniform(1, 2.0);
        let hiddens_vec = vec![hidden.clone()];

        let loss = distiller.layer_wise_cosine_loss(&hiddens_vec, &hiddens_vec);

        // Cosine loss = 1 - cosine_similarity
        // For identical inputs, cosine_similarity = 1, so loss = 0
        prop_assert!(loss.abs() < 1e-5);
    }

    /// MSE loss should be symmetric
    #[test]
    fn prop_mse_symmetric(
        h1 in hiddens_strategy(2, 4),
        h2 in hiddens_strategy(2, 4),
    ) {
        use super::progressive::ProgressiveDistiller;

        let distiller = ProgressiveDistiller::uniform(1, 2.0);

        let loss1 = distiller.layer_wise_mse_loss(&[h1.clone()], &[h2.clone()]);
        let loss2 = distiller.layer_wise_mse_loss(&[h2.clone()], &[h1.clone()]);

        prop_assert!((loss1 - loss2).abs() < 1e-5);
    }

    /// Cosine similarity should be symmetric
    #[test]
    fn prop_cosine_symmetric(
        h1 in hiddens_strategy(2, 4),
        h2 in hiddens_strategy(2, 4),
    ) {
        use super::progressive::ProgressiveDistiller;

        let distiller = ProgressiveDistiller::uniform(1, 2.0);

        let loss1 = distiller.layer_wise_cosine_loss(&[h1.clone()], &[h2.clone()]);
        let loss2 = distiller.layer_wise_cosine_loss(&[h2.clone()], &[h1.clone()]);

        prop_assert!((loss1 - loss2).abs() < 1e-5);
    }

    /// Progressive distillation loss should be finite and non-negative
    #[test]
    fn prop_progressive_loss_valid(
        temperature in 0.5f32..10.0,
        alpha in 0.0f32..=1.0,
        beta in 0.0f32..=1.0,
    ) {
        use super::progressive::ProgressiveDistiller;

        let distiller = ProgressiveDistiller::uniform(2, temperature);

        let student_logits = array![[2.0, 1.0, 0.5]];
        let teacher_logits = array![[1.8, 1.1, 0.6]];
        let student_hiddens = vec![array![[1.0, 2.0]], array![[3.0, 4.0]]];
        let teacher_hiddens = vec![array![[1.1, 2.1]], array![[3.1, 4.1]]];
        let labels = vec![0];

        let loss = distiller.combined_loss(
            &student_logits,
            &teacher_logits,
            &student_hiddens,
            &teacher_hiddens,
            &labels,
            alpha,
            beta,
        );

        prop_assert!(loss >= 0.0);
        prop_assert!(loss.is_finite());
    }

    /// Weighted ensemble should respect weights
    #[test]
    fn prop_ensemble_respects_weights(
        w1 in 0.1f32..10.0,
        w2 in 0.1f32..10.0,
    ) {
        let distiller = EnsembleDistiller::new(vec![w1, w2], 2.0);

        // Weights should be normalized
        let sum: f32 = distiller.weights.iter().sum();
        prop_assert!((sum - 1.0).abs() < 1e-5);

        // Individual weights should be proportional to input
        let expected_w1 = w1 / (w1 + w2);
        let expected_w2 = w2 / (w1 + w2);
        prop_assert!((distiller.weights[0] - expected_w1).abs() < 1e-5);
        prop_assert!((distiller.weights[1] - expected_w2).abs() < 1e-5);
    }
}

// Integration tests
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_distillation() {
        // Simulate a simple distillation scenario
        let loss_fn = DistillationLoss::new(3.0, 0.7);

        // Teacher has confident predictions
        let teacher = array![[10.0, 2.0, 1.0], [1.0, 12.0, 2.0], [2.0, 1.0, 11.0]];

        // Student initially random
        let student = array![[5.0, 5.0, 5.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]];

        let labels = vec![0, 1, 2];

        let initial_loss = loss_fn.forward(&student, &teacher, &labels);

        // After "training", student should have improved predictions
        let student_improved = array![[9.0, 3.0, 2.0], [2.0, 10.0, 3.0], [3.0, 2.0, 9.0]];

        let final_loss = loss_fn.forward(&student_improved, &teacher, &labels);

        // Loss should decrease after improvement
        assert!(final_loss < initial_loss);
    }

    #[test]
    fn test_multi_teacher_ensemble() {
        let distiller = EnsembleDistiller::new(vec![2.0, 1.0, 1.0], 2.0);

        let t1 = array![[10.0, 1.0, 1.0]]; // Strong on class 0
        let t2 = array![[1.0, 10.0, 1.0]]; // Strong on class 1
        let t3 = array![[1.0, 1.0, 10.0]]; // Strong on class 2
        let teachers = vec![t1, t2, t3];

        let ensemble = distiller.combine_teachers(&teachers);

        // Ensemble should be influenced most by t1 (weight=2.0)
        assert!(ensemble[[0, 0]] > ensemble[[0, 1]]);
        assert!(ensemble[[0, 0]] > ensemble[[0, 2]]);
    }

    #[test]
    fn test_progressive_multi_layer() {
        use super::progressive::ProgressiveDistiller;

        // More weight on later layers
        let distiller = ProgressiveDistiller::new(vec![0.5, 1.0, 2.0], 2.0);

        let student_hiddens = vec![array![[1.0, 2.0]], array![[3.0, 4.0]], array![[5.0, 6.0]]];

        let teacher_hiddens = vec![
            array![[1.1, 2.1]],
            array![[3.1, 4.1]],
            array![[5.5, 6.5]], // Larger diff in last layer
        ];

        let loss = distiller.layer_wise_mse_loss(&student_hiddens, &teacher_hiddens);

        // Loss should be dominated by last layer due to higher weight
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }
}
