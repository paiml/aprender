//! Multi-teacher ensemble distillation

use ndarray::{Array2, Axis};

/// Multi-Teacher Ensemble Distillation
///
/// Distills knowledge from multiple teacher models into a single student.
/// Combines teacher predictions via averaging or weighted averaging.
///
/// # Methods
///
/// - **Average**: Simple mean of all teacher logits
/// - **Weighted**: Weighted combination based on teacher confidence/accuracy
///
/// # Example
///
/// ```
/// use entrenar::distill::EnsembleDistiller;
/// use ndarray::array;
///
/// let distiller = EnsembleDistiller::new(vec![1.0, 1.0], 2.0);
/// let teachers = vec![
///     array![[2.0, 1.0, 0.5]],
///     array![[1.5, 1.2, 0.8]],
/// ];
/// let ensemble_logits = distiller.combine_teachers(&teachers);
/// ```
#[derive(Debug, Clone)]
pub struct EnsembleDistiller {
    /// Weights for each teacher (normalized to sum to 1)
    pub weights: Vec<f32>,
    /// Temperature for distillation
    pub temperature: f32,
}

impl EnsembleDistiller {
    /// Create a new ensemble distiller with given teacher weights
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight for each teacher (will be normalized)
    /// * `temperature` - Temperature for softening distributions
    ///
    /// # Panics
    ///
    /// Panics if weights are empty, all zero, or temperature <= 0
    pub fn new(weights: Vec<f32>, temperature: f32) -> Self {
        assert!(!weights.is_empty(), "Must have at least one teacher");
        assert!(temperature > 0.0, "Temperature must be positive, got {temperature}");

        let sum: f32 = weights.iter().sum();
        assert!(sum > 0.0, "Teacher weights must sum to positive value");

        // Normalize weights
        let normalized_weights: Vec<f32> = weights.iter().map(|&w| w / sum).collect();

        Self { weights: normalized_weights, temperature }
    }

    /// Create an ensemble with uniform weights
    pub fn uniform(num_teachers: usize, temperature: f32) -> Self {
        Self::new(vec![1.0; num_teachers], temperature)
    }

    /// Combine multiple teacher logits into ensemble prediction
    ///
    /// # Arguments
    ///
    /// * `teacher_logits` - Vector of teacher logits, each [batch_size, num_classes]
    ///
    /// # Returns
    ///
    /// Weighted average of teacher logits [batch_size, num_classes]
    pub fn combine_teachers(&self, teacher_logits: &[Array2<f32>]) -> Array2<f32> {
        assert_eq!(
            teacher_logits.len(),
            self.weights.len(),
            "Number of teachers must match number of weights"
        );
        assert!(!teacher_logits.is_empty(), "Must have at least one teacher");

        // Check all teachers have same shape
        let shape = teacher_logits[0].shape();
        for t in teacher_logits.iter().skip(1) {
            assert_eq!(t.shape(), shape, "All teacher logits must have the same shape");
        }

        // Weighted average
        let mut ensemble = Array2::zeros((shape[0], shape[1]));

        for (teacher, &weight) in teacher_logits.iter().zip(&self.weights) {
            ensemble = ensemble + teacher * weight;
        }

        ensemble
    }

    /// Combine teachers via probability distribution averaging (more stable)
    ///
    /// Converts each teacher's logits to probabilities, averages them,
    /// then converts back to logits.
    pub fn combine_via_probabilities(&self, teacher_logits: &[Array2<f32>]) -> Array2<f32> {
        assert_eq!(
            teacher_logits.len(),
            self.weights.len(),
            "Number of teachers must match number of weights"
        );
        assert!(!teacher_logits.is_empty(), "Must have at least one teacher");

        let shape = teacher_logits[0].shape();

        // Convert each teacher to probabilities
        let teacher_probs: Vec<Array2<f32>> =
            teacher_logits.iter().map(|logits| softmax_2d(&(logits / self.temperature))).collect();

        // Weighted average of probabilities
        let mut ensemble_probs = Array2::zeros((shape[0], shape[1]));
        for (probs, &weight) in teacher_probs.iter().zip(&self.weights) {
            ensemble_probs = ensemble_probs + probs * weight;
        }

        // Convert back to logits (inverse softmax via log)
        // Note: This is approximate - exact inverse doesn't exist
        ensemble_probs.mapv(|p: f32| (p + 1e-10_f32).max(f32::MIN_POSITIVE).ln() * self.temperature)
    }

    /// Compute ensemble distillation loss
    ///
    /// # Arguments
    ///
    /// * `student_logits` - Logits from student [batch_size, num_classes]
    /// * `teacher_logits` - Vector of teacher logits
    /// * `labels` - Ground truth labels `[batch_size]`
    /// * `alpha` - Weight for distillation vs hard loss
    pub fn distillation_loss(
        &self,
        student_logits: &Array2<f32>,
        teacher_logits: &[Array2<f32>],
        labels: &[usize],
        alpha: f32,
    ) -> f32 {
        use super::loss::DistillationLoss;

        let ensemble_logits = self.combine_teachers(teacher_logits);
        let loss_fn = DistillationLoss::new(self.temperature, alpha);

        loss_fn.forward(student_logits, &ensemble_logits, labels)
    }
}

/// Compute softmax along last axis for 2D array
fn softmax_2d(x: &Array2<f32>) -> Array2<f32> {
    let mut result = x.clone();

    for mut row in result.axis_iter_mut(Axis(0)) {
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        row.mapv_inplace(|v| (v - max_val).exp());
        let sum: f32 = row.sum();
        row.mapv_inplace(|v| v / sum);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_uniform_ensemble() {
        let distiller = EnsembleDistiller::uniform(3, 2.0);
        assert_eq!(distiller.weights.len(), 3);
        assert_relative_eq!(distiller.weights.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
        for &w in &distiller.weights {
            assert_relative_eq!(w, 1.0 / 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_weighted_ensemble() {
        let distiller = EnsembleDistiller::new(vec![1.0, 2.0, 3.0], 2.0);
        assert_relative_eq!(distiller.weights.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(distiller.weights[0], 1.0 / 6.0, epsilon = 1e-6);
        assert_relative_eq!(distiller.weights[1], 2.0 / 6.0, epsilon = 1e-6);
        assert_relative_eq!(distiller.weights[2], 3.0 / 6.0, epsilon = 1e-6);
    }

    #[test]
    fn test_combine_teachers() {
        let distiller = EnsembleDistiller::uniform(2, 2.0);

        let t1 = array![[1.0, 2.0, 3.0]];
        let t2 = array![[3.0, 2.0, 1.0]];
        let teachers = vec![t1, t2];

        let ensemble = distiller.combine_teachers(&teachers);

        // Should be average: (1+3)/2=2, (2+2)/2=2, (3+1)/2=2
        assert_relative_eq!(ensemble[[0, 0]], 2.0, epsilon = 1e-6);
        assert_relative_eq!(ensemble[[0, 1]], 2.0, epsilon = 1e-6);
        assert_relative_eq!(ensemble[[0, 2]], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_weighted_combine() {
        let distiller = EnsembleDistiller::new(vec![1.0, 3.0], 2.0);

        let t1 = array![[1.0, 2.0, 3.0]];
        let t2 = array![[3.0, 2.0, 1.0]];
        let teachers = vec![t1, t2];

        let ensemble = distiller.combine_teachers(&teachers);

        // Should be weighted average: (1*0.25 + 3*0.75) = 2.5
        assert_relative_eq!(ensemble[[0, 0]], 2.5, epsilon = 1e-6);
        // (2*0.25 + 2*0.75) = 2.0
        assert_relative_eq!(ensemble[[0, 1]], 2.0, epsilon = 1e-6);
        // (3*0.25 + 1*0.75) = 1.5
        assert_relative_eq!(ensemble[[0, 2]], 1.5, epsilon = 1e-6);
    }

    #[test]
    fn test_combine_via_probabilities() {
        let distiller = EnsembleDistiller::uniform(2, 2.0);

        let t1 = array![[2.0, 1.0, 0.5]];
        let t2 = array![[1.5, 1.2, 0.8]];
        let teachers = vec![t1, t2];

        let ensemble = distiller.combine_via_probabilities(&teachers);

        // Result should be finite and reasonable
        assert!(ensemble.iter().all(|&x| x.is_finite()));
    }

    #[test]
    #[should_panic(expected = "Must have at least one teacher")]
    fn test_empty_weights_panics() {
        EnsembleDistiller::new(vec![], 2.0);
    }

    #[test]
    #[should_panic(expected = "Teacher weights must sum to positive")]
    fn test_zero_weights_panics() {
        EnsembleDistiller::new(vec![0.0, 0.0], 2.0);
    }

    #[test]
    #[should_panic(expected = "Number of teachers must match")]
    fn test_mismatched_teachers_panics() {
        let distiller = EnsembleDistiller::uniform(2, 2.0);
        let teachers = vec![array![[1.0, 2.0]]]; // Only 1 teacher
        distiller.combine_teachers(&teachers);
    }

    #[test]
    fn test_distillation_loss() {
        let distiller = EnsembleDistiller::uniform(2, 2.0);

        let student = array![[2.0, 1.0, 0.5]];
        let t1 = array![[1.8, 1.1, 0.6]];
        let t2 = array![[1.9, 0.9, 0.7]];
        let teachers = vec![t1, t2];
        let labels = vec![0];

        let loss = distiller.distillation_loss(&student, &teachers, &labels, 0.7);

        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }
}
