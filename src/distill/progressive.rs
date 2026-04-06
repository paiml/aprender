//! Progressive layer-wise distillation

use ndarray::{Array2, Axis};

/// Progressive Layer-wise Distillation
///
/// Distills knowledge progressively from intermediate layers of the teacher
/// to corresponding layers of the student. This helps the student learn
/// better intermediate representations, not just final predictions.
///
/// # Approach
///
/// Instead of only matching final logits, progressive distillation also
/// matches intermediate layer outputs (hidden states, attention weights, etc.)
/// between teacher and student at multiple depths.
///
/// # Example
///
/// ```
/// use entrenar::distill::ProgressiveDistiller;
///
/// let distiller = ProgressiveDistiller::new(vec![1.0, 1.0, 2.0], 2.0);
///
/// // Match intermediate layers (e.g., layer 3, 6, 9 of teacher to student)
/// // let loss = distiller.layer_wise_loss(&student_hiddens, &teacher_hiddens);
/// ```
#[derive(Debug, Clone)]
pub struct ProgressiveDistiller {
    /// Weight for each layer's distillation loss
    pub layer_weights: Vec<f32>,
    /// Temperature for distillation
    pub temperature: f32,
}

impl ProgressiveDistiller {
    /// Create a new progressive distiller
    ///
    /// # Arguments
    ///
    /// * `layer_weights` - Weight for each layer (will be normalized)
    /// * `temperature` - Temperature for softening distributions
    ///
    /// # Panics
    ///
    /// Panics if layer_weights is empty or temperature <= 0
    pub fn new(layer_weights: Vec<f32>, temperature: f32) -> Self {
        assert!(!layer_weights.is_empty(), "Must have at least one layer weight");
        assert!(temperature > 0.0, "Temperature must be positive, got {temperature}");

        let sum: f32 = layer_weights.iter().sum();
        assert!(sum > 0.0, "Layer weights must sum to positive value");

        // Normalize weights
        let normalized: Vec<f32> = layer_weights.iter().map(|&w| w / sum).collect();

        Self { layer_weights: normalized, temperature }
    }

    /// Create progressive distiller with uniform layer weights
    pub fn uniform(num_layers: usize, temperature: f32) -> Self {
        Self::new(vec![1.0; num_layers], temperature)
    }

    /// Compute layer-wise MSE loss between student and teacher hidden states
    ///
    /// # Arguments
    ///
    /// * `student_hiddens` - Hidden states from student layers `[num_layers]`
    /// * `teacher_hiddens` - Hidden states from teacher layers `[num_layers]`
    ///
    /// # Returns
    ///
    /// Weighted MSE loss across all layers
    pub fn layer_wise_mse_loss(
        &self,
        student_hiddens: &[Array2<f32>],
        teacher_hiddens: &[Array2<f32>],
    ) -> f32 {
        assert_eq!(
            student_hiddens.len(),
            teacher_hiddens.len(),
            "Number of layers must match (student vs teacher)"
        );
        assert_eq!(
            student_hiddens.len(),
            self.layer_weights.len(),
            "Number of layers must match (student vs weights)"
        );

        let mut total_loss = 0.0;

        for ((student, teacher), &weight) in
            student_hiddens.iter().zip(teacher_hiddens).zip(&self.layer_weights)
        {
            assert_eq!(
                student.shape(),
                teacher.shape(),
                "Student and teacher hidden states must have same shape"
            );

            let mse = mse_loss(student, teacher);
            total_loss += weight * mse;
        }

        total_loss
    }

    /// Compute layer-wise cosine similarity loss
    ///
    /// Encourages student representations to have similar direction to teacher,
    /// which can be more robust than MSE.
    pub fn layer_wise_cosine_loss(
        &self,
        student_hiddens: &[Array2<f32>],
        teacher_hiddens: &[Array2<f32>],
    ) -> f32 {
        assert_eq!(
            student_hiddens.len(),
            teacher_hiddens.len(),
            "Number of layers must match (student vs teacher)"
        );
        assert_eq!(
            student_hiddens.len(),
            self.layer_weights.len(),
            "Number of layers must match (student vs weights)"
        );

        let mut total_loss = 0.0;

        for ((student, teacher), &weight) in
            student_hiddens.iter().zip(teacher_hiddens).zip(&self.layer_weights)
        {
            assert_eq!(
                student.shape(),
                teacher.shape(),
                "Student and teacher hidden states must have same shape"
            );

            // Cosine loss = 1 - cosine_similarity
            let cos_sim = cosine_similarity(student, teacher);
            total_loss += weight * (1.0 - cos_sim);
        }

        total_loss
    }

    /// Combined progressive distillation loss
    ///
    /// Combines final logit distillation with intermediate layer matching.
    ///
    /// # Arguments
    ///
    /// * `student_logits` - Final logits from student
    /// * `teacher_logits` - Final logits from teacher
    /// * `student_hiddens` - Intermediate hidden states from student
    /// * `teacher_hiddens` - Intermediate hidden states from teacher
    /// * `labels` - Ground truth labels
    /// * `alpha` - Weight for distillation vs hard loss
    /// * `beta` - Weight for hidden state matching vs logit matching
    #[allow(clippy::too_many_arguments)]
    pub fn combined_loss(
        &self,
        student_logits: &Array2<f32>,
        teacher_logits: &Array2<f32>,
        student_hiddens: &[Array2<f32>],
        teacher_hiddens: &[Array2<f32>],
        labels: &[usize],
        alpha: f32,
        beta: f32,
    ) -> f32 {
        use super::loss::DistillationLoss;

        // Final logit distillation
        let logit_loss = DistillationLoss::new(self.temperature, alpha);
        let logit_distill = logit_loss.forward(student_logits, teacher_logits, labels);

        // Intermediate layer matching
        let hidden_loss = self.layer_wise_cosine_loss(student_hiddens, teacher_hiddens);

        // Combine
        (1.0 - beta) * logit_distill + beta * hidden_loss
    }
}

/// Compute MSE loss between two arrays
fn mse_loss(student: &Array2<f32>, teacher: &Array2<f32>) -> f32 {
    assert_eq!(student.shape(), teacher.shape());

    let diff = student - teacher;
    let squared = diff.mapv(|x| x * x);
    squared.mean().unwrap_or(0.0)
}

/// Compute cosine similarity between two arrays
///
/// cosine_sim(a, b) = (a · b) / (||a|| * ||b||)
///
/// Averaged over batch dimension.
fn cosine_similarity(student: &Array2<f32>, teacher: &Array2<f32>) -> f32 {
    assert_eq!(student.shape(), teacher.shape());

    let batch_size = student.nrows();
    if batch_size == 0 {
        return 0.0;
    }

    let mut total_sim = 0.0;

    for (s_row, t_row) in student.axis_iter(Axis(0)).zip(teacher.axis_iter(Axis(0))) {
        let dot: f32 = s_row.iter().zip(t_row.iter()).map(|(a, b)| a * b).sum();
        let s_norm: f32 = s_row.iter().map(|x| x * x).sum::<f32>().sqrt();
        let t_norm: f32 = t_row.iter().map(|x| x * x).sum::<f32>().sqrt();

        if s_norm > 1e-10 && t_norm > 1e-10 {
            total_sim += dot / (s_norm * t_norm);
        }
    }

    total_sim / batch_size as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array;

    #[test]
    fn test_uniform_progressive() {
        let distiller = ProgressiveDistiller::uniform(3, 2.0);
        assert_eq!(distiller.layer_weights.len(), 3);
        assert_relative_eq!(distiller.layer_weights.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
        for &w in &distiller.layer_weights {
            assert_relative_eq!(w, 1.0 / 3.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_weighted_progressive() {
        let distiller = ProgressiveDistiller::new(vec![1.0, 2.0, 3.0], 2.0);
        assert_relative_eq!(distiller.layer_weights.iter().sum::<f32>(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_loss_zero_for_identical() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mse = mse_loss(&a, &a);
        assert_relative_eq!(mse, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_loss_positive() {
        let a = array![[1.0, 2.0, 3.0]];
        let b = array![[2.0, 3.0, 4.0]];
        let mse = mse_loss(&a, &b);
        assert!(mse > 0.0);
        // MSE = mean((1,1,1)^2) = 1.0
        assert_relative_eq!(mse, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_similarity_one_for_identical() {
        let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let cos = cosine_similarity(&a, &a);
        assert_relative_eq!(cos, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_for_orthogonal() {
        let a = array![[1.0, 0.0]];
        let b = array![[0.0, 1.0]];
        let cos = cosine_similarity(&a, &b);
        assert_relative_eq!(cos, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_similarity_positive() {
        let a = array![[1.0, 2.0, 3.0]];
        let b = array![[2.0, 4.0, 6.0]]; // Same direction, scaled
        let cos = cosine_similarity(&a, &b);
        assert_relative_eq!(cos, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_layer_wise_mse_loss() {
        let distiller = ProgressiveDistiller::uniform(2, 2.0);

        let student_hiddens = vec![array![[1.0, 2.0], [3.0, 4.0]], array![[5.0, 6.0], [7.0, 8.0]]];
        let teacher_hiddens = vec![array![[1.1, 2.1], [3.1, 4.1]], array![[5.1, 6.1], [7.1, 8.1]]];

        let loss = distiller.layer_wise_mse_loss(&student_hiddens, &teacher_hiddens);
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_layer_wise_cosine_loss() {
        let distiller = ProgressiveDistiller::uniform(2, 2.0);

        let student_hiddens = vec![array![[1.0, 2.0], [3.0, 4.0]], array![[5.0, 6.0], [7.0, 8.0]]];
        let teacher_hiddens = vec![array![[1.1, 2.1], [3.1, 4.1]], array![[5.1, 6.1], [7.1, 8.1]]];

        let loss = distiller.layer_wise_cosine_loss(&student_hiddens, &teacher_hiddens);
        assert!(loss >= 0.0); // Cosine loss should be >= 0
        assert!(loss.is_finite());
    }

    #[test]
    fn test_combined_loss() {
        let distiller = ProgressiveDistiller::uniform(2, 2.0);

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
            0.7, // alpha
            0.3, // beta
        );

        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    #[should_panic(expected = "Must have at least one layer weight")]
    fn test_empty_layers_panics() {
        ProgressiveDistiller::new(vec![], 2.0);
    }

    #[test]
    #[should_panic(expected = "Temperature must be positive")]
    fn test_invalid_temperature_panics() {
        ProgressiveDistiller::new(vec![1.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "Number of layers must match")]
    fn test_mismatched_layers_panics() {
        let distiller = ProgressiveDistiller::uniform(2, 2.0);
        let student = vec![array![[1.0, 2.0]]]; // 1 layer
        let teacher = vec![array![[1.0, 2.0]], array![[3.0, 4.0]]]; // 2 layers
        distiller.layer_wise_mse_loss(&student, &teacher);
    }
}
