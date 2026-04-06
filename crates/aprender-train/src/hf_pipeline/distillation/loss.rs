//! Knowledge Distillation Loss
//!
//! Implements Hinton et al. (2015) distillation loss.

use ndarray::{Array1, Array2};

use super::utils::{cross_entropy_loss, kl_divergence, log_softmax, softmax};

/// Knowledge Distillation Loss
///
/// Implements Hinton et al. (2015) distillation loss:
///
/// ```text
/// L_KD = α * T² * KL(softmax(z_s/T) || softmax(z_t/T)) + (1-α) * CE(y, z_s)
/// ```
///
/// Where:
/// - `z_s` = student logits
/// - `z_t` = teacher logits
/// - `T` = temperature (higher = softer targets)
/// - `α` = weight for distillation loss vs hard label loss
/// - `y` = ground truth labels
#[derive(Debug, Clone)]
pub struct DistillationLoss {
    /// Temperature for softening distributions (typical: 2-20)
    pub temperature: f32,
    /// Weight for soft loss vs hard loss (typical: 0.5-0.9)
    pub alpha: f32,
}

impl Default for DistillationLoss {
    fn default() -> Self {
        Self::new(4.0, 0.7)
    }
}

impl DistillationLoss {
    /// Create new distillation loss
    ///
    /// # Arguments
    ///
    /// * `temperature` - Temperature for softening (2-20 typical)
    /// * `alpha` - Weight for soft loss (0.5-0.9 typical)
    #[must_use]
    pub fn new(temperature: f32, alpha: f32) -> Self {
        Self { temperature, alpha }
    }

    /// Compute distillation loss for single sample
    ///
    /// # Arguments
    ///
    /// * `student_logits` - Student model output logits
    /// * `teacher_logits` - Teacher model output logits
    /// * `target` - Ground truth label index
    ///
    /// # Returns
    ///
    /// Combined distillation loss
    pub fn forward_single(
        &self,
        student_logits: &Array1<f32>,
        teacher_logits: &Array1<f32>,
        target: usize,
    ) -> f32 {
        let t = self.temperature;

        // Temperature-scaled logits
        let student_scaled: Array1<f32> = student_logits.mapv(|x| x / t);
        let teacher_scaled: Array1<f32> = teacher_logits.mapv(|x| x / t);

        // Soft targets from teacher
        let teacher_soft = softmax(&teacher_scaled);
        let student_log_soft = log_softmax(&student_scaled);

        // KL divergence (scaled by T²)
        let kl_loss = kl_divergence(&student_log_soft, &teacher_soft) * t * t;

        // Hard label cross-entropy
        let ce_loss = cross_entropy_loss(student_logits, target);

        // Combined loss
        self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss
    }

    /// Compute distillation loss for batch
    ///
    /// # Arguments
    ///
    /// * `student_logits` - [batch_size, vocab_size]
    /// * `teacher_logits` - [batch_size, vocab_size]
    /// * `targets` - Ground truth labels
    ///
    /// # Returns
    ///
    /// Mean batch loss
    pub fn forward(
        &self,
        student_logits: &Array2<f32>,
        teacher_logits: &Array2<f32>,
        targets: &[usize],
    ) -> f32 {
        let batch_size = student_logits.nrows();
        assert_eq!(batch_size, teacher_logits.nrows());
        assert_eq!(batch_size, targets.len());

        let mut total_loss = 0.0;
        for (i, &target) in targets.iter().enumerate() {
            let s_row = student_logits.row(i).to_owned();
            let t_row = teacher_logits.row(i).to_owned();
            total_loss += self.forward_single(&s_row, &t_row, target);
        }

        total_loss / batch_size as f32
    }

    /// Compute soft loss only (no hard labels)
    pub fn soft_loss(&self, student_logits: &Array1<f32>, teacher_logits: &Array1<f32>) -> f32 {
        let t = self.temperature;

        let student_scaled: Array1<f32> = student_logits.mapv(|x| x / t);
        let teacher_scaled: Array1<f32> = teacher_logits.mapv(|x| x / t);

        let teacher_soft = softmax(&teacher_scaled);
        let student_log_soft = log_softmax(&student_scaled);

        kl_divergence(&student_log_soft, &teacher_soft) * t * t
    }
}
