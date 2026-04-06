//! Knowledge Distillation Trainer implementation.

use super::config::TrainerConfig;
use super::state::TrainingState;
use crate::hf_pipeline::fine_tune::FineTuneMethod;
use crate::hf_pipeline::loader::TeacherModel;

/// Knowledge Distillation Trainer
///
/// Orchestrates the training loop for distilling knowledge from
/// a teacher model to a student model.
pub struct DistillationTrainer<T: TeacherModel> {
    /// Configuration
    pub config: TrainerConfig,
    /// Teacher model
    teacher: T,
    /// Training state
    state: TrainingState,
}

impl<T: TeacherModel> DistillationTrainer<T> {
    /// Create new trainer with teacher model
    pub fn new(config: TrainerConfig, teacher: T) -> Self {
        Self { config, teacher, state: TrainingState::new() }
    }

    /// Get current training state
    #[must_use]
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get teacher model reference
    #[must_use]
    pub fn teacher(&self) -> &T {
        &self.teacher
    }

    /// Compute total loss for a batch
    ///
    /// Combines distillation loss with optional progressive and attention transfer.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn compute_loss(
        &self,
        student_logits: &ndarray::Array2<f32>,
        teacher_logits: &ndarray::Array2<f32>,
        targets: &[usize],
        student_hidden: Option<&[ndarray::Array2<f32>]>,
        teacher_hidden: Option<&[ndarray::Array2<f32>]>,
        student_attention: Option<&[ndarray::Array2<f32>]>,
        teacher_attention: Option<&[ndarray::Array2<f32>]>,
    ) -> f32 {
        // Contract: cross-entropy-kernel-v1.yaml precondition (pv codegen)
        contract_pre_cross_entropy!();

        // Base distillation loss
        let mut total_loss =
            self.config.distillation_loss.forward(student_logits, teacher_logits, targets);

        // Progressive distillation (hidden state matching)
        if let (Some(prog), Some(sh), Some(th)) =
            (&self.config.progressive, student_hidden, teacher_hidden)
        {
            total_loss += prog.hidden_state_loss(sh, th);
        }

        // Attention transfer
        if let (Some(at), Some(sa), Some(ta)) =
            (&self.config.attention_transfer, student_attention, teacher_attention)
        {
            total_loss += at.loss(sa, ta);
        }

        contract_post_cross_entropy!(total_loss);
        total_loss
    }

    /// Check if using LoRA/QLoRA for student fine-tuning
    #[must_use]
    pub fn is_parameter_efficient(&self) -> bool {
        matches!(
            self.config.fine_tune.method,
            FineTuneMethod::LoRA(_)
                | FineTuneMethod::QLoRA { .. }
                | FineTuneMethod::PrefixTuning { .. }
        )
    }

    /// Estimate total memory requirements
    #[must_use]
    pub fn estimate_total_memory(&self) -> u64 {
        let teacher_mem = self.teacher.estimate_memory(
            self.config.fine_tune.batch_size,
            self.config.fine_tune.max_seq_length,
        );
        let student_mem = self.config.fine_tune.estimate_memory(self.teacher.param_count() / 4); // Assume 4x smaller student

        teacher_mem.total() + student_mem.total()
    }

    /// Simulate one training step (for testing)
    pub fn simulate_step(&mut self, loss: f32) {
        self.state.record_loss(loss);
        self.state.step();
    }

    /// Simulate epoch boundary
    pub fn simulate_epoch(&mut self) {
        self.state.new_epoch();
    }
}
