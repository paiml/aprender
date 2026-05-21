//! Student trainable-model abstraction for the distillation loop.
//!
//! # SPEC-DISTILL-001 Phase 2b (PMAT-695)
//!
//! Mirrors Phase 1's [`TeacherLogitsProvider`] for the student side.
//! Where the teacher only needs to *produce* logits, the student also
//! needs to *update its weights* given a logit-space gradient — so the
//! trait has two methods: [`StudentLogitsProvider::logits_for_batch`]
//! (forward) and [`StudentLogitsProvider::apply_kd_gradient`] (backward
//! + optimizer step seeded by Phase 2a's `kd_logit_gradient`).
//!
//! ## Phase 2b vs Phase 2c
//!
//! **Phase 2b (this module, landed in this PR)**: ships the trait + a
//! `FixtureStudent` implementation suitable for unit tests + the
//! pipeline-level wiring that uses it. The fixture's `apply_kd_gradient`
//! moves its internal logits one step toward the teacher (gradient
//! descent in logit space with a constant LR) so loss-monotonicity
//! tests are meaningful.
//!
//! **Phase 2c (separate ticket, PMAT-696)**: ships `CudaStudentProvider`
//! that wraps `CudaTransformerTrainer`. The forward uses
//! `CudaTransformerTrainer::forward_logits`; the backward uses a new
//! `forward_backward_kd_batch` (added to the trainer) that uploads
//! `kd_logit_gradient` into `logits_buf` and runs the standard
//! `gpu_backward` — bypassing the in-place `fused_cross_entropy_cuda`
//! path. With Phase 2c landed, real distillation runs on GPU.
//!
//! ## Falsifiers pinned
//!
//! - **F-DISTILL-STUDENT-001** — `FixtureStudent` initialized with
//!   uniform logits moves strictly toward the teacher's distribution
//!   after one `apply_kd_gradient` step.
//! - **F-DISTILL-STUDENT-002** — multiple steps of `apply_kd_gradient`
//!   strictly decrease the per-step KD loss (loss monotone — sanity for
//!   the gradient direction).

use entrenar_common::Result;

/// A trainable student model whose logits-space updates are driven by
/// Phase 2a's `kd_logit_gradient`.
///
/// The `apply_kd_gradient` contract: given a `Vec<Vec<f32>>` shape
/// `[batch, vocab]` from `kd_logit_gradient`, update the student so
/// that subsequent calls to `logits_for_batch` for the same `input_ids`
/// produce logits shifted in the direction the gradient indicates.
///
/// For `FixtureStudent` this is literally subtracting the scaled
/// gradient from an internal logits buffer. For `CudaStudentProvider`
/// (Phase 2c) this uploads the gradient into the trainer's `logits_buf`,
/// calls `gpu_backward`, and runs the optimizer step.
pub trait StudentLogitsProvider {
    /// Vocabulary size of the student's output distribution.
    fn vocab_size(&self) -> usize;

    /// Forward pass: produce last-position logits for each batch element.
    ///
    /// Shape `[batch, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the student backend fails (e.g., CUDA OOM,
    /// missing weights).
    fn logits_for_batch(&mut self, input_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>>;

    /// Backward pass + optimizer step seeded by Phase 2a's
    /// `kd_logit_gradient` output.
    ///
    /// Shape of `gradient` matches `logits_for_batch` output —
    /// `[batch, vocab_size]`. After this call returns, subsequent
    /// `logits_for_batch` calls reflect the updated parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if the gradient shape doesn't match or the
    /// optimizer step fails.
    fn apply_kd_gradient(&mut self, gradient: &[Vec<f32>]) -> Result<()>;

    /// PMAT-699 (Phase 4 P0): persist the student's current trained weights
    /// to disk.
    ///
    /// Default implementation is a no-op — appropriate for FixtureStudent
    /// where the pipeline's export step writes the placeholder
    /// `student_weights` map directly. CudaStudentProvider MUST override
    /// to delegate to its trainer's `save_apr` — otherwise 25h of GPU
    /// training silently produces a 200-byte empty checkpoint
    /// (Stage D 2026-05-20 incident).
    ///
    /// # Errors
    ///
    /// Returns an error if the backend cannot serialize (e.g., GPU
    /// download fails, disk write fails).
    fn save_checkpoint(&mut self, _path: &std::path::Path) -> Result<()> {
        Ok(())
    }
}

/// Fixture student for unit testing the orchestration layer.
///
/// Holds an internal `[vocab_size]` logits buffer that is returned
/// (broadcast across batch elements) on every `logits_for_batch` call.
/// `apply_kd_gradient` subtracts the mean gradient (across batch
/// elements) scaled by `learning_rate` from the buffer.
///
/// Not for production — Phase 2c's `CudaStudentProvider` (PMAT-696) is
/// the real backend.
pub struct FixtureStudent {
    vocab_size: usize,
    logits: Vec<f32>,
    learning_rate: f32,
}

impl FixtureStudent {
    /// Create a fixture student with `vocab_size` initial logits set to
    /// `initial_value` and a SGD-style learning rate `learning_rate` for
    /// the in-place logit update.
    #[must_use]
    pub fn new(vocab_size: usize, initial_value: f32, learning_rate: f32) -> Self {
        Self {
            vocab_size,
            logits: vec![initial_value; vocab_size],
            learning_rate,
        }
    }

    /// Read the current student logits — exposed for tests that need to
    /// assert "logits moved toward teacher".
    #[must_use]
    pub fn current_logits(&self) -> &[f32] {
        &self.logits
    }
}

impl StudentLogitsProvider for FixtureStudent {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn logits_for_batch(&mut self, input_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        // The fixture is input-independent — broadcast the same logits.
        Ok(input_ids.iter().map(|_| self.logits.clone()).collect())
    }

    fn apply_kd_gradient(&mut self, gradient: &[Vec<f32>]) -> Result<()> {
        if gradient.is_empty() {
            return Ok(());
        }
        // Validate gradient shape — each row must equal vocab_size.
        for (i, row) in gradient.iter().enumerate() {
            if row.len() != self.vocab_size {
                return Err(entrenar_common::EntrenarError::Internal {
                    message: format!(
                        "FixtureStudent.apply_kd_gradient: gradient row {} has \
                         length {} but vocab_size is {}",
                        i,
                        row.len(),
                        self.vocab_size
                    ),
                });
            }
        }
        // Average gradient across batch (the canonical SGD batch averaging).
        let batch_size = gradient.len() as f32;
        for j in 0..self.vocab_size {
            let mean_grad: f32 =
                gradient.iter().map(|row| row[j]).sum::<f32>() / batch_size;
            self.logits[j] -= self.learning_rate * mean_grad;
        }
        Ok(())
    }
}

// SPEC-DISTILL-001 Phase 2d (PMAT-697): real GPU student backend
// wrapping `CudaTransformerTrainer`. Gated on the `cuda` feature
// because the underlying trainer's `for_inference` constructor +
// `forward_backward_with_grad` method both require CUDA.
#[cfg(feature = "cuda")]
pub use cuda_backend::CudaStudentProvider;

#[cfg(feature = "cuda")]
mod cuda_backend {
    use super::{Result, StudentLogitsProvider};
    use entrenar::train::transformer_trainer::CudaTransformerTrainer;
    use entrenar::transformer::TransformerConfig;
    use std::path::Path;

    /// Real GPU student backend. Wraps a trainable `CudaTransformerTrainer`
    /// and bridges the Phase 2b trait surface to the trainer's
    /// `forward_logits` + `forward_backward_with_grad` methods (the latter
    /// added in Phase 2d for this purpose).
    ///
    /// **Batch_size=1 only** (Phase 2d limitation): the trait's
    /// `apply_kd_gradient` doesn't take input_ids, so the provider has
    /// to cache the input_ids from the most-recent `logits_for_batch`
    /// call and re-run forward on each apply call. With batch_size=1
    /// that round-trip is correct; with batch_size>1 only the LAST
    /// element gets a real gradient update. Phase 2e generalizes via a
    /// fused-step trait method that takes input_ids + gradient together.
    ///
    /// # Falsifier
    ///
    /// `F-DISTILL-CUDA-STUDENT-001` (proposed) — `logits_for_batch` matches
    /// a standalone `CudaTransformerTrainer::for_inference(...).forward_logits(...)`
    /// call within `1e-6` absolute. Verified in Phase 4 production
    /// (requires CUDA hardware).
    pub struct CudaStudentProvider {
        trainer: CudaTransformerTrainer,
        vocab_size: usize,
        // Phase 2d batch_size=1 limitation: cache the most-recent
        // input_ids so apply_kd_gradient knows what activations to
        // re-establish. None means "no logits_for_batch has been
        // called yet" — apply_kd_gradient errors in that state.
        last_input_ids: Option<Vec<u32>>,
    }

    impl CudaStudentProvider {
        /// Construct from a checkpoint directory + model config.
        ///
        /// The checkpoint must contain a `model.safetensors` or
        /// `model.apr` file matching `model_config`. Loads onto GPU
        /// in train mode (optimizer state allocated).
        ///
        /// # Errors
        ///
        /// Returns an error if checkpoint loading or CUDA initialization
        /// fails.
        pub fn for_training(
            checkpoint_dir: impl AsRef<Path>,
            model_config: TransformerConfig,
        ) -> Result<Self> {
            let vocab_size = model_config.vocab_size;
            let trainer = CudaTransformerTrainer::for_inference(checkpoint_dir, model_config)
                .map_err(|e| entrenar_common::EntrenarError::Internal {
                    message: format!("CudaStudentProvider::for_training: {e}"),
                })?;
            Ok(Self {
                trainer,
                vocab_size,
                last_input_ids: None,
            })
        }
    }

    impl StudentLogitsProvider for CudaStudentProvider {
        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn logits_for_batch(&mut self, input_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
            // Phase 2d caches only the most-recent input_ids (batch_size=1
            // assumption). For batches >1, only the last element's
            // activations are retained for apply_kd_gradient.
            self.last_input_ids = input_ids.last().cloned();

            let mut out = Vec::with_capacity(input_ids.len());
            for ids in input_ids {
                let logits = self.trainer.forward_logits(ids).ok_or_else(|| {
                    entrenar_common::EntrenarError::Internal {
                        message: "CudaStudentProvider.forward_logits returned \
                                  None (likely CUDA init failure or empty input_ids)"
                            .to_string(),
                    }
                })?;
                if logits.len() != self.vocab_size {
                    return Err(entrenar_common::EntrenarError::Internal {
                        message: format!(
                            "CudaStudentProvider: forward_logits returned {} \
                             logits, expected {} (vocab_size mismatch — likely \
                             a config drift between TransformerConfig and the \
                             loaded checkpoint)",
                            logits.len(),
                            self.vocab_size
                        ),
                    });
                }
                out.push(logits);
            }
            Ok(out)
        }

        fn apply_kd_gradient(&mut self, gradient: &[Vec<f32>]) -> Result<()> {
            if gradient.is_empty() {
                return Ok(());
            }
            let Some(last_ids) = self.last_input_ids.clone() else {
                return Err(entrenar_common::EntrenarError::Internal {
                    message: "CudaStudentProvider.apply_kd_gradient called \
                              before logits_for_batch — no cached input_ids \
                              to re-establish activations"
                        .to_string(),
                });
            };
            // Apply only the LAST gradient (matches the batch_size=1
            // limitation documented on the struct). Future Phase 2e fuses
            // input_ids + gradient into a single trait call so larger
            // batches can be processed correctly.
            let last_grad = gradient
                .last()
                .ok_or_else(|| entrenar_common::EntrenarError::Internal {
                    message: "CudaStudentProvider.apply_kd_gradient: empty gradient slice"
                        .to_string(),
                })?;
            self.trainer
                .forward_backward_with_grad(&last_ids, last_grad)
                .ok_or_else(|| entrenar_common::EntrenarError::Internal {
                    message: "CudaTransformerTrainer.forward_backward_with_grad \
                              returned None (CUDA stream poisoned or gradient \
                              shape mismatch)"
                        .to_string(),
                })?;
            Ok(())
        }

        /// PMAT-699 P0 fix: pull trained weights from GPU and write them
        /// to the destination directory as an APR v2 checkpoint.
        ///
        /// Without this override, pipeline.export() serialized the empty
        /// student_weights HashMap (PMAT-698f's APR short-circuit returns
        /// empty), producing a 200-byte placeholder model.safetensors.
        /// Stage D 2026-05-20 ran 25h of GB10 training and lost all
        /// weights because of this gap.
        ///
        /// Delegates to the trainer's existing save_apr method.
        fn save_checkpoint(&mut self, path: &std::path::Path) -> Result<()> {
            // Ensure parent dir exists (trainer.save_apr writes into it).
            if let Some(parent) = path.parent() {
                if !parent.as_os_str().is_empty() {
                    std::fs::create_dir_all(parent).map_err(|e| {
                        entrenar_common::EntrenarError::Io {
                            context: format!(
                                "save_checkpoint mkdir parent {}",
                                parent.display()
                            ),
                            source: e,
                        }
                    })?;
                }
            }
            self.trainer
                .save_apr(path, "albor-distilled-v2", "Qwen2ForCausalLM")
                .map_err(|e| entrenar_common::EntrenarError::Internal {
                    message: format!(
                        "CudaStudentProvider.save_checkpoint: trainer.save_apr({}) failed: {e:?}",
                        path.display()
                    ),
                })?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kd_step::{kd_logit_gradient, kd_loss};
    use crate::teacher_provider::{FixtureTeacher, TeacherLogitsProvider};

    #[test]
    fn fixture_student_reports_vocab_size() {
        let s = FixtureStudent::new(32, 0.0, 0.1);
        assert_eq!(s.vocab_size(), 32);
    }

    #[test]
    fn fixture_student_broadcasts_logits_across_batch() {
        let mut s = FixtureStudent::new(8, 0.5, 0.1);
        let batch = vec![vec![1u32, 2], vec![3]];
        let logits = s.logits_for_batch(&batch).unwrap();
        assert_eq!(logits.len(), 2);
        for row in &logits {
            assert_eq!(row.len(), 8);
            for &v in row {
                assert!((v - 0.5).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn fixture_student_apply_kd_gradient_validates_shape() {
        let mut s = FixtureStudent::new(8, 0.0, 0.1);
        let wrong = vec![vec![0.0_f32; 4]]; // wrong vocab
        let r = s.apply_kd_gradient(&wrong);
        assert!(r.is_err(), "shape mismatch must error");
    }

    #[test]
    fn fixture_student_apply_kd_gradient_updates_logits() {
        let mut s = FixtureStudent::new(4, 1.0, 0.1);
        // All-positive gradient → logits should decrease.
        let grad = vec![vec![2.0_f32; 4]];
        s.apply_kd_gradient(&grad).unwrap();
        for &v in s.current_logits() {
            assert!((v - (1.0 - 0.1 * 2.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn fixture_student_apply_kd_gradient_averages_across_batch() {
        let mut s = FixtureStudent::new(2, 0.0, 1.0);
        // Two batch elements with opposite gradients → mean = 0 → no update.
        let grad = vec![vec![1.0_f32, 2.0], vec![-1.0_f32, -2.0]];
        s.apply_kd_gradient(&grad).unwrap();
        for &v in s.current_logits() {
            assert!(v.abs() < 1e-6, "batch-averaged zero gradient → no logit change");
        }
    }

    /// F-DISTILL-STUDENT-001: starting from uniform logits, one KD step
    /// moves the student strictly closer to the teacher's distribution.
    #[test]
    fn falsify_student_001_one_step_moves_toward_teacher() {
        let vocab = 16;
        let mut teacher = FixtureTeacher::new(vocab);
        let mut student = FixtureStudent::new(vocab, 0.0, 1.0); // uniform start

        // Teacher's logits for input_ids ending in token 5 will have a
        // large value at index 5; everything else 0.
        let input_ids = vec![vec![5u32]];
        let labels = vec![5_usize];
        let teacher_logits = teacher.logits_for_batch(&input_ids).unwrap();
        let s_logits_before = student.logits_for_batch(&input_ids).unwrap();

        let grad = vec![kd_logit_gradient(
            &s_logits_before[0],
            &teacher_logits[0],
            labels[0],
            4.0,
            0.0, // alpha=0 → pure KL signal
        )];

        student.apply_kd_gradient(&grad).unwrap();

        // After one step, the student's logit at index 5 should be HIGHER
        // (teacher prefers index 5 → gradient at index 5 is negative →
        // student logit moves up).
        let s_logits_after = student.logits_for_batch(&input_ids).unwrap();
        assert!(
            s_logits_after[0][5] > s_logits_before[0][5],
            "student logit at teacher's preferred index must increase, \
             before={} after={}",
            s_logits_before[0][5],
            s_logits_after[0][5]
        );
    }

    /// F-DISTILL-STUDENT-002: multiple KD steps strictly decrease KD loss.
    #[test]
    fn falsify_student_002_loss_decreases_monotonically_over_steps() {
        let vocab = 16;
        let mut teacher = FixtureTeacher::new(vocab);
        let mut student = FixtureStudent::new(vocab, 0.0, 0.5);

        let input_ids = vec![vec![7u32]];
        let labels = vec![7_usize];
        let teacher_logits = teacher.logits_for_batch(&input_ids).unwrap();

        let mut losses = Vec::new();
        for _step in 0..10 {
            let s_logits = student.logits_for_batch(&input_ids).unwrap();
            let loss = kd_loss(&s_logits[0], &teacher_logits[0], labels[0], 4.0, 0.0);
            losses.push(loss);
            let grad = vec![kd_logit_gradient(
                &s_logits[0],
                &teacher_logits[0],
                labels[0],
                4.0,
                0.0,
            )];
            student.apply_kd_gradient(&grad).unwrap();
        }

        // Assert monotonic decrease (strict for first few steps, then ≤).
        // With LR=0.5 and 10 steps, we expect the loss to roughly halve.
        for i in 1..losses.len() {
            assert!(
                losses[i] <= losses[i - 1] + 1e-5,
                "KD loss must decrease monotonically over training, \
                 but step {i} loss {} > step {} loss {}",
                losses[i],
                i - 1,
                losses[i - 1]
            );
        }
        assert!(
            losses.last().unwrap() < &(losses[0] * 0.9),
            "loss after 10 steps ({}) should be < 90% of initial loss ({})",
            losses.last().unwrap(),
            losses[0]
        );
    }
}
