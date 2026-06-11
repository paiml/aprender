//! Teacher logits provider for the distillation training loop.
//!
//! # SPEC-DISTILL-001 Phase 1 (PMAT-691)
//!
//! Replaces the synthetic-logits stub in `pipeline.rs::train()`. The
//! [`TeacherLogitsProvider`] trait abstracts the "given a batch of
//! `input_ids`, return the teacher's logits" operation so that the
//! pipeline can swap backends (real teacher inference, frozen fixture
//! for tests, mock for unit tests) without touching the training loop.
//!
//! ## Design — online instead of cached (SPEC-DISTILL-001 v1.1.0)
//!
//! The original v1.0.0 plan cached top-K logits to disk. Storage math:
//! 1.24B tokens × 64 entries × 6 bytes ≈ 476 GB, which exceeds the
//! lambda-vector NVMe budget. Online teacher inference (the DistilBERT /
//! Distil-Qwen actual practice) trades ~2× student-step time for zero
//! cache footprint. A future Phase 1.5 may add an in-memory ring buffer
//! that hides teacher latency behind student compute.
//!
//! ## Falsifier
//!
//! `F-DISTILL-TEACHER-001`: the provider's output matches the wrapped
//! backend's standalone logits computation on the same `input_ids` within
//! `1e-3` absolute error. Verified in `tests/`.

use entrenar_common::Result;

/// A teacher whose logits the student tries to match during distillation.
///
/// Implementations decide how to load + run the teacher; the pipeline only
/// needs `logits_for_batch`. Returned shape is `[batch, vocab_size]` for
/// last-position logits (suitable for next-token-prediction KD).
pub trait TeacherLogitsProvider {
    /// Vocabulary size of the teacher's output distribution. The pipeline
    /// uses this to size the KD-loss soft-targets without first calling
    /// `logits_for_batch`.
    fn vocab_size(&self) -> usize;

    /// Run the teacher on `input_ids` (one `Vec<u32>` per batch element)
    /// and return last-position logits.
    ///
    /// The returned `Vec<Vec<f32>>` is shape `[batch, vocab]`. Empty
    /// `input_ids` is a programming error; implementations may panic
    /// or return an error.
    ///
    /// # Errors
    ///
    /// Returns an error if the teacher backend fails (e.g., CUDA OOM,
    /// missing weights). The pipeline aborts on error rather than
    /// silently fall back to non-distillation training.
    fn logits_for_batch(&mut self, input_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>>;

    /// Run the teacher and return logits at EVERY position of each input
    /// window: shape `[batch][position][vocab]`. This is the per-position
    /// next-token signal used by full-sequence KD — every position predicts
    /// its successor, giving up to `seq_len`× more distillation signal per
    /// forward pass than the last-position-only [`Self::logits_for_batch`].
    ///
    /// The default impl wraps `logits_for_batch` as a single trailing
    /// position, so existing providers (including the CUDA backend) keep
    /// working unchanged — they expose one position until they override this
    /// with a true all-positions forward. Per-position-capable providers
    /// (e.g. `FixtureTeacher`) override it.
    ///
    /// # Errors
    ///
    /// Propagates backend errors, same as [`Self::logits_for_batch`].
    fn logits_per_position(&mut self, input_ids: &[Vec<u32>]) -> Result<Vec<Vec<Vec<f32>>>> {
        Ok(self
            .logits_for_batch(input_ids)?
            .into_iter()
            .map(|row| vec![row])
            .collect())
    }
}

/// A frozen-fixture teacher used in unit tests. Returns logits whose
/// `argmax` is the last input token (so the student can be tested for
/// "next-token-matches-last-token" sanity), with all other vocab entries
/// at zero.
///
/// Not for production use — see `realizar_teacher::RealizarTeacher`
/// (Phase 1b, PMAT-692) for the real backend wired through `aprender-serve`.
pub struct FixtureTeacher {
    vocab_size: usize,
}

impl FixtureTeacher {
    /// Create a fixture teacher with the given vocabulary size.
    #[must_use]
    pub const fn new(vocab_size: usize) -> Self {
        Self { vocab_size }
    }
}

impl TeacherLogitsProvider for FixtureTeacher {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    fn logits_for_batch(&mut self, input_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(input_ids.len());
        for ids in input_ids {
            let mut logits = vec![0.0_f32; self.vocab_size];
            // Use the last token as the "predicted next token" — set its
            // logit to a high value so argmax(softmax(logits)) recovers it.
            // Out-of-vocab IDs collapse to position 0 to keep the fixture
            // well-defined under all inputs.
            #[allow(clippy::cast_possible_truncation)]
            let predicted_token = ids.last().copied().unwrap_or(0) as usize;
            let idx = if predicted_token < self.vocab_size {
                predicted_token
            } else {
                0
            };
            logits[idx] = 10.0;
            out.push(logits);
        }
        Ok(out)
    }

    fn logits_per_position(&mut self, input_ids: &[Vec<u32>]) -> Result<Vec<Vec<Vec<f32>>>> {
        // Per-position fixture: position p predicts the genuine next token
        // within the window (ids[p+1]); the final position (no successor in
        // the window) falls back to its own token. Each position's argmax is
        // therefore its target, giving the student a distinct, learnable
        // per-position signal.
        let mut out = Vec::with_capacity(input_ids.len());
        for ids in input_ids {
            let mut rows = Vec::with_capacity(ids.len());
            for p in 0..ids.len() {
                let mut logits = vec![0.0_f32; self.vocab_size];
                #[allow(clippy::cast_possible_truncation)]
                let predicted = if p + 1 < ids.len() {
                    ids[p + 1] as usize
                } else {
                    ids[p] as usize
                };
                let idx = if predicted < self.vocab_size {
                    predicted
                } else {
                    0
                };
                logits[idx] = 10.0;
                rows.push(logits);
            }
            out.push(rows);
        }
        Ok(out)
    }
}

// SPEC-DISTILL-001 Phase 1b (PMAT-693): real teacher backend delegating
// to entrenar's CudaTransformerTrainer in inference-only mode. Gated on
// the `cuda` feature because the underlying trainer's `for_inference`
// constructor requires CUDA. Without the feature, only `FixtureTeacher`
// is available (which is sufficient for unit tests but cannot drive a
// real distillation training run).
#[cfg(feature = "cuda")]
pub use cuda_backend::CudaTrainerTeacher;

#[cfg(feature = "cuda")]
mod cuda_backend {
    use super::{Result, TeacherLogitsProvider};
    use entrenar::train::transformer_trainer::CudaTransformerTrainer;
    use entrenar::transformer::TransformerConfig;
    use std::path::Path;

    /// Real teacher backend wrapping a CUDA inference-only trainer.
    ///
    /// Constructed via `CudaTrainerTeacher::for_inference(checkpoint_dir,
    /// model_config)` which loads SafeTensors (preferring APR format if
    /// present) and stages weights on the GPU.
    ///
    /// `logits_for_batch` invokes the trainer's `forward_logits(&tokens)`
    /// per batch element and returns the last-position logits for each.
    /// Output shape is `[batch, vocab_size]`.
    ///
    /// # Falsifier
    ///
    /// `F-DISTILL-TEACHER-002` — the per-batch-element logits must equal
    /// what a standalone `CudaTransformerTrainer::for_inference(...)
    /// .forward_logits(...)` call produces on the same input within
    /// `1e-6` absolute (no extra processing layer; this provider is a
    /// thin delegation).
    pub struct CudaTrainerTeacher {
        trainer: CudaTransformerTrainer,
        vocab_size: usize,
    }

    impl CudaTrainerTeacher {
        /// Construct from a checkpoint directory + model config.
        ///
        /// # Errors
        ///
        /// Returns an error if the checkpoint cannot be loaded or CUDA
        /// initialization fails.
        pub fn for_inference(
            checkpoint_dir: impl AsRef<Path>,
            model_config: TransformerConfig,
        ) -> Result<Self> {
            let vocab_size = model_config.vocab_size;
            let trainer = CudaTransformerTrainer::for_inference(checkpoint_dir, model_config)
                .map_err(|e| entrenar_common::EntrenarError::Internal {
                    message: format!("CudaTrainerTeacher::for_inference: {e}"),
                })?;
            Ok(Self {
                trainer,
                vocab_size,
            })
        }
    }

    impl TeacherLogitsProvider for CudaTrainerTeacher {
        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn logits_for_batch(&mut self, input_ids: &[Vec<u32>]) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(input_ids.len());
            for ids in input_ids {
                let logits = self.trainer.forward_logits(ids).ok_or_else(|| {
                    entrenar_common::EntrenarError::Internal {
                        message: "CudaTransformerTrainer.forward_logits returned \
                                  None (likely missing weights or CUDA init failure)"
                            .to_string(),
                    }
                })?;
                // Defensive size check — the trainer must return vocab_size
                // logits; if not, something is mis-configured (e.g., the
                // model_config vocab doesn't match the checkpoint).
                if logits.len() != self.vocab_size {
                    return Err(entrenar_common::EntrenarError::Internal {
                        message: format!(
                            "CudaTrainerTeacher: forward_logits returned {} \
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_teacher_reports_correct_vocab_size() {
        let t = FixtureTeacher::new(151_936);
        assert_eq!(t.vocab_size(), 151_936);
    }

    #[test]
    fn fixture_teacher_returns_one_logits_vec_per_batch_element() {
        let mut t = FixtureTeacher::new(32);
        let batch = vec![vec![1, 2, 3], vec![4, 5], vec![6]];
        let logits = t.logits_for_batch(&batch).unwrap();
        assert_eq!(logits.len(), 3, "one logits vec per batch element");
        for v in &logits {
            assert_eq!(v.len(), 32, "each logits vec is vocab_size long");
        }
    }

    #[test]
    fn fixture_teacher_argmax_recovers_last_token() {
        // F-DISTILL-FIXTURE-001: the fixture's argmax is the last input
        // token. This lets pipeline tests verify the KD signal is being
        // applied correctly: if the student aligns with the teacher,
        // student.argmax should converge to the fixture's argmax.
        let mut t = FixtureTeacher::new(16);
        let batch = vec![vec![1, 2, 7], vec![3], vec![15, 0, 5]];
        let logits = t.logits_for_batch(&batch).unwrap();

        for (i, expected_argmax) in [7usize, 3, 5].iter().enumerate() {
            let argmax = logits[i]
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(
                argmax, *expected_argmax,
                "batch element {i}: argmax should equal last input token"
            );
        }
    }

    #[test]
    fn fixture_teacher_out_of_vocab_token_collapses_to_zero() {
        // Robustness: if input contains a token >= vocab_size (which
        // shouldn't happen in correct usage but is a useful test for the
        // provider's bounds-handling), the fixture must not panic.
        let mut t = FixtureTeacher::new(8);
        let batch = vec![vec![999_999]]; // out of 8-vocab
        let logits = t.logits_for_batch(&batch).unwrap();
        assert_eq!(logits[0].len(), 8);
        // The first slot gets the high value (collapse rule), not a panic.
        let argmax = logits[0]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(argmax, 0, "out-of-vocab token argmax collapses to 0");
    }

    #[test]
    fn fixture_teacher_empty_batch_returns_empty_vec() {
        let mut t = FixtureTeacher::new(32);
        let logits = t.logits_for_batch(&[]).unwrap();
        assert!(logits.is_empty());
    }

    #[test]
    fn fixture_teacher_logits_are_deterministic_across_calls() {
        // The fixture must be stateless / pure — same input → same output.
        // Required so that pipeline unit tests are reproducible.
        let mut t1 = FixtureTeacher::new(64);
        let mut t2 = FixtureTeacher::new(64);
        let batch = vec![vec![10, 20, 30]];
        let l1 = t1.logits_for_batch(&batch).unwrap();
        let l2 = t2.logits_for_batch(&batch).unwrap();
        assert_eq!(l1, l2);

        // And the same instance returns the same values on repeated calls.
        let l3 = t1.logits_for_batch(&batch).unwrap();
        assert_eq!(l1, l3);
    }
}
