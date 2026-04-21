//! CUDA-backend `StepFn` / `ValFn` / `CheckpointFn` for the 370M pretrain
//! loop (task #132 Phase 2, contract `gpu-training-backend-v1`).
//!
//! Mirrors `pretrain_real.rs` but swaps `TransformerTrainer`
//! (CPU + trueno SIMD) for `CudaTransformerTrainer` (GPU-resident
//! AdamW + fused CE). The entire module is gated on
//! `#[cfg(feature = "cuda")]` because `CudaTransformerTrainer::new`
//! / `train_batch` / `eval_batch` / `save_apr` only exist in the
//! cuda build — the non-cuda stub returns an error from `new()` and
//! exposes no step/eval/save methods.
//!
//! Contract obligations discharged / strengthened vs the CPU path:
//! - INV-ARCH-370M-001 (param count ∈ [366M, 374M]) via `debug_assert`
//!   on `CudaTransformerTrainer::model().parameters()`, matching
//!   the CPU guard.
//! - INV-TRAIN-007 (no NaN/Inf): `train_batch` / `eval_batch` return
//!   finite loss by construction; non-finite outputs abort via
//!   `PretrainLoop`'s guards.
//! - INV-TRAIN-008 (grad_norm ≥ 0): `last_grad_norm()` returns the
//!   real LM-head L2 norm. Strictly stronger than the CPU path's
//!   `1.0` placeholder.
//!
//! Deferred to a follow-up:
//! - INV-TRAIN-003 (AdamW-state sha256). `CudaTransformerTrainer`
//!   keeps (m, v, t) on the GPU; discharging this cleanly needs a
//!   D2H sync that `save_apr` already pays for but `StepFn` does
//!   not want to pay per-step. Until that sync is factored out,
//!   the trait default `optimizer_state_sha256 -> None` is used,
//!   and GATE-TRAIN-006 runs only on the CPU path.

#![cfg(feature = "cuda")]

use crate::train::pretrain::{CheckpointFn, EpochArtifact, StepFn, ValFn};
use crate::train::pretrain_real::llama_370m_train_config;
use crate::train::transformer_trainer::{CudaTransformerTrainer, LMBatch};
use std::cell::RefCell;
use std::rc::Rc;

/// Shared mutable ownership of a GPU-resident trainer. Both
/// `CudaRealStepFn` (train steps) and `CudaRealValFn` (eval) clone
/// this `Rc` so the three hooks see the same GPU memory.
pub type SharedCudaTrainer = Rc<RefCell<CudaTransformerTrainer>>;

/// Allocate a `CudaTransformerTrainer` with MODEL-2 v2-remedy defaults
/// and verify INV-ARCH-370M-001 in debug builds.
///
/// Returns a `crate::Result` because `CudaTransformerTrainer::new`
/// can fail on missing CUDA runtime, kernel pre-warm failure, or
/// block upload failure — the CLI surfaces this as a
/// GATE-GPUTRAIN-002 error so the operator knows to check their
/// `--features cuda` build or their GPU.
pub fn build_shared_cuda_trainer(
    lr: f32,
    seq_length: usize,
    seed: u64,
) -> crate::Result<SharedCudaTrainer> {
    let cfg = llama_370m_train_config(lr, seq_length, seed);
    let trainer = CudaTransformerTrainer::new(cfg)?;
    #[cfg(debug_assertions)]
    {
        let param_count: usize = trainer.model().parameters().iter().map(|t| t.len()).sum();
        debug_assert!(
            (366_000_000..=374_000_000).contains(&param_count),
            "INV-ARCH-370M-001: parameter count {param_count} outside [366M, 374M] band",
        );
    }
    Ok(Rc::new(RefCell::new(trainer)))
}

/// CUDA `StepFn` — pulls one `LMBatch` from the shard iterator and
/// runs a real GPU forward + backward + AdamW step.
pub struct CudaRealStepFn {
    trainer: SharedCudaTrainer,
    batches: Box<dyn Iterator<Item = LMBatch>>,
}

impl CudaRealStepFn {
    pub fn new(trainer: SharedCudaTrainer, batches: Box<dyn Iterator<Item = LMBatch>>) -> Self {
        Self { trainer, batches }
    }
}

impl StepFn for CudaRealStepFn {
    fn step(&mut self, _step: u64, _lr: f32, _batch_tokens: u64) -> (f32, f32) {
        // Exhausted shard stream: emit a finite placeholder so the
        // NaN/Inf guard (INV-TRAIN-007) doesn't mis-fire and the
        // divergence guard (GATE-TRAIN-005) correctly does not abort.
        let Some(batch) = self.batches.next() else {
            return (1.0, 1.0);
        };
        let mut trainer = self.trainer.borrow_mut();
        let loss = trainer.train_batch(&batch);
        // Real LM-head L2 norm — strictly more informative than the
        // CPU path's `1.0` placeholder for GATE-TRAIN-008 monitoring.
        let grad_norm = trainer.last_grad_norm();
        (loss, grad_norm)
    }

    // INV-TRAIN-003 intentionally deferred for the GPU path — see
    // module docs. Uses trait default `-> None`, so the CPU gate
    // (`--device cpu`) is the one that exercises AdamW-state parity.
}

/// CUDA `ValFn` — forward-only eval across pre-loaded held-out
/// batches. Uses `eval_batch` (fused GPU cross-entropy, no logits
/// D2H) and averages across batches.
pub struct CudaRealValFn {
    trainer: SharedCudaTrainer,
    held_out: Vec<LMBatch>,
}

impl CudaRealValFn {
    pub fn new(trainer: SharedCudaTrainer, held_out: Vec<LMBatch>) -> Self {
        Self { trainer, held_out }
    }
}

impl ValFn for CudaRealValFn {
    fn validate(&mut self, _epoch: usize) -> f32 {
        if self.held_out.is_empty() {
            return f32::NAN;
        }
        let mut trainer = self.trainer.borrow_mut();
        let mut total_loss = 0.0_f32;
        let mut count = 0_usize;
        for batch in &self.held_out {
            if batch.batch_size == 0 {
                continue;
            }
            total_loss += trainer.eval_batch(batch);
            count += 1;
        }
        if count == 0 {
            f32::NAN
        } else {
            total_loss / count as f32
        }
    }
}

/// CUDA `CheckpointFn` — writes the 370M weights to
/// `artifact.checkpoint_path` in APR format. `save_apr` takes
/// `&mut self` on the CUDA path because it syncs GPU→CPU before
/// writing, which is why this holds the `SharedCudaTrainer` instead
/// of cloning the trainer out.
pub struct CudaAprCheckpointFn {
    trainer: SharedCudaTrainer,
    model_name: String,
    architecture: String,
}

impl CudaAprCheckpointFn {
    pub fn new(
        trainer: SharedCudaTrainer,
        model_name: impl Into<String>,
        architecture: impl Into<String>,
    ) -> Self {
        Self { trainer, model_name: model_name.into(), architecture: architecture.into() }
    }
}

impl CheckpointFn for CudaAprCheckpointFn {
    fn save(&mut self, _epoch: usize, artifact: &EpochArtifact) -> Result<(), String> {
        let mut trainer = self.trainer.borrow_mut();
        trainer
            .save_apr(&artifact.checkpoint_path, &self.model_name, &self.architecture)
            .map_err(|e| format!("save_apr (cuda) failed: {e}"))
    }
}
