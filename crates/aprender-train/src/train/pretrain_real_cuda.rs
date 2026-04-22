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

use crate::train::cycling_iter::next_batch_or_panic;
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
        // INV-TRAIN-011 / GATE-TRAIN-EXHAUST: same contract as the CPU
        // peer (`RealStepFn::step`). Exhaustion must surface either as
        // real cycled compute (caller wraps the iterator in
        // `CyclingBatchIter`) or as a `GATE-TRAIN-EXHAUST` panic —
        // never as the silent `(1.0, 1.0)` placeholder that was the
        // task #141 defect.
        let batch = next_batch_or_panic(&mut *self.batches);
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

#[cfg(test)]
mod tests {
    //! GATE-TRAIN-EXHAUST CUDA-peer discharge. These tests require a
    //! real CUDA GPU because `CudaTransformerTrainer::new` allocates
    //! on-device — they mirror `pretrain_real.rs` tests one-for-one
    //! and are `#[ignore]`d by default, matching the project pattern
    //! (`cuda_cublas_parity.rs`). Run with:
    //!     cargo test -p aprender-train --features cuda -- --ignored cuda_stepfn
    use super::*;
    use crate::train::transformer_trainer::TransformerTrainConfig;
    use crate::transformer::TransformerConfig;

    fn tiny_cuda_trainer() -> SharedCudaTrainer {
        // Bypass `build_shared_cuda_trainer` (370M debug_assert) and
        // construct the GPU trainer with a minimum synthetic config.
        let mut tiny = TransformerConfig::llama2_7b();
        tiny.hidden_size = 64;
        tiny.num_attention_heads = 4;
        tiny.num_kv_heads = 4;
        tiny.num_hidden_layers = 2;
        tiny.intermediate_size = 128;
        tiny.vocab_size = 256;
        let cfg = TransformerTrainConfig::new(tiny);
        let trainer = CudaTransformerTrainer::new(cfg)
            .expect("tiny CUDA trainer init — requires real GPU, #[ignore]d test");
        Rc::new(RefCell::new(trainer))
    }

    /// INV-TRAIN-011 / GATE-TRAIN-EXHAUST CUDA peer (a): confirm
    /// `CudaRealStepFn::step` panics with `GATE-TRAIN-EXHAUST` on
    /// shard-stream exhaustion — never returns the `(1.0, 1.0)`
    /// placeholder tuple. Mirrors the CPU-peer assertion in
    /// `pretrain_real.rs::tests::cpu_stepfn_exhaustion_does_not_emit_constant_placeholder`.
    #[test]
    #[ignore] // Requires GPU — run with: cargo test --features cuda -- --ignored
    fn cuda_stepfn_exhaustion_does_not_emit_constant_placeholder() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let trainer = tiny_cuda_trainer();
        let empty_iter: Box<dyn Iterator<Item = LMBatch>> = Box::new(std::iter::empty::<LMBatch>());
        let mut step = CudaRealStepFn::new(trainer, empty_iter);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = step.step(0, 1.0e-4, 128);
        }));
        let err = result.expect_err(
            "INV-TRAIN-011 (cuda): exhausted shard stream MUST panic, \
             not return a placeholder tuple",
        );
        let msg = err
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| err.downcast_ref::<&str>().copied())
            .unwrap_or("");
        assert!(
            msg.contains(crate::train::cycling_iter::EXHAUST_PANIC_PREFIX),
            "panic must cite GATE-TRAIN-EXHAUST, got: {msg:?}"
        );
    }

    /// INV-TRAIN-011 / GATE-TRAIN-EXHAUST CUDA peer (b): when the
    /// caller opts into cycling via `CyclingBatchIter`, the CUDA
    /// `StepFn` keeps emitting real GPU compute across the cycle
    /// boundary. Mirrors `pretrain_real.rs::tests::cpu_stepfn_exhaustion_cycles_or_halts`.
    #[test]
    #[ignore] // Requires GPU — run with: cargo test --features cuda -- --ignored
    fn cuda_stepfn_exhaustion_cycles_or_halts() {
        use crate::train::cycling_iter::CyclingBatchIter;
        let trainer = tiny_cuda_trainer();
        let factory = || -> Box<dyn Iterator<Item = LMBatch>> {
            let sequences: Vec<Vec<u32>> = vec![(0..5u32).collect()];
            Box::new(vec![LMBatch::from_sequences(&sequences, 0, 0)].into_iter())
        };
        let cycling = CyclingBatchIter::new(factory);
        let flag = cycling.has_cycled_flag();
        let mut step = CudaRealStepFn::new(trainer, Box::new(cycling));
        for i in 0..3 {
            let (loss, grad_norm) = step.step(i, 1.0e-4, 128);
            assert!(loss.is_finite(), "call {i}: loss must be finite");
            assert!(grad_norm.is_finite() && grad_norm >= 0.0, "call {i}: grad_norm guard");
        }
        assert!(
            flag.load(std::sync::atomic::Ordering::SeqCst),
            "CyclingBatchIter must have cycled at least once in 3 calls",
        );
    }
}
