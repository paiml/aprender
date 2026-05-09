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
use crate::train::pretrain_real::{
    build_transformer_config, llama_370m_train_config, load_init_tensors_from_apr,
    populate_trainer_from_init_tensors, validate_pretrain_init_arch_compatible,
};
use crate::train::transformer_trainer::{CudaTransformerTrainer, LMBatch, TransformerTrainConfig};
use crate::transformer::{Transformer, TransformerConfig};
use std::cell::RefCell;
use std::path::Path;
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

/// Polymorphic CUDA trainer builder for `apr pretrain --init --device cuda`
/// (§50.4 step 5f.5 — symmetric to the CPU `build_shared_trainer_with_init`).
///
/// Composes the same §50.4 step-5f machinery as the CPU path, but runs
/// it against `CudaTransformerTrainer::with_model` so the populated
/// init weights flow through GPU upload (transformer blocks via
/// `upload_blocks`, final RMSNorm via `from_host`, lm_head /
/// embed_tokens.weight via `from_host`):
///   - 5c: `build_transformer_config(init_arch)` — polymorphic dispatch
///   - 5f.1: `validate_pretrain_init_arch_compatible(init_arch)` — encoder rejection
///   - 5f.2: `load_init_tensors_from_apr(path)` — read APR weights
///   - 5f.3: `populate_trainer_from_init_tensors(transformer, &tensors)` — populate CPU model
///   - 5f.5: `CudaTransformerTrainer::with_model(populated_model, train_cfg)` — GPU upload
///
/// Behaviour:
///   init = None  → identical to `build_shared_cuda_trainer` (Llama370M
///                  from-scratch baseline with INV-ARCH-370M-001 enforced).
///   init = Some  → builds a CUDA trainer whose GPU weights derive from
///                  the populated CPU model (the populated `Transformer`
///                  is moved into `with_model` which uploads its blocks /
///                  norm / lm_head to GPU). INV-ARCH-370M-001 is NOT
///                  enforced — arch is whatever the init APR has.
///
/// Spec: SPEC-SHIP-TWO-001 §52.4 (CPU 5f.4 wireup) + §54-§56 (Qwen
/// 5g.0/5g.1 prerequisites) + this §50.4 step 5f.5 (CUDA wireup).
///
/// # Errors
///
/// Returns Err when:
/// - `init_arch.is_some() != init_path.is_some()` (caller bug — same
///   diagnostic as the CPU path's `build_shared_trainer_with_init`).
/// - `init_arch` is `Some` with `architecture = Encoder`
///   (FALSIFY-APR-PRETRAIN-ARCH-007 / FALSIFY-APR-PRETRAIN-INIT-001).
/// - `load_init_tensors_from_apr` fails (FALSIFY-APR-PRETRAIN-INIT-006).
/// - `populate_trainer_from_init_tensors` fails (FALSIFY-APR-PRETRAIN-INIT-007).
/// - `CudaTransformerTrainer::with_model` fails (CUDA init / kernel
///   pre-warm / block upload — surfaces as GATE-GPUTRAIN-002).
///
/// # Caller Contract
///
/// The caller MUST have built the binary with `--features cuda`. This
/// function is gated on `#[cfg(feature = "cuda")]` so a non-cuda build
/// will not see this symbol; the apr-cli dispatch layer routes
/// `--device cuda` to `drive_real_cuda` which calls this builder, and
/// the non-cuda stub for `drive_real_cuda` already returns the
/// rebuild-with-cuda error per `feedback_cuda_feature_footgun.md`.
pub fn build_shared_cuda_trainer_with_init(
    lr: f32,
    seq_length: usize,
    seed: u64,
    init_arch: Option<&TransformerConfig>,
    init_path: Option<&Path>,
) -> crate::Result<SharedCudaTrainer> {
    if init_arch.is_some() != init_path.is_some() {
        return Err(crate::error::Error::ConfigError(format!(
            "build_shared_cuda_trainer_with_init: init_arch and init_path must both be Some \
             or both None (caller bug; init_arch.is_some()={}, init_path.is_some()={})",
            init_arch.is_some(),
            init_path.is_some()
        )));
    }

    if let Some(cfg) = init_arch {
        validate_pretrain_init_arch_compatible(cfg).map_err(crate::error::Error::ConfigError)?;
    }

    let model_cfg = build_transformer_config(init_arch);
    let mut train_cfg = TransformerTrainConfig::new(model_cfg);
    train_cfg.lr = lr;
    train_cfg.max_seq_len = seq_length;
    train_cfg.seed = seed;

    // Build the CPU model first; populate init weights into it; then
    // hand it to CudaTransformerTrainer::with_model which uploads the
    // populated blocks, final RMSNorm, and lm_head/embed_tokens to GPU.
    // This is the symmetric path to CPU's build_shared_trainer_with_init,
    // exercising the SAME populate_trainer_from_init_tensors helper so
    // the population semantics are identical between backends.
    let mut transformer = Transformer::new(&train_cfg.model_config);

    if let Some(path) = init_path {
        let tensors = load_init_tensors_from_apr(path).map_err(crate::error::Error::ConfigError)?;
        populate_trainer_from_init_tensors(&mut transformer, &tensors)
            .map_err(crate::error::Error::ConfigError)?;
    } else {
        // From-scratch CUDA path with init=None: enforce the
        // INV-ARCH-370M-001 param-count band. Mirrors the CPU
        // `build_shared_trainer` invariant exactly.
        #[cfg(debug_assertions)]
        {
            let param_count: usize = transformer.parameters().iter().map(|t| t.len()).sum();
            debug_assert!(
                (366_000_000..=374_000_000).contains(&param_count),
                "INV-ARCH-370M-001: parameter count {param_count} outside [366M, 374M] band \
                 (from-scratch CUDA path with init=None)",
            );
        }
    }

    let trainer = CudaTransformerTrainer::with_model(transformer, train_cfg)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// FALSIFY-APR-PRETRAIN-INIT-CUDA-002 (paired-args invariant):
    /// `build_shared_cuda_trainer_with_init` MUST reject the
    /// (Some, None) and (None, Some) caller-bug states identically
    /// to the CPU `build_shared_trainer_with_init`. The two fields
    /// are paired by construction — separately optional fields are
    /// a defect class because they let a caller pass an arch
    /// without weights (silent random-init at the GPU boundary) or
    /// weights without an arch (silently fall back to Llama370M).
    ///
    /// This test fires WITHOUT a CUDA device — the args check
    /// happens before any GPU allocation — so it runs on any host
    /// even when CUDA runtime is unavailable.
    #[test]
    fn build_shared_cuda_trainer_with_init_rejects_unpaired_args() {
        use std::path::PathBuf;
        // Arch without path — Err. Use Qwen 0.5B as a concrete
        // non-Llama370M decoder config to prove the paired-args
        // gate fires before any architectural inspection.
        let cfg = TransformerConfig::qwen2_0_5b();
        let result_arch_only =
            build_shared_cuda_trainer_with_init(1.0e-4, 128, 42, Some(&cfg), None);
        assert!(
            matches!(result_arch_only, Err(_)),
            "(Some(arch), None) MUST Err — caller-bug guard"
        );

        // Path without arch — Err.
        let dummy = PathBuf::from("/tmp/does-not-exist.apr");
        let result_path_only =
            build_shared_cuda_trainer_with_init(1.0e-4, 128, 42, None, Some(&dummy));
        assert!(
            matches!(result_path_only, Err(_)),
            "(None, Some(path)) MUST Err — caller-bug guard"
        );

        // Both Err messages name the function so callers can grep
        // back to the offending invocation. We extract the message
        // by destructuring (CudaTransformerTrainer is not Debug, so
        // unwrap_err() doesn't compile) — the err is a ConfigError.
        let err_arch = match result_arch_only {
            Err(crate::error::Error::ConfigError(s)) => s,
            other => panic!("expected ConfigError, got: {:?}", other.is_ok()),
        };
        let err_path = match result_path_only {
            Err(crate::error::Error::ConfigError(s)) => s,
            other => panic!("expected ConfigError, got: {:?}", other.is_ok()),
        };
        assert!(
            err_arch.contains("build_shared_cuda_trainer_with_init"),
            "Err MUST name the function for grep-ability: {err_arch}"
        );
        assert!(
            err_path.contains("build_shared_cuda_trainer_with_init"),
            "Err MUST name the function for grep-ability: {err_path}"
        );
    }

    /// FALSIFY-APR-PRETRAIN-INIT-CUDA-003 (encoder family rejection):
    /// passing an Encoder-architecture init config to
    /// `build_shared_cuda_trainer_with_init` MUST Err — same semantic
    /// as the CPU path's `validate_pretrain_init_arch_compatible`.
    /// This proves the symmetric builder threads the §50.4 step 5f.1
    /// encoder rejection through the CUDA backend.
    ///
    /// Fires WITHOUT a CUDA device — the encoder check happens
    /// before any GPU allocation.
    #[test]
    fn build_shared_cuda_trainer_with_init_rejects_encoder_family() {
        use crate::transformer::ModelArchitecture;
        use std::path::PathBuf;
        let mut encoder_cfg = TransformerConfig::qwen2_0_5b();
        encoder_cfg.architecture = ModelArchitecture::Encoder;
        let dummy = PathBuf::from("/tmp/does-not-exist.apr");
        let result =
            build_shared_cuda_trainer_with_init(1.0e-4, 128, 42, Some(&encoder_cfg), Some(&dummy));
        assert!(matches!(result, Err(_)), "Encoder-family init MUST Err under §50.4 step 5f.1");
    }
}
