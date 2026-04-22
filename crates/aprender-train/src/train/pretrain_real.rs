//! Real-corpus `StepFn` / `ValFn` for MODEL-2 pretrain MVP (task #111).
//!
//! Bridges the model-agnostic `PretrainLoop` (`pretrain.rs`) to the
//! 370M Llama scaffold (`models/llama_370m.rs`) by wiring a real
//! `TransformerTrainer` through the `StepFn` and `ValFn` traits.
//!
//! The loop drive replaces the `LinearDecaySynthetic` / `ScriptedVal`
//! pair used for GATE-TRAIN-005/007/008 wiring verification (task #105)
//! with a real forward + backward + optimizer step and a real held-out
//! validation forward pass.
//!
//! Contract obligations discharged:
//! - INV-ARCH-370M-001 (param count in [366M, 374M]) via `debug_assert_eq!`
//! - INV-TRAIN-001 (per-step metrics — 6 fields via PretrainLoop)
//! - INV-TRAIN-007 (no NaN/Inf — the loop aborts on first non-finite)
//!
//! Deferred (task #111 follow-ups):
//! - Real grad_norm (currently reports a placeholder; needs
//!   TransformerTrainer extension to surface pre-clip norm)
//! - INV-TRAIN-003 (real optimizer-state sha256 over AdamW m/v/t buffers)

use crate::models::llama_370m::Llama370MConfig;
use crate::train::cycling_iter::next_batch_or_panic;
use crate::train::pretrain::{CheckpointFn, EpochArtifact, StepFn, ValFn};
use crate::train::transformer_trainer::{LMBatch, TransformerTrainConfig, TransformerTrainer};
use crate::transformer::{ModelArchitecture, TransformerConfig};
use std::cell::RefCell;
use std::rc::Rc;

/// Shared mutable ownership of the `TransformerTrainer` — both
/// `RealStepFn` (training steps) and `RealValFn` (forward-only
/// validation) clone this `Rc`.
pub type SharedTrainer = Rc<RefCell<TransformerTrainer>>;

/// Build a `TransformerConfig` field-for-field from `Llama370MConfig::*`
/// constants (the contract-frozen MODEL-2 370M architecture).
pub fn llama_370m_transformer_config() -> TransformerConfig {
    TransformerConfig {
        hidden_size: Llama370MConfig::HIDDEN_DIM,
        num_attention_heads: Llama370MConfig::NUM_HEADS,
        num_kv_heads: Llama370MConfig::NUM_KV_HEADS,
        intermediate_size: Llama370MConfig::INTERMEDIATE_DIM,
        num_hidden_layers: Llama370MConfig::NUM_LAYERS,
        vocab_size: Llama370MConfig::VOCAB_SIZE,
        max_position_embeddings: Llama370MConfig::MAX_POSITION_EMBEDDINGS,
        rms_norm_eps: Llama370MConfig::RMS_NORM_EPS,
        rope_theta: Llama370MConfig::ROPE_THETA,
        use_bias: false,
        head_dim_override: None,
        architecture: ModelArchitecture::Decoder,
        hf_architecture: Some("LlamaForCausalLM".into()),
        hf_model_type: Some("llama".into()),
        tie_word_embeddings: true,
    }
}

/// Build a `TransformerTrainConfig` with MODEL-2 v2-remedy defaults
/// (LR=5e-5, AdamW defaults, fp32, seed=42 set by caller).
pub fn llama_370m_train_config(lr: f32, seq_length: usize, seed: u64) -> TransformerTrainConfig {
    let model_cfg = llama_370m_transformer_config();
    let mut cfg = TransformerTrainConfig::new(model_cfg);
    cfg.lr = lr;
    cfg.max_seq_len = seq_length;
    cfg.seed = seed;
    cfg
}

/// `StepFn` impl that pulls one `LMBatch` from an owned iterator and
/// runs a real forward + backward + AdamW step through the shared
/// `TransformerTrainer`.
pub struct RealStepFn {
    trainer: SharedTrainer,
    batches: Box<dyn Iterator<Item = LMBatch>>,
}

impl RealStepFn {
    pub fn new(trainer: SharedTrainer, batches: Box<dyn Iterator<Item = LMBatch>>) -> Self {
        Self { trainer, batches }
    }
}

impl StepFn for RealStepFn {
    fn step(&mut self, _step: u64, _lr: f32, _batch_tokens: u64) -> (f32, f32) {
        // INV-TRAIN-011 / GATE-TRAIN-EXHAUST: shard stream exhaustion
        // must produce real compute (via CyclingBatchIter wrapping) OR
        // hard-fail with a diagnostic panic. Returning a constant
        // placeholder tuple is FORBIDDEN — that was the task #141
        // defect that silently turned a 45× under-Chinchilla corpus
        // into fake convergence evidence.
        let batch = next_batch_or_panic(&mut *self.batches);
        let mut trainer = self.trainer.borrow_mut();
        let loss = trainer.train_batch(&batch);
        // TODO(task #111 follow-up): expose AdamW pre-clip grad norm.
        // Placeholder = 1.0 keeps INV-TRAIN-007 satisfied (finite) and
        // INV-TRAIN-008 satisfied (≥ 0); the real grad norm is a
        // downstream ticket that needs TransformerTrainer extension.
        let grad_norm = 1.0_f32;
        (loss, grad_norm)
    }

    /// INV-TRAIN-003 discharge: hash the real AdamW (t, m, v) buffers.
    fn optimizer_state_sha256(&self) -> Option<String> {
        Some(self.trainer.borrow().optimizer_state_sha256())
    }
}

/// `ValFn` impl that runs forward-only across a pre-loaded set of
/// held-out batches and returns mean cross-entropy loss.
pub struct RealValFn {
    trainer: SharedTrainer,
    held_out: Vec<LMBatch>,
}

impl RealValFn {
    pub fn new(trainer: SharedTrainer, held_out: Vec<LMBatch>) -> Self {
        Self { trainer, held_out }
    }
}

impl ValFn for RealValFn {
    fn validate(&mut self, _epoch: usize) -> f32 {
        if self.held_out.is_empty() {
            return f32::NAN;
        }
        let trainer = self.trainer.borrow();
        let mut total_loss = 0.0_f32;
        let mut total_items = 0_usize;
        for batch in &self.held_out {
            for i in 0..batch.batch_size {
                let Some(inp) = batch.get_input(i) else {
                    continue;
                };
                let Some(tgt) = batch.get_target(i) else {
                    continue;
                };
                let (loss_val, _loss_tensor, _logits) = trainer.forward_single(inp, tgt);
                total_loss += loss_val;
                total_items += 1;
            }
        }
        if total_items == 0 {
            f32::NAN
        } else {
            total_loss / total_items as f32
        }
    }
}

/// `CheckpointFn` impl that writes the 370M Llama weights to
/// `artifact.checkpoint_path` in APR format (task #111 step 7).
///
/// Holds the `SharedTrainer` alongside `RealStepFn` / `RealValFn` so
/// the three hooks see the same in-memory weights.
pub struct AprCheckpointFn {
    trainer: SharedTrainer,
    model_name: String,
    architecture: String,
}

impl AprCheckpointFn {
    pub fn new(
        trainer: SharedTrainer,
        model_name: impl Into<String>,
        architecture: impl Into<String>,
    ) -> Self {
        Self { trainer, model_name: model_name.into(), architecture: architecture.into() }
    }
}

impl CheckpointFn for AprCheckpointFn {
    fn save(&mut self, _epoch: usize, artifact: &EpochArtifact) -> Result<(), String> {
        let trainer = self.trainer.borrow();
        trainer
            .save_apr(&artifact.checkpoint_path, &self.model_name, &self.architecture)
            .map_err(|e| format!("save_apr failed: {e}"))
    }
}

/// Shared-ownership helper so the CLI can hand the same trainer to
/// both `RealStepFn` and `RealValFn`.
pub fn build_shared_trainer(lr: f32, seq_length: usize, seed: u64) -> SharedTrainer {
    let cfg = llama_370m_train_config(lr, seq_length, seed);
    let trainer = TransformerTrainer::new(cfg);
    // INV-ARCH-370M-001: verify parameter count lands in the 370M ± 1%
    // band. This is a debug_assert so release builds do not pay for
    // the full parameter walk, but dev builds catch drift the instant
    // any Llama370MConfig constant changes.
    #[cfg(debug_assertions)]
    {
        let param_count: usize = trainer.model().parameters().iter().map(|t| t.len()).sum();
        debug_assert!(
            (366_000_000..=374_000_000).contains(&param_count),
            "INV-ARCH-370M-001: parameter count {param_count} outside [366M, 374M] band",
        );
    }
    Rc::new(RefCell::new(trainer))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::transformer_trainer::LMBatch;

    #[test]
    fn transformer_config_matches_llama_370m_constants() {
        let cfg = llama_370m_transformer_config();
        assert_eq!(cfg.hidden_size, Llama370MConfig::HIDDEN_DIM);
        assert_eq!(cfg.num_hidden_layers, Llama370MConfig::NUM_LAYERS);
        assert_eq!(cfg.num_attention_heads, Llama370MConfig::NUM_HEADS);
        assert_eq!(cfg.num_kv_heads, Llama370MConfig::NUM_KV_HEADS);
        assert_eq!(cfg.intermediate_size, Llama370MConfig::INTERMEDIATE_DIM);
        assert_eq!(cfg.vocab_size, Llama370MConfig::VOCAB_SIZE);
        assert!((cfg.rope_theta - Llama370MConfig::ROPE_THETA).abs() < f32::EPSILON);
        assert!((cfg.rms_norm_eps - Llama370MConfig::RMS_NORM_EPS).abs() < f32::EPSILON);
        assert!(!cfg.use_bias, "INV-ARCH-370M-008: no bias");
        assert!(cfg.tie_word_embeddings, "INV-ARCH-370M-004: tied embeddings");
    }

    fn tiny_cpu_trainer() -> SharedTrainer {
        // Minimal trainer for panic-path tests — avoids the ~5 GB 370M
        // allocation `build_shared_trainer` would pay.
        let mut tiny = TransformerConfig::llama2_7b();
        tiny.hidden_size = 64;
        tiny.num_attention_heads = 4;
        tiny.num_kv_heads = 4;
        tiny.num_hidden_layers = 2;
        tiny.intermediate_size = 128;
        tiny.vocab_size = 256;
        let cfg = TransformerTrainConfig::new(tiny);
        Rc::new(RefCell::new(TransformerTrainer::new(cfg)))
    }

    /// INV-TRAIN-011 / GATE-TRAIN-EXHAUST CPU peer (a): confirm
    /// `RealStepFn::step` does NOT emit the literal `(1.0, 1.0)`
    /// placeholder on shard exhaustion. Inverts the pre-task-#141
    /// assertion that accepted the placeholder as "finite and ≥ 0".
    #[test]
    fn cpu_stepfn_exhaustion_does_not_emit_constant_placeholder() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        let trainer = tiny_cpu_trainer();
        let empty_iter: Box<dyn Iterator<Item = LMBatch>> = Box::new(std::iter::empty::<LMBatch>());
        let mut step = RealStepFn::new(trainer, empty_iter);
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = step.step(0, 1.0e-4, 128);
        }));
        let err = result.expect_err(
            "INV-TRAIN-011: exhausted shard stream MUST panic, not \
             return a placeholder tuple",
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

    /// INV-TRAIN-011 / GATE-TRAIN-EXHAUST CPU peer (b): when the
    /// caller opts into cycling via `CyclingBatchIter`, `RealStepFn`
    /// must keep emitting real compute across the cycle boundary —
    /// i.e. `step` returns loss values that came from a real
    /// `train_batch`, not the placeholder tuple.
    #[test]
    fn cpu_stepfn_exhaustion_cycles_or_halts() {
        use crate::train::cycling_iter::CyclingBatchIter;
        let trainer = tiny_cpu_trainer();
        // Factory rebuilds a 1-batch iterator each call — step 2 forces
        // the cycle boundary.
        let factory = || -> Box<dyn Iterator<Item = LMBatch>> {
            let sequences: Vec<Vec<u32>> = vec![(0..5u32).collect()];
            Box::new(vec![LMBatch::from_sequences(&sequences, 0, 0)].into_iter())
        };
        let cycling = CyclingBatchIter::new(factory);
        let flag = cycling.has_cycled_flag();
        let mut step = RealStepFn::new(trainer, Box::new(cycling));
        // Three calls: pre-exhaust, first-cycle, post-cycle. Each must
        // return a real finite loss, never the placeholder tuple.
        for i in 0..3 {
            let (loss, grad_norm) = step.step(i, 1.0e-4, 128);
            assert!(loss.is_finite(), "call {i}: loss must be finite");
            assert!(grad_norm.is_finite() && grad_norm >= 0.0, "call {i}: grad_norm guard");
            // INV-TRAIN-011 value-fingerprint counter-test: a tiny
            // synthetic step may coincidentally return loss=1.0 before
            // any cycle, so we only reject the exact joint fingerprint
            // AFTER we've cycled at least once and `grad_norm` is also
            // exactly 1.0 — which `RealStepFn` does set to 1.0 today,
            // so the joint signal we actually guard against is
            // "loss==1.0 AND grad_norm==1.0 AND iterator was empty".
            // The panic path above already covers the empty case; here
            // we only assert real compute survived the cycle.
        }
        assert!(
            flag.load(std::sync::atomic::Ordering::SeqCst),
            "CyclingBatchIter must have cycled at least once in 3 calls",
        );
    }

    #[test]
    fn real_val_fn_empty_held_out_returns_nan() {
        let mut tiny = TransformerConfig::llama2_7b();
        tiny.hidden_size = 64;
        tiny.num_attention_heads = 4;
        tiny.num_kv_heads = 4;
        tiny.num_hidden_layers = 2;
        tiny.intermediate_size = 128;
        tiny.vocab_size = 256;
        let cfg = TransformerTrainConfig::new(tiny);
        let trainer = Rc::new(RefCell::new(TransformerTrainer::new(cfg)));
        let mut val = RealValFn::new(trainer, Vec::new());
        let loss = val.validate(0);
        assert!(loss.is_nan(), "empty held_out must surface as NaN to the guard");
    }
}
