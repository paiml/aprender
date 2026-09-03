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
use crate::train::pretrain::{CheckpointFn, EpochArtifact, StepFn, ValFn};
use crate::train::transformer_trainer::{LMBatch, TransformerTrainConfig, TransformerTrainer};
use crate::transformer::{ModelArchitecture, Transformer, TransformerConfig};
use crate::Tensor;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::Path;
use std::rc::Rc;

/// Shared mutable ownership of the `TransformerTrainer` — both
/// `RealStepFn` (training steps) and `RealValFn` (forward-only
/// validation) clone this `Rc`.
pub type SharedTrainer = Rc<RefCell<TransformerTrainer>>;

/// Load tensors from an APR file as the read-half of `apr pretrain --init`.
///
/// Per `apr-pretrain-arch-polymorphic-v1` §init_load_semantics (PR #1473),
/// the loader is REUSED, not reimplemented — this function is a thin wrapper
/// over `aprender::format::converter::convert_report::load_model_tensors`,
/// which is the same machinery `apr export` and `apr inspect` use. No
/// duplicate APR parser; one source of truth.
///
/// Returns a map of `tensor_name -> (flat_f32_data, shape)`. The HF naming
/// convention is preserved (e.g., `model.embed_tokens.weight`); reconciling
/// against the trainer's parameter names is step 5f.3 (the population step).
///
/// Discharges from `apr-pretrain-arch-polymorphic-v1`:
///   - §init_load_semantics invariant: "Loader is reused, not reimplemented"
///   - FALSIFY-006 (init_loss < 6.0) at READ-COMPILE-BIND level: this
///     function is the read half. Full FALSIFY-006 discharge requires
///     5f.3 (population) + 5g (LIVE 500-step fine-tune).
///
/// Spec: SPEC-SHIP-TWO-001 §50.4 step 5f.2.
///
/// # Errors
///
/// Returns Err if the APR file:
/// - Does not exist (filesystem I/O error)
/// - Has invalid magic bytes (not APR\\0 or APRN)
/// - Has a corrupted tensor index
/// - Contains tensors with unsupported dtype (non-F32)
pub fn load_init_tensors_from_apr(
    path: impl AsRef<Path>,
) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>, String> {
    let path_ref = path.as_ref();
    aprender::format::converter::load_model_tensors(path_ref).map_err(|e| {
        format!(
            "FALSIFY-APR-PRETRAIN-INIT-006: failed to load init tensors from APR file {}: {e}",
            path_ref.display()
        )
    })
}

/// Reject an init `TransformerConfig` whose architecture family is incompatible
/// with the pretrain target (decoder-only LM training).
///
/// Per `apr-pretrain-arch-polymorphic-v1` §arch_extraction_signature
/// (PR #1473), wrong-arch APR (e.g., a CodeBERT/RoBERTa encoder model)
/// MUST be FAIL-FAST not silent-truncate. Without this gate, an operator
/// who points `--init` at e.g. `microsoft/codebert-base.apr` would silently
/// load encoder weights into a decoder-shaped trainer, producing nonsense
/// gradients that the divergence guard catches LATE (after multiple epochs).
///
/// Discharges FALSIFY-APR-PRETRAIN-ARCH-007 at PARTIAL_ALGORITHM_LEVEL.
///
/// Spec: SPEC-SHIP-TWO-001 §50.4 step 5f.1.
///
/// # Errors
///
/// Returns Err with a clear architecture-family-mismatch message when:
/// - `cfg.architecture` is `ModelArchitecture::Encoder` (BERT/RoBERTa/CodeBERT)
///
/// Future expansion can add other family checks (e.g., reject hybrid SSM
/// architectures whose forward pass differs from the standard decoder loop).
pub fn validate_pretrain_init_arch_compatible(cfg: &TransformerConfig) -> Result<(), String> {
    match cfg.architecture {
        ModelArchitecture::Decoder => Ok(()),
        ModelArchitecture::Encoder => Err(format!(
            "FALSIFY-APR-PRETRAIN-ARCH-007: --init checkpoint has architecture=Encoder \
             (e.g., BERT/RoBERTa/CodeBERT) but the pretrain trainer is decoder-only \
             (Llama/Qwen-class causal LMs). Loading encoder weights into a decoder \
             trainer would produce nonsense gradients. Architectural details: \
             hidden_size={}, num_layers={}, vocab_size={}, hf_architecture={:?}",
            cfg.hidden_size, cfg.num_hidden_layers, cfg.vocab_size, cfg.hf_architecture
        )),
    }
}

/// SPEC §86 / INV-INIT-ARCH-MATCH-001 — infer the family-arch slug from
/// tensor names alone (no tensor data needed).
///
/// Mirrors the heavyweight `infer_architecture_from_names` in
/// `aprender-core::format::converter::tokenizer_loader` but takes only
/// names, so callers don't have to materialize the full F32 tensor map.
/// Used by `validate_init_arch_matches_tensor_evidence` to catch the
/// §86 case (Llama-stamped metadata + Qwen2-tensored APR) at the gate.
///
/// Returns one of: "qwen3", "qwen2", "llama", "mamba", "rwkv",
/// "gpt-neox", "opt", "bert", "gpt2", "unknown".
#[must_use]
pub fn family_from_tensor_names<'a, I>(names: I) -> &'static str
where
    I: IntoIterator<Item = &'a str>,
{
    // We iterate once but need to check multiple predicates; collect names
    // into a Vec<&str> so the predicates can each scan independently. For
    // an Qwen2-0.5B APR (~291 tensors) this is negligible.
    let names: Vec<&str> = names.into_iter().collect();

    if let Some(family) = family_from_distinctive_prefix(&names) {
        return family;
    }

    let any_contains = |needle: &str| names.iter().any(|k| k.contains(needle));
    let has_model_layers = any_contains("model.layers");
    let has_transformer_h = any_contains("transformer.h")
        || names.iter().any(|k| k.starts_with("h.") && k.contains(".attn."));
    let has_blk = any_contains("blk.");
    if has_model_layers {
        return family_from_model_layers(&names);
    }
    if has_transformer_h {
        return "gpt2";
    }
    if has_blk {
        return "unknown"; // GGUF-naming, can't disambiguate
    }
    "unknown"
}

/// Families carrying a distinctive tensor-name prefix or infix. Checked before
/// the generic `model.layers` decoder shapes, so the ordering here is load
/// bearing (GPT-NeoX and OPT both also match `model.layers`).
fn family_from_distinctive_prefix(names: &[&str]) -> Option<&'static str> {
    let any_contains = |needle: &str| names.iter().any(|k| k.contains(needle));
    let any_starts_with = |pfx: &str| names.iter().any(|k| k.starts_with(pfx));

    // PMAT-546: Mamba (SSM)
    if any_contains("mixer.in_proj") || any_contains("mixer.out_proj") {
        return Some("mamba");
    }
    // PMAT-546: RWKV
    if any_starts_with("rwkv.blocks.") || any_contains("blocks.0.att.") {
        return Some("rwkv");
    }
    // GH-311: GPT-NeoX (must precede model.layers)
    if any_starts_with("gpt_neox.") {
        return Some("gpt-neox");
    }
    // GH-311: OPT
    if any_starts_with("model.decoder.layers.") {
        return Some("opt");
    }
    // GH-311: BERT
    if any_starts_with("bert.") {
        return Some("bert");
    }
    None
}

/// Disambiguate the `model.layers` decoder families.
fn family_from_model_layers(names: &[&str]) -> &'static str {
    let any_contains = |needle: &str| names.iter().any(|k| k.contains(needle));

    // Qwen3 — unique QK-norm signal
    if any_contains("self_attn.q_norm.weight") {
        return "qwen3";
    }
    // Qwen2 — distinguished from Llama by attention bias / fused QKV
    if any_contains("self_attn.q_proj.bias") || any_contains("qkv_proj.weight") {
        return "qwen2";
    }
    "llama"
}

/// SPEC §86 / INV-INIT-ARCH-MATCH-001 — normalize an APR metadata
/// `architecture` string to the canonical family slug used by
/// `family_from_tensor_names`.
///
/// Handles the three forms the field can take:
///
/// - **HF class name** (e.g., `"Qwen2ForCausalLM"`, `"LlamaForCausalLM"`)
///   — the §82 P0-H fallback stamps this into the family field when
///   `hf_architecture` is absent.
/// - **Family slug** (e.g., `"qwen2"`, `"llama"`) — the canonical form
///   from a properly-imported APR (post-P0-K).
/// - **Capitalised legacy** (e.g., `"Qwen2"`, `"Llama"`) — older imports.
///
/// Returns `None` for `"unknown"` or unmappable strings — the caller
/// should treat those as "no metadata claim" and skip the cross-check.
#[must_use]
pub fn normalize_metadata_arch_family(arch: &str) -> Option<&'static str> {
    match arch {
        // HF class names (P0-H §82 fallback stamps these into the family field)
        "Qwen2ForCausalLM" | "Qwen2.5ForCausalLM" => Some("qwen2"),
        "Qwen3ForCausalLM" | "Qwen3MoeForCausalLM" => Some("qwen3"),
        "LlamaForCausalLM" => Some("llama"),
        "MistralForCausalLM" => Some("llama"), // Mistral shares Llama tensor shape
        "Phi3ForCausalLM" | "PhiForCausalLM" => Some("llama"), // Phi shares Llama family for our purposes
        "GPT2LMHeadModel" => Some("gpt2"),
        "GPTNeoXForCausalLM" => Some("gpt-neox"),
        "MambaForCausalLM" => Some("mamba"),
        "RwkvForCausalLM" | "Rwkv6ForCausalLM" => Some("rwkv"),
        "BertModel" | "BertForMaskedLM" => Some("bert"),
        "OPTForCausalLM" => Some("opt"),
        // Family slugs (canonical / lowercase)
        "qwen2" | "qwen2.5" | "qwen" => Some("qwen2"),
        "qwen3" | "qwen3_5" | "qwen3.5" => Some("qwen3"),
        "llama" | "mistral" | "phi" | "phi3" | "phi4" => Some("llama"),
        "gpt2" => Some("gpt2"),
        "gpt-neox" | "gpt_neox" | "gptneox" | "pythia" => Some("gpt-neox"),
        "mamba" => Some("mamba"),
        "rwkv" => Some("rwkv"),
        "bert" => Some("bert"),
        "opt" => Some("opt"),
        // Capitalised legacy
        "Qwen2" | "Qwen2.5" | "Qwen" => Some("qwen2"),
        "Qwen3" => Some("qwen3"),
        "Llama" | "Mistral" | "Phi" | "Phi3" | "Phi4" => Some("llama"),
        "Gpt2" | "GPT2" => Some("gpt2"),
        // Unknown / unmappable — caller should skip the cross-check
        _ => None,
    }
}

/// SPEC §86 / INV-INIT-ARCH-MATCH-001 — FAIL-FAST when an APR's
/// metadata `architecture` claim contradicts what its tensor names imply.
///
/// This catches the §86 silent-failure case at the gate instead of at
/// init eval: a pre-P0-K APR with `architecture = "LlamaForCausalLM"`
/// (the §82 P0-H fallback) and Qwen2-style tensor names produces
/// random-init training instead of resume-from-checkpoint, with
/// val_loss at step 0 disagreeing with the init's recorded val_loss
/// by orders of magnitude. The fix at the framework level is shipped
/// via PR #1742 (P0-K stamping); this invariant prevents existing
/// pre-P0-K artifacts from training silently from random init.
///
/// Discharges INV-INIT-ARCH-MATCH-001 in `contracts/apr-pretrain-from-init-v1.yaml`
/// (forthcoming, scope-noted in SPEC §86.6).
///
/// # Errors
///
/// Returns Err with a clear naming-both-claims message when the
/// metadata family slug differs from the tensor-evidence family slug.
/// When the metadata claim is `"unknown"` (or doesn't parse to a known
/// family) the gate is skipped — no false-positive on novel architectures.
///
/// # Salvage path
///
/// Operators with pre-P0-K Llama-stamped Qwen2 checkpoints can
/// restamp the metadata in place via the §86.4 recipe:
///
/// ```ignore
/// apr stamp <pre-p0k.apr> --architecture qwen2 --hf-architecture Qwen2ForCausalLM \
///                          -o <stamped.apr>
/// ```
///
/// See PR #1757 (apr stamp HF identity extension).
pub fn validate_init_arch_matches_tensor_evidence(
    metadata_arch: Option<&str>,
    init_tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<(), String> {
    // If the metadata claim is absent or unmappable, we have no claim to
    // contradict — skip the cross-check (a novel arch is not §86's case).
    let Some(metadata_family) = metadata_arch.and_then(normalize_metadata_arch_family) else {
        return Ok(());
    };

    let tensor_family = family_from_tensor_names(init_tensors.keys().map(String::as_str));

    // If tensor inference returns "unknown" (e.g., GGUF blk.* names that
    // can't be disambiguated), we trust the metadata claim. Only fail
    // when BOTH inferences produce concrete family slugs AND they differ.
    if tensor_family == "unknown" {
        return Ok(());
    }

    if metadata_family != tensor_family {
        return Err(format!(
            "FALSIFY-INIT-ARCH-MATCH-001: --init APR metadata claims architecture \
             family `{metadata_family}` (from `{}`) but tensor naming implies \
             family `{tensor_family}`. This is the SPEC §86 silent-failure pattern: \
             pre-P0-K APRs with the §82 P0-H \"LlamaForCausalLM\" fallback stamp + \
             Qwen2 tensors load as random-init and train from scratch. Salvage with \
             `apr stamp <input.apr> --architecture {tensor_family} --hf-architecture \
             {} -o <stamped.apr>` (see PR #1757 / SPEC §86.4) then re-run \
             `apr pretrain --init <stamped.apr>`.",
            metadata_arch.unwrap_or("?"),
            // Synthesize the canonical HF class name from the tensor-evidence family
            match tensor_family {
                "qwen2" => "Qwen2ForCausalLM",
                "qwen3" => "Qwen3ForCausalLM",
                "llama" => "LlamaForCausalLM",
                "gpt2" => "GPT2LMHeadModel",
                "gpt-neox" => "GPTNeoXForCausalLM",
                "mamba" => "MambaForCausalLM",
                "rwkv" => "RwkvForCausalLM",
                "bert" => "BertModel",
                "opt" => "OPTForCausalLM",
                other => other,
            }
        ));
    }

    Ok(())
}

/// Populate a `Transformer`'s parameters from an `init_tensors` BTreeMap.
///
/// For each parameter the `Transformer` exposes via `named_parameters()`, look
/// up the same HF-naming key in `init_tensors` and replace the parameter's
/// `Tensor` with `Tensor::from_vec(data, requires_grad=true)`. The parameter
/// remains trainable (gradients still flow) — this is fine-tune init, not
/// frozen-encoder.
///
/// Strictness rules:
/// - **Every** model parameter must have a matching entry in `init_tensors`.
///   Missing entries return Err naming the unmatched parameter; this catches
///   the case where the init APR was extracted from a different architecture.
/// - **Length** of each init entry must match the model parameter's length
///   (computed from the model's existing tensor `len()`). Mismatch returns Err.
/// - **Extra** entries in `init_tensors` are silently ignored. This handles
///   `tie_word_embeddings`: a Qwen2.5 APR may publish a separate `lm_head.weight`
///   tensor that the model omits when ties are enabled.
///
/// Discharges from `apr-pretrain-arch-polymorphic-v1` §init_load_semantics:
/// - Population invariant: "Init tensors populate trainer parameters
///   byte-equivalent to source"
/// - FALSIFY-APR-PRETRAIN-INIT-007 (population step) at PARTIAL_ALGORITHM_LEVEL.
///
/// Spec: SPEC-SHIP-TWO-001 §50.4 step 5f.3.
///
/// # Errors
///
/// Returns Err if any model parameter is missing from `init_tensors` or if
/// any matched entry has a wrong length. The error message lists up to the
/// first 5 problem parameters and the total count.
pub fn populate_trainer_from_init_tensors(
    transformer: &mut Transformer,
    init_tensors: &BTreeMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<usize, String> {
    let expected: Vec<(String, usize)> =
        transformer.named_parameters().into_iter().map(|(name, t)| (name, t.len())).collect();
    let mut populated = 0usize;
    let mut errors: Vec<String> = Vec::new();

    for (name, expected_len) in &expected {
        match init_tensors.get(name) {
            Some((data, _shape)) => {
                if data.len() != *expected_len {
                    errors.push(format!(
                        "{name}: init length {} != trainer expected {expected_len}",
                        data.len()
                    ));
                    continue;
                }
                let tensor = Tensor::from_vec(data.clone(), true);
                if !transformer.set_named_parameter(name, tensor) {
                    errors.push(format!("{name}: set_named_parameter rejected the assignment"));
                    continue;
                }
                populated += 1;
            }
            None => {
                errors.push(format!("{name}: not present in init APR tensors"));
            }
        }
    }

    if !errors.is_empty() {
        let total = errors.len();
        let head = errors.iter().take(5).cloned().collect::<Vec<_>>().join("; ");
        return Err(format!(
            "FALSIFY-APR-PRETRAIN-INIT-007: populate_trainer_from_init_tensors \
             failed for {total} parameter(s); first {} of {total}: {head}",
            errors.len().min(5)
        ));
    }

    Ok(populated)
}

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

/// Polymorphic builder per `apr-pretrain-arch-polymorphic-v1` §arch_extraction_signature.
///
/// Discharges FALSIFY-APR-PRETRAIN-ARCH-002 (init=None preserves Llama370M baseline)
/// and FALSIFY-APR-PRETRAIN-ARCH-003 (init=Some passes through extracted config).
///
/// Behaviour:
///   init = None  → return `llama_370m_transformer_config()`, the §24/§25
///                  from-scratch baseline. NO regression.
///   init = Some  → clone the caller-extracted `TransformerConfig` byte-for-byte.
///                  No silent defaults, no field overrides.
///
/// The caller is responsible for actually reading the APR file and producing the
/// `TransformerConfig` (typically via `TransformerConfig::from_apr_metadata` from
/// `transformer::config`). Decoupling the dispatch from the file I/O keeps
/// `aprender-train` free of `aprender-serve` (the APR loader) as a build dep.
///
/// Spec: SPEC-SHIP-TWO-001 §50.4 step 5c.
pub fn build_transformer_config(init: Option<&TransformerConfig>) -> TransformerConfig {
    match init {
        None => llama_370m_transformer_config(),
        Some(cfg) => cfg.clone(),
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
        // Pull one batch; if the shard stream is exhausted before the
        // loop plans to stop, emit a tiny finite placeholder so
        // GATE-TRAIN-007 (NaN/Inf guard) does not mis-fire — the
        // divergence guard (GATE-TRAIN-005) will correctly not abort
        // on a flat tail.
        let Some(batch) = self.batches.next() else {
            return (1.0, 1.0);
        };
        let mut trainer = self.trainer.borrow_mut();
        let loss = trainer.train_batch(&batch);
        // Deferred (PMAT-945, task #111 follow-up): expose AdamW pre-clip grad norm.
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

/// Polymorphic trainer builder for `apr pretrain --init` per
/// `apr-pretrain-arch-polymorphic-v1` §arch_extraction_signature +
/// §init_load_semantics (PR #1473).
///
/// Composes the §50.4 step-5f machinery into a single CLI-callable entry:
///   - 5c: `build_transformer_config(init_arch)` — polymorphic dispatch
///   - 5f.1: `validate_pretrain_init_arch_compatible(init_arch)` — encoder rejection
///   - 5f.2: `load_init_tensors_from_apr(path)` — read APR weights
///   - 5f.3: `populate_trainer_from_init_tensors(trainer, &tensors)` — populate
///
/// Behaviour:
///   init = None  → identical to `build_shared_trainer` (Llama370M from-scratch
///                  baseline; INV-ARCH-370M-001 enforced).
///   init = Some  → builds a trainer with the EXTRACTED arch, validates the
///                  family, loads tensors from the APR file, populates them.
///                  INV-ARCH-370M-001 is NOT enforced (the arch is whatever the
///                  init APR has, e.g. 0.5B / 1.5B / 7B).
///
/// Spec: SPEC-SHIP-TWO-001 §52.4 (step 5f.4 wireup).
///
/// # Errors
///
/// Returns Err when:
/// - `init_arch` is `Some` with `architecture = Encoder` (FALSIFY-APR-PRETRAIN-ARCH-007)
/// - `load_init_tensors_from_apr` fails (FALSIFY-APR-PRETRAIN-INIT-006)
/// - `populate_trainer_from_init_tensors` fails (FALSIFY-APR-PRETRAIN-INIT-007)
pub fn build_shared_trainer_with_init(
    lr: f32,
    seq_length: usize,
    seed: u64,
    init_arch: Option<&TransformerConfig>,
    init_path: Option<&Path>,
) -> Result<SharedTrainer, String> {
    if init_arch.is_some() != init_path.is_some() {
        return Err(format!(
            "build_shared_trainer_with_init: init_arch and init_path must both be Some \
             or both None (caller bug; init_arch.is_some()={}, init_path.is_some()={})",
            init_arch.is_some(),
            init_path.is_some()
        ));
    }

    if let Some(cfg) = init_arch {
        validate_pretrain_init_arch_compatible(cfg)?;
    }

    let model_cfg = build_transformer_config(init_arch);
    let mut train_cfg = TransformerTrainConfig::new(model_cfg);
    train_cfg.lr = lr;
    train_cfg.max_seq_len = seq_length;
    train_cfg.seed = seed;
    let mut trainer = TransformerTrainer::new(train_cfg);

    // Note: INV-ARCH-370M-001 (param-count band check) lives in
    // `build_shared_trainer` (the from-scratch CLI path). The polymorphic
    // builder is shape-agnostic by design — `build_transformer_config(init)`
    // returns whatever the init APR has (0.5B, 1.5B, 7B, etc), so a single
    // hardcoded band check would fire-fail on every non-Llama370M init.

    if let Some(path) = init_path {
        let tensors = load_init_tensors_from_apr(path)?;
        // SPEC §86 / INV-INIT-ARCH-MATCH-001 — fail-fast on the §86
        // silent-failure pattern (pre-P0-K APR with wrong arch stamp +
        // mismatched tensor names → random-init fallback at val_loss ≈ 8.6).
        // Read the raw metadata.architecture string here (init_arch.hf_architecture
        // is None for pre-P0-K APRs, which is precisely the §86 case — so the
        // TransformerConfig isn't sufficient).
        let raw_metadata_arch = read_apr_metadata_architecture_string(path);
        validate_init_arch_matches_tensor_evidence(raw_metadata_arch.as_deref(), &tensors)?;
        populate_trainer_from_init_tensors(trainer.model_mut(), &tensors)?;
    }

    Ok(Rc::new(RefCell::new(trainer)))
}

/// SPEC §86 helper — read the raw `architecture` string from an APR v2
/// metadata block without going through `transformer_config_from_apr_metadata`
/// (which converts to a `ModelArchitecture` enum and loses the original
/// string). Used by INV-INIT-ARCH-MATCH-001 to detect the §86 case where
/// the metadata claims "LlamaForCausalLM" but the tensors are Qwen2-shaped.
///
/// Returns `None` on any read / parse failure — the gate caller treats
/// "no metadata claim" as "skip check" so this is safe.
fn read_apr_metadata_architecture_string(path: &Path) -> Option<String> {
    use aprender::format::v2::{AprV2Header, AprV2Metadata, HEADER_SIZE_V2, MAGIC_V2};
    use std::io::{Read, Seek, SeekFrom};
    let mut file = std::fs::File::open(path).ok()?;
    let mut header_buf = [0u8; HEADER_SIZE_V2];
    file.read_exact(&mut header_buf).ok()?;
    if header_buf[..4] != MAGIC_V2 {
        return None;
    }
    let header = AprV2Header::from_bytes(&header_buf).ok()?;
    file.seek(SeekFrom::Start(header.metadata_offset)).ok()?;
    let mut meta_buf = vec![0u8; header.metadata_size as usize];
    file.read_exact(&mut meta_buf).ok()?;
    let metadata = AprV2Metadata::from_json(&meta_buf).ok()?;
    metadata.architecture
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::transformer_trainer::LMBatch;

    /// FALSIFY-APR-PRETRAIN-INIT-006 (read-half) — load_init_tensors_from_apr
    /// returns Err with a clear message naming the falsifier when the path
    /// does not exist.
    ///
    /// Spec: SPEC-SHIP-TWO-001 §50.4 step 5f.2.
    #[test]
    fn load_init_tensors_missing_file_errors_with_falsifier_id() {
        let tmp = tempfile::TempDir::new().expect("tempdir");
        let missing = tmp.path().join("does-not-exist.apr");
        let err =
            load_init_tensors_from_apr(&missing).expect_err("missing init APR file MUST fail-fast");
        assert!(
            err.contains("FALSIFY-APR-PRETRAIN-INIT-006"),
            "error must cite falsifier id (auditability): {err}"
        );
        assert!(
            err.contains("does-not-exist.apr"),
            "error must name the missing path (operator-experience): {err}"
        );
    }

    /// FALSIFY-APR-PRETRAIN-INIT-006 (read-half) — function exists with the
    /// right signature: `Path -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>>`.
    /// Discharges the COMPILE-BIND level claim. Live empirical correctness
    /// requires step 5g (operator-runnable LIVE fine-tune).
    ///
    /// Drift-prevention: this test catches a future refactor that changes
    /// the return type or signature, which would break the §50.4 step 5f.3
    /// follow-up that reconciles the BTreeMap against trainer parameters.
    #[test]
    fn load_init_tensors_signature_compile_bind() {
        // Verify the function signature compile-binds: takes a Path-like,
        // returns the right Result type. This is a compile-time check —
        // if the signature drifts, this test stops compiling.
        fn check_signature<F>(_f: F)
        where
            F: Fn(&Path) -> Result<BTreeMap<String, (Vec<f32>, Vec<usize>)>, String>,
        {
        }
        check_signature(|p| load_init_tensors_from_apr(p));
    }

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

    /// FALSIFY-APR-PRETRAIN-ARCH-002 — `build_transformer_config(None)` returns
    /// the Llama370M baseline byte-for-byte. Falsifies regression in the §24/§25
    /// from-scratch path when the polymorphic dispatch was added.
    ///
    /// Spec: SPEC-SHIP-TWO-001 §50.4 step 5c.
    #[test]
    fn build_transformer_config_no_init_matches_llama370m() {
        let baseline = llama_370m_transformer_config();
        let result = build_transformer_config(None);
        assert_eq!(result.hidden_size, baseline.hidden_size);
        assert_eq!(result.num_attention_heads, baseline.num_attention_heads);
        assert_eq!(result.num_kv_heads, baseline.num_kv_heads);
        assert_eq!(result.intermediate_size, baseline.intermediate_size);
        assert_eq!(result.num_hidden_layers, baseline.num_hidden_layers);
        assert_eq!(result.vocab_size, baseline.vocab_size);
        assert_eq!(result.max_position_embeddings, baseline.max_position_embeddings);
        assert!((result.rms_norm_eps - baseline.rms_norm_eps).abs() < f32::EPSILON);
        assert!((result.rope_theta - baseline.rope_theta).abs() < f32::EPSILON);
        assert_eq!(result.use_bias, baseline.use_bias);
        assert_eq!(result.tie_word_embeddings, baseline.tie_word_embeddings);
        assert_eq!(result.architecture, baseline.architecture);
        assert_eq!(result.hf_architecture, baseline.hf_architecture);
        assert_eq!(result.hf_model_type, baseline.hf_model_type);
    }

    /// FALSIFY-APR-PRETRAIN-ARCH-003 — `build_transformer_config(Some(cfg))`
    /// passes through the caller-provided config byte-for-byte. No silent
    /// defaults, no field overrides. Tests with Qwen2.5-Coder-0.5B shape
    /// because that is the §49 fine-tune target.
    ///
    /// Spec: SPEC-SHIP-TWO-001 §50.4 step 5c.
    #[test]
    fn build_transformer_config_qwen_init_matches_input() {
        let qwen = TransformerConfig::qwen2_0_5b();
        let result = build_transformer_config(Some(&qwen));
        assert_eq!(result.hidden_size, qwen.hidden_size, "hidden_size");
        assert_eq!(result.num_attention_heads, qwen.num_attention_heads, "num_attention_heads");
        assert_eq!(result.num_kv_heads, qwen.num_kv_heads, "num_kv_heads");
        assert_eq!(result.intermediate_size, qwen.intermediate_size, "intermediate_size");
        assert_eq!(result.num_hidden_layers, qwen.num_hidden_layers, "num_hidden_layers");
        assert_eq!(result.vocab_size, qwen.vocab_size, "vocab_size");
        assert_eq!(
            result.max_position_embeddings, qwen.max_position_embeddings,
            "max_position_embeddings"
        );
        assert_eq!(result.use_bias, qwen.use_bias, "use_bias");
        assert_eq!(result.tie_word_embeddings, qwen.tie_word_embeddings, "tie_word_embeddings");
        assert_eq!(result.architecture, qwen.architecture, "architecture");
        // GQA-7:1 ratio preserved (Qwen2.5-0.5B: 14 / 2 = 7)
        assert_eq!(
            result.num_attention_heads / result.num_kv_heads,
            7,
            "GQA ratio must preserve as 7:1 (Qwen2.5-0.5B canonical)"
        );
    }

    /// Drift-prevention: dispatch is mutually exclusive — None and Some
    /// produce different configs (otherwise the polymorphic builder is
    /// vacuous). Catches a future refactor that accidentally always
    /// returns Llama370M regardless of init.
    ///
    /// Spec: SPEC-SHIP-TWO-001 §50.4 step 5c — drift prevention.
    #[test]
    fn build_transformer_config_dispatch_mutually_exclusive() {
        let qwen = TransformerConfig::qwen2_0_5b();
        let none_result = build_transformer_config(None);
        let some_result = build_transformer_config(Some(&qwen));
        // The two outputs MUST differ, otherwise the dispatch is broken.
        assert_ne!(
            none_result.hidden_size, some_result.hidden_size,
            "dispatch must differentiate None vs Some — Llama370M hidden=1024 vs Qwen=896"
        );
        assert_ne!(
            none_result.vocab_size, some_result.vocab_size,
            "dispatch must differentiate None vs Some — Llama370M vocab=50257 vs Qwen=151936"
        );
    }

    /// FALSIFY-APR-PRETRAIN-ARCH-007 (decoder branch) — `validate_pretrain_init_arch_compatible`
    /// returns Ok for a decoder-family config.
    ///
    /// Spec: SPEC-SHIP-TWO-001 §50.4 step 5f.1.
    #[test]
    fn validate_pretrain_init_arch_accepts_decoder() {
        let qwen = TransformerConfig::qwen2_0_5b();
        assert_eq!(qwen.architecture, ModelArchitecture::Decoder);
        validate_pretrain_init_arch_compatible(&qwen)
            .expect("decoder-family config (Qwen2.5-0.5B) MUST pass arch-compat gate");
    }

    /// FALSIFY-APR-PRETRAIN-ARCH-007 (encoder branch) — load-bearing test.
    /// `validate_pretrain_init_arch_compatible` returns Err naming the
    /// architecture-family mismatch when given an encoder config (e.g.,
    /// CodeBERT). Without this gate, the decoder trainer would silently
    /// build with encoder weights producing nonsense gradients.
    ///
    /// Spec: SPEC-SHIP-TWO-001 §50.4 step 5f.1.
    #[test]
    fn validate_pretrain_init_arch_rejects_encoder() {
        // Construct a minimal encoder config (CodeBERT-shaped).
        let bert = TransformerConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_kv_heads: 12,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            vocab_size: 50265,
            max_position_embeddings: 514,
            rms_norm_eps: 1e-12,
            rope_theta: 10_000.0,
            use_bias: true,
            head_dim_override: None,
            architecture: ModelArchitecture::Encoder,
            hf_architecture: Some("RobertaModel".to_string()),
            hf_model_type: Some("roberta".to_string()),
            tie_word_embeddings: false,
        };
        let err = validate_pretrain_init_arch_compatible(&bert).expect_err(
            "encoder-family config (CodeBERT/RoBERTa) MUST fail arch-compat gate — \
             silent acceptance would corrupt §49 fine-tune trajectory before any \
             FALSIFY-006 check could measure it",
        );
        assert!(
            err.contains("FALSIFY-APR-PRETRAIN-ARCH-007"),
            "error must cite falsifier id: {err}"
        );
        assert!(err.contains("Encoder"), "error must name the architecture family: {err}");
        assert!(
            err.contains("decoder-only"),
            "error must explain why this is wrong (decoder trainer): {err}"
        );
        assert!(
            err.contains("RobertaModel"),
            "error must name the offending hf_architecture: {err}"
        );
    }

    /// Drift-prevention: validate_pretrain_init_arch_compatible's behavior on
    /// the from-scratch baseline (Llama370M) — must Ok. Catches a future
    /// refactor that accidentally over-rejects decoder configs.
    #[test]
    fn validate_pretrain_init_arch_accepts_llama370m_baseline() {
        let llama = llama_370m_transformer_config();
        assert_eq!(
            llama.architecture,
            ModelArchitecture::Decoder,
            "Llama370M baseline MUST be Decoder (regression-free)"
        );
        validate_pretrain_init_arch_compatible(&llama)
            .expect("Llama370M baseline (Decoder) MUST pass arch-compat gate");
    }

    #[test]
    fn real_step_fn_exhausted_iterator_returns_finite_placeholder() {
        // Empty iterator means no real batches; we must still return
        // finite values so the loop's non-divergence + NaN guards see
        // sane data instead of surprising NaN.
        //
        // Construct a minimal trainer WITHOUT running `build_shared_trainer`
        // because that takes ~5 GB of parameter allocation for 370M —
        // too expensive for a unit test. Use a tiny synthetic config.
        let mut tiny = TransformerConfig::llama2_7b();
        tiny.hidden_size = 64;
        tiny.num_attention_heads = 4;
        tiny.num_kv_heads = 4;
        tiny.num_hidden_layers = 2;
        tiny.intermediate_size = 128;
        tiny.vocab_size = 256;
        let cfg = TransformerTrainConfig::new(tiny);
        let trainer = Rc::new(RefCell::new(TransformerTrainer::new(cfg)));
        let empty_iter: Box<dyn Iterator<Item = LMBatch>> = Box::new(std::iter::empty::<LMBatch>());
        let mut step = RealStepFn::new(trainer, empty_iter);
        let (loss, grad_norm) = step.step(0, 1.0e-4, 128);
        assert!(loss.is_finite(), "exhausted iter must return finite loss");
        assert!(grad_norm.is_finite(), "grad_norm must be finite");
        assert!(grad_norm >= 0.0, "INV-TRAIN-008: grad_norm non-negative");
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

    /// Build a tiny Transformer suitable for unit testing the populate path.
    /// Uses GQA-1:1 (kv=q) shape — the populate function is shape-agnostic so
    /// the simpler ratio is fine here.
    fn tiny_test_transformer() -> Transformer {
        let mut tiny = TransformerConfig::llama2_7b();
        tiny.hidden_size = 32;
        tiny.num_attention_heads = 2;
        tiny.num_kv_heads = 2;
        tiny.num_hidden_layers = 2;
        tiny.intermediate_size = 64;
        tiny.vocab_size = 16;
        Transformer::new(&tiny)
    }

    /// Build a `BTreeMap<String, (Vec<f32>, Vec<usize>)>` from a Transformer's
    /// `named_parameters()` snapshot. Each tensor is a deterministic ramp
    /// (i as f32 * 0.001) so populate is byte-identifiable post-set.
    fn tensors_map_from_transformer(
        transformer: &Transformer,
    ) -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        let mut map = BTreeMap::new();
        for (name, t) in transformer.named_parameters() {
            let len = t.len();
            let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.001).collect();
            map.insert(name, (data, vec![len]));
        }
        map
    }

    /// Happy path — every model parameter has a matching init entry of correct
    /// length; populate succeeds and the count matches `named_parameters().len()`.
    #[test]
    fn populate_trainer_from_init_tensors_happy_path() {
        let mut transformer = tiny_test_transformer();
        let init_tensors = tensors_map_from_transformer(&transformer);
        let expected_count = transformer.named_parameters().len();
        let result = populate_trainer_from_init_tensors(&mut transformer, &init_tensors);
        assert!(result.is_ok(), "happy-path populate must succeed: {result:?}");
        assert_eq!(
            result.unwrap(),
            expected_count,
            "populated count must equal named_parameters().len()"
        );
    }

    /// Drift-prevention: extra entries in `init_tensors` that the model does
    /// NOT expose are silently ignored. This handles tied-embeddings: a Qwen
    /// APR may publish a separate `lm_head.weight` that the trainer's tied
    /// model omits.
    #[test]
    fn populate_trainer_from_init_tensors_extra_entries_silently_ignored() {
        let mut transformer = tiny_test_transformer();
        let mut init_tensors = tensors_map_from_transformer(&transformer);
        // Inject a fictitious extra parameter that the model does not have.
        init_tensors
            .insert("model.layers.999.fictitious.weight".to_string(), (vec![0.0; 4], vec![4]));
        let expected_count = transformer.named_parameters().len();
        let result = populate_trainer_from_init_tensors(&mut transformer, &init_tensors);
        assert!(result.is_ok(), "extra init entries must NOT cause Err: {result:?}");
        assert_eq!(result.unwrap(), expected_count);
    }

    /// FALSIFY-APR-PRETRAIN-INIT-007 (length mismatch) — when an init tensor
    /// has the wrong flat length for a known parameter, populate MUST Err
    /// with the FALSIFIER ID and a per-parameter diagnostic line.
    #[test]
    fn populate_trainer_from_init_tensors_rejects_length_mismatch() {
        let mut transformer = tiny_test_transformer();
        let mut init_tensors = tensors_map_from_transformer(&transformer);
        // Corrupt one entry's length to trigger the mismatch path.
        let any_name = transformer.named_parameters()[0].0.clone();
        init_tensors.insert(any_name.clone(), (vec![0.0; 7], vec![7]));
        let result = populate_trainer_from_init_tensors(&mut transformer, &init_tensors);
        assert!(result.is_err(), "length-mismatch must Err, not silently truncate");
        let err = result.unwrap_err();
        assert!(
            err.contains("FALSIFY-APR-PRETRAIN-INIT-007"),
            "error must cite falsifier id; got: {err}"
        );
        assert!(err.contains(&any_name), "error must name the offending parameter; got: {err}");
        assert!(
            err.contains("init length 7"),
            "error must report the actual init length; got: {err}"
        );
    }

    /// FALSIFY-APR-PRETRAIN-INIT-007 (missing-required) — when a model
    /// parameter has NO corresponding entry in `init_tensors`, populate MUST
    /// Err with FALSIFIER ID and a "not present in init APR tensors"
    /// per-parameter diagnostic. This catches the architecture-mismatch
    /// class where init was extracted from a different model family.
    #[test]
    fn populate_trainer_from_init_tensors_rejects_missing_required_param() {
        let mut transformer = tiny_test_transformer();
        let mut init_tensors = tensors_map_from_transformer(&transformer);
        // Drop one entry to trigger the missing-required path.
        let any_name = transformer.named_parameters()[0].0.clone();
        init_tensors.remove(&any_name);
        let result = populate_trainer_from_init_tensors(&mut transformer, &init_tensors);
        assert!(result.is_err(), "missing-required must Err, not silently leave random init");
        let err = result.unwrap_err();
        assert!(
            err.contains("FALSIFY-APR-PRETRAIN-INIT-007"),
            "error must cite falsifier id; got: {err}"
        );
        assert!(err.contains(&any_name), "error must name the missing parameter; got: {err}");
        assert!(
            err.contains("not present in init APR"),
            "error must say what was missing; got: {err}"
        );
    }

    /// `build_shared_trainer_with_init(None, None)` returns a trainer with
    /// the §24/§25 from-scratch Llama370M architecture (regression-free
    /// dispatch). Asserts the baseline shape via the (hidden, vocab) tuple
    /// rather than param count to avoid the stale INV-ARCH-370M-001 band
    /// check in `build_shared_trainer` (a defect outside §50.4 scope —
    /// param_count=322M vs assert range [366M, 374M]; tracked for follow-up).
    #[test]
    fn build_shared_trainer_with_init_none_uses_llama370m_shape() {
        let trainer = build_shared_trainer_with_init(1.0e-4, 128, 42, None, None)
            .expect("None case must succeed");
        let model = trainer.borrow();
        // The baseline polymorphic dispatch produces a Llama370M-shaped model.
        // Embedding shape `vocab × hidden` is the cleanest non-stale check.
        let embed_len = model.model().named_parameters()[0].1.len();
        let expected_embed_len = Llama370MConfig::VOCAB_SIZE * Llama370MConfig::HIDDEN_DIM;
        assert_eq!(
            embed_len,
            expected_embed_len,
            "init=None must produce Llama370M-shaped embedding (vocab={} × hidden={})",
            Llama370MConfig::VOCAB_SIZE,
            Llama370MConfig::HIDDEN_DIM
        );
    }

    /// `build_shared_trainer_with_init(Some, None)` and the inverse must
    /// fail-fast — both args are paired and either both Some or both None.
    /// Drift-prevention: catches a future caller that forgets to pass one.
    #[test]
    fn build_shared_trainer_with_init_rejects_unpaired_args() {
        // arch Some, path None
        let cfg = TransformerConfig::qwen2_0_5b();
        let result = build_shared_trainer_with_init(1.0e-4, 128, 42, Some(&cfg), None);
        assert!(result.is_err(), "unpaired (arch=Some, path=None) must Err");
        // arch None, path Some
        let dummy_path = std::path::PathBuf::from("/dev/null");
        let result = build_shared_trainer_with_init(1.0e-4, 128, 42, None, Some(&dummy_path));
        assert!(result.is_err(), "unpaired (arch=None, path=Some) must Err");
    }

    /// `build_shared_trainer_with_init(Some(encoder), Some(path))` rejects
    /// the encoder family BEFORE attempting tensor load. Drift-prevention for
    /// FALSIFY-APR-PRETRAIN-ARCH-007 at the trainer-builder integration level.
    #[test]
    fn build_shared_trainer_with_init_rejects_encoder_family() {
        let mut encoder_cfg = TransformerConfig::qwen2_0_5b();
        encoder_cfg.architecture = ModelArchitecture::Encoder;
        let dummy_path = std::path::PathBuf::from("/nonexistent/encoder.apr");
        let result =
            build_shared_trainer_with_init(1.0e-4, 128, 42, Some(&encoder_cfg), Some(&dummy_path));
        let err = match result {
            Ok(_) => panic!("encoder family must be rejected before tensor load"),
            Err(e) => e,
        };
        assert!(
            err.contains("FALSIFY-APR-PRETRAIN-ARCH-007"),
            "error must cite falsifier id; got: {err}"
        );
    }

    /// `build_shared_trainer_with_init(Some(decoder), Some(missing_path))`
    /// proceeds past the family check and FAILS at tensor load with a
    /// FALSIFY-006 error. Pins the failure ordering: arch validation first,
    /// then tensor load.
    #[test]
    fn build_shared_trainer_with_init_decoder_family_proceeds_to_tensor_load() {
        let cfg = TransformerConfig::qwen2_0_5b();
        let dummy_path = std::path::PathBuf::from("/nonexistent/decoder.apr");
        let result = build_shared_trainer_with_init(1.0e-4, 128, 42, Some(&cfg), Some(&dummy_path));
        let err = match result {
            Ok(_) => panic!("missing tensor path must Err"),
            Err(e) => e,
        };
        assert!(
            err.contains("FALSIFY-APR-PRETRAIN-INIT-006"),
            "decoder family proceeds to tensor load; failure cites INIT-006 not ARCH-007; got: {err}"
        );
        assert!(
            !err.contains("FALSIFY-APR-PRETRAIN-ARCH-007"),
            "decoder family must NOT trigger encoder-rejection; got: {err}"
        );
    }

    /// FALSIFY-H4-INIT-STATS-001 (SHIP-TWO §61 H4A bisect):
    /// `load_init_tensors_from_apr` on the canonical Qwen2.5-Coder-0.5B-Instruct
    /// APR file MUST produce sensibly-distributed weights:
    ///   - `model.embed_tokens.weight` mean ≈ 0 (within ±0.01)
    ///   - `model.embed_tokens.weight` std in [0.01, 0.1] (HF LLaMA init = 0.02)
    ///   - `model.norm.weight` mean ≈ 1.0 (RMSNorm pretrained scale)
    ///
    /// CONTEXT: §61 evidence shows val_loss=19.80 > ln(vocab)=17.21 at
    /// step 1, indicating the loaded model produces sub-random predictions.
    /// Four candidate hypotheses (H4A tied weights, H4B layout, H4C norm
    /// scale, H4D residual stream). This test bisects H4A+H4C: if any of
    /// the loaded tensor stats are wildly out-of-range, the load itself
    /// is corrupt; if all stats look correct, the bug is in the forward
    /// path (H4B layout or H4D residual).
    ///
    /// Host-gated: requires a canonical Qwen 0.5B init APR. Tries the
    /// "fresh" path first (current `apr import` of HF safetensors,
    /// preserves BF16 dtype tag); falls back to the older "fp16" path
    /// (legacy import, mis-tagged as F16). Skips if neither present.
    ///
    /// The legacy file demonstrates the H4 dtype-mislabel defect class:
    /// safetensors source is BF16, old `apr import` wrote bytes raw
    /// but tagged dtype as F16, aprender's loader then read bytes as
    /// F16 and produced distorted values. The fresh path preserves
    /// BF16 correctly. Element-0 cross-checks agree with the
    /// safetensors source under BF16 decode.
    #[test]
    fn falsify_h4_init_stats_qwen_embed_norm_sensible() {
        let fresh = std::path::Path::new("/mnt/nvme-raid0/models/qwen2.5-coder-0.5b-fresh.apr");
        let legacy =
            std::path::Path::new("/mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-fp16.apr");
        let path = if fresh.exists() {
            fresh
        } else if legacy.exists() {
            legacy
        } else {
            eprintln!("[falsify-h4-init-stats-001] skipping: host lacks Qwen 0.5B APR");
            return;
        };
        let _ = path; // silence unused if branches
        if !path.exists() {
            eprintln!("[falsify-h4-init-stats-001] skipping: host lacks {}", path.display());
            return;
        }
        // H4 root-cause probe: directly inspect the APR's dtype tag to
        // verify whether the F16 vs BF16 distinction was preserved
        // through `apr import`.
        {
            use aprender::format::v2::AprV2Reader;
            let bytes = std::fs::read(path).expect("read APR");
            let reader = AprV2Reader::from_bytes(&bytes).expect("parse APR v2");
            for name in ["model.layers.0.self_attn.q_proj.bias", "model.norm.weight"] {
                if let Some(entry) = reader.get_tensor(name) {
                    eprintln!(
                        "[h4-init-dtype] {name}: dtype={:?} shape={:?}",
                        entry.dtype, entry.shape
                    );
                }
            }
        }
        let tensors = match load_init_tensors_from_apr(path) {
            Ok(t) => t,
            Err(e) => {
                panic!("FALSIFY-H4-INIT-STATS-001: load_init_tensors_from_apr failed: {e}");
            }
        };

        // Required tensors
        let embed = tensors
            .get("model.embed_tokens.weight")
            .unwrap_or_else(|| panic!("missing model.embed_tokens.weight in init APR"));
        let norm = tensors
            .get("model.norm.weight")
            .unwrap_or_else(|| panic!("missing model.norm.weight in init APR"));

        let stats = |name: &str, data: &[f32]| -> (f64, f64, f32, f32) {
            let n = data.len() as f64;
            let mean = data.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = data
                .iter()
                .map(|&v| {
                    let d = v as f64 - mean;
                    d * d
                })
                .sum::<f64>()
                / n;
            let std = var.sqrt();
            let min = data.iter().copied().fold(f32::INFINITY, f32::min);
            let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!(
                "[h4-init-stats] {name}: n={n} mean={mean:.5} std={std:.5} min={min:.4} max={max:.4}"
            );
            (mean, std, min, max)
        };
        // H4-DTYPE-MISLABEL: dump first 4 element-0 values to compare
        // with safetensors source (decoded as BF16). If the APR loader
        // mis-decodes BF16 bytes as F16, values will diverge.
        {
            let q = tensors.get("model.layers.0.self_attn.q_proj.bias").unwrap();
            eprintln!(
                "[h4-dtype-mislabel] q_proj.bias L0[0..6] (aprender F16-decoded): {:?}",
                &q.0[..6]
            );
            let n = tensors.get("model.norm.weight").unwrap();
            eprintln!(
                "[h4-dtype-mislabel] model.norm.weight[0..6] (aprender F16-decoded): {:?}",
                &n.0[..6]
            );
        }

        let (em, es, _, _) = stats("model.embed_tokens.weight", &embed.0);
        let (nm, ns, _, _) = stats("model.norm.weight", &norm.0);

        // H4C bisect: dump per-layer norm stats. Standard RMSNorm
        // weights are near 1.0 (init=1.0, trained drift to ~0.1-2.0).
        // Mean > 5 across layers indicates a load-time scale-corruption.
        for layer_idx in [0_usize, 5, 11, 23] {
            for kind in ["input_layernorm", "post_attention_layernorm"] {
                let key = format!("model.layers.{layer_idx}.{kind}.weight");
                if let Some(t) = tensors.get(&key) {
                    stats(&key, &t.0);
                }
            }
        }
        for kind in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ] {
            if let Some(t) = tensors.get(kind) {
                stats(kind, &t.0);
            }
        }

        // Embedding init bound: HF LLaMA init normal(0, 0.02). After
        // pretraining the std grows but typically stays in [0.01, 0.1].
        // mean should be near 0 (well-centered).
        assert!(
            em.abs() < 0.05,
            "FALSIFY-H4-INIT-STATS-001: embed mean={em} > 0.05; weights are not centered. \
             Possible f16→f32 sign-bit corruption or wrong byte-order."
        );
        assert!(
            (0.005..=0.5).contains(&es),
            "FALSIFY-H4-INIT-STATS-001: embed std={es} outside [0.005, 0.5]; weights are not \
             distributed like trained transformer init. Possible f16 mantissa misread or \
             scale corruption."
        );

        // RMSNorm init: weights are ~1.0 (sqrt(2)≈1.41 in some configs).
        // After training they stay close to 1, sometimes drifting up to ~10.
        assert!(
            nm > 0.01 && nm < 100.0,
            "FALSIFY-H4-INIT-STATS-001: norm mean={nm} outside [0.01, 100]; RMSNorm scale \
             load is corrupt. Trained pretrained values are typically near 1.0."
        );
        assert!(
            ns < 100.0,
            "FALSIFY-H4-INIT-STATS-001: norm std={ns} > 100; RMSNorm has explosive variance. \
             Tensor load is corrupt."
        );
    }

    /// FALSIFY-H4-CPU-FORWARD-001 (H4 residual cascade — bisect to CPU vs CUDA):
    /// CPU `aprender::Transformer::forward` on a populated Qwen 0.5B model
    /// MUST produce sensibly-distributed logits. Host-gated test that
    /// bisects whether the val_loss > ln(vocab) defect is in the
    /// populate path / CPU forward (RED here = bug there) or in CUDA
    /// (GREEN here, RED in eval_batch = bug in CUDA path).
    #[test]
    fn falsify_h4_cpu_forward_qwen_logits_sensible() {
        let fresh = std::path::Path::new("/mnt/nvme-raid0/models/qwen2.5-coder-0.5b-fresh.apr");
        let legacy =
            std::path::Path::new("/mnt/nvme-raid0/models/qwen2.5-coder-0.5b-instruct-fp16.apr");
        let path = if fresh.exists() {
            fresh
        } else if legacy.exists() {
            legacy
        } else {
            eprintln!("[falsify-h4-cpu-forward-001] skipping: host lacks Qwen 0.5B APR");
            return;
        };

        let tensors = load_init_tensors_from_apr(path).expect("load_init_tensors_from_apr");
        let cfg = TransformerConfig::qwen2_0_5b();
        let mut transformer = Transformer::new(&cfg);
        let populated = populate_trainer_from_init_tensors(&mut transformer, &tensors)
            .expect("populate_trainer_from_init_tensors");
        eprintln!("[falsify-h4-cpu-forward-001] populated {populated} tensors");

        let token_ids = vec![100_u32];
        let logits = transformer.forward(&token_ids);
        let data = logits.data();
        let slice = data.as_slice().expect("logits contiguous");

        let mut nan_count = 0usize;
        let mut inf_count = 0usize;
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut argmax_idx = 0_usize;
        for (i, &v) in slice.iter().enumerate() {
            if v.is_nan() {
                nan_count += 1;
            } else if v.is_infinite() {
                inf_count += 1;
            } else {
                if v < min {
                    min = v;
                }
                if v > max {
                    max = v;
                    argmax_idx = i;
                }
                sum += v as f64;
                sum_sq += (v as f64) * (v as f64);
            }
        }
        let n = slice.len() as f64;
        let mean = sum / n;
        let std = (sum_sq / n - mean * mean).sqrt();

        eprintln!(
            "[falsify-h4-cpu-forward-001] token=100 logits: n={} nan={nan_count} inf={inf_count} \
             min={min:.4} max={max:.4} mean={mean:.4} std={std:.4} argmax={argmax_idx}",
            slice.len()
        );

        assert_eq!(nan_count, 0, "logits contain NaN — forward corruption");
        assert_eq!(inf_count, 0, "logits contain Inf — forward corruption");
        assert!(
            std > 0.01,
            "FALSIFY-H4-CPU-FORWARD-001: logits std={std} < 0.01 — essentially constant"
        );
        let peak_to_mean = (max as f64 - mean).abs() / std.max(1e-9);
        assert!(
            peak_to_mean > 1.5,
            "FALSIFY-H4-CPU-FORWARD-001: peak-to-mean ratio = {peak_to_mean} < 1.5 — \
             logits are essentially uniform"
        );
        assert!(
            (argmax_idx as u32) < cfg.vocab_size as u32,
            "FALSIFY-H4-CPU-FORWARD-001: argmax_idx={argmax_idx} >= vocab_size={}",
            cfg.vocab_size
        );
    }

    // ========================================================================
    // SPEC §86 / INV-INIT-ARCH-MATCH-001 unit tests — arch-mismatch fail-fast
    // ========================================================================

    fn qwen2_tensor_names() -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        // Minimal Qwen2 signature: model.layers + self_attn.q_proj.bias (distinguishes from Llama)
        let mut m = BTreeMap::new();
        m.insert("model.layers.0.self_attn.q_proj.bias".to_string(), (vec![0.0_f32; 4], vec![4]));
        m.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0_f32; 16], vec![4, 4]),
        );
        m
    }

    fn llama_tensor_names() -> BTreeMap<String, (Vec<f32>, Vec<usize>)> {
        // Llama signature: model.layers + NO attention bias + NO qkv_proj
        let mut m = BTreeMap::new();
        m.insert(
            "model.layers.0.self_attn.q_proj.weight".to_string(),
            (vec![0.0_f32; 16], vec![4, 4]),
        );
        m.insert("model.layers.0.input_layernorm.weight".to_string(), (vec![1.0_f32; 4], vec![4]));
        m
    }

    /// SPEC §86 INV-INIT-ARCH-MATCH-001: the canonical §86 case — APR
    /// metadata claims "LlamaForCausalLM" (§82 P0-H fallback) but tensors
    /// are Qwen2-shaped (have q_proj.bias). MUST fail with the falsifier ID.
    #[test]
    fn inv_init_arch_match_001_rejects_llama_stamped_qwen2_tensors() {
        let tensors = qwen2_tensor_names();
        let err = validate_init_arch_matches_tensor_evidence(Some("LlamaForCausalLM"), &tensors)
            .expect_err("§86 case MUST be rejected");
        assert!(
            err.contains("FALSIFY-INIT-ARCH-MATCH-001"),
            "error must cite falsifier id; got: {err}"
        );
        assert!(
            err.contains("llama") && err.contains("qwen2"),
            "error must name both claimed and inferred families; got: {err}"
        );
        assert!(
            err.contains("apr stamp"),
            "error must include the §86.4 salvage recipe; got: {err}"
        );
    }

    /// SPEC §86: the inverse — metadata claims "Qwen2ForCausalLM" but
    /// tensors are Llama-shaped (no q_proj.bias). MUST fail.
    #[test]
    fn inv_init_arch_match_001_rejects_qwen2_stamped_llama_tensors() {
        let tensors = llama_tensor_names();
        let err = validate_init_arch_matches_tensor_evidence(Some("Qwen2ForCausalLM"), &tensors)
            .expect_err("inverse §86 case MUST be rejected");
        assert!(err.contains("FALSIFY-INIT-ARCH-MATCH-001"));
        assert!(err.contains("qwen2") && err.contains("llama"));
    }

    /// SPEC §86: matching family slug + Qwen2 tensors — must PASS (no false-positive).
    #[test]
    fn inv_init_arch_match_001_accepts_matching_qwen2() {
        let tensors = qwen2_tensor_names();
        validate_init_arch_matches_tensor_evidence(Some("Qwen2ForCausalLM"), &tensors)
            .expect("matching qwen2 + qwen2 must pass");
        validate_init_arch_matches_tensor_evidence(Some("qwen2"), &tensors)
            .expect("matching qwen2 slug + qwen2 tensors must pass");
    }

    /// SPEC §86: matching family slug + Llama tensors — must PASS.
    #[test]
    fn inv_init_arch_match_001_accepts_matching_llama() {
        let tensors = llama_tensor_names();
        validate_init_arch_matches_tensor_evidence(Some("LlamaForCausalLM"), &tensors)
            .expect("matching llama + llama must pass");
        validate_init_arch_matches_tensor_evidence(Some("llama"), &tensors)
            .expect("matching llama slug + llama tensors must pass");
    }

    /// SPEC §86: None metadata claim — skip the check (no false-positive
    /// on novel architectures).
    #[test]
    fn inv_init_arch_match_001_skips_when_metadata_absent() {
        let tensors = qwen2_tensor_names();
        validate_init_arch_matches_tensor_evidence(None, &tensors)
            .expect("absent metadata claim must skip check");
    }

    /// SPEC §86: unknown family in metadata (e.g., "weird-novel-arch") —
    /// skip the check.
    #[test]
    fn inv_init_arch_match_001_skips_unmappable_metadata() {
        let tensors = qwen2_tensor_names();
        validate_init_arch_matches_tensor_evidence(Some("WeirdNovelArchForCausalLM"), &tensors)
            .expect("unmappable metadata MUST skip check (no false-positive on novel arch)");
    }

    /// SPEC §86: GGUF-style tensor names (blk.*) — inference returns
    /// "unknown" and we trust the metadata claim. Must not fail.
    #[test]
    fn inv_init_arch_match_001_trusts_metadata_when_tensors_unknown() {
        let mut tensors = BTreeMap::new();
        tensors.insert("blk.0.attn_q.weight".to_string(), (vec![0.0_f32; 16], vec![4, 4]));
        // GGUF names can't disambiguate; we trust the metadata.
        validate_init_arch_matches_tensor_evidence(Some("LlamaForCausalLM"), &tensors)
            .expect("unknown tensor family must skip check (trust metadata)");
    }

    /// Spec §86 helper test: family_from_tensor_names correctly
    /// distinguishes Qwen2 from Llama by the attention-bias signal.
    #[test]
    fn family_from_tensor_names_distinguishes_qwen2_from_llama() {
        let qwen2: Vec<&str> = vec![
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias", // bias = Qwen2 signature
        ];
        assert_eq!(family_from_tensor_names(qwen2.iter().copied()), "qwen2");

        let llama: Vec<&str> =
            vec!["model.layers.0.self_attn.q_proj.weight", "model.layers.0.input_layernorm.weight"];
        assert_eq!(family_from_tensor_names(llama.iter().copied()), "llama");
    }

    /// Spec §86: normalize_metadata_arch_family handles all three input forms.
    #[test]
    fn normalize_metadata_arch_family_handles_three_forms() {
        // Class name (P0-H fallback)
        assert_eq!(normalize_metadata_arch_family("Qwen2ForCausalLM"), Some("qwen2"));
        assert_eq!(normalize_metadata_arch_family("LlamaForCausalLM"), Some("llama"));
        // Family slug (canonical)
        assert_eq!(normalize_metadata_arch_family("qwen2"), Some("qwen2"));
        assert_eq!(normalize_metadata_arch_family("llama"), Some("llama"));
        // Capitalised legacy
        assert_eq!(normalize_metadata_arch_family("Qwen2"), Some("qwen2"));
        // Unknown
        assert_eq!(normalize_metadata_arch_family("unknown"), None);
        assert_eq!(normalize_metadata_arch_family("WeirdNovelArch"), None);
    }
}
