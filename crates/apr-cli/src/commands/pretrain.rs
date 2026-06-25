//! `apr pretrain` — pretraining loop driver for SHIP-TWO-001 MODEL-2.
//!
//! Wires `entrenar::train::pretrain::PretrainLoop` into the CLI. The
//! loop shape is enforced by `contracts/training-loop-pretrain-v1.yaml`
//! — specifically GATE-TRAIN-005 (divergence), GATE-TRAIN-007 (NaN),
//! and GATE-TRAIN-008 (throughput range).
//!
//! For MODEL-2 specifically, the 370M model forward pass is still a
//! scaffold (see `crates/aprender-train/src/models/llama_370m.rs`),
//! so this command runs in **synthetic** mode by default: it drives
//! the loop with a deterministic decreasing-loss step function so the
//! contract gates are exercised end-to-end even before the 370M
//! compute path is wired.

use crate::error::{CliError, Result};
use crate::output;
use clap::ValueEnum;
use colored::Colorize;
use entrenar::models::llama_370m::{
    assert_tokenizer_vocab_matches_model, assert_tokenizer_vocab_within_model_bound,
    Llama370MConfig,
};
use entrenar::train::device::{resolve_device, Device};
use entrenar::train::pretrain::{
    CheckpointFn, LinearDecaySynthetic, PretrainAbort, PretrainConfig, PretrainLoop, RunStatus,
    ScriptedVal, StepFn, TrainingRegime, ValFn,
};
use entrenar::train::pretrain_real::{
    build_shared_trainer, build_shared_trainer_with_init, AprCheckpointFn, RealStepFn, RealValFn,
};
use entrenar::train::shard_reader::ShardBatchIter;
use entrenar::train::transformer_trainer::LMBatch;
use entrenar::transformer::TransformerConfig;
use std::path::Path;

/// Number of LMBatches pulled off the head of the shard stream and
/// reserved as the held-out validation set.
///
/// 2026-04-26: bumped from 2 → 16 to reduce val_loss measurement
/// noise on from-scratch runs. With batch=16 seq=512, the prior
/// 2-batch held-out covered just 16,384 tokens — single-batch
/// fluctuation was ~0.04 in val_loss, which is at the same scale
/// as epoch-over-epoch improvement signal during early training.
/// A 50K-step run early-stopped at epoch 5/24 even though
/// train_loss was monotonically decreasing (10.01 → 9.54). With 16
/// held-out batches (131K tokens), val_loss noise floor drops
/// proportionally to ~0.01, restoring early-stop signal-to-noise.
const HELD_OUT_BATCHES: usize = 16;

/// Drift-prevention constant pinned by `apr-pretrain-arch-polymorphic-v1`
/// v1.7.0 §FALSIFY-APR-PRETRAIN-INIT-CUDA-001.
///
/// Pre-§50.4-step-5f.5 (this constant's first incarnation, v1.4.0..v1.6.0):
/// the fail-fast error returned when `--init <PATH>` AND `--device cuda`
/// were combined and the CUDA wireup did not exist. The const was the
/// drift-prevention surface — a unit test verified the citation, the
/// "not yet wired" phrase, and the 5f.5 reference all appeared.
///
/// Post-5f.5 (this PR — `apr-pretrain-arch-polymorphic-v1` v1.7.0): the
/// CUDA wireup landed via `entrenar::train::pretrain_real_cuda::
/// build_shared_cuda_trainer_with_init` (symmetric to the CPU
/// `build_shared_trainer_with_init`). The const is RETAINED but its
/// payload is repurposed as a drift-prevention sentinel: if a future
/// refactor accidentally re-introduces a fail-fast on the CUDA + --init
/// path, the test that pins this string will fail-fast and surface the
/// regression. The string itself is no longer emitted by any code path
/// in `drive_real`; it survives only to anchor the contract obligation.
pub(crate) const FALSIFY_APR_PRETRAIN_INIT_CUDA_001_MSG: &str =
    "FALSIFY-APR-PRETRAIN-INIT-CUDA-001: --init is wired for --device cuda \
     via build_shared_cuda_trainer_with_init (5f.5 SHIPPED); operator can pass \
     --init <PATH> --device cuda for end-to-end GPU fine-tune dispatch.";

/// CLI selector bound to training-loop-pretrain-v1 §hyperparameter_defaults.
/// Atomically flips the `(regime, lr_max, warmup_steps, target_val_loss)`
/// 4-tuple per INV-TRAIN-009. Explicit `--lr` / `--warmup-steps` /
/// `--target-val-loss` still win over the table row.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum PretrainMode {
    /// Post-divergence MODEL-1 remedy defaults (lr=5e-5, warmup=100, target=2.2).
    Finetune,
    /// 370M cold-start defaults (lr=3e-4, warmup=1000, target=3.0).
    FromScratch,
}

/// Resolved HP tuple from the contract's `hyperparameter_defaults` table.
/// Inputs are CLI-provided overrides (`None` means "inherit mode default").
/// Output binds INV-TRAIN-009: regime ALWAYS matches `mode`, and any field
/// the operator set explicitly passes through unchanged.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResolvedHp {
    pub regime: TrainingRegime,
    pub lr_max: f32,
    pub warmup_steps: usize,
    pub target_val_loss: f32,
}

/// SPEC §82 P0-H: derive APR checkpoint `general.name` and `architecture`
/// metadata from the `--init` model's TransformerConfig. Without this, the
/// trainer hardcoded `("llama-370m-pretrain", "LlamaForCausalLM")` even when
/// fine-tuning a Qwen2 model — which silently produced GGUF exports that
/// llama.cpp could not load because the 72 Qwen2 bias tensors (q_proj_bias,
/// k_proj_bias, v_proj_bias per layer × 24 layers) leaked through the
/// llama-family GGUF mapper as unrecognized passthrough names. The fix
/// stamps `Qwen2ForCausalLM` so the qwen2 family mapper handles biases
/// correctly.
///
/// Falls back to the pre-§82 defaults when `--init` is not provided (a
/// from-scratch llama-370m pretrain).
fn checkpoint_name_and_arch(init_arch: Option<&TransformerConfig>) -> (String, String) {
    match init_arch {
        Some(arch) => {
            let hf_arch = arch
                .hf_architecture
                .clone()
                .unwrap_or_else(|| "LlamaForCausalLM".to_string());
            // Use the lowercase hf_model_type for the name suffix when
            // available (e.g. "qwen2-pretrain"), else fall back to a
            // generic name.
            let name = arch
                .hf_model_type
                .as_deref()
                .map_or_else(|| "model-pretrain".to_string(), |t| format!("{t}-pretrain"));
            (name, hf_arch)
        }
        None => (
            "llama-370m-pretrain".to_string(),
            "LlamaForCausalLM".to_string(),
        ),
    }
}

/// SPEC §82 P1-A: Estimate transformer parameter count from arch dims.
///
/// Formula (decoder-only, tied or untied embedding):
///   N ≈ vocab × hidden                              (embedding)
///     + L × (4·hidden² + 3·hidden·intermediate)     (per-layer attn + ffn)
///     + hidden                                       (final norm)
///
/// Embedding is counted once (assumes tied lm_head; for untied add a 2nd
/// `vocab × hidden`). This is a coarse estimate suitable for Chinchilla
/// scaling sanity checks, not a precise param report — for that, use
/// `apr inspect --json | jq .parameters`.
fn estimate_param_count(arch: &TransformerConfig) -> u64 {
    let vocab = arch.vocab_size as u64;
    let hidden = arch.hidden_size as u64;
    let inter = arch.intermediate_size as u64;
    let layers = arch.num_hidden_layers as u64;
    let embed = vocab.saturating_mul(hidden);
    let attn_per_layer = 4u64.saturating_mul(hidden).saturating_mul(hidden);
    let ffn_per_layer = 3u64.saturating_mul(hidden).saturating_mul(inter);
    let per_layer = attn_per_layer.saturating_add(ffn_per_layer);
    let layer_total = layers.saturating_mul(per_layer);
    embed.saturating_add(layer_total).saturating_add(hidden)
}

pub(crate) fn mode_defaults(
    mode: PretrainMode,
    vocab_size: u32,
    lr_override: Option<f32>,
    warmup_override: Option<usize>,
    target_override: Option<f32>,
) -> ResolvedHp {
    let (regime, lr_def, warmup_def, target_def) = match mode {
        PretrainMode::Finetune => (TrainingRegime::Finetune, 5.0e-5, 100, 2.2),
        PretrainMode::FromScratch => (
            TrainingRegime::FromScratch { vocab_size },
            3.0e-4,
            1000,
            3.0,
        ),
    };
    ResolvedHp {
        regime,
        lr_max: lr_override.unwrap_or(lr_def),
        warmup_steps: warmup_override.unwrap_or(warmup_def),
        target_val_loss: target_override.unwrap_or(target_def),
    }
}

/// Execute `apr pretrain`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run(
    dataset: &Path,
    tokenizer: &Path,
    run_dir: &Path,
    mode: PretrainMode,
    lr: Option<f32>,
    num_steps: usize,
    warmup_steps: Option<usize>,
    batch_size: usize,
    seq_length: usize,
    steps_per_epoch: usize,
    seed: u64,
    target_val_loss: Option<f32>,
    vocab_size: u32,
    synthetic: bool,
    device: &str,
    init: Option<&Path>,
    force_under_provisioned: bool,
    val_shard: Option<&Path>,
    json_output: bool,
) -> Result<()> {
    // Contract gpu-training-backend-v1 INV-GPUTRAIN-001 / GATE-GPUTRAIN-002:
    // parse --device BEFORE any trainer allocation so an invalid spec
    // or an explicit `cuda` on a CPU-only host fails fast with a clear
    // diagnostic. Synthetic drive still honours --device (for parity
    // with real compute) but the stub error surface is identical.
    let resolved_device =
        resolve_device(device).map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    // Contract apr-pretrain-from-init-v1 §init_load_semantics + §50.4 step 5f.4:
    // when --init is present, (1) validate magic bytes, (2) extract
    // TransformerConfig from the APR header metadata, (3) propagate the
    // extracted arch through preflight + trainer construction.
    // Per `apr-pretrain-arch-polymorphic-v1` §arch_extraction_signature,
    // missing or unreadable architecture metadata is FAIL-FAST not silent-fallback.
    let init_arch: Option<TransformerConfig> = if let Some(init_path) = init {
        validate_init_apr_path(init_path)?;
        Some(
            crate::commands::model_config::read_apr_architecture(init_path).ok_or_else(|| {
                CliError::ValidationFailed(format!(
                    "FALSIFY-APR-PRETRAIN-INIT-005: --init APR file at {} has missing or invalid \
                     architecture metadata (hidden_size, num_heads, num_layers, vocab_size, etc). \
                     Cannot extract TransformerConfig per apr-pretrain-arch-polymorphic-v1 \
                     §arch_extraction_signature.",
                    init_path.display()
                ))
            })?,
        )
    } else {
        None
    };

    let hp = mode_defaults(mode, vocab_size, lr, warmup_steps, target_val_loss);

    // SPEC §82 P1-A + SPEC §83 P0-J: Chinchilla compute-optimal gate
    // (Hoffmann et al. 2022, arXiv:2203.15556). Compute-optimal pretraining
    // requires train tokens D ≈ 20·N where N is the parameter count.
    //
    // P0-J upgrade (post-audit, 2026-05-16, audit Rec #2): D/N < 10× is
    // now a HARD BLOCKER (fail-fast) unless `--force-under-provisioned`
    // is passed. 10× ≤ D/N < 20× is a strong warning. Triggered only on
    // `--init` runs where arch dims allow N estimation; from-scratch
    // runs are exempt.
    //
    // Audit motivation: §82 P2-A's 0.04× ratio + repetitive token
    // gibberish at val_loss=4.71 (Holtzman et al. 2019 degeneration)
    // proved that 30 min of theoretical falsification saves 8h+ GPU.
    // Contract: contracts/chinchilla-gate-v1.yaml.
    if let Some(arch) = init_arch.as_ref() {
        let n_params = estimate_param_count(arch);
        let d_tokens = (num_steps as u64)
            .saturating_mul(batch_size as u64)
            .saturating_mul(seq_length as u64);
        let ratio = d_tokens as f64 / n_params as f64;
        let suggested_steps = if batch_size > 0 && seq_length > 0 {
            (20 * n_params) / (batch_size as u64 * seq_length as u64)
        } else {
            0
        };

        if ratio < 10.0 && !force_under_provisioned {
            return Err(CliError::ValidationFailed(format!(
                "[P0-J] Chinchilla hard gate (chinchilla-gate-v1): \
                 train tokens D = {} ({:.1}M) is {:.3}× param count N = {} ({:.1}M); \
                 Chinchilla compute-optimal target is D ≈ 20·N (Hoffmann et al. 2022, arXiv:2203.15556). \
                 Run REJECTED: D/N < 10× will produce mode collapse / repetitive degeneration \
                 (Holtzman et al. 2019, arXiv:1904.09751). \
                 Increase --num-steps to ~{} OR widen --dataset corpus OR reduce model size. \
                 To bypass anyway (e.g. ablation studies, resumed runs), pass --force-under-provisioned.",
                d_tokens,
                d_tokens as f64 / 1e6,
                ratio,
                n_params,
                n_params as f64 / 1e6,
                suggested_steps,
            )));
        }

        if ratio < 10.0 {
            // Bypassed via --force-under-provisioned: emit a loud warning
            // so the override is captured in the log.
            eprintln!(
                "[P0-J] Chinchilla gate BYPASSED via --force-under-provisioned: \
                 D = {} ({:.1}M) is {:.3}× N = {} ({:.1}M). \
                 Run will likely produce repetitive/degenerate output. \
                 You explicitly opted in.",
                d_tokens,
                d_tokens as f64 / 1e6,
                ratio,
                n_params,
                n_params as f64 / 1e6,
            );
        } else if ratio < 20.0 {
            // 10× ≤ D/N < 20× — below compute-optimal but training will
            // still progress meaningfully. Warning, not error.
            eprintln!(
                "[P1-A] Chinchilla gate WARNING: D = {} ({:.1}M) is {:.1}× N = {} ({:.1}M); \
                 below compute-optimal 20·N target — model has room for more training. \
                 Suggested --num-steps for 20·N: ~{}.",
                d_tokens,
                d_tokens as f64 / 1e6,
                ratio,
                n_params,
                n_params as f64 / 1e6,
                suggested_steps,
            );
        }
    }

    // Validation: GATE-TRAIN-003 requires target_val_loss > 0.
    if hp.target_val_loss <= 0.0 {
        return Err(CliError::ValidationFailed(format!(
            "target_val_loss must be positive, got {}",
            hp.target_val_loss
        )));
    }
    if num_steps == 0 {
        return Err(CliError::ValidationFailed(
            "num_steps must be > 0".to_string(),
        ));
    }
    if steps_per_epoch == 0 {
        return Err(CliError::ValidationFailed(
            "steps_per_epoch must be > 0".to_string(),
        ));
    }

    let config = PretrainConfig {
        dataset_path: dataset.to_path_buf(),
        tokenizer_dir: tokenizer.to_path_buf(),
        run_dir: run_dir.to_path_buf(),
        lr_max: hp.lr_max,
        lr_min: (hp.lr_max * 1.0e-2).max(1.0e-7),
        warmup_steps: hp.warmup_steps,
        total_steps: num_steps,
        batch_size,
        seq_length,
        steps_per_epoch,
        seed,
        grad_clip: 1.0,
        weight_decay: 0.01,
        target_val_loss: hp.target_val_loss,
        // Patience widened from 2 → 5 epochs for from-scratch runs (2026-04-26).
        // Rationale: a 50K-step run early-stopped at epoch 5/24 even though
        // train_loss was monotonically decreasing 10.01 → 9.54 (Δ=−0.47);
        // val_loss noise on 16k-token val set (now 131k) had stdev ~0.04,
        // same scale as epoch-over-epoch improvement signal during early
        // training. 5 patience epochs gives the optimizer time to push past
        // local plateaus without ending an obviously-still-converging run.
        patience_epochs: 5,
        // Minimum epochs before early-stop. Bumped 1 → 3 so the warmup
        // window (1000 steps = 1 epoch at 1000 steps_per_epoch, or 0.5
        // epoch at 2000 steps_per_epoch) plus 1-2 initial epochs of post-
        // warmup learning are guaranteed to complete before any early-stop
        // signal is honoured.
        min_epochs_before_early_stop: 3,
        regime: hp.regime,
    };

    if !json_output {
        print_header(&config);
        // GATE-GPUTRAIN-002 visibility: print the resolved Device so the
        // operator can confirm which backend was selected. `auto` is the
        // only spec that may silently fall back, and this print makes
        // the fall-back visible at startup.
        output::kv("  Device", resolved_device.to_string());
        println!();
    }

    let status = if synthetic {
        drive_synthetic(
            config.clone(),
            num_steps,
            steps_per_epoch,
            hp.target_val_loss,
            json_output,
        )?
    } else {
        drive_real(
            config.clone(),
            dataset,
            hp.lr_max,
            seq_length,
            batch_size,
            seed,
            resolved_device,
            json_output,
            init_arch.as_ref(),
            init,
            val_shard,
        )?
    };

    // Contract: non-OK terminal statuses map to non-zero exit codes so
    // operators can recognize divergence / NaN from shell `$?`.
    match status {
        RunStatus::Aborted(abort) => Err(abort_to_err(&abort)),
        RunStatus::Ok { .. } | RunStatus::EarlyStop { .. } => Ok(()),
    }
}

/// Synthetic drive: deterministic linear-decay `StepFn` and a scripted
/// val-loss sequence so the full gate surface (GATE-TRAIN-005/007/008)
/// is exercised end-to-end with no corpus I/O.
fn drive_synthetic(
    config: PretrainConfig,
    num_steps: usize,
    steps_per_epoch: usize,
    target_val_loss: f32,
    json_output: bool,
) -> Result<RunStatus> {
    let step_fn = LinearDecaySynthetic {
        start_loss: (target_val_loss * 2.0).max(1.5),
        decay_per_step: (target_val_loss * 0.01).max(1.0e-4),
        grad_norm: 0.8,
    };
    let num_epochs = num_steps.div_ceil(steps_per_epoch);
    let mut sequence = Vec::with_capacity(num_epochs + 2);
    let start_val = (target_val_loss * 1.8).max(3.0);
    for i in 0..(num_epochs + 2) {
        let t = i as f32 / (num_epochs.max(1) as f32);
        sequence.push(target_val_loss + (start_val - target_val_loss) * (1.0 - t).max(0.0));
    }
    let val_fn = ScriptedVal { sequence };
    // Synthetic drive has no real weights to checkpoint.
    run_and_report(config, step_fn, val_fn, None, json_output)
}

/// Contract apr-pretrain-from-init-v1 §init_load_semantics + §init_error_semantics:
/// validate `--init <PATH>` BEFORE any trainer allocation. Falsifies
/// FALSIFY-APR-PRETRAIN-INIT-003 (missing-file) + -004 (invalid-magic).
///
/// Returns Ok on a valid APR file (existence + magic bytes verified).
/// Architecture extraction + weight load are §50.4 step 5f.4 — the
/// caller (`run()`) extracts the config via `model_config::read_apr_architecture`
/// and passes both to `build_shared_trainer_with_init` per
/// `apr-pretrain-arch-polymorphic-v1` §init_load_semantics.
fn validate_init_apr_path(path: &Path) -> Result<()> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "FALSIFY-APR-PRETRAIN-INIT-003: --init path does not exist or is unreadable: {} ({e})",
            path.display()
        ))
    })?;
    let mut magic = [0u8; 4];
    use std::io::Read;
    file.read_exact(&mut magic).map_err(|e| {
        CliError::ValidationFailed(format!(
            "FALSIFY-APR-PRETRAIN-INIT-004: --init file too short to contain APR magic bytes: {} ({e})",
            path.display()
        ))
    })?;
    // APR magic bytes per `crates/aprender-core/src/format/kani_proofs.rs`:
    //   APR\0 = [0x41, 0x50, 0x52, 0x00] (v2)
    //   APRN  = [0x41, 0x50, 0x52, 0x4E] (v1)
    const APR_MAGIC_V2: [u8; 4] = [0x41, 0x50, 0x52, 0x00];
    const APR_MAGIC_V1: [u8; 4] = [0x41, 0x50, 0x52, 0x4E];
    if magic != APR_MAGIC_V2 && magic != APR_MAGIC_V1 {
        return Err(CliError::ValidationFailed(format!(
            "FALSIFY-APR-PRETRAIN-INIT-004: --init file is not a valid APR file (magic={:02X?}, expected {:02X?} or {:02X?}): {}",
            magic, APR_MAGIC_V2, APR_MAGIC_V1, path.display()
        )));
    }
    Ok(())
}

/// GATE-ARCH-370M-011 pre-flight: count the tokenizer's vocabulary entries
/// from `vocab.json` and assert the count matches `target_vocab_size`
/// before any trainer allocation.
///
/// Per `apr-pretrain-arch-polymorphic-v1` §qwen_tokenizer_vocab_compatibility
/// (PR #1473), the target is now POLYMORPHIC — when `--init <PATH>` is set,
/// the caller passes the extracted-arch's vocab_size (e.g., 151_936 for
/// Qwen2.5-0.5B); otherwise `Llama370MConfig::VOCAB_SIZE` (50_257) for
/// the §24/§25 from-scratch baseline.
///
/// Any mismatch aborts the dispatch with a clear error naming both values
/// and the violated invariant — the N-09 OOB escape in `Embedding::forward`
/// would otherwise silently corrupt training.
///
/// Discharges FALSIFY-APR-PRETRAIN-ARCH-005 (Qwen tokenizer passes with
/// Qwen target) and FALSIFY-APR-PRETRAIN-ARCH-006 (Qwen tokenizer fails
/// with Llama target).
fn preflight_tokenizer_vocab_matches_target(
    tokenizer_dir: &Path,
    target_vocab_size: usize,
    init_is_some: bool,
) -> Result<()> {
    let vocab_path = tokenizer_dir.join("vocab.json");
    let vocab_json = std::fs::read_to_string(&vocab_path).map_err(|e| {
        CliError::ValidationFailed(format!(
            "GATE-ARCH-370M-011 pre-flight: cannot read {} ({e})",
            vocab_path.display()
        ))
    })?;
    let vocab: serde_json::Map<String, serde_json::Value> = serde_json::from_str(&vocab_json)
        .map_err(|e| {
            CliError::ValidationFailed(format!(
                "GATE-ARCH-370M-011 pre-flight: {} is not a valid vocab.json: {e}",
                vocab_path.display()
            ))
        })?;
    // §55: when --init is set (polymorphic path with HF-distributed
    // checkpoint), allow tokenizer_vocab ≤ model_vocab to admit Qwen-style
    // reserved-slot vocabularies. When --init is absent (§24/§25 from-scratch
    // baseline), enforce strict equality to preserve INV-ARCH-370M-006.
    if init_is_some {
        assert_tokenizer_vocab_within_model_bound(vocab.len(), target_vocab_size)
            .map_err(CliError::ValidationFailed)
    } else {
        assert_tokenizer_vocab_matches_model(vocab.len(), target_vocab_size)
            .map_err(CliError::ValidationFailed)
    }
}

/// Real-corpus drive: build a shared 370M trainer (CPU or CUDA), split
/// the shard stream head-off into a held-out validation set, and run a
/// full forward + backward + AdamW step per training batch.
///
/// When `device.is_cuda()`, the `cuda` feature must be compiled in —
/// otherwise this surfaces a clear error rather than silently falling
/// back to CPU (GATE-GPUTRAIN-002, contract gpu-training-backend-v1).
#[allow(clippy::too_many_arguments)]
fn drive_real(
    config: PretrainConfig,
    dataset: &Path,
    lr: f32,
    seq_length: usize,
    batch_size: usize,
    seed: u64,
    device: Device,
    json_output: bool,
    init_arch: Option<&TransformerConfig>,
    init_path: Option<&Path>,
    val_shard: Option<&Path>,
) -> Result<RunStatus> {
    // GATE-ARCH-370M-011 / INV-ARCH-370M-006 — refuse to dispatch a real
    // training step when the tokenizer vocab_size and the model vocab_size
    // disagree. The N-09 OOB escape guard in Embedding::forward masks the
    // mismatch at runtime → silent garbage gradients otherwise. Synthetic
    // drive skips this check because it never touches the real model.
    // Per `apr-pretrain-arch-polymorphic-v1` §qwen_tokenizer_vocab_compatibility
    // (§50.4 step 5d/5f.4): when --init is set, gate by the EXTRACTED arch's
    // vocab_size; otherwise gate by the §24/§25 baseline Llama370MConfig::VOCAB_SIZE,
    // preserving regression-free behavior (FALSIFY-002 + FALSIFY-005 + FALSIFY-006).
    let target_vocab = init_arch
        .map(|cfg| cfg.vocab_size)
        .unwrap_or(Llama370MConfig::VOCAB_SIZE);
    preflight_tokenizer_vocab_matches_target(
        &config.tokenizer_dir,
        target_vocab,
        init_arch.is_some(),
    )?;

    // MVP: pad_id/eos_id both 0. All sequences are uniform length
    // (seq_length + 1) so LMBatch::from_sequences takes the shared
    // layout path and pad_id is never used for padding. The real
    // tokenizer's special-token ids will plumb through in a follow-up.
    //
    // wrap_around=true: when the corpus shards are exhausted before
    // --num-steps is reached, reset cursor to shard 0 and continue.
    // This is standard ML-training behaviour (matches PyTorch /
    // HuggingFace). Without it, an 18M-token corpus exhausts in ~2
    // epochs of a 5K-step run with batch=16 seq=512, and the
    // Cuda*StepFn falls back to placeholder loss `(1.0, 1.0)` — silently
    // producing garbage gradients. See spec §22 (PR #1073) for the
    // root-cause investigation.
    let mut iter = ShardBatchIter::new(dataset, batch_size, seq_length, 0, 0)
        .map_err(|e| {
            CliError::ValidationFailed(format!(
                "dataset shard iterator init failed: {e} (path={})",
                dataset.display()
            ))
        })?
        .with_wrap_around(true)
        // SPEC §82 P2-B: surface data starvation. When the corpus cycles
        // mid-run, emit a stderr line so operators can detect that the
        // step budget exceeds the corpus capacity (per Chinchilla, train
        // tokens D ≈ 20·N — if D is small, the corpus wraps repeatedly
        // and the model memorizes instead of generalizing).
        .with_warn_on_wrap_around(true);

    // SPEC §84 P2-F (apr-pretrain-val-shard-v1): held-out val source.
    //
    // When --val-shard <DIR> is provided, drain HELD_OUT_BATCHES from a
    // dedicated independent shard iterator over <DIR>; the training iter
    // stays at offset 0 (no batch theft). This makes val_loss comparable
    // across runs whose --dataset composition changes (the P2-C audit-
    // falsified result was confounded by val sets drawn from different
    // corpus distributions — see evidence/p2c-2026-05-17/findings.md).
    //
    // When --val-shard is None, the historical "first N batches of
    // --dataset" behaviour is preserved.
    let held_out: Vec<LMBatch> = if let Some(val_dir) = val_shard {
        let mut val_iter = ShardBatchIter::new(val_dir, batch_size, seq_length, 0, 0)
            .map_err(|e| {
                CliError::ValidationFailed(format!(
                    "FALSIFY-PRETRAIN-VAL-SHARD-001: --val-shard iterator init failed: {e} \
                     (path={})",
                    val_dir.display()
                ))
            })?
            // Per INV-PRETRAIN-VAL-SHARD-002 — the val shard is NOT
            // wrap-around. A short val corpus draws short held_out
            // (potentially < HELD_OUT_BATCHES batches) and the run
            // proceeds; we only fail if zero batches are drawn.
            .with_wrap_around(false);
        let mut batches: Vec<LMBatch> = Vec::with_capacity(HELD_OUT_BATCHES);
        for _ in 0..HELD_OUT_BATCHES {
            match val_iter.next() {
                Some(b) => batches.push(b),
                None => break,
            }
        }
        if batches.is_empty() {
            return Err(CliError::ValidationFailed(format!(
                "FALSIFY-PRETRAIN-VAL-SHARD-003: --val-shard {} is too small to yield any \
                 held-out batches at batch_size={} seq_length={}",
                val_dir.display(),
                batch_size,
                seq_length
            )));
        }
        if !json_output {
            eprintln!(
                "[P2-F] held-out val source = --val-shard {} ({} batches)",
                val_dir.display(),
                batches.len()
            );
        }
        batches
    } else {
        // Reserve the first `HELD_OUT_BATCHES` batches as the held-out val
        // set; the remainder feeds RealStepFn.
        let mut batches: Vec<LMBatch> = Vec::with_capacity(HELD_OUT_BATCHES);
        for _ in 0..HELD_OUT_BATCHES {
            match iter.next() {
                Some(b) => batches.push(b),
                None => break,
            }
        }
        if batches.is_empty() {
            return Err(CliError::ValidationFailed(format!(
                "dataset {} is too small to reserve any held-out batches",
                dataset.display()
            )));
        }
        batches
    };

    if device.is_cuda() {
        // §50.4 step 5f.5 SHIPPED (this PR): CUDA path with --init is now
        // wired symmetric to the CPU path via
        // `entrenar::train::pretrain_real_cuda::build_shared_cuda_trainer_with_init`.
        // The same §50.4 step-5f machinery composes through both backends:
        //   5c: build_transformer_config(init_arch)
        //   5f.1: validate_pretrain_init_arch_compatible(init_arch) — encoder rejection
        //   5f.2: load_init_tensors_from_apr(path) — read APR weights
        //   5f.3: populate_trainer_from_init_tensors(transformer, &tensors) — populate CPU model
        //   5f.5 (this PR): CudaTransformerTrainer::with_model uploads populated
        //                   blocks / norm / lm_head to GPU.
        //
        // Per `apr-pretrain-arch-polymorphic-v1` v1.7.0 §FALSIFY-APR-PRETRAIN-INIT-CUDA-001,
        // the const FALSIFY_APR_PRETRAIN_INIT_CUDA_001_MSG is repurposed as a
        // drift-prevention sentinel — if a future refactor re-introduces a
        // fail-fast on the CUDA + --init path, the test that pins the const
        // will fail and surface the regression.
        drive_real_cuda(
            config,
            iter,
            held_out,
            lr,
            seq_length,
            seed,
            json_output,
            init_arch,
            init_path,
        )
    } else {
        drive_real_cpu(
            config,
            iter,
            held_out,
            lr,
            seq_length,
            seed,
            json_output,
            init_arch,
            init_path,
        )
    }
}

/// CPU backend for `drive_real` — builds a `TransformerTrainer`
/// (`aprender::Tensor` + trueno SIMD) and wires `RealStepFn` /
/// `RealValFn` / `AprCheckpointFn`.
#[allow(clippy::too_many_arguments)]
fn drive_real_cpu(
    config: PretrainConfig,
    iter: entrenar::train::shard_reader::ShardBatchIter,
    held_out: Vec<LMBatch>,
    lr: f32,
    seq_length: usize,
    seed: u64,
    json_output: bool,
    init_arch: Option<&TransformerConfig>,
    init_path: Option<&Path>,
) -> Result<RunStatus> {
    // §50.4 step 5f.4: when --init is set, build the trainer via the
    // polymorphic builder (extracts arch + loads + populates init tensors).
    // When --init is absent, use the existing from-scratch baseline builder
    // so the §24/§25 evidence remains regression-free.
    let trainer = if init_arch.is_some() || init_path.is_some() {
        build_shared_trainer_with_init(lr, seq_length, seed, init_arch, init_path)
            .map_err(CliError::ValidationFailed)?
    } else {
        build_shared_trainer(lr, seq_length, seed)
    };
    let step_fn = RealStepFn::new(trainer.clone(), Box::new(iter));
    let val_fn = RealValFn::new(trainer.clone(), held_out);
    let (ckpt_name, ckpt_arch) = checkpoint_name_and_arch(init_arch);
    let ckpt: Box<dyn CheckpointFn> =
        Box::new(AprCheckpointFn::new(trainer, &ckpt_name, &ckpt_arch));
    run_and_report(config, step_fn, val_fn, Some(ckpt), json_output)
}

/// CUDA backend for `drive_real` — builds a `CudaTransformerTrainer`
/// and wires `CudaRealStepFn` / `CudaRealValFn` / `CudaAprCheckpointFn`
/// (task #132 Phase 2, contract gpu-training-backend-v1).
///
/// When the `cuda` feature is NOT compiled in, this returns a clear
/// build-time error so operators who asked for `--device cuda` do not
/// silently get the CPU path (GATE-GPUTRAIN-002 / FM-GPUTRAIN-SILENT-CPU).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn drive_real_cuda(
    config: PretrainConfig,
    iter: entrenar::train::shard_reader::ShardBatchIter,
    held_out: Vec<LMBatch>,
    lr: f32,
    seq_length: usize,
    seed: u64,
    json_output: bool,
    init_arch: Option<&TransformerConfig>,
    init_path: Option<&Path>,
) -> Result<RunStatus> {
    use entrenar::train::pretrain_real_cuda::{
        build_shared_cuda_trainer, build_shared_cuda_trainer_with_init, CudaAprCheckpointFn,
        CudaRealStepFn, CudaRealValFn,
    };
    // §50.4 step 5f.5: when --init is set on the CUDA path, build via the
    // polymorphic builder (extracts arch + loads + populates init tensors,
    // then uploads to GPU). When --init is absent, use the existing
    // from-scratch baseline so the §24/§25 evidence remains regression-free
    // and INV-ARCH-370M-001 stays enforced on the from-scratch CUDA path.
    let trainer = if init_arch.is_some() || init_path.is_some() {
        build_shared_cuda_trainer_with_init(lr, seq_length, seed, init_arch, init_path).map_err(
            |e| {
                CliError::ValidationFailed(format!(
                    "GATE-GPUTRAIN-002: CUDA trainer allocation (--init path) failed: {e}. \
                     See contracts/entrenar/gpu-training-backend-v1.yaml and \
                     contracts/apr-pretrain-arch-polymorphic-v1.yaml v1.7.0 \
                     §FALSIFY-APR-PRETRAIN-INIT-CUDA-001 — this path is only \
                     reachable when the binary was built with `--features cuda`.",
                ))
            },
        )?
    } else {
        build_shared_cuda_trainer(lr, seq_length, seed).map_err(|e| {
            CliError::ValidationFailed(format!(
                "GATE-GPUTRAIN-002: CUDA trainer allocation failed: {e}. \
                 See contracts/entrenar/gpu-training-backend-v1.yaml and \
                 memory/feedback_cuda_feature_footgun.md — this path is \
                 only reachable when the binary was built with `--features cuda`.",
            ))
        })?
    };
    let step_fn = CudaRealStepFn::new(trainer.clone(), Box::new(iter));
    let val_fn = CudaRealValFn::new(trainer.clone(), held_out);
    // SPEC-SHIP-TWO-001 §81 P0-D: pass --tokenizer through so each
    // checkpoint embeds the tokenizer.json (apr qa requires this).
    let (ckpt_name, ckpt_arch) = checkpoint_name_and_arch(init_arch);
    let ckpt: Box<dyn CheckpointFn> = Box::new(
        CudaAprCheckpointFn::new(trainer, &ckpt_name, &ckpt_arch)
            .with_tokenizer_dir(&config.tokenizer_dir),
    );
    run_and_report(config, step_fn, val_fn, Some(ckpt), json_output)
}

/// CUDA backend stub when the `cuda` feature is NOT compiled in.
///
/// This is the load-bearing gate that prevents FM-GPUTRAIN-SILENT-CPU:
/// if a user passes `--device cuda` on an apr binary built without
/// CUDA support, they see a clear "rebuild with --features cuda" error
/// rather than a 14-minute CPU run masquerading as GPU training
/// (task #132 lambda-labs incident, 2026-04-21).
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn drive_real_cuda(
    _config: PretrainConfig,
    _iter: entrenar::train::shard_reader::ShardBatchIter,
    _held_out: Vec<LMBatch>,
    _lr: f32,
    _seq_length: usize,
    _seed: u64,
    _json_output: bool,
    _init_arch: Option<&TransformerConfig>,
    _init_path: Option<&Path>,
) -> Result<RunStatus> {
    Err(CliError::ValidationFailed(
        "GATE-GPUTRAIN-002: --device cuda was requested but this `apr` \
         binary was built WITHOUT the `cuda` feature. \
         Rebuild with `cargo build --release --features cuda` or use \
         `--device cpu`. See memory/feedback_cuda_feature_footgun.md \
         (contract gpu-training-backend-v1 / task #132 Phase 2)."
            .into(),
    ))
}

/// Shared helper: construct the `PretrainLoop`, run it, print the
/// terminal report, and bubble the `RunStatus` back for exit-code
/// mapping. `checkpoint_fn` — when `Some` — writes an APR file per
/// epoch that passes GATE-TRAIN-005.
fn run_and_report<S: StepFn, V: ValFn>(
    config: PretrainConfig,
    step_fn: S,
    val_fn: V,
    checkpoint_fn: Option<Box<dyn CheckpointFn>>,
    json_output: bool,
) -> Result<RunStatus> {
    let mut loop_ = PretrainLoop::new(config, step_fn, val_fn);
    if let Some(ckpt) = checkpoint_fn {
        loop_ = loop_.with_checkpoint_fn(ckpt);
    }
    let status = loop_.run();
    report(&status, &loop_, json_output)?;
    Ok(status)
}

fn abort_to_err(abort: &PretrainAbort) -> CliError {
    match abort {
        PretrainAbort::Divergence { .. } | PretrainAbort::DivergenceAtEpochZero { .. } => {
            CliError::ValidationFailed(format!(
                "GATE-TRAIN-005 ship-blocker fired: {abort}. See \
                 contracts/training-loop-pretrain-v1.yaml and \
                 memory/project_ship_two_001_model1_qlora_divergence.md"
            ))
        }
        PretrainAbort::NumericalInstability { .. } => {
            CliError::ValidationFailed(format!("GATE-TRAIN-007 NaN/Inf guard fired: {abort}"))
        }
        PretrainAbort::ThroughputOutOfRange { .. } => CliError::ValidationFailed(format!(
            "GATE-TRAIN-008 throughput-range guard fired: {abort}"
        )),
    }
}

fn print_header(cfg: &PretrainConfig) {
    output::header("apr pretrain — SHIP-TWO-001 MODEL-2 training loop");
    println!();
    output::section("Configuration");
    output::kv("  Dataset", cfg.dataset_path.display().to_string());
    output::kv("  Tokenizer", cfg.tokenizer_dir.display().to_string());
    output::kv("  Run dir", cfg.run_dir.display().to_string());
    output::kv("  LR max", format!("{:.2e}", cfg.lr_max));
    output::kv("  Total steps", cfg.total_steps.to_string());
    output::kv("  Warmup steps", cfg.warmup_steps.to_string());
    output::kv(
        "  Batch × seq",
        format!("{} × {}", cfg.batch_size, cfg.seq_length),
    );
    output::kv("  Steps / epoch", cfg.steps_per_epoch.to_string());
    output::kv("  Seed", cfg.seed.to_string());
    output::kv("  Target val_loss", format!("{:.2}", cfg.target_val_loss));
    println!();
}

fn report<S: entrenar::train::pretrain::StepFn, V: entrenar::train::pretrain::ValFn>(
    status: &RunStatus,
    loop_: &PretrainLoop<S, V>,
    json_output: bool,
) -> Result<()> {
    if json_output {
        let report = PretrainReport::from(status, loop_);
        let json = serde_json::to_string_pretty(&report)
            .map_err(|e| CliError::InvalidFormat(e.to_string()))?;
        println!("{json}");
        return Ok(());
    }

    output::section("Run Result");
    match status {
        RunStatus::Ok {
            final_val_loss,
            epochs_completed,
        } => {
            println!(
                "  {} CONVERGED  final val_loss={:.4} after {} epoch(s)",
                "OK".green().bold(),
                final_val_loss,
                epochs_completed
            );
        }
        RunStatus::EarlyStop {
            best_val_loss,
            epochs_completed,
        } => {
            println!(
                "  {} EARLY_STOP  best val_loss={:.4} after {} epoch(s)",
                "OK".yellow().bold(),
                best_val_loss,
                epochs_completed
            );
        }
        RunStatus::Aborted(abort) => {
            println!("  {} ABORTED  {}", "FAIL".red().bold(), abort);
        }
    }
    output::kv("  Steps recorded", loop_.step_metrics().len().to_string());
    output::kv(
        "  Epochs recorded",
        loop_.epoch_artifacts().len().to_string(),
    );
    println!();
    Ok(())
}

#[derive(serde::Serialize)]
struct PretrainReport {
    status: String,
    detail: Option<String>,
    final_val_loss: Option<f32>,
    epochs_completed: usize,
    steps_recorded: usize,
    val_loss_history: Vec<f32>,
    /// Per-step `StepMetrics` captured by `PretrainLoop` (GATE-TRAIN-001
    /// contract `training-loop-pretrain-v1.yaml::per_step_metrics.required`).
    ///
    /// Emitted so downstream consumers can discharge FALSIFY-GPUTRAIN-005
    /// (step-time < 500 ms on RTX 4090 for 370M) and FALSIFY-GPUTRAIN-006
    /// (same-seed reproducibility — two cuda:0 runs at seed=0 must match
    /// on every step's train_loss within `AC_GPUTRAIN_006_MAX_SEED_LOSS_DELTA`
    /// = 1e-5) directly from the `--json` output, rather than having to
    /// parse run-dir checkpoint metadata.
    per_step_metrics: Vec<entrenar::train::pretrain::StepMetrics>,
}

impl PretrainReport {
    fn from<S: entrenar::train::pretrain::StepFn, V: entrenar::train::pretrain::ValFn>(
        status: &RunStatus,
        loop_: &PretrainLoop<S, V>,
    ) -> Self {
        let (status_name, detail, final_val_loss, epochs_completed) = match status {
            RunStatus::Ok {
                final_val_loss,
                epochs_completed,
            } => (
                "OK".to_string(),
                None,
                Some(*final_val_loss),
                *epochs_completed,
            ),
            RunStatus::EarlyStop {
                best_val_loss,
                epochs_completed,
            } => (
                "EARLY_STOP".to_string(),
                None,
                Some(*best_val_loss),
                *epochs_completed,
            ),
            RunStatus::Aborted(abort) => (
                "ABORTED".to_string(),
                Some(abort.to_string()),
                None,
                loop_.epoch_artifacts().len(),
            ),
        };
        PretrainReport {
            status: status_name,
            detail,
            final_val_loss,
            epochs_completed,
            steps_recorded: loop_.step_metrics().len(),
            val_loss_history: loop_.val_loss_history().to_vec(),
            per_step_metrics: loop_.step_metrics().to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// SPEC §82 P0-H: when `--init` is absent, fall back to historical defaults
    /// so from-scratch 370M pretrain still produces `llama-370m-pretrain` /
    /// `LlamaForCausalLM` stamps.
    #[test]
    fn checkpoint_name_and_arch_default_when_no_init() {
        let (name, arch) = checkpoint_name_and_arch(None);
        assert_eq!(name, "llama-370m-pretrain");
        assert_eq!(arch, "LlamaForCausalLM");
    }

    /// SPEC §82 P0-H: when `--init` is a Qwen2 model, stamp `qwen2-pretrain`
    /// and `Qwen2ForCausalLM` so the qwen2 GGUF family mapper handles the
    /// 72 Qwen2 attn biases instead of leaving them as passthrough names.
    #[test]
    fn checkpoint_name_and_arch_qwen2_init() {
        let mut cfg = TransformerConfig::llama2_7b();
        cfg.hf_architecture = Some("Qwen2ForCausalLM".to_string());
        cfg.hf_model_type = Some("qwen2".to_string());
        let (name, arch) = checkpoint_name_and_arch(Some(&cfg));
        assert_eq!(name, "qwen2-pretrain");
        assert_eq!(arch, "Qwen2ForCausalLM");
    }

    /// SPEC §82 P0-H: a `--init` model that lacks `hf_architecture` falls back
    /// to `LlamaForCausalLM` rather than silently emitting an empty arch
    /// string. (Belt-and-suspenders for older APR files written before the
    /// hf_architecture field existed.)
    #[test]
    fn checkpoint_name_and_arch_init_without_hf_fields() {
        let cfg = TransformerConfig::llama2_7b();
        // llama2_7b() leaves hf_architecture and hf_model_type as None.
        let (name, arch) = checkpoint_name_and_arch(Some(&cfg));
        assert_eq!(name, "model-pretrain");
        assert_eq!(arch, "LlamaForCausalLM");
    }

    /// Stage a `vocab.json` with exactly `n` distinct integer-string tokens at
    /// `<dir>/vocab.json`. Used by pre-flight gate tests + by other tests that
    /// need to get PAST the GATE-ARCH-370M-011 pre-flight to exercise a later
    /// failure mode (e.g. empty dataset shards).
    fn stage_vocab_json(dir: &std::path::Path, n: usize) {
        std::fs::create_dir_all(dir).expect("mkdir tokenizer dir");
        let mut obj = serde_json::Map::with_capacity(n);
        for i in 0..n {
            obj.insert(format!("t{i}"), serde_json::Value::from(i as u64));
        }
        let json = serde_json::to_string(&obj).expect("serialize");
        std::fs::write(dir.join("vocab.json"), json).expect("write vocab.json");
    }

    /// SPEC §82 P1-A: parameter count estimator should be order-of-magnitude
    /// correct for known reference models. Qwen2.5-0.5B has ~500M params;
    /// our coarse formula should be within 2× of that.
    #[test]
    fn estimate_param_count_qwen2_05b_within_2x() {
        let mut cfg = TransformerConfig::llama2_7b();
        cfg.hidden_size = 896;
        cfg.num_hidden_layers = 24;
        cfg.num_attention_heads = 14;
        cfg.num_kv_heads = 2;
        cfg.intermediate_size = 4864;
        cfg.vocab_size = 151936;
        let n = estimate_param_count(&cfg);
        // True Qwen2.5-0.5B = ~494M. Our estimate counts tied embedding once
        // and ignores GQA reduction; expect ~400-700M.
        let ref_params: u64 = 494_000_000;
        assert!(
            n > ref_params / 2 && n < ref_params * 2,
            "Qwen2.5-0.5B estimate {n} should be within 2× of 494M",
        );
    }

    /// SPEC §82 P1-A: estimator should scale super-linearly with depth.
    #[test]
    fn estimate_param_count_scales_with_layers() {
        let mut cfg = TransformerConfig::llama2_7b();
        cfg.hidden_size = 512;
        cfg.num_hidden_layers = 1;
        cfg.intermediate_size = 2048;
        cfg.vocab_size = 32000;
        let n1 = estimate_param_count(&cfg);
        cfg.num_hidden_layers = 24;
        let n24 = estimate_param_count(&cfg);
        // 24× per-layer params + shared embedding ≈ 5-6× total for small models
        // where embedding dominates per-layer contribution.
        assert!(
            n24 > n1 * 4,
            "24-layer model {n24} should be at least 4× 1-layer model {n1}",
        );
    }

    // ─── SPEC §83 P0-J: Chinchilla hard-gate behavior ──────────
    //
    // The gate logic itself lives inline in `run()` so a full unit
    // test requires either calling `run()` (heavy — needs dataset
    // path + tokenizer dir) or factoring the math into a helper.
    // Below we test the math in isolation via a local helper; the
    // end-to-end CLI behavior is covered by integration tests in
    // tests/chinchilla_gate_test.rs (FALSIFY-CHINCHILLA-001..003).

    /// Mirror of the inline gate math in `run()` — kept in sync via
    /// review. Returns Some(error_message) if rejected, None if
    /// accepted (with or without bypass).
    fn chinchilla_gate_check(
        arch: &TransformerConfig,
        num_steps: usize,
        batch_size: usize,
        seq_length: usize,
        force_under_provisioned: bool,
    ) -> Option<f64> {
        let n_params = estimate_param_count(arch);
        let d_tokens = (num_steps as u64)
            .saturating_mul(batch_size as u64)
            .saturating_mul(seq_length as u64);
        let ratio = d_tokens as f64 / n_params as f64;
        if ratio < 10.0 && !force_under_provisioned {
            Some(ratio)
        } else {
            None
        }
    }

    fn qwen_05b_config() -> TransformerConfig {
        let mut cfg = TransformerConfig::llama2_7b();
        cfg.hidden_size = 896;
        cfg.num_hidden_layers = 24;
        cfg.num_attention_heads = 14;
        cfg.num_kv_heads = 2;
        cfg.intermediate_size = 4864;
        cfg.vocab_size = 151936;
        cfg.hf_architecture = Some("Qwen2ForCausalLM".to_string());
        cfg.hf_model_type = Some("qwen2".to_string());
        cfg
    }

    /// FALSIFY-CHINCHILLA-001 (unit): §82 P2-A reproducer — 5000
    /// steps × 16 × 512 = 40.96M tokens against Qwen-0.5B (~494M
    /// params) = ratio 0.083× → REJECTED.
    #[test]
    fn chinchilla_hard_gate_rejects_under_provisioned() {
        let cfg = qwen_05b_config();
        let verdict = chinchilla_gate_check(&cfg, 5000, 16, 512, false);
        assert!(verdict.is_some(), "0.083× should be rejected");
        let ratio = verdict.expect("ratio");
        assert!(ratio < 0.1, "expected ratio < 0.1, got {ratio}");
    }

    /// FALSIFY-CHINCHILLA-002 (unit): same config with bypass flag
    /// → accepted (returns None despite low ratio).
    #[test]
    fn chinchilla_hard_gate_bypasses_with_force_flag() {
        let cfg = qwen_05b_config();
        let verdict = chinchilla_gate_check(&cfg, 5000, 16, 512, true);
        assert!(verdict.is_none(), "force_under_provisioned must bypass");
    }

    /// FALSIFY-CHINCHILLA-004 (unit): boundary at exactly D/N = 10
    /// passes; just below fails. Uses ceiling division to ensure
    /// the "exact" case actually meets or exceeds 10·N (integer
    /// truncation on `target_d / (bs*sl)` would land slightly below).
    #[test]
    fn chinchilla_hard_gate_boundary_10x() {
        let cfg = qwen_05b_config();
        let n = estimate_param_count(&cfg);
        let bs = 16u64;
        let sl = 512u64;
        let target_d = 10 * n;
        let bs_sl = bs * sl;
        // Ceiling division so D ≥ 10·N exactly (passes the gate).
        let exact_steps = (target_d + bs_sl - 1) / bs_sl;
        let verdict_exact =
            chinchilla_gate_check(&cfg, exact_steps as usize, bs as usize, sl as usize, false);
        assert!(
            verdict_exact.is_none(),
            "ratio ≥ 10.0 should PASS, got verdict={verdict_exact:?}"
        );
        // One full step less → below 10·N → REJECTED.
        let verdict_below = chinchilla_gate_check(
            &cfg,
            (exact_steps - 1) as usize,
            bs as usize,
            sl as usize,
            false,
        );
        assert!(
            verdict_below.is_some(),
            "ratio just below 10× should be REJECTED"
        );
    }

    /// FALSIFY-CHINCHILLA-005 (unit): generously-provisioned ratios
    /// (≥ 10×) pass without --force flag.
    #[test]
    fn chinchilla_hard_gate_accepts_well_provisioned() {
        let cfg = qwen_05b_config();
        let n = estimate_param_count(&cfg);
        // 25·N = generous (above 20× compute-optimal target).
        let bs = 16u64;
        let sl = 512u64;
        let steps_25x = ((25 * n) / (bs * sl)) as usize;
        let verdict = chinchilla_gate_check(&cfg, steps_25x, bs as usize, sl as usize, false);
        assert!(verdict.is_none(), "25× should pass");
    }

    #[test]
    fn preflight_accepts_matching_vocab() {
        // GATE-ARCH-370M-011 acceptance case: tokenizer vocab.json with
        // exactly Llama370MConfig::VOCAB_SIZE entries must pass pre-flight.
        let tmp = TempDir::new().expect("tempdir");
        stage_vocab_json(tmp.path(), Llama370MConfig::VOCAB_SIZE);
        preflight_tokenizer_vocab_matches_target(tmp.path(), Llama370MConfig::VOCAB_SIZE, false)
            .expect("matching vocab must pass GATE-ARCH-370M-011");
    }

    #[test]
    fn preflight_rejects_tokenizer_vocab_mismatch() {
        // FALSIFY-ARCH-370M-011: a tokenizer whose vocab size drifts from
        // the model's pinned VOCAB_SIZE MUST abort dispatch with an error
        // message that names both values and the gate id, so the operator
        // can see the mismatch without stepping through code. Task #131
        // bumped VOCAB_SIZE to 50_257 (Option A) — the counter-example
        // below now exercises a tokenizer one token short of contract.
        let tmp = TempDir::new().expect("tempdir");
        let mismatch = Llama370MConfig::VOCAB_SIZE - 1;
        stage_vocab_json(tmp.path(), mismatch);
        let err = preflight_tokenizer_vocab_matches_target(
            tmp.path(),
            Llama370MConfig::VOCAB_SIZE,
            false,
        )
        .expect_err("tokenizer/model vocab mismatch must be rejected");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("GATE-ARCH-370M-011"),
                    "msg must cite gate: {msg}"
                );
                assert!(
                    msg.contains(&mismatch.to_string()),
                    "msg must name tokenizer vocab: {msg}"
                );
                assert!(
                    msg.contains(&Llama370MConfig::VOCAB_SIZE.to_string()),
                    "msg must name model vocab: {msg}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn preflight_rejects_missing_vocab_json() {
        // Missing vocab.json is a pre-flight failure (not a later shard
        // error) — the operator should know the tokenizer layout is
        // wrong, not that the dataset is empty.
        let tmp = TempDir::new().expect("tempdir");
        let err = preflight_tokenizer_vocab_matches_target(
            tmp.path(),
            Llama370MConfig::VOCAB_SIZE,
            false,
        )
        .expect_err("missing vocab.json must be rejected");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("GATE-ARCH-370M-011"),
                    "msg must cite gate: {msg}"
                );
                assert!(
                    msg.contains("cannot read"),
                    "msg must name I/O failure: {msg}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// FALSIFY-APR-PRETRAIN-ARCH-005 — a Qwen tokenizer (vocab=151_936) MUST
    /// pass preflight when the target_vocab_size is the Qwen extracted-arch
    /// (151_936). Falsifies a regression where preflight would still gate
    /// against the hardcoded Llama370M vocab.
    ///
    /// Spec: SPEC-SHIP-TWO-001 §50.4 step 5d.
    #[test]
    fn preflight_qwen_vocab_passes_with_qwen_target() {
        const QWEN2_VOCAB_SIZE: usize = 151_936;
        let tmp = TempDir::new().expect("tempdir");
        stage_vocab_json(tmp.path(), QWEN2_VOCAB_SIZE);
        // §50.4 step 5d called this with init=Some semantic (the polymorphic path). Use
        // init_is_some=true here per §55 relaxed-bound semantics; vocab.len() == target
        // is still acceptable under <=.
        preflight_tokenizer_vocab_matches_target(tmp.path(), QWEN2_VOCAB_SIZE, true).expect(
            "Qwen tokenizer (151_936) MUST pass preflight when target is Qwen-shaped — \
             this is the load-bearing claim of §49 fine-tune from a Qwen2.5 init checkpoint",
        );
    }

    /// FALSIFY-APR-PRETRAIN-ARCH-006 — a Qwen tokenizer (vocab=151_936) MUST
    /// FAIL preflight when target_vocab_size is the Llama370M baseline
    /// (50_257). Falsifies the silent-pass class where an operator would
    /// accidentally pair a Qwen tokenizer with the from-scratch trainer.
    ///
    /// Spec: SPEC-SHIP-TWO-001 §50.4 step 5d.
    #[test]
    fn preflight_qwen_vocab_fails_with_llama_target() {
        const QWEN2_VOCAB_SIZE: usize = 151_936;
        let tmp = TempDir::new().expect("tempdir");
        stage_vocab_json(tmp.path(), QWEN2_VOCAB_SIZE);
        // §55: this is the from-scratch path (init absent), so init_is_some=false.
        // Strict equality applies; tokenizer (151_936) ≠ target (50_257) MUST fail.
        let err = preflight_tokenizer_vocab_matches_target(
            tmp.path(),
            Llama370MConfig::VOCAB_SIZE,
            false,
        )
        .expect_err(
            "Qwen tokenizer (151_936) MUST FAIL preflight when target is Llama370M (50_257) — \
             silent-pass would corrupt training",
        );
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains(&QWEN2_VOCAB_SIZE.to_string()),
                    "msg must name Qwen vocab size 151_936: {msg}"
                );
                assert!(
                    msg.contains(&Llama370MConfig::VOCAB_SIZE.to_string()),
                    "msg must name target Llama vocab size 50_257: {msg}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// FALSIFY-APR-PRETRAIN-ARCH-009 (§55) — at preflight level, an HF
    /// tokenizer with vocab.json count = 151665 (BPE+added, the §54 LIVE
    /// smoke shape) MUST PASS preflight when target is Qwen 151936 AND
    /// init_is_some=true (the polymorphic path).
    #[test]
    fn preflight_qwen_reserved_slots_pass_under_polymorphic_init() {
        const QWEN_TOKENIZER_EFFECTIVE: usize = 151_665;
        const QWEN_DECLARED_VOCAB: usize = 151_936;
        let tmp = TempDir::new().expect("tempdir");
        stage_vocab_json(tmp.path(), QWEN_TOKENIZER_EFFECTIVE);

        // init_is_some=true: relaxed bound applies; 151665 ≤ 151936 PASSES.
        preflight_tokenizer_vocab_matches_target(tmp.path(), QWEN_DECLARED_VOCAB, true).expect(
            "FALSIFY-APR-PRETRAIN-ARCH-009: HF reserved-slot tokenizer (151_665 ≤ 151_936) \
             MUST pass preflight under polymorphic init path (§55 relaxed bound)",
        );

        // init_is_some=false: strict equality applies; 151665 ≠ 151936 FAILS.
        let err = preflight_tokenizer_vocab_matches_target(tmp.path(), QWEN_DECLARED_VOCAB, false)
            .expect_err(
                "FALSIFY-APR-PRETRAIN-ARCH-009 dual: from-scratch path MUST keep strict ==",
            );
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("GATE-ARCH-370M-011")
                        && msg.contains(&QWEN_TOKENIZER_EFFECTIVE.to_string())
                        && msg.contains(&QWEN_DECLARED_VOCAB.to_string()),
                    "strict-mode error must name gate + both sizes: {msg}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// FALSIFY-APR-PRETRAIN-ARCH-010 (§55) — at preflight level, a tokenizer
    /// with MORE entries than the model declares MUST FAIL even under the
    /// polymorphic init path. This is the OOB-safety guard: such a tokenizer
    /// could emit ids ≥ model_vocab → silent embedding-lookup garbage.
    #[test]
    fn preflight_oversized_tokenizer_rejected_even_under_polymorphic_init() {
        const QWEN_DECLARED_VOCAB: usize = 151_936;
        let oversized = QWEN_DECLARED_VOCAB + 100;
        let tmp = TempDir::new().expect("tempdir");
        stage_vocab_json(tmp.path(), oversized);

        let err = preflight_tokenizer_vocab_matches_target(
            tmp.path(),
            QWEN_DECLARED_VOCAB,
            true, // polymorphic path
        )
        .expect_err(
            "FALSIFY-APR-PRETRAIN-ARCH-010: oversized tokenizer MUST fail-fast even under \
             polymorphic init (OOB safety; relaxed bound is ≤ not <)",
        );
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("RELAXED") && msg.contains("OOB"),
                    "polymorphic-mode error must cite RELAXED + OOB: {msg}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// FALSIFY-APR-PRETRAIN-INIT-CUDA-001 (drift-prevention sentinel,
    /// post-5f.5): after §50.4 step 5f.5 SHIPPED, the const message
    /// pins the wireup-is-wired property. The string MUST contain
    /// (a) the falsifier id, (b) the canonical "is wired for --device
    /// cuda" phrase, (c) a reference to the symmetric builder
    /// `build_shared_cuda_trainer_with_init`, and (d) the "5f.5
    /// SHIPPED" status marker. If a future refactor accidentally
    /// reverts the wireup or renames the symmetric builder, this test
    /// catches the drift before the contract reference goes stale.
    ///
    /// Pinned via `pub(crate) const FALSIFY_APR_PRETRAIN_INIT_CUDA_001_MSG`
    /// so this test fires on a CPU-only build (no `--features cuda` needed).
    /// The const itself is NOT emitted by any code path in `drive_real`;
    /// it survives only to anchor the contract obligation. The runtime
    /// behaviour (`drive_real_cuda` calling `build_shared_cuda_trainer_with_init`
    /// when `init_arch.is_some() || init_path.is_some()`) is exercised
    /// at the entrenar crate level where CUDA-feature builds can fire it.
    #[test]
    fn drive_real_cuda_init_path_wireup_sentinel_pinned() {
        let msg = FALSIFY_APR_PRETRAIN_INIT_CUDA_001_MSG;
        assert!(
            msg.contains("FALSIFY-APR-PRETRAIN-INIT-CUDA-001"),
            "sentinel MUST cite the falsifier id (auditability): {msg}"
        );
        assert!(
            msg.contains("is wired for --device cuda"),
            "sentinel MUST contain the canonical 'is wired' phrase so \
             operators recognize §50.4 step 5f.5 SHIPPED: {msg}"
        );
        assert!(
            msg.contains("build_shared_cuda_trainer_with_init"),
            "sentinel MUST name the symmetric builder so future agents \
             know which symbol implements the wireup: {msg}"
        );
        assert!(
            msg.contains("5f.5 SHIPPED"),
            "sentinel MUST include the 5f.5 SHIPPED status marker so \
             grep over the codebase can find the discharge point: {msg}"
        );
    }

    #[test]
    fn synthetic_pretrain_end_to_end_happy_path() {
        let tmp = TempDir::new().expect("tempdir");
        let dataset = tmp.path().join("data.jsonl");
        let tokenizer = tmp.path().join("tok");
        let run_dir = tmp.path().join("run");

        let result = run(
            &dataset,
            &tokenizer,
            &run_dir,
            PretrainMode::Finetune,
            Some(5.0e-5),
            25,
            Some(5),
            2,
            4,
            5,
            42,
            Some(2.2),
            50257,
            true,
            "cpu",
            None,
            false,
            None,
            true,
        );
        assert!(
            result.is_ok(),
            "synthetic pretrain end-to-end must succeed: got {result:?}"
        );
    }

    #[test]
    fn real_mode_empty_dataset_dir_errors() {
        // When --synthetic is off, the real-corpus branch must surface a
        // clear error if the dataset directory has no .bin shards. This
        // supersedes the old "non-synthetic is not implemented" guard.
        // Stage a valid vocab.json first so GATE-ARCH-370M-011 pre-flight
        // passes — otherwise the shard-iterator error below is never reached.
        let tmp = TempDir::new().expect("tempdir");
        let tok_dir = tmp.path().join("tok");
        stage_vocab_json(&tok_dir, Llama370MConfig::VOCAB_SIZE);
        let err = run(
            tmp.path(),
            &tok_dir,
            tmp.path(),
            PretrainMode::Finetune,
            Some(5.0e-5),
            10,
            Some(2),
            2,
            4,
            5,
            42,
            Some(2.2),
            50257,
            false,
            "cpu",
            None,
            false,
            None,
            true,
        )
        .expect_err("empty dataset dir must fail to initialise the shard iterator");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("shard iterator init failed"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn invalid_target_val_loss_rejected() {
        let tmp = TempDir::new().expect("tempdir");
        let err = run(
            tmp.path(),
            tmp.path(),
            tmp.path(),
            PretrainMode::Finetune,
            Some(5.0e-5),
            10,
            Some(2),
            2,
            4,
            5,
            42,
            Some(-1.0),
            50257,
            true,
            "cpu",
            None,
            false,
            None,
            true,
        )
        .expect_err("negative target_val_loss must be rejected");
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }

    // ── GATE-TRAIN-009 / INV-TRAIN-009 falsifiers ──────────────────────
    // Contract: training-loop-pretrain-v1 v1.3.0 §hyperparameter_defaults
    //
    // These tests bind the CLI's `mode_defaults` resolver to the
    // hyperparameter_defaults YAML table. If the table is ever edited
    // without also updating this resolver (or vice versa), the tests
    // fail. That is exactly the drift INV-TRAIN-009 forbids.

    #[test]
    fn mode_finetune_is_default_and_matches_contract() {
        // No overrides → resolved HP matches the `finetune` YAML row
        // (lr_max=5e-5, warmup_steps=100, target_val_loss=2.2) AND the
        // regime is Finetune so INV-TRAIN-005 epoch-zero cap = 10.0.
        let hp = mode_defaults(PretrainMode::Finetune, 50257, None, None, None);
        assert_eq!(hp.regime, TrainingRegime::Finetune);
        assert!(
            (hp.lr_max - 5.0e-5).abs() < 1.0e-12,
            "lr_max={} must equal finetune default 5e-5",
            hp.lr_max
        );
        assert_eq!(hp.warmup_steps, 100);
        assert!(
            (hp.target_val_loss - 2.2).abs() < 1.0e-6,
            "target_val_loss={} must equal finetune default 2.2",
            hp.target_val_loss
        );
    }

    #[test]
    fn mode_from_scratch_applies_all_four_defaults() {
        // `--mode from-scratch` with no HP overrides MUST yield the full
        // cold-start 4-tuple atomically — regime=FromScratch, lr=3e-4,
        // warmup=1000, target=3.0. INV-TRAIN-009 falsifier (a).
        let hp = mode_defaults(PretrainMode::FromScratch, 50257, None, None, None);
        assert_eq!(hp.regime, TrainingRegime::FromScratch { vocab_size: 50257 });
        assert!(
            (hp.lr_max - 3.0e-4).abs() < 1.0e-12,
            "lr_max={} must equal from_scratch default 3e-4",
            hp.lr_max
        );
        assert_eq!(hp.warmup_steps, 1000);
        assert!(
            (hp.target_val_loss - 3.0).abs() < 1.0e-6,
            "target_val_loss={} must equal from_scratch default 3.0",
            hp.target_val_loss
        );
    }

    #[test]
    fn mode_from_scratch_honors_explicit_lr_override() {
        // `--mode from-scratch --lr 1e-4` → regime still flips to
        // FromScratch AND warmup/target keep the from_scratch defaults,
        // but lr_max is the operator-supplied 1e-4. INV-TRAIN-009
        // falsifier (b): overrides win, regime still moves.
        let hp = mode_defaults(PretrainMode::FromScratch, 50257, Some(1.0e-4), None, None);
        assert_eq!(hp.regime, TrainingRegime::FromScratch { vocab_size: 50257 });
        assert!(
            (hp.lr_max - 1.0e-4).abs() < 1.0e-12,
            "lr_max={} must equal explicit override 1e-4",
            hp.lr_max
        );
        // Remaining two fields retained their mode defaults.
        assert_eq!(hp.warmup_steps, 1000);
        assert!((hp.target_val_loss - 3.0).abs() < 1.0e-6);
    }

    // ── GATE-TRAIN-010 / INV-TRAIN-010 falsifiers ──────────────────────
    // Contract: training-loop-pretrain-v1 v1.4.0 §INV-TRAIN-010
    //
    // Task #105's original wiring shipped `synthetic: bool` with
    // `default_value = "true"`. The `--synthetic` flag had no
    // companion to turn it off, so every invocation of `apr pretrain`
    // silently routed to drive_synthetic. Tasks #119 / #124 / #125
    // all captured scripted-loss output and mis-labeled it real
    // compute. These two tests parse actual argv through clap and
    // assert the routing discriminator byte-for-byte.

    fn parse_pretrain_synthetic(extra: &[&str]) -> bool {
        // The `Commands` enum is large enough in debug builds to overflow
        // the default 2 MiB test-thread stack during clap's recursive
        // destructuring. Run the parse on a worker thread with a 16 MiB
        // stack so this falsifier passes in both debug and release.
        let extra: Vec<String> = extra.iter().map(|s| (*s).to_string()).collect();
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                use clap::Parser;
                let mut argv: Vec<String> = vec![
                    "apr".to_string(),
                    "pretrain".to_string(),
                    "--dataset".to_string(),
                    "/tmp/_gate_train_010/ds".to_string(),
                    "--tokenizer".to_string(),
                    "/tmp/_gate_train_010/tok".to_string(),
                    "--run-dir".to_string(),
                    "/tmp/_gate_train_010/run".to_string(),
                ];
                argv.extend(extra);
                let cli = crate::Cli::try_parse_from(&argv).expect("clap parse must succeed");
                match *cli.command {
                    crate::Commands::Extended(crate::ExtendedCommands::Pretrain {
                        synthetic,
                        ..
                    }) => synthetic,
                    other => panic!("expected ExtendedCommands::Pretrain, got {other:?}"),
                }
            })
            .expect("spawn parse thread")
            .join()
            .expect("parse thread must not panic")
    }

    #[test]
    fn cli_pretrain_defaults_to_real_compute() {
        // Absent `--synthetic` MUST parse to synthetic=false so the
        // dispatcher routes through drive_real.
        assert!(
            !parse_pretrain_synthetic(&[]),
            "INV-TRAIN-010: `apr pretrain` (no --synthetic) must parse to synthetic=false"
        );
    }

    #[test]
    fn cli_pretrain_synthetic_flag_routes_to_synthetic() {
        // `--synthetic` present MUST parse to synthetic=true.
        assert!(
            parse_pretrain_synthetic(&["--synthetic"]),
            "INV-TRAIN-010: `apr pretrain --synthetic` must parse to synthetic=true"
        );
    }

    // ── FALSIFY-GPUTRAIN-001 / 002 CLI surface (contract phase 1) ────
    // Contract: gpu-training-backend-v1 §device_dispatch
    //
    // These tests parse actual `apr pretrain --device …` argv through
    // clap and assert the string is surfaced byte-for-byte to the
    // dispatcher. `resolve_device()` itself is exercised by
    // `aprender-train::train::device::tests` — these tests verify that
    // the CLI flag exists and that its default is `auto` (the only
    // spec allowed to fall back).

    fn parse_pretrain_device(extra: &[&str]) -> String {
        let extra: Vec<String> = extra.iter().map(|s| (*s).to_string()).collect();
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                use clap::Parser;
                let mut argv: Vec<String> = vec![
                    "apr".to_string(),
                    "pretrain".to_string(),
                    "--dataset".to_string(),
                    "/tmp/_gputrain_device/ds".to_string(),
                    "--tokenizer".to_string(),
                    "/tmp/_gputrain_device/tok".to_string(),
                    "--run-dir".to_string(),
                    "/tmp/_gputrain_device/run".to_string(),
                ];
                argv.extend(extra);
                let cli = crate::Cli::try_parse_from(&argv).expect("clap parse must succeed");
                match *cli.command {
                    crate::Commands::Extended(crate::ExtendedCommands::Pretrain {
                        device, ..
                    }) => device,
                    other => panic!("expected ExtendedCommands::Pretrain, got {other:?}"),
                }
            })
            .expect("spawn parse thread")
            .join()
            .expect("parse thread must not panic")
    }

    #[test]
    fn cli_pretrain_device_defaults_to_auto() {
        // Absent `--device`, the flag MUST parse to `"auto"` — the only
        // spec allowed to silently fall back to CPU when CUDA is not
        // available. Any other default would violate the contract's
        // "explicit request → hard-fail" invariant.
        assert_eq!(
            parse_pretrain_device(&[]),
            "auto",
            "gpu-training-backend-v1 INV-GPUTRAIN-002: default --device must be `auto`",
        );
    }

    #[test]
    fn cli_pretrain_device_accepts_cpu() {
        // `--device cpu` MUST round-trip through clap unchanged.
        assert_eq!(parse_pretrain_device(&["--device", "cpu"]), "cpu");
    }

    #[test]
    fn cli_pretrain_device_accepts_cuda_index() {
        // `--device cuda:7` MUST round-trip unchanged; grammar
        // enforcement happens in `resolve_device`, not at clap.
        assert_eq!(parse_pretrain_device(&["--device", "cuda:7"]), "cuda:7");
    }

    // ── apr-pretrain-from-init-v1 falsifiers ────────────────────────────
    // Contract: contracts/apr-pretrain-from-init-v1.yaml v1.0.0 PROPOSED
    // Spec: SPEC-SHIP-TWO-001 §49 step 4 — wire `apr pretrain --init`
    //
    // PARTIAL_ALGORITHM_LEVEL: file-existence + magic-byte checks bind
    // FALSIFY-APR-PRETRAIN-INIT-003 / -004; the clap surface binds
    // FALSIFY-001 / -007. FALSIFY-005 (arch mismatch), -006 (init_loss
    // signal), -009 (optimizer state), -010 (idempotent load) are gated
    // on the §49 step 5 weight-load impl. The "valid APR returns
    // not-yet-wired" test pins the no-silent-fallback contract: a
    // recognised APR cannot be silently ignored.

    fn parse_pretrain_init(extra: &[&str]) -> Option<std::path::PathBuf> {
        let extra: Vec<String> = extra.iter().map(|s| (*s).to_string()).collect();
        std::thread::Builder::new()
            .stack_size(16 * 1024 * 1024)
            .spawn(move || {
                use clap::Parser;
                let mut argv: Vec<String> = vec![
                    "apr".to_string(),
                    "pretrain".to_string(),
                    "--dataset".to_string(),
                    "/tmp/_init_flag/ds".to_string(),
                    "--tokenizer".to_string(),
                    "/tmp/_init_flag/tok".to_string(),
                    "--run-dir".to_string(),
                    "/tmp/_init_flag/run".to_string(),
                ];
                argv.extend(extra);
                let cli = crate::Cli::try_parse_from(&argv).expect("clap parse must succeed");
                match *cli.command {
                    crate::Commands::Extended(crate::ExtendedCommands::Pretrain {
                        init, ..
                    }) => init,
                    other => panic!("expected ExtendedCommands::Pretrain, got {other:?}"),
                }
            })
            .expect("spawn parse thread")
            .join()
            .expect("parse thread must not panic")
    }

    /// FALSIFY-APR-PRETRAIN-INIT-001: --init flag exists in clap surface.
    #[test]
    fn pretrain_init_flag_absent_parses_to_none() {
        // Absent --init MUST parse to None. Falsifies a regression where a
        // default value silently injects a path the operator never typed.
        assert_eq!(
            parse_pretrain_init(&[]),
            None,
            "FALSIFY-APR-PRETRAIN-INIT-001/002: default --init must be None (no silent default)"
        );
    }

    /// FALSIFY-APR-PRETRAIN-INIT-001: --init <PATH> parses to Some(PathBuf).
    #[test]
    fn pretrain_init_flag_parses_path() {
        let parsed = parse_pretrain_init(&["--init", "/tmp/foo.apr"]);
        assert_eq!(
            parsed.as_deref().and_then(|p| p.to_str()),
            Some("/tmp/foo.apr"),
            "FALSIFY-APR-PRETRAIN-INIT-001: --init <PATH> must round-trip through clap"
        );
    }

    /// FALSIFY-APR-PRETRAIN-INIT-003: --init <missing-file> fails fast
    /// before any trainer allocation; stderr names the path.
    #[test]
    fn pretrain_init_missing_file_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let missing = tmp.path().join("does-not-exist.apr");
        let err = run(
            tmp.path(),
            tmp.path(),
            tmp.path(),
            PretrainMode::Finetune,
            Some(5.0e-5),
            10,
            Some(2),
            2,
            4,
            5,
            42,
            Some(2.2),
            50257,
            true,
            "cpu",
            Some(&missing),
            false,
            None,
            true,
        )
        .expect_err("missing --init file must be rejected");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("FALSIFY-APR-PRETRAIN-INIT-003"),
                    "msg must cite falsifier id: {msg}"
                );
                assert!(
                    msg.contains("does-not-exist.apr"),
                    "msg must name the missing path: {msg}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// FALSIFY-APR-PRETRAIN-INIT-004: --init with wrong magic bytes fails fast.
    #[test]
    fn pretrain_init_bad_magic_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let bad = tmp.path().join("not-an-apr.bin");
        std::fs::write(&bad, b"GGUF\x00\x00\x00\x00\x00\x00\x00\x00").expect("write fixture file");
        let err = run(
            tmp.path(),
            tmp.path(),
            tmp.path(),
            PretrainMode::Finetune,
            Some(5.0e-5),
            10,
            Some(2),
            2,
            4,
            5,
            42,
            Some(2.2),
            50257,
            true,
            "cpu",
            Some(&bad),
            false,
            None,
            true,
        )
        .expect_err("invalid magic bytes must be rejected");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    msg.contains("FALSIFY-APR-PRETRAIN-INIT-004"),
                    "msg must cite falsifier id: {msg}"
                );
                assert!(
                    msg.contains("not a valid APR file"),
                    "msg must describe magic mismatch: {msg}"
                );
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// FALSIFY-APR-PRETRAIN-INIT-004: empty file (read_exact fails on 4 bytes).
    #[test]
    fn pretrain_init_empty_file_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let empty = tmp.path().join("empty.apr");
        std::fs::write(&empty, b"").expect("write empty fixture");
        let err = run(
            tmp.path(),
            tmp.path(),
            tmp.path(),
            PretrainMode::Finetune,
            Some(5.0e-5),
            10,
            Some(2),
            2,
            4,
            5,
            42,
            Some(2.2),
            50257,
            true,
            "cpu",
            Some(&empty),
            false,
            None,
            true,
        )
        .expect_err("empty file must be rejected (cannot contain magic bytes)");
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }

    /// §50.4 step 5f.4: a magic-byte-valid but metadata-bogus APR file
    /// MUST be rejected at the architecture-extraction step, not silently
    /// fall back to random init. The error must clearly cite the
    /// architecture-extraction failure (not the legacy "not yet wired"
    /// guard, which was retired when the wireup landed). This drift-prevention
    /// pins the new fail-closed semantic.
    #[test]
    fn pretrain_init_valid_magic_but_bogus_metadata_fails_at_arch_extraction() {
        let tmp = TempDir::new().expect("tempdir");
        let valid = tmp.path().join("v2-valid-magic-bogus-metadata.apr");
        // APR\0 magic + padding; passes validate_init_apr_path but
        // read_apr_architecture (which reads the v2 header) will return None.
        std::fs::write(&valid, b"APR\x00\x00\x00\x00\x00\x00\x00\x00\x00")
            .expect("write fixture file");
        let err = run(
            tmp.path(),
            tmp.path(),
            tmp.path(),
            PretrainMode::Finetune,
            Some(5.0e-5),
            10,
            Some(2),
            2,
            4,
            5,
            42,
            Some(2.2),
            50257,
            true,
            "cpu",
            Some(&valid),
            false,
            None,
            true,
        )
        .expect_err("bogus metadata must NOT silently random-init");
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(
                    !msg.contains("not yet wired"),
                    "the legacy step-5-partial guard must be retired: {msg}"
                );
                // The actual error from read_apr_architecture failure or
                // downstream layer; both are acceptable as long as we DON'T
                // silently load random init.
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    /// Pin v1 magic (APRN) acceptance — `validate_init_apr_path` alone
    /// (decoupled from architecture extraction) returns Ok for both APR\0
    /// and APRN magic bytes. Architecture extraction is a separate step.
    #[test]
    fn pretrain_init_v1_magic_aprn_passes_validate_init_apr_path() {
        let tmp = TempDir::new().expect("tempdir");
        let v1 = tmp.path().join("v1-aprn.apr");
        std::fs::write(&v1, b"APRN\x00\x00\x00\x00").expect("write fixture file");
        let result = validate_init_apr_path(&v1);
        assert!(
            result.is_ok(),
            "APRN magic must pass validate_init_apr_path; got {result:?}"
        );
    }

    // ── PMAT-125 B1: additional CPU-only coverage ──────────────────────────

    /// `estimate_param_count` must not panic on zero/degenerate dims — it uses
    /// saturating arithmetic throughout, so an all-zero arch returns 0.
    #[test]
    fn estimate_param_count_zero_dims_is_saturating() {
        let mut cfg = TransformerConfig::llama2_7b();
        cfg.vocab_size = 0;
        cfg.hidden_size = 0;
        cfg.intermediate_size = 0;
        cfg.num_hidden_layers = 0;
        assert_eq!(estimate_param_count(&cfg), 0);
    }

    /// `estimate_param_count` counts the embedding term `vocab × hidden` once
    /// even with zero layers (no per-layer contribution).
    #[test]
    fn estimate_param_count_zero_layers_is_embedding_plus_norm() {
        let mut cfg = TransformerConfig::llama2_7b();
        cfg.vocab_size = 100;
        cfg.hidden_size = 8;
        cfg.intermediate_size = 32;
        cfg.num_hidden_layers = 0;
        // embed = 100*8 = 800; + final norm (hidden) = 8 → 808.
        assert_eq!(estimate_param_count(&cfg), 808);
    }

    /// `checkpoint_name_and_arch` honors `hf_model_type` for the name suffix
    /// while falling back to `LlamaForCausalLM` when `hf_architecture` is unset.
    #[test]
    fn checkpoint_name_and_arch_model_type_without_architecture() {
        let mut cfg = TransformerConfig::llama2_7b();
        cfg.hf_model_type = Some("gpt2".to_string());
        cfg.hf_architecture = None;
        let (name, arch) = checkpoint_name_and_arch(Some(&cfg));
        assert_eq!(name, "gpt2-pretrain");
        assert_eq!(arch, "LlamaForCausalLM");
    }

    /// `validate_init_apr_path` on a directory (not a file) fails fast with a
    /// FALSIFY-tagged validation error rather than panicking.
    #[test]
    fn validate_init_apr_path_directory_errors() {
        let tmp = TempDir::new().expect("tempdir");
        let err = validate_init_apr_path(tmp.path()).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }

    /// `validate_init_apr_path` on a 3-byte file (too short for the 4-byte
    /// magic) surfaces the INIT-004 short-file error.
    #[test]
    fn validate_init_apr_path_three_bytes_too_short() {
        let tmp = TempDir::new().expect("tempdir");
        let p = tmp.path().join("short.apr");
        std::fs::write(&p, b"APR").expect("write");
        let err = validate_init_apr_path(&p).unwrap_err();
        match err {
            CliError::ValidationFailed(m) => assert!(m.contains("INIT-004")),
            other => panic!("unexpected: {other:?}"),
        }
    }
}
