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
use entrenar::train::cycling_iter::CyclingBatchIter;
use entrenar::train::device::{resolve_device, Device};
use entrenar::train::pretrain::{
    CheckpointFn, LinearDecaySynthetic, PretrainConfig, PretrainLoop, RunStatus, ScriptedVal,
    StepFn, TrainingRegime, ValFn,
};
use entrenar::train::pretrain_real::{
    build_shared_trainer, AprCheckpointFn, RealStepFn, RealValFn,
};
use entrenar::train::shard_reader::ShardBatchIter;
use entrenar::train::transformer_trainer::LMBatch;
use std::path::Path;

#[path = "pretrain_preflight.rs"]
mod preflight;
use preflight::{preflight_dispatch_budget, preflight_tokenizer_vocab_matches_model};

#[path = "pretrain_report.rs"]
mod report_mod;
use report_mod::{abort_to_err, print_header, report};

/// Number of LMBatches pulled off the head of the shard stream and
/// reserved as the held-out validation set. Chosen as a small constant
/// for MVP; follow-up ticket will plumb an explicit `--val-shards`
/// flag so training and validation can target disjoint shard files.
const HELD_OUT_BATCHES: usize = 2;

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
    allow_shard_cycle: bool,
    json_output: bool,
) -> Result<()> {
    // Contract gpu-training-backend-v1 INV-GPUTRAIN-001 / GATE-GPUTRAIN-002:
    // parse --device BEFORE any trainer allocation so an invalid spec
    // or an explicit `cuda` on a CPU-only host fails fast with a clear
    // diagnostic. Synthetic drive still honours --device (for parity
    // with real compute) but the stub error surface is identical.
    let resolved_device =
        resolve_device(device).map_err(|e| CliError::ValidationFailed(e.to_string()))?;

    let hp = mode_defaults(mode, vocab_size, lr, warmup_steps, target_val_loss);

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
        patience_epochs: 2,
        min_epochs_before_early_stop: 1,
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
            num_steps,
            seq_length,
            batch_size,
            seed,
            resolved_device,
            allow_shard_cycle,
            json_output,
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
    num_steps: usize,
    seq_length: usize,
    batch_size: usize,
    seed: u64,
    device: Device,
    allow_shard_cycle: bool,
    json_output: bool,
) -> Result<RunStatus> {
    // GATE-ARCH-370M-011 / INV-ARCH-370M-006 — refuse to dispatch a real
    // training step when the tokenizer vocab_size and the model vocab_size
    // disagree. The N-09 OOB escape guard in Embedding::forward masks the
    // mismatch at runtime → silent garbage gradients otherwise. Synthetic
    // drive skips this check because it never touches the real model.
    preflight_tokenizer_vocab_matches_model(&config.tokenizer_dir)?;

    // GATE-CORPUS-PREFLIGHT / FALSIFY-CORPUS-004 — refuse to dispatch
    // when the planned token budget exceeds corpus total_tokens unless
    // `--allow-shard-cycle` is set. Runs BEFORE trainer allocation so
    // a wrong-sized corpus costs zero GPU time.
    let (planned_tokens, total_tokens) = preflight_dispatch_budget(
        dataset,
        num_steps,
        batch_size,
        seq_length,
        allow_shard_cycle,
    )?;

    // MVP: pad_id/eos_id both 0. All sequences are uniform length
    // (seq_length + 1) so LMBatch::from_sequences takes the shared
    // layout path and pad_id is never used for padding. The real
    // tokenizer's special-token ids will plumb through in a follow-up.
    let mut iter = ShardBatchIter::new(dataset, batch_size, seq_length, 0, 0).map_err(|e| {
        CliError::ValidationFailed(format!(
            "dataset shard iterator init failed: {e} (path={})",
            dataset.display()
        ))
    })?;

    // Reserve the first `HELD_OUT_BATCHES` batches as the held-out val
    // set; the remainder feeds RealStepFn.
    let mut held_out: Vec<LMBatch> = Vec::with_capacity(HELD_OUT_BATCHES);
    for _ in 0..HELD_OUT_BATCHES {
        match iter.next() {
            Some(b) => held_out.push(b),
            None => break,
        }
    }
    if held_out.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "dataset {} is too small to reserve any held-out batches",
            dataset.display()
        )));
    }

    // Resolve the training iterator. When the operator opted into
    // cycling AND the planned budget exceeds the corpus, wrap the
    // shard reader in a `CyclingBatchIter` that re-opens the dir
    // (skipping held-out batches) on each cycle boundary and emits a
    // single INFO log at the first cycle. Otherwise use the plain
    // iterator so `GATE-TRAIN-EXHAUST` fires on over-run.
    let train_iter: Box<dyn Iterator<Item = LMBatch>> =
        if allow_shard_cycle && planned_tokens > total_tokens {
            let dataset_pb = dataset.to_path_buf();
            let factory = move || -> Box<dyn Iterator<Item = LMBatch>> {
                let mut fresh = ShardBatchIter::new(&dataset_pb, batch_size, seq_length, 0, 0)
                    .expect("GATE-CORPUS-PREFLIGHT cycle factory: shard re-open failed");
                // Keep the train/val split stable across cycles by
                // skipping past the held-out prefix on every rebuild.
                for _ in 0..HELD_OUT_BATCHES {
                    if fresh.next().is_none() {
                        break;
                    }
                }
                Box::new(fresh)
            };
            Box::new(CyclingBatchIter::new(factory))
        } else {
            Box::new(iter)
        };

    if device.is_cuda() {
        drive_real_cuda(
            config,
            train_iter,
            held_out,
            lr,
            seq_length,
            seed,
            json_output,
        )
    } else {
        drive_real_cpu(
            config,
            train_iter,
            held_out,
            lr,
            seq_length,
            seed,
            json_output,
        )
    }
}

/// CPU backend for `drive_real` — builds a `TransformerTrainer`
/// (`aprender::Tensor` + trueno SIMD) and wires `RealStepFn` /
/// `RealValFn` / `AprCheckpointFn`. Accepts a pre-boxed iterator so
/// the caller can choose between a plain `ShardBatchIter` (INV-TRAIN-011
/// hard-fail on exhaustion) and a `CyclingBatchIter` wrapper
/// (INV-TRAIN-011 path a, `--allow-shard-cycle`).
#[allow(clippy::too_many_arguments)]
fn drive_real_cpu(
    config: PretrainConfig,
    iter: Box<dyn Iterator<Item = LMBatch>>,
    held_out: Vec<LMBatch>,
    lr: f32,
    seq_length: usize,
    seed: u64,
    json_output: bool,
) -> Result<RunStatus> {
    let trainer = build_shared_trainer(lr, seq_length, seed);
    let step_fn = RealStepFn::new(trainer.clone(), iter);
    let val_fn = RealValFn::new(trainer.clone(), held_out);
    let ckpt: Box<dyn CheckpointFn> = Box::new(AprCheckpointFn::new(
        trainer,
        "llama-370m-pretrain",
        "LlamaForCausalLM",
    ));
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
    iter: Box<dyn Iterator<Item = LMBatch>>,
    held_out: Vec<LMBatch>,
    lr: f32,
    seq_length: usize,
    seed: u64,
    json_output: bool,
) -> Result<RunStatus> {
    use entrenar::train::pretrain_real_cuda::{
        build_shared_cuda_trainer, CudaAprCheckpointFn, CudaRealStepFn, CudaRealValFn,
    };
    let trainer = build_shared_cuda_trainer(lr, seq_length, seed).map_err(|e| {
        CliError::ValidationFailed(format!(
            "GATE-GPUTRAIN-002: CUDA trainer allocation failed: {e}. \
             See contracts/entrenar/gpu-training-backend-v1.yaml and \
             memory/feedback_cuda_feature_footgun.md — this path is \
             only reachable when the binary was built with `--features cuda`.",
        ))
    })?;
    let step_fn = CudaRealStepFn::new(trainer.clone(), iter);
    let val_fn = CudaRealValFn::new(trainer.clone(), held_out);
    let ckpt: Box<dyn CheckpointFn> = Box::new(CudaAprCheckpointFn::new(
        trainer,
        "llama-370m-pretrain",
        "LlamaForCausalLM",
    ));
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
    _iter: Box<dyn Iterator<Item = LMBatch>>,
    _held_out: Vec<LMBatch>,
    _lr: f32,
    _seq_length: usize,
    _seed: u64,
    _json_output: bool,
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

#[cfg(test)]
#[path = "pretrain_tests.rs"]
mod tests;
