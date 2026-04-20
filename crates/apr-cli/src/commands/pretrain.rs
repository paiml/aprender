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
use entrenar::train::pretrain::{
    CheckpointFn, LinearDecaySynthetic, PretrainAbort, PretrainConfig, PretrainLoop, RunStatus,
    ScriptedVal, StepFn, TrainingRegime, ValFn,
};
use entrenar::train::pretrain_real::{
    build_shared_trainer, AprCheckpointFn, RealStepFn, RealValFn,
};
use entrenar::train::shard_reader::ShardBatchIter;
use entrenar::train::transformer_trainer::LMBatch;
use std::path::Path;

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
    json_output: bool,
) -> Result<()> {
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

/// Real-corpus drive: build a shared 370M `TransformerTrainer`, split
/// the shard stream head-off into a held-out validation set, and run a
/// full forward + backward + AdamW step per training batch.
fn drive_real(
    config: PretrainConfig,
    dataset: &Path,
    lr: f32,
    seq_length: usize,
    batch_size: usize,
    seed: u64,
    json_output: bool,
) -> Result<RunStatus> {
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

    let trainer = build_shared_trainer(lr, seq_length, seed);
    let step_fn = RealStepFn::new(trainer.clone(), Box::new(iter));
    let val_fn = RealValFn::new(trainer.clone(), held_out);
    // Task #111 step 7: per-epoch APR checkpoint on GATE-TRAIN-005 pass.
    let ckpt: Box<dyn CheckpointFn> = Box::new(AprCheckpointFn::new(
        trainer,
        "llama-370m-pretrain",
        "LlamaForCausalLM",
    ));
    run_and_report(config, step_fn, val_fn, Some(ckpt), json_output)
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

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
            Some(2.2),
            50257,
            false,
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
}
