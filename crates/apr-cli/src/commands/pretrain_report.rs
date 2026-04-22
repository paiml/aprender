//! Terminal report + error-mapping helpers for `apr pretrain`
//! (extracted from `pretrain.rs` to keep the file-size invariant).
//!
//! Contract: `contracts/training-loop-pretrain-v1.yaml` — exit-code
//! mapping for GATE-TRAIN-005 / GATE-TRAIN-007 / GATE-TRAIN-008.

use crate::error::{CliError, Result};
use crate::output;
use colored::Colorize;
use entrenar::train::pretrain::{PretrainAbort, PretrainConfig, PretrainLoop, RunStatus};

pub(super) fn abort_to_err(abort: &PretrainAbort) -> CliError {
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

pub(super) fn print_header(cfg: &PretrainConfig) {
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

pub(super) fn report<S: entrenar::train::pretrain::StepFn, V: entrenar::train::pretrain::ValFn>(
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
