//! `apr grad-norm` — CRUX-F-09 gradient-norm telemetry analysis.
//!
//! Reads a JSON file of per-step records:
//!   [{ "step": u64, "grad_norm": f64,
//!      "grad_norm_clipped": f64|null, "loss": f64|null }, ...]
//! Dispatches `aprender::metrics::grad_norm::analyze_history` and emits
//! an aggregated report (text or `--json`).
//!
//! Spec: `contracts/crux-F-09-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use aprender::metrics::grad_norm::{analyze_history, HistoryReport, StepRecord};
use serde::Deserialize;

use crate::error::{CliError, Result};

#[derive(Debug, Deserialize)]
struct RawRecord {
    step: u64,
    grad_norm: f64,
    #[serde(default)]
    grad_norm_clipped: Option<f64>,
    #[serde(default)]
    loss: Option<f64>,
}

impl From<RawRecord> for StepRecord {
    fn from(r: RawRecord) -> Self {
        Self {
            step: r.step,
            grad_norm: r.grad_norm,
            grad_norm_clipped: r.grad_norm_clipped,
            loss: r.loss,
        }
    }
}

pub(crate) fn run(
    history_file: &Path,
    max_grad_norm: Option<f64>,
    spike_window: usize,
    spike_multiplier: f64,
    json: bool,
) -> Result<()> {
    if !history_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(history_file)));
    }

    let body = std::fs::read_to_string(history_file)?;

    let raw: Vec<RawRecord> = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr grad-norm: failed to parse JSON records from {}: {e}",
            history_file.display()
        ))
    })?;

    if raw.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "history file {} contains zero records",
            history_file.display()
        )));
    }

    if spike_multiplier <= 0.0 {
        return Err(CliError::ValidationFailed(format!(
            "--spike-multiplier must be > 0 (got {spike_multiplier})"
        )));
    }

    let records: Vec<StepRecord> = raw.into_iter().map(Into::into).collect();
    let report = analyze_history(&records, max_grad_norm, spike_window, spike_multiplier);

    if !report.all_non_negative {
        print_report(&report, history_file, max_grad_norm, json);
        return Err(CliError::ValidationFailed(
            "grad_norm field contains negative or non-finite value".to_string(),
        ));
    }
    if !report.clipping_non_expansive {
        print_report(&report, history_file, max_grad_norm, json);
        return Err(CliError::ValidationFailed(
            "grad_norm_clipped > grad_norm on at least one step (clipping cannot amplify)"
                .to_string(),
        ));
    }
    if report.max_exceeds_cap {
        print_report(&report, history_file, max_grad_norm, json);
        return Err(CliError::ValidationFailed(
            "grad_norm_clipped exceeds --max-grad-norm cap on at least one step".to_string(),
        ));
    }

    print_report(&report, history_file, max_grad_norm, json);
    Ok(())
}

fn print_report(report: &HistoryReport, path: &Path, cap: Option<f64>, json: bool) {
    if json {
        let v = serde_json::json!({
            "num_steps": report.num_steps,
            "min": report.min,
            "max": report.max,
            "mean": report.mean,
            "num_spikes": report.num_spikes,
            "all_non_negative": report.all_non_negative,
            "clipping_non_expansive": report.clipping_non_expansive,
            "max_exceeds_cap": report.max_exceeds_cap,
            "max_grad_norm": cap,
            "history_path": path.display().to_string(),
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("grad-norm report for {}", path.display());
        println!("  num_steps: {}", report.num_steps);
        println!("  min:       {:.6}", report.min);
        println!("  max:       {:.6}", report.max);
        println!("  mean:      {:.6}", report.mean);
        println!(
            "  num_spikes (rolling-median threshold): {}",
            report.num_spikes
        );
        println!("  all_non_negative:       {}", report.all_non_negative);
        println!(
            "  clipping_non_expansive: {}",
            report.clipping_non_expansive
        );
        println!("  max_exceeds_cap:        {}", report.max_exceeds_cap);
    }
}
