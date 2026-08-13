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
    // GH-2391: a NaN cap makes `norm > cap` false for every step, so the
    // cap-violation check reports zero violations for telemetry it never read.
    use crate::commands::threshold_arg;
    threshold_arg::guard_opt("--max-grad-norm", max_grad_norm, threshold_arg::TOLERANCE)?;
    threshold_arg::guard(
        "--spike-multiplier",
        spike_multiplier,
        threshold_arg::TOLERANCE,
    )?;

    if !history_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(history_file)));
    }

    let body = std::fs::read_to_string(history_file)?;

    // NOT InvalidFormat: that variant's Display is "Invalid APR format", and a
    // grad-norm history is a JSON telemetry file, not a model. The old wording
    // sent users hunting for a corrupt .apr that was never involved.
    let raw: Vec<RawRecord> = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidInput(format!(
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

// ─── Error wording (dogfood 0.63.0, issue #2374 finding 15) ──────────────────
//
// `apr grad-norm --history-file bad.json` reported a malformed JSON telemetry
// file as "Invalid APR format", sending users hunting for a corrupt model that
// this command never touches.
#[cfg(test)]
mod tests {
    use super::*;

    /// Write `body` to a scratch file and run `grad-norm` against it.
    fn run_on(body: &str, name: &str) -> Result<()> {
        let path =
            std::env::temp_dir().join(format!("apr-2374-gn-{}-{name}.json", std::process::id()));
        std::fs::write(&path, body).expect("scratch write should succeed");
        let outcome = run(&path, None, 5, 3.0, true);
        let _ = std::fs::remove_file(&path);
        outcome
    }

    #[test]
    fn malformed_history_does_not_blame_the_apr_format() {
        let err = run_on("not json", "bare").expect_err("malformed JSON must be rejected");
        let msg = err.to_string();
        assert!(
            !msg.contains("APR"),
            "grad-norm never reads a model; the error must not mention APR: {msg}"
        );
        assert!(
            msg.contains("failed to parse JSON records"),
            "the accurate inner message must survive: {msg}"
        );
    }

    #[test]
    fn malformed_history_keeps_exit_code_4() {
        // The class of failure is unchanged — only the artifact named. Users
        // and CI scripts keying on exit 4 must not break.
        let err =
            run_on("{ this is not json", "trunc").expect_err("malformed JSON must be rejected");
        assert_eq!(err.exit_code(), std::process::ExitCode::from(4));
    }

    #[test]
    fn well_formed_but_empty_history_is_still_a_validation_failure() {
        // Guards against over-broadening: a parseable-but-empty history keeps
        // its own distinct error, it is not swept into the parse failure.
        let err = run_on("[]", "empty").expect_err("an empty history must be rejected");
        let msg = err.to_string();
        assert!(msg.contains("zero records"), "unexpected error: {msg}");
        assert!(
            !msg.contains("failed to parse"),
            "an empty array parses fine: {msg}"
        );
    }
}
