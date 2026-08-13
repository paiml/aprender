//! CRUX-B-08 — `apr awq-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the three AWQ classifiers in `awq_classifier.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "quality": {
//!     "p_fp16":    0.50,
//!     "p_awq":     0.45,
//!     "threshold": 0.80
//!   },
//!   "compression": {
//!     "fp16_bytes":  1_000_000_000,
//!     "awq_bytes":     250_000_000,
//!     "max_ratio":           0.30
//!   },
//!   "flags": {
//!     "argv":             ["model.safetensors", "--scheme", "int4", "-o", "out.apr"],
//!     "expected_outcome": "accepted"   // accepted (alias: ok) | rejected
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-B-08
//! stderr stamp on any failing gate.
//!
//! The `flags` gate asks the SHIPPED clap parser (`commands::quantize_flag_parity`)
//! whether `apr quantize <argv>` is accepted — it used to ask a hand-rolled
//! matcher that understood `--method`/`--bits`/`--group-size`, none of which
//! `apr quantize` has ever taken (aprender#2377 finding 2).

use super::lint_error::{load_json_observation, LintError};
use crate::commands::awq_classifier::{
    classify_compression_ratio, classify_quality_retention, CompressionOutcome, QualityRetention,
    AWQ_MAX_COMPRESSION_RATIO, AWQ_MIN_QUALITY_RETENTION,
};
use crate::commands::quantize_flag_parity::evaluate_flags_observation;
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct AwqLintArgs {
    pub observation_file: String,
    pub json: bool,
}

#[derive(Debug, Clone, serde::Serialize)]
struct GateReport {
    gate: &'static str,
    falsify_id: &'static str,
    outcome: String,
    passed: bool,
}

pub fn run(args: AwqLintArgs) -> Result<(), LintError> {
    let obs: Value = load_json_observation(&args.observation_file, "FALSIFY-CRUX-B-08")?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(q) = obs.get("quality") {
        let (report, err) = run_quality_gate(q);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(c) = obs.get("compression") {
        let (report, err) = run_compression_gate(c);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(f) = obs.get("flags") {
        let (report, err) = run_flags_gate(f)?;
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err(LintError::unusable(
            "FALSIFY-CRUX-B-08: observation has none of quality/compression/flags",
        ));
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-B-08",
            "gates": reports,
        });
        println!("{}", serde_json::to_string_pretty(&payload).unwrap());
    } else {
        for r in &reports {
            let tag = if r.passed { "PASS" } else { "FAIL" };
            println!("[{tag}] {} ({}): {}", r.gate, r.falsify_id, r.outcome);
        }
    }

    if !failures.is_empty() {
        return Err(LintError::gate_failed(failures.join("\n")));
    }
    Ok(())
}

fn run_quality_gate(v: &Value) -> (GateReport, Option<String>) {
    let p_fp16 = v.get("p_fp16").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let p_awq = v.get("p_awq").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let threshold = v
        .get("threshold")
        .and_then(|x| x.as_f64())
        .unwrap_or(AWQ_MIN_QUALITY_RETENTION);
    let outcome = classify_quality_retention(p_fp16, p_awq, threshold);
    let (passed, desc) = match outcome {
        QualityRetention::Retained { ratio } => (
            true,
            format!("ratio={ratio:.4} >= {threshold} (p_fp16={p_fp16}, p_awq={p_awq})"),
        ),
        QualityRetention::Degraded { ratio, threshold } => (
            false,
            format!("ratio={ratio:.4} < {threshold} (p_fp16={p_fp16}, p_awq={p_awq})"),
        ),
    };
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-08-001 quality gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "quality",
            falsify_id: "FALSIFY-CRUX-B-08-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_compression_gate(v: &Value) -> (GateReport, Option<String>) {
    let fp16 = v.get("fp16_bytes").and_then(|x| x.as_u64()).unwrap_or(0);
    let awq = v.get("awq_bytes").and_then(|x| x.as_u64()).unwrap_or(0);
    let max_ratio = v
        .get("max_ratio")
        .and_then(|x| x.as_f64())
        .unwrap_or(AWQ_MAX_COMPRESSION_RATIO);
    let outcome = classify_compression_ratio(fp16, awq, max_ratio);
    let (passed, desc) = match outcome {
        CompressionOutcome::Compressed { ratio } => (
            true,
            format!("ratio={ratio:.4} (max={max_ratio}, fp16={fp16}, awq={awq})"),
        ),
        CompressionOutcome::Insufficient { ratio, max_ratio } => (
            false,
            format!("ratio={ratio:.4} > max={max_ratio} (fp16={fp16}, awq={awq})"),
        ),
    };
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-08-003 compression gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "compression",
            falsify_id: "FALSIFY-CRUX-B-08-003",
            outcome: desc,
            passed,
        },
        err,
    )
}

/// The CLI-surface gate: does the SHIPPED `apr quantize` agree with what the
/// observation claims about this argv?
///
/// `Err` (exit 4) means the observation cannot be used — a missing argv, a
/// missing expectation, or an `expected_outcome` written in the vocabulary of
/// the hand-rolled parser this gate no longer owns. `Ok` with a failure string
/// (exit 5) means the observation was usable and the real parser disagreed.
fn run_flags_gate(v: &Value) -> Result<(GateReport, Option<String>), LintError> {
    const FALSIFY_ID: &str = "FALSIFY-CRUX-B-08-002";
    let gate = evaluate_flags_observation(v, FALSIFY_ID).map_err(LintError::unusable)?;
    let err = gate
        .failure
        .map(|m| format!("{FALSIFY_ID} flags gate failed: {m}"));
    Ok((
        GateReport {
            gate: "flags",
            falsify_id: FALSIFY_ID,
            outcome: gate.outcome,
            passed: gate.passed,
        },
        err,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_obs(json: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(json.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    fn args_for(f: &NamedTempFile) -> AwqLintArgs {
        AwqLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = AwqLintArgs {
            observation_file: "/no/such/awq.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err().to_string();
        // The whole *-lint family reports a missing input identically:
        // "File not found: <path>" with exit 3 (commands::lint_error).
        assert!(err.contains("File not found"), "got: {err}");
        assert!(err.contains("/no/such/awq.json"), "got: {err}");
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs("  \n");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("xx");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("none of quality/compression/flags"));
    }

    #[test]
    fn quality_gate_retained_passes() {
        // ratio = 0.85/0.90 = 0.944 >= default 0.80.
        let f = write_obs(r#"{"quality": {"p_fp16": 0.90, "p_awq": 0.85}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn quality_gate_degraded_fails() {
        let f = write_obs(r#"{"quality": {"p_fp16": 0.90, "p_awq": 0.50}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-B-08-001"));
    }

    #[test]
    fn compression_gate_compressed_passes() {
        // ratio = 200000/1000000 = 0.2 <= default 0.30.
        let f = write_obs(r#"{"compression": {"fp16_bytes": 1000000, "awq_bytes": 200000}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn compression_gate_insufficient_fails() {
        let f = write_obs(r#"{"compression": {"fp16_bytes": 1000000, "awq_bytes": 900000}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-B-08-003"));
    }

    #[test]
    fn flags_gate_accepts_a_real_quantize_invocation() {
        let f = write_obs(
            r#"{"flags": {"argv": ["model.safetensors", "--scheme", "int4", "-o", "out.apr"], "expected_outcome": "accepted"}}"#,
        );
        run(args_for(&f)).expect("the shipped `apr quantize` accepts this argv");
    }

    /// FALSIFY-LINTFLAG-004 (awq half). Before aprender#2377 finding 2 this
    /// observation PASSED: `parse_awq_flags` + `validate_awq_flags` reported
    /// `Ok { bits: 4, group_size: 128 }` for an argv the shipped `apr quantize`
    /// has never accepted. The gate now runs the real clap parser and refuses.
    #[test]
    fn flags_gate_rejects_the_method_bits_argv_the_shipped_cli_never_took() {
        let f = write_obs(
            r#"{"flags": {"argv": ["--method", "awq", "--bits", "4", "--group-size", "128"], "expected_outcome": "ok"}}"#,
        );
        let err = run(args_for(&f))
            .expect_err("shipped `apr quantize` has no --method/--bits/--group-size")
            .to_string();
        assert!(err.contains("FALSIFY-CRUX-B-08-002"), "got: {err}");
        assert!(err.contains("REJECTED"), "got: {err}");
        assert!(
            err.contains("--scheme"),
            "the operator must be told which flags `apr quantize` does accept; got: {err}"
        );
    }

    /// An observation may legitimately record that the CLI refuses something.
    #[test]
    fn flags_gate_passes_when_the_observation_expects_a_rejection() {
        let f = write_obs(
            r#"{"flags": {"argv": ["--method", "awq", "--bits", "4"], "expected_outcome": "rejected"}}"#,
        );
        run(args_for(&f)).expect("observer asserted the refusal that actually happens");
    }

    /// Exit 4, not exit 5: a label the real parser cannot emit is a broken
    /// capture, not a contract violation by the system under test.
    #[test]
    fn flags_gate_stale_vocabulary_is_unusable_input() {
        let f = write_obs(
            r#"{"flags": {"argv": ["--bits", "4"], "expected_outcome": "missing_method"}}"#,
        );
        let err = run(args_for(&f)).expect_err("`missing_method` is not a clap verdict");
        assert_eq!(err.exit_code_value(), 4, "got: {err}");
    }

    #[test]
    fn json_mode_ok() {
        let f = write_obs(r#"{"quality": {"p_fp16": 1.0, "p_awq": 1.0}}"#);
        let args = AwqLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
