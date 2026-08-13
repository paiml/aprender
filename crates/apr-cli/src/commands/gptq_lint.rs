//! CRUX-B-09 — `apr gptq-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the three GPTQ classifiers in `gptq_classifier.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "compression": {
//!     "fp16_bytes":  1_000_000_000,
//!     "gptq_bytes":    250_000_000,
//!     "max_ratio":          0.30
//!   },
//!   "cosine": {
//!     "pairs": [
//!       { "fp16": [..], "gptq": [..] },
//!       ...
//!     ],
//!     "threshold": 0.98
//!   },
//!   "flags": {
//!     "argv":             ["model.safetensors", "--scheme", "int4", "-o", "out.apr"],
//!     "expected_outcome": "accepted"   // accepted (alias: ok) | rejected
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-B-09
//! stderr stamp on any failing gate.
//!
//! The `flags` gate asks the SHIPPED clap parser (`commands::quantize_flag_parity`)
//! whether `apr quantize <argv>` is accepted — it used to ask a hand-rolled
//! matcher that understood `--method`/`--bits`/`--group-size`, none of which
//! `apr quantize` has ever taken (aprender#2377 finding 2).

use super::lint_error::{load_json_observation, LintError};
use crate::commands::gptq_classifier::{
    classify_compression_ratio, classify_mean_cosine, CompressionOutcome, CosineFidelity,
    GPTQ_MAX_COMPRESSION_RATIO, GPTQ_MIN_MEAN_COSINE,
};
use crate::commands::quantize_flag_parity::evaluate_flags_observation;
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct GptqLintArgs {
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

pub fn run(args: GptqLintArgs) -> Result<(), LintError> {
    let obs: Value = load_json_observation(&args.observation_file, "FALSIFY-CRUX-B-09")?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(cmp) = obs.get("compression") {
        let (report, err) = run_compression_gate(cmp);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(cos) = obs.get("cosine") {
        let (report, err) = run_cosine_gate(cos);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(fl) = obs.get("flags") {
        let (report, err) = run_flags_gate(fl)?;
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err(LintError::unusable(
            "FALSIFY-CRUX-B-09: observation has none of compression/cosine/flags",
        ));
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-B-09",
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

fn run_compression_gate(v: &Value) -> (GateReport, Option<String>) {
    let fp16 = v.get("fp16_bytes").and_then(|x| x.as_u64()).unwrap_or(0);
    let gptq = v.get("gptq_bytes").and_then(|x| x.as_u64()).unwrap_or(0);
    let max_ratio = v
        .get("max_ratio")
        .and_then(|x| x.as_f64())
        .unwrap_or(GPTQ_MAX_COMPRESSION_RATIO);
    let outcome = classify_compression_ratio(fp16, gptq, max_ratio);
    let (passed, desc) = match outcome {
        CompressionOutcome::Compressed { ratio } => (
            true,
            format!("ratio={ratio:.4} (max={max_ratio}, fp16={fp16}, gptq={gptq})"),
        ),
        CompressionOutcome::Insufficient { ratio, max_ratio } => (
            false,
            format!("ratio={ratio:.4} > max={max_ratio} (fp16={fp16}, gptq={gptq})"),
        ),
    };
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-09-001 compression gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "compression",
            falsify_id: "FALSIFY-CRUX-B-09-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn read_f64_array(v: &Value) -> Vec<f64> {
    v.as_array()
        .map(|a| a.iter().filter_map(|n| n.as_f64()).collect())
        .unwrap_or_default()
}

fn run_cosine_gate(v: &Value) -> (GateReport, Option<String>) {
    let threshold = v
        .get("threshold")
        .and_then(|x| x.as_f64())
        .unwrap_or(GPTQ_MIN_MEAN_COSINE);
    let Some(pairs_val) = v.get("pairs").and_then(|x| x.as_array()) else {
        let desc = "cosine.pairs missing".to_string();
        return (
            GateReport {
                gate: "cosine",
                falsify_id: "FALSIFY-CRUX-B-09-002",
                outcome: desc.clone(),
                passed: false,
            },
            Some(format!("FALSIFY-CRUX-B-09-002 cosine gate failed: {desc}")),
        );
    };

    let vecs: Vec<(Vec<f64>, Vec<f64>)> = pairs_val
        .iter()
        .map(|p| {
            (
                p.get("fp16").map(read_f64_array).unwrap_or_default(),
                p.get("gptq").map(read_f64_array).unwrap_or_default(),
            )
        })
        .collect();
    let borrowed: Vec<(&[f64], &[f64])> = vecs
        .iter()
        .map(|(a, b)| (a.as_slice(), b.as_slice()))
        .collect();
    let fidelity = classify_mean_cosine(&borrowed, threshold);

    let (passed, desc) = match fidelity {
        CosineFidelity::Ok { mean, n } => {
            (true, format!("mean_cos={mean:.6} >= {threshold} (n={n})"))
        }
        CosineFidelity::Degraded { mean, threshold, n } => {
            (false, format!("mean_cos={mean:.6} < {threshold} (n={n})"))
        }
        CosineFidelity::NoSamples => (false, "no valid pairs (all length-mismatched)".to_string()),
    };
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-09-002 cosine gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "cosine",
            falsify_id: "FALSIFY-CRUX-B-09-002",
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
    const FALSIFY_ID: &str = "FALSIFY-CRUX-B-09-003";
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

    fn args_for(f: &NamedTempFile) -> GptqLintArgs {
        GptqLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = GptqLintArgs {
            observation_file: "/no/such/gptq.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err().to_string();
        // The whole *-lint family reports a missing input identically:
        // "File not found: <path>" with exit 3 (commands::lint_error).
        assert!(err.contains("File not found"), "got: {err}");
        assert!(err.contains("/no/such/gptq.json"), "got: {err}");
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs(" ");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("not-json");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("none of compression/cosine/flags"));
    }

    #[test]
    fn compression_gate_compressed_passes() {
        let f = write_obs(r#"{"compression": {"fp16_bytes": 1000000, "gptq_bytes": 200000}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn compression_gate_insufficient_fails() {
        let f = write_obs(r#"{"compression": {"fp16_bytes": 1000000, "gptq_bytes": 950000}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-B-09-001"));
    }

    #[test]
    fn cosine_gate_identical_vectors_pass() {
        let f = write_obs(
            r#"{"cosine": {"pairs": [{"fp16": [1.0, 2.0, 3.0], "gptq": [1.0, 2.0, 3.0]}]}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn cosine_gate_missing_pairs_fails() {
        let f = write_obs(r#"{"cosine": {}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-B-09-002"));
    }

    #[test]
    fn flags_gate_accepts_a_real_quantize_invocation() {
        let f = write_obs(
            r#"{"flags": {"argv": ["model.safetensors", "--scheme", "int4", "-o", "out.apr"], "expected_outcome": "accepted"}}"#,
        );
        run(args_for(&f)).expect("the shipped `apr quantize` accepts this argv");
    }

    /// FALSIFY-LINTFLAG-004 (gptq half). Before aprender#2377 finding 2 this
    /// observation PASSED: `parse_gptq_flags` + `validate_gptq_flags` reported
    /// `Ok { bits: 4, group_size: 128 }` for an argv the shipped `apr quantize`
    /// has never accepted. The gate now runs the real clap parser and refuses.
    #[test]
    fn flags_gate_rejects_the_method_bits_argv_the_shipped_cli_never_took() {
        let f = write_obs(
            r#"{"flags": {"argv": ["--method", "gptq", "--bits", "4", "--group-size", "128"], "expected_outcome": "ok"}}"#,
        );
        let err = run(args_for(&f))
            .expect_err("shipped `apr quantize` has no --method/--bits/--group-size")
            .to_string();
        assert!(err.contains("FALSIFY-CRUX-B-09-003"), "got: {err}");
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
            r#"{"flags": {"argv": ["--method", "gptq", "--bits", "4"], "expected_outcome": "rejected"}}"#,
        );
        run(args_for(&f)).expect("observer asserted the refusal that actually happens");
    }

    /// Exit 4, not exit 5: a label the real parser cannot emit is a broken
    /// capture, not a contract violation by the system under test.
    #[test]
    fn flags_gate_stale_vocabulary_is_unusable_input() {
        let f = write_obs(
            r#"{"flags": {"argv": ["--method", "gptq"], "expected_outcome": "missing_bits"}}"#,
        );
        let err = run(args_for(&f)).expect_err("`missing_bits` is not a clap verdict");
        assert_eq!(err.exit_code_value(), 4, "got: {err}");
    }

    #[test]
    fn json_mode_ok() {
        let f = write_obs(r#"{"compression": {"fp16_bytes": 100, "gptq_bytes": 10}}"#);
        let args = GptqLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
