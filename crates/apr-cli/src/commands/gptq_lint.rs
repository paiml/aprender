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
//!     "argv":             ["--method", "gptq", "--bits", "4", "--group-size", "128"],
//!     "expected_outcome": "ok"    // ok | missing_method | wrong_method | invalid_bits | missing_bits | invalid_group_size
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-B-09
//! stderr stamp on any failing gate.

use crate::commands::gptq_classifier::{
    classify_compression_ratio, classify_mean_cosine, parse_gptq_flags, validate_gptq_flags,
    CompressionOutcome, CosineFidelity, GptqFlagValidation, GPTQ_MAX_COMPRESSION_RATIO,
    GPTQ_MIN_MEAN_COSINE,
};
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

pub fn run(args: GptqLintArgs) -> Result<(), String> {
    let path = Path::new(&args.observation_file);
    if !path.exists() {
        return Err(format!(
            "FALSIFY-CRUX-B-09: observation file not found: {}",
            args.observation_file
        ));
    }
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("FALSIFY-CRUX-B-09: failed to read observation: {e}"))?;
    if raw.trim().is_empty() {
        return Err("FALSIFY-CRUX-B-09: observation file is empty".to_string());
    }
    let obs: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("FALSIFY-CRUX-B-09: observation is not valid JSON: {e}"))?;

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
        let (report, err) = run_flags_gate(fl);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err("FALSIFY-CRUX-B-09: observation has none of compression/cosine/flags".into());
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
        return Err(failures.join("\n"));
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

fn run_flags_gate(v: &Value) -> (GateReport, Option<String>) {
    let argv_owned: Vec<String> = v
        .get("argv")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let argv: Vec<&str> = argv_owned.iter().map(|s| s.as_str()).collect();
    let flags = parse_gptq_flags(&argv);
    let validation = validate_gptq_flags(&flags);

    let expected = v
        .get("expected_outcome")
        .and_then(|x| x.as_str())
        .unwrap_or("ok");

    let got = match &validation {
        GptqFlagValidation::Ok { .. } => "ok",
        GptqFlagValidation::MissingMethod => "missing_method",
        GptqFlagValidation::WrongMethod { .. } => "wrong_method",
        GptqFlagValidation::InvalidBits { .. } => "invalid_bits",
        GptqFlagValidation::MissingBits => "missing_bits",
        GptqFlagValidation::InvalidGroupSize { .. } => "invalid_group_size",
    };
    let passed = got == expected;
    let desc = format!("expected={expected} got={got} ({validation:?})");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-09-003 flags gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "flags",
            falsify_id: "FALSIFY-CRUX-B-09-003",
            outcome: desc,
            passed,
        },
        err,
    )
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
        let err = run(args).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-09"));
        assert!(err.contains("not found"));
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs(" ");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("not-json");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err();
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
        let err = run(args_for(&f)).unwrap_err();
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
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-09-002"));
    }

    #[test]
    fn flags_gate_valid_gptq_passes() {
        let f = write_obs(
            r#"{"flags": {"argv": ["--method", "gptq", "--bits", "4", "--group-size", "128"], "expected_outcome": "ok"}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn flags_gate_missing_bits_classified() {
        let f = write_obs(
            r#"{"flags": {"argv": ["--method", "gptq"], "expected_outcome": "missing_bits"}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn flags_gate_mismatch_fails() {
        let f = write_obs(r#"{"flags": {"argv": ["--method", "gptq"], "expected_outcome": "ok"}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-09-003"));
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
