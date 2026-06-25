//! CRUX-B-11 — `apr fp8-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the two FP8 (E4M3) classifiers in `fp8_classifier.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "frobenius": {
//!     "original":      [f32, ...],
//!     "reconstructed": [f32, ...],
//!     "threshold":     0.01
//!   },
//!   "capability": { "sm": 90 }
//! }
//! ```
//!
//! Any missing top-level key is skipped. The CLI exits non-zero on any
//! failing gate and stamps the FALSIFY id in stderr.

use crate::commands::fp8_classifier::{
    classify_frobenius_error, classify_sm_capability, CapabilityOutcome, FrobeniusOutcome,
    FP8_MAX_FROBENIUS_REL_ERR,
};
use serde_json::Value;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Fp8LintArgs {
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

pub fn run(args: Fp8LintArgs) -> Result<(), String> {
    let path = Path::new(&args.observation_file);
    if !path.exists() {
        return Err(format!(
            "FALSIFY-CRUX-B-11: observation file not found: {}",
            args.observation_file
        ));
    }
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("FALSIFY-CRUX-B-11: failed to read observation: {e}"))?;
    if raw.trim().is_empty() {
        return Err("FALSIFY-CRUX-B-11: observation file is empty".to_string());
    }
    let obs: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("FALSIFY-CRUX-B-11: observation is not valid JSON: {e}"))?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(frob) = obs.get("frobenius") {
        let (report, err) = run_frobenius_gate(frob);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(cap) = obs.get("capability") {
        let (report, err) = run_capability_gate(cap);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err("FALSIFY-CRUX-B-11: observation has neither frobenius nor capability".into());
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-B-11",
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

fn read_f32_array(v: &Value) -> Vec<f32> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|n| n.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_default()
}

fn run_frobenius_gate(v: &Value) -> (GateReport, Option<String>) {
    let original = v.get("original").map(read_f32_array).unwrap_or_default();
    let reconstructed = v
        .get("reconstructed")
        .map(read_f32_array)
        .unwrap_or_default();
    let threshold = v
        .get("threshold")
        .and_then(|x| x.as_f64())
        .unwrap_or(FP8_MAX_FROBENIUS_REL_ERR);
    let outcome = classify_frobenius_error(&original, &reconstructed, threshold);
    let passed = matches!(outcome, FrobeniusOutcome::Ok { .. });
    let desc = format!("{outcome:?}");
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-11-001 frobenius gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "frobenius",
            falsify_id: "FALSIFY-CRUX-B-11-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_capability_gate(v: &Value) -> (GateReport, Option<String>) {
    let sm = v.get("sm").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let outcome = classify_sm_capability(sm);
    let passed = matches!(outcome, CapabilityOutcome::Capable { .. });
    let desc = format!("{outcome:?}");
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-11-002 capability gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "capability",
            falsify_id: "FALSIFY-CRUX-B-11-002",
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

    fn args_for(f: &NamedTempFile) -> Fp8LintArgs {
        Fp8LintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = Fp8LintArgs {
            observation_file: "/no/such/fp8.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-11"));
        assert!(err.contains("not found"));
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs("");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("nope");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("neither frobenius nor capability"));
    }

    #[test]
    fn frobenius_gate_identical_passes() {
        let f = write_obs(
            r#"{"frobenius": {"original": [1.0, 2.0, 3.0], "reconstructed": [1.0, 2.0, 3.0]}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn frobenius_gate_large_error_fails() {
        let f = write_obs(
            r#"{"frobenius": {"original": [1.0, 2.0, 3.0], "reconstructed": [10.0, 20.0, 30.0]}}"#,
        );
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-11-001"));
    }

    #[test]
    fn capability_gate_hopper_passes() {
        let f = write_obs(r#"{"capability": {"sm": 90}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn capability_gate_old_arch_fails() {
        let f = write_obs(r#"{"capability": {"sm": 80}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-11-002"));
    }

    #[test]
    fn json_mode_ok() {
        let f = write_obs(r#"{"capability": {"sm": 100}}"#);
        let args = Fp8LintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
