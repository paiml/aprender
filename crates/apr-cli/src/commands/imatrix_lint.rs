//! CRUX-B-07 — `apr imatrix-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the imatrix classifiers in `imatrix_classifier.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "improvement": {
//!     "ppl_naive":  100.0,
//!     "ppl_calib":   90.0,
//!     "threshold":   0.005
//!   },
//!   "leakage": {
//!     "calib_hashes": ["a", "b"],
//!     "eval_hashes":  ["c", "d"]
//!   },
//!   "flags": {
//!     "argv":          ["quantize", "model.apr", "--imatrix", "calib.jsonl"],
//!     "expected_path": "calib.jsonl"      // or null for "expected absent"
//!   },
//!   "provenance": {
//!     "calib_bytes_utf8": "calib-v1",     // OR
//!     "expected_sha256":  "abc...",       // one of these is required
//!     "recorded":         "abc..."        // Option<String>
//!   }
//! }
//! ```
//!
//! Any missing top-level key is skipped. Non-zero exit + FALSIFY-CRUX-B-07
//! stderr stamp on any failing gate.

use crate::commands::imatrix_classifier::{
    calibration_eval_disjoint, classify_imatrix_improvement, compute_provenance_sha256,
    parse_imatrix_flag, validate_recorded_provenance, ImprovementOutcome, ProvenanceOutcome,
    MIN_PPL_IMPROVEMENT,
};
use serde_json::Value;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ImatrixLintArgs {
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

pub fn run(args: ImatrixLintArgs) -> Result<(), String> {
    let path = Path::new(&args.observation_file);
    if !path.exists() {
        return Err(format!(
            "FALSIFY-CRUX-B-07: observation file not found: {}",
            args.observation_file
        ));
    }
    let raw = fs::read_to_string(path)
        .map_err(|e| format!("FALSIFY-CRUX-B-07: failed to read observation: {e}"))?;
    if raw.trim().is_empty() {
        return Err("FALSIFY-CRUX-B-07: observation file is empty".to_string());
    }
    let obs: Value = serde_json::from_str(&raw)
        .map_err(|e| format!("FALSIFY-CRUX-B-07: observation is not valid JSON: {e}"))?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(v) = obs.get("improvement") {
        let (report, err) = run_improvement_gate(v);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("leakage") {
        let (report, err) = run_leakage_gate(v);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("flags") {
        let (report, err) = run_flags_gate(v);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(v) = obs.get("provenance") {
        let (report, err) = run_provenance_gate(v);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err(
            "FALSIFY-CRUX-B-07: observation has none of improvement/leakage/flags/provenance"
                .into(),
        );
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-B-07",
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

fn run_improvement_gate(v: &Value) -> (GateReport, Option<String>) {
    let ppl_naive = v.get("ppl_naive").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let ppl_calib = v.get("ppl_calib").and_then(|x| x.as_f64()).unwrap_or(0.0);
    let threshold = v
        .get("threshold")
        .and_then(|x| x.as_f64())
        .unwrap_or(MIN_PPL_IMPROVEMENT);
    let outcome = classify_imatrix_improvement(ppl_naive, ppl_calib, threshold);
    let (passed, desc) = match outcome {
        ImprovementOutcome::Improved { delta } => (
            true,
            format!("Δ={delta:.4} >= {threshold} (naive={ppl_naive}, calib={ppl_calib})"),
        ),
        ImprovementOutcome::Insufficient { delta, threshold } => (
            false,
            format!("Δ={delta:.4} < {threshold} (naive={ppl_naive}, calib={ppl_calib})"),
        ),
    };
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-07-001 improvement gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "improvement",
            falsify_id: "FALSIFY-CRUX-B-07-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_leakage_gate(v: &Value) -> (GateReport, Option<String>) {
    let calib: BTreeSet<String> = v
        .get("calib_hashes")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let eval: BTreeSet<String> = v
        .get("eval_hashes")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|s| s.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let disjoint = calibration_eval_disjoint(&calib, &eval);
    let overlap: Vec<&String> = calib.intersection(&eval).collect();
    let desc = if disjoint {
        format!("disjoint (|calib|={}, |eval|={})", calib.len(), eval.len())
    } else {
        format!(
            "leakage detected: {} overlapping item(s): {:?}",
            overlap.len(),
            overlap
        )
    };
    let err = if disjoint {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-07-001 leakage invariant violated: {desc}"
        ))
    };
    (
        GateReport {
            gate: "leakage",
            falsify_id: "FALSIFY-CRUX-B-07-001",
            outcome: desc,
            passed: disjoint,
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
    let got = parse_imatrix_flag(&argv);
    let expected = v
        .get("expected_path")
        .and_then(|x| if x.is_null() { None } else { x.as_str() })
        .map(|s| s.to_string());
    let passed = got == expected;
    let desc = format!("expected={expected:?} got={got:?}");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-B-07-002 flags gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "flags",
            falsify_id: "FALSIFY-CRUX-B-07-002",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_provenance_gate(v: &Value) -> (GateReport, Option<String>) {
    let expected = if let Some(bytes) = v.get("calib_bytes_utf8").and_then(|x| x.as_str()) {
        compute_provenance_sha256(bytes.as_bytes())
    } else if let Some(s) = v.get("expected_sha256").and_then(|x| x.as_str()) {
        s.to_string()
    } else {
        return (
            GateReport {
                gate: "provenance",
                falsify_id: "FALSIFY-CRUX-B-07-003",
                outcome: "missing expected sha256 input".to_string(),
                passed: false,
            },
            Some(
                "FALSIFY-CRUX-B-07-003 provenance gate failed: observation needs either calib_bytes_utf8 or expected_sha256"
                    .to_string(),
            ),
        );
    };
    let recorded = v.get("recorded").and_then(|x| x.as_str());
    let outcome = validate_recorded_provenance(recorded, &expected);
    let (passed, desc) = match &outcome {
        ProvenanceOutcome::Match => (true, format!("match (sha256={expected})")),
        ProvenanceOutcome::Missing => (false, "no imatrix_source_sha256 recorded".to_string()),
        ProvenanceOutcome::Mismatch { recorded, expected } => (
            false,
            format!("mismatch: recorded={recorded} expected={expected}"),
        ),
    };
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-B-07-003 provenance gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "provenance",
            falsify_id: "FALSIFY-CRUX-B-07-003",
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

    fn args_for(f: &NamedTempFile) -> ImatrixLintArgs {
        ImatrixLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = ImatrixLintArgs {
            observation_file: "/no/such/im.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-07"));
        assert!(err.contains("not found"));
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs("  ");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("##");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("none of improvement/leakage/flags/provenance"));
    }

    #[test]
    fn improvement_gate_better_ppl_passes() {
        // delta = (100-90)/100 = 0.1 >= 0.005.
        let f = write_obs(r#"{"improvement": {"ppl_naive": 100.0, "ppl_calib": 90.0}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn improvement_gate_no_gain_fails() {
        let f = write_obs(r#"{"improvement": {"ppl_naive": 100.0, "ppl_calib": 100.0}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-07-001"));
    }

    #[test]
    fn leakage_gate_disjoint_passes() {
        let f =
            write_obs(r#"{"leakage": {"calib_hashes": ["a", "b"], "eval_hashes": ["c", "d"]}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn leakage_gate_overlap_fails() {
        let f =
            write_obs(r#"{"leakage": {"calib_hashes": ["a", "b"], "eval_hashes": ["b", "c"]}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-07-001"));
        assert!(err.contains("leakage"));
    }

    #[test]
    fn flags_gate_present_path_passes() {
        let f = write_obs(
            r#"{"flags": {"argv": ["quantize", "model.apr", "--imatrix", "calib.jsonl"], "expected_path": "calib.jsonl"}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn flags_gate_absent_path_passes_when_null_expected() {
        let f =
            write_obs(r#"{"flags": {"argv": ["quantize", "model.apr"], "expected_path": null}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn flags_gate_mismatch_fails() {
        let f = write_obs(r#"{"flags": {"argv": ["quantize"], "expected_path": "calib.jsonl"}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-07-002"));
    }

    #[test]
    fn provenance_gate_matching_sha_passes() {
        let sha = "a".repeat(64);
        let obs = serde_json::json!({
            "provenance": { "expected_sha256": sha, "recorded": sha }
        });
        let f = write_obs(&obs.to_string());
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn provenance_gate_mismatch_fails() {
        let obs = serde_json::json!({
            "provenance": { "expected_sha256": "a".repeat(64), "recorded": "b".repeat(64) }
        });
        let f = write_obs(&obs.to_string());
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-07-003"));
    }

    #[test]
    fn provenance_gate_missing_input_fails() {
        let f = write_obs(r#"{"provenance": {}}"#);
        let err = run(args_for(&f)).unwrap_err();
        assert!(err.contains("FALSIFY-CRUX-B-07-003"));
    }

    #[test]
    fn json_mode_ok() {
        let f = write_obs(r#"{"improvement": {"ppl_naive": 50.0, "ppl_calib": 40.0}}"#);
        let args = ImatrixLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
