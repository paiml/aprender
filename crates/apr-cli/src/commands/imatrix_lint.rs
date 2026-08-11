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

use super::lint_error::{load_json_observation, LintError};
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

/// Falsify id of the calibration/eval leakage invariant.
///
/// It used to share `FALSIFY-CRUX-B-07-001` with the perplexity-improvement
/// gate, so a `--json` consumer keying on `falsify_id` could not tell the two
/// apart — and a leakage failure was filed under the improvement test's id.
/// Every sibling envelope-B lint (awq, gptq, fp8, embeddings) already stamps a
/// distinct id per gate; contracts/crux-B-07-v1.yaml now carries -004.
const LEAKAGE_FALSIFY_ID: &str = "FALSIFY-CRUX-B-07-004";

#[derive(Debug, Clone, serde::Serialize)]
struct GateReport {
    gate: &'static str,
    falsify_id: &'static str,
    outcome: String,
    passed: bool,
}

pub fn run(args: ImatrixLintArgs) -> Result<(), LintError> {
    let obs: Value = load_json_observation(&args.observation_file, "FALSIFY-CRUX-B-07")?;

    let (reports, failures) = build_reports(&obs);

    if reports.is_empty() {
        // No section present at all: the run judged nothing, which must not be
        // reported as a pass. UnusableInput, not GateFailed — nothing was
        // rejected, there was simply nothing to reject.
        return Err(LintError::unusable(
            "FALSIFY-CRUX-B-07: observation has none of improvement/leakage/flags/provenance",
        ));
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
        return Err(LintError::gate_failed(failures.join("\n")));
    }
    Ok(())
}

/// Dispatch every gate present in the observation, returning the reports and
/// the failure strings. Split out of [`run`] so tests can assert on the
/// `falsify_id` stamps directly instead of scraping stdout.
fn build_reports(obs: &Value) -> (Vec<GateReport>, Vec<String>) {
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

    (reports, failures)
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

/// Canonicalise one `*_hashes` element into the string used for set
/// membership. A hash may legitimately be serialised as a JSON string or as
/// a JSON number; both are accepted and compared on the same footing so that
/// `[1, 2]` vs `[1, 2]` is detected as leakage rather than silently dropped.
/// Composite values (array/object/null) are not hashes and are rejected.
fn canonical_hash(el: &Value) -> Result<String, String> {
    match el {
        Value::String(s) => Ok(s.clone()),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        other => Err(format!("element is not a scalar hash: {other}")),
    }
}

/// Extract one hash set, refusing to invent an empty set out of missing or
/// wrongly-typed evidence. Returning `Err` makes the gate FAIL: an
/// unreadable observation is unknown, not disjoint.
fn extract_hash_set(v: &Value, field: &str) -> Result<BTreeSet<String>, String> {
    let Some(raw) = v.get(field) else {
        return Err(format!("missing `{field}`"));
    };
    let Some(arr) = raw.as_array() else {
        return Err(format!("`{field}` is not an array"));
    };
    let mut out = BTreeSet::new();
    for (i, el) in arr.iter().enumerate() {
        match canonical_hash(el) {
            Ok(h) => {
                out.insert(h);
            }
            Err(why) => return Err(format!("`{field}`[{i}]: {why}")),
        }
    }
    Ok(out)
}

fn run_leakage_gate(v: &Value) -> (GateReport, Option<String>) {
    // A non-object `leakage`, a missing array, or non-scalar elements used to
    // collapse to two empty sets and be reported as `disjoint (|calib|=0,
    // |eval|=0)` — a statement about data that was never read.
    if !v.is_object() {
        return leakage_unreadable(format!("`leakage` is not an object: {v}"));
    }
    let calib = match extract_hash_set(v, "calib_hashes") {
        Ok(s) => s,
        Err(why) => return leakage_unreadable(why),
    };
    let eval = match extract_hash_set(v, "eval_hashes") {
        Ok(s) => s,
        Err(why) => return leakage_unreadable(why),
    };
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
            "{LEAKAGE_FALSIFY_ID} leakage invariant violated: {desc}"
        ))
    };
    (
        GateReport {
            gate: "leakage",
            falsify_id: LEAKAGE_FALSIFY_ID,
            outcome: desc,
            passed: disjoint,
        },
        err,
    )
}

/// Build the FAIL report used when the `leakage` section cannot be read.
fn leakage_unreadable(why: String) -> (GateReport, Option<String>) {
    let desc = format!("leakage evidence unreadable: {why}");
    let err = Some(format!(
        "{LEAKAGE_FALSIFY_ID} leakage gate could not be evaluated: {desc}"
    ));
    (
        GateReport {
            gate: "leakage",
            falsify_id: LEAKAGE_FALSIFY_ID,
            outcome: desc,
            passed: false,
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
        let err = run(args).unwrap_err().to_string();
        // The whole *-lint family reports a missing input identically:
        // "File not found: <path>" with exit 3 (commands::lint_error).
        assert!(err.contains("File not found"), "got: {err}");
        assert!(err.contains("/no/such/im.json"), "got: {err}");
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs("  ");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("observation file is empty"));
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("##");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("not valid JSON"));
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err().to_string();
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
        let err = run(args_for(&f)).unwrap_err().to_string();
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
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-B-07-004"), "got: {err}");
        assert!(err.contains("leakage"));
    }

    /// FALSIFY-CRUX-B-07-004 — the four gates must be distinguishable by
    /// `falsify_id`. `improvement` and `leakage` both stamped -001, so a
    /// consumer keying on the id could not tell which invariant broke.
    #[test]
    fn every_gate_carries_a_distinct_falsify_id() {
        let obs: Value = serde_json::from_str(
            r#"{"improvement": {"ppl_naive": 100.0, "ppl_calib": 90.0},
                "leakage": {"calib_hashes": ["a"], "eval_hashes": ["b"]},
                "flags": {"argv": ["quantize"], "expected_path": null},
                "provenance": {"expected_sha256": "ab", "recorded": "ab"}}"#,
        )
        .expect("obs parses");
        let (reports, failures) = build_reports(&obs);
        assert_eq!(reports.len(), 4, "all four gates must run");
        assert!(failures.is_empty(), "all-pass body: {failures:?}");
        let ids: BTreeSet<&str> = reports.iter().map(|r| r.falsify_id).collect();
        assert_eq!(
            ids.len(),
            reports.len(),
            "duplicate falsify_id across gates: {:?}",
            reports
                .iter()
                .map(|r| (r.gate, r.falsify_id))
                .collect::<Vec<_>>()
        );
    }

    /// The id a failing leakage gate stamps must be the leakage id, not the
    /// perplexity-improvement one: they fail for unrelated reasons.
    #[test]
    fn leakage_failure_is_not_filed_under_the_improvement_id() {
        let obs: Value =
            serde_json::from_str(r#"{"leakage": {"calib_hashes": ["a"], "eval_hashes": ["a"]}}"#)
                .expect("obs parses");
        let (reports, failures) = build_reports(&obs);
        assert_eq!(reports[0].falsify_id, "FALSIFY-CRUX-B-07-004");
        assert!(!reports[0].passed);
        assert_eq!(failures.len(), 1);
        assert!(
            !failures[0].contains("FALSIFY-CRUX-B-07-001"),
            "leakage failure filed under the improvement id: {}",
            failures[0]
        );
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
        let err = run(args_for(&f)).unwrap_err().to_string();
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
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-B-07-003"));
    }

    #[test]
    fn provenance_gate_missing_input_fails() {
        let f = write_obs(r#"{"provenance": {}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-B-07-003"));
    }

    // ── leakage gate: degenerate evidence must not read as "disjoint" ─────
    //
    // Dogfood 0.63.0 #2377 finding 4. The gate reported
    // `[PASS] leakage ...: disjoint (|calib|=0, |eval|=0)` for bodies it had
    // never actually read, because `filter_map(as_str)` on a non-object or on
    // integer-typed hashes yields two empty sets.

    #[test]
    fn leakage_overlapping_integer_hashes_are_detected() {
        // Same overlap as the string case below, only serialized as numbers.
        // Pre-fix this PASSED with "disjoint (|calib|=0, |eval|=0)".
        let f = write_obs(r#"{"leakage": {"calib_hashes": [1, 2], "eval_hashes": [1, 2]}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(
            err.contains("leakage invariant violated"),
            "integer-typed overlapping hashes must be leakage, got: {err}"
        );
        assert!(err.contains('1'), "overlap must be named: {err}");
    }

    #[test]
    fn leakage_overlapping_string_hashes_are_detected() {
        // Control: the type the gate always handled correctly.
        let f = write_obs(r#"{"leakage": {"calib_hashes": ["a"], "eval_hashes": ["a"]}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("leakage invariant violated"), "got: {err}");
    }

    #[test]
    fn leakage_scalar_section_is_unreadable_not_disjoint() {
        let f = write_obs(r#"{"leakage": "nonsense"}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(
            err.contains("could not be evaluated"),
            "a non-object leakage section must fail the gate, got: {err}"
        );
    }

    #[test]
    fn leakage_missing_eval_hashes_is_unreadable() {
        let f = write_obs(r#"{"leakage": {"calib_hashes": ["a"]}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("eval_hashes"), "got: {err}");
    }

    #[test]
    fn leakage_non_scalar_element_is_unreadable() {
        let f = write_obs(r#"{"leakage": {"calib_hashes": [["a"]], "eval_hashes": ["b"]}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("not a scalar hash"), "got: {err}");
    }

    #[test]
    fn leakage_genuinely_disjoint_still_passes() {
        // The fix must not turn a real pass into a failure.
        let f = write_obs(r#"{"leakage": {"calib_hashes": ["a"], "eval_hashes": ["b"]}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn leakage_empty_but_explicit_sets_are_disjoint() {
        let f = write_obs(r#"{"leakage": {"calib_hashes": [], "eval_hashes": []}}"#);
        assert!(run(args_for(&f)).is_ok());
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
