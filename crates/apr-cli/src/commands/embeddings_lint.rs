//! CRUX-C-13 — `apr embeddings-lint` CLI wiring (CRUX-SHIP-001 g2/g3 proof).
//!
//! Dispatches the four pure classifiers in `embeddings_classifier.rs` over a
//! captured JSON observation file:
//!
//! ```jsonc
//! {
//!   "shape":       { "input_len": 3, "hidden_size": 4,
//!                    "data": [{ "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] }, ...] },
//!   "determinism": { "v1": [...], "v2": [...] },
//!   "usage":       { "prompt": 8, "total": 8 },
//!   "flag":        { "argv": ["apr", "serve", "--embeddings-enabled"] }
//! }
//! ```
//!
//! Any missing top-level key is skipped (captured observations may only
//! cover one gate at a time). The CLI exits non-zero on any failing gate
//! and stamps the FALSIFY id in stderr so CI log scrapers can pinpoint
//! the violation.

use crate::commands::embeddings_classifier::{
    classify_determinism, classify_embeddings_response_shape, classify_usage_tokens,
    parse_embeddings_flag, DeterminismOutcome, EmbeddingRow, EmbeddingsFlagOutcome,
    EmbeddingsShapeOutcome, UsageOutcome, EMBEDDINGS_COSINE_TOLERANCE,
};
use crate::commands::lint_input;
use crate::error::CliError;
use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct EmbeddingsLintArgs {
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

pub fn run(args: EmbeddingsLintArgs) -> crate::error::Result<()> {
    let path = Path::new(&args.observation_file);
    let obs = lint_input::read_json_observation("apr embeddings-lint", path)?;

    let mut reports: Vec<GateReport> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    if let Some(shape) = obs.get("shape") {
        let (report, err) = run_shape_gate(shape);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(det) = obs.get("determinism") {
        let (report, err) = run_determinism_gate(det);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(usage) = obs.get("usage") {
        let (report, err) = run_usage_gate(usage);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }
    if let Some(flag) = obs.get("flag") {
        let (report, err) = run_flag_gate(flag);
        reports.push(report);
        if let Some(e) = err {
            failures.push(e);
        }
    }

    if reports.is_empty() {
        return Err(CliError::ValidationFailed(
            "FALSIFY-CRUX-C-13: observation has none of shape/determinism/usage/flag".to_string(),
        ));
    }

    if args.json {
        let payload = serde_json::json!({
            "contract": "CRUX-C-13",
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
        return Err(CliError::ValidationFailed(failures.join("\n")));
    }
    Ok(())
}

/// The three fields the shape gate reasons over must all be present and of
/// the right JSON type. Anything else is unknown evidence, not a pass.
fn require_shape_fields(v: &Value) -> Result<(), String> {
    if !v.is_object() {
        return Err(format!("`shape` is not an object: {v}"));
    }
    for field in ["input_len", "hidden_size"] {
        match v.get(field) {
            None => return Err(format!("missing `{field}`")),
            Some(x) if x.as_u64().is_none() => {
                return Err(format!("`{field}` is not a non-negative integer: {x}"))
            }
            Some(_) => {}
        }
    }
    match v.get("data") {
        None => Err("missing `data`".to_string()),
        Some(x) if !x.is_array() => Err(format!("`data` is not an array: {x}")),
        Some(_) => Ok(()),
    }
}

fn run_shape_gate(v: &Value) -> (GateReport, Option<String>) {
    // `{"shape": {}}` used to default input_len/hidden_size to 0 and `data`
    // to the empty vector, so 0 == 0 and the gate reported
    // `Ok { n_rows: 0 }` — a verdict about evidence that was never present.
    if let Err(why) = require_shape_fields(v) {
        let desc = format!("shape evidence unreadable: {why}");
        let err = Some(format!(
            "FALSIFY-CRUX-C-13-001 shape gate could not be evaluated: {desc}"
        ));
        return (
            GateReport {
                gate: "shape",
                falsify_id: "FALSIFY-CRUX-C-13-001",
                outcome: desc,
                passed: false,
            },
            err,
        );
    }
    let input_len = v.get("input_len").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
    let hidden_size = v.get("hidden_size").and_then(|x| x.as_u64()).unwrap_or(0) as usize;

    let rows_raw: Vec<(u64, Vec<f32>)> = v
        .get("data")
        .and_then(|x| x.as_array())
        .map(|arr| {
            arr.iter()
                .map(|row| {
                    let index = row.get("index").and_then(|x| x.as_u64()).unwrap_or(0);
                    let embedding: Vec<f32> = row
                        .get("embedding")
                        .and_then(|x| x.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|n| n.as_f64().map(|f| f as f32))
                                .collect()
                        })
                        .unwrap_or_default();
                    (index, embedding)
                })
                .collect()
        })
        .unwrap_or_default();

    let rows: Vec<EmbeddingRow<'_>> = rows_raw
        .iter()
        .map(|(i, e)| EmbeddingRow {
            index: *i,
            embedding: e.as_slice(),
        })
        .collect();

    let outcome = classify_embeddings_response_shape(input_len, &rows, hidden_size);
    let passed = matches!(outcome, EmbeddingsShapeOutcome::Ok { .. });
    let desc = format!("{outcome:?}");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-C-13-001 shape gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "shape",
            falsify_id: "FALSIFY-CRUX-C-13-001",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_determinism_gate(v: &Value) -> (GateReport, Option<String>) {
    let v1: Vec<f32> = v
        .get("v1")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|n| n.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_default();
    let v2: Vec<f32> = v
        .get("v2")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|n| n.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_default();
    let outcome = classify_determinism(&v1, &v2, EMBEDDINGS_COSINE_TOLERANCE);
    let passed = matches!(outcome, DeterminismOutcome::Deterministic { .. });
    let desc = format!("{outcome:?}");
    let err = if passed {
        None
    } else {
        Some(format!(
            "FALSIFY-CRUX-C-13-002 determinism gate failed: {desc}"
        ))
    };
    (
        GateReport {
            gate: "determinism",
            falsify_id: "FALSIFY-CRUX-C-13-002",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_usage_gate(v: &Value) -> (GateReport, Option<String>) {
    let prompt = v.get("prompt").and_then(|x| x.as_u64()).unwrap_or(0);
    let total = v.get("total").and_then(|x| x.as_u64()).unwrap_or(0);
    let outcome = classify_usage_tokens(prompt, total);
    let passed = matches!(outcome, UsageOutcome::Ok { .. });
    let desc = format!("{outcome:?}");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-C-13-003 usage gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "usage",
            falsify_id: "FALSIFY-CRUX-C-13-003",
            outcome: desc,
            passed,
        },
        err,
    )
}

fn run_flag_gate(v: &Value) -> (GateReport, Option<String>) {
    let argv: Vec<String> = v
        .get("argv")
        .and_then(|x| x.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|n| n.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    let argv_refs: Vec<&str> = argv.iter().map(String::as_str).collect();
    let expected = v
        .get("expected")
        .and_then(|x| x.as_str())
        .unwrap_or("enabled");
    let outcome = parse_embeddings_flag(&argv_refs);
    let observed = match &outcome {
        EmbeddingsFlagOutcome::Enabled => "enabled",
        EmbeddingsFlagOutcome::Disabled => "disabled",
        EmbeddingsFlagOutcome::MalformedFlag { .. } => "malformed",
    };
    let passed = observed == expected;
    let desc = format!("{outcome:?} (expected={expected}, observed={observed})");
    let err = if passed {
        None
    } else {
        Some(format!("FALSIFY-CRUX-C-13-004 flag gate failed: {desc}"))
    };
    (
        GateReport {
            gate: "flag",
            falsify_id: "FALSIFY-CRUX-C-13-004",
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

    fn args_for(f: &NamedTempFile) -> EmbeddingsLintArgs {
        EmbeddingsLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: false,
        }
    }

    #[test]
    fn missing_file_is_falsify_error() {
        let args = EmbeddingsLintArgs {
            observation_file: "/no/such/emb.json".to_string(),
            json: false,
        };
        let err = run(args).unwrap_err();
        // #2377-8: this used to be exit 1, the same code a *failing falsifier*
        // produced, because `run` returned `Result<(), String>` and dispatch had
        // no class to map. A CI job could not tell the two apart.
        assert_eq!(
            err.exit_code(),
            std::process::ExitCode::from(3),
            "a missing observation file must be exit 3: {err}"
        );
    }

    #[test]
    fn empty_file_is_error() {
        let f = write_obs("  ");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("is empty"), "{err}");
    }

    #[test]
    fn invalid_json_is_error() {
        let f = write_obs("][");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("failed to parse JSON"), "{err}");
        // #2377-9: a captured JSON observation is not an APR model.
        assert!(!err.contains("Invalid APR format"), "{err}");
    }

    #[test]
    fn empty_object_has_no_gates() {
        let f = write_obs("{}");
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("none of shape/determinism/usage/flag"));
    }

    #[test]
    fn shape_gate_well_formed_passes() {
        let f = write_obs(
            r#"{"shape": {"input_len": 2, "hidden_size": 3,
                "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]},
                         {"index": 1, "embedding": [0.4, 0.5, 0.6]}]}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn shape_gate_row_count_mismatch_fails() {
        let f = write_obs(
            r#"{"shape": {"input_len": 5, "hidden_size": 3,
                "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]}}"#,
        );
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-C-13-001"));
    }

    #[test]
    fn determinism_gate_identical_vectors_pass() {
        let f = write_obs(r#"{"determinism": {"v1": [0.1, 0.2, 0.3], "v2": [0.1, 0.2, 0.3]}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn usage_gate_matching_tokens_pass() {
        let f = write_obs(r#"{"usage": {"prompt": 8, "total": 8}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn usage_gate_mismatch_fails() {
        let f = write_obs(r#"{"usage": {"prompt": 8, "total": 9}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("FALSIFY-CRUX-C-13-003"));
    }

    #[test]
    fn flag_gate_enabled_passes() {
        let f = write_obs(
            r#"{"flag": {"argv": ["apr", "serve", "--embeddings-enabled"], "expected": "enabled"}}"#,
        );
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn flag_gate_disabled_default_passes() {
        let f = write_obs(r#"{"flag": {"argv": ["apr", "serve"], "expected": "disabled"}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    // ── shape gate: an empty section is unknown, not Ok ───────────────────
    //
    // Dogfood 0.63.0 #2377 finding 4: `{"shape": {}}` defaulted input_len,
    // hidden_size and data to 0/0/[] and reported `Ok { n_rows: 0 }`.

    #[test]
    fn shape_gate_empty_section_is_unreadable_not_ok() {
        let f = write_obs(r#"{"shape": {}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(
            err.contains("could not be evaluated"),
            "an empty shape section must fail the gate, got: {err}"
        );
    }

    #[test]
    fn shape_gate_missing_data_is_unreadable() {
        let f = write_obs(r#"{"shape": {"input_len": 0, "hidden_size": 4}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("missing `data`"), "got: {err}");
    }

    #[test]
    fn shape_gate_scalar_section_is_unreadable() {
        let f = write_obs(r#"{"shape": "nonsense"}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("not an object"), "got: {err}");
    }

    #[test]
    fn shape_gate_wrong_typed_input_len_is_unreadable() {
        let f = write_obs(r#"{"shape": {"input_len": "2", "hidden_size": 3, "data": []}}"#);
        let err = run(args_for(&f)).unwrap_err().to_string();
        assert!(err.contains("input_len"), "got: {err}");
    }

    #[test]
    fn shape_gate_explicit_zero_rows_still_passes() {
        // A genuine empty-input response is still a pass — the fix rejects
        // absent evidence, not a legitimately empty one.
        let f = write_obs(r#"{"shape": {"input_len": 0, "hidden_size": 4, "data": []}}"#);
        assert!(run(args_for(&f)).is_ok());
    }

    #[test]
    fn json_mode_ok() {
        let f = write_obs(r#"{"usage": {"prompt": 4, "total": 4}}"#);
        let args = EmbeddingsLintArgs {
            observation_file: f.path().to_string_lossy().into_owned(),
            json: true,
        };
        assert!(run(args).is_ok());
    }
}
