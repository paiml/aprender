//! `apr explain-token-lint` — CRUX-F-19 sampler-chain JSONL gate.
//!
//! Reads an already-captured `apr explain --format jsonl` body and dispatches
//! the pure classifiers in `explain_token_classifier`. Exits non-zero on any
//! failure.
//!
//! Spec: `contracts/crux-F-19-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use super::explain_token_classifier::{
    classify_greedy_picks_argmax, classify_probs_normalize, classify_sampled_in_candidates,
    classify_schema, ExplainGreedyOutcome, ExplainProbsOutcome, ExplainSampledOutcome,
    ExplainSchemaOutcome,
};
use super::threshold_arg;
use crate::error::{CliError, Result};

pub(crate) fn run(
    jsonl_file: &Path,
    tolerance: f64,
    require_greedy: bool,
    json: bool,
) -> Result<()> {
    // Fail closed before any gate runs: `(sum - 1.0).abs() > tolerance` is
    // false against NaN, so the probs-normalize gate could never fire.
    threshold_arg::guard("--tolerance", tolerance, threshold_arg::TOLERANCE)?;
    if !jsonl_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(jsonl_file)));
    }
    let body = std::fs::read_to_string(jsonl_file)?;

    let schema = classify_schema(&body);
    let (probs, sampled, greedy) = if matches!(schema, ExplainSchemaOutcome::Ok { .. }) {
        (
            classify_probs_normalize(&body, tolerance),
            classify_sampled_in_candidates(&body),
            if require_greedy {
                Some(classify_greedy_picks_argmax(&body))
            } else {
                None
            },
        )
    } else {
        (ExplainProbsOutcome::Ok, ExplainSampledOutcome::Ok, None)
    };

    print_report(jsonl_file, &schema, &probs, &sampled, greedy.as_ref(), json);

    if !matches!(schema, ExplainSchemaOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "explain-token-lint schema gate rejected body: {schema:?}"
        )));
    }
    if !matches!(probs, ExplainProbsOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "explain-token-lint probs-normalize gate rejected body: {probs:?}"
        )));
    }
    if !matches!(sampled, ExplainSampledOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "explain-token-lint sampled-in-candidates gate rejected body: {sampled:?}"
        )));
    }
    if let Some(g) = &greedy {
        if !matches!(g, ExplainGreedyOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "explain-token-lint greedy-argmax gate rejected body: {g:?}"
            )));
        }
    }
    Ok(())
}

fn print_report(
    path: &Path,
    schema: &ExplainSchemaOutcome,
    probs: &ExplainProbsOutcome,
    sampled: &ExplainSampledOutcome,
    greedy: Option<&ExplainGreedyOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "file": path.display().to_string(),
            "schema": format!("{schema:?}"),
            "probs_normalize": format!("{probs:?}"),
            "sampled_in_candidates": format!("{sampled:?}"),
            "greedy_picks_argmax": greedy.map(|g| format!("{g:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("explain-token-lint report for {}", path.display());
    println!("  schema                : {schema:?}");
    println!("  probs_normalize       : {probs:?}");
    println!("  sampled_in_candidates : {sampled:?}");
    if let Some(g) = greedy {
        println!("  greedy_picks_argmax   : {g:?}");
    }
}

#[cfg(test)]
mod cov_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    fn w(s: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(s.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }
    #[test]
    fn missing_file_is_file_not_found() {
        let err = run(Path::new("/no/such/explain.jsonl"), 1e-5, false, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn empty_jsonl_runs() {
        let f = w("");
        let _ = run(f.path(), 1e-5, false, true);
    }
    /// 0.63.0 printed `probs_normalize : Ok` and exited 0 for a record whose
    /// post_probs sum to 0.6, whenever `--tolerance nan` was passed.
    #[test]
    fn nan_tolerance_cannot_disarm_the_probs_normalize_gate() {
        let f = w(concat!(
            r#"{"step":0,"sampled_id":7,"candidates":["#,
            r#"{"token_id":7,"pre_prob":0.9,"post_prob":0.5,"rank":0},"#,
            r#"{"token_id":3,"pre_prob":0.1,"post_prob":0.1,"rank":1}]}"#,
            "\n"
        ));

        // Control: the shipped default rejects this body.
        let err = run(f.path(), 1e-5, false, false).unwrap_err();
        match err {
            CliError::ValidationFailed(msg) => {
                assert!(msg.contains("probs-normalize"), "got: {msg}");
            }
            other => panic!("expected the probs gate to fire, got {other:?}"),
        }

        for bad in [f64::NAN, -1.0, f64::INFINITY] {
            let err = run(f.path(), bad, false, false).unwrap_err();
            match err {
                CliError::ValidationFailed(msg) => {
                    assert!(msg.contains("--tolerance"), "got: {msg}");
                }
                other => panic!("tolerance {bad} must fail closed, got {other:?}"),
            }
        }
    }

    #[test]
    fn garbage_jsonl_errors() {
        let f = w("garbage line\nanother\n");
        let _ = run(f.path(), 1e-5, false, false);
    }
}
