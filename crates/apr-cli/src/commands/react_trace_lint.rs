//! `apr react-trace-lint` — CRUX-I-06 ReAct agent trace gate.
//!
//! Reads a captured `apr agent --trace OUT.json` body and dispatches the
//! pure classifiers in `react_trace_classifier`. Exits non-zero on any
//! failure.
//!
//! Spec: `contracts/crux-I-06-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::react_trace_classifier::{
    classify_iteration_bound, classify_scratchpad_grammar, classify_termination, ReactBoundOutcome,
    ReactGrammarOutcome, ReactTerminationOutcome,
};
use crate::error::{CliError, Result};

pub(crate) fn run(
    trace_file: &Path,
    max_iterations: Option<i64>,
    require_grammar: bool,
    json: bool,
) -> Result<()> {
    if !trace_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(trace_file)));
    }
    let body_text = std::fs::read_to_string(trace_file)?;
    let body: Value = serde_json::from_str(&body_text).map_err(|e| {
        CliError::InvalidInput(format!(
            "apr react-trace-lint: failed to parse JSON from {}: {e}",
            trace_file.display()
        ))
    })?;

    let term = classify_termination(&body);
    let bound = max_iterations.map(|n| classify_iteration_bound(&body, n));
    let grammar = if require_grammar {
        let sp = body.get("scratchpad").and_then(Value::as_str).unwrap_or("");
        Some(classify_scratchpad_grammar(sp))
    } else {
        None
    };

    print_report(trace_file, &term, bound.as_ref(), grammar.as_ref(), json);

    if !matches!(term, ReactTerminationOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "react-trace-lint termination gate rejected body: {term:?}"
        )));
    }
    if let Some(o) = &bound {
        if !matches!(o, ReactBoundOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "react-trace-lint iteration-bound gate rejected body: {o:?}"
            )));
        }
    }
    if let Some(o) = &grammar {
        if !matches!(o, ReactGrammarOutcome::Ok { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "react-trace-lint scratchpad-grammar gate rejected body: {o:?}"
            )));
        }
    }
    Ok(())
}

fn print_report(
    path: &Path,
    term: &ReactTerminationOutcome,
    bound: Option<&ReactBoundOutcome>,
    grammar: Option<&ReactGrammarOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "file": path.display().to_string(),
            "termination": term,
            "iteration_bound": bound.map(|o| format!("{o:?}")),
            "scratchpad_grammar": grammar.map(|o| format!("{o:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("react-trace-lint report for {}", path.display());
    println!("  termination       : {term:?}");
    if let Some(o) = bound {
        println!("  iteration_bound   : {o:?}");
    }
    if let Some(o) = grammar {
        println!("  scratchpad_grammar: {o:?}");
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
        let err = run(Path::new("/no/such/react.json"), None, false, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn invalid_json_is_invalid_format() {
        let f = w("xx");
        let err = run(f.path(), None, false, false).unwrap_err();
        assert!(matches!(err, CliError::InvalidInput(_)));
    }
    #[test]
    fn empty_object_runs() {
        let f = w("{}");
        let _ = run(f.path(), None, false, true);
    }
}
