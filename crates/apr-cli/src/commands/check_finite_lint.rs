//! `apr check-finite-lint` — CRUX-F-11 NaN/Inf activation-check gate.
//!
//! Reads an already-captured `apr trace --check-finite` (error JSON on
//! stderr from a poisoned model) and/or `apr trace --check-finite --list`
//! (layer coverage on stdout) body and dispatches the pure classifiers in
//! `check_finite_classifier`. Exits non-zero on any failure.
//!
//! Spec: `contracts/crux-F-11-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::Path;

use super::check_finite_classifier::{
    classify_error_json, classify_layer_coverage, CheckFiniteCoverageOutcome,
    CheckFiniteErrorOutcome, F11_REQUIRED_OP_PREFIXES,
};
use super::lint_input;
use crate::error::{CliError, Result};

pub(crate) fn run(
    error_file: Option<&Path>,
    list_file: Option<&Path>,
    min_layers: usize,
    json: bool,
) -> Result<()> {
    if error_file.is_none() && list_file.is_none() {
        return Err(CliError::ValidationFailed(
            "apr check-finite-lint: at least one of --error-file or --list-file is required"
                .to_string(),
        ));
    }

    let err_outcome = match error_file {
        Some(p) => Some(classify_error_json(&lint_input::read_json_observation(
            "apr check-finite-lint",
            p,
        )?)),
        None => None,
    };

    let cov_outcome = match list_file {
        Some(p) => Some(classify_layer_coverage(
            &lint_input::read_json_observation("apr check-finite-lint", p)?,
            min_layers,
            F11_REQUIRED_OP_PREFIXES,
        )),
        None => None,
    };

    print_report(
        error_file,
        list_file,
        err_outcome.as_ref(),
        cov_outcome.as_ref(),
        json,
    );

    if let Some(o) = &err_outcome {
        if !matches!(o, CheckFiniteErrorOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "check-finite-lint error-json gate rejected body: {o:?}"
            )));
        }
    }
    if let Some(o) = &cov_outcome {
        if !matches!(o, CheckFiniteCoverageOutcome::Ok { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "check-finite-lint layer-coverage gate rejected body: {o:?}"
            )));
        }
    }
    Ok(())
}

fn print_report(
    error_file: Option<&Path>,
    list_file: Option<&Path>,
    err: Option<&CheckFiniteErrorOutcome>,
    cov: Option<&CheckFiniteCoverageOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "error_file": error_file.map(|p| p.display().to_string()),
            "list_file": list_file.map(|p| p.display().to_string()),
            "error_json": err.map(|o| format!("{o:?}")),
            "layer_coverage": cov.map(|o| format!("{o:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("check-finite-lint report");
    if let Some(p) = error_file {
        println!("  error_file    : {}", p.display());
    }
    if let Some(p) = list_file {
        println!("  list_file     : {}", p.display());
    }
    if let Some(o) = err {
        println!("  error_json    : {o:?}");
    }
    if let Some(o) = cov {
        println!("  layer_coverage: {o:?}");
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
    fn no_inputs_is_validation_failed() {
        let err = run(None, None, 0, false).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }
    #[test]
    fn missing_error_file_is_file_not_found() {
        let err = run(Some(Path::new("/no/such/err.json")), None, 0, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn missing_list_file_is_file_not_found() {
        let err = run(None, Some(Path::new("/no/such/list.txt")), 0, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn invalid_error_json_errors() {
        let f = w("nope");
        let err = run(Some(f.path()), None, 0, false);
        assert!(err.is_err());
    }
}
