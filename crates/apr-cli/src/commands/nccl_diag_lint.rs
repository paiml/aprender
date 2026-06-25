//! `apr nccl-diag-lint` — CRUX-F-15 NCCL failure-diagnostics gate.
//!
//! Reads a captured stderr JSON diagnostic emitted by `apr train` on NCCL
//! error and dispatches the pure classifiers in `nccl_diag_classifier`.
//! Exits non-zero on any failure.
//!
//! Spec: `contracts/crux-F-15-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::nccl_diag_classifier::{
    classify_doc_link, classify_exit_code, classify_schema, NcclDocLinkOutcome, NcclExitOutcome,
    NcclSchemaOutcome, F15_MIN_EXIT_CODE,
};
use crate::error::{CliError, Result};

pub(crate) fn run(
    diag_file: &Path,
    exit_code: Option<i32>,
    require_doc_link: bool,
    json: bool,
) -> Result<()> {
    if !diag_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(diag_file)));
    }
    let body_text = std::fs::read_to_string(diag_file)?;
    let body: Value = serde_json::from_str(&body_text).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr nccl-diag-lint: failed to parse JSON from {}: {e}",
            diag_file.display()
        ))
    })?;

    let schema = classify_schema(&body);
    let doc_link = if require_doc_link && matches!(schema, NcclSchemaOutcome::Ok) {
        Some(classify_doc_link(&body))
    } else {
        None
    };
    let exit = exit_code.map(|c| classify_exit_code(c, F15_MIN_EXIT_CODE));

    print_report(diag_file, &schema, doc_link.as_ref(), exit.as_ref(), json);

    if !matches!(schema, NcclSchemaOutcome::Ok) {
        return Err(CliError::ValidationFailed(format!(
            "nccl-diag-lint schema gate rejected body: {schema:?}"
        )));
    }
    if let Some(o) = &doc_link {
        if !matches!(o, NcclDocLinkOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "nccl-diag-lint doc-link gate rejected body: {o:?}"
            )));
        }
    }
    if let Some(o) = &exit {
        if !matches!(o, NcclExitOutcome::Ok { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "nccl-diag-lint exit-code gate rejected: {o:?}"
            )));
        }
    }
    Ok(())
}

fn print_report(
    path: &Path,
    schema: &NcclSchemaOutcome,
    doc_link: Option<&NcclDocLinkOutcome>,
    exit: Option<&NcclExitOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "file": path.display().to_string(),
            "schema": format!("{schema:?}"),
            "doc_link": doc_link.map(|o| format!("{o:?}")),
            "exit_code": exit.map(|o| format!("{o:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("nccl-diag-lint report for {}", path.display());
    println!("  schema    : {schema:?}");
    if let Some(o) = doc_link {
        println!("  doc_link  : {o:?}");
    }
    if let Some(o) = exit {
        println!("  exit_code : {o:?}");
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
        let err = run(Path::new("/no/such/nccl.json"), None, false, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn invalid_json_is_invalid_format() {
        let f = w("xx");
        let err = run(f.path(), None, false, false).unwrap_err();
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }
    #[test]
    fn empty_object_runs() {
        let f = w("{}");
        let _ = run(f.path(), None, false, true);
    }
}
