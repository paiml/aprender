//! `apr oom-lint` — CRUX-F-13 CUDA OOM postmortem gate.
//!
//! Reads an already-captured `/tmp/apr-oom-<ts>.json` postmortem (from apr
//! or any tool that emits the CRUX-F-13 report shape) and dispatches the
//! pure classifiers in `oom_classifier`. Exits non-zero on any failure.
//!
//! Optional `--stderr-file` lets the caller point at a captured stderr log
//! so the `OOM_REPORT path=...` breadcrumb (FALSIFY-CRUX-F-13-004) can be
//! verified in the same invocation.
//!
//! Spec: `contracts/crux-F-13-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::oom_classifier::{
    classify_oom_breadcrumb, classify_oom_invariants, classify_oom_schema, classify_oom_size,
    OomBreadcrumbOutcome, OomInvariantsOutcome, OomSchemaOutcome, OomSizeOutcome,
};
use crate::error::{CliError, Result};

pub(crate) fn run(report_file: &Path, stderr_file: Option<&Path>, json: bool) -> Result<()> {
    if !report_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(report_file)));
    }
    let body = std::fs::read_to_string(report_file)?;
    let bytes = body.len() as u64;

    let report: Value = serde_json::from_str(&body).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr oom-lint: failed to parse JSON from {}: {e}",
            report_file.display()
        ))
    })?;

    let schema = classify_oom_schema(&report);
    let invariants = if matches!(schema, OomSchemaOutcome::Ok) {
        classify_oom_invariants(&report)
    } else {
        OomInvariantsOutcome::Ok
    };
    let size = classify_oom_size(bytes);

    let breadcrumb = match stderr_file {
        Some(path) => {
            if !path.exists() {
                return Err(CliError::FileNotFound(PathBuf::from(path)));
            }
            let stderr_body = std::fs::read_to_string(path)?;
            Some(classify_oom_breadcrumb(&stderr_body))
        }
        None => None,
    };

    print_report(
        report_file,
        &schema,
        &invariants,
        &size,
        breadcrumb.as_ref(),
        json,
    );

    match (&schema, &invariants, &size) {
        (OomSchemaOutcome::Ok, OomInvariantsOutcome::Ok, OomSizeOutcome::Ok { .. }) => {}
        (bad, _, _) if !matches!(bad, OomSchemaOutcome::Ok) => {
            return Err(CliError::ValidationFailed(format!(
                "oom-lint schema gate rejected report: {bad:?}"
            )));
        }
        (_, bad, _) if !matches!(bad, OomInvariantsOutcome::Ok) => {
            return Err(CliError::ValidationFailed(format!(
                "oom-lint invariants gate rejected report: {bad:?}"
            )));
        }
        (_, _, bad) => {
            return Err(CliError::ValidationFailed(format!(
                "oom-lint size gate rejected report: {bad:?}"
            )));
        }
    }

    if let Some(outcome) = &breadcrumb {
        if !matches!(outcome, OomBreadcrumbOutcome::Ok { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "oom-lint breadcrumb gate rejected stderr: {outcome:?}"
            )));
        }
    }

    Ok(())
}

fn print_report(
    path: &Path,
    schema: &OomSchemaOutcome,
    invariants: &OomInvariantsOutcome,
    size: &OomSizeOutcome,
    breadcrumb: Option<&OomBreadcrumbOutcome>,
    json: bool,
) {
    if json {
        let mut v = serde_json::json!({
            "report_path": path.display().to_string(),
            "schema_outcome": format!("{:?}", schema),
            "invariants_outcome": format!("{:?}", invariants),
            "size_outcome": format!("{:?}", size),
            "schema_ok": matches!(schema, OomSchemaOutcome::Ok),
            "invariants_ok": matches!(invariants, OomInvariantsOutcome::Ok),
            "size_ok": matches!(size, OomSizeOutcome::Ok { .. }),
        });
        if let Some(b) = breadcrumb {
            v["breadcrumb_outcome"] = serde_json::json!(format!("{:?}", b));
            v["breadcrumb_ok"] = serde_json::json!(matches!(b, OomBreadcrumbOutcome::Ok { .. }));
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&v).unwrap_or_else(|_| v.to_string())
        );
    } else {
        println!("oom-lint for {}", path.display());
        println!("  schema_outcome:     {schema:?}");
        println!("  invariants_outcome: {invariants:?}");
        println!("  size_outcome:       {size:?}");
        if let Some(b) = breadcrumb {
            println!("  breadcrumb_outcome: {b:?}");
        }
    }
}

#[cfg(test)]
mod cov_tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_obs(s: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(s.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn missing_report_is_file_not_found() {
        let err = run(Path::new("/no/such/oom.json"), None, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    #[test]
    fn invalid_json_is_invalid_format() {
        let f = write_obs("not json");
        let err = run(f.path(), None, false).unwrap_err();
        assert!(matches!(err, CliError::InvalidFormat(_)));
    }

    #[test]
    fn missing_stderr_file_is_file_not_found() {
        let report = write_obs("{}");
        let err = run(report.path(), Some(Path::new("/no/such/stderr.log")), false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }

    #[test]
    fn empty_report_fails_schema_gate() {
        let f = write_obs("{}");
        let err = run(f.path(), None, false).unwrap_err();
        assert!(matches!(err, CliError::ValidationFailed(_)));
    }
}
