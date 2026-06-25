//! `apr prometheus-lint` — CRUX-K-07 Prometheus /metrics exposition gate.
//!
//! Reads an already-captured `/metrics` HTTP response body (and optionally
//! the `Content-Type` response header) and dispatches the pure classifiers
//! in `prometheus_classifier`. Exits non-zero on any failure.
//!
//! Spec: `contracts/crux-K-07-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use super::prometheus_classifier::{
    classify_content_type, classify_required_metrics, classify_text_format, PromContentTypeOutcome,
    PromRequiredOutcome, PromTextFormatOutcome, K07_REQUIRED_METRICS,
};
use crate::error::{CliError, Result};

pub(crate) fn run(
    metrics_file: &Path,
    content_type: Option<&str>,
    require_k07_metrics: bool,
    json: bool,
) -> Result<()> {
    if !metrics_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(metrics_file)));
    }
    let body = std::fs::read_to_string(metrics_file)?;

    let text = classify_text_format(&body);
    let required = if require_k07_metrics {
        Some(classify_required_metrics(&body, K07_REQUIRED_METRICS))
    } else {
        None
    };
    let ct = content_type.map(classify_content_type);

    print_report(metrics_file, &text, required.as_ref(), ct.as_ref(), json);

    if !matches!(text, PromTextFormatOutcome::Ok { .. }) {
        return Err(CliError::ValidationFailed(format!(
            "prometheus-lint text-format gate rejected body: {text:?}"
        )));
    }
    if let Some(outcome) = &required {
        if !matches!(outcome, PromRequiredOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "prometheus-lint required-metrics gate rejected body: {outcome:?}"
            )));
        }
    }
    if let Some(outcome) = &ct {
        if !matches!(outcome, PromContentTypeOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "prometheus-lint content-type gate rejected header: {outcome:?}"
            )));
        }
    }
    Ok(())
}

fn print_report(
    path: &Path,
    text: &PromTextFormatOutcome,
    required: Option<&PromRequiredOutcome>,
    ct: Option<&PromContentTypeOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "file": path.display().to_string(),
            "text_format": format!("{text:?}"),
            "required_metrics": required.map(|r| format!("{r:?}")),
            "content_type": ct.map(|c| format!("{c:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("prometheus-lint report for {}", path.display());
    println!("  text_format     : {text:?}");
    if let Some(r) = required {
        println!("  required_metrics: {r:?}");
    }
    if let Some(c) = ct {
        println!("  content_type    : {c:?}");
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
        let err = run(Path::new("/no/such/metrics.txt"), None, false, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn empty_metrics_runs() {
        let f = w("");
        let _ = run(f.path(), None, false, true);
    }
    #[test]
    fn some_metrics_runs() {
        let f = w("# HELP foo bar\nfoo 1\n");
        let _ = run(f.path(), Some("text/plain"), false, false);
    }
}
