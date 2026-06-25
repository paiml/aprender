//! `apr otlp-lint` — CRUX-K-08 OpenTelemetry OTLP trace gate.
//!
//! Reads an already-captured OTLP/JSON `ExportTraceServiceRequest` body and
//! dispatches the pure classifiers in `otlp_classifier`. Exits non-zero on
//! any failure.
//!
//! Spec: `contracts/crux-K-08-v1.yaml`. CRUX-SHIP-001 g2/g3 surface.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::otlp_classifier::{
    classify_genai_attributes, classify_span_present, classify_trace_propagation,
    OtlpAttributesOutcome, OtlpSpanPresentOutcome, OtlpTracePropagationOutcome,
    K08_REQUIRED_ATTRIBUTES, K08_ROOT_SPAN_NAME,
};
use crate::error::{CliError, Result};

pub(crate) fn run(
    otlp_file: &Path,
    require_apr_span: bool,
    require_genai_attrs: bool,
    expect_trace_id: Option<&str>,
    json: bool,
) -> Result<()> {
    if !otlp_file.exists() {
        return Err(CliError::FileNotFound(PathBuf::from(otlp_file)));
    }
    let body_text = std::fs::read_to_string(otlp_file)?;
    let body: Value = serde_json::from_str(&body_text).map_err(|e| {
        CliError::InvalidFormat(format!(
            "apr otlp-lint: failed to parse JSON from {}: {e}",
            otlp_file.display()
        ))
    })?;

    let span = if require_apr_span {
        Some(classify_span_present(&body, K08_ROOT_SPAN_NAME))
    } else {
        None
    };
    let attrs = if require_genai_attrs {
        Some(classify_genai_attributes(&body, K08_REQUIRED_ATTRIBUTES))
    } else {
        None
    };
    let trace = expect_trace_id.map(|tid| classify_trace_propagation(&body, tid));

    print_report(
        otlp_file,
        span.as_ref(),
        attrs.as_ref(),
        trace.as_ref(),
        json,
    );

    if let Some(outcome) = &span {
        if !matches!(outcome, OtlpSpanPresentOutcome::Ok { .. }) {
            return Err(CliError::ValidationFailed(format!(
                "otlp-lint span-present gate rejected body: {outcome:?}"
            )));
        }
    }
    if let Some(outcome) = &attrs {
        if !matches!(outcome, OtlpAttributesOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "otlp-lint genai-attributes gate rejected body: {outcome:?}"
            )));
        }
    }
    if let Some(outcome) = &trace {
        if !matches!(outcome, OtlpTracePropagationOutcome::Ok) {
            return Err(CliError::ValidationFailed(format!(
                "otlp-lint trace-propagation gate rejected body: {outcome:?}"
            )));
        }
    }
    Ok(())
}

fn print_report(
    path: &Path,
    span: Option<&OtlpSpanPresentOutcome>,
    attrs: Option<&OtlpAttributesOutcome>,
    trace: Option<&OtlpTracePropagationOutcome>,
    json: bool,
) {
    if json {
        let obj = serde_json::json!({
            "file": path.display().to_string(),
            "span_present": span.map(|s| format!("{s:?}")),
            "genai_attributes": attrs.map(|a| format!("{a:?}")),
            "trace_propagation": trace.map(|t| format!("{t:?}")),
        });
        println!("{}", serde_json::to_string_pretty(&obj).unwrap_or_default());
        return;
    }
    println!("otlp-lint report for {}", path.display());
    if let Some(s) = span {
        println!("  span_present     : {s:?}");
    }
    if let Some(a) = attrs {
        println!("  genai_attributes : {a:?}");
    }
    if let Some(t) = trace {
        println!("  trace_propagation: {t:?}");
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
        let err = run(Path::new("/no/such/otlp.json"), false, false, None, false).unwrap_err();
        assert!(matches!(err, CliError::FileNotFound(_)));
    }
    #[test]
    fn malformed_content_errors() {
        let f = w("definitely not otlp json");
        let err = run(f.path(), false, false, None, false);
        assert!(err.is_err());
    }
    #[test]
    fn empty_json_object_runs() {
        let f = w("{}");
        let _ = run(f.path(), false, false, None, true);
    }
}
