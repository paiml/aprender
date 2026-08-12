//! `apr otlp-lint` — CRUX-K-08 OpenTelemetry OTLP trace gate.
//!
//! Reads an already-captured OTLP/JSON `ExportTraceServiceRequest` body and
//! dispatches the pure classifiers in `otlp_classifier`. Exits non-zero on
//! any failure.
//!
//! Every check is opt-in, so at least one of `--require-apr-span`,
//! `--require-genai-attrs` or `--expect-trace-id` must be given. A run that
//! selects no gate is rejected rather than reported as clean.
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
        CliError::InvalidInput(format!(
            "apr otlp-lint: failed to parse JSON from {}: {e}",
            otlp_file.display()
        ))
    })?;

    // All three checks are opt-in, so the documented bare invocation
    // `apr otlp-lint --otlp-file body.json` ran zero of them and exited 0 for
    // any parseable JSON — including the scalar `42`. A CI step wired to it
    // was permanently green.
    if !require_apr_span && !require_genai_attrs && expect_trace_id.is_none() {
        return Err(CliError::ValidationFailed(format!(
            "otlp-lint: VACUOUS RUN — no gate was selected, so nothing in {} was checked. Pass at \
             least one of --require-apr-span, --require-genai-attrs or --expect-trace-id <ID>.",
            otlp_file.display()
        )));
    }

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
    /// Was `empty_json_object_runs`, which discarded the result entirely and
    /// so asserted nothing about the very command whose job is to assert.
    #[test]
    fn falsifier_no_gate_flags_is_a_vacuous_run() {
        // Six bodies the 0.63.0 binary all accepted with rc=0 and a report
        // header that reads as a clean PASS.
        for body in [
            "{}",
            r#"{"totally":"unrelated"}"#,
            "[]",
            r#""str""#,
            "42",
            "false",
        ] {
            let f = w(body);
            for json in [false, true] {
                let err = run(f.path(), false, false, None, json).unwrap_err();
                match err {
                    CliError::ValidationFailed(msg) => {
                        assert!(msg.contains("VACUOUS RUN"), "{body}: {msg}");
                        assert!(msg.contains("--require-apr-span"), "{body}: {msg}");
                    }
                    other => panic!("{body}: expected ValidationFailed, got {other:?}"),
                }
            }
        }
    }

    /// The gates themselves still work, and an armed run on a good body still
    /// passes — the fix is to the default, not to the checks.
    #[test]
    fn armed_gate_still_distinguishes_good_from_bad() {
        let bad = w("[]");
        assert!(run(bad.path(), true, false, None, false).is_err());

        let good = w(r#"{"resourceSpans":[{"scopeSpans":[{"spans":[
            {"name":"apr.inference","traceId":"0123456789abcdef0123456789abcdef"}]}]}]}"#);
        assert!(
            run(good.path(), true, false, None, false).is_ok(),
            "an armed span-present gate must still pass a well-formed body"
        );
    }
}
