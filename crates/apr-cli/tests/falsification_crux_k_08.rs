//! CRUX-K-08 — end-to-end falsification harness for `apr otlp-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-K-08-{001,002,003}
//! gate the classifier discharges has a matching captured OTLP/JSON body
//! that the binary must classify exactly as the harness expects.

use serde_json::json;
use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_otlp(body: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-k-08-otlp-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(
        serde_json::to_vec_pretty(body)
            .expect("serialize")
            .as_slice(),
    )
    .expect("write");
    f.flush().expect("flush");
    f
}

fn good_otlp_body() -> serde_json::Value {
    json!({
        "resourceSpans": [{
            "resource": {"attributes": [
                {"key": "service.name", "value": {"stringValue": "apr-serve"}}
            ]},
            "scopeSpans": [{
                "scope": {"name": "apr"},
                "spans": [{
                    "traceId": "0af7651916cd43dd8448eb211c80319c",
                    "spanId": "1234567890abcdef",
                    "parentSpanId": "00f067aa0ba902b7",
                    "name": "apr.inference",
                    "attributes": [
                        {"key": "gen_ai.system", "value": {"stringValue": "apr"}},
                        {"key": "gen_ai.request.model", "value": {"stringValue": "qwen3-1.7b"}},
                        {"key": "apr.tokens.prompt", "value": {"intValue": "42"}},
                        {"key": "apr.tokens.output", "value": {"intValue": "10"}}
                    ]
                }]
            }]
        }]
    })
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_k_08_cli_help_advertises_flags() {
    let out = apr_binary()
        .args(["otlp-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--otlp-file"),
        "--help must advertise --otlp-file; got:\n{stdout}"
    );
    assert!(
        stdout.contains("--require-apr-span"),
        "--help must advertise --require-apr-span; got:\n{stdout}"
    );
    assert!(
        stdout.contains("--require-genai-attrs"),
        "--help must advertise --require-genai-attrs; got:\n{stdout}"
    );
    assert!(
        stdout.contains("--expect-trace-id"),
        "--help must advertise --expect-trace-id; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_k_08_cli_missing_file_fails() {
    let out = apr_binary()
        .args([
            "otlp-lint",
            "--otlp-file",
            "/nonexistent/crux-k-08-missing.json",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

#[test]
fn falsify_crux_k_08_cli_malformed_json_fails() {
    let mut f = tempfile::Builder::new()
        .prefix("crux-k-08-bad-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(b"{ not json").expect("write");
    f.flush().expect("flush");
    let out = apr_binary()
        .args(["otlp-lint", "--otlp-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "malformed JSON must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_k_08_001_span_present_ok_on_apr_inference() {
    let f = write_otlp(&good_otlp_body());
    let out = apr_binary()
        .args(["otlp-lint", "--otlp-file"])
        .arg(f.path())
        .arg("--require-apr-span")
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good OTLP body must pass span-present gate; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_k_08_001_span_present_reports_missing_apr_inference() {
    let body =
        json!({"resourceSpans":[{"scopeSpans":[{"spans":[{"name":"http.server.request"}]}]}]});
    let f = write_otlp(&body);
    let out = apr_binary()
        .args(["otlp-lint", "--otlp-file"])
        .arg(f.path())
        .arg("--require-apr-span")
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "missing apr.inference span must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("SpanNameNotFound"),
        "stderr must name SpanNameNotFound outcome; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_k_08_002_genai_attributes_ok_on_full_body() {
    let f = write_otlp(&good_otlp_body());
    let out = apr_binary()
        .args(["otlp-lint", "--otlp-file"])
        .arg(f.path())
        .arg("--require-genai-attrs")
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "full body must pass genai-attributes gate; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_k_08_002_genai_attributes_reports_partial_set() {
    let body = json!({
        "resourceSpans":[{"scopeSpans":[{"spans":[{
            "name":"apr.inference",
            "attributes":[
                {"key":"gen_ai.system","value":{"stringValue":"apr"}}
            ]
        }]}]}]
    });
    let f = write_otlp(&body);
    let out = apr_binary()
        .args(["otlp-lint", "--otlp-file"])
        .arg(f.path())
        .arg("--require-genai-attrs")
        .output()
        .expect("run");
    assert!(!out.status.success(), "partial attribute set must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Missing") && stderr.contains("apr.tokens.prompt"),
        "stderr must list missing keys; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_k_08_003_trace_propagation_ok_on_canonical_traceparent() {
    let f = write_otlp(&good_otlp_body());
    let out = apr_binary()
        .args(["otlp-lint", "--otlp-file"])
        .arg(f.path())
        .args(["--expect-trace-id", "0af7651916cd43dd8448eb211c80319c"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "canonical traceparent must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_k_08_003_trace_propagation_rejects_short_id() {
    let f = write_otlp(&good_otlp_body());
    let out = apr_binary()
        .args(["otlp-lint", "--otlp-file"])
        .arg(f.path())
        .args(["--expect-trace-id", "shortid"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "short trace-id must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("InvalidExpectedTraceId"),
        "stderr must name InvalidExpectedTraceId outcome; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_k_08_003_trace_propagation_reports_mismatch() {
    let f = write_otlp(&good_otlp_body());
    let out = apr_binary()
        .args(["otlp-lint", "--otlp-file"])
        .arg(f.path())
        .args(["--expect-trace-id", "ffffffffffffffffffffffffffffffff"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "unmatched trace-id must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("TraceIdNotFound"),
        "stderr must name TraceIdNotFound outcome; got:\n{stderr}"
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_k_08_json_output_contains_outcomes() {
    let f = write_otlp(&good_otlp_body());
    let out = apr_binary()
        .args(["--json", "otlp-lint", "--otlp-file"])
        .arg(f.path())
        .arg("--require-apr-span")
        .arg("--require-genai-attrs")
        .args(["--expect-trace-id", "0af7651916cd43dd8448eb211c80319c"])
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good body must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["span_present"]
        .as_str()
        .expect("span_present")
        .contains("Ok"));
    assert!(parsed["genai_attributes"]
        .as_str()
        .expect("genai_attributes")
        .contains("Ok"));
    assert!(parsed["trace_propagation"]
        .as_str()
        .expect("trace_propagation")
        .contains("Ok"));
}
