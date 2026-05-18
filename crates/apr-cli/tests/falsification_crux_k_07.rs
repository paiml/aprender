//! CRUX-K-07 — end-to-end falsification harness for `apr prometheus-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-K-07-{001,002,003}
//! gate the classifier discharges has a matching captured `/metrics` body
//! that the binary must classify exactly as the harness expects.

use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_body(body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-k-07-body-")
        .suffix(".txt")
        .tempfile()
        .expect("tempfile");
    f.write_all(body.as_bytes()).expect("write body");
    f.flush().expect("flush");
    f
}

fn good_k07_body() -> String {
    concat!(
        "# HELP apr_num_requests_running running requests\n",
        "# TYPE apr_num_requests_running gauge\n",
        "apr_num_requests_running 3\n",
        "# HELP apr_num_requests_waiting queued requests\n",
        "# TYPE apr_num_requests_waiting gauge\n",
        "apr_num_requests_waiting 0\n",
        "# HELP apr_gpu_cache_usage_perc kv cache\n",
        "# TYPE apr_gpu_cache_usage_perc gauge\n",
        "apr_gpu_cache_usage_perc 0.42\n",
        "# TYPE apr_time_to_first_token_seconds histogram\n",
        "apr_time_to_first_token_seconds_bucket{le=\"0.5\"} 100\n",
        "apr_time_to_first_token_seconds_sum 12.5\n",
        "apr_time_to_first_token_seconds_count 200\n",
        "# TYPE apr_time_per_output_token_seconds histogram\n",
        "apr_time_per_output_token_seconds_bucket{le=\"0.05\"} 80\n",
        "apr_time_per_output_token_seconds_sum 1.5\n",
        "apr_time_per_output_token_seconds_count 40\n",
        "# TYPE apr_e2e_request_latency_seconds histogram\n",
        "apr_e2e_request_latency_seconds_bucket{le=\"1.0\"} 50\n",
        "apr_e2e_request_latency_seconds_sum 30.0\n",
        "apr_e2e_request_latency_seconds_count 60\n",
    )
    .to_string()
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_k_07_cli_help_advertises_metrics_file() {
    let out = apr_binary()
        .args(["prometheus-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--metrics-file"),
        "--help must advertise --metrics-file; got:\n{stdout}"
    );
    assert!(
        stdout.contains("--content-type"),
        "--help must advertise --content-type; got:\n{stdout}"
    );
    assert!(
        stdout.contains("--require-k07-metrics"),
        "--help must advertise --require-k07-metrics; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_k_07_cli_missing_file_fails() {
    let out = apr_binary()
        .args([
            "prometheus-lint",
            "--metrics-file",
            "/nonexistent/crux-k-07-missing.txt",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_k_07_001_text_format_ok_on_good_body() {
    let f = write_body(&good_k07_body());
    let out = apr_binary()
        .args(["prometheus-lint", "--metrics-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good K-07 body must pass text-format gate; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_k_07_001_text_format_rejects_sample_before_type() {
    let body = "apr_x 1\n";
    let f = write_body(body);
    let out = apr_binary()
        .args(["prometheus-lint", "--metrics-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "sample-before-TYPE must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("SampleBeforeType"),
        "stderr must name SampleBeforeType outcome; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_k_07_001_text_format_rejects_nonnumeric_sample() {
    let body = "# TYPE apr_x gauge\napr_x banana\n";
    let f = write_body(body);
    let out = apr_binary()
        .args(["prometheus-lint", "--metrics-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "non-numeric sample must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("SampleValueNotNumeric"),
        "stderr must name SampleValueNotNumeric outcome; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_k_07_002_required_metrics_pass_on_full_body() {
    let f = write_body(&good_k07_body());
    let out = apr_binary()
        .args(["prometheus-lint", "--metrics-file"])
        .arg(f.path())
        .arg("--require-k07-metrics")
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "full K-07 body must pass required-metrics gate; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_k_07_002_required_metrics_reports_partial_set() {
    let body = "# TYPE apr_num_requests_running gauge\napr_num_requests_running 1\n";
    let f = write_body(body);
    let out = apr_binary()
        .args(["prometheus-lint", "--metrics-file"])
        .arg(f.path())
        .arg("--require-k07-metrics")
        .output()
        .expect("run");
    assert!(!out.status.success(), "partial K-07 set must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Missing") && stderr.contains("apr_e2e_request_latency_seconds"),
        "stderr must list missing K-07 metrics; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_k_07_001_content_type_ok_on_canonical_header() {
    let f = write_body(&good_k07_body());
    let out = apr_binary()
        .args(["prometheus-lint", "--metrics-file"])
        .arg(f.path())
        .args(["--content-type", "text/plain; version=0.0.4; charset=utf-8"])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "canonical Content-Type must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_k_07_001_content_type_rejects_application_json() {
    let f = write_body(&good_k07_body());
    let out = apr_binary()
        .args(["prometheus-lint", "--metrics-file"])
        .arg(f.path())
        .args(["--content-type", "application/json"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "application/json must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("WrongMediaType"),
        "stderr must name WrongMediaType outcome; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_k_07_001_content_type_rejects_missing_version() {
    let f = write_body(&good_k07_body());
    let out = apr_binary()
        .args(["prometheus-lint", "--metrics-file"])
        .arg(f.path())
        .args(["--content-type", "text/plain; charset=utf-8"])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing version parameter must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MissingVersion"),
        "stderr must name MissingVersion outcome; got:\n{stderr}"
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_k_07_json_output_contains_outcomes() {
    let f = write_body(&good_k07_body());
    let out = apr_binary()
        .args(["--json", "prometheus-lint", "--metrics-file"])
        .arg(f.path())
        .arg("--require-k07-metrics")
        .args(["--content-type", "text/plain; version=0.0.4"])
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good body must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["text_format"]
        .as_str()
        .expect("text_format string")
        .contains("Ok"));
    assert!(parsed["required_metrics"]
        .as_str()
        .expect("required_metrics string")
        .contains("Ok"));
    assert!(parsed["content_type"]
        .as_str()
        .expect("content_type string")
        .contains("Ok"));
}
