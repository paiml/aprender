//! CRUX-F-07 — end-to-end falsification harness for `apr gpu-memtrace-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-F-07-{001..003}
//! gate the classifier discharges has a matching captured Chrome Trace
//! JSON body that the binary must classify exactly as the harness expects.

use serde_json::json;
use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_body(body: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-f-07-")
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

fn good_trace() -> serde_json::Value {
    json!({
        "displayTimeUnit": "ns",
        "traceEvents": [
            {"ph": "i", "ts": 0,    "name": "alloc", "pid": 0, "tid": 1, "args": {"bytes": 1024, "addr": "0xAAAA"}},
            {"ph": "i", "ts": 100,  "name": "alloc", "pid": 0, "tid": 1, "args": {"bytes": 2048, "addr": "0xBBBB"}},
            {"ph": "i", "ts": 200,  "name": "free",  "pid": 0, "tid": 1, "args": {"bytes": 1024, "addr": "0xAAAA"}},
            {"ph": "i", "ts": 300,  "name": "free",  "pid": 0, "tid": 1, "args": {"bytes": 2048, "addr": "0xBBBB"}}
        ]
    })
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_f_07_cli_help_advertises_trace_file() {
    let out = apr_binary()
        .args(["gpu-memtrace-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--trace-file"),
        "--help must advertise --trace-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_f_07_cli_missing_file_fails() {
    let out = apr_binary()
        .args([
            "gpu-memtrace-lint",
            "--trace-file",
            "/nonexistent/crux-f-07-missing.json",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing file must not exit 0");
}

#[test]
fn falsify_crux_f_07_cli_malformed_json_fails() {
    let mut f = tempfile::Builder::new()
        .prefix("crux-f-07-bad-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(b"{ not json").expect("write");
    f.flush().expect("flush");
    let out = apr_binary()
        .args(["gpu-memtrace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "malformed JSON must not exit 0");
}

// ===== g3: classifier discharges =====

#[test]
fn falsify_crux_f_07_001_schema_ok_on_good_body() {
    let f = write_body(&good_trace());
    let out = apr_binary()
        .args(["gpu-memtrace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "good Chrome Trace must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_f_07_001_schema_rejects_missing_trace_events() {
    let body = json!({"otherKey": []});
    let f = write_body(&body);
    let out = apr_binary()
        .args(["gpu-memtrace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing traceEvents must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("MissingTraceEvents"),
        "stderr must name MissingTraceEvents; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_07_001_schema_rejects_event_missing_ph() {
    let body = json!({"traceEvents": [{"ts": 0, "name": "alloc"}]});
    let f = write_body(&body);
    let out = apr_binary()
        .args(["gpu-memtrace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "missing ph must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("EventMissingField"),
        "stderr must name EventMissingField; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_07_002_alloc_free_pairing_rejects_orphan_alloc() {
    let body = json!({
        "traceEvents": [
            {"ph": "i", "ts": 0, "name": "alloc", "args": {"addr": "0xLEAK"}},
            {"ph": "i", "ts": 1, "name": "alloc", "args": {"addr": "0xOK"}},
            {"ph": "i", "ts": 2, "name": "free",  "args": {"addr": "0xOK"}}
        ]
    });
    let f = write_body(&body);
    let out = apr_binary()
        .args(["gpu-memtrace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "orphan alloc must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("OrphanAllocs") && stderr.contains("0xLEAK"),
        "stderr must name OrphanAllocs with the leaked addr; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_07_002_alloc_free_pairing_rejects_orphan_free() {
    let body = json!({
        "traceEvents": [
            {"ph": "i", "ts": 0, "name": "alloc", "args": {"addr": "0xAAA"}},
            {"ph": "i", "ts": 1, "name": "free",  "args": {"addr": "0xAAA"}},
            {"ph": "i", "ts": 2, "name": "free",  "args": {"addr": "0xPHANTOM"}}
        ]
    });
    let f = write_body(&body);
    let out = apr_binary()
        .args(["gpu-memtrace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "orphan free must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("OrphanFrees") && stderr.contains("0xPHANTOM"),
        "stderr must name OrphanFrees with the phantom addr; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_07_003_monotonic_timestamps_rejects_violation() {
    let body = json!({
        "traceEvents": [
            {"ph": "i", "ts": 100, "name": "alloc", "pid": 0, "tid": 1, "args": {"addr": "0xA"}},
            {"ph": "i", "ts":  50, "name": "free",  "pid": 0, "tid": 1, "args": {"addr": "0xA"}}
        ]
    });
    let f = write_body(&body);
    let out = apr_binary()
        .args(["gpu-memtrace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(!out.status.success(), "non-monotonic ts must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("NonMonotonic"),
        "stderr must name NonMonotonic; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_f_07_003_monotonic_timestamps_allows_cross_stream_interleave() {
    // Different (pid, tid) streams can interleave freely.
    let body = json!({
        "traceEvents": [
            {"ph": "i", "ts": 100, "name": "alloc", "pid": 0, "tid": 1, "args": {"addr": "0xA"}},
            {"ph": "i", "ts":  50, "name": "alloc", "pid": 0, "tid": 2, "args": {"addr": "0xB"}},
            {"ph": "i", "ts": 200, "name": "free",  "pid": 0, "tid": 1, "args": {"addr": "0xA"}},
            {"ph": "i", "ts": 250, "name": "free",  "pid": 0, "tid": 2, "args": {"addr": "0xB"}}
        ]
    });
    let f = write_body(&body);
    let out = apr_binary()
        .args(["gpu-memtrace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "independent streams may interleave; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ===== JSON output shape =====

#[test]
fn falsify_crux_f_07_json_output_contains_outcomes() {
    let f = write_body(&good_trace());
    let out = apr_binary()
        .args(["--json", "gpu-memtrace-lint", "--trace-file"])
        .arg(f.path())
        .output()
        .expect("run");
    assert!(out.status.success(), "json + good body must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value = serde_json::from_str(&stdout).expect("json output must parse");
    assert!(parsed["schema"].as_str().expect("schema").contains("Ok"));
    assert!(parsed["alloc_free_pairing"]
        .as_str()
        .expect("pairing")
        .contains("Ok"));
    assert!(parsed["monotonic_timestamps"]
        .as_str()
        .expect("ts")
        .contains("Ok"));
}
