//! End-to-end falsification tests for CRUX-F-13 — CUDA OOM postmortem.
//!
//! Contract: `contracts/crux-F-13-v1.yaml` (v1.1.0).
//!
//! CRUX-SHIP-001 compliance:
//! - g1_classifier_green: `commands::oom_classifier` in-crate (28 tests).
//! - g2_cli_reachable: `apr oom-lint --help` surfaces `--report-file` and
//!   `--stderr-file`.
//! - g3_e2e_runs: subprocess invocation of the real binary runs the
//!   classifier end-to-end over a user-supplied captured `/tmp/apr-oom-*.json`
//!   postmortem. The live CUDA OOM trigger path in aprender-serve remains
//!   PARTIAL_ALGORITHM_LEVEL under BLOCKER-UPSTREAM-MISSING.

#![allow(clippy::unwrap_used)]

use std::io::Write;

use assert_cmd::Command;
use predicates::prelude::*;

fn write_json(v: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    let body = serde_json::to_vec_pretty(v).unwrap();
    f.write_all(&body).unwrap();
    f.flush().unwrap();
    f
}

fn write_text(body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(body.as_bytes()).unwrap();
    f.flush().unwrap();
    f
}

fn well_formed_report() -> serde_json::Value {
    serde_json::json!({
        "peak_allocated_bytes": 2_000_000u64,
        "peak_reserved_bytes": 3_000_000u64,
        "largest_alloc_stack": ["apr::run", "realizar::load", "cuda::malloc"],
        "tensor_histogram": {"1-64KB": 12u64, "64KB-1MB": 4u64, "1MB+": 1u64},
        "last_100_ops": [{"op": "matmul", "bytes": 4096u64, "ts_ns": 1u64}],
        "exit_code": 137i64,
        "timestamp": "2026-04-21T18:30:00Z",
    })
}

// ═══ g2_cli_reachable ═══

#[test]
fn falsify_crux_f_13_help_advertises_report_file_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--report-file"));
}

#[test]
fn falsify_crux_f_13_help_advertises_stderr_file_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--stderr-file"));
}

#[test]
fn falsify_crux_f_13_rejects_bare_invocation_without_file() {
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint"])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "oom-lint without --report-file must fail",
    );
}

// ═══ g3_e2e_runs — FALSIFY-CRUX-F-13-002 schema gate ═══

#[test]
fn falsify_crux_f_13_schema_accepts_well_formed_report() {
    let f = write_json(&well_formed_report());
    Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint", "--report-file", f.path().to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("schema_outcome:     Ok"))
        .stdout(predicate::str::contains("invariants_outcome: Ok"));
}

#[test]
fn falsify_crux_f_13_schema_rejects_missing_required_key() {
    let mut bad = well_formed_report();
    bad.as_object_mut().unwrap().remove("timestamp");
    let f = write_json(&bad);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint", "--report-file", f.path().to_str().unwrap()])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "missing timestamp must fail schema gate",
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("MissingRequiredKey")
            || stderr.contains("timestamp")
            || stderr.contains("schema gate"),
        "stderr should explain missing-key rejection: {stderr}",
    );
}

#[test]
fn falsify_crux_f_13_schema_rejects_malformed_json() {
    let f = write_text("{ not valid json");
    Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint", "--report-file", f.path().to_str().unwrap()])
        .assert()
        .failure();
}

// ═══ g3_e2e_runs — FALSIFY-CRUX-F-13-005 invariants gate ═══

#[test]
fn falsify_crux_f_13_invariants_rejects_reserved_less_than_allocated() {
    let mut bad = well_formed_report();
    bad["peak_allocated_bytes"] = serde_json::json!(5_000_000u64);
    bad["peak_reserved_bytes"] = serde_json::json!(1_000_000u64);
    let f = write_json(&bad);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint", "--report-file", f.path().to_str().unwrap()])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "reserved < allocated must fail invariants gate",
    );
}

#[test]
fn falsify_crux_f_13_invariants_rejects_silent_exit_code_zero() {
    let mut bad = well_formed_report();
    bad["exit_code"] = serde_json::json!(0i64);
    let f = write_json(&bad);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint", "--report-file", f.path().to_str().unwrap()])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "exit_code=0 must fail invariants gate (silent OOM-swallow)",
    );
}

#[test]
fn falsify_crux_f_13_invariants_rejects_more_than_100_ops() {
    let mut bad = well_formed_report();
    let ops: Vec<serde_json::Value> = (0u64..101)
        .map(|i| serde_json::json!({"op": "x", "bytes": 1u64, "ts_ns": i}))
        .collect();
    bad["last_100_ops"] = serde_json::json!(ops);
    let f = write_json(&bad);
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint", "--report-file", f.path().to_str().unwrap()])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "last_100_ops.len()=101 must fail invariants gate",
    );
}

#[test]
fn falsify_crux_f_13_invariants_rejects_empty_histogram() {
    let mut bad = well_formed_report();
    bad["tensor_histogram"] = serde_json::json!({});
    let f = write_json(&bad);
    Command::cargo_bin("apr")
        .unwrap()
        .args(["oom-lint", "--report-file", f.path().to_str().unwrap()])
        .assert()
        .failure();
}

// ═══ g3_e2e_runs — FALSIFY-CRUX-F-13-004 breadcrumb gate ═══

#[test]
fn falsify_crux_f_13_breadcrumb_accepts_well_formed_stderr() {
    let f = write_json(&well_formed_report());
    let stderr =
        write_text("some noise\nOOM_REPORT path=/tmp/apr-oom-1745000000.json\nmore noise\n");
    Command::cargo_bin("apr")
        .unwrap()
        .args([
            "oom-lint",
            "--report-file",
            f.path().to_str().unwrap(),
            "--stderr-file",
            stderr.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("breadcrumb_outcome: Ok"));
}

#[test]
fn falsify_crux_f_13_breadcrumb_rejects_missing_line() {
    let f = write_json(&well_formed_report());
    let stderr = write_text("no OOM_REPORT line here\n");
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "oom-lint",
            "--report-file",
            f.path().to_str().unwrap(),
            "--stderr-file",
            stderr.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "missing breadcrumb must fail the breadcrumb gate",
    );
}

#[test]
fn falsify_crux_f_13_breadcrumb_rejects_wrong_prefix() {
    let f = write_json(&well_formed_report());
    let stderr = write_text("OOM_REPORT path=/var/log/apr-oom-1.json\n");
    Command::cargo_bin("apr")
        .unwrap()
        .args([
            "oom-lint",
            "--report-file",
            f.path().to_str().unwrap(),
            "--stderr-file",
            stderr.path().to_str().unwrap(),
        ])
        .assert()
        .failure();
}

// ═══ g3_e2e_runs — JSON output contract ═══

#[test]
fn falsify_crux_f_13_json_output_structure() {
    let f = write_json(&well_formed_report());
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "--json",
            "oom-lint",
            "--report-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(output.status.success(), "well-formed report must exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("--json must emit valid JSON: {e}\nstdout={stdout}"));
    assert_eq!(v["schema_ok"], serde_json::json!(true));
    assert_eq!(v["invariants_ok"], serde_json::json!(true));
    assert_eq!(v["size_ok"], serde_json::json!(true));
}

// ═══ FALSIFY-CRUX-F-13-001 file-not-found ═══

#[test]
fn falsify_crux_f_13_rejects_nonexistent_file() {
    Command::cargo_bin("apr")
        .unwrap()
        .args([
            "oom-lint",
            "--report-file",
            "/tmp/apr-oom-does-not-exist-99999999.json",
        ])
        .assert()
        .failure();
}
