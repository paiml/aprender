//! E2E falsification tests for `apr fp8-lint` (CRUX-B-11).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #972: exercise the CLI surface
//! end-to-end on captured JSON observations and assert the classifier
//! verdicts + non-zero exit codes on known-bad input.

use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_tmp_json(name: &str, body: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix(name)
        .suffix(".json")
        .tempfile()
        .expect("create tempfile");
    f.write_all(body.as_bytes()).expect("write tempfile");
    f.flush().expect("flush tempfile");
    f
}

// ---- help surface (g2 proof) ----------------------------------------------

#[test]
fn falsify_crux_b_11_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["fp8-lint", "--help"])
        .output()
        .expect("run apr fp8-lint --help");
    assert!(out.status.success(), "apr fp8-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_b_11_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("fp8-lint")
        .output()
        .expect("run apr fp8-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr fp8-lint` must exit non-zero"
    );
}

// ---- frobenius gate (FALSIFY-CRUX-B-11-001) -------------------------------

#[test]
fn falsify_crux_b_11_001_frobenius_ok_on_identical_vectors() {
    let tmp = write_tmp_json(
        "fp8-frob-ok",
        r#"{ "frobenius": {
               "original":      [0.1, 0.2, 0.3, 0.4],
               "reconstructed": [0.1, 0.2, 0.3, 0.4],
               "threshold": 0.01
             } }"#,
    );
    let out = apr_binary()
        .args(["fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint");
    assert!(
        out.status.success(),
        "identical vectors must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_11_001_frobenius_rejects_degraded_reconstruction() {
    let tmp = write_tmp_json(
        "fp8-frob-deg",
        r#"{ "frobenius": {
               "original":      [1.0, 0.0, 0.0],
               "reconstructed": [0.5, 0.5, 0.5],
               "threshold": 0.01
             } }"#,
    );
    let out = apr_binary()
        .args(["fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-11-001"));
}

#[test]
fn falsify_crux_b_11_001_frobenius_rejects_length_mismatch() {
    let tmp = write_tmp_json(
        "fp8-frob-len",
        r#"{ "frobenius": {
               "original":      [1.0, 2.0],
               "reconstructed": [1.0, 2.0, 3.0],
               "threshold": 0.01
             } }"#,
    );
    let out = apr_binary()
        .args(["fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-11-001"));
}

#[test]
fn falsify_crux_b_11_001_frobenius_rejects_zero_original() {
    let tmp = write_tmp_json(
        "fp8-frob-zero",
        r#"{ "frobenius": {
               "original":      [0.0, 0.0, 0.0],
               "reconstructed": [0.0, 0.0, 0.0],
               "threshold": 0.01
             } }"#,
    );
    let out = apr_binary()
        .args(["fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-11-001"));
}

// ---- capability gate (FALSIFY-CRUX-B-11-002) ------------------------------

#[test]
fn falsify_crux_b_11_002_capability_ok_on_sm_90() {
    let tmp = write_tmp_json("fp8-cap-90", r#"{ "capability": { "sm": 90 } }"#);
    let out = apr_binary()
        .args(["fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint");
    assert!(
        out.status.success(),
        "sm_90 must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_11_002_capability_ok_on_sm_100_blackwell() {
    let tmp = write_tmp_json("fp8-cap-100", r#"{ "capability": { "sm": 100 } }"#);
    let out = apr_binary()
        .args(["fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint");
    assert!(
        out.status.success(),
        "sm_100 Blackwell must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_11_002_capability_rejects_sm_80_ampere() {
    let tmp = write_tmp_json("fp8-cap-80", r#"{ "capability": { "sm": 80 } }"#);
    let out = apr_binary()
        .args(["fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-11-002"));
}

#[test]
fn falsify_crux_b_11_002_capability_rejects_sm_0_unknown() {
    let tmp = write_tmp_json("fp8-cap-0", r#"{ "capability": { "sm": 0 } }"#);
    let out = apr_binary()
        .args(["fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-11-002"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_b_11_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("fp8-empty", "");
    let out = apr_binary()
        .args(["fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_b_11_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "fp8-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr fp8-lint");
    assert!(!out.status.success());
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_b_11_json_output_shape() {
    let tmp = write_tmp_json(
        "fp8-json",
        r#"{
          "frobenius": {
            "original":      [0.1, 0.2, 0.3],
            "reconstructed": [0.1, 0.2, 0.3],
            "threshold": 0.01
          },
          "capability": { "sm": 90 }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "fp8-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr fp8-lint --json");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"frobenius\""));
    assert!(stdout.contains("\"capability\""));
}
