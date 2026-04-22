//! E2E falsification tests for `apr gbnf-lint` (CRUX-C-10).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #977: exercise the CLI surface
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
fn falsify_crux_c_10_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["gbnf-lint", "--help"])
        .output()
        .expect("run apr gbnf-lint --help");
    assert!(out.status.success(), "apr gbnf-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_c_10_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("gbnf-lint")
        .output()
        .expect("run apr gbnf-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr gbnf-lint` must exit non-zero"
    );
}

// ---- json-output gate (FALSIFY-CRUX-C-10-001) -----------------------------

#[test]
fn falsify_crux_c_10_001_json_ok_on_valid_json_with_stop() {
    let tmp = write_tmp_json(
        "gbnf-json-ok",
        r#"{ "output": "{\"a\":1}", "finish_reason": "stop" }"#,
    );
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(
        out.status.success(),
        "valid JSON + stop must exit 0; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_10_001_json_rejects_non_json_output() {
    let tmp = write_tmp_json(
        "gbnf-json-notjson",
        r#"{ "output": "not valid json", "finish_reason": "stop" }"#,
    );
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(!out.status.success(), "non-JSON output must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-C-10-001"),
        "stderr must cite FALSIFY-CRUX-C-10-001; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_c_10_001_json_rejects_wrong_finish_reason() {
    let tmp = write_tmp_json(
        "gbnf-json-wrongfinish",
        r#"{ "output": "{}", "finish_reason": "tool_calls" }"#,
    );
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-10-001"));
}

// ---- diagnostic gate (FALSIFY-CRUX-C-10-002) ------------------------------

#[test]
fn falsify_crux_c_10_002_diagnostic_ok_on_nonzero_exit_with_keyword() {
    let tmp = write_tmp_json(
        "gbnf-diag-ok",
        r#"{
          "grammar_error": {
            "exit_code": 1,
            "stderr":    "error: invalid grammar at line 1"
          }
        }"#,
    );
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(
        out.status.success(),
        "nonzero exit + 'grammar' keyword must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_10_002_diagnostic_rejects_zero_exit() {
    let tmp = write_tmp_json(
        "gbnf-diag-zero",
        r#"{
          "grammar_error": {
            "exit_code": 0,
            "stderr":    "grammar was fine"
          }
        }"#,
    );
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-10-002"));
}

#[test]
fn falsify_crux_c_10_002_diagnostic_rejects_missing_keyword() {
    let tmp = write_tmp_json(
        "gbnf-diag-nokw",
        r#"{
          "grammar_error": {
            "exit_code": 1,
            "stderr":    "unrelated parse error"
          }
        }"#,
    );
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-10-002"));
}

// ---- masking gate (FALSIFY-CRUX-C-10-001 illegal-token sub-claim) ---------

#[test]
fn falsify_crux_c_10_001_masking_ok_when_illegal_positions_are_null() {
    // null → -Infinity per the CLI convention.
    let tmp = write_tmp_json(
        "gbnf-mask-ok",
        r#"{
          "masking": {
            "logits":     [1.0, null, 2.0, null],
            "legal_mask": [true, false, true, false]
          }
        }"#,
    );
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(
        out.status.success(),
        "null (= -Infinity) at illegal positions must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_10_001_masking_rejects_finite_at_illegal_position() {
    let tmp = write_tmp_json(
        "gbnf-mask-finite",
        r#"{
          "masking": {
            "logits":     [1.0, 2.0, 3.0],
            "legal_mask": [true, false, true]
          }
        }"#,
    );
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-10-001"));
}

#[test]
fn falsify_crux_c_10_001_masking_rejects_length_mismatch() {
    let tmp = write_tmp_json(
        "gbnf-mask-lenmismatch",
        r#"{
          "masking": {
            "logits":     [1.0, 2.0],
            "legal_mask": [true, false, true]
          }
        }"#,
    );
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-10-001"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_c_10_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("gbnf-empty", "");
    let out = apr_binary()
        .args(["gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_c_10_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "gbnf-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr gbnf-lint");
    assert!(!out.status.success());
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_c_10_json_output_shape() {
    let tmp = write_tmp_json(
        "gbnf-json-shape",
        r#"{
          "output": "{\"k\":1}",
          "finish_reason": "stop",
          "grammar_error": { "exit_code": 1, "stderr": "grammar bad" },
          "masking": {
            "logits":     [1.0, null],
            "legal_mask": [true, false]
          }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "gbnf-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gbnf-lint --json");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"json\""), "--json must emit json key");
    assert!(
        stdout.contains("\"diagnostic\""),
        "--json must emit diagnostic key"
    );
    assert!(
        stdout.contains("\"masking\""),
        "--json must emit masking key"
    );
}
