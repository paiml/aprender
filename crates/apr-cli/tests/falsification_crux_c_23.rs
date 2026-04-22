//! E2E falsification tests for `apr dry-sampling-lint` (CRUX-C-23).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #983: exercise the CLI surface
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
fn falsify_crux_c_23_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["dry-sampling-lint", "--help"])
        .output()
        .expect("run apr dry-sampling-lint --help");
    assert!(
        out.status.success(),
        "apr dry-sampling-lint --help must exit 0"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_c_23_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("dry-sampling-lint")
        .output()
        .expect("run apr dry-sampling-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr dry-sampling-lint` must exit non-zero"
    );
}

// ---- params gate (FALSIFY-CRUX-C-23-001) ----------------------------------

#[test]
fn falsify_crux_c_23_001_params_ok_on_defaults() {
    let tmp = write_tmp_json(
        "dry-params-ok",
        r#"{ "params": { "multiplier": 0.8, "base": 1.75, "allowed_length": 2 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(
        out.status.success(),
        "defaults must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_23_001_params_rejects_negative_multiplier() {
    let tmp = write_tmp_json(
        "dry-params-negmul",
        r#"{ "params": { "multiplier": -0.1, "base": 1.75, "allowed_length": 2 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-23-001"));
}

#[test]
fn falsify_crux_c_23_001_params_rejects_base_below_one() {
    let tmp = write_tmp_json(
        "dry-params-base",
        r#"{ "params": { "multiplier": 0.8, "base": 0.5, "allowed_length": 2 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-23-001"));
}

#[test]
fn falsify_crux_c_23_001_params_rejects_allowed_length_zero() {
    let tmp = write_tmp_json(
        "dry-params-al0",
        r#"{ "params": { "multiplier": 0.8, "base": 1.75, "allowed_length": 0 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-23-001"));
}

// ---- identity gate (FALSIFY-CRUX-C-23-001) --------------------------------

#[test]
fn falsify_crux_c_23_001_identity_ok_when_unchanged() {
    let tmp = write_tmp_json(
        "dry-id-ok",
        r#"{ "identity": { "logits_before": [0.1, 0.5, -0.3],
                          "logits_after":  [0.1, 0.5, -0.3],
                          "multiplier": 0.0 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(
        out.status.success(),
        "identity must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_23_001_identity_rejects_changed_logit() {
    let tmp = write_tmp_json(
        "dry-id-chg",
        r#"{ "identity": { "logits_before": [0.1, 0.5, -0.3],
                          "logits_after":  [0.1, 0.3, -0.3],
                          "multiplier": 0.0 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(!out.status.success(), "changed logits at mul=0 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-23-001"));
}

// ---- match_len gate (FALSIFY-CRUX-C-23-002) -------------------------------

#[test]
fn falsify_crux_c_23_002_match_len_ok_on_repeated_trigram() {
    // ctx = [1 2 3 1 2], candidate = 3 → expected match_len = 3
    let tmp = write_tmp_json(
        "dry-ml-ok",
        r#"{ "match_len": { "ctx": [1,2,3,1,2], "candidate": 3,
                            "seq_breakers": [], "expected_match_len": 3 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(
        out.status.success(),
        "match_len=3 expected; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_23_002_match_len_rejects_wrong_expected() {
    // actual = 3, expected = 7 → mismatch
    let tmp = write_tmp_json(
        "dry-ml-mismatch",
        r#"{ "match_len": { "ctx": [1,2,3,1,2], "candidate": 3,
                            "seq_breakers": [], "expected_match_len": 7 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-23-002"));
}

#[test]
fn falsify_crux_c_23_002_match_len_seq_breaker_zero() {
    // ctx = [1,2,9,1,2] with 9 as a breaker → expected match_len = 0
    let tmp = write_tmp_json(
        "dry-ml-breaker",
        r#"{ "match_len": { "ctx": [1,2,9,1,2], "candidate": 3,
                            "seq_breakers": [9], "expected_match_len": 0 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(
        out.status.success(),
        "seq_breaker reset should pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---- penalty gate (FALSIFY-CRUX-C-23-002) ---------------------------------

#[test]
fn falsify_crux_c_23_002_penalty_ok_at_threshold() {
    // match_len=2, allowed=2 → exponent=0 → penalty = multiplier
    let tmp = write_tmp_json(
        "dry-pen-ok",
        r#"{ "penalty": { "match_len": 2, "allowed_length": 2,
                          "multiplier": 0.8, "base": 1.75 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(
        out.status.success(),
        "penalty at threshold should pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_23_002_penalty_rejects_invalid_input() {
    // multiplier negative is InvalidInput, not Negative — but still fails
    let tmp = write_tmp_json(
        "dry-pen-neg",
        r#"{ "penalty": { "match_len": 5, "allowed_length": 2,
                          "multiplier": -0.1, "base": 1.75 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-23-002"));
}

// ---- monotone gate (FALSIFY-CRUX-C-23-002) --------------------------------

#[test]
fn falsify_crux_c_23_002_monotone_ok_non_decreasing() {
    // 3 → 5 above threshold, penalty grows
    let tmp = write_tmp_json(
        "dry-mono-ok",
        r#"{ "monotone": { "match_len_a": 3, "match_len_b": 5,
                           "allowed_length": 2, "multiplier": 0.8, "base": 1.75 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(
        out.status.success(),
        "monotone non-decreasing should pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_23_002_monotone_rejects_decreasing_args() {
    // a > b is InvalidInput — fails
    let tmp = write_tmp_json(
        "dry-mono-bad",
        r#"{ "monotone": { "match_len_a": 5, "match_len_b": 3,
                           "allowed_length": 2, "multiplier": 0.8, "base": 1.75 } }"#,
    );
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-23-002"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_c_23_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("dry-empty", "");
    let out = apr_binary()
        .args(["dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_c_23_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "dry-sampling-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr dry-sampling-lint");
    assert!(!out.status.success());
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_c_23_json_output_shape() {
    let tmp = write_tmp_json(
        "dry-json",
        r#"{
          "params":    { "multiplier": 0.8, "base": 1.75, "allowed_length": 2 },
          "identity":  { "logits_before": [0.1, 0.5],
                         "logits_after":  [0.1, 0.5],
                         "multiplier": 0.0 },
          "match_len": { "ctx": [1,2,3,1,2], "candidate": 3,
                         "seq_breakers": [], "expected_match_len": 3 },
          "penalty":   { "match_len": 5, "allowed_length": 2,
                         "multiplier": 0.8, "base": 1.75 },
          "monotone":  { "match_len_a": 3, "match_len_b": 5,
                         "allowed_length": 2, "multiplier": 0.8, "base": 1.75 }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "dry-sampling-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr dry-sampling-lint --json");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"params\""));
    assert!(stdout.contains("\"identity\""));
    assert!(stdout.contains("\"match_len\""));
    assert!(stdout.contains("\"penalty\""));
    assert!(stdout.contains("\"monotone\""));
}
