//! E2E falsification tests for `apr speculative-lint` (CRUX-C-09).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #976: exercise the CLI surface
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
fn falsify_crux_c_09_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["speculative-lint", "--help"])
        .output()
        .expect("run apr speculative-lint --help");
    assert!(out.status.success(), "apr speculative-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
    assert!(
        stdout.contains("--alpha-min"),
        "--help must advertise --alpha-min; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_c_09_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("speculative-lint")
        .output()
        .expect("run apr speculative-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr speculative-lint` must exit non-zero"
    );
}

// ---- parity gate (FALSIFY-CRUX-C-09-001) ----------------------------------

#[test]
fn falsify_crux_c_09_001_parity_ok_on_matching_tokens() {
    let tmp = write_tmp_json(
        "spec-parity-ok",
        r#"{ "base_tokens": [1,2,3], "spec_tokens": [1,2,3] }"#,
    );
    let out = apr_binary()
        .args(["speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint");
    assert!(
        out.status.success(),
        "matching parity must exit 0; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_09_001_parity_rejects_length_mismatch() {
    let tmp = write_tmp_json(
        "spec-parity-len",
        r#"{ "base_tokens": [1,2,3], "spec_tokens": [1,2,3,4] }"#,
    );
    let out = apr_binary()
        .args(["speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint");
    assert!(!out.status.success(), "length mismatch must exit non-zero");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-C-09-001"),
        "stderr must cite FALSIFY-CRUX-C-09-001; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_c_09_001_parity_rejects_token_divergence() {
    let tmp = write_tmp_json(
        "spec-parity-div",
        r#"{ "base_tokens": [1,2,3], "spec_tokens": [1,2,9] }"#,
    );
    let out = apr_binary()
        .args(["speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-09-001"));
}

// ---- uplift gate (FALSIFY-CRUX-C-09-002) ----------------------------------

#[test]
fn falsify_crux_c_09_002_uplift_ok_above_threshold() {
    let tmp = write_tmp_json(
        "spec-uplift-ok",
        r#"{ "base_tps": 100.0, "spec_tps": 140.0 }"#,
    );
    let out = apr_binary()
        .args(["speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint");
    assert!(out.status.success(), "40% uplift must pass 30% threshold");
}

#[test]
fn falsify_crux_c_09_002_uplift_rejects_regression() {
    let tmp = write_tmp_json(
        "spec-uplift-reg",
        r#"{ "base_tps": 100.0, "spec_tps": 80.0 }"#,
    );
    let out = apr_binary()
        .args(["speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-09-002"));
}

// ---- compat gate (FALSIFY-CRUX-C-09-003) ----------------------------------

#[test]
fn falsify_crux_c_09_003_compat_ok_on_matching_tokenizers() {
    let tmp = write_tmp_json(
        "spec-compat-ok",
        r#"{
          "draft_tokenizer_sha256":  "abc123",
          "target_tokenizer_sha256": "abc123",
          "draft_vocab_size":        50000,
          "target_vocab_size":       50000
        }"#,
    );
    let out = apr_binary()
        .args(["speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_c_09_003_compat_rejects_vocab_mismatch() {
    let tmp = write_tmp_json(
        "spec-compat-vocab",
        r#"{
          "draft_tokenizer_sha256":  "abc123",
          "target_tokenizer_sha256": "abc123",
          "draft_vocab_size":        50000,
          "target_vocab_size":       32000
        }"#,
    );
    let out = apr_binary()
        .args(["speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-09-003"));
}

// ---- acceptance_rate gate (FALSIFY-CRUX-C-09-004) -------------------------

#[test]
fn falsify_crux_c_09_004_acceptance_rate_rejects_out_of_range() {
    let tmp = write_tmp_json(
        "spec-ar-oor",
        r#"{ "speculative": { "acceptance_rate": 1.5 } }"#,
    );
    let out = apr_binary()
        .args(["speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-09-004"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_c_09_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("spec-empty", "");
    let out = apr_binary()
        .args(["speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_c_09_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "speculative-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr speculative-lint");
    assert!(!out.status.success());
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_c_09_json_output_shape() {
    let tmp = write_tmp_json(
        "spec-json-ok",
        r#"{
          "base_tokens": [1,2], "spec_tokens": [1,2],
          "base_tps": 100.0, "spec_tps": 140.0,
          "draft_tokenizer_sha256":  "abc",
          "target_tokenizer_sha256": "abc",
          "draft_vocab_size":        100,
          "target_vocab_size":       100,
          "speculative": { "acceptance_rate": 0.75 }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "speculative-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr speculative-lint --json");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"parity\""), "--json must emit parity key");
    assert!(stdout.contains("\"uplift\""), "--json must emit uplift key");
    assert!(stdout.contains("\"compat\""), "--json must emit compat key");
    assert!(
        stdout.contains("\"acceptance\""),
        "--json must emit acceptance key"
    );
}
