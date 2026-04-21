//! E2E falsification tests for `apr embeddings-lint` (CRUX-C-13).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #973: exercise the CLI surface
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
fn falsify_crux_c_13_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["embeddings-lint", "--help"])
        .output()
        .expect("run apr embeddings-lint --help");
    assert!(out.status.success(), "apr embeddings-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_c_13_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("embeddings-lint")
        .output()
        .expect("run apr embeddings-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr embeddings-lint` must exit non-zero"
    );
}

// ---- shape gate (FALSIFY-CRUX-C-13-001) -----------------------------------

#[test]
fn falsify_crux_c_13_001_shape_ok_on_three_rows() {
    let tmp = write_tmp_json(
        "em-shape-ok",
        r#"{ "shape": {
               "input_len": 3,
               "hidden_size": 4,
               "data": [
                 { "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] },
                 { "index": 1, "embedding": [0.5, 0.6, 0.7, 0.8] },
                 { "index": 2, "embedding": [0.9, 1.0, 1.1, 1.2] }
               ]
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(
        out.status.success(),
        "shape well-formed must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_13_001_shape_rejects_row_count_mismatch() {
    let tmp = write_tmp_json(
        "em-shape-rc",
        r#"{ "shape": {
               "input_len": 2, "hidden_size": 4,
               "data": [ { "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] } ]
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-13-001"));
}

#[test]
fn falsify_crux_c_13_001_shape_rejects_vector_dim_mismatch() {
    let tmp = write_tmp_json(
        "em-shape-dim",
        r#"{ "shape": {
               "input_len": 2, "hidden_size": 4,
               "data": [
                 { "index": 0, "embedding": [0.1, 0.2, 0.3, 0.4] },
                 { "index": 1, "embedding": [0.5, 0.6, 0.7] }
               ]
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-13-001"));
}

#[test]
fn falsify_crux_c_13_001_shape_rejects_index_out_of_order() {
    let tmp = write_tmp_json(
        "em-shape-idx",
        r#"{ "shape": {
               "input_len": 2, "hidden_size": 2,
               "data": [
                 { "index": 0, "embedding": [0.1, 0.2] },
                 { "index": 5, "embedding": [0.3, 0.4] }
               ]
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-13-001"));
}

// ---- determinism gate (FALSIFY-CRUX-C-13-002) -----------------------------

#[test]
fn falsify_crux_c_13_002_determinism_ok_on_identical() {
    let tmp = write_tmp_json(
        "em-det-ok",
        r#"{ "determinism": {
               "v1": [0.1, 0.2, 0.3, 0.4],
               "v2": [0.1, 0.2, 0.3, 0.4]
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(
        out.status.success(),
        "identical vectors must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_13_002_determinism_rejects_drift() {
    let tmp = write_tmp_json(
        "em-det-drift",
        r#"{ "determinism": {
               "v1": [1.0, 0.0, 0.0],
               "v2": [0.0, 1.0, 0.0]
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-13-002"));
}

#[test]
fn falsify_crux_c_13_002_determinism_rejects_zero_vector() {
    let tmp = write_tmp_json(
        "em-det-zero",
        r#"{ "determinism": { "v1": [0.0, 0.0, 0.0], "v2": [0.0, 0.0, 0.0] } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-13-002"));
}

// ---- usage gate (FALSIFY-CRUX-C-13-003) -----------------------------------

#[test]
fn falsify_crux_c_13_003_usage_ok_on_matching_prompt_total() {
    let tmp = write_tmp_json(
        "em-usage-ok",
        r#"{ "usage": { "prompt": 8, "total": 8 } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(
        out.status.success(),
        "matching usage must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_13_003_usage_rejects_mismatch() {
    let tmp = write_tmp_json(
        "em-usage-mis",
        r#"{ "usage": { "prompt": 5, "total": 7 } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-13-003"));
}

#[test]
fn falsify_crux_c_13_003_usage_rejects_zero_prompt() {
    let tmp = write_tmp_json(
        "em-usage-zero",
        r#"{ "usage": { "prompt": 0, "total": 0 } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-13-003"));
}

// ---- flag gate (FALSIFY-CRUX-C-13-004) ------------------------------------

#[test]
fn falsify_crux_c_13_004_flag_bare_form_enabled() {
    let tmp = write_tmp_json(
        "em-flag-bare",
        r#"{ "flag": {
               "argv": ["apr", "serve", "--embeddings-enabled"],
               "expected": "enabled"
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(
        out.status.success(),
        "bare flag must enable; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_13_004_flag_equals_false_disabled() {
    let tmp = write_tmp_json(
        "em-flag-false",
        r#"{ "flag": {
               "argv": ["apr", "serve", "--embeddings-enabled=false"],
               "expected": "disabled"
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(
        out.status.success(),
        "=false must be disabled; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_13_004_flag_rejects_wrong_expectation() {
    let tmp = write_tmp_json(
        "em-flag-wrong",
        r#"{ "flag": {
               "argv": ["apr", "serve", "--embeddings-enabled=false"],
               "expected": "enabled"
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-13-004"));
}

#[test]
fn falsify_crux_c_13_004_flag_malformed_is_rejected() {
    let tmp = write_tmp_json(
        "em-flag-malformed",
        r#"{ "flag": {
               "argv": ["apr", "serve", "--embeddings-enabled=maybe"],
               "expected": "enabled"
             } }"#,
    );
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-13-004"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_c_13_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("em-empty", "");
    let out = apr_binary()
        .args(["embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_c_13_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "embeddings-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr embeddings-lint");
    assert!(!out.status.success());
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_c_13_json_output_shape() {
    let tmp = write_tmp_json(
        "em-json",
        r#"{
          "shape": {
            "input_len": 1, "hidden_size": 2,
            "data": [ { "index": 0, "embedding": [0.1, 0.2] } ]
          },
          "determinism": {
            "v1": [0.1, 0.2, 0.3],
            "v2": [0.1, 0.2, 0.3]
          },
          "usage": { "prompt": 4, "total": 4 },
          "flag": {
            "argv": ["apr", "serve", "--embeddings-enabled"],
            "expected": "enabled"
          }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "embeddings-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr embeddings-lint --json");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"shape\""));
    assert!(stdout.contains("\"determinism\""));
    assert!(stdout.contains("\"usage\""));
    assert!(stdout.contains("\"flag\""));
}
