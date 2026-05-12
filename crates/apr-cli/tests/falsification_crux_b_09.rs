//! E2E falsification tests for `apr gptq-lint` (CRUX-B-09).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #970: exercise the CLI surface
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
fn falsify_crux_b_09_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["gptq-lint", "--help"])
        .output()
        .expect("run apr gptq-lint --help");
    assert!(out.status.success(), "apr gptq-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_b_09_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("gptq-lint")
        .output()
        .expect("run apr gptq-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr gptq-lint` must exit non-zero"
    );
}

// ---- compression gate (FALSIFY-CRUX-B-09-001) -----------------------------

#[test]
fn falsify_crux_b_09_001_compression_ok_under_ceiling() {
    let tmp = write_tmp_json(
        "gptq-cmp-ok",
        r#"{ "compression": { "fp16_bytes": 1000000, "gptq_bytes": 200000, "max_ratio": 0.30 } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(
        out.status.success(),
        "20% ratio must pass 30% ceiling; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_09_001_compression_rejects_over_ceiling() {
    let tmp = write_tmp_json(
        "gptq-cmp-bad",
        r#"{ "compression": { "fp16_bytes": 1000000, "gptq_bytes": 400000, "max_ratio": 0.30 } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09-001"));
}

#[test]
fn falsify_crux_b_09_001_compression_rejects_zero_source() {
    let tmp = write_tmp_json(
        "gptq-cmp-zero",
        r#"{ "compression": { "fp16_bytes": 0, "gptq_bytes": 100 } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09-001"));
}

// ---- cosine gate (FALSIFY-CRUX-B-09-002) ----------------------------------

#[test]
fn falsify_crux_b_09_002_cosine_ok_on_identical_logits() {
    let tmp = write_tmp_json(
        "gptq-cos-ok",
        r#"{ "cosine": {
               "pairs": [
                 { "fp16": [1.0, 0.0, 0.5, -0.3], "gptq": [1.0, 0.0, 0.5, -0.3] },
                 { "fp16": [0.2, 0.7, 0.1, 0.9], "gptq": [0.2, 0.7, 0.1, 0.9] }
               ],
               "threshold": 0.98
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(
        out.status.success(),
        "identical logits must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_09_002_cosine_rejects_orthogonal_pair() {
    let tmp = write_tmp_json(
        "gptq-cos-bad",
        r#"{ "cosine": {
               "pairs": [ { "fp16": [1.0, 0.0], "gptq": [0.0, 1.0] } ],
               "threshold": 0.98
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09-002"));
}

#[test]
fn falsify_crux_b_09_002_cosine_rejects_all_mismatched_lengths() {
    let tmp = write_tmp_json(
        "gptq-cos-mis",
        r#"{ "cosine": {
               "pairs": [ { "fp16": [1.0, 0.0], "gptq": [1.0, 0.0, 0.5] } ],
               "threshold": 0.98
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09-002"));
}

#[test]
fn falsify_crux_b_09_002_cosine_rejects_missing_pairs_key() {
    let tmp = write_tmp_json("gptq-cos-nopairs", r#"{ "cosine": { "threshold": 0.98 } }"#);
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09-002"));
}

// ---- flags gate (FALSIFY-CRUX-B-09-003) -----------------------------------

#[test]
fn falsify_crux_b_09_003_flags_ok_on_canonical_argv() {
    let tmp = write_tmp_json(
        "gptq-flg-ok",
        r#"{ "flags": {
               "argv": ["--method", "gptq", "--bits", "4", "--group-size", "128"],
               "expected_outcome": "ok"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(
        out.status.success(),
        "canonical flags must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_09_003_flags_ok_on_equals_form() {
    let tmp = write_tmp_json(
        "gptq-flg-eq",
        r#"{ "flags": {
               "argv": ["--method=gptq", "--bits=4", "--group-size=-1"],
               "expected_outcome": "ok"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(
        out.status.success(),
        "--method=gptq per-tensor group_size must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_09_003_flags_rejects_wrong_method() {
    let tmp = write_tmp_json(
        "gptq-flg-wm",
        r#"{ "flags": {
               "argv": ["--method", "awq", "--bits", "4"],
               "expected_outcome": "ok"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09-003"));
}

#[test]
fn falsify_crux_b_09_003_flags_rejects_invalid_bits() {
    let tmp = write_tmp_json(
        "gptq-flg-bits",
        r#"{ "flags": {
               "argv": ["--method", "gptq", "--bits", "5"],
               "expected_outcome": "ok"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09-003"));
}

#[test]
fn falsify_crux_b_09_003_flags_rejects_missing_bits() {
    let tmp = write_tmp_json(
        "gptq-flg-mb",
        r#"{ "flags": {
               "argv": ["--method", "gptq"],
               "expected_outcome": "ok"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09-003"));
}

#[test]
fn falsify_crux_b_09_003_flags_observer_can_assert_expected_failure() {
    // Observer captured a deliberate negative case: argv has wrong method,
    // and the observation asserts that the gate SHOULD produce wrong_method.
    let tmp = write_tmp_json(
        "gptq-flg-neg",
        r#"{ "flags": {
               "argv": ["--method", "awq", "--bits", "4"],
               "expected_outcome": "wrong_method"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(
        out.status.success(),
        "expected_outcome=wrong_method must match observed wrong_method; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_b_09_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("gptq-empty", "");
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_b_09_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "gptq-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_b_09_observation_without_known_keys_rejected() {
    let tmp = write_tmp_json("gptq-emptyobj", r#"{ "other": 1 }"#);
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09"));
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_b_09_json_output_shape() {
    let tmp = write_tmp_json(
        "gptq-json",
        r#"{
          "compression": { "fp16_bytes": 1000000, "gptq_bytes": 250000, "max_ratio": 0.30 },
          "flags":       { "argv": ["--method","gptq","--bits","4"], "expected_outcome": "ok" }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint --json");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"compression\""));
    assert!(stdout.contains("\"flags\""));
    assert!(stdout.contains("CRUX-B-09"));
}
