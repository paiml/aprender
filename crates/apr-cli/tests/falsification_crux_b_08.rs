//! E2E falsification tests for `apr awq-lint` (CRUX-B-08).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #969: exercise the CLI surface
//! end-to-end on captured JSON observations and assert the classifier
//! verdicts + non-zero exit codes on known-bad input.
//!
//! Observation shape:
//! ```jsonc
//! {
//!   "quality":     { "p_fp16", "p_awq", "threshold" },         // FALSIFY-CRUX-B-08-001
//!   "flags":       { "argv": [..], "expected_outcome": "ok" }, // FALSIFY-CRUX-B-08-002
//!   "compression": { "fp16_bytes", "awq_bytes", "max_ratio" }  // FALSIFY-CRUX-B-08-003
//! }
//! ```

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
fn falsify_crux_b_08_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["awq-lint", "--help"])
        .output()
        .expect("run apr awq-lint --help");
    assert!(out.status.success(), "apr awq-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_b_08_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("awq-lint")
        .output()
        .expect("run apr awq-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr awq-lint` must exit non-zero"
    );
}

// ---- quality gate (FALSIFY-CRUX-B-08-001) ---------------------------------

#[test]
fn falsify_crux_b_08_001_quality_retained_ok() {
    // ratio = 0.45/0.50 = 0.90 ≥ 0.80 → Retained
    let tmp = write_tmp_json(
        "awq-quality-ok",
        r#"{ "quality": { "p_fp16": 0.50, "p_awq": 0.45, "threshold": 0.80 } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(
        out.status.success(),
        "retained quality must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_08_001_quality_rejects_degraded() {
    // ratio = 0.30/0.50 = 0.60 < 0.80 → Degraded
    let tmp = write_tmp_json(
        "awq-quality-bad",
        r#"{ "quality": { "p_fp16": 0.50, "p_awq": 0.30, "threshold": 0.80 } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "degraded quality must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-B-08-001"),
        "stderr must stamp FALSIFY-CRUX-B-08-001; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_b_08_001_quality_rejects_zero_baseline() {
    // p_fp16 = 0 → Degraded (NaN ratio, baseline broken)
    let tmp = write_tmp_json(
        "awq-quality-zero",
        r#"{ "quality": { "p_fp16": 0.0, "p_awq": 0.45, "threshold": 0.80 } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "zero baseline must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-08-001"));
}

// ---- compression gate (FALSIFY-CRUX-B-08-003) -----------------------------

#[test]
fn falsify_crux_b_08_003_compression_ok() {
    // 250M / 1G = 0.25 ≤ 0.30 → Compressed
    let tmp = write_tmp_json(
        "awq-comp-ok",
        r#"{ "compression": { "fp16_bytes": 1000000000, "awq_bytes": 250000000, "max_ratio": 0.30 } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(
        out.status.success(),
        "compression under ceiling must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_08_003_compression_rejects_over_ceiling() {
    // 400M / 1G = 0.40 > 0.30 → Insufficient
    let tmp = write_tmp_json(
        "awq-comp-bad",
        r#"{ "compression": { "fp16_bytes": 1000000000, "awq_bytes": 400000000, "max_ratio": 0.30 } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "compression over ceiling must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-08-003"));
}

#[test]
fn falsify_crux_b_08_003_compression_rejects_zero_source() {
    // fp16_bytes = 0 → Insufficient (∞ ratio, source missing)
    let tmp = write_tmp_json(
        "awq-comp-zero",
        r#"{ "compression": { "fp16_bytes": 0, "awq_bytes": 100, "max_ratio": 0.30 } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "zero source must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-08-003"));
}

// ---- flags gate (FALSIFY-CRUX-B-08-002) -----------------------------------

#[test]
fn falsify_crux_b_08_002_flags_ok_canonical() {
    // --method awq --bits 4 --group-size 128 → Ok
    let tmp = write_tmp_json(
        "awq-flags-ok",
        r#"{ "flags": { "argv": ["--method", "awq", "--bits", "4", "--group-size", "128"],
                        "expected_outcome": "ok" } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(
        out.status.success(),
        "canonical flags must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_08_002_flags_ok_equals_form() {
    // --method=awq --bits=4 → Ok (group_size defaults to 128)
    let tmp = write_tmp_json(
        "awq-flags-eq",
        r#"{ "flags": { "argv": ["--method=awq", "--bits=4"],
                        "expected_outcome": "ok" } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(
        out.status.success(),
        "equals-form flags must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_08_002_flags_rejects_wrong_method() {
    // Observer expected ok, but method=gptq → UnknownMethod → mismatch fails
    let tmp = write_tmp_json(
        "awq-flags-wrongmethod",
        r#"{ "flags": { "argv": ["--method", "gptq", "--bits", "4"],
                        "expected_outcome": "ok" } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "wrong method must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-08-002"));
}

#[test]
fn falsify_crux_b_08_002_flags_rejects_invalid_bits() {
    // --bits 5 is not in AWQ_ALLOWED_BITS → InvalidBits
    let tmp = write_tmp_json(
        "awq-flags-bits",
        r#"{ "flags": { "argv": ["--method", "awq", "--bits", "5"],
                        "expected_outcome": "ok" } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "invalid bits must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-08-002"));
}

#[test]
fn falsify_crux_b_08_002_flags_rejects_missing_bits() {
    // Missing --bits → InvalidBits { got: 0 }
    let tmp = write_tmp_json(
        "awq-flags-nobits",
        r#"{ "flags": { "argv": ["--method", "awq"],
                        "expected_outcome": "ok" } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "missing bits must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-08-002"));
}

#[test]
fn falsify_crux_b_08_002_flags_rejects_invalid_group_size() {
    // --group-size 96 is not in AWQ_ALLOWED_GROUP_SIZES → InvalidGroupSize
    let tmp = write_tmp_json(
        "awq-flags-gs",
        r#"{ "flags": { "argv": ["--method", "awq", "--bits", "4", "--group-size", "96"],
                        "expected_outcome": "ok" } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "invalid group size must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-08-002"));
}

#[test]
fn falsify_crux_b_08_002_flags_observer_can_assert_expected_failure() {
    // Observer deliberately captured a wrong-method case and expects that
    // classification → PASS (the captured outcome matches expectation).
    let tmp = write_tmp_json(
        "awq-flags-negobs",
        r#"{ "flags": { "argv": ["--method", "gptq", "--bits", "4"],
                        "expected_outcome": "unknown_method" } }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(
        out.status.success(),
        "observer-asserts-expected-failure must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_b_08_empty_file_rejected() {
    let tmp = write_tmp_json("awq-empty", "");
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_b_08_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "awq-lint",
            "--observation-file",
            "/nonexistent/path/awq-obs.json",
        ])
        .output()
        .expect("run apr awq-lint");
    assert!(!out.status.success(), "missing file must be rejected");
}

#[test]
fn falsify_crux_b_08_only_unknown_keys_rejected() {
    // No quality/compression/flags → nothing to classify
    let tmp = write_tmp_json(
        "awq-nogates",
        r#"{ "unrelated": 1, "also_irrelevant": "hi" }"#,
    );
    let out = apr_binary()
        .args(["awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint");
    assert!(
        !out.status.success(),
        "observation without any gate keys must be rejected"
    );
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_b_08_json_output_shape() {
    let tmp = write_tmp_json(
        "awq-json",
        r#"{
          "quality":     { "p_fp16": 0.50, "p_awq": 0.45, "threshold": 0.80 },
          "compression": { "fp16_bytes": 1000000000, "awq_bytes": 250000000, "max_ratio": 0.30 },
          "flags":       { "argv": ["--method", "awq", "--bits", "4", "--group-size", "128"],
                           "expected_outcome": "ok" }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "awq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr awq-lint --json");
    assert!(
        out.status.success(),
        "all-pass bundle under --json must exit 0; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"contract\""), "stdout:\n{stdout}");
    assert!(stdout.contains("\"CRUX-B-08\""));
    assert!(stdout.contains("\"gates\""));
    assert!(stdout.contains("\"quality\""));
    assert!(stdout.contains("\"compression\""));
    assert!(stdout.contains("\"flags\""));
    assert!(stdout.contains("FALSIFY-CRUX-B-08-001"));
    assert!(stdout.contains("FALSIFY-CRUX-B-08-002"));
    assert!(stdout.contains("FALSIFY-CRUX-B-08-003"));
}
