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
//
// The verdict comes from the SHIPPED clap parser (`commands::quantize_flag_parity`),
// not from a hand-rolled matcher living beside the assertion. `expected_outcome`
// is `accepted` (alias `ok`) or `rejected`; the pre-fix per-flag labels are
// refused as an unusable observation. See aprender#2377 finding 2 and
// `contracts/apr-lint-flag-parity-v1.yaml`.

#[test]
fn falsify_crux_b_09_003_flags_ok_on_a_real_quantize_argv() {
    let tmp = write_tmp_json(
        "gptq-flg-ok",
        r#"{ "flags": {
               "argv": ["model.safetensors", "--scheme", "int4", "-o", "out.apr"],
               "expected_outcome": "accepted"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(
        out.status.success(),
        "a real `apr quantize <FILE> --scheme int4 -o out.apr` must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The load-bearing falsifier. Before aprender#2377 finding 2 this exact
/// observation exited 0: the gate asked `parse_gptq_flags`/`validate_gptq_flags`,
/// which reported `Ok { bits: 4, group_size: 128 }`. `apr quantize` has no
/// `--method`, no `--bits` and no `--group-size`.
#[test]
fn falsify_crux_b_09_003_flags_reject_method_bits_group_size() {
    let tmp = write_tmp_json(
        "gptq-flg-legacy",
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
    assert_eq!(
        out.status.code(),
        Some(5),
        "a gate rejection is exit 5; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-09-003"), "got: {stderr}");
    assert!(stderr.contains("REJECTED"), "got: {stderr}");
    assert!(
        stderr.contains("--scheme"),
        "the failure must name the flags `apr quantize` does accept; got: {stderr}"
    );
}

#[test]
fn falsify_crux_b_09_003_flags_reject_an_unknown_flag() {
    let tmp = write_tmp_json(
        "gptq-flg-unknown",
        r#"{ "flags": {
               "argv": ["m.apr", "--totally-made-up"],
               "expected_outcome": "accepted"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert_eq!(out.status.code(), Some(5));
}

#[test]
fn falsify_crux_b_09_003_flags_observer_can_assert_expected_rejection() {
    // Observer captured a deliberate negative case and asserts the refusal
    // that the shipped parser actually produces.
    let tmp = write_tmp_json(
        "gptq-flg-neg",
        r#"{ "flags": {
               "argv": ["--method", "awq", "--bits", "4"],
               "expected_outcome": "rejected"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert!(
        out.status.success(),
        "expected_outcome=rejected must match the shipped parser's refusal; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Exit 4, not 5: an `expected_outcome` the real parser cannot emit means the
/// capture step is stale, not that the system under test broke a contract.
#[test]
fn falsify_crux_b_09_003_stale_per_flag_label_is_unusable_input() {
    let tmp = write_tmp_json(
        "gptq-flg-stale",
        r#"{ "flags": {
               "argv": ["--method", "gptq"],
               "expected_outcome": "missing_bits"
             } }"#,
    );
    let out = apr_binary()
        .args(["gptq-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr gptq-lint");
    assert_eq!(
        out.status.code(),
        Some(4),
        "stale vocabulary is unusable input (exit 4); stderr={}",
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
          "flags":       { "argv": ["m.apr","--scheme","int4","-o","o.apr"], "expected_outcome": "accepted" }
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
