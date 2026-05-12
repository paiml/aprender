//! E2E falsification tests for `apr typical-p-lint` (CRUX-C-22).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #982: exercise the CLI surface
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
fn falsify_crux_c_22_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["typical-p-lint", "--help"])
        .output()
        .expect("run apr typical-p-lint --help");
    assert!(
        out.status.success(),
        "apr typical-p-lint --help must exit 0"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_c_22_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("typical-p-lint")
        .output()
        .expect("run apr typical-p-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr typical-p-lint` must exit non-zero"
    );
}

// ---- range gate (FALSIFY-CRUX-C-22-001 — parameter range) -----------------

#[test]
fn falsify_crux_c_22_001_range_ok_on_valid_p() {
    let tmp = write_tmp_json("typ-range-ok", r#"{ "range": { "p": 0.95 } }"#);
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(
        out.status.success(),
        "p=0.95 must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_22_001_range_rejects_zero_p() {
    let tmp = write_tmp_json("typ-range-0", r#"{ "range": { "p": 0.0 } }"#);
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(!out.status.success(), "p=0 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-22-001"));
}

#[test]
fn falsify_crux_c_22_001_range_rejects_above_one() {
    let tmp = write_tmp_json("typ-range-above", r#"{ "range": { "p": 1.5 } }"#);
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-22-001"));
}

// ---- identity gate (FALSIFY-CRUX-C-22-001) --------------------------------

#[test]
fn falsify_crux_c_22_001_identity_ok_on_all_kept() {
    let tmp = write_tmp_json(
        "typ-id-ok",
        r#"{ "identity": { "kept_indices": [0,1,2,3], "total_tokens": 4, "p": 1.0 } }"#,
    );
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_c_22_001_identity_rejects_dropped_tokens() {
    let tmp = write_tmp_json(
        "typ-id-drop",
        r#"{ "identity": { "kept_indices": [0,1], "total_tokens": 4, "p": 1.0 } }"#,
    );
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(!out.status.success(), "dropped tokens at p=1.0 must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-22-001"));
}

// ---- mass coverage gate (FALSIFY-CRUX-C-22-002) ---------------------------

#[test]
fn falsify_crux_c_22_002_mass_ok_when_meets_threshold() {
    let tmp = write_tmp_json(
        "typ-mass-ok",
        r#"{ "mass": { "kept_probs": [0.5, 0.3, 0.15], "p": 0.9 } }"#,
    );
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_c_22_002_mass_rejects_insufficient() {
    let tmp = write_tmp_json(
        "typ-mass-low",
        r#"{ "mass": { "kept_probs": [0.3, 0.2], "p": 0.9 } }"#,
    );
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-22-002"));
}

// ---- renormalization gate (FALSIFY-CRUX-C-22-002) -------------------------

#[test]
fn falsify_crux_c_22_002_renorm_ok_when_sums_to_one() {
    let tmp = write_tmp_json(
        "typ-renorm-ok",
        r#"{ "renorm": { "filtered_probs": [0.6, 0.4] } }"#,
    );
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_c_22_002_renorm_rejects_under_normalized() {
    let tmp = write_tmp_json(
        "typ-renorm-under",
        r#"{ "renorm": { "filtered_probs": [0.3, 0.3] } }"#,
    );
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-22-002"));
}

// ---- sort order gate (FALSIFY-CRUX-C-22-002) ------------------------------

#[test]
fn falsify_crux_c_22_002_sort_ok_on_uniform() {
    // uniform: all c_i equal → any order is valid
    let tmp = write_tmp_json(
        "typ-sort-uniform",
        r#"{ "sort": {
             "all_probs": [0.25, 0.25, 0.25, 0.25],
             "kept_probs_in_sort_order": [0.25, 0.25]
           } }"#,
    );
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_c_22_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("typ-empty", "");
    let out = apr_binary()
        .args(["typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_c_22_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "typical-p-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr typical-p-lint");
    assert!(!out.status.success());
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_c_22_json_output_shape() {
    let tmp = write_tmp_json(
        "typ-json-shape",
        r#"{
          "range":    { "p": 0.95 },
          "identity": { "kept_indices": [0,1,2], "total_tokens": 3, "p": 1.0 },
          "mass":     { "kept_probs": [0.5, 0.3, 0.15], "p": 0.9 },
          "renorm":   { "filtered_probs": [0.6, 0.4] }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "typical-p-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr typical-p-lint --json");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"range\""));
    assert!(stdout.contains("\"identity\""));
    assert!(stdout.contains("\"mass\""));
    assert!(stdout.contains("\"renorm\""));
}
