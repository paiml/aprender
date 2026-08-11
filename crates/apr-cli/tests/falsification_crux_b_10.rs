//! E2E falsification tests for `apr nf4-lint` (CRUX-B-10).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #971: exercise the CLI surface
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

// bnb canonical NF4 codebook (16 entries)
const NF4_CANONICAL: &str = "[-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, \
                             -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, \
                             0.07958029955625534, 0.16093020141124725, 0.24611230194568634, \
                             0.33791524171829224, 0.44070982933044434, 0.5626170039176941, \
                             0.7229568362236023, 1.0]";

// ---- help surface (g2 proof) ----------------------------------------------

#[test]
fn falsify_crux_b_10_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["nf4-lint", "--help"])
        .output()
        .expect("run apr nf4-lint --help");
    assert!(out.status.success(), "apr nf4-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_b_10_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("nf4-lint")
        .output()
        .expect("run apr nf4-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr nf4-lint` must exit non-zero"
    );
}

// ---- codebook gate (FALSIFY-CRUX-B-10-001) --------------------------------

#[test]
fn falsify_crux_b_10_001_codebook_ok_on_canonical() {
    let body = format!(r#"{{ "codebook": {{ "expected": {NF4_CANONICAL} }} }}"#);
    let tmp = write_tmp_json("nf4-cb-ok", &body);
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(
        out.status.success(),
        "canonical codebook must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_10_001_codebook_without_expected_is_vacuous_not_a_pass() {
    // This test used to assert `success()` — that a `codebook` section carrying
    // no `expected` array "falls back to a length check" and passes. #2449 made
    // a section that supplies no expectation VACUOUS and non-zero, because a
    // gate that compared the codebook against nothing has discharged nothing.
    // The assertion was never updated, so this target has been red on main; it
    // is not run by CI (`workspace-test` runs `--lib` plus a fixed 18-command
    // chain that does not include it). Asserting the old shape here would lock
    // the pre-#2449 defect back in.
    let tmp = write_tmp_json("nf4-cb-implicit", r#"{ "codebook": {} }"#);
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a codebook section with no `expected` proves nothing and must not pass; stderr={stderr}"
    );
    assert!(
        stderr.contains("VACUOUS"),
        "the rejection must name the reason, not just fail; stderr={stderr}"
    );
    assert_eq!(
        out.status.code(),
        Some(5),
        "a gate verdict is exit 5 — distinct from a missing (3) or unparseable (4) \
         observation; stderr={stderr}"
    );
}

#[test]
fn falsify_crux_b_10_001_codebook_rejects_length_mismatch() {
    let tmp = write_tmp_json(
        "nf4-cb-short",
        r#"{ "codebook": { "expected": [-1.0, 0.0, 1.0] } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-10-001"));
}

#[test]
fn falsify_crux_b_10_001_codebook_rejects_wrong_values() {
    // 16 entries but intentionally wrong
    let tmp = write_tmp_json(
        "nf4-cb-bad",
        r#"{ "codebook": { "expected": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6] } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-10-001"));
}

// ---- roundtrip gate (FALSIFY-CRUX-B-10-003) -------------------------------

#[test]
fn falsify_crux_b_10_003_roundtrip_ok_on_small_block() {
    let tmp = write_tmp_json(
        "nf4-rt-ok",
        r#"{ "roundtrip": { "weights": [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], "max_rel_l2": 0.5 } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(
        out.status.success(),
        "small-block roundtrip within envelope must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_10_003_roundtrip_rejects_tight_bound() {
    let tmp = write_tmp_json(
        "nf4-rt-tight",
        r#"{ "roundtrip": { "weights": [0.1, -0.23, 0.37, -0.41], "max_rel_l2": 0.00001 } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-10-003"));
}

#[test]
fn falsify_crux_b_10_003_roundtrip_rejects_empty_weights() {
    let tmp = write_tmp_json(
        "nf4-rt-empty",
        r#"{ "roundtrip": { "weights": [], "max_rel_l2": 0.15 } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-10-003"));
}

// ---- storage gate (FALSIFY-CRUX-B-10-002) ---------------------------------

#[test]
fn falsify_crux_b_10_002_storage_ok_single_quant() {
    let tmp = write_tmp_json(
        "nf4-st-ok",
        r#"{ "storage": {
               "n_weights":                    1000000,
               "block_size":                   64,
               "double_quant":                 false,
               "expected_min_bytes_per_weight": 0.50,
               "expected_max_bytes_per_weight": 0.70
             } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(
        out.status.success(),
        "single-quant storage in envelope must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_10_002_storage_rejects_tight_envelope() {
    let tmp = write_tmp_json(
        "nf4-st-tight",
        r#"{ "storage": {
               "n_weights":                    1000000,
               "block_size":                   64,
               "double_quant":                 false,
               "expected_min_bytes_per_weight": 0.01,
               "expected_max_bytes_per_weight": 0.05
             } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-10-002"));
}

#[test]
fn falsify_crux_b_10_002_storage_rejects_zero_n_weights() {
    let tmp = write_tmp_json(
        "nf4-st-zero",
        r#"{ "storage": { "n_weights": 0, "block_size": 64 } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-10-002"));
}

// ---- parity gate (FALSIFY-CRUX-B-10-004) ----------------------------------

#[test]
fn falsify_crux_b_10_004_parity_ok_on_zero_target() {
    let tmp = write_tmp_json(
        "nf4-par-ok",
        r#"{ "parity": { "target": 0.0, "expected_index": 7 } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(
        out.status.success(),
        "zero target must map to codebook index 7; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_10_004_parity_ok_on_extreme_positive() {
    let tmp = write_tmp_json(
        "nf4-par-pos",
        r#"{ "parity": { "target": 1.0, "expected_index": 15 } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(
        out.status.success(),
        "+1.0 must map to index 15; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_10_004_parity_rejects_wrong_index() {
    let tmp = write_tmp_json(
        "nf4-par-bad",
        r#"{ "parity": { "target": 0.0, "expected_index": 0 } }"#,
    );
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-10-004"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_b_10_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("nf4-empty", "");
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_b_10_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "nf4-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_b_10_observation_without_known_keys_rejected() {
    let tmp = write_tmp_json("nf4-empty-obj", r#"{ "other": 1 }"#);
    let out = apr_binary()
        .args(["nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-10"));
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_b_10_json_output_shape() {
    let body = format!(
        r#"{{
          "codebook": {{ "expected": {NF4_CANONICAL} }},
          "parity":   {{ "target": 0.0, "expected_index": 7 }}
        }}"#
    );
    let tmp = write_tmp_json("nf4-json", &body);
    let out = apr_binary()
        .args(["--json", "nf4-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr nf4-lint --json");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"codebook\""));
    assert!(stdout.contains("\"parity\""));
    assert!(stdout.contains("CRUX-B-10"));
}
