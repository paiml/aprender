//! E2E falsification tests for `apr imatrix-lint` (CRUX-B-07).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #968: exercise the CLI surface
//! end-to-end on captured JSON observations and assert the classifier
//! verdicts + non-zero exit codes on known-bad input.
//!
//! Observation shape:
//! ```jsonc
//! {
//!   "improvement": { "ppl_naive", "ppl_calib", "threshold" },   // FALSIFY-001
//!   "leakage":     { "calib_hashes": [..], "eval_hashes": [..] },// FALSIFY-001 invariant
//!   "flags":       { "argv": [..], "expected_path": "..." },    // FALSIFY-002
//!   "provenance":  { "calib_bytes_utf8"|"expected_sha256",
//!                    "recorded": "..." }                         // FALSIFY-003
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
fn falsify_crux_b_07_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["imatrix-lint", "--help"])
        .output()
        .expect("run apr imatrix-lint --help");
    assert!(out.status.success(), "apr imatrix-lint --help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_b_07_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("imatrix-lint")
        .output()
        .expect("run apr imatrix-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr imatrix-lint` must exit non-zero"
    );
}

// ---- improvement gate (FALSIFY-CRUX-B-07-001) -----------------------------

#[test]
fn falsify_crux_b_07_001_improvement_ok_at_threshold() {
    // Δ = (100-99.5)/100 = 0.005 = threshold → Improved
    let tmp = write_tmp_json(
        "imat-impr-ok",
        r#"{ "improvement": { "ppl_naive": 100.0, "ppl_calib": 99.5, "threshold": 0.005 } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(
        out.status.success(),
        "Δ at threshold must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_07_001_improvement_rejects_below_threshold() {
    // Δ = (100-99.8)/100 = 0.002 < 0.005 → Insufficient
    let tmp = write_tmp_json(
        "imat-impr-bad",
        r#"{ "improvement": { "ppl_naive": 100.0, "ppl_calib": 99.8, "threshold": 0.005 } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "below threshold must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-07-001"));
}

#[test]
fn falsify_crux_b_07_001_improvement_rejects_regression() {
    // Δ = (100-110)/100 = -0.10 → Insufficient
    let tmp = write_tmp_json(
        "imat-impr-regress",
        r#"{ "improvement": { "ppl_naive": 100.0, "ppl_calib": 110.0, "threshold": 0.005 } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "regression must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-07-001"));
}

#[test]
fn falsify_crux_b_07_001_improvement_rejects_zero_baseline() {
    // ppl_naive = 0 → Insufficient (upstream pipeline bug)
    let tmp = write_tmp_json(
        "imat-impr-zero",
        r#"{ "improvement": { "ppl_naive": 0.0, "ppl_calib": 5.0, "threshold": 0.005 } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "zero baseline must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-07-001"));
}

// ---- leakage gate (FALSIFY-CRUX-B-07-004 invariant) -----------------------

#[test]
fn falsify_crux_b_07_001_leakage_ok_disjoint() {
    let tmp = write_tmp_json(
        "imat-leak-ok",
        r#"{ "leakage": { "calib_hashes": ["a", "b"], "eval_hashes": ["c", "d"] } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(
        out.status.success(),
        "disjoint sets must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_07_001_leakage_rejects_overlap() {
    let tmp = write_tmp_json(
        "imat-leak-bad",
        r#"{ "leakage": { "calib_hashes": ["a", "b", "c"], "eval_hashes": ["c", "d"] } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "calib/eval leakage must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    // The leakage invariant is its own falsifiable claim: it must NOT be
    // filed under -001, the perplexity-improvement id (issue #2391).
    assert!(
        stderr.contains("FALSIFY-CRUX-B-07-004"),
        "leakage failure must stamp its own id; got: {stderr}"
    );
    assert!(
        !stderr.contains("FALSIFY-CRUX-B-07-001"),
        "leakage failure filed under the improvement id: {stderr}"
    );
}

// ---- flags gate (FALSIFY-CRUX-B-07-002) -----------------------------------

#[test]
fn falsify_crux_b_07_002_flags_ok_space_form() {
    let tmp = write_tmp_json(
        "imat-flag-sp",
        r#"{ "flags": { "argv": ["quantize", "model.apr", "--imatrix", "calib.jsonl"],
                        "expected_path": "calib.jsonl" } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(
        out.status.success(),
        "space-form flag must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_07_002_flags_ok_equals_form() {
    let tmp = write_tmp_json(
        "imat-flag-eq",
        r#"{ "flags": { "argv": ["quantize", "--imatrix=calib.jsonl"],
                        "expected_path": "calib.jsonl" } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(
        out.status.success(),
        "equals-form flag must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_07_002_flags_ok_absent_as_expected() {
    // Observer captured an argv WITHOUT --imatrix and expected null
    let tmp = write_tmp_json(
        "imat-flag-absent",
        r#"{ "flags": { "argv": ["quantize", "--method", "q4k"],
                        "expected_path": null } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(
        out.status.success(),
        "absent-as-expected must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_07_002_flags_rejects_wrong_path() {
    // argv has --imatrix foo but observer expected bar → mismatch
    let tmp = write_tmp_json(
        "imat-flag-mismatch",
        r#"{ "flags": { "argv": ["quantize", "--imatrix", "foo.jsonl"],
                        "expected_path": "bar.jsonl" } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "path mismatch must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-07-002"));
}

#[test]
fn falsify_crux_b_07_002_flags_rejects_similar_named_flag() {
    // --imatrix-force must NOT match; observer expected "calib.jsonl"
    let tmp = write_tmp_json(
        "imat-flag-similar",
        r#"{ "flags": { "argv": ["quantize", "--imatrix-force", "true"],
                        "expected_path": "calib.jsonl" } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "similar-named flag must not match");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-07-002"));
}

#[test]
fn falsify_crux_b_07_002_flags_rejects_trailing_no_value() {
    // --imatrix at end of argv with no value → parser returns None
    let tmp = write_tmp_json(
        "imat-flag-noval",
        r#"{ "flags": { "argv": ["quantize", "model.apr", "--imatrix"],
                        "expected_path": "calib.jsonl" } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(
        !out.status.success(),
        "trailing --imatrix with no value must fail"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-07-002"));
}

// ---- provenance gate (FALSIFY-CRUX-B-07-003) ------------------------------

#[test]
fn falsify_crux_b_07_003_provenance_match_via_expected_sha256() {
    // Pure literal expected sha256 → Match
    let tmp = write_tmp_json(
        "imat-prov-match",
        r#"{ "provenance": {
               "expected_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
               "recorded":        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
             } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(
        out.status.success(),
        "matching sha256 must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_07_003_provenance_match_is_case_insensitive() {
    let tmp = write_tmp_json(
        "imat-prov-case",
        r#"{ "provenance": {
               "expected_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
               "recorded":        "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855"
             } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(
        out.status.success(),
        "case-insensitive match must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_b_07_003_provenance_rejects_missing_recorded() {
    // expected present, recorded absent → Missing
    let tmp = write_tmp_json(
        "imat-prov-miss",
        r#"{ "provenance": {
               "expected_sha256": "abc123"
             } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "missing recorded must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-07-003"));
}

#[test]
fn falsify_crux_b_07_003_provenance_rejects_mismatch() {
    let tmp = write_tmp_json(
        "imat-prov-bad",
        r#"{ "provenance": {
               "expected_sha256": "aaaa",
               "recorded":        "bbbb"
             } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "sha256 mismatch must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-07-003"));
}

#[test]
fn falsify_crux_b_07_003_provenance_rejects_no_expected_input() {
    // No calib_bytes_utf8 and no expected_sha256 → FAIL
    let tmp = write_tmp_json(
        "imat-prov-noexp",
        r#"{ "provenance": { "recorded": "abc" } }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "missing expected input must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-B-07-003"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_b_07_empty_file_rejected() {
    let tmp = write_tmp_json("imat-empty", "");
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_b_07_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "imatrix-lint",
            "--observation-file",
            "/nonexistent/path/imat.json",
        ])
        .output()
        .expect("run apr imatrix-lint");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_b_07_only_unknown_keys_rejected() {
    let tmp = write_tmp_json(
        "imat-nogates",
        r#"{ "unrelated": 1, "also_irrelevant": "hi" }"#,
    );
    let out = apr_binary()
        .args(["imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint");
    assert!(
        !out.status.success(),
        "observation without any gate keys must be rejected"
    );
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_b_07_json_output_shape() {
    let tmp = write_tmp_json(
        "imat-json",
        r#"{
          "improvement": { "ppl_naive": 100.0, "ppl_calib": 90.0, "threshold": 0.005 },
          "leakage":     { "calib_hashes": ["a"], "eval_hashes": ["b"] },
          "flags":       { "argv": ["quantize", "--imatrix", "calib.jsonl"],
                           "expected_path": "calib.jsonl" },
          "provenance":  { "expected_sha256": "deadbeef", "recorded": "deadbeef" }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "imatrix-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr imatrix-lint --json");
    assert!(
        out.status.success(),
        "all-pass bundle under --json must exit 0; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"contract\""));
    assert!(stdout.contains("\"CRUX-B-07\""));
    assert!(stdout.contains("\"gates\""));
    assert!(stdout.contains("\"improvement\""));
    assert!(stdout.contains("\"leakage\""));
    assert!(stdout.contains("\"flags\""));
    assert!(stdout.contains("\"provenance\""));
    assert!(stdout.contains("FALSIFY-CRUX-B-07-001"));
    assert!(stdout.contains("FALSIFY-CRUX-B-07-002"));
    assert!(stdout.contains("FALSIFY-CRUX-B-07-003"));
    assert!(stdout.contains("FALSIFY-CRUX-B-07-004"));
    // A --json consumer keys on falsify_id: every gate needs its own.
    let ids: Vec<&str> = stdout
        .lines()
        .filter(|l| l.contains("\"falsify_id\""))
        .collect();
    let unique: std::collections::BTreeSet<&&str> = ids.iter().collect();
    assert_eq!(
        ids.len(),
        unique.len(),
        "duplicate falsify_id in --json gates: {ids:?}"
    );
}
