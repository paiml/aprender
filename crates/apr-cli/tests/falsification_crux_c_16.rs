//! E2E falsification tests for `apr lora-hotswap-lint` (CRUX-C-16).
//!
//! Discharges g3 of CRUX-SHIP-001 for PR #980: exercise the CLI surface
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
fn falsify_crux_c_16_help_advertises_observation_file_flag() {
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--help"])
        .output()
        .expect("run apr lora-hotswap-lint --help");
    assert!(
        out.status.success(),
        "apr lora-hotswap-lint --help must exit 0"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_c_16_rejects_bare_invocation_without_file() {
    let out = apr_binary()
        .arg("lora-hotswap-lint")
        .output()
        .expect("run apr lora-hotswap-lint without args");
    assert!(
        !out.status.success(),
        "bare `apr lora-hotswap-lint` must exit non-zero"
    );
}

// ---- hotswap_parity gate (FALSIFY-CRUX-C-16-001) --------------------------

#[test]
fn falsify_crux_c_16_001_hotswap_parity_ok_on_identical() {
    let tmp = write_tmp_json(
        "lora-parity-ok",
        r#"{ "hotswap_parity": { "merged_tokens": [1,2,3], "hotswap_tokens": [1,2,3] } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(
        out.status.success(),
        "identical tokens must pass; stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_16_001_hotswap_parity_rejects_token_divergence() {
    let tmp = write_tmp_json(
        "lora-parity-div",
        r#"{ "hotswap_parity": { "merged_tokens": [1,2,3], "hotswap_tokens": [1,9,3] } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(!out.status.success(), "divergent tokens must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-C-16-001"),
        "stderr must cite FALSIFY-CRUX-C-16-001; got:\n{stderr}"
    );
}

// ---- load_latency gate (FALSIFY-CRUX-C-16-002) ----------------------------

#[test]
fn falsify_crux_c_16_002_load_latency_ok_under_budget() {
    let tmp = write_tmp_json(
        "lora-lat-ok",
        r#"{ "load_latency": { "samples_seconds": [0.1, 0.15, 0.2], "budget_seconds": 2.0 } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_c_16_002_load_latency_rejects_exceeded() {
    // 9x 0.1 + 1x 2.5 → 10-sample nearest-rank P99 = 2.5 > budget 2.0
    let tmp = write_tmp_json(
        "lora-lat-ex",
        r#"{ "load_latency": {
             "samples_seconds": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 2.5],
             "budget_seconds": 2.0
           } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(!out.status.success(), "exceeded budget must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-16-002"));
}

#[test]
fn falsify_crux_c_16_002_load_latency_rejects_invalid_budget() {
    let tmp = write_tmp_json(
        "lora-lat-bad-budget",
        r#"{ "load_latency": { "samples_seconds": [0.1], "budget_seconds": 0.0 } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(!out.status.success(), "zero budget must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-16-002"));
}

// ---- adapter_compat gate (FALSIFY-CRUX-C-16-003) --------------------------

#[test]
fn falsify_crux_c_16_003_adapter_compat_ok_on_matching() {
    let tmp = write_tmp_json(
        "lora-compat-ok",
        r#"{ "adapter_compat": {
             "base_sha256": "abc123",
             "adapter_base_sha256": "abc123",
             "base_module_names": ["q_proj", "k_proj", "v_proj"],
             "adapter_target_modules": ["q_proj", "v_proj"],
             "adapter_rank": 64
           } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_c_16_003_adapter_compat_rejects_sha_mismatch() {
    let tmp = write_tmp_json(
        "lora-compat-sha",
        r#"{ "adapter_compat": {
             "base_sha256": "abc123",
             "adapter_base_sha256": "def456",
             "base_module_names": ["q_proj"],
             "adapter_target_modules": ["q_proj"],
             "adapter_rank": 16
           } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-16-003"));
}

#[test]
fn falsify_crux_c_16_003_adapter_compat_rejects_rank_too_large() {
    let tmp = write_tmp_json(
        "lora-compat-rank",
        r#"{ "adapter_compat": {
             "base_sha256": "abc",
             "adapter_base_sha256": "abc",
             "base_module_names": ["q_proj"],
             "adapter_target_modules": ["q_proj"],
             "adapter_rank": 1024
           } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-16-003"));
}

#[test]
fn falsify_crux_c_16_003_adapter_compat_rejects_unknown_modules() {
    let tmp = write_tmp_json(
        "lora-compat-unknown",
        r#"{ "adapter_compat": {
             "base_sha256": "abc",
             "adapter_base_sha256": "abc",
             "base_module_names": ["q_proj", "k_proj"],
             "adapter_target_modules": ["q_proj", "x_proj"],
             "adapter_rank": 16
           } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-16-003"));
}

// ---- unload_restore gate (FALSIFY-CRUX-C-16-004) --------------------------

#[test]
fn falsify_crux_c_16_004_unload_restore_ok_on_identical() {
    let tmp = write_tmp_json(
        "lora-unload-ok",
        r#"{ "unload_restore": { "fresh_tokens": [1,2,3], "after_unload_tokens": [1,2,3] } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_c_16_004_unload_restore_rejects_divergence() {
    let tmp = write_tmp_json(
        "lora-unload-div",
        r#"{ "unload_restore": { "fresh_tokens": [1,2,3], "after_unload_tokens": [1,9,3] } }"#,
    );
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-C-16-004"));
}

// ---- input validation -----------------------------------------------------

#[test]
fn falsify_crux_c_16_empty_file_rejected_via_cli() {
    let tmp = write_tmp_json("lora-empty", "");
    let out = apr_binary()
        .args(["lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(!out.status.success(), "empty file must be rejected");
}

#[test]
fn falsify_crux_c_16_nonexistent_file_rejected() {
    let out = apr_binary()
        .args([
            "lora-hotswap-lint",
            "--observation-file",
            "/nonexistent/path/obs.json",
        ])
        .output()
        .expect("run apr lora-hotswap-lint");
    assert!(!out.status.success());
}

// ---- --json shape ---------------------------------------------------------

#[test]
fn falsify_crux_c_16_json_output_shape() {
    let tmp = write_tmp_json(
        "lora-json-shape",
        r#"{
          "hotswap_parity": { "merged_tokens": [1], "hotswap_tokens": [1] },
          "load_latency":   { "samples_seconds": [0.1], "budget_seconds": 2.0 },
          "adapter_compat": {
            "base_sha256": "abc",
            "adapter_base_sha256": "abc",
            "base_module_names": ["q_proj"],
            "adapter_target_modules": ["q_proj"],
            "adapter_rank": 16
          },
          "unload_restore": { "fresh_tokens": [1], "after_unload_tokens": [1] }
        }"#,
    );
    let out = apr_binary()
        .args(["--json", "lora-hotswap-lint", "--observation-file"])
        .arg(tmp.path())
        .output()
        .expect("run apr lora-hotswap-lint --json");
    assert!(
        out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"hotswap_parity\""),
        "--json must emit hotswap_parity key"
    );
    assert!(stdout.contains("\"load_latency\""));
    assert!(stdout.contains("\"adapter_compat\""));
    assert!(stdout.contains("\"unload_restore\""));
}
