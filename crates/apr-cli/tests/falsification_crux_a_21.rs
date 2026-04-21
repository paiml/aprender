//! CRUX-A-21 — end-to-end falsification harness for `apr shared-cache-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-A-21-{001,002}
//! gate the classifier discharges has a matching e2e JSON observation
//! that the binary must classify exactly as the harness expects.

use serde_json::json;
use std::io::Write;
use std::process::Command;

fn apr_binary() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_apr"));
    cmd.env("NO_COLOR", "1");
    cmd
}

fn write_obs(json_body: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .prefix("crux-a-21-obs-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(serde_json::to_vec_pretty(json_body).unwrap().as_slice())
        .expect("write obs");
    f.flush().expect("flush");
    f
}

const HEX_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HEX_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_a_21_cli_help_advertises_observation_file() {
    let out = apr_binary()
        .args(["shared-cache-lint", "--help"])
        .output()
        .expect("run");
    assert!(out.status.success(), "--help must exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--observation-file"),
        "--help must advertise --observation-file; got:\n{stdout}"
    );
}

#[test]
fn falsify_crux_a_21_cli_bare_invocation_is_usage_error() {
    let out = apr_binary().arg("shared-cache-lint").output().expect("run");
    assert!(
        !out.status.success(),
        "bare invocation must not exit 0 — missing required --observation-file"
    );
}

#[test]
fn falsify_crux_a_21_cli_missing_file_fails_with_stamp() {
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            "/nonexistent/crux-a-21-missing.json",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-A-21") || stderr.contains("not found"),
        "stderr must stamp FALSIFY-CRUX-A-21; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_a_21_cli_empty_file_fails() {
    let tmp = tempfile::Builder::new()
        .prefix("crux-a-21-empty-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_a_21_cli_invalid_json_fails() {
    let mut tmp = tempfile::Builder::new()
        .prefix("crux-a-21-bad-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    tmp.write_all(b"{not json").unwrap();
    tmp.flush().unwrap();
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_a_21_cli_no_gates_fails() {
    let tmp = write_obs(&json!({"unrelated": true}));
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

// ===== FALSIFY-CRUX-A-21-001 dedup (resolve + blob_path_for) =====

#[test]
fn falsify_crux_a_21_001_dedup_env_wins_and_paths_collapse() {
    let obs = json!({
        "dedup": {
            "apr_models_env": "/var/lib/apr/models",
            "home":           "/home/user",
            "expected_root":  "/var/lib/apr/models",
            "sha256_hex_a":   HEX_A,
            "sha256_hex_b":   HEX_A,
            "expected_same_path": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "env wins + identical hashes must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] dedup"));
}

#[test]
fn falsify_crux_a_21_001_dedup_home_fallback_when_env_absent() {
    let obs = json!({
        "dedup": {
            "home":           "/home/u",
            "expected_root":  "/home/u/.apr/models",
            "sha256_hex_a":   HEX_A,
            "sha256_hex_b":   HEX_B,
            "expected_same_path": false
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success(), "home fallback must resolve");
}

#[test]
fn falsify_crux_a_21_001_dedup_wrong_expected_root_fails() {
    let obs = json!({
        "dedup": {
            "apr_models_env": "/var/lib/apr/models",
            "home":           "/home/u",
            "expected_root":  "/home/u/.apr/models"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-21-001"));
}

#[test]
fn falsify_crux_a_21_001_dedup_invalid_hex_fails() {
    let obs = json!({
        "dedup": {
            "apr_models_env": "/var/apr",
            "home":           "/home/u",
            "expected_root":  "/var/apr",
            "sha256_hex_a":   "not-hex",
            "sha256_hex_b":   HEX_A,
            "expected_same_path": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "malformed hex must fail");
}

#[test]
fn falsify_crux_a_21_001_dedup_missing_home_errors() {
    let obs = json!({
        "dedup": {
            "home": "",
            "expected_root": "/anything"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "no env and empty home must fail");
}

// ===== FALSIFY-CRUX-A-21-002 permission (EACCES → exit 13) =====

#[test]
fn falsify_crux_a_21_002_permission_eacces_exit_13() {
    let obs = json!({
        "permission": {
            "kind":              "permission_denied",
            "expected_outcome":  "permission_denied",
            "expected_exit_code": 13,
            "expected_hint_substring": "daemon"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "EACCES → exit 13 must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] permission"));
}

#[test]
fn falsify_crux_a_21_002_permission_not_found_exit_1() {
    let obs = json!({
        "permission": {
            "kind":              "not_found",
            "expected_outcome":  "not_found",
            "expected_exit_code": 1
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_a_21_002_permission_timed_out_is_other() {
    let obs = json!({
        "permission": {
            "kind":              "timed_out",
            "expected_outcome":  "other",
            "expected_exit_code": 1
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_a_21_002_permission_wrong_exit_code_fails() {
    let obs = json!({
        "permission": {
            "kind":              "permission_denied",
            "expected_outcome":  "permission_denied",
            "expected_exit_code": 1
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-21-002"));
}

#[test]
fn falsify_crux_a_21_002_permission_wrong_hint_substring_fails() {
    let obs = json!({
        "permission": {
            "kind":              "permission_denied",
            "expected_outcome":  "permission_denied",
            "expected_exit_code": 13,
            "expected_hint_substring": "kittens"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "wrong hint must fail");
}

#[test]
fn falsify_crux_a_21_002_permission_declaring_ok_on_eacces_fails() {
    let obs = json!({
        "permission": {
            "kind":              "permission_denied",
            "expected_outcome":  "ok"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "EACCES can never be Ok");
}

#[test]
fn falsify_crux_a_21_002_permission_unknown_kind_fails() {
    let obs = json!({
        "permission": {
            "kind": "banana",
            "expected_outcome": "other"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

// ===== Multi-gate + JSON shape =====

#[test]
fn falsify_crux_a_21_multi_gate_all_pass() {
    let obs = json!({
        "dedup": {
            "apr_models_env": "/var/lib/apr/models",
            "home":           "/home/u",
            "expected_root":  "/var/lib/apr/models",
            "sha256_hex_a":   HEX_A,
            "sha256_hex_b":   HEX_A,
            "expected_same_path": true
        },
        "permission": {
            "kind":              "permission_denied",
            "expected_outcome":  "permission_denied",
            "expected_exit_code": 13,
            "expected_hint_substring": "daemon"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "both gates green must exit 0; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] dedup"));
    assert!(stdout.contains("[PASS] permission"));
}

#[test]
fn falsify_crux_a_21_json_output_shape() {
    let obs = json!({
        "dedup": {
            "apr_models_env": "/var/apr",
            "home":           "/home/u",
            "expected_root":  "/var/apr"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "--json",
            "shared-cache-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--json must emit valid JSON");
    assert_eq!(parsed["contract"], "CRUX-A-21");
    assert_eq!(parsed["gates"][0]["gate"], "dedup");
    assert_eq!(parsed["gates"][0]["falsify_id"], "FALSIFY-CRUX-A-21-001");
    assert_eq!(parsed["gates"][0]["passed"], true);
}
