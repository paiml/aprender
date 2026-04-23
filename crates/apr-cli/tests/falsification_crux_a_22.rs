//! CRUX-A-22 — end-to-end falsification harness for `apr registry-quota-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-A-22-{001,002,003}
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
        .prefix("crux-a-22-obs-")
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(serde_json::to_vec_pretty(json_body).unwrap().as_slice())
        .expect("write obs");
    f.flush().expect("flush");
    f
}

// ===== g2: CLI shape =====

#[test]
fn falsify_crux_a_22_cli_help_advertises_observation_file() {
    let out = apr_binary()
        .args(["registry-quota-lint", "--help"])
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
fn falsify_crux_a_22_cli_bare_invocation_is_usage_error() {
    let out = apr_binary().arg("registry-quota-lint").output().expect("run");
    assert!(
        !out.status.success(),
        "bare invocation must not exit 0 — missing required --observation-file"
    );
}

#[test]
fn falsify_crux_a_22_cli_missing_file_fails_with_stamp() {
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            "/nonexistent/crux-a-22-missing.json",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-A-22") || stderr.contains("not found"),
        "stderr must stamp FALSIFY-CRUX-A-22; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_a_22_cli_empty_file_fails() {
    let tmp = tempfile::Builder::new()
        .prefix("crux-a-22-empty-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_a_22_cli_invalid_json_fails() {
    let mut tmp = tempfile::Builder::new()
        .prefix("crux-a-22-bad-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    tmp.write_all(b"{not json").unwrap();
    tmp.flush().unwrap();
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_a_22_cli_no_gates_fails() {
    let tmp = write_obs(&json!({"unrelated": true}));
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

// ===== FALSIFY-CRUX-A-22-001 quota decision + error body =====

#[test]
fn falsify_crux_a_22_001_reject_one_byte_over() {
    let obs = json!({
        "quota": {
            "quota": 1000,
            "used": 600,
            "incoming": 401,
            "expected_outcome": "reject"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "reject verdict must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] quota"));
}

#[test]
fn falsify_crux_a_22_001_allow_under_budget() {
    let obs = json!({
        "quota": {
            "quota": 1000,
            "used": 200,
            "incoming": 300,
            "expected_outcome": "allow"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_a_22_001_wrong_expected_outcome_fails() {
    // Classifier returns allow, observer claimed reject → FAIL.
    let obs = json!({
        "quota": {
            "quota": 1000,
            "used": 200,
            "incoming": 300,
            "expected_outcome": "reject"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-22-001"));
}

// ===== FALSIFY-CRUX-A-22-002 atomic (pre-download purity + determinism) =====

#[test]
fn falsify_crux_a_22_002_atomic_reject_is_deterministic() {
    let obs = json!({
        "atomic": {
            "quota": 500,
            "used": 400,
            "incoming": 200,
            "expected_outcome": "reject"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "deterministic reject must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] atomic"));
}

#[test]
fn falsify_crux_a_22_002_atomic_allow_is_deterministic() {
    let obs = json!({
        "atomic": {
            "quota": 1_000_000,
            "used": 0,
            "incoming": 100,
            "expected_outcome": "allow"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_a_22_002_atomic_wrong_outcome_fails() {
    let obs = json!({
        "atomic": {
            "quota": 500,
            "used": 400,
            "incoming": 200,
            "expected_outcome": "allow"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-22-002"));
}

// ===== FALSIFY-CRUX-A-22-003 ceiling (post-pull invariant) =====

#[test]
fn falsify_crux_a_22_003_ceiling_allow_preserves_invariant() {
    let obs = json!({
        "ceiling": {
            "quota": 1000,
            "used": 200,
            "incoming": 300,
            "expected_outcome": "allow",
            "expected_post_used_le_quota": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "allow + post_used ≤ quota must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] ceiling"));
}

#[test]
fn falsify_crux_a_22_003_ceiling_exact_fit_allow() {
    // used + incoming == quota → Allow, invariant holds with equality.
    let obs = json!({
        "ceiling": {
            "quota": 1000,
            "used": 600,
            "incoming": 400,
            "expected_outcome": "allow",
            "expected_post_used_le_quota": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success(), "exact fit is Allow");
}

#[test]
fn falsify_crux_a_22_003_ceiling_reject_invariant_false() {
    // Reject verdict → post_used would exceed quota; invariant=false.
    let obs = json!({
        "ceiling": {
            "quota": 1000,
            "used": 900,
            "incoming": 200,
            "expected_outcome": "reject",
            "expected_post_used_le_quota": false
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success(), "reject + invariant=false must pass");
}

#[test]
fn falsify_crux_a_22_003_ceiling_wrong_invariant_fails() {
    // Observer claims invariant holds even though verdict is Reject.
    let obs = json!({
        "ceiling": {
            "quota": 1000,
            "used": 900,
            "incoming": 200,
            "expected_outcome": "reject",
            "expected_post_used_le_quota": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-22-003"));
}

// ===== Multi-gate + JSON shape =====

#[test]
fn falsify_crux_a_22_multi_gate_all_pass() {
    let obs = json!({
        "quota": {
            "quota": 1000, "used": 600, "incoming": 401,
            "expected_outcome": "reject"
        },
        "atomic": {
            "quota": 1000, "used": 600, "incoming": 401,
            "expected_outcome": "reject"
        },
        "ceiling": {
            "quota": 1000, "used": 200, "incoming": 300,
            "expected_outcome": "allow",
            "expected_post_used_le_quota": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "all three gates green must exit 0; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] quota"));
    assert!(stdout.contains("[PASS] atomic"));
    assert!(stdout.contains("[PASS] ceiling"));
}

#[test]
fn falsify_crux_a_22_json_output_shape() {
    let obs = json!({
        "quota": {
            "quota": 1000, "used": 600, "incoming": 401,
            "expected_outcome": "reject"
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "--json",
            "registry-quota-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--json must emit valid JSON");
    assert_eq!(parsed["contract"], "CRUX-A-22");
    assert_eq!(parsed["gates"][0]["gate"], "quota");
    assert_eq!(parsed["gates"][0]["falsify_id"], "FALSIFY-CRUX-A-22-001");
    assert_eq!(parsed["gates"][0]["passed"], true);
}
