//! CRUX-A-23 — end-to-end falsification harness for `apr unified-search-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-A-23-{001,002}
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
        .prefix("crux-a-23-obs-")
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
fn falsify_crux_a_23_cli_help_advertises_observation_file() {
    let out = apr_binary()
        .args(["unified-search-lint", "--help"])
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
fn falsify_crux_a_23_cli_bare_invocation_is_usage_error() {
    let out = apr_binary()
        .arg("unified-search-lint")
        .output()
        .expect("run");
    assert!(
        !out.status.success(),
        "bare invocation must not exit 0 — missing required --observation-file"
    );
}

#[test]
fn falsify_crux_a_23_cli_missing_file_fails_with_stamp() {
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            "/nonexistent/crux-a-23-missing.json",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-A-23") || stderr.contains("not found"),
        "stderr must stamp FALSIFY-CRUX-A-23; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_a_23_cli_empty_file_fails() {
    let tmp = tempfile::Builder::new()
        .prefix("crux-a-23-empty-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_a_23_cli_invalid_json_fails() {
    let mut tmp = tempfile::Builder::new()
        .prefix("crux-a-23-bad-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    tmp.write_all(b"{bad json").unwrap();
    tmp.flush().unwrap();
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_a_23_cli_no_gates_fails() {
    let tmp = write_obs(&json!({"unrelated": true}));
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

// ===== FALSIFY-CRUX-A-23-001 offline =====

#[test]
fn falsify_crux_a_23_001_offline_local_only_surfaces() {
    // Hub absent (offline); local rows must appear with source=LOCAL.
    let obs = json!({
        "offline": {
            "local": [
                { "repo": "gpt2",              "cached": true },
                { "repo": "bert-base-uncased", "cached": true }
            ],
            "expected_count": 2,
            "expected_sources": { "gpt2": "LOCAL", "bert-base-uncased": "LOCAL" }
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "offline with 2 local rows must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] offline"));
}

#[test]
fn falsify_crux_a_23_001_offline_empty_hub_and_local_is_empty() {
    // Classifier returns 0 rows, observer pre-declared expected_count=0 — PASS.
    let obs = json!({
        "offline": {
            "local": [],
            "expected_count": 0,
            "expected_sources": {}
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success(), "empty-both expected empty must pass");
}

#[test]
fn falsify_crux_a_23_001_offline_count_mismatch_fails() {
    let obs = json!({
        "offline": {
            "local": [{ "repo": "gpt2", "cached": true }],
            "expected_count": 2
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "count mismatch must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-23-001"));
}

#[test]
fn falsify_crux_a_23_001_offline_wrong_source_fails() {
    // local-only repo but observer claimed source=HUB → classifier reports LOCAL → FAIL.
    let obs = json!({
        "offline": {
            "local": [{ "repo": "gpt2", "cached": true }],
            "expected_count": 1,
            "expected_sources": { "gpt2": "HUB" }
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "wrong expected source must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-23-001"));
}

// ===== FALSIFY-CRUX-A-23-002 dedup =====

#[test]
fn falsify_crux_a_23_002_dedup_repo_in_both_halves_is_both() {
    let obs = json!({
        "dedup": {
            "hub":   [{ "repo": "gpt2", "downloads": 1000, "likes": 10 }],
            "local": [{ "repo": "gpt2", "cached": true }],
            "expected_count": 1,
            "expected_sources": { "gpt2": "BOTH" }
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "hub+local overlap must collapse to BOTH; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn falsify_crux_a_23_002_dedup_disjoint_halves_both_appear() {
    let obs = json!({
        "dedup": {
            "hub":   [{ "repo": "gpt2", "downloads": 500 }],
            "local": [{ "repo": "bert", "cached": true }],
            "expected_count": 2,
            "expected_sources": { "gpt2": "HUB", "bert": "LOCAL" }
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
}

#[test]
fn falsify_crux_a_23_002_dedup_count_mismatch_fails() {
    let obs = json!({
        "dedup": {
            "hub":   [{ "repo": "gpt2" }],
            "local": [{ "repo": "gpt2" }],
            "expected_count": 2
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "dedup forces count=1, not 2");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-23-002"));
}

#[test]
fn falsify_crux_a_23_002_dedup_source_both_not_hub() {
    // Overlap → source=BOTH; declaring HUB must FAIL.
    let obs = json!({
        "dedup": {
            "hub":   [{ "repo": "gpt2" }],
            "local": [{ "repo": "gpt2" }],
            "expected_count": 1,
            "expected_sources": { "gpt2": "HUB" }
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "BOTH claimed as HUB must fail");
}

#[test]
fn falsify_crux_a_23_002_dedup_missing_repo_fails() {
    let obs = json!({
        "dedup": {
            "hub":   [{ "repo": "gpt2" }],
            "local": [],
            "expected_sources": { "nonexistent-repo": "HUB" }
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "declaring missing repo must fail");
}

// ===== Multi-gate + JSON shape =====

#[test]
fn falsify_crux_a_23_multi_gate_all_pass() {
    let obs = json!({
        "offline": {
            "local": [{ "repo": "gpt2", "cached": true }],
            "expected_count": 1,
            "expected_sources": { "gpt2": "LOCAL" }
        },
        "dedup": {
            "hub":   [{ "repo": "bert" }],
            "local": [{ "repo": "bert", "cached": true }],
            "expected_count": 1,
            "expected_sources": { "bert": "BOTH" }
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "unified-search-lint",
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
    assert!(stdout.contains("[PASS] offline"));
    assert!(stdout.contains("[PASS] dedup"));
}

#[test]
fn falsify_crux_a_23_json_output_shape() {
    let obs = json!({
        "offline": {
            "local": [{ "repo": "gpt2", "cached": true }],
            "expected_count": 1
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "--json",
            "unified-search-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--json must emit valid JSON");
    assert_eq!(parsed["contract"], "CRUX-A-23");
    assert_eq!(parsed["gates"][0]["gate"], "offline");
    assert_eq!(parsed["gates"][0]["falsify_id"], "FALSIFY-CRUX-A-23-001");
    assert_eq!(parsed["gates"][0]["passed"], true);
}
