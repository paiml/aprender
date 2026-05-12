//! CRUX-A-25 — end-to-end falsification harness for `apr rm-gc-lint`.
//!
//! CRUX-SHIP-001 gate g3 evidence: every FALSIFY-CRUX-A-25-{001,002,003}
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
        .prefix("crux-a-25-obs-")
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
fn falsify_crux_a_25_cli_help_advertises_observation_file() {
    let out = apr_binary()
        .args(["rm-gc-lint", "--help"])
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
fn falsify_crux_a_25_cli_bare_invocation_is_usage_error() {
    let out = apr_binary().arg("rm-gc-lint").output().expect("run");
    assert!(
        !out.status.success(),
        "bare invocation must not exit 0 — missing required --observation-file"
    );
}

#[test]
fn falsify_crux_a_25_cli_missing_file_fails_with_stamp() {
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            "/nonexistent/crux-a-25-missing.json",
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("FALSIFY-CRUX-A-25") || stderr.contains("not found"),
        "stderr must stamp FALSIFY-CRUX-A-25; got:\n{stderr}"
    );
}

#[test]
fn falsify_crux_a_25_cli_empty_file_fails() {
    let tmp = tempfile::Builder::new()
        .prefix("crux-a-25-empty-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_a_25_cli_invalid_json_fails() {
    let mut tmp = tempfile::Builder::new()
        .prefix("crux-a-25-bad-")
        .suffix(".json")
        .tempfile()
        .unwrap();
    tmp.write_all(b"{not json").unwrap();
    tmp.flush().unwrap();
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

#[test]
fn falsify_crux_a_25_cli_no_gates_fails() {
    let tmp = write_obs(&json!({"unrelated": true}));
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
}

// ===== FALSIFY-CRUX-A-25-001 rm frees unique-owned blobs =====

#[test]
fn falsify_crux_a_25_001_rm_frees_unique_owned_blobs() {
    // Two blobs owned only by gpt2:latest → rm should flag both as orphans.
    let obs = json!({
        "rm": {
            "manifests": [
                { "tag": "gpt2:latest", "blobs": ["sha1", "sha2"] }
            ],
            "tag_to_rm": "gpt2:latest",
            "all_blobs": ["sha1", "sha2"],
            "expected_freed": ["sha1", "sha2"]
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "rm of last referrer must free both blobs; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] rm"));
}

#[test]
fn falsify_crux_a_25_001_rm_absent_tag_frees_nothing() {
    // tag_to_rm doesn't match any manifest → no change, no orphans.
    let obs = json!({
        "rm": {
            "manifests": [
                { "tag": "gpt2:latest", "blobs": ["sha1"] }
            ],
            "tag_to_rm": "nonexistent:tag",
            "all_blobs": ["sha1"],
            "expected_freed": []
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success(), "rm of absent tag must free nothing");
}

#[test]
fn falsify_crux_a_25_001_rm_count_mismatch_fails() {
    // Observer claims freed=[sha1,sha2] but classifier will compute [sha1] only.
    let obs = json!({
        "rm": {
            "manifests": [
                { "tag": "a", "blobs": ["sha1"] },
                { "tag": "b", "blobs": ["sha2"] }
            ],
            "tag_to_rm": "a",
            "all_blobs": ["sha1", "sha2"],
            "expected_freed": ["sha1", "sha2"]
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success(), "wrong expected_freed must fail");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-25-001"));
}

// ===== FALSIFY-CRUX-A-25-002 safety — referenced blobs never freed =====

#[test]
fn falsify_crux_a_25_002_safety_duplicate_tag_frees_nothing() {
    // gpt2:latest and gpt2:dup share sha1; rm of :latest leaves :dup, frees nothing.
    let obs = json!({
        "safety": {
            "manifests": [
                { "tag": "gpt2:latest", "blobs": ["sha1"] },
                { "tag": "gpt2:dup",    "blobs": ["sha1"] }
            ],
            "tag_to_rm": "gpt2:latest",
            "all_blobs": ["sha1"],
            "expected_freed": []
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "duplicate-tag rm must preserve shared blob; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] safety"));
}

#[test]
fn falsify_crux_a_25_002_safety_wrong_expected_fails() {
    // Observer wrongly claims the surviving-shared blob is freed → violates invariant.
    let obs = json!({
        "safety": {
            "manifests": [
                { "tag": "x", "blobs": ["sha1"] },
                { "tag": "y", "blobs": ["sha1"] }
            ],
            "tag_to_rm": "x",
            "all_blobs": ["sha1"],
            "expected_freed": ["sha1"]
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-25-002"));
}

#[test]
fn falsify_crux_a_25_002_safety_preserves_disjoint_manifests() {
    // rm of tag-a must not touch tag-b's blobs; orphans=[sha-a] only.
    let obs = json!({
        "safety": {
            "manifests": [
                { "tag": "a", "blobs": ["sha-a"] },
                { "tag": "b", "blobs": ["sha-b"] }
            ],
            "tag_to_rm": "a",
            "all_blobs": ["sha-a", "sha-b"],
            "expected_freed": ["sha-a"]
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
}

// ===== FALSIFY-CRUX-A-25-003 dry-run purity + idempotence =====

#[test]
fn falsify_crux_a_25_003_dryrun_idempotent_on_orphan() {
    // One orphan blob + no referencing manifests → plan ≠ empty, post-gc is idempotent.
    let obs = json!({
        "dryrun": {
            "manifests": [ { "tag": "x", "blobs": [] } ],
            "all_blobs": ["sha-orphan"],
            "expected_idempotent": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        out.status.success(),
        "dry-run + idempotence must pass; stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("[PASS] dryrun"));
}

#[test]
fn falsify_crux_a_25_003_dryrun_all_live_is_noop() {
    // All blobs referenced → plan is empty, second pass is trivially idempotent.
    let obs = json!({
        "dryrun": {
            "manifests": [ { "tag": "a", "blobs": ["sha1", "sha2"] } ],
            "all_blobs": ["sha1", "sha2"],
            "expected_idempotent": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success(), "all-live must pass");
}

#[test]
fn falsify_crux_a_25_003_dryrun_wrong_expected_fails() {
    // Observer claims non-idempotent; classifier will compute idempotent=true → mismatch.
    let obs = json!({
        "dryrun": {
            "manifests": [ { "tag": "x", "blobs": [] } ],
            "all_blobs": ["sha-orphan"],
            "expected_idempotent": false
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("FALSIFY-CRUX-A-25-003"));
}

#[test]
fn falsify_crux_a_25_003_dryrun_empty_blobs_is_noop() {
    // Nothing to GC at all → plan is empty, idempotent trivially.
    let obs = json!({
        "dryrun": {
            "manifests": [],
            "all_blobs": [],
            "expected_idempotent": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
}

// ===== Multi-gate + JSON shape =====

#[test]
fn falsify_crux_a_25_multi_gate_all_pass() {
    let obs = json!({
        "rm": {
            "manifests": [ { "tag": "gpt2:latest", "blobs": ["sha1"] } ],
            "tag_to_rm": "gpt2:latest",
            "all_blobs": ["sha1"],
            "expected_freed": ["sha1"]
        },
        "safety": {
            "manifests": [
                { "tag": "a", "blobs": ["shared"] },
                { "tag": "b", "blobs": ["shared"] }
            ],
            "tag_to_rm": "a",
            "all_blobs": ["shared"],
            "expected_freed": []
        },
        "dryrun": {
            "manifests": [ { "tag": "z", "blobs": [] } ],
            "all_blobs": ["sha-orphan"],
            "expected_idempotent": true
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "rm-gc-lint",
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
    assert!(stdout.contains("[PASS] rm"));
    assert!(stdout.contains("[PASS] safety"));
    assert!(stdout.contains("[PASS] dryrun"));
}

#[test]
fn falsify_crux_a_25_json_output_shape() {
    let obs = json!({
        "rm": {
            "manifests": [ { "tag": "gpt2:latest", "blobs": ["sha1"] } ],
            "tag_to_rm": "gpt2:latest",
            "all_blobs": ["sha1"],
            "expected_freed": ["sha1"]
        }
    });
    let tmp = write_obs(&obs);
    let out = apr_binary()
        .args([
            "--json",
            "rm-gc-lint",
            "--observation-file",
            tmp.path().to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout).expect("--json must emit valid JSON");
    assert_eq!(parsed["contract"], "CRUX-A-25");
    assert_eq!(parsed["gates"][0]["gate"], "rm");
    assert_eq!(parsed["gates"][0]["falsify_id"], "FALSIFY-CRUX-A-25-001");
    assert_eq!(parsed["gates"][0]["passed"], true);
}
