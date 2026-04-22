//! End-to-end falsification tests for CRUX-F-09 — Gradient-norm telemetry per step.
//!
//! Contract: `contracts/crux-F-09-v1.yaml` (v1.1.0).
//!
//! CRUX-SHIP-001 compliance:
//! - g1_classifier_green: `aprender::metrics::grad_norm` unit tests in-crate.
//! - g2_cli_reachable: `apr grad-norm --help` surfaces `--history-file`.
//! - g3_e2e_runs: subprocess invocation of the real binary exercises
//!   classifier-via-CLI end-to-end. Live training-loop hook (one JSON line
//!   per step from `apr finetune`/`apr pretrain`) is PARTIAL_ALGORITHM_LEVEL
//!   under BLOCKER-UPSTREAM-MISSING.

#![allow(clippy::unwrap_used)]

use std::io::Write;

use assert_cmd::Command;
use predicates::prelude::*;

fn write_history(records: &serde_json::Value) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    let body = serde_json::to_vec(records).unwrap();
    f.write_all(&body).unwrap();
    f.flush().unwrap();
    f
}

// ═══ g2_cli_reachable ═══

#[test]
fn falsify_crux_f_09_help_advertises_history_file_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["grad-norm", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--history-file"));
}

#[test]
fn falsify_crux_f_09_help_advertises_max_grad_norm_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["grad-norm", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--max-grad-norm"));
}

#[test]
fn falsify_crux_f_09_rejects_bare_invocation_without_file() {
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["grad-norm"])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "bare `apr grad-norm` (no --history-file) must fail"
    );
}

// ═══ g3_e2e_runs: classifier-via-CLI ═══

#[test]
fn falsify_crux_f_09_json_emits_aggregate_keys() {
    // Build a clean 20-step history, all norms around 1.0, no spikes.
    let mut recs = Vec::new();
    for step in 0..20u64 {
        recs.push(serde_json::json!({
            "step": step,
            "grad_norm": 1.0 + (step as f64) * 0.001,
            "grad_norm_clipped": 0.9,
            "loss": 2.0,
        }));
    }
    let f = write_history(&serde_json::Value::Array(recs));

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "--json",
            "grad-norm",
            "--history-file",
            f.path().to_str().unwrap(),
            "--spike-window",
            "8",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        output.status.success(),
        "apr --json grad-norm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    for key in [
        "num_steps",
        "min",
        "max",
        "mean",
        "num_spikes",
        "all_non_negative",
        "clipping_non_expansive",
        "max_exceeds_cap",
    ] {
        assert!(v.get(key).is_some(), "missing key {key} in {stdout}");
    }
    assert_eq!(v["num_steps"].as_u64(), Some(20));
    assert_eq!(v["num_spikes"].as_u64(), Some(0));
    assert_eq!(v["all_non_negative"].as_bool(), Some(true));
    assert_eq!(v["clipping_non_expansive"].as_bool(), Some(true));
}

#[test]
fn falsify_crux_f_09_detects_spike_via_cli() {
    // 16 flat steps at 1.0 then one spike at 100.0 (>10x median).
    let mut recs = Vec::new();
    for step in 0..16u64 {
        recs.push(serde_json::json!({
            "step": step,
            "grad_norm": 1.0,
            "grad_norm_clipped": serde_json::Value::Null,
            "loss": serde_json::Value::Null,
        }));
    }
    recs.push(serde_json::json!({
        "step": 16,
        "grad_norm": 100.0,
        "grad_norm_clipped": serde_json::Value::Null,
        "loss": serde_json::Value::Null,
    }));
    let f = write_history(&serde_json::Value::Array(recs));

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "--json",
            "grad-norm",
            "--history-file",
            f.path().to_str().unwrap(),
            "--spike-window",
            "16",
            "--spike-multiplier",
            "10.0",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        output.status.success(),
        "apr grad-norm failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let v: serde_json::Value =
        serde_json::from_str(&String::from_utf8_lossy(&output.stdout)).unwrap();
    assert_eq!(
        v["num_spikes"].as_u64(),
        Some(1),
        "expected exactly one spike for 16 flat steps followed by 100.0; got {v}"
    );
}

#[test]
fn falsify_crux_f_09_rejects_clip_cap_violation() {
    // grad_norm_clipped=2.5 with --max-grad-norm=1.0 must be rejected.
    let recs = serde_json::json!([
        { "step": 0, "grad_norm": 5.0, "grad_norm_clipped": 2.5, "loss": 1.0 }
    ]);
    let f = write_history(&recs);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "grad-norm",
            "--history-file",
            f.path().to_str().unwrap(),
            "--max-grad-norm",
            "1.0",
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "grad_norm_clipped=2.5 > max_grad_norm=1.0 must fail"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("cap") || stderr.contains("max-grad-norm"),
        "stderr should explain cap violation; got: {stderr}"
    );
}

#[test]
fn falsify_crux_f_09_rejects_expansive_clipping() {
    // grad_norm_clipped > grad_norm is physically impossible for L2 projection.
    let recs = serde_json::json!([
        { "step": 0, "grad_norm": 1.0, "grad_norm_clipped": 2.0, "loss": 1.0 }
    ]);
    let f = write_history(&recs);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["grad-norm", "--history-file", f.path().to_str().unwrap()])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "grad_norm_clipped > grad_norm (clipping amplifies) must fail"
    );
}

#[test]
fn falsify_crux_f_09_rejects_empty_history() {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(b"[]").unwrap();
    f.flush().unwrap();
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["grad-norm", "--history-file", f.path().to_str().unwrap()])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "empty history file must be rejected (no silent pass)"
    );
}

// ═══ Classifier direct invocation ═══

#[test]
fn falsify_crux_f_09_clip_non_expansive_invariant() {
    use aprender::metrics::grad_norm::{clip_grad_norm, ClipOutcome};
    let mut grads = [3.0_f64, 4.0]; // ||g|| = 5
    match clip_grad_norm(&mut grads, 1.0) {
        ClipOutcome::Ok {
            pre_norm,
            post_norm,
        } => {
            assert!(
                post_norm <= pre_norm + 1e-9,
                "clipping must be non-expansive: pre={pre_norm}, post={post_norm}"
            );
            assert!(
                post_norm <= 1.0 + 1e-6,
                "post-clip norm must respect max_norm cap"
            );
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn falsify_crux_f_09_l2_norm_matches_pytorch_formula() {
    use aprender::metrics::grad_norm::{compute_grad_norm_l2, GradNormOutcome};
    // PyTorch: torch.tensor([3.0, 4.0]).norm().item() == 5.0
    match compute_grad_norm_l2(&[3.0, 4.0]) {
        GradNormOutcome::Ok(v) => assert!((v - 5.0).abs() < 1e-12),
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn falsify_crux_f_09_empty_grads_distinct_outcome() {
    use aprender::metrics::grad_norm::{compute_grad_norm_l2, GradNormOutcome};
    assert!(matches!(
        compute_grad_norm_l2(&[]),
        GradNormOutcome::EmptyGradients
    ));
}
