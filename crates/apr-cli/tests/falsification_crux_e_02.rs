//! End-to-end falsification tests for CRUX-E-02 — Perplexity on held-out corpus.
//!
//! Contract: `contracts/crux-E-02-v1.yaml` (v1.1.0).
//!
//! CRUX-SHIP-001 compliance:
//! - g1_classifier_green: `aprender::metrics::perplexity` unit tests in-crate.
//! - g2_cli_reachable: `apr ppl --help` surfaces `--log-probs-file`.
//! - g3_e2e_runs: subprocess invocation of the real binary exercises
//!   classifier-via-CLI end-to-end. Live-model PPL (real GGUF/APR over a
//!   corpus) remains PARTIAL_ALGORITHM_LEVEL under BLOCKER-UPSTREAM-MISSING.

#![allow(clippy::unwrap_used)]

use std::io::Write;

use assert_cmd::Command;
use predicates::prelude::*;

fn write_log_probs(log_probs: &[f64]) -> tempfile::NamedTempFile {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    let body = serde_json::to_vec(log_probs).unwrap();
    f.write_all(&body).unwrap();
    f.flush().unwrap();
    f
}

// ═══ g2_cli_reachable ═══

#[test]
fn falsify_crux_e_02_help_advertises_log_probs_file_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["ppl", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--log-probs-file"));
}

#[test]
fn falsify_crux_e_02_rejects_bare_ppl_without_file() {
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["ppl"])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "bare `apr ppl` (no --log-probs-file) must fail"
    );
}

// ═══ g3_e2e_runs: classifier-via-CLI ═══

#[test]
fn falsify_crux_e_02_json_emits_ppl_key() {
    // ln(2) ≈ 0.693; mean NLL = ln(2); PPL = 2.
    let ln2 = std::f64::consts::LN_2;
    let log_probs: Vec<f64> = vec![-ln2; 8];
    let f = write_log_probs(&log_probs);

    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "--json",
            "ppl",
            "--log-probs-file",
            f.path().to_str().unwrap(),
        ])
        .output()
        .expect("apr binary runs");
    assert!(
        output.status.success(),
        "apr --json ppl failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let v: serde_json::Value = serde_json::from_str(&stdout).expect("valid JSON");
    let ppl = v["ppl"].as_f64().expect("ppl key present and numeric");
    assert!(
        (ppl - 2.0).abs() < 1e-9,
        "FALSIFY-CRUX-E-02-001: expected ppl ≈ 2.0 for log p = -ln(2), got {ppl}"
    );
    assert_eq!(v["num_tokens"].as_u64(), Some(8));
}

#[test]
fn falsify_crux_e_02_004_ppl_at_least_one_and_finite() {
    use aprender::metrics::perplexity::{compute_perplexity, PerplexityOutcome};
    let samples = [-1.0_f64, -0.5, -2.3, -0.01];
    match compute_perplexity(&samples) {
        PerplexityOutcome::Ok { ppl, .. } => {
            assert!(ppl >= 1.0 && ppl.is_finite(), "ppl = {ppl}");
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn falsify_crux_e_02_perfect_prediction_ppl_is_one() {
    use aprender::metrics::perplexity::{compute_perplexity, PerplexityOutcome};
    match compute_perplexity(&[0.0, 0.0, 0.0]) {
        PerplexityOutcome::Ok { ppl, .. } => {
            assert!(
                (ppl - 1.0).abs() < 1e-12,
                "perfect prediction must give PPL=1.0, got {ppl}"
            );
        }
        other => panic!("expected Ok, got {other:?}"),
    }
}

#[test]
fn falsify_crux_e_02_empty_distinct_outcome() {
    use aprender::metrics::perplexity::{compute_perplexity, PerplexityOutcome};
    assert!(matches!(
        compute_perplexity(&[]),
        PerplexityOutcome::EmptyLogProbs
    ));
}

#[test]
fn falsify_crux_e_02_nan_log_prob_rejected_via_cli() {
    let f = {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        // Write raw JSON with a NaN-alike: serde_json can't write NaN, so
        // emit a positive log-prob which also trips no-silent-pass.
        file.write_all(br"[-1.0, 0.5, -2.0]").unwrap();
        file.flush().unwrap();
        file
    };
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["ppl", "--log-probs-file", f.path().to_str().unwrap()])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "positive log-prob must be rejected (no silent pass)"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("strictly positive") || stderr.contains("log-prob"),
        "stderr should explain the rejection; got: {stderr}"
    );
}

#[test]
fn falsify_crux_e_02_empty_file_rejected_via_cli() {
    let mut f = tempfile::NamedTempFile::new().unwrap();
    f.write_all(b"[]").unwrap();
    f.flush().unwrap();
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["ppl", "--log-probs-file", f.path().to_str().unwrap()])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "empty log-probs file must be rejected"
    );
}

#[test]
fn falsify_crux_e_02_ppl_monotone_in_nll() {
    use aprender::metrics::perplexity::{compute_perplexity, PerplexityOutcome};
    let a = [-0.5_f64, -0.5, -0.5];
    let b = [-2.0_f64, -2.0, -2.0];
    let pa = match compute_perplexity(&a) {
        PerplexityOutcome::Ok { ppl, .. } => ppl,
        _ => panic!(),
    };
    let pb = match compute_perplexity(&b) {
        PerplexityOutcome::Ok { ppl, .. } => ppl,
        _ => panic!(),
    };
    assert!(pa < pb, "ppl({pa}) < ppl({pb}) must hold");
}
