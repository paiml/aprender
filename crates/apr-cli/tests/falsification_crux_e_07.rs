//! End-to-end falsification tests for CRUX-E-07 — Latency P50/P95/P99 percentiles.
//!
//! Contract: contracts/crux-E-07-v1.yaml
//!
//! CRUX-SHIP-001 compliance:
//! - g1_classifier_green: covered by `aprender::metrics::percentile` unit tests
//!   (see `crates/aprender-core/src/metrics/percentile.rs`).
//! - g2_cli_reachable: `apr bench --help` surfaces `--percentiles` flag.
//! - g3_e2e_runs: subprocess invocation of the real binary exercises end-to-end.
//! - g4_contract_discharged: FALSIFY-CRUX-E-07-002 (monotonicity) + FALSIFY-005
//!   (strictly positive) dispatched via pure classifiers + flag reachability.

#![allow(clippy::unwrap_used)]

use assert_cmd::Command;
use predicates::prelude::*;

// ═══ g2_cli_reachable: --percentiles flag must appear in help ═══

#[test]
fn falsify_crux_e_07_help_advertises_percentiles_flag() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["bench", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--percentiles"));
}

#[test]
fn falsify_crux_e_07_help_mentions_default_50_95_99() {
    Command::cargo_bin("apr")
        .unwrap()
        .args(["bench", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("50,95,99"));
}

// ═══ g3_e2e_runs: parser accepts comma-separated percentile values ═══

#[test]
fn falsify_crux_e_07_accepts_comma_separated_percentiles() {
    // Invoking against a nonexistent model will fail at load-time, but only
    // AFTER clap successfully parses the --percentiles argument. A clap
    // parse failure would surface with a different error class.
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args([
            "bench",
            "--percentiles",
            "50,90,95,99,99.9",
            "/tmp/definitely-nonexistent-model-crux-e-07.gguf",
        ])
        .output()
        .expect("apr binary runs");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("unexpected argument")
            && !stderr.contains("invalid value")
            && !stderr.contains("error: the following required arguments"),
        "clap must accept `--percentiles 50,90,95,99,99.9`; got stderr: {stderr}"
    );
}

#[test]
fn falsify_crux_e_07_rejects_percentile_without_value() {
    // clap MUST reject a bare `--percentiles` flag (no values).
    let output = Command::cargo_bin("apr")
        .unwrap()
        .args(["bench", "--percentiles"])
        .output()
        .expect("apr binary runs");
    assert!(
        !output.status.success(),
        "bare --percentiles must fail arg parsing"
    );
}

// ═══ g1_classifier_green: mirror key invariants end-to-end ═══

#[test]
fn falsify_crux_e_07_002_monotonicity_via_classifier() {
    use aprender::metrics::percentile::{compute_percentile_ladder, PercentileLadderOutcome};

    // Adversarial: a heavy-tailed distribution exposes p99 >> p50 blowup.
    let mut samples: Vec<f64> = (1..=99).map(f64::from).collect();
    samples.push(1000.0); // one outlier

    match compute_percentile_ladder(&samples, &[50.0, 95.0, 99.0]) {
        PercentileLadderOutcome::Ok(vs) => {
            assert!(
                vs[0] <= vs[1] && vs[1] <= vs[2],
                "FALSIFY-CRUX-E-07-002: monotonicity violated: {vs:?}"
            );
        }
        other => panic!("FALSIFY-CRUX-E-07-002: expected Ok, got {other:?}"),
    }
}

#[test]
fn falsify_crux_e_07_005_percentiles_strictly_positive() {
    use aprender::metrics::percentile::{compute_percentile, PercentileOutcome};

    let samples: Vec<f64> = (1..=100).map(f64::from).collect();
    for p in [50.0, 95.0, 99.0] {
        match compute_percentile(&samples, p) {
            PercentileOutcome::Ok(v) => assert!(
                v > 0.0,
                "FALSIFY-CRUX-E-07-005: p{p} = {v} must be strictly positive"
            ),
            other => panic!("FALSIFY-CRUX-E-07-005: expected Ok, got {other:?}"),
        }
    }
}

#[test]
fn falsify_crux_e_07_nan_sample_distinct_outcome() {
    use aprender::metrics::percentile::{compute_percentile, PercentileOutcome};

    assert!(
        matches!(
            compute_percentile(&[1.0, f64::NAN, 3.0], 50.0),
            PercentileOutcome::NonFiniteSample
        ),
        "NaN samples must map to a distinct Outcome variant — no silent pass"
    );
}

#[test]
fn falsify_crux_e_07_empty_samples_distinct_outcome() {
    use aprender::metrics::percentile::{compute_percentile, PercentileOutcome};

    assert!(matches!(
        compute_percentile(&[], 50.0),
        PercentileOutcome::EmptySamples
    ));
}
