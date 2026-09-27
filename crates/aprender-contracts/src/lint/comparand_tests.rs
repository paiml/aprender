use super::*;

use std::path::Path;

use crate::lint::sigma_gate::{run_sigma_gate, SigmaOutcome};

fn fixture(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/ont")
        .join(name)
}

#[test]
fn no_comparand_measures_nothing() {
    let mut called = false;
    let got = baseline_at(None, |_| {
        called = true;
        Some(7)
    });
    assert_eq!(got, None, "no comparand is not a baseline of any value");
    assert!(!called, "nothing is measured when no comparand is named");
}

#[test]
fn the_measure_runs_over_the_named_dir() {
    let dir = Path::new("/nonexistent/comparand/contracts");
    let got = baseline_at(Some(dir), |d| {
        assert_eq!(d, dir);
        Some(3)
    });
    assert_eq!(got, Some(3));
}

#[test]
fn the_comparand_run_sees_no_comparand() {
    // The comparand's own gate run must not recurse into a comparand of the comparand.
    let dir = Path::new("/nonexistent/base");
    let got = baseline_at(Some(dir), |_| {
        assert!(comparand_dir().is_none(), "a nested run sees no comparand");
        baseline_at(Some(dir), |_| Some(99))
    });
    assert_eq!(got, None, "the nested measure answers None, never 99");
}

#[test]
fn the_guard_is_cleared_after_the_call_and_after_a_panic() {
    let dir = Path::new("/nonexistent/base");
    assert_eq!(baseline_at(Some(dir), |_| Some(1)), Some(1));
    assert_eq!(
        baseline_at(Some(dir), |_| Some(2)),
        Some(2),
        "a second call still measures"
    );
    let panicked =
        std::panic::catch_unwind(|| baseline_at(Some(dir), |_| panic!("measure failed")));
    assert!(panicked.is_err());
    assert_eq!(
        baseline_at(Some(dir), |_| Some(4)),
        Some(4),
        "a panicking measure must not leave the thread believing it is still measuring"
    );
}

#[test]
fn count_in_reads_the_gate_payload() {
    let SigmaOutcome::Ran { result, .. } = run_sigma_gate(&fixture("sigma-ok")) else {
        panic!("sigma-ok runs");
    };
    assert_eq!(count_in(&result, "contracts_checked"), Some(1));
    assert_eq!(count_in(&result, "formal_prose"), Some(0));
    assert_eq!(count_in(&result, "no_such_key"), None);
}
