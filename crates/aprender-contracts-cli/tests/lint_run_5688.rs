//! #5688: behaviour of `commands::lint::run`, driven through the `pv` binary.
//!
//! Every case runs in a scratch directory so `.pv/` caches never land in the repo.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pv_in(dir: &Path, args: &[&str]) -> (i32, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_pv"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("failed to spawn pv");
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    (out.status.code().unwrap_or(-1), text)
}

/// A scratch dir holding `c/relu-kernel-v1.yaml` and an empty `empty/`.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("pv-lint-run-5688-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("c")).expect("mkdir c");
    std::fs::create_dir_all(dir.join("empty")).expect("mkdir empty");
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/relu-kernel-v1.yaml");
    std::fs::copy(fixture, dir.join("c/relu-kernel-v1.yaml")).expect("copy fixture");
    dir
}

#[test]
fn missing_corpus_is_declined_with_exit_2() {
    let d = scratch("missing");
    let (rc, out) = pv_in(&d, &["lint", "no-such-dir"]);
    assert_eq!(rc, 2, "{out}");
    assert!(out.contains("0 contracts under no-such-dir"), "{out}");
}

#[test]
fn empty_corpus_is_declined_with_exit_2() {
    let d = scratch("empty");
    let (rc, out) = pv_in(&d, &["lint", "empty"]);
    assert_eq!(rc, 2, "{out}");
    assert!(out.contains("0 contracts under empty"), "{out}");
    assert!(!out.contains("Result: PASS"), "{out}");
}

#[test]
fn show_trend_without_history_prints_hint_and_succeeds() {
    let d = scratch("trend");
    let (rc, out) = pv_in(&d, &["lint", "c", "--show-trend"]);
    assert_eq!(rc, 0, "{out}");
    assert!(out.contains("No trend data"), "{out}");
    assert!(
        !out.contains("Result:"),
        "show-trend must not run the lint: {out}"
    );
}

#[test]
fn coverage_below_minimum_is_refused_naming_both_figures() {
    let d = scratch("cov");
    let (rc, out) = pv_in(
        &d,
        &[
            "lint",
            "c",
            "--no-cache",
            "--coverage",
            "--min-coverage",
            "101",
        ],
    );
    assert_eq!(rc, 1, "{out}");
    assert!(
        out.contains("Contract Coverage: 1/1 at Standard+ (100.0%)"),
        "{out}"
    );
    assert!(
        out.contains("contract coverage 100.0% is below minimum 101.0%"),
        "{out}"
    );
}

#[test]
fn coverage_at_or_above_minimum_adds_no_coverage_error() {
    let d = scratch("covok");
    let (_, out) = pv_in(
        &d,
        &[
            "lint",
            "c",
            "--no-cache",
            "--coverage",
            "--min-coverage",
            "50",
        ],
    );
    assert!(out.contains("Contract Coverage: 1/1"), "{out}");
    assert!(!out.contains("is below minimum"), "{out}");
}

#[test]
fn single_gate_runs_only_that_gate() {
    let d = scratch("gate1");
    let (rc, out) = pv_in(&d, &["lint", "c", "--no-cache", "--gate", "validate"]);
    assert_eq!(rc, 1, "{out}");
    assert!(out.contains("PROVABILITY-001"), "{out}");
    assert!(out.contains("0/1 armed gates passed"), "{out}");
}

#[test]
fn unknown_gate_in_a_list_is_refused() {
    let d = scratch("gate2");
    let (rc, out) = pv_in(
        &d,
        &[
            "lint",
            "c",
            "--no-cache",
            "--gate",
            "validate",
            "--gate",
            "nosuch",
        ],
    );
    assert_eq!(rc, 1, "{out}");
    assert!(
        out.contains("--gate nosuch: not a gate this build runs alone"),
        "{out}"
    );
}

#[test]
fn single_file_with_strict_test_binding_is_refused_before_any_lint() {
    let d = scratch("strict");
    let (rc, out) = pv_in(
        &d,
        &["lint", "c/relu-kernel-v1.yaml", "--strict-test-binding"],
    );
    assert_eq!(rc, 1, "{out}");
    assert!(
        out.contains("cannot run over a single contract file"),
        "{out}"
    );
    assert!(!out.contains("Result:"), "{out}");
}
