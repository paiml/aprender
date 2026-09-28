// Integration tests: unwrap()/panic!() are idiomatic; strict workspace lints relaxed here.
#![allow(clippy::disallowed_methods)]
//! M001-M010 (the subset that runs the binary): Measurement Tools (cbtop) Falsification Tests
//!
//! Per spec: docs/specifications/archive/qwen2.5-coder-showcase-demo.md S9.4
//!
//! These lived in `aprender-core/tests/falsification_measurement_tests.rs` and each
//! one ran `cargo run -p apr-cli --bin apr` from inside the test. Under nextest that is
//! a SECOND cargo build per test, with apr-cli's own feature set rather than the
//! workspace-unified one the test binaries were built with, and six of them queued on
//! the same build-directory lock: x86-main reported all six SLOW, five of them past
//! 120 s. Here they run the `apr` that cargo already built for this target
//! (`CARGO_BIN_EXE_apr`), so no test compiles anything. The arithmetic-only M-tests
//! (M004-M006, M009, M011+) stay in aprender-core.
//!
//! The move also removes the "apr binary not found" SKIP arms and the `if success`
//! guards: the binary is a build input of this target and cannot be absent, so a
//! failing run is a failure, not a skip.

use std::process::{Command, Output};

fn cbtop(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_apr"))
        .args(["cbtop", "--headless", "--simulated"])
        .args(args)
        .output()
        .expect("spawn the apr this target was built with")
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// M001: cbtop --headless --simulated exits cleanly with code 0
#[test]
fn m001_headless_exits_cleanly() {
    let out = cbtop(&["--iterations", "10"]);
    assert!(
        out.status.success(),
        "M001 FALSIFIED: cbtop --headless exited with error: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// M002: JSON output is valid JSON
#[test]
fn m002_json_output_valid() {
    let out = cbtop(&["--json", "--iterations", "10"]);
    assert!(
        out.status.success(),
        "M002: cbtop --json failed:\n{}",
        text(&out)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let trimmed = stdout.trim();
    assert!(
        trimmed.starts_with('{') && trimmed.ends_with('}'),
        "M002 FALSIFIED: Output is not valid JSON object"
    );
    assert!(
        trimmed.contains("\"model\""),
        "M002 FALSIFIED: JSON missing 'model' field"
    );
    assert!(
        trimmed.contains("\"throughput\""),
        "M002 FALSIFIED: JSON missing 'throughput' field"
    );
    assert!(
        trimmed.contains("\"brick_scores\""),
        "M002 FALSIFIED: JSON missing 'brick_scores' field"
    );
}

/// M003: Brick scores present in JSON output
#[test]
fn m003_brick_scores_present() {
    let out = cbtop(&["--json", "--iterations", "10"]);
    assert!(
        out.status.success(),
        "M003: cbtop --json failed:\n{}",
        text(&out)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let brick_count = stdout.matches("\"name\":").count();
    assert!(
        brick_count >= 7,
        "M003 FALSIFIED: Expected 7 brick scores, found {}",
        brick_count
    );
}

/// M007: CI mode returns exit code 1 on threshold failure
#[test]
fn m007_ci_exit_code_on_failure() {
    let out = cbtop(&["--ci", "--throughput", "999999", "--iterations", "10"]);
    let t = text(&out);
    assert!(
        !out.status.success(),
        "M007 FALSIFIED: CI mode should return non-zero on threshold failure"
    );
    // A non-zero exit alone is also what a crash looks like; the verdict must be the reason.
    assert!(
        t.contains("Status: FAIL"),
        "M007 FALSIFIED: non-zero exit without a `Status: FAIL` verdict:\n{t}"
    );
}

/// M008: CI mode returns exit code 0 on threshold pass
#[test]
fn m008_ci_exit_code_on_pass() {
    let out = cbtop(&["--ci", "--throughput", "100", "--iterations", "10"]);
    // `--simulated` draws jittered brick timings, so whether the thresholds are met is a
    // coin flip per run (measured 2026-09-11: "Falsification: 3/7 passed", CV 88 %). The
    // contract M008 names is that the EXIT CODE follows the verdict: 0 iff the run prints
    // `Status: PASS`. Assert that equivalence, which is deterministic, instead of assuming
    // the simulated run passes.
    let t = text(&out);
    let verdict_pass = t.contains("Status: PASS");
    let verdict_fail = t.contains("Status: FAIL");
    assert!(
        verdict_pass || verdict_fail,
        "M008 FALSIFIED: CI mode printed no `Status: PASS|FAIL` verdict:\n{t}"
    );
    assert_eq!(
        out.status.success(),
        verdict_pass,
        "M008 FALSIFIED: CI exit code must be 0 exactly when the verdict is PASS (success={}, verdict_pass={})",
        out.status.success(),
        verdict_pass
    );
}

/// M010: Output file written when --output specified
#[test]
fn m010_output_file_created() {
    // Per-target scratch dir, not a fixed /tmp path two concurrent runs would share.
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("m010_test_output.json");
    let _ = std::fs::remove_file(&path);
    let p = path.to_str().expect("utf-8 path");
    let out = cbtop(&["--json", "--output", p, "--iterations", "10"]);
    assert!(
        out.status.success(),
        "M010: cbtop --output failed:\n{}",
        text(&out)
    );
    assert!(
        path.exists(),
        "M010 FALSIFIED: Output file not created at {}",
        path.display()
    );
    let _ = std::fs::remove_file(&path);
}
