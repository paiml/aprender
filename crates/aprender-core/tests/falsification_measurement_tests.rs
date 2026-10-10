#![allow(clippy::disallowed_methods)]
//! M001-M010: Measurement Tools (cbtop) Falsification Tests
//!
//! Per spec: docs/specifications/archive/qwen2.5-coder-showcase-demo.md S9.4
//!
//! These tests verify the cbtop measurement tool works correctly.
//! Measurement tools must be accurate and reliable for optimization.
//!
//! FALSIFICATION: If measurement is unreliable, optimization is impossible.

use std::process::Command;

/// `cargo` for the nested `cargo run -p apr-cli` builds below, minus this test
/// target's `CONTRACT_*` vars. cargo hands a package's build-script `rustc-env`
/// vars to that package's test processes, so the nested build inherited
/// aprender-core's `CONTRACT_BINDING_SOURCE=binding.yaml`. apr-cli's build emits
/// no binding registry, so every `#[contract]` site in it then failed the #4368
/// registry check and apr-cli did not compile (#5001).
fn nested_cargo() -> Command {
    let mut cmd = Command::new("cargo");
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("CONTRACT_") {
            cmd.env_remove(key);
        }
    }
    cmd
}

/// Run `apr cbtop <args>` through the nested `cargo run -p apr-cli` build.
///
/// Panics NOT MEASURED when cargo cannot be spawned or when the nested build
/// fails before `apr` starts. cargo prints its `Running` line only once the
/// binary is built. Quiet and verbose are pinned off, because verbose mode also
/// prints a `Running` line per rustc call, and the line must name `cbtop`.
/// Colour is pinned off, because `CARGO_TERM_COLOR=always` puts ANSI codes in
/// front of `Running` and the check would then miss a binary that did run.
///
/// The tests used to read a failed build as a result: M002/M003/M010 skipped
/// their asserts unless the exit status was success, and M007 asserted only
/// "non-zero", which a compile error (101) satisfies. All four passed while
/// apr-cli did not compile (#5001, #5003).
fn run_cbtop(id: &str, args: &[&str]) -> std::process::Output {
    let output = nested_cargo()
        .args(["run", "-p", "apr-cli", "--bin", "apr", "--", "cbtop"])
        .args(args)
        .env("CARGO_TERM_QUIET", "false")
        .env("CARGO_TERM_VERBOSE", "false")
        .env("CARGO_TERM_COLOR", "never")
        .output()
        .unwrap_or_else(|e| panic!("{id} NOT MEASURED: could not spawn cargo: {e}"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr
            .lines()
            .any(|l| l.trim_start().starts_with("Running `") && l.contains(" cbtop")),
        "{id} NOT MEASURED: the nested apr-cli build failed before apr ran (exit {:?}):\n{stderr}",
        output.status.code()
    );
    output
}

include!("includes/falsification_measurement.rs");
include!("includes/falsification_measurement_scoring.rs");
