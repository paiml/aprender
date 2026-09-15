//! e2e: the `cli` transport, exercised against the SHIPPED BINARY.
//!
//! Declared in the root `Cargo.toml` as
//! `[package.metadata.transports] cli = { e2e = "e2e_cli_t", features = ["cli"] }`
//! and run by `scripts/dogfood.sh`'s `interface-parity` gate.
//!
//! Every test here spawns the built `apr` artifact — the binary a user installs
//! — rather than calling the library. That is the whole point of
//! the gate: a library-level suite is structurally blind to a subcommand that
//! is unreachable through the binary (a missing feature gate, an unregistered
//! `clap` subcommand, a `main` that never dispatches). Only spawning the
//! artifact can see it.
//!
//! Hermetic: no network, no model outside the repo, no writes outside the
//! process's own pipes.

use std::path::PathBuf;
use std::process::{Command, Output};

/// The shipped binary, located through the bin-exe environment variable cargo
/// sets for integration tests of the package that declares the `apr` bin
/// target. It points at the artifact cargo just built — never at whatever `apr`
/// happens to be on `$PATH` (four of them once coexisted on the dev box; a bare
/// `apr` resolved to a 26-day-old copy).
///
/// The literal appears exactly once in this file, at the call below: the
/// dogfood `interface-parity` gate proves reachability by grepping the target's
/// source for it, and a mention in prose would satisfy that grep while the test
/// spawned something else entirely.
fn apr() -> Command {
    Command::new(env!("CARGO_BIN_EXE_apr"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Render a failed invocation with both streams, so a red run names the cause
/// instead of only the exit code.
fn describe(what: &str, out: &Output) -> String {
    format!(
        "{what}: exit={:?}\n--- stdout ---\n{}\n--- stderr ---\n{}",
        out.status.code(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    )
}

#[test]
fn version_exits_zero_and_prints_the_crate_version() {
    let out = apr()
        .arg("--version")
        .output()
        .expect("spawn the shipped apr binary with --version");

    assert!(out.status.success(), "{}", describe("apr --version", &out));

    let stdout = String::from_utf8_lossy(&out.stdout);
    let want = env!("CARGO_PKG_VERSION");
    assert!(
        stdout.contains(want),
        "apr --version printed {stdout:?}, which does not contain the crate \
         version {want:?} — the shipped artifact disagrees with the manifest \
         it was built from"
    );
}

#[test]
fn inspect_reads_a_tracked_apr_v2_fixture() {
    // `golden_v2.apr` (1.07 KiB, tracked) is the smallest fixture in the tree
    // the current binary accepts. The v1 goldens and the legacy `APRN/APR1/APR2`
    // models are rejected by design ("Only APR v2 (APR\0) is supported"), so
    // they would test the error path, not the transport.
    let fixture = repo_root().join("crates/apr-format/tests/fixtures/golden_v2.apr");
    assert!(
        fixture.is_file(),
        "fixture missing: {} — this test is hermetic and must not download one",
        fixture.display()
    );

    let out = apr()
        .arg("inspect")
        .arg(&fixture)
        .output()
        .expect("spawn the shipped apr binary with inspect");

    assert!(out.status.success(), "{}", describe("apr inspect", &out));

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("APR v2"),
        "apr inspect exited 0 but never reported the format; stdout was:\n{stdout}"
    );
}
