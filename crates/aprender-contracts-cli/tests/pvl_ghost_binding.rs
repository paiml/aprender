//! PVL-001 EV-2 (PVL-2, PMAT-4080) — `pv proof-status --binding` RESOLVES bindings,
//! and a ghost binding is a reject.
//!
//! GROUND TRUTH (PVL-001 EV-2 facts; scripts/dogfood.sh records the same): given a
//! binding that claims `implemented` for a function that exists nowhere,
//! `pv proof-status --binding` printed its report and exited 0. It counted binding
//! ENTRIES and never resolved them, while `pv verify-bindings` on the same file
//! exited 1. And a MISSING binding file exits 1 on "Failed to read", which is why
//! the v2 `done_when` "passed" with nothing implemented: this file asserts the
//! promised line and the symbol, never a bare exit code.
//!
//! DISCRIMINATION: the control binding names a function that DOES exist, so "every
//! binding is a ghost" fails it; the missing-file case proves a read error is not
//! mistaken for the ghost verdict.

use std::path::PathBuf;
use std::process::Command;

/// The `pv` built from THIS tree (never a stale one off `$PATH`, #2552).
fn pv_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_pv"))
}

/// The workspace root. The binding resolver is CWD-sensitive by design (it scans
/// the local `src/` and the binding's derived source root), and PVL-001 EV-2's probe
/// runs from the worktree root, so these runs do too.
fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

fn pv(args: &[&str]) -> Run {
    let out = Command::new(pv_bin())
        .current_dir(root())
        .args(args)
        .output()
        .expect("failed to spawn pv");
    Run {
        code: out.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

const GHOST: &str = "tests/fixtures/pvl/ghost-binding.yaml";
const RESOLVED: &str = "tests/fixtures/pvl/resolved-binding.yaml";
const ONE_CONTRACT: &str = "contracts/softmax-kernel-v1.yaml";

/// PVL-001 EV-2's probe, verbatim in effect: default contract path, the ghost fixture.
#[test]
fn a_ghost_binding_is_a_reject() {
    let r = pv(&["proof-status", "--binding", GHOST]);
    assert_eq!(
        r.code, 1,
        "a ghost binding must be rejected (exit 1)\n{}\n{}",
        r.stdout, r.stderr
    );
    assert!(
        r.stdout.lines().any(|l| l == "GHOST BINDINGS (1)"),
        "stdout must carry the line `GHOST BINDINGS (1)`:\n{}",
        r.stdout
    );
    assert!(
        r.stdout.contains("function_that_does_not_exist_xyz"),
        "the ghost symbol must be named:\n{}",
        r.stdout
    );
}

/// `--verify-bindings` is a no-op alias: resolution happens with or without it.
#[test]
fn verify_bindings_is_a_noop_alias() {
    for extra in [
        &[][..],
        &["--verify-bindings"][..],
        &["--verify-bindings", "."][..],
    ] {
        let mut args = vec!["proof-status", ONE_CONTRACT, "--binding", GHOST];
        args.extend_from_slice(extra);
        let r = pv(&args);
        assert_eq!(r.code, 1, "{args:?}\n{}\n{}", r.stdout, r.stderr);
        assert!(
            r.stdout.lines().any(|l| l == "GHOST BINDINGS (1)"),
            "{args:?}\n{}",
            r.stdout
        );
    }
}

/// Control: a binding whose function exists resolves, is accepted, and prints no
/// ghost block, so the reject above cannot be "every binding is a ghost".
#[test]
fn control_a_resolved_binding_is_accepted() {
    let r = pv(&["proof-status", ONE_CONTRACT, "--binding", RESOLVED]);
    assert_eq!(
        r.code, 0,
        "a resolved binding must be accepted\n{}\n{}",
        r.stdout, r.stderr
    );
    assert!(!r.stdout.contains("GHOST BINDINGS"), "{}", r.stdout);
}

/// A binding file that does not exist is a read error, NOT the ghost verdict: it
/// must not print `GHOST BINDINGS`. (This is the v2 hole: rc 1 alone proved nothing.)
#[test]
fn a_missing_binding_file_is_not_the_ghost_verdict() {
    let r = pv(&[
        "proof-status",
        ONE_CONTRACT,
        "--binding",
        "tests/fixtures/pvl/does-not-exist.yaml",
    ]);
    assert_ne!(r.code, 0, "{}\n{}", r.stdout, r.stderr);
    assert!(!r.stdout.contains("GHOST BINDINGS"), "{}", r.stdout);
}
