//! ONT-10 S11 surface gate for the `aprender-verify` binary: `contracts/bin-aprender-verify-ml--aprender-verify-v1.yaml`.
//!
//! Every assertion runs the binary built from this tree (`CARGO_BIN_EXE_aprender-verify`), which
//! cannot fall back to a stale copy on PATH. A test that only asserted success on good
//! input would pass against `fn main() {}`, so each invariant here is one the binary
//! can actually fail: a rejected input must exit non-zero AND must not be a panic.

use std::path::PathBuf;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_aprender-verify");

/// The pinned surface. Adding or removing a subcommand edits this list and the
/// contract's `surface_equals_snapshot` formula together.
const SUBCOMMANDS: &[&str] = &[
    "generate", "stats", "corpus", "depyler", "verify", "train", "evaluate", "distill", "codex",
    "export", "mutate",
];

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("spawn the binary under test")
}

fn advertised() -> Vec<String> {
    let out = run(&["--help"]);
    assert!(out.status.success(), "--help must exit 0");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .skip_while(|l| !l.starts_with("Commands:"))
        .skip(1)
        .take_while(|l| !l.trim().is_empty())
        .filter_map(|l| l.split_whitespace().next().map(str::to_owned))
        .filter(|c| c != "help")
        .collect()
}

/// Three inputs no subcommand can process: a missing path, an empty file, garbage bytes.
fn unusable_inputs() -> [String; 3] {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("aprender-verify-surface-gate");
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let empty = dir.join("empty.bin");
    let garbage = dir.join("garbage.bin");
    std::fs::write(&empty, b"").expect("write empty fixture");
    std::fs::write(&garbage, b"not a model {{{\x00\x01\xff").expect("write garbage fixture");
    [
        dir.join("does-not-exist").display().to_string(),
        empty.display().to_string(),
        garbage.display().to_string(),
    ]
}

fn assert_rejected(args: &[&str]) {
    let out = run(args);
    let code = out.status.code();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        matches!(code, Some(c) if c != 0 && c != 101) && !stderr.contains("panicked"),
        "`aprender-verify {}` must reject its input with a clean non-zero exit, got {code:?}\nstderr: {stderr}",
        args.join(" ")
    );
}

#[test]
fn version_reports_the_crate_version() {
    let out = run(&["--version"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(env!("CARGO_PKG_VERSION")),
        "--version printed {stdout:?}, not {}",
        env!("CARGO_PKG_VERSION")
    );
}

#[test]
fn advertised_subcommands_are_the_pinned_surface() {
    assert_eq!(advertised(), SUBCOMMANDS);
}

#[test]
fn every_advertised_subcommand_is_reachable() {
    let cmds = advertised();
    assert!(
        !cmds.is_empty(),
        "--help advertised no subcommands; the parser is broken"
    );
    for cmd in &cmds {
        let out = run(&[cmd, "--help"]);
        assert!(
            out.status.success(),
            "`aprender-verify {cmd} --help` exited {:?}",
            out.status.code()
        );
    }
}

#[test]
fn unknown_subcommand_is_a_usage_error() {
    assert_eq!(run(&["no-such-subcommand"]).status.code(), Some(2));
}

#[test]
fn unknown_flag_is_a_usage_error() {
    assert_eq!(run(&["--no-such-flag"]).status.code(), Some(2));
}

#[test]
fn unusable_input_is_rejected() {
    let bad = unusable_inputs();
    for b in &bad {
        // A file (not a directory) panicked at read_dir's expect() until ONT-10 S11.
        assert_rejected(&["verify", "-i", b]);
    }
    // A missing input "trained" and wrote a model, exit 0, until ONT-10 S11.
    let out = format!("{}.model", bad[0]);
    assert_rejected(&["train", "-i", &bad[0], "-o", &out]);
}
