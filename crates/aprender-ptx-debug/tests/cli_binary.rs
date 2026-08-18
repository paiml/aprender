//! End-to-end checks that the parsed command actually reaches its handler.
//!
//! `tests/cli_args.rs` proves the argument grammar; these tests prove the
//! dispatch behind it, so a subcommand cannot be parsed correctly and then
//! wired to nothing.

use std::process::Command;

/// Path to the freshly built binary, supplied by cargo. Never resolve a binary
/// through `$PATH` or a hardcoded path.
const BIN: &str = env!("CARGO_BIN_EXE_aprender-ptx-debug");

/// A path that cannot exist, used to reach a handler without a PTX fixture:
/// only the handler itself can produce the "Failed to read" diagnostic.
const MISSING_PTX: &str = "/nonexistent/aprender-ptx-debug/fixture.ptx";

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

fn run(args: &[&str]) -> Run {
    let out = Command::new(BIN)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn {BIN}: {e}"));
    Run {
        code: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

#[test]
fn unknown_flag_fails_the_process() {
    let r = run(&["analyze", "kernel.ptx", "--seed", "42"]);
    assert_eq!(r.code, Some(1), "stderr: {}", r.stderr);
    assert!(
        r.stderr.contains("--seed"),
        "the rejected flag should be named; stderr: {}",
        r.stderr
    );
    assert!(
        !r.stdout.contains("PTX Analysis Report"),
        "analysis must not run when parsing failed; stdout: {}",
        r.stdout
    );
}

#[test]
fn no_arguments_exits_one() {
    let r = run(&[]);
    assert_eq!(r.code, Some(1));
}

#[test]
fn analyze_subcommand_reaches_its_handler() {
    let r = run(&["analyze", MISSING_PTX]);
    assert_eq!(r.code, Some(1), "stderr: {}", r.stderr);
    assert!(
        r.stderr.contains("Failed to read"),
        "analyze should have reached the file read; stderr: {}",
        r.stderr
    );
}

#[test]
fn gen_fkr_subcommand_reaches_its_handler() {
    let r = run(&["gen-fkr", MISSING_PTX]);
    assert_eq!(r.code, Some(1), "stderr: {}", r.stderr);
    assert!(
        r.stderr.contains("Failed to read"),
        "gen-fkr should have reached the file read; stderr: {}",
        r.stderr
    );
}

#[test]
fn version_subcommand_and_version_flag_do_not_drift() {
    let sub = run(&["version"]);
    let flag = run(&["--version"]);
    assert_eq!(sub.code, Some(0), "stderr: {}", sub.stderr);
    assert_eq!(flag.code, Some(0), "stderr: {}", flag.stderr);
    assert!(
        sub.stdout.contains(env!("CARGO_PKG_VERSION")),
        "stdout: {}",
        sub.stdout
    );
    assert_eq!(
        sub.stdout, flag.stdout,
        "`version` and `--version` must print the same string"
    );
}

#[test]
fn help_lists_every_subcommand() {
    let r = run(&["help"]);
    assert_eq!(r.code, Some(0), "stderr: {}", r.stderr);
    for expected in ["analyze", "gen-fkr", "version", "EXIT CODES", "EXAMPLES"] {
        assert!(
            r.stdout.contains(expected),
            "help should mention `{expected}`; stdout: {}",
            r.stdout
        );
    }
}
