//! ONT-10 surface gate for the `simular` binary: `contracts/bin-aprender-simulate--simular-v1.yaml` (aprender#4079).
//!
//! Every assertion runs the binary built from this tree (`CARGO_BIN_EXE_simular`), which
//! cannot fall back to a stale copy on PATH. A test that only asserted success on good
//! input would pass against `fn main() {}`, so each invariant here is one the binary
//! can actually fail: a rejected input must exit non-zero AND must not be a panic.

use std::path::PathBuf;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_simular");

/// An argument list the parser must refuse as an unknown subcommand or extra positional.
const UNKNOWN_SUB: &[&str] = &["no-such-subcommand"];

/// The pinned surface. Adding or removing a subcommand edits this list and the
/// contract's FALSIFY-BIN-*-004 prediction together.
const SUBCOMMANDS: &[&str] = &[
    "run",
    "render",
    "validate",
    "verify",
    "emc-check",
    "emc-validate",
    "list-emc",
    "version",
];

fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("spawn the binary under test")
}

fn help_section(header: &str) -> Vec<String> {
    // `-h` is clap's one-line-per-entry form; `--help` may be the multi-line long form.
    let out = run(&["-h"]);
    assert!(out.status.success(), "-h must exit 0");
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .skip_while(|l| !l.starts_with(header))
        .skip(1)
        .take_while(|l| !l.is_empty() && !l.ends_with(':'))
        .map(str::to_owned)
        .collect()
}

fn advertised() -> Vec<String> {
    help_section("Commands:")
        .iter()
        .filter_map(|l| l.split_whitespace().next().map(str::to_owned))
        .filter(|c| c != "help")
        .collect()
}

/// Three inputs no command can process: a missing path, an empty file, garbage bytes.
fn unusable_inputs() -> Vec<String> {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("simular-surface-gate");
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let empty = dir.join("empty.bin");
    let garbage = dir.join("garbage.bin");
    std::fs::write(&empty, b"").expect("write empty fixture");
    std::fs::write(&garbage, b"not a model {{{\x00\x01\xff").expect("write garbage fixture");
    let extra = dir.join("schemaless.yaml");
    std::fs::write(&extra, b"foo: 1\n").expect("write schemaless.yaml fixture");
    vec![
        dir.join("does-not-exist").display().to_string(),
        empty.display().to_string(),
        garbage.display().to_string(),
        extra.display().to_string(),
    ]
}

fn assert_rejected(args: &[&str]) {
    let out = run(args);
    let code = out.status.code();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        matches!(code, Some(c) if c != 0 && c != 101) && !stderr.contains("panicked"),
        "`simular {}` must reject its input with a clean non-zero exit, got {code:?}\nstderr: {stderr}",
        args.join(" ")
    );
}

fn assert_usage_error(args: &[&str]) {
    let out = run(args);
    assert_eq!(
        out.status.code(),
        Some(2),
        "`simular {}` must be a usage error (exit 2)\nstderr: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn version_reports_the_crate_version() {
    let out = run(&["--version"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let want = format!("simular {}", env!("CARGO_PKG_VERSION"));
    assert!(
        stdout.starts_with(&want),
        "--version printed {stdout:?}, not {want:?}"
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
            "`simular {cmd} --help` exited {:?}",
            out.status.code()
        );
    }
}

#[test]
fn unknown_subcommand_is_a_usage_error() {
    assert_usage_error(UNKNOWN_SUB);
}

#[test]
fn unknown_flag_is_a_usage_error() {
    assert_usage_error(&["--no-such-flag"]);
}

/// Usage errors the contract names: missing required values and values outside an
/// enumerated set must be refused by the parser, not reach the command.
#[test]
fn malformed_arguments_are_usage_errors() {
    assert_usage_error(&["render", "--domain", "nosuch"]);
    assert_usage_error(&["render", "--format", "nosuch"]);
}

/// Before #4079 `validate` on an empty file, or on `foo: 1`, printed
/// "Schema validation PASSED" and exited 0: the schema check was a stub returning Ok.
#[test]
fn unusable_input_is_rejected() {
    for b in &unusable_inputs() {
        assert_rejected(&["run", b.as_str()]);
        assert_rejected(&["validate", b.as_str()]);
        assert_rejected(&["verify", b.as_str()]);
        assert_rejected(&["emc-check", b.as_str()]);
        assert_rejected(&["emc-validate", b.as_str()]);
    }
}
