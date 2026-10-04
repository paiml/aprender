//! ONT-10 surface gate for the `aprender-zram-generator` binary: `contracts/bin-aprender-zram-generator--aprender-zram-generator-v1.yaml` (aprender#4079).
//!
//! Every assertion runs the binary built from this tree (`CARGO_BIN_EXE_aprender-zram-generator`), which
//! cannot fall back to a stale copy on PATH. A test that only asserted success on good
//! input would pass against `fn main() {}`, so each invariant here is one the binary
//! can actually fail: a rejected input must exit non-zero AND must not be a panic.

use std::path::PathBuf;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_aprender-zram-generator");

/// An argument list the parser must refuse as an unknown subcommand or extra positional.
const UNKNOWN_SUB: &[&str] = &["a", "b", "c", "one-too-many"];

/// The pinned surface: `aprender-zram-generator` has no subcommands, so its options are the command set.
/// Adding or removing an option edits this list and the contract's FALSIFY-BIN-*-004
/// prediction together. `--help`/`--version` are clap's and are not listed.
const OPTIONS: &[&str] = &[];

/// The positionals, in order.
const POSITIONALS: &[&str] = &["NORMAL_DIR", "EARLY_DIR", "LATE_DIR"];

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

fn advertised_options() -> Vec<String> {
    help_section("Options:")
        .iter()
        .filter_map(|l| l.split_whitespace().find(|w| w.starts_with("--")))
        .map(|w| w.trim_end_matches(',').to_owned())
        .filter(|o| o != "--help" && o != "--version")
        .collect()
}

fn assert_rejected(args: &[&str]) {
    let out = run(args);
    let code = out.status.code();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        matches!(code, Some(c) if c != 0 && c != 101) && !stderr.contains("panicked"),
        "`aprender-zram-generator {}` must reject its input with a clean non-zero exit, got {code:?}\nstderr: {stderr}",
        args.join(" ")
    );
}

fn assert_usage_error(args: &[&str]) {
    let out = run(args);
    assert_eq!(
        out.status.code(),
        Some(2),
        "`aprender-zram-generator {}` must be a usage error (exit 2)\nstderr: {}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
}

#[test]
fn version_reports_the_crate_version() {
    let out = run(&["--version"]);
    assert!(out.status.success());
    let stdout = String::from_utf8_lossy(&out.stdout);
    let want = format!("aprender-zram-generator {}", env!("CARGO_PKG_VERSION"));
    assert!(
        stdout.starts_with(&want),
        "--version printed {stdout:?}, not {want:?}"
    );
}

#[test]
fn advertised_options_are_the_pinned_surface() {
    assert!(
        help_section("Commands:").is_empty(),
        "a subcommand appeared; pin SUBCOMMANDS instead of OPTIONS"
    );
    assert_eq!(advertised_options(), OPTIONS);
    let args: Vec<String> = help_section("Arguments:")
        .iter()
        .filter_map(|l| l.split_whitespace().next())
        .map(|w| {
            w.trim_matches(|c| matches!(c, '<' | '>' | '[' | ']'))
                .to_owned()
        })
        .collect();
    assert_eq!(args, POSITIONALS);
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
    assert_usage_error(&[]);
}

/// A generator creates its output directory, so a missing NORMAL_DIR is valid input;
/// the unusable one is a path under a regular file, which cannot be a directory.
#[test]
fn unusable_input_is_rejected() {
    let dir =
        PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("aprender-zram-generator-surface-gate");
    std::fs::create_dir_all(&dir).expect("create fixture dir");
    let file = dir.join("a-regular-file");
    std::fs::write(&file, b"").expect("write fixture");
    let sub = file.join("sub").display().to_string();
    assert_rejected(&[sub.as_str()]);
    let out = run(&[sub.as_str()]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.starts_with("aprender-zram-generator: "),
        "the error names the binary, not a legacy name: {stderr:?}"
    );
}
