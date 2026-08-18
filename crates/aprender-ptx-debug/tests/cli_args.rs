//! Falsification tests for the `aprender-ptx-debug` argument parser.
//!
//! This CLI used hand-rolled `match args[1]` dispatch. The identical pattern in
//! a sibling crate silently dropped `--seed`: unknown flags fell through a
//! catch-all arm, a flag given without a value was discarded, and an
//! unparseable value became a default instead of an error. Each test below pins
//! one of those failure modes to a hard parse error, so a regression back to a
//! permissive parser turns the suite red.

use clap::error::ErrorKind;
use clap::{CommandFactory, Parser};
use trueno_ptx_debug::cli::{exit_code_for_parse_error, AnalyzeArgs, Cli, Command, GenFkrArgs};

fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
    Cli::try_parse_from(args)
}

/// Parse arguments that are expected to fail, returning the error kind.
fn parse_err_kind(args: &[&str]) -> ErrorKind {
    match parse(args) {
        Ok(cli) => panic!(
            "expected `{}` to be rejected, parsed {cli:?}",
            args.join(" ")
        ),
        Err(e) => e.kind(),
    }
}

fn analyze_args(args: &[&str]) -> AnalyzeArgs {
    match parse(args)
        .unwrap_or_else(|e| panic!("expected `{}` to parse: {e}", args.join(" ")))
        .command
    {
        Command::Analyze(a) => a,
        other => panic!("expected `analyze`, got {other:?}"),
    }
}

fn gen_fkr_args(args: &[&str]) -> GenFkrArgs {
    match parse(args)
        .unwrap_or_else(|e| panic!("expected `{}` to parse: {e}", args.join(" ")))
        .command
    {
        Command::GenFkr(a) => a,
        other => panic!("expected `gen-fkr`, got {other:?}"),
    }
}

/// clap's own structural validation of the command tree.
#[test]
fn command_tree_is_valid() {
    Cli::command().debug_assert();
}

// --- Failure mode 1: unknown flags must not be silently ignored -------------

#[test]
fn unknown_flag_is_rejected_not_ignored() {
    // The literal defect from the sibling crate: `--seed` is not a flag of this
    // CLI, so it must be an error rather than being dropped on the floor.
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "analyze", "k.ptx", "--seed", "42"]),
        ErrorKind::UnknownArgument
    );
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "analyze", "k.ptx", "--nope"]),
        ErrorKind::UnknownArgument
    );
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "gen-fkr", "k.ptx", "--seed", "42"]),
        ErrorKind::UnknownArgument
    );
}

#[test]
fn unknown_short_flag_is_rejected() {
    // `-o` belongs to gen-fkr only; analyze must not quietly accept it.
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "analyze", "k.ptx", "-o", "out.rs"]),
        ErrorKind::UnknownArgument
    );
}

#[test]
fn extra_positional_is_rejected() {
    // The hand-rolled parser let the last positional silently win.
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "analyze", "a.ptx", "b.ptx"]),
        ErrorKind::UnknownArgument
    );
}

#[test]
fn unknown_subcommand_is_rejected() {
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "bogus"]),
        ErrorKind::InvalidSubcommand
    );
}

// --- Failure mode 2: a valued flag given without a value must be an error ---

#[test]
fn valued_flag_without_value_is_rejected() {
    for args in [
        &["aprender-ptx-debug", "analyze", "k.ptx", "--min-score"][..],
        &["aprender-ptx-debug", "analyze", "k.ptx", "--html"][..],
        &["aprender-ptx-debug", "gen-fkr", "k.ptx", "-o"][..],
    ] {
        assert_eq!(
            parse_err_kind(args),
            ErrorKind::InvalidValue,
            "`{}` must not discard the dangling flag",
            args.join(" ")
        );
    }
}

#[test]
fn missing_required_file_is_rejected() {
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "analyze"]),
        ErrorKind::MissingRequiredArgument
    );
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "gen-fkr"]),
        ErrorKind::MissingRequiredArgument
    );
}

#[test]
fn no_arguments_is_rejected() {
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug"]),
        ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
    );
}

// --- Failure mode 3: an unparseable value must error, not fall back ---------

#[test]
fn unparseable_min_score_is_an_error_not_the_default() {
    assert_eq!(
        parse_err_kind(&[
            "aprender-ptx-debug",
            "analyze",
            "k.ptx",
            "--min-score",
            "notanumber",
        ]),
        ErrorKind::ValueValidation
    );

    // Positive control: the flag really is wired up, so the assertion above
    // cannot be passing merely because `--min-score` is ignored outright.
    let ok = analyze_args(&[
        "aprender-ptx-debug",
        "analyze",
        "k.ptx",
        "--min-score",
        "91.5",
    ]);
    assert!(
        (ok.min_score - 91.5).abs() < f64::EPSILON,
        "min_score should be 91.5, got {}",
        ok.min_score
    );
    assert!(
        (ok.min_score - 70.0).abs() > f64::EPSILON,
        "min_score must not fall back to the 70.0 default"
    );
}

// --- Every subcommand is reachable -----------------------------------------

#[test]
fn every_subcommand_is_reachable() {
    assert!(matches!(
        parse(&["aprender-ptx-debug", "analyze", "k.ptx"]).map(|c| c.command),
        Ok(Command::Analyze(_))
    ));
    assert!(matches!(
        parse(&["aprender-ptx-debug", "gen-fkr", "k.ptx"]).map(|c| c.command),
        Ok(Command::GenFkr(_))
    ));
    assert!(matches!(
        parse(&["aprender-ptx-debug", "version"]).map(|c| c.command),
        Ok(Command::Version)
    ));
    // `help`, `--help` and `--version` are reported by clap as errors that the
    // binary turns into a successful exit.
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "help"]),
        ErrorKind::DisplayHelp
    );
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "--help"]),
        ErrorKind::DisplayHelp
    );
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "-h"]),
        ErrorKind::DisplayHelp
    );
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "--version"]),
        ErrorKind::DisplayVersion
    );
    assert_eq!(
        parse_err_kind(&["aprender-ptx-debug", "-V"]),
        ErrorKind::DisplayVersion
    );
}

// --- Flags and defaults preserved from the hand-rolled parser ---------------

#[test]
fn analyze_defaults_match_the_documented_behaviour() {
    let args = analyze_args(&["aprender-ptx-debug", "analyze", "kernel.ptx"]);
    assert_eq!(args.file, "kernel.ptx");
    assert!(!args.falsify);
    assert!(!args.json);
    assert_eq!(args.html, None);
    assert!(
        (args.min_score - 70.0).abs() < f64::EPSILON,
        "documented default min-score is 70, got {}",
        args.min_score
    );
}

#[test]
fn analyze_accepts_every_documented_flag() {
    let args = analyze_args(&[
        "aprender-ptx-debug",
        "analyze",
        "kernel.ptx",
        "--falsify",
        "--min-score",
        "90",
        "--html",
        "report.html",
        "--json",
    ]);
    assert_eq!(args.file, "kernel.ptx");
    assert!(args.falsify);
    assert!(args.json);
    assert_eq!(args.html.as_deref(), Some("report.html"));
    assert!((args.min_score - 90.0).abs() < f64::EPSILON);
}

#[test]
fn gen_fkr_accepts_its_output_flag() {
    let defaults = gen_fkr_args(&["aprender-ptx-debug", "gen-fkr", "kernel.ptx"]);
    assert_eq!(defaults.file, "kernel.ptx");
    assert_eq!(defaults.output, None, "gen-fkr defaults to stdout");

    let with_output = gen_fkr_args(&[
        "aprender-ptx-debug",
        "gen-fkr",
        "kernel.ptx",
        "-o",
        "tests/kernel_fkr.rs",
    ]);
    assert_eq!(with_output.output.as_deref(), Some("tests/kernel_fkr.rs"));
}

// --- Exit code mapping ------------------------------------------------------

#[test]
fn help_and_version_exit_zero_every_other_parse_failure_exits_one() {
    let code = |args: &[&str]| match parse(args) {
        Ok(_) => panic!("`{}` should not parse cleanly", args.join(" ")),
        Err(e) => exit_code_for_parse_error(&e),
    };

    assert_eq!(code(&["aprender-ptx-debug", "--help"]), 0);
    assert_eq!(code(&["aprender-ptx-debug", "help"]), 0);
    assert_eq!(code(&["aprender-ptx-debug", "--version"]), 0);

    // Preserved from the hand-rolled parser: usage failures exit 1.
    assert_eq!(code(&["aprender-ptx-debug"]), 1);
    assert_eq!(code(&["aprender-ptx-debug", "bogus"]), 1);
    assert_eq!(code(&["aprender-ptx-debug", "analyze"]), 1);
    assert_eq!(
        code(&["aprender-ptx-debug", "analyze", "k.ptx", "--seed", "42"]),
        1
    );
}
