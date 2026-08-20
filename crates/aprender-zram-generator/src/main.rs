//! systemd generator for zram device configuration.
//!
//! This generator runs during early boot to create systemd units
//! for zram device setup based on system configuration.
//!
//! Argument parsing is DECLARATIVE (clap derive). It used to index
//! `std::env::args()` directly, which meant `--help` printed 0 BYTES and an
//! unknown flag was ACCEPTED at exit 0 — a typo'd flag became the generator's
//! output DIRECTORY, so `trueno-zram-generator --help` created a directory
//! literally named `--help` and wrote unit files into it.
//! See `scripts/check_no_hand_rolled_parsers.sh`.

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]
#![cfg_attr(test, allow(clippy::disallowed_methods))] // test assertions — unwrap acceptable

mod config;
mod fstab;
mod unit;

use clap::Parser;
use std::process::ExitCode;

/// systemd generator for zram device configuration.
///
/// systemd invokes generators as `generator normal_dir early_dir late_dir`
/// (see `systemd.generator(7)`). The three positional arguments below preserve
/// that protocol exactly: same order, same meaning.
#[derive(Debug, Parser)]
#[command(
    name = "trueno-zram-generator",
    version,
    about = "systemd generator for zram device configuration",
    long_about = "systemd generator for zram device configuration.\n\n\
                  systemd invokes generators as `generator <NORMAL_DIR> <EARLY_DIR> <LATE_DIR>`; \
                  see systemd.generator(7). Units are written into NORMAL_DIR."
)]
// The `_dir` postfix is systemd's own vocabulary for these three arguments
// (systemd.generator(7)); renaming them to satisfy `struct_field_names` would
// make the protocol harder to recognise, not easier.
#[allow(clippy::struct_field_names)]
struct Cli {
    /// systemd normal-priority generator directory (units are written here).
    #[arg(value_name = "NORMAL_DIR")]
    normal_dir: String,

    /// systemd early-priority generator directory (accepted, currently unused).
    // Part of the systemd generator protocol: systemd always passes it. It is
    // accepted so the argument grammar matches what systemd sends, and so a
    // typo'd flag is REJECTED instead of sliding into a directory slot.
    #[allow(dead_code)]
    #[arg(value_name = "EARLY_DIR")]
    early_dir: Option<String>,

    /// systemd late-priority generator directory (accepted, currently unused).
    #[allow(dead_code)]
    #[arg(value_name = "LATE_DIR")]
    late_dir: Option<String>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("trueno-zram-generator: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    // Load configuration
    let config = config::load_config()?;

    // Generate systemd units
    unit::generate_units(&cli.normal_dir, &config)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::error::ErrorKind;
    use clap::{CommandFactory, Parser};

    /// Catches duplicate short options and other grammar defects.
    ///
    /// clap only runs this under `#[cfg(debug_assertions)]`, so without an
    /// explicit test a release build ships the ambiguity.
    #[test]
    fn test_cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    /// The systemd protocol: three positional directories, in order.
    #[test]
    fn test_systemd_three_positional_directories_in_order() {
        let cli = Cli::try_parse_from([
            "trueno-zram-generator",
            "/run/systemd/generator",
            "/run/systemd/generator.early",
            "/run/systemd/generator.late",
        ])
        .expect("the systemd generator invocation must parse");
        assert_eq!(cli.normal_dir, "/run/systemd/generator");
        assert_eq!(
            cli.early_dir.as_deref(),
            Some("/run/systemd/generator.early")
        );
        assert_eq!(cli.late_dir.as_deref(), Some("/run/systemd/generator.late"));
    }

    /// The previous parser required only `normal_dir`; that stays true.
    #[test]
    fn test_normal_dir_alone_is_accepted() {
        let cli = Cli::try_parse_from(["trueno-zram-generator", "/run/systemd/generator"])
            .expect("normal_dir alone must still parse");
        assert_eq!(cli.normal_dir, "/run/systemd/generator");
        assert!(cli.early_dir.is_none());
        assert!(cli.late_dir.is_none());
    }

    #[test]
    fn test_missing_normal_dir_is_error() {
        let err =
            Cli::try_parse_from(["trueno-zram-generator"]).expect_err("normal_dir is required");
        assert_eq!(err.kind(), ErrorKind::MissingRequiredArgument);
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
    }

    /// THE headline defect: an unknown flag used to be accepted at exit 0 and
    /// treated as the output DIRECTORY.
    #[test]
    fn test_unknown_flag_is_error_not_a_directory() {
        let err = Cli::try_parse_from([
            "trueno-zram-generator",
            "--definitely-not-a-real-flag-xyz",
            "/run/systemd/generator",
        ])
        .expect_err("an unknown flag must be rejected, never used as a directory");
        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
    }

    /// The same flag as the ONLY argument: it must not become `normal_dir`.
    #[test]
    fn test_lone_unknown_flag_is_error() {
        let err = Cli::try_parse_from(["trueno-zram-generator", "--help-me-please"])
            .expect_err("a lone unknown flag must be rejected");
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
        assert_ne!(
            err.kind(),
            ErrorKind::DisplayHelp,
            "an unknown flag is not a help request"
        );
    }

    /// A fourth positional has no meaning in the generator protocol.
    #[test]
    fn test_too_many_positionals_is_error() {
        let err = Cli::try_parse_from([
            "trueno-zram-generator",
            "/run/systemd/generator",
            "/run/systemd/generator.early",
            "/run/systemd/generator.late",
            "/run/systemd/generator.extra",
        ])
        .expect_err("a fourth directory must be rejected");
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
    }

    /// `--help` must exit 0 and print real help. It used to print 0 BYTES and
    /// create a directory named `--help`.
    #[test]
    fn test_help_exits_zero_with_substantial_output() {
        let err = Cli::try_parse_from(["trueno-zram-generator", "--help"])
            .expect_err("--help short-circuits parsing via an Err carrying the help text");
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
        assert_eq!(err.exit_code(), 0, "--help must exit 0");

        let rendered = err.render().to_string();
        assert!(
            rendered.len() > 100,
            "help must be substantial, got {} bytes",
            rendered.len()
        );
        for expected in ["NORMAL_DIR", "EARLY_DIR", "LATE_DIR"] {
            assert!(
                rendered.contains(expected),
                "help must document {expected}, got:\n{rendered}"
            );
        }
    }
}
