//! Trueno development task runner (xtask pattern)
//!
//! Provides development utilities:
//! - `check-simd`: Validate SIMD backend detection
//! - `install-hooks`: Install git hooks
//! - `validate-examples`: Check example quality
//!
//! Argument parsing is DECLARATIVE (clap derive). It used to be a
//! `match args[1]` over `std::env::args()`, which made `--help` an
//! "Unknown command" that exited 1. See `scripts/check_no_hand_rolled_parsers.sh`.

// Development-phase lint allows
#![allow(clippy::useless_vec)]
#![cfg_attr(test, allow(clippy::disallowed_methods))]

mod check_simd;
mod install_hooks;
mod validate_examples;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::process::ExitCode;

/// Trueno development task runner.
#[derive(Debug, Parser)]
#[command(
    version,
    about = "Trueno development task runner (xtask pattern)",
    long_about = "Trueno development task runner (xtask pattern).\n\n\
                  Normally invoked as `cargo xtask <COMMAND>`."
)]
struct Cli {
    /// Development task to run.
    #[command(subcommand)]
    command: Commands,
}

/// Development tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Subcommand)]
enum Commands {
    /// Check SIMD attributes (pre-commit validation)
    CheckSimd,
    /// Install git pre-commit hooks
    InstallHooks,
    /// Validate book examples meet EXTREME TDD quality
    ValidateExamples,
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match dispatch(cli.command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {}", e);
            ExitCode::FAILURE
        }
    }
}

/// Run a parsed command (pure dispatch, testable).
fn dispatch(command: Commands) -> Result<()> {
    match command {
        Commands::CheckSimd => check_simd::run(),
        Commands::InstallHooks => install_hooks::run(),
        Commands::ValidateExamples => validate_examples::run(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::error::ErrorKind;
    use clap::CommandFactory;

    /// Catches duplicate short options and other grammar defects.
    ///
    /// clap only runs this under `#[cfg(debug_assertions)]`, so without an
    /// explicit test a release build ships the ambiguity.
    #[test]
    fn test_cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_parse_check_simd() {
        let cli = Cli::try_parse_from(["xtask", "check-simd"]).expect("check-simd must parse");
        assert_eq!(cli.command, Commands::CheckSimd);
    }

    #[test]
    fn test_parse_install_hooks() {
        let cli =
            Cli::try_parse_from(["xtask", "install-hooks"]).expect("install-hooks must parse");
        assert_eq!(cli.command, Commands::InstallHooks);
    }

    #[test]
    fn test_parse_validate_examples() {
        let cli =
            Cli::try_parse_from(["xtask", "validate-examples"]).expect("validate-examples parses");
        assert_eq!(cli.command, Commands::ValidateExamples);
    }

    #[test]
    fn test_no_subcommand_is_error() {
        let err = Cli::try_parse_from(["xtask"]).expect_err("a bare invocation must not succeed");
        assert_eq!(
            err.kind(),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
        assert!(
            err.render().to_string().contains("check-simd"),
            "the error must still list the available commands"
        );
    }

    /// The hand-rolled parser's catch-all turned every unknown token into
    /// "Unknown command". clap must reject it as an unknown SUBCOMMAND.
    #[test]
    fn test_unknown_subcommand_is_error() {
        let err = Cli::try_parse_from(["xtask", "unknown"]).expect_err("unknown must be rejected");
        assert_eq!(err.kind(), ErrorKind::InvalidSubcommand);
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
    }

    /// An unknown FLAG must be an error, never silently ignored.
    #[test]
    fn test_unknown_flag_is_error() {
        let err = Cli::try_parse_from(["xtask", "--definitely-not-a-real-flag-xyz"])
            .expect_err("an unknown flag must be rejected");
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
        assert_ne!(
            err.kind(),
            ErrorKind::DisplayHelp,
            "an unknown flag is not a help request"
        );
    }

    /// An unknown flag AFTER a valid subcommand must also be rejected.
    #[test]
    fn test_unknown_flag_after_subcommand_is_error() {
        let err = Cli::try_parse_from(["xtask", "check-simd", "--nope"])
            .expect_err("an unknown flag after a subcommand must be rejected");
        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
    }

    /// A stray positional after a subcommand must be rejected, not ignored.
    ///
    /// The hand-rolled parser read `args[1]` only, so `xtask check-simd junk`
    /// silently ran check-simd.
    #[test]
    fn test_extra_positional_is_error() {
        let err = Cli::try_parse_from(["xtask", "check-simd", "junk"])
            .expect_err("a stray positional must be rejected");
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
    }

    /// `--help` must exit 0 and print real help. It used to exit 1 with
    /// "Error: Unknown command: --help".
    #[test]
    fn test_help_exits_zero_with_substantial_output() {
        let err = Cli::try_parse_from(["xtask", "--help"])
            .expect_err("--help short-circuits parsing via an Err carrying the help text");
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
        assert_eq!(err.exit_code(), 0, "--help must exit 0");

        let rendered = err.render().to_string();
        assert!(
            rendered.len() > 100,
            "help must be substantial, got {} bytes",
            rendered.len()
        );
        for expected in ["check-simd", "install-hooks", "validate-examples"] {
            assert!(
                rendered.contains(expected),
                "help must document {expected}, got:\n{rendered}"
            );
        }
    }

    #[test]
    fn test_help_subcommand_and_short_flag_agree() {
        for args in [
            vec!["xtask", "-h"],
            vec!["xtask", "help"],
            vec!["xtask", "--help"],
        ] {
            let err = Cli::try_parse_from(args.clone())
                .expect_err("help forms short-circuit parsing with an Err");
            assert_eq!(err.exit_code(), 0, "{args:?} must exit 0");
        }
    }

    #[test]
    fn test_dispatch_validate_examples() {
        // Fails to find examples/ in the test environment, which proves dispatch
        // reached validate_examples rather than returning a parse error.
        let result = dispatch(Commands::ValidateExamples);
        assert!(result.is_err());
    }

    #[test]
    fn test_dispatch_check_simd_does_not_panic() {
        // Runs against actual source files; the outcome depends on the tree.
        let _ = dispatch(Commands::CheckSimd);
    }

    #[test]
    fn test_dispatch_install_hooks_does_not_panic() {
        // Requires a .git directory; the outcome depends on the checkout.
        let _ = dispatch(Commands::InstallHooks);
    }
}
