//! Update README.md certification table from models.csv.
//!
//! Usage: apr-qa-readme-sync [--csv PATH] [--readme PATH]
//!
//! Argument parsing is DECLARATIVE (clap derive). It used to be a hand-rolled
//! `while i < args.len()` loop whose `_ => {}` catch-all silently swallowed
//! unknown flags, stray positionals, and `--csv` given without a value.
//! See `scripts/check_no_hand_rolled_parsers.sh`.

#![forbid(unsafe_code)]

use aprender_qa_certify::{
    generate_summary, generate_table, parse_csv, update_readme, CertifyError, END_MARKER,
    START_MARKER,
};
use chrono::Utc;
use clap::Parser;
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

fn find_project_root() -> Option<PathBuf> {
    let mut current = env::current_dir().ok()?;
    loop {
        if current.join("Cargo.toml").exists() && current.join("crates").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

/// Update the README.md certification table from models.csv.
#[derive(Debug, Parser)]
#[command(
    name = "apr-qa-readme-sync",
    version,
    about = "Updates README.md certification table from models.csv",
    long_about = None
)]
struct Cli {
    /// Path to models.csv (default: docs/certifications/models.csv)
    #[arg(long, value_name = "PATH")]
    csv: Option<PathBuf>,

    /// Path to README.md (default: README.md)
    #[arg(long, value_name = "PATH")]
    readme: Option<PathBuf>,
}

fn validate_readme_markers(content: &str) -> Result<(), CertifyError> {
    if !content.contains(START_MARKER) {
        return Err(CertifyError::MarkerNotFound(format!(
            "README is missing start marker: {START_MARKER}"
        )));
    }
    if !content.contains(END_MARKER) {
        return Err(CertifyError::MarkerNotFound(format!(
            "README is missing end marker: {END_MARKER}"
        )));
    }
    Ok(())
}

fn run(cli: Cli) -> Result<(), CertifyError> {
    // Find project root
    let root = find_project_root().ok_or_else(|| {
        CertifyError::MarkerNotFound(
            "Could not find project root (looking for Cargo.toml + crates/)".to_string(),
        )
    })?;

    let csv_path = cli
        .csv
        .unwrap_or_else(|| root.join("docs/certifications/models.csv"));
    let readme_path = cli.readme.unwrap_or_else(|| root.join("README.md"));

    // Read CSV
    eprintln!("Reading CSV from: {}", csv_path.display());
    let csv_content = fs::read_to_string(&csv_path)?;
    let models = parse_csv(&csv_content)?;
    eprintln!("Loaded {} models", models.len());

    // Generate content
    let timestamp = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
    let summary = generate_summary(&models, &timestamp);
    let table = generate_table(&models);
    let full_content = format!("{summary}\n\n{table}");

    // Read and update README
    eprintln!("Reading README from: {}", readme_path.display());
    let readme_content = fs::read_to_string(&readme_path)?;
    validate_readme_markers(&readme_content)?;

    let updated_readme = update_readme(&readme_content, &full_content)?;

    // Write updated README
    fs::write(&readme_path, updated_readme)?;
    eprintln!("Updated {}", readme_path.display());
    eprintln!("Done. Commit both README.md and docs/certifications/models.csv together.");

    Ok(())
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::error::ErrorKind;
    use clap::{CommandFactory, Parser};
    use std::path::PathBuf;

    /// Catches duplicate short options and other grammar defects.
    ///
    /// clap only runs this under `#[cfg(debug_assertions)]`, so without an
    /// explicit test a release build ships the ambiguity.
    #[test]
    fn test_cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn test_no_args_leaves_both_paths_unset() {
        let cli = Cli::try_parse_from(["apr-qa-readme-sync"]).expect("no args must parse");
        assert!(cli.csv.is_none(), "csv default is resolved at runtime");
        assert!(
            cli.readme.is_none(),
            "readme default is resolved at runtime"
        );
    }

    #[test]
    fn test_both_flags_are_honoured() {
        let cli = Cli::try_parse_from([
            "apr-qa-readme-sync",
            "--csv",
            "/tmp/models.csv",
            "--readme",
            "/tmp/README.md",
        ])
        .expect("both flags must parse");
        assert_eq!(cli.csv, Some(PathBuf::from("/tmp/models.csv")));
        assert_eq!(cli.readme, Some(PathBuf::from("/tmp/README.md")));
    }

    /// An unknown flag must be an ERROR.
    ///
    /// The hand-rolled loop had a `_ => {}` catch-all, so a typo'd flag was
    /// silently dropped and the tool ran with defaults.
    #[test]
    fn test_unknown_flag_is_error() {
        let err = Cli::try_parse_from(["apr-qa-readme-sync", "--definitely-not-a-real-flag-xyz"])
            .expect_err("an unknown flag must be rejected");
        assert_eq!(err.kind(), ErrorKind::UnknownArgument);
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
    }

    /// A near-miss of a real flag must be rejected too, not silently ignored.
    #[test]
    fn test_misspelled_known_flag_is_error() {
        let err = Cli::try_parse_from(["apr-qa-readme-sync", "--csvv", "/tmp/models.csv"])
            .expect_err("--csvv must be rejected");
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
    }

    /// A stray positional must be an ERROR. This binary takes no positionals,
    /// and the hand-rolled loop ignored them.
    #[test]
    fn test_stray_positional_is_error() {
        let err = Cli::try_parse_from(["apr-qa-readme-sync", "models.csv"])
            .expect_err("a stray positional must be rejected");
        assert_ne!(err.exit_code(), 0, "must exit non-zero");
    }

    /// `--csv` with no value must be an ERROR, not a silent fallback to the
    /// default. The hand-rolled arm was guarded by `if i + 1 < args.len()`, so
    /// a trailing `--csv` fell into the catch-all and vanished.
    #[test]
    fn test_flag_without_value_is_error() {
        for flag in ["--csv", "--readme"] {
            let parsed = Cli::try_parse_from(["apr-qa-readme-sync", flag]);
            assert!(
                parsed.is_err(),
                "{flag} with no value must be an error, not a silent default"
            );
            let err = parsed.expect_err("checked is_err above");
            assert_ne!(
                err.exit_code(),
                0,
                "{flag} with no value must exit non-zero"
            );
            assert_ne!(
                err.kind(),
                ErrorKind::DisplayHelp,
                "{flag} with no value is not a help request"
            );
        }
    }

    /// `--help` must exit 0 and print real help.
    #[test]
    fn test_help_exits_zero_with_substantial_output() {
        let err = Cli::try_parse_from(["apr-qa-readme-sync", "--help"])
            .expect_err("--help short-circuits parsing via an Err carrying the help text");
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
        assert_eq!(err.exit_code(), 0, "--help must exit 0");

        let rendered = err.render().to_string();
        assert!(
            rendered.len() > 100,
            "help must be substantial, got {} bytes",
            rendered.len()
        );
        for expected in ["--csv", "--readme", "models.csv"] {
            assert!(
                rendered.contains(expected),
                "help must document {expected}, got:\n{rendered}"
            );
        }
    }

    /// `-h` was accepted by the old parser and must still be.
    #[test]
    fn test_short_help_still_accepted() {
        let err = Cli::try_parse_from(["apr-qa-readme-sync", "-h"])
            .expect_err("-h short-circuits parsing");
        assert_eq!(err.kind(), ErrorKind::DisplayHelp);
        assert_eq!(err.exit_code(), 0, "-h must exit 0");
    }
}
