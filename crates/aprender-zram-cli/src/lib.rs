//! trueno-zram CLI - zramctl replacement with SIMD acceleration.
//!
//! The command surface lives here rather than in `main.rs` so that something
//! other than the `trueno-zram` binary can reach it. A command enum declared in
//! a binary target is importable by nothing: the standalone binary was the only
//! way to run any of this, which is exactly what the APR-MONO consolidation is
//! meant to end. `apr zram <cmd>` and `trueno-zram <cmd>` now call the SAME
//! [`dispatch`], so the two surfaces cannot drift.

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]

pub mod commands;
pub mod output;

use clap::{Parser, Subcommand};
use std::process::ExitCode;

/// trueno-zram: SIMD-accelerated zram management
#[derive(Parser)]
#[command(name = "trueno-zram")]
#[command(author, version = concat!(env!("CARGO_PKG_VERSION"), " (", env!("APR_GIT_SHA"), ")"), about, long_about = None)]
pub struct Cli {
    /// Output format
    #[arg(long, default_value = "table")]
    pub format: output::OutputFormat,

    /// The command to run
    #[command(subcommand)]
    pub command: Commands,
}

/// The arguments `apr zram` takes: the same subcommands as [`Cli`] plus the
/// same `--format`, so no output format is reachable only from the standalone
/// binary. `--format` is optional here because `apr` already has a global
/// `--json`; [`resolve_format`] reconciles the two (#4060).
#[derive(clap::Args, Debug)]
pub struct ZramArgs {
    /// Output format [default: json under `apr --json`, otherwise table]
    #[arg(long, value_enum)]
    pub format: Option<output::OutputFormat>,

    /// The command to run
    #[command(subcommand)]
    pub command: Commands,
}

/// The format `apr zram` runs with: an explicit `--format` wins, then apr's
/// global `--json`, then the standalone binary's default, `table`.
#[must_use]
pub fn resolve_format(explicit: Option<output::OutputFormat>, json: bool) -> output::OutputFormat {
    match (explicit, json) {
        (Some(format), _) => format,
        (None, true) => output::OutputFormat::Json,
        (None, false) => output::OutputFormat::Table,
    }
}

/// Every zram management operation the CLI offers.
#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Create and configure a zram device
    Create(commands::CreateArgs),

    /// Remove a zram device
    Remove(commands::RemoveArgs),

    /// Show zram device status
    Status(commands::StatusArgs),

    /// Run compression benchmarks
    Benchmark(commands::BenchmarkArgs),
}

/// Parse `argv` and run one command. This is the whole of the standalone
/// binary.
#[must_use]
pub fn run() -> ExitCode {
    let cli = Cli::parse();
    match dispatch(&cli.command, cli.format) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Run one already-parsed command.
///
/// Split out from [`run`] so a caller that did its own parsing -- `apr zram` --
/// executes the identical code path instead of a copy of it. Returns the error
/// rather than an exit code so each front end can report it in its own idiom.
///
/// # Errors
/// Propagates whatever the selected command returns.
pub fn dispatch(
    command: &Commands,
    format: output::OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match command {
        Commands::Create(args) => commands::create(args),
        Commands::Remove(args) => commands::remove(args),
        Commands::Status(args) => commands::status(args, format),
        Commands::Benchmark(args) => commands::benchmark(args),
    }
}

#[cfg(test)]
#[allow(clippy::panic)] // a test reports a rejected word by panicking
mod zram_args_tests {
    use super::{output::OutputFormat, resolve_format, ZramArgs};
    use clap::Parser;

    #[derive(Parser)]
    struct Host {
        #[command(flatten)]
        args: ZramArgs,
    }

    #[test]
    fn explicit_format_wins_over_json() {
        assert!(matches!(
            resolve_format(Some(OutputFormat::Raw), true),
            OutputFormat::Raw
        ));
        assert!(matches!(
            resolve_format(Some(OutputFormat::Table), true),
            OutputFormat::Table
        ));
    }

    #[test]
    fn absent_format_follows_json_then_table() {
        assert!(matches!(resolve_format(None, true), OutputFormat::Json));
        assert!(matches!(resolve_format(None, false), OutputFormat::Table));
    }

    #[test]
    fn every_format_the_binary_accepts_parses_through_zram_args() {
        for (word, want) in [
            ("table", OutputFormat::Table),
            ("json", OutputFormat::Json),
            ("raw", OutputFormat::Raw),
        ] {
            let host = Host::try_parse_from(["apr-zram", "--format", word, "status"])
                .unwrap_or_else(|e| panic!("--format {word} rejected: {e}"));
            assert_eq!(
                std::mem::discriminant(&host.args.format.expect("format parsed")),
                std::mem::discriminant(&want),
                "--format {word}"
            );
        }
        let host = Host::try_parse_from(["apr-zram", "status"]).expect("no --format parses");
        assert!(host.args.format.is_none());
    }
}
