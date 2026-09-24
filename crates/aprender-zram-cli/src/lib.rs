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
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Output format
    #[arg(long, default_value = "table")]
    pub format: output::OutputFormat,

    /// The command to run
    #[command(subcommand)]
    pub command: Commands,
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
