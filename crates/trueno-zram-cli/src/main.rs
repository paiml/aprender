//! trueno-zram CLI - zramctl replacement with SIMD acceleration.

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]

mod commands;
mod output;

use clap::{Parser, Subcommand};
use std::process::ExitCode;

/// trueno-zram: SIMD-accelerated zram management
#[derive(Parser)]
#[command(name = "trueno-zram")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Output format
    #[arg(long, default_value = "table")]
    format: output::OutputFormat,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Create and configure a zram device
    Create(commands::CreateArgs),

    /// Remove a zram device
    Remove(commands::RemoveArgs),

    /// Show zram device status
    Status(commands::StatusArgs),

    /// Run compression benchmarks
    Benchmark(commands::BenchmarkArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match &cli.command {
        Commands::Create(args) => commands::create(args),
        Commands::Remove(args) => commands::remove(args),
        Commands::Status(args) => commands::status(args, cli.format),
        Commands::Benchmark(args) => commands::benchmark(args),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("Error: {e}");
            ExitCode::FAILURE
        }
    }
}
