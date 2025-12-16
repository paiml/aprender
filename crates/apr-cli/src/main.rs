//! apr - APR Model Operations CLI
//!
//! Toyota Way: Genchi Genbutsu (Go and See)
//! See the actual model data, not abstractions.
//!
//! Usage:
//!   apr inspect model.apr           # Inspect model metadata
//!   apr debug model.apr             # Simple debugging output
//!   apr debug model.apr --drama     # Theatrical debugging
//!   apr validate model.apr          # Validate model integrity
//!   apr diff model1.apr model2.apr  # Compare models
//!   apr tensors model.apr           # List tensor names and shapes
//!   apr tensors model.apr --stats   # Show tensor statistics
//!   apr trace model.apr             # Layer-by-layer analysis
//!   apr probar model.apr -o ./out   # Export for probar visual testing

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::ExitCode;

mod commands;
mod error;
mod output;

use commands::{debug, diff, inspect, probar, tensors, trace, validate};

/// apr - APR Model Operations Tool
///
/// Inspect, debug, and manage .apr model files.
/// Toyota Way: Genchi Genbutsu - Go and see the actual data.
#[derive(Parser)]
#[command(name = "apr")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Output as JSON
    #[arg(long, global = true)]
    json: bool,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Quiet mode (errors only)
    #[arg(short, long, global = true)]
    quiet: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect model metadata, vocab, and structure
    Inspect {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show vocabulary details
        #[arg(long)]
        vocab: bool,

        /// Show filter/security details
        #[arg(long)]
        filters: bool,

        /// Show weight statistics
        #[arg(long)]
        weights: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Simple debugging output ("drama" mode available)
    Debug {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Theatrical "drama" mode output
        #[arg(long)]
        drama: bool,

        /// Show hex dump
        #[arg(long)]
        hex: bool,

        /// Extract ASCII strings
        #[arg(long)]
        strings: bool,

        /// Limit output lines
        #[arg(long, default_value = "256")]
        limit: usize,
    },

    /// Validate model integrity and quality
    Validate {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show 100-point quality assessment
        #[arg(long)]
        quality: bool,

        /// Strict validation (fail on warnings)
        #[arg(long)]
        strict: bool,

        /// Minimum score to pass (0-100)
        #[arg(long)]
        min_score: Option<u8>,
    },

    /// Compare two models
    Diff {
        /// First model file
        #[arg(value_name = "FILE1")]
        file1: PathBuf,

        /// Second model file
        #[arg(value_name = "FILE2")]
        file2: PathBuf,

        /// Show weight-level differences
        #[arg(long)]
        weights: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List tensor names and shapes
    Tensors {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Show tensor statistics (mean, std, min, max)
        #[arg(long)]
        stats: bool,

        /// Filter tensors by name pattern
        #[arg(long)]
        filter: Option<String>,

        /// Limit number of tensors shown
        #[arg(long, default_value = "100")]
        limit: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Layer-by-layer trace analysis
    Trace {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Filter layers by name pattern
        #[arg(long)]
        layer: Option<String>,

        /// Compare with reference model
        #[arg(long)]
        reference: Option<PathBuf>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Verbose output with per-layer stats
        #[arg(short, long)]
        verbose: bool,
    },

    /// Export for probar visual testing
    Probar {
        /// Path to .apr model file
        #[arg(value_name = "FILE")]
        file: PathBuf,

        /// Output directory for test artifacts
        #[arg(short, long, default_value = "./probar-export")]
        output: PathBuf,

        /// Export format: json, png, or both
        #[arg(long, default_value = "both")]
        format: String,

        /// Golden reference directory for comparison
        #[arg(long)]
        golden: Option<PathBuf>,

        /// Filter layers by name pattern
        #[arg(long)]
        layer: Option<String>,
    },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    let result = match cli.command {
        Commands::Inspect {
            file,
            vocab,
            filters,
            weights,
            json,
        } => inspect::run(&file, vocab, filters, weights, json || cli.json),

        Commands::Debug {
            file,
            drama,
            hex,
            strings,
            limit,
        } => debug::run(&file, drama, hex, strings, limit),

        Commands::Validate {
            file,
            quality,
            strict,
            min_score,
        } => validate::run(&file, quality, strict, min_score),

        Commands::Diff {
            file1,
            file2,
            weights,
            json,
        } => diff::run(&file1, &file2, weights, json || cli.json),

        Commands::Tensors {
            file,
            stats,
            filter,
            limit,
            json,
        } => tensors::run(&file, stats, filter.as_deref(), json || cli.json, limit),

        Commands::Trace {
            file,
            layer,
            reference,
            json,
            verbose,
        } => trace::run(
            &file,
            layer.as_deref(),
            reference.as_deref(),
            json || cli.json,
            verbose || cli.verbose,
        ),

        Commands::Probar {
            file,
            output,
            format,
            golden,
            layer,
        } => {
            let export_format = format.parse().unwrap_or(probar::ExportFormat::Both);
            probar::run(
                &file,
                &output,
                export_format,
                golden.as_deref(),
                layer.as_deref(),
            )
        }
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            e.exit_code()
        }
    }
}
