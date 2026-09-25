//! simular CLI - Unified Simulation Engine
//!
//! Minimal entry point. All logic is in the `cli` module.

use clap::Parser;
use simular::cli::{run_cli, Cli};
use std::process::ExitCode;

/// Printed before anything runs: this binary is one of the duplicates `apr`
/// absorbed (#4060, EPIC #4057). It still works, through the same library code
/// `apr sim` calls, so scripts that call it keep running until its consumers
/// have moved; then the bin target is removed.
const DEPRECATED: &str = "warning: `simular` is deprecated and will be removed; run `apr sim` instead (the same code path). See paiml/aprender#4057.";

fn main() -> ExitCode {
    eprintln!("{DEPRECATED}");
    run_cli(Cli::parse())
}
