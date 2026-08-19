//! simular CLI - Unified Simulation Engine
//!
//! Minimal entry point. All logic is in the `cli` module.

use clap::Parser;
use simular::cli::{run_cli, Args};
use std::process::ExitCode;

fn main() -> ExitCode {
    run_cli(Args::parse())
}
