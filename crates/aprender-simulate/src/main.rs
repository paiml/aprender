//! simular CLI - Unified Simulation Engine
//!
//! Minimal entry point. All logic is in the `cli` module.

use clap::Parser;
use simular::cli::{run_cli, Cli};
use std::process::ExitCode;

fn main() -> ExitCode {
    #[cfg(not(target_arch = "wasm32"))]
    sovereign_update::hook!("simular"); // EPIC #4232: `simular update`, and the startup notice
    run_cli(Cli::parse())
}
