//! apr - APR Model Operations CLI
//!
//! Entry point shim. See lib.rs for implementation.

use apr_cli::{execute_command, Cli};
use clap::Parser;
use colored::control;
use std::process::ExitCode;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> ExitCode {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();
    // Force colored output (matches pmat behavior) — users can disable with NO_COLOR=1
    control::set_override(true);
    let cli = Cli::parse();
    match execute_command(&cli) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            e.exit_code()
        }
    }
}
