//! aprender-ptx-debug CLI
//!
//! Pure Rust PTX debugging and static analysis tool.
//!
//! Usage:
//!   aprender-ptx-debug analyze <file.ptx> [--falsify] [--min-score N]
//!   aprender-ptx-debug gen-fkr <file.ptx> [-o tests.rs]
//!
//! Argument parsing is declarative and lives in `trueno_ptx_debug::cli`.

use std::process;

// Imported anonymously: `clap::Parser` would otherwise collide with the PTX
// `Parser` used by the library.
use clap::Parser as _;

use trueno_ptx_debug::cli::{exit_code_for_parse_error, Cli};

/// #4062: `apr ptx-debug` runs the same code; this binary is on its way out.
const DEPRECATED: &str = "warning: `aprender-ptx-debug` is deprecated and will be removed; run `apr ptx-debug` instead (the same code path). See paiml/aprender#4057.";

fn main() {
    eprintln!("{DEPRECATED}");
    let cli = match Cli::try_parse() {
        Ok(cli) => cli,
        Err(err) => {
            // clap picks stdout for --help/--version and stderr for real
            // failures; the exit code is chosen the same way.
            let _ = err.print();
            process::exit(exit_code_for_parse_error(&err));
        }
    };

    match trueno_ptx_debug::run::run(cli.command) {
        Ok(code) => process::exit(code),
        Err(e) => {
            eprintln!("Error: {}", e);
            process::exit(1);
        }
    }
}
