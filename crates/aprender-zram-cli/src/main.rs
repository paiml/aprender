//! trueno-zram binary. The command surface lives in the library
//! (`aprender_zram_cli`) so `apr zram` can reach the same code; see the module
//! docs there.

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]

use std::process::ExitCode;

/// Printed before anything runs: this binary is one of the duplicates `apr`
/// absorbed (#4060, EPIC #4057). It still works, through the same library code
/// `apr zram` calls, so scripts that call it keep running until its consumers
/// have moved; then the bin target is removed.
const DEPRECATED: &str = "warning: `trueno-zram` is deprecated and will be removed; run `apr zram` instead (the same code path). See paiml/aprender#4057.";

fn main() -> ExitCode {
    eprintln!("{DEPRECATED}");
    aprender_zram_cli::run()
}
