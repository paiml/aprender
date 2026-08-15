//! trueno-zram binary. The command surface lives in the library
//! (`aprender_zram_cli`) so `apr zram` can reach the same code; see the module
//! docs there.

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]

use std::process::ExitCode;

fn main() -> ExitCode {
    aprender_zram_cli::run()
}
