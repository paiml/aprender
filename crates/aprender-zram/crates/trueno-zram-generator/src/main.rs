//! systemd generator for zram device configuration.
//!
//! This generator runs during early boot to create systemd units
//! for zram device setup based on system configuration.

#![deny(missing_docs)]
#![deny(clippy::panic)]
#![warn(clippy::all, clippy::pedantic)]

mod config;
mod fstab;
mod unit;

use std::process::ExitCode;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("trueno-zram-generator: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();

    // systemd generators receive: normal_dir early_dir late_dir
    if args.len() < 2 {
        return Err("Usage: trueno-zram-generator <normal_dir> [early_dir] [late_dir]".into());
    }

    let normal_dir = &args[1];
    let _early_dir = args.get(2);
    let _late_dir = args.get(3);

    // Load configuration
    let config = config::load_config()?;

    // Generate systemd units
    unit::generate_units(normal_dir, &config)?;

    Ok(())
}
