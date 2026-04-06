//! Remove zram device command.
//!
//! This is a pure shim that delegates to `trueno_zram_core::zram`.

use clap::Args;
use trueno_zram_core::zram::{SysfsOps, ZramOps};

/// Arguments for removing a zram device.
#[derive(Args)]
pub struct RemoveArgs {
    /// Device number to remove.
    #[arg(short, long, default_value = "0")]
    pub device: u32,

    /// Force removal even if in use.
    #[arg(short, long)]
    pub force: bool,
}

/// Remove a zram device.
pub fn remove(args: &RemoveArgs) -> Result<(), Box<dyn std::error::Error>> {
    let ops = SysfsOps::new();
    ops.remove(args.device, args.force)?;

    println!("Removed zram{}", args.device);
    Ok(())
}
