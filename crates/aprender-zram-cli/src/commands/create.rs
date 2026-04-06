//! Create zram device command.
//!
//! This is a pure shim that delegates to `trueno_zram_core::zram`.

use clap::Args;
use trueno_zram_core::zram::{format_size, parse_size, SysfsOps, ZramConfig, ZramOps};

/// Arguments for creating a zram device.
#[derive(Args)]
pub struct CreateArgs {
    /// Device number (0-16).
    #[arg(short, long, default_value = "0", value_parser = clap::value_parser!(u32).range(0..=16))]
    pub device: u32,

    /// Device size (e.g., "4G", "512M", "ram/2").
    #[arg(short, long)]
    pub size: String,

    /// Compression algorithm (lz4, zstd).
    #[arg(short, long, default_value = "lz4")]
    pub algorithm: String,

    /// Number of compression streams (0 = auto).
    #[arg(long, default_value = "0")]
    pub streams: u32,
}

/// Create and configure a zram device.
pub fn create(args: &CreateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let size_bytes = parse_size(&args.size)?;

    let config = ZramConfig {
        device: args.device,
        size: size_bytes,
        algorithm: args.algorithm.clone(),
        streams: args.streams,
    };

    let ops = SysfsOps::new();
    ops.create(&config)?;

    println!(
        "Created zram{} with size {} using {}",
        args.device,
        format_size(size_bytes),
        args.algorithm
    );

    Ok(())
}
