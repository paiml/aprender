//! Writeback command - triggers writeback to backing device

use super::WritebackArgs;
use crate::device::UblkDevice;
use anyhow::Result;

pub fn run(args: WritebackArgs) -> Result<()> {
    let device = UblkDevice::open(&args.device)?;

    let mode = if args.all {
        "all"
    } else if args.idle {
        "idle"
    } else if args.huge {
        "huge"
    } else {
        "idle" // default
    };

    tracing::info!(device = %device.name(), mode = mode, "Triggering writeback");
    device.writeback(args.idle, args.huge, args.all)?;

    Ok(())
}
