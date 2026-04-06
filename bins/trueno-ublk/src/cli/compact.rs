//! Compact command - triggers memory compaction

use super::DeviceArg;
use crate::device::UblkDevice;
use anyhow::Result;

pub fn run(args: DeviceArg) -> Result<()> {
    let device = UblkDevice::open(&args.device)?;
    tracing::info!(device = %device.name(), "Triggering compaction");
    device.compact()?;
    Ok(())
}
