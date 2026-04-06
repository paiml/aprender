//! Idle command - marks pages as idle for writeback tracking

use super::DeviceArg;
use crate::device::UblkDevice;
use anyhow::Result;

pub fn run(args: DeviceArg) -> Result<()> {
    let device = UblkDevice::open(&args.device)?;
    tracing::info!(device = %device.name(), "Marking pages as idle");
    device.mark_idle()?;
    Ok(())
}
