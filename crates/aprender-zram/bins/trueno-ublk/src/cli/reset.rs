//! Reset command - removes trueno-ublk devices

use super::ResetArgs;
use crate::device::UblkDevice;
use anyhow::Result;

pub fn run(args: ResetArgs) -> Result<()> {
    let devices = if args.all {
        UblkDevice::list_all()?
    } else if args.devices.is_empty() {
        anyhow::bail!("No devices specified. Use --all to reset all devices.");
    } else {
        args.devices.iter().map(|p| UblkDevice::open(p)).collect::<Result<Vec<_>>>()?
    };

    for device in devices {
        tracing::info!(device = %device.name(), "Removing device");
        device.remove()?;
    }

    Ok(())
}
