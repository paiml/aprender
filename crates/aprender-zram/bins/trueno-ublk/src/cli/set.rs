//! Set command - live reconfiguration of device parameters

use super::{parse_size, SetArgs};
use crate::device::UblkDevice;
use anyhow::Result;

pub fn run(args: SetArgs) -> Result<()> {
    let device = UblkDevice::open(&args.device)?;

    if let Some(ref limit) = args.mem_limit {
        let bytes = parse_size(limit)?;
        tracing::info!(device = %device.name(), limit = bytes, "Setting memory limit");
        device.set_mem_limit(bytes)?;
    }

    if let Some(gpu) = args.gpu {
        tracing::info!(device = %device.name(), gpu = gpu, "Setting GPU mode");
        device.set_gpu_enabled(gpu)?;
    }

    if let Some(threshold) = args.entropy_skip {
        tracing::info!(
            device = %device.name(),
            threshold = threshold,
            "Setting entropy skip threshold"
        );
        device.set_entropy_threshold(threshold)?;
    }

    if let Some(ref limit) = args.writeback_limit {
        let bytes = parse_size(limit)?;
        tracing::info!(device = %device.name(), limit = bytes, "Setting writeback limit");
        device.set_writeback_limit(bytes)?;
    }

    if let Some(enable) = args.writeback_limit_enable {
        tracing::info!(
            device = %device.name(),
            enable = enable,
            "Setting writeback limit enabled"
        );
        device.set_writeback_limit_enabled(enable)?;
    }

    if args.reset_mem_used_max {
        tracing::info!(device = %device.name(), "Resetting mem_used_max watermark");
        device.reset_mem_used_max()?;
    }

    Ok(())
}
