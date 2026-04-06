//! H0-MON-20 through H0-MON-24: GpuDeviceInfo construction tests

use crate::monitor::*;

// =========================================================================
// H0-MON-20: GpuDeviceInfo construction
// =========================================================================

#[test]
fn h0_mon_20_device_info_basic() {
    let info = GpuDeviceInfo::new(0, "Test GPU", GpuVendor::Nvidia, GpuBackend::Vulkan);

    assert_eq!(info.index, 0);
    assert_eq!(info.name, "Test GPU");
    assert_eq!(info.vendor, GpuVendor::Nvidia);
    assert_eq!(info.backend, GpuBackend::Vulkan);
    assert_eq!(info.vram_total, 0);
    assert!(info.compute_capability.is_none());
}

#[test]
fn h0_mon_21_device_info_builder() {
    let info = GpuDeviceInfo::new(0, "RTX 4090", GpuVendor::Nvidia, GpuBackend::Vulkan)
        .with_vram(24_000_000_000)
        .with_compute_capability(8, 9)
        .with_driver_version("535.154.05")
        .with_pci_bus_id("0000:01:00.0");

    assert_eq!(info.vram_total, 24_000_000_000);
    assert_eq!(info.compute_capability, Some((8, 9)));
    assert_eq!(info.driver_version, Some("535.154.05".to_string()));
    assert_eq!(info.pci_bus_id, Some("0000:01:00.0".to_string()));
}

#[test]
fn h0_mon_22_device_info_vram_helpers() {
    let info = GpuDeviceInfo::new(0, "Test", GpuVendor::Nvidia, GpuBackend::Vulkan)
        .with_vram(24 * 1024 * 1024 * 1024); // 24 GB

    assert_eq!(info.vram_mb(), 24 * 1024);
    assert!((info.vram_gb() - 24.0).abs() < 0.01);
}

#[test]
fn h0_mon_23_device_info_supports_cuda() {
    let nvidia = GpuDeviceInfo::new(0, "RTX", GpuVendor::Nvidia, GpuBackend::Vulkan);
    let amd = GpuDeviceInfo::new(0, "RX", GpuVendor::Amd, GpuBackend::Vulkan);

    assert!(nvidia.supports_cuda());
    assert!(!amd.supports_cuda());
}

#[test]
fn h0_mon_24_device_info_display() {
    let info = GpuDeviceInfo::new(0, "RTX 4090", GpuVendor::Nvidia, GpuBackend::Vulkan)
        .with_vram(24 * 1024 * 1024 * 1024);

    let display = format!("{info}");
    assert!(display.contains("RTX 4090"));
    assert!(display.contains("Vulkan"));
    assert!(display.contains("24.0"));
}
