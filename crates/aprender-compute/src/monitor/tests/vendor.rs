//! H0-MON-01 through H0-MON-06: GpuVendor identification tests

use crate::monitor::*;

// =========================================================================
// H0-MON-01: GpuVendor identification
// =========================================================================

#[test]
fn h0_mon_01_vendor_nvidia_id() {
    let vendor = GpuVendor::from_vendor_id(0x10de);
    assert_eq!(vendor, GpuVendor::Nvidia);
    assert!(vendor.is_nvidia());
    assert_eq!(vendor.name(), "NVIDIA");
}

#[test]
fn h0_mon_02_vendor_amd_id() {
    let vendor = GpuVendor::from_vendor_id(0x1002);
    assert_eq!(vendor, GpuVendor::Amd);
    assert!(!vendor.is_nvidia());
    assert_eq!(vendor.name(), "AMD");
}

#[test]
fn h0_mon_03_vendor_intel_id() {
    let vendor = GpuVendor::from_vendor_id(0x8086);
    assert_eq!(vendor, GpuVendor::Intel);
    assert!(!vendor.is_nvidia());
    assert_eq!(vendor.name(), "Intel");
}

#[test]
fn h0_mon_04_vendor_apple_id() {
    let vendor = GpuVendor::from_vendor_id(0x106b);
    assert_eq!(vendor, GpuVendor::Apple);
    assert!(!vendor.is_nvidia());
    assert_eq!(vendor.name(), "Apple");
}

#[test]
fn h0_mon_05_vendor_unknown_id() {
    let vendor = GpuVendor::from_vendor_id(0x9999);
    assert_eq!(vendor, GpuVendor::Unknown(0x9999));
    assert!(!vendor.is_nvidia());
    assert_eq!(vendor.name(), "Unknown");
}

#[test]
fn h0_mon_06_vendor_display() {
    assert_eq!(format!("{}", GpuVendor::Nvidia), "NVIDIA");
    assert_eq!(format!("{}", GpuVendor::Amd), "AMD");
    assert_eq!(format!("{}", GpuVendor::Unknown(0x1234)), "Unknown (0x1234)");
}
