//! H031-H035: ErrorMockDevice error propagation, ThrottleReason, DeviceType, DeviceId coverage

use super::*;

// =========================================================================
// H031: Error-Propagating Mock Device
// =========================================================================

#[test]
fn h031_error_mock_memory_usage_percent_error() {
    let mock = ErrorMockDevice::new(true);
    // Should propagate the error from memory_used_bytes
    let result = mock.memory_usage_percent();
    assert!(result.is_err());
}

#[test]
fn h031_error_mock_memory_available_bytes_error() {
    let mock = ErrorMockDevice::new(true);
    // Should propagate the error from memory_used_bytes
    let result = mock.memory_available_bytes();
    assert!(result.is_err());
}

#[test]
fn h031_error_mock_memory_used_mb_error() {
    let mock = ErrorMockDevice::new(true);
    let result = mock.memory_used_mb();
    assert!(result.is_err());
}

#[test]
fn h031_error_mock_memory_total_mb_error() {
    let mock = ErrorMockDevice::new(true);
    let result = mock.memory_total_mb();
    assert!(result.is_err());
}

#[test]
fn h031_error_mock_memory_total_gb_error() {
    let mock = ErrorMockDevice::new(true);
    let result = mock.memory_total_gb();
    assert!(result.is_err());
}

#[test]
fn h031_error_mock_power_usage_percent_error() {
    let mock = ErrorMockDevice::new(true);
    // Should propagate the error from compute_power_watts
    let result = mock.power_usage_percent();
    assert!(result.is_err());
}

#[test]
fn h031_error_mock_is_thermal_throttling_error() {
    let mock = ErrorMockDevice::new(true);
    // Should propagate the error from compute_temperature_c
    let result = mock.is_thermal_throttling();
    assert!(result.is_err());
}

#[test]
fn h031_error_mock_is_power_throttling_error() {
    let mock = ErrorMockDevice::new(true);
    // Should propagate the error from power_usage_percent -> compute_power_watts
    let result = mock.is_power_throttling();
    assert!(result.is_err());
}

#[test]
fn h031_error_mock_working_correctly() {
    let mock = ErrorMockDevice::new(false);
    // When not returning errors, should work
    assert!(mock.memory_usage_percent().is_ok());
    assert!(mock.memory_available_bytes().is_ok());
    assert!(mock.memory_used_mb().is_ok());
    assert!(mock.memory_total_mb().is_ok());
    assert!(mock.memory_total_gb().is_ok());
    assert!(mock.power_usage_percent().is_ok());
    assert!(mock.is_thermal_throttling().is_ok());
    assert!(mock.is_power_throttling().is_ok());
}

// =========================================================================
// H032: DeviceSnapshot with Errors
// =========================================================================

#[test]
fn h032_device_snapshot_with_errors() {
    let mock = ErrorMockDevice::new(true);
    // DeviceSnapshot::capture uses unwrap_or defaults
    let snapshot = DeviceSnapshot::capture(&mock);
    assert!(snapshot.is_ok());

    let snap = snapshot.expect("test");
    // Should use defaults when metrics fail
    assert_eq!(snap.compute_utilization, 0.0);
    assert_eq!(snap.memory_used_bytes, 0);
    assert_eq!(snap.memory_total_bytes, 0);
    assert_eq!(snap.temperature_c, 0.0);
    assert_eq!(snap.power_watts, 0.0);
    assert_eq!(snap.clock_mhz, 0);
}

// =========================================================================
// H033: ThrottleReason Complete Coverage
// =========================================================================

#[test]
fn h033_throttle_reason_clone() {
    let reason = ThrottleReason::Power;
    let cloned = reason.clone();
    assert_eq!(reason, cloned);
}

#[test]
fn h033_throttle_reason_copy() {
    let reason = ThrottleReason::Thermal;
    let copied: ThrottleReason = reason; // Copy
    assert_eq!(reason, copied);
}

#[test]
fn h033_throttle_reason_equality() {
    assert_eq!(ThrottleReason::None, ThrottleReason::None);
    assert_eq!(ThrottleReason::Thermal, ThrottleReason::Thermal);
    assert_ne!(ThrottleReason::None, ThrottleReason::Thermal);
    assert_ne!(ThrottleReason::Power, ThrottleReason::Thermal);
}

#[test]
fn h033_throttle_reason_debug() {
    let reason = ThrottleReason::HwSlowdown;
    let debug_str = format!("{:?}", reason);
    assert!(debug_str.contains("HwSlowdown"));
}

// =========================================================================
// H034: DeviceType Complete Coverage
// =========================================================================

#[test]
fn h034_device_type_clone() {
    let dt = DeviceType::NvidiaGpu;
    let cloned = dt.clone();
    assert_eq!(dt, cloned);
}

#[test]
fn h034_device_type_copy() {
    let dt = DeviceType::AmdGpu;
    let copied: DeviceType = dt; // Copy
    assert_eq!(dt, copied);
}

#[test]
fn h034_device_type_equality() {
    assert_eq!(DeviceType::Cpu, DeviceType::Cpu);
    assert_ne!(DeviceType::Cpu, DeviceType::NvidiaGpu);
    assert_ne!(DeviceType::NvidiaGpu, DeviceType::AmdGpu);
    assert_ne!(DeviceType::AmdGpu, DeviceType::IntelGpu);
    assert_ne!(DeviceType::IntelGpu, DeviceType::AppleSilicon);
}

#[test]
fn h034_device_type_hash() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(DeviceType::Cpu);
    set.insert(DeviceType::NvidiaGpu);
    set.insert(DeviceType::AmdGpu);
    set.insert(DeviceType::IntelGpu);
    set.insert(DeviceType::AppleSilicon);

    assert_eq!(set.len(), 5);

    // Duplicate should not increase size
    set.insert(DeviceType::Cpu);
    assert_eq!(set.len(), 5);
}

// =========================================================================
// H035: DeviceId Complete Coverage
// =========================================================================

#[test]
fn h035_device_id_copy() {
    let id = DeviceId::nvidia(0);
    let copied: DeviceId = id; // Copy
    assert_eq!(id, copied);
}

#[test]
fn h035_device_id_new_with_all_types() {
    // Test DeviceId::new with all DeviceTypes
    let cpu = DeviceId::new(DeviceType::Cpu, 0);
    let nvidia = DeviceId::new(DeviceType::NvidiaGpu, 1);
    let amd = DeviceId::new(DeviceType::AmdGpu, 2);
    let intel = DeviceId::new(DeviceType::IntelGpu, 3);
    let apple = DeviceId::new(DeviceType::AppleSilicon, 4);

    assert_eq!(cpu.device_type, DeviceType::Cpu);
    assert_eq!(cpu.index, 0);
    assert_eq!(nvidia.device_type, DeviceType::NvidiaGpu);
    assert_eq!(nvidia.index, 1);
    assert_eq!(amd.device_type, DeviceType::AmdGpu);
    assert_eq!(amd.index, 2);
    assert_eq!(intel.device_type, DeviceType::IntelGpu);
    assert_eq!(intel.index, 3);
    assert_eq!(apple.device_type, DeviceType::AppleSilicon);
    assert_eq!(apple.index, 4);
}
