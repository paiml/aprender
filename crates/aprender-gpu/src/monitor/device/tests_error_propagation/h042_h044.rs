//! H042-H044: PartialErrorMockDevice, DeviceSnapshot timestamp, CpuDevice debug

use super::*;

// =========================================================================
// H042: Partial Error Mock for Second-Call Error Propagation
// =========================================================================

#[test]
fn h042_partial_error_memory_usage_percent_total_error() {
    let mock = PartialErrorMockDevice::with_total_error();
    // memory_used_bytes succeeds, but memory_total_bytes fails
    // Should propagate the error from the second call
    let result = mock.memory_usage_percent();
    assert!(result.is_err());
}

#[test]
fn h042_partial_error_memory_available_bytes_total_error() {
    let mock = PartialErrorMockDevice::with_total_error();
    // memory_used_bytes succeeds, but memory_total_bytes fails
    let result = mock.memory_available_bytes();
    assert!(result.is_err());
}

#[test]
fn h042_partial_error_power_usage_percent_limit_error() {
    let mock = PartialErrorMockDevice::with_limit_error();
    // compute_power_watts succeeds, but compute_power_limit_watts fails
    let result = mock.power_usage_percent();
    assert!(result.is_err());
}

#[test]
fn h042_partial_error_device_name() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert_eq!(mock.device_name(), "PartialErrorMock");
}

#[test]
fn h042_partial_error_device_type() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert_eq!(mock.device_type(), DeviceType::Cpu);
}

#[test]
fn h042_partial_error_device_id() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert_eq!(mock.device_id(), DeviceId::cpu());
}

#[test]
fn h042_partial_error_compute_utilization() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert!((mock.compute_utilization().expect("test") - 50.0).abs() < 0.01);
}

#[test]
fn h042_partial_error_compute_clock() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert_eq!(mock.compute_clock_mhz().expect("test"), 3000);
}

#[test]
fn h042_partial_error_compute_temperature() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert!((mock.compute_temperature_c().expect("test") - 50.0).abs() < 0.01);
}

#[test]
fn h042_partial_error_compute_power() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert!((mock.compute_power_watts().expect("test") - 100.0).abs() < 0.01);
}

#[test]
fn h042_partial_error_memory_used() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert_eq!(mock.memory_used_bytes().expect("test"), 1024);
}

#[test]
fn h042_partial_error_compute_units() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert_eq!(mock.compute_unit_count(), 8);
    assert_eq!(mock.active_compute_units().expect("test"), 8);
}

#[test]
fn h042_partial_error_pcie_metrics() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert_eq!(mock.pcie_generation(), 0);
    assert_eq!(mock.pcie_width(), 0);
    assert!(mock.pcie_tx_bytes_per_sec().is_err());
    assert!(mock.pcie_rx_bytes_per_sec().is_err());
}

#[test]
fn h042_partial_error_memory_bandwidth() {
    let mock = PartialErrorMockDevice::with_total_error();
    assert!(mock.memory_bandwidth_gbps().is_err());
}

#[test]
fn h042_partial_error_refresh() {
    let mut mock = PartialErrorMockDevice::with_total_error();
    assert!(mock.refresh().is_ok());
}

// =========================================================================
// H043: DeviceSnapshot Timestamp Coverage
// =========================================================================

#[test]
fn h043_device_snapshot_timestamp_non_zero() {
    let mock = MockDevice::new(1024, 2048, 100.0, 200.0, 50.0);
    let snapshot = DeviceSnapshot::capture(&mock).expect("test");

    // Timestamp should be non-zero (based on system time)
    assert!(snapshot.timestamp_ms > 0);
}

#[test]
fn h043_device_snapshot_all_fields_populated() {
    let mock = MockDevice::new(1024, 2048, 100.0, 200.0, 50.0);
    let snapshot = DeviceSnapshot::capture(&mock).expect("test");

    // Verify all fields are populated from the mock
    assert_eq!(snapshot.device_id, DeviceId::cpu());
    assert!((snapshot.compute_utilization - 50.0).abs() < 0.01);
    assert_eq!(snapshot.memory_used_bytes, 1024);
    assert_eq!(snapshot.memory_total_bytes, 2048);
    assert!((snapshot.temperature_c - 50.0).abs() < 0.01);
    assert!((snapshot.power_watts - 100.0).abs() < 0.01);
    assert_eq!(snapshot.clock_mhz, 3000);
}

// =========================================================================
// H044: CpuDevice Debug Trait Coverage
// =========================================================================

#[test]
fn h044_cpu_device_debug() {
    let cpu = CpuDevice::new();
    let debug_str = format!("{:?}", cpu);
    assert!(debug_str.contains("CpuDevice"));
}
