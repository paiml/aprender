//! H036-H041: CpuDevice edge cases, boundary tests, MockDevice trait methods,
//! ErrorMockDevice additional coverage

use super::*;

// =========================================================================
// H036: CpuDevice Partial Coverage via Edge Cases
// =========================================================================

#[test]
fn h036_cpu_device_memory_used_realistic() {
    let mut cpu = CpuDevice::new();
    cpu.refresh().expect("test");

    // memory_used should be between 0 and total
    let used = cpu.memory_used_bytes().expect("test");
    let total = cpu.memory_total_bytes().expect("test");
    assert!(used <= total);
}

// =========================================================================
// H037: Boundary Tests for Default Implementations
// =========================================================================

#[test]
fn h037_memory_usage_percent_boundary_values() {
    // Test 0% usage
    let mock_empty = MockDevice::new(0, 1000, 0.0, 0.0, 0.0);
    assert!((mock_empty.memory_usage_percent().expect("test") - 0.0).abs() < 0.01);

    // Test 100% usage
    let mock_full = MockDevice::new(1000, 1000, 0.0, 0.0, 0.0);
    assert!((mock_full.memory_usage_percent().expect("test") - 100.0).abs() < 0.01);
}

#[test]
fn h037_memory_available_boundary() {
    // All memory available
    let mock_empty = MockDevice::new(0, 1000, 0.0, 0.0, 0.0);
    assert_eq!(mock_empty.memory_available_bytes().expect("test"), 1000);

    // No memory available
    let mock_full = MockDevice::new(1000, 1000, 0.0, 0.0, 0.0);
    assert_eq!(mock_full.memory_available_bytes().expect("test"), 0);
}

#[test]
fn h037_power_usage_percent_boundary() {
    // 0% power
    let mock_idle = MockDevice::new(0, 0, 0.0, 100.0, 0.0);
    assert!((mock_idle.power_usage_percent().expect("test") - 0.0).abs() < 0.01);

    // 100% power
    let mock_max = MockDevice::new(0, 0, 100.0, 100.0, 0.0);
    assert!((mock_max.power_usage_percent().expect("test") - 100.0).abs() < 0.01);
}

// =========================================================================
// H038: DeviceSnapshot Memory Percent Edge Cases
// =========================================================================

#[test]
fn h038_snapshot_memory_percent_edge_cases() {
    // Test with various memory ratios
    let snap_50 = DeviceSnapshot {
        device_id: DeviceId::cpu(),
        timestamp_ms: 12345,
        compute_utilization: 25.0,
        memory_used_bytes: 500,
        memory_total_bytes: 1000,
        temperature_c: 60.0,
        power_watts: 75.0,
        clock_mhz: 2500,
    };
    assert!((snap_50.memory_usage_percent() - 50.0).abs() < 0.01);

    // Test 0%
    let snap_0 = DeviceSnapshot {
        device_id: DeviceId::cpu(),
        timestamp_ms: 0,
        compute_utilization: 0.0,
        memory_used_bytes: 0,
        memory_total_bytes: 1000,
        temperature_c: 0.0,
        power_watts: 0.0,
        clock_mhz: 0,
    };
    assert!((snap_0.memory_usage_percent() - 0.0).abs() < 0.01);

    // Test 100%
    let snap_100 = DeviceSnapshot {
        device_id: DeviceId::cpu(),
        timestamp_ms: 0,
        compute_utilization: 0.0,
        memory_used_bytes: 1000,
        memory_total_bytes: 1000,
        temperature_c: 0.0,
        power_watts: 0.0,
        clock_mhz: 0,
    };
    assert!((snap_100.memory_usage_percent() - 100.0).abs() < 0.01);
}

// =========================================================================
// H039: CpuDevice Method Coverage via Direct Tests
// =========================================================================

#[test]
fn h039_cpu_device_device_id_and_type() {
    let cpu = CpuDevice::new();
    assert_eq!(cpu.device_id(), DeviceId::cpu());
    assert_eq!(cpu.device_type(), DeviceType::Cpu);
}

#[test]
fn h039_cpu_utilization_initial() {
    let cpu = CpuDevice::new();
    // Initially cpu_usage is 0.0 before refresh
    let util = cpu.compute_utilization().expect("test");
    assert!(util >= 0.0 && util <= 100.0);
}

#[test]
fn h039_cpu_memory_used_initial() {
    let cpu = CpuDevice::new();
    // Initially memory_used is 0 before refresh
    let used = cpu.memory_used_bytes().expect("test");
    assert!(used >= 0);
}

#[test]
fn h039_cpu_active_units_equals_total() {
    let cpu = CpuDevice::new();
    let total = cpu.compute_unit_count();
    let active = cpu.active_compute_units().expect("test");
    assert_eq!(total, active);
}

// =========================================================================
// H040: Additional MockDevice Trait Methods
// =========================================================================

#[test]
fn h040_mock_device_power_limit_access() {
    let mock = MockDevice::new(0, 0, 50.0, 100.0, 0.0);
    let limit = mock.compute_power_limit_watts().expect("test");
    assert!((limit - 100.0).abs() < 0.01);
}

#[test]
fn h040_mock_device_memory_total_access() {
    let mock = MockDevice::new(500, 1000, 0.0, 0.0, 0.0);
    let total = mock.memory_total_bytes().expect("test");
    assert_eq!(total, 1000);
}

// =========================================================================
// H041: ErrorMockDevice Additional Coverage
// =========================================================================

#[test]
fn h041_error_mock_device_name() {
    let mock = ErrorMockDevice::new(false);
    assert_eq!(mock.device_name(), "ErrorMock");
}

#[test]
fn h041_error_mock_device_type() {
    let mock = ErrorMockDevice::new(false);
    assert_eq!(mock.device_type(), DeviceType::Cpu);
}

#[test]
fn h041_error_mock_compute_units() {
    let mock = ErrorMockDevice::new(false);
    assert_eq!(mock.compute_unit_count(), 8);
    assert_eq!(mock.active_compute_units().expect("test"), 8);
}

#[test]
fn h041_error_mock_pcie_metrics() {
    let mock = ErrorMockDevice::new(false);
    assert_eq!(mock.pcie_generation(), 0);
    assert_eq!(mock.pcie_width(), 0);
    assert!(mock.pcie_tx_bytes_per_sec().is_err());
    assert!(mock.pcie_rx_bytes_per_sec().is_err());
}

#[test]
fn h041_error_mock_refresh() {
    let mut mock = ErrorMockDevice::new(false);
    assert!(mock.refresh().is_ok());
}

#[test]
fn h041_error_mock_memory_bandwidth() {
    let mock = ErrorMockDevice::new(false);
    assert!(mock.memory_bandwidth_gbps().is_err());
}
