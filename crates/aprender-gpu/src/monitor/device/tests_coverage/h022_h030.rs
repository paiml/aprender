//! Edge case tests, snapshot field coverage, CpuDevice internal methods,
//! read functions, clock, temperature, CPU usage, memory, and MockDevice
//! additional coverage (H022-H030).

use super::*;

// =========================================================================
// H022: Edge Case Tests for Default Trait Implementations
// =========================================================================

#[test]
fn h022_mock_device_thermal_throttling_at_threshold() {
    // Test exactly at the 80 degree threshold
    let mock_at_80 = MockDevice::new(0, 0, 0.0, 0.0, 80.0);
    // 80.0 is not > 80.0, so no throttling
    assert!(!mock_at_80.is_thermal_throttling().expect("test"));

    // Just above threshold
    let mock_at_80_1 = MockDevice::new(0, 0, 0.0, 0.0, 80.1);
    assert!(mock_at_80_1.is_thermal_throttling().expect("test"));
}

#[test]
fn h022_mock_device_power_throttling_at_threshold() {
    // Test exactly at the 95% threshold
    let mock_at_95 = MockDevice::new(0, 0, 95.0, 100.0, 0.0);
    // 95.0 is not > 95.0, so no throttling
    assert!(!mock_at_95.is_power_throttling().expect("test"));

    // Just above threshold
    let mock_at_95_1 = MockDevice::new(0, 0, 95.1, 100.0, 0.0);
    assert!(mock_at_95_1.is_power_throttling().expect("test"));
}

#[test]
fn h022_mock_device_memory_usage_full() {
    // Test 100% memory usage
    let mock_full = MockDevice::new(100, 100, 0.0, 0.0, 0.0);
    assert!((mock_full.memory_usage_percent().expect("test") - 100.0).abs() < 0.01);
    assert_eq!(mock_full.memory_available_bytes().expect("test"), 0);
}

// =========================================================================
// H023: DeviceSnapshot Additional Field Coverage
// =========================================================================

#[test]
fn h023_device_snapshot_field_access() {
    let mock = MockDevice::new(8 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024, 250.0, 350.0, 72.0);
    let snapshot = DeviceSnapshot::capture(&mock).expect("test");

    // Verify all fields are accessible and have expected values
    assert_eq!(snapshot.memory_used_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(snapshot.memory_total_bytes, 32 * 1024 * 1024 * 1024);
    assert!((snapshot.temperature_c - 72.0).abs() < 0.01);
    assert!((snapshot.power_watts - 250.0).abs() < 0.01);
    assert_eq!(snapshot.clock_mhz, 3000);

    // Test memory_usage_percent calculation
    // 8GB / 32GB = 25%
    assert!((snapshot.memory_usage_percent() - 25.0).abs() < 0.01);
}

// =========================================================================
// H024: CpuDevice Internal Method Coverage via Refresh
// =========================================================================

#[test]
fn h024_cpu_device_refresh_populates_fields() {
    let mut cpu = CpuDevice::new();

    // First refresh
    let result = cpu.refresh();
    assert!(result.is_ok());

    // After refresh, utilization should be populated (may be 0 if just started)
    let util = cpu.compute_utilization();
    assert!(util.is_ok());
    let util_val = util.expect("test");
    assert!(util_val >= 0.0 && util_val <= 100.0);

    // Memory used should be reasonable
    let mem = cpu.memory_used_bytes();
    assert!(mem.is_ok());
}

#[test]
fn h024_cpu_device_refresh_multiple_times() {
    let mut cpu = CpuDevice::new();

    // Refresh multiple times should always succeed
    for _ in 0..5 {
        assert!(cpu.refresh().is_ok());
    }

    // Values should still be accessible
    assert!(cpu.compute_utilization().is_ok());
    assert!(cpu.memory_used_bytes().is_ok());
}

// =========================================================================
// H025: CpuDevice Direct Read Functions Coverage
// =========================================================================

#[test]
fn h025_cpu_device_read_core_count() {
    // read_core_count is called in CpuDevice::new()
    // On Linux it reads from /proc/cpuinfo
    // Verify the result is a valid positive number
    let cpu = CpuDevice::new();
    let count = cpu.compute_unit_count();
    assert!(count >= 1, "Should have at least 1 core");
    assert!(count <= 1024, "Sanity check: should have fewer than 1024 cores");
}

#[test]
fn h025_cpu_device_read_total_memory() {
    // read_total_memory is called in CpuDevice::new()
    let cpu = CpuDevice::new();
    let total = cpu.memory_total_bytes().expect("test");
    // System should have at least 1GB and less than 1TB typically
    assert!(total >= 1024 * 1024 * 1024, "Should have at least 1GB");
    assert!(total < 100 * 1024 * 1024 * 1024 * 1024, "Sanity: < 100TB");
}

#[test]
fn h025_cpu_device_read_cpu_name() {
    // read_cpu_name is called in CpuDevice::new()
    let cpu = CpuDevice::new();
    let name = cpu.device_name();
    assert!(!name.is_empty(), "CPU name should not be empty");
    // Name could be "Unknown CPU" if /proc/cpuinfo doesn't have model name
}

// =========================================================================
// H026: CpuDevice Compute Clock Coverage
// =========================================================================

#[test]
fn h026_cpu_device_compute_clock_value() {
    let cpu = CpuDevice::new();
    // On systems with frequency scaling, this should return Ok
    // On systems without, it returns NotSupported
    match cpu.compute_clock_mhz() {
        Ok(mhz) => {
            // Valid frequency range: 100 MHz to 10 GHz
            assert!(mhz >= 100, "Clock should be at least 100 MHz");
            assert!(mhz <= 10000, "Clock should be at most 10 GHz");
        }
        Err(GpuError::NotSupported(_)) => {
            // Expected on systems without frequency info
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

// =========================================================================
// H027: CpuDevice Temperature Coverage
// =========================================================================

#[test]
fn h027_cpu_device_temperature_value() {
    let mut cpu = CpuDevice::new();
    cpu.refresh().expect("test");

    // Temperature may or may not be available depending on hardware/permissions
    match cpu.compute_temperature_c() {
        Ok(temp) => {
            // Valid temperature range: 0 to 150 Celsius
            assert!(temp >= 0.0, "Temperature should be non-negative");
            assert!(temp <= 150.0, "Temperature should be at most 150C");
        }
        Err(GpuError::NotSupported(_)) => {
            // Expected on systems without temperature sensors
        }
        Err(e) => panic!("Unexpected error: {:?}", e),
    }
}

// =========================================================================
// H028: CpuDevice CPU Usage Coverage
// =========================================================================

#[test]
fn h028_cpu_device_cpu_usage_after_refresh() {
    let mut cpu = CpuDevice::new();
    cpu.refresh().expect("test");

    let usage = cpu.compute_utilization().expect("test");
    // CPU usage should be between 0 and 100
    assert!(usage >= 0.0, "CPU usage should be non-negative");
    assert!(usage <= 100.0, "CPU usage should be at most 100%");
}

// =========================================================================
// H029: CpuDevice Memory Used Coverage
// =========================================================================

#[test]
fn h029_cpu_device_memory_used_after_refresh() {
    let mut cpu = CpuDevice::new();
    cpu.refresh().expect("test");

    let used = cpu.memory_used_bytes().expect("test");
    let total = cpu.memory_total_bytes().expect("test");

    // Used should be <= total
    assert!(used <= total, "Used memory should not exceed total");
    // At least some memory should be used (kernel, etc.)
    assert!(used > 0, "Some memory should be in use");
}

// =========================================================================
// H030: MockDevice Additional Coverage
// =========================================================================

#[test]
fn h030_mock_device_device_name() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
    assert_eq!(mock.device_name(), "Mock");
}

#[test]
fn h030_mock_device_device_type() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
    assert!(matches!(mock.device_type(), DeviceType::Cpu));
}

#[test]
fn h030_mock_device_device_id() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
    assert_eq!(mock.device_id(), DeviceId::cpu());
}

#[test]
fn h030_mock_device_compute_utilization() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
    assert!((mock.compute_utilization().expect("test") - 50.0).abs() < 0.01);
}

#[test]
fn h030_mock_device_compute_clock() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
    assert_eq!(mock.compute_clock_mhz().expect("test"), 3000);
}

#[test]
fn h030_mock_device_compute_temperature() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 45.0);
    assert!((mock.compute_temperature_c().expect("test") - 45.0).abs() < 0.01);
}

#[test]
fn h030_mock_device_compute_power() {
    let mock = MockDevice::new(0, 0, 200.0, 300.0, 0.0);
    assert!((mock.compute_power_watts().expect("test") - 200.0).abs() < 0.01);
    assert!((mock.compute_power_limit_watts().expect("test") - 300.0).abs() < 0.01);
}

#[test]
fn h030_mock_device_compute_unit_count() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
    assert_eq!(mock.compute_unit_count(), 8);
}

#[test]
fn h030_mock_device_memory_bytes() {
    let mock = MockDevice::new(1000, 2000, 0.0, 0.0, 0.0);
    assert_eq!(mock.memory_used_bytes().expect("test"), 1000);
    assert_eq!(mock.memory_total_bytes().expect("test"), 2000);
}
