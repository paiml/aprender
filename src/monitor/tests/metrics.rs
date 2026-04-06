//! H0-MON-30 through H0-MON-61: Memory, thermal, power, and GPU metrics tests

use std::time::Duration;

use crate::monitor::*;

// =========================================================================
// H0-MON-30: GpuMemoryMetrics
// =========================================================================

#[test]
fn h0_mon_30_memory_metrics_basic() {
    let mem = GpuMemoryMetrics::new(24_000_000_000, 8_000_000_000, 16_000_000_000);

    assert_eq!(mem.total, 24_000_000_000);
    assert_eq!(mem.used, 8_000_000_000);
    assert_eq!(mem.free, 16_000_000_000);
}

#[test]
fn h0_mon_31_memory_metrics_usage_percent() {
    let mem = GpuMemoryMetrics::new(100, 25, 75);
    assert!((mem.usage_percent() - 25.0).abs() < 0.01);
}

#[test]
fn h0_mon_32_memory_metrics_usage_percent_zero_total() {
    let mem = GpuMemoryMetrics::new(0, 0, 0);
    assert!((mem.usage_percent() - 0.0).abs() < 0.01);
}

#[test]
fn h0_mon_33_memory_metrics_mb_helpers() {
    let mem = GpuMemoryMetrics::new(
        24 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
    );

    assert_eq!(mem.used_mb(), 8 * 1024);
    assert_eq!(mem.free_mb(), 16 * 1024);
}

// =========================================================================
// H0-MON-40: GpuThermalMetrics
// =========================================================================

#[test]
fn h0_mon_40_thermal_safe() {
    let thermal = GpuThermalMetrics { temperature_celsius: 50, ..Default::default() };
    assert!(thermal.is_safe());
    assert!(!thermal.is_critical());
    assert_eq!(thermal.status(), "COOL");
}

#[test]
fn h0_mon_41_thermal_warm() {
    let thermal = GpuThermalMetrics { temperature_celsius: 65, ..Default::default() };
    assert!(thermal.is_safe());
    assert!(!thermal.is_critical());
    assert_eq!(thermal.status(), "WARM");
}

#[test]
fn h0_mon_42_thermal_hot() {
    let thermal = GpuThermalMetrics { temperature_celsius: 82, ..Default::default() };
    assert!(!thermal.is_safe());
    assert!(!thermal.is_critical());
    assert_eq!(thermal.status(), "HOT");
}

#[test]
fn h0_mon_43_thermal_critical() {
    let thermal = GpuThermalMetrics { temperature_celsius: 95, ..Default::default() };
    assert!(!thermal.is_safe());
    assert!(thermal.is_critical());
    assert_eq!(thermal.status(), "CRITICAL");
}

// =========================================================================
// H0-MON-50: GpuPowerMetrics
// =========================================================================

#[test]
fn h0_mon_50_power_usage_percent() {
    let power =
        GpuPowerMetrics { power_draw_watts: 225.0, power_limit_watts: 450.0, power_state: 0 };
    assert!((power.usage_percent() - 50.0).abs() < 0.01);
}

#[test]
fn h0_mon_51_power_usage_percent_zero_limit() {
    let power = GpuPowerMetrics { power_draw_watts: 100.0, power_limit_watts: 0.0, power_state: 0 };
    assert!((power.usage_percent() - 0.0).abs() < 0.01);
}

// =========================================================================
// H0-MON-60: GpuMetrics
// =========================================================================

#[test]
fn h0_mon_60_metrics_creation() {
    let mem = GpuMemoryMetrics::new(1000, 500, 500);
    let metrics = GpuMetrics::new(0, mem);

    assert_eq!(metrics.device_index, 0);
    assert_eq!(metrics.memory.total, 1000);
    assert!(metrics.thermal.is_none());
    assert!(metrics.power.is_none());
}

#[test]
fn h0_mon_61_metrics_age() {
    let mem = GpuMemoryMetrics::new(1000, 500, 500);
    let metrics = GpuMetrics::new(0, mem);

    // Age should be very small immediately after creation
    assert!(metrics.age() < Duration::from_millis(100));
}
