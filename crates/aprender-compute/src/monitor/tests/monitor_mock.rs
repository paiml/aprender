//! H0-MON-100 through H0-MON-106: GpuMonitor mock tests

use std::time::Duration;

use crate::monitor::*;

// =========================================================================
// H0-MON-100: GpuMonitor (mock)
// =========================================================================

#[test]
fn h0_mon_100_monitor_mock_creation() {
    let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
        .with_vram(24 * 1024 * 1024 * 1024);
    let config = MonitorConfig::default();
    let monitor = GpuMonitor::mock(info, config);

    assert_eq!(monitor.device_info().name, "Mock GPU");
    assert_eq!(monitor.config().poll_interval, Duration::from_millis(100));
}

#[test]
fn h0_mon_101_monitor_collect() {
    let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan)
        .with_vram(24 * 1024 * 1024 * 1024);
    let config = MonitorConfig::default();
    let monitor = GpuMonitor::mock(info, config);

    // Initially no samples
    assert_eq!(monitor.sample_count(), 0);

    // Collect a sample
    let metrics = monitor.collect().expect("collect should work");
    assert_eq!(metrics.device_index, 0);
    assert_eq!(monitor.sample_count(), 1);

    // Collect more samples
    monitor.collect().expect("collect should work");
    monitor.collect().expect("collect should work");
    assert_eq!(monitor.sample_count(), 3);
}

#[test]
fn h0_mon_102_monitor_history_buffer() {
    let info =
        GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan).with_vram(1024);

    // Small history size to test ring buffer
    let config = MonitorConfig { history_size: 3, ..Default::default() };
    let monitor = GpuMonitor::mock(info, config);

    // Fill beyond capacity
    for _ in 0..5 {
        monitor.collect().expect("collect should work");
    }

    // Should only have 3 samples (ring buffer)
    assert_eq!(monitor.sample_count(), 3);

    // History should return 3 items
    let history = monitor.history();
    assert_eq!(history.len(), 3);
}

#[test]
fn h0_mon_103_monitor_latest() {
    let info =
        GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan).with_vram(1024);
    let config = MonitorConfig::default();
    let monitor = GpuMonitor::mock(info, config);

    // No samples yet - should error
    assert!(monitor.latest().is_err());

    // After collecting, latest should work
    monitor.collect().expect("collect should work");
    let latest = monitor.latest().expect("latest should work");
    assert_eq!(latest.device_index, 0);
}

#[test]
fn h0_mon_104_monitor_clear_history() {
    let info =
        GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan).with_vram(1024);
    let config = MonitorConfig::default();
    let monitor = GpuMonitor::mock(info, config);

    // Collect some samples
    monitor.collect().expect("collect should work");
    monitor.collect().expect("collect should work");
    assert_eq!(monitor.sample_count(), 2);

    // Clear history
    monitor.clear_history();
    assert_eq!(monitor.sample_count(), 0);
}

#[test]
fn h0_mon_105_monitor_is_collecting() {
    let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan);
    let config = MonitorConfig::default();
    let monitor = GpuMonitor::mock(info, config);

    // Mock monitor is not actively collecting in background
    assert!(!monitor.is_collecting());
}

#[test]
fn h0_mon_106_monitor_stop() {
    let info = GpuDeviceInfo::new(0, "Mock GPU", GpuVendor::Nvidia, GpuBackend::Vulkan);
    let config = MonitorConfig::default();
    let monitor = GpuMonitor::mock(info, config);

    // Stop should not panic even without background collection
    monitor.stop();
}
