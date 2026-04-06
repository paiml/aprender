//! Core device tests: Device ID, Device Type, CPU Device basics,
//! Device Snapshot, Throttle Reason, and MockDevice (H001-H012)

use super::*;

// =========================================================================
// H001: Device ID Tests
// =========================================================================

#[test]
fn h001_device_id_cpu() {
    let id = DeviceId::cpu();
    assert_eq!(id.device_type, DeviceType::Cpu);
    assert_eq!(id.index, 0);
    assert_eq!(format!("{}", id), "CPU");
}

#[test]
fn h001_device_id_nvidia() {
    let id = DeviceId::nvidia(0);
    assert_eq!(id.device_type, DeviceType::NvidiaGpu);
    assert_eq!(id.index, 0);
    assert_eq!(format!("{}", id), "NVIDIA:0");

    let id2 = DeviceId::nvidia(1);
    assert_eq!(format!("{}", id2), "NVIDIA:1");
}

#[test]
fn h001_device_id_amd() {
    let id = DeviceId::amd(0);
    assert_eq!(id.device_type, DeviceType::AmdGpu);
    assert_eq!(id.index, 0);
    assert_eq!(format!("{}", id), "AMD:0");
}

#[test]
fn h001_device_id_equality() {
    let id1 = DeviceId::nvidia(0);
    let id2 = DeviceId::nvidia(0);
    let id3 = DeviceId::nvidia(1);
    let id4 = DeviceId::amd(0);

    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
    assert_ne!(id1, id4);
}

// =========================================================================
// H002: Device Type Tests
// =========================================================================

#[test]
fn h002_device_type_display() {
    assert_eq!(format!("{}", DeviceType::Cpu), "CPU");
    assert_eq!(format!("{}", DeviceType::NvidiaGpu), "NVIDIA GPU");
    assert_eq!(format!("{}", DeviceType::AmdGpu), "AMD GPU");
    assert_eq!(format!("{}", DeviceType::IntelGpu), "Intel GPU");
    assert_eq!(format!("{}", DeviceType::AppleSilicon), "Apple Silicon");
}

// =========================================================================
// H003: CPU Device Tests
// =========================================================================

#[test]
fn h003_cpu_device_creation() {
    let cpu = CpuDevice::new();
    assert_eq!(cpu.device_type(), DeviceType::Cpu);
    assert_eq!(cpu.device_id(), DeviceId::cpu());
    assert!(cpu.compute_unit_count() > 0);
}

#[test]
fn h003_cpu_device_default() {
    let cpu = CpuDevice::default();
    assert!(cpu.compute_unit_count() > 0);
}

#[test]
fn h003_cpu_device_name() {
    let cpu = CpuDevice::new();
    // Name should be non-empty
    assert!(!cpu.device_name().is_empty());
}

#[test]
fn h003_cpu_device_memory_total() {
    let cpu = CpuDevice::new();
    // Should have some memory (at least 1GB in practice)
    let total = cpu.memory_total_bytes().unwrap_or(0);
    assert!(total > 0, "CPU should report total memory");
}

#[test]
fn h003_cpu_device_refresh() {
    let mut cpu = CpuDevice::new();
    assert!(cpu.refresh().is_ok());
    // After refresh, metrics should be populated
}

// =========================================================================
// H004: Device Snapshot Tests
// =========================================================================

#[test]
fn h004_device_snapshot_capture() {
    let cpu = CpuDevice::new();
    let snapshot = DeviceSnapshot::capture(&cpu);
    assert!(snapshot.is_ok());

    let snap = snapshot.expect("test");
    assert_eq!(snap.device_id, DeviceId::cpu());
    assert!(snap.timestamp_ms > 0);
}

#[test]
fn h004_device_snapshot_memory_percent() {
    let snap = DeviceSnapshot {
        device_id: DeviceId::cpu(),
        timestamp_ms: 0,
        compute_utilization: 50.0,
        memory_used_bytes: 50 * 1024 * 1024 * 1024, // 50 GB
        memory_total_bytes: 100 * 1024 * 1024 * 1024, // 100 GB
        temperature_c: 45.0,
        power_watts: 100.0,
        clock_mhz: 3000,
    };

    assert!((snap.memory_usage_percent() - 50.0).abs() < 0.01);
}

#[test]
fn h004_device_snapshot_memory_percent_zero_total() {
    let snap = DeviceSnapshot {
        device_id: DeviceId::cpu(),
        timestamp_ms: 0,
        compute_utilization: 0.0,
        memory_used_bytes: 0,
        memory_total_bytes: 0, // Division by zero case
        temperature_c: 0.0,
        power_watts: 0.0,
        clock_mhz: 0,
    };

    assert!((snap.memory_usage_percent() - 0.0).abs() < 0.01);
}

// =========================================================================
// H005: Throttle Reason Tests
// =========================================================================

#[test]
fn h005_throttle_reason_display() {
    assert_eq!(format!("{}", ThrottleReason::None), "None");
    assert_eq!(format!("{}", ThrottleReason::Thermal), "Thermal");
    assert_eq!(format!("{}", ThrottleReason::Power), "Power");
}

// =========================================================================
// H006: Derived Metrics Tests (ComputeDevice trait defaults)
// =========================================================================

#[test]
fn h006_memory_usage_percent() {
    let cpu = CpuDevice::new();
    let percent = cpu.memory_usage_percent();
    // On Linux, this should always succeed
    assert!(percent.is_ok());
    let p = percent.expect("test");
    assert!(p >= 0.0 && p <= 100.0);
}

#[test]
fn h006_memory_available_bytes() {
    let cpu = CpuDevice::new();
    // On Linux, these should always succeed
    let avail = cpu.memory_available_bytes().expect("test");
    let total = cpu.memory_total_bytes().expect("test");
    assert!(avail <= total);
}

#[test]
fn h006_memory_mb_helpers() {
    let cpu = CpuDevice::new();
    // On Linux, these should always succeed
    let used_mb = cpu.memory_used_mb().expect("test");
    let total_mb = cpu.memory_total_mb().expect("test");
    assert!(used_mb <= total_mb);
}

#[test]
fn h006_memory_gb_helper() {
    let cpu = CpuDevice::new();
    // On Linux, this should always succeed
    let total_gb = cpu.memory_total_gb().expect("test");
    // Should be positive (most systems have > 1GB)
    assert!(total_gb > 0.0);
}

// =========================================================================
// H007: Thermal Throttling Detection
// =========================================================================

#[test]
fn h007_thermal_throttling_detection() {
    let cpu = CpuDevice::new();
    // Just verify it doesn't panic
    let _ = cpu.is_thermal_throttling();
}

// =========================================================================
// H008: Power Throttling Detection
// =========================================================================

#[test]
fn h008_power_throttling_detection() {
    let cpu = CpuDevice::new();
    // CPU doesn't support power metrics, but shouldn't panic
    let _ = cpu.is_power_throttling();
}

// =========================================================================
// H009: Edge Cases
// =========================================================================

#[test]
fn h009_cpu_unsupported_metrics() {
    let cpu = CpuDevice::new();

    // PCIe metrics should return NotSupported
    assert!(cpu.pcie_tx_bytes_per_sec().is_err());
    assert!(cpu.pcie_rx_bytes_per_sec().is_err());
    assert_eq!(cpu.pcie_generation(), 0);
    assert_eq!(cpu.pcie_width(), 0);
}

#[test]
fn h009_device_id_hash() {
    use std::collections::HashSet;

    let mut set = HashSet::new();
    set.insert(DeviceId::cpu());
    set.insert(DeviceId::nvidia(0));
    set.insert(DeviceId::nvidia(1));
    set.insert(DeviceId::amd(0));

    assert_eq!(set.len(), 4);

    // Duplicate should not increase size
    set.insert(DeviceId::cpu());
    assert_eq!(set.len(), 4);
}

// =========================================================================
// H010: Additional Display Coverage
// =========================================================================

#[test]
fn h010_device_id_intel_display() {
    let id = DeviceId::new(DeviceType::IntelGpu, 0);
    assert_eq!(format!("{}", id), "Intel:0");
    assert_eq!(id.device_type, DeviceType::IntelGpu);

    let id2 = DeviceId::new(DeviceType::IntelGpu, 2);
    assert_eq!(format!("{}", id2), "Intel:2");
}

#[test]
fn h010_device_id_apple_display() {
    let id = DeviceId::new(DeviceType::AppleSilicon, 0);
    assert_eq!(format!("{}", id), "Apple:0");
    assert_eq!(id.device_type, DeviceType::AppleSilicon);
}

#[test]
fn h010_throttle_reason_all_variants() {
    // Cover all ThrottleReason Display variants
    assert_eq!(format!("{}", ThrottleReason::None), "None");
    assert_eq!(format!("{}", ThrottleReason::Thermal), "Thermal");
    assert_eq!(format!("{}", ThrottleReason::Power), "Power");
    assert_eq!(format!("{}", ThrottleReason::ApplicationClocks), "AppClocks");
    assert_eq!(format!("{}", ThrottleReason::SwPowerCap), "SwPowerCap");
    assert_eq!(format!("{}", ThrottleReason::HwSlowdown), "HwSlowdown");
    assert_eq!(format!("{}", ThrottleReason::SyncBoost), "SyncBoost");
}

// =========================================================================
// H011: Default Trait Impl Edge Cases
// =========================================================================

/// Mock device to test default trait implementations with controlled values
struct MockDevice {
    mem_used: u64,
    mem_total: u64,
    power_current: f64,
    power_limit: f64,
    temperature: f64,
}

impl MockDevice {
    fn new(
        mem_used: u64,
        mem_total: u64,
        power_current: f64,
        power_limit: f64,
        temperature: f64,
    ) -> Self {
        Self { mem_used, mem_total, power_current, power_limit, temperature }
    }
}

impl ComputeDevice for MockDevice {
    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }
    fn device_name(&self) -> &str {
        "Mock"
    }
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }
    fn compute_utilization(&self) -> Result<f64, GpuError> {
        Ok(50.0)
    }
    fn compute_clock_mhz(&self) -> Result<u32, GpuError> {
        Ok(3000)
    }
    fn compute_temperature_c(&self) -> Result<f64, GpuError> {
        Ok(self.temperature)
    }
    fn compute_power_watts(&self) -> Result<f64, GpuError> {
        Ok(self.power_current)
    }
    fn compute_power_limit_watts(&self) -> Result<f64, GpuError> {
        Ok(self.power_limit)
    }
    fn memory_used_bytes(&self) -> Result<u64, GpuError> {
        Ok(self.mem_used)
    }
    fn memory_total_bytes(&self) -> Result<u64, GpuError> {
        Ok(self.mem_total)
    }
    fn memory_bandwidth_gbps(&self) -> Result<f64, GpuError> {
        Err(GpuError::NotSupported("mock".into()))
    }
    fn compute_unit_count(&self) -> u32 {
        8
    }
    fn active_compute_units(&self) -> Result<u32, GpuError> {
        Ok(8)
    }
    fn pcie_tx_bytes_per_sec(&self) -> Result<u64, GpuError> {
        Err(GpuError::NotSupported("mock".into()))
    }
    fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError> {
        Err(GpuError::NotSupported("mock".into()))
    }
    fn pcie_generation(&self) -> u8 {
        0
    }
    fn pcie_width(&self) -> u8 {
        0
    }
    fn refresh(&mut self) -> Result<(), GpuError> {
        Ok(())
    }
}

#[test]
fn h011_memory_usage_percent_zero_total() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);
    // Zero total should return 0.0, not divide by zero
    assert!((mock.memory_usage_percent().expect("test") - 0.0).abs() < 0.01);
}

#[test]
fn h011_memory_usage_percent_normal() {
    let mock = MockDevice::new(50 * 1024 * 1024 * 1024, 100 * 1024 * 1024 * 1024, 0.0, 0.0, 0.0);
    // 50% usage
    assert!((mock.memory_usage_percent().expect("test") - 50.0).abs() < 0.01);
}

#[test]
fn h011_memory_available_bytes() {
    let mock = MockDevice::new(30 * 1024 * 1024 * 1024, 100 * 1024 * 1024 * 1024, 0.0, 0.0, 0.0);
    // 70GB available
    let available = mock.memory_available_bytes().expect("test");
    assert_eq!(available, 70 * 1024 * 1024 * 1024);
}

#[test]
fn h011_memory_mb_gb_conversions() {
    let mock = MockDevice::new(1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024, 0.0, 0.0, 0.0);
    // 1GB used = 1024MB
    assert_eq!(mock.memory_used_mb().expect("test"), 1024);
    // 16GB total = 16384MB
    assert_eq!(mock.memory_total_mb().expect("test"), 16384);
    // 16GB as f64
    assert!((mock.memory_total_gb().expect("test") - 16.0).abs() < 0.01);
}

#[test]
fn h011_power_usage_percent_zero_limit() {
    let mock = MockDevice::new(0, 0, 100.0, 0.0, 0.0);
    // Zero limit should return 0.0, not divide by zero
    assert!((mock.power_usage_percent().expect("test") - 0.0).abs() < 0.01);
}

#[test]
fn h011_power_usage_percent_normal() {
    let mock = MockDevice::new(0, 0, 150.0, 300.0, 0.0);
    // 50% power usage
    assert!((mock.power_usage_percent().expect("test") - 50.0).abs() < 0.01);
}

#[test]
fn h011_thermal_throttling_below_threshold() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 75.0);
    // Below 80C - no throttling
    assert!(!mock.is_thermal_throttling().expect("test"));
}

#[test]
fn h011_thermal_throttling_above_threshold() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 85.0);
    // Above 80C - throttling
    assert!(mock.is_thermal_throttling().expect("test"));
}

#[test]
fn h011_power_throttling_below_threshold() {
    let mock = MockDevice::new(0, 0, 90.0, 100.0, 0.0);
    // 90% - below 95% threshold
    assert!(!mock.is_power_throttling().expect("test"));
}

#[test]
fn h011_power_throttling_above_threshold() {
    let mock = MockDevice::new(0, 0, 98.0, 100.0, 0.0);
    // 98% - above 95% threshold
    assert!(mock.is_power_throttling().expect("test"));
}

// =========================================================================
// H012: DeviceSnapshot Edge Cases
// =========================================================================

#[test]
fn h012_device_snapshot_from_mock() {
    let mock = MockDevice::new(8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024, 150.0, 300.0, 65.0);
    let snapshot = DeviceSnapshot::capture(&mock).expect("test");

    assert_eq!(snapshot.device_id, DeviceId::cpu());
    assert!((snapshot.compute_utilization - 50.0).abs() < 0.01);
    assert_eq!(snapshot.memory_used_bytes, 8 * 1024 * 1024 * 1024);
    assert_eq!(snapshot.memory_total_bytes, 16 * 1024 * 1024 * 1024);
    assert!((snapshot.temperature_c - 65.0).abs() < 0.01);
    assert!((snapshot.power_watts - 150.0).abs() < 0.01);
    assert_eq!(snapshot.clock_mhz, 3000);
}
