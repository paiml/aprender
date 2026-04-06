//! Extended coverage tests: CpuDevice coverage, MockDevice extended,
//! derived metrics, and boundary tests (H013-H030)

use super::*;

// =========================================================================
// MockDevice (shared test fixture)
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

// =========================================================================
// H013: CpuDevice Additional Coverage
// =========================================================================

#[test]
fn h013_cpu_device_refresh() {
    let mut cpu = CpuDevice::new();
    // Refresh should not panic
    let _ = cpu.refresh();
    // After refresh, metrics should still work
    let _ = cpu.compute_utilization();
    let _ = cpu.memory_used_bytes();
}

#[test]
fn h013_cpu_device_multiple_refreshes() {
    let mut cpu = CpuDevice::new();
    // Multiple refreshes should be idempotent
    for _ in 0..3 {
        let _ = cpu.refresh();
    }
}

#[test]
fn h013_cpu_device_name_not_empty() {
    let cpu = CpuDevice::new();
    assert!(!cpu.device_name().is_empty());
}

#[test]
fn h013_cpu_device_core_count_positive() {
    let cpu = CpuDevice::new();
    assert!(cpu.compute_unit_count() > 0);
}

#[test]
fn h013_cpu_clock_speed() {
    let cpu = CpuDevice::new();
    // Clock speed should be positive (or NotSupported error on some systems)
    // Just verify we can call it without panic
    let _ = cpu.compute_clock_mhz();
}

#[test]
fn h013_cpu_temperature() {
    let cpu = CpuDevice::new();
    // Temperature may not be available on all systems
    // Just verify we can call it without panic
    let _ = cpu.compute_temperature_c();
}

// =========================================================================
// H014: MockDevice Extended Coverage
// =========================================================================

#[test]
fn h014_mock_device_all_methods() {
    let mock = MockDevice::new(1024, 2048, 10.0, 100.0, 30.0);

    // Test all trait method implementations
    assert_eq!(mock.device_id(), DeviceId::cpu());
    assert_eq!(mock.device_name(), "Mock");
    assert!(matches!(mock.device_type(), DeviceType::Cpu));
    assert_eq!(mock.compute_unit_count(), 8);
    assert_eq!(mock.memory_used_bytes().expect("test"), 1024);
    assert_eq!(mock.memory_total_bytes().expect("test"), 2048);
    assert!((mock.compute_utilization().expect("test") - 50.0).abs() < 0.01); // MockDevice always returns 50.0
    assert!((mock.compute_temperature_c().expect("test") - 30.0).abs() < 0.01);
    assert!((mock.compute_power_watts().expect("test") - 10.0).abs() < 0.01);
    assert_eq!(mock.compute_clock_mhz().expect("test"), 3000);
}

#[test]
fn h014_mock_device_derived_metrics() {
    let mock = MockDevice::new(1024, 2048, 10.0, 100.0, 30.0);

    // Derived metrics
    let usage_percent = mock.memory_usage_percent().expect("test");
    assert!((usage_percent - 50.0).abs() < 0.01); // 1024/2048 = 50%

    let available = mock.memory_available_bytes().expect("test");
    assert_eq!(available, 1024); // 2048 - 1024
}

#[test]
fn h014_mock_device_mb_gb_helpers() {
    let mock = MockDevice::new(1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 10.0, 100.0, 30.0);

    let used_mb = mock.memory_used_mb().expect("test");
    assert_eq!(used_mb, 1024); // 1 GB = 1024 MB

    let total_gb = mock.memory_total_gb().expect("test");
    assert!((total_gb - 2.0).abs() < 0.1); // 2 GB
}

// =========================================================================
// H015: DeviceId Additional Coverage
// =========================================================================

#[test]
fn h015_device_id_display() {
    assert_eq!(format!("{}", DeviceId::cpu()), "CPU");
    assert_eq!(format!("{}", DeviceId::nvidia(0)), "NVIDIA:0");
    assert_eq!(format!("{}", DeviceId::nvidia(1)), "NVIDIA:1");
    assert_eq!(format!("{}", DeviceId::amd(0)), "AMD:0");
}

#[test]
fn h015_device_id_debug() {
    let cpu_id = DeviceId::cpu();
    let debug_str = format!("{:?}", cpu_id);
    assert!(debug_str.contains("Cpu"));
}

#[test]
fn h015_device_id_clone() {
    let id1 = DeviceId::nvidia(0);
    let id2 = id1.clone();
    assert_eq!(id1, id2);
}

// =========================================================================
// H016: DeviceSnapshot Additional Coverage
// =========================================================================

#[test]
fn h016_snapshot_debug() {
    let mock = MockDevice::new(1024, 2048, 10.0, 100.0, 30.0);
    let snapshot = DeviceSnapshot::capture(&mock).expect("test");
    let debug_str = format!("{:?}", snapshot);
    assert!(debug_str.contains("DeviceSnapshot"));
}

#[test]
fn h016_snapshot_clone() {
    let mock = MockDevice::new(1024, 2048, 10.0, 100.0, 30.0);
    let snapshot1 = DeviceSnapshot::capture(&mock).expect("test");
    let snapshot2 = snapshot1.clone();
    assert_eq!(snapshot1.device_id, snapshot2.device_id);
    assert_eq!(snapshot1.memory_used_bytes, snapshot2.memory_used_bytes);
}

// =========================================================================
// H017: DeviceType Coverage
// =========================================================================

#[test]
fn h017_device_type_display() {
    assert_eq!(format!("{}", DeviceType::Cpu), "CPU");
    assert_eq!(format!("{}", DeviceType::NvidiaGpu), "NVIDIA GPU");
    assert_eq!(format!("{}", DeviceType::AmdGpu), "AMD GPU");
    assert_eq!(format!("{}", DeviceType::IntelGpu), "Intel GPU");
}

#[test]
fn h017_device_type_debug() {
    let gpu = DeviceType::NvidiaGpu;
    let debug_str = format!("{:?}", gpu);
    assert!(debug_str.contains("NvidiaGpu"));
}

// =========================================================================
// H018: Error Path Coverage (Best Effort)
// =========================================================================

#[test]
fn h018_cpu_unsupported_pcie_metrics() {
    let cpu = CpuDevice::new();

    // PCIe metrics return NotSupported
    let tx = cpu.pcie_tx_bytes_per_sec();
    assert!(matches!(tx, Err(GpuError::NotSupported(_))));

    let rx = cpu.pcie_rx_bytes_per_sec();
    assert!(matches!(rx, Err(GpuError::NotSupported(_))));
}

#[test]
fn h018_cpu_power_metrics() {
    let cpu = CpuDevice::new();

    // Power metrics return NotSupported for CPU
    let power = cpu.compute_power_watts();
    assert!(matches!(power, Err(GpuError::NotSupported(_))));

    let power_limit = cpu.compute_power_limit_watts();
    assert!(matches!(power_limit, Err(GpuError::NotSupported(_))));

    let bw = cpu.memory_bandwidth_gbps();
    assert!(matches!(bw, Err(GpuError::NotSupported(_))));
}

// =========================================================================
// H019: CpuDevice active_compute_units Test
// =========================================================================

#[test]
fn h019_cpu_active_compute_units() {
    let cpu = CpuDevice::new();

    // active_compute_units should return the core count
    let active = cpu.active_compute_units();
    assert!(active.is_ok());
    let count = active.expect("test");
    assert!(count > 0, "Should have at least one active compute unit");
    assert_eq!(count, cpu.compute_unit_count(), "Active should equal total cores");
}

// =========================================================================
// H020: MockDevice Full Coverage Tests
// =========================================================================

#[test]
fn h020_mock_device_pcie_metrics() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);

    // PCIe metrics should return NotSupported
    assert!(matches!(mock.pcie_tx_bytes_per_sec(), Err(GpuError::NotSupported(_))));
    assert!(matches!(mock.pcie_rx_bytes_per_sec(), Err(GpuError::NotSupported(_))));
    assert_eq!(mock.pcie_generation(), 0);
    assert_eq!(mock.pcie_width(), 0);
}

#[test]
fn h020_mock_device_active_compute_units() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);

    let active = mock.active_compute_units();
    assert!(active.is_ok());
    assert_eq!(active.expect("test"), 8); // MockDevice returns 8 compute units
}

#[test]
fn h020_mock_device_memory_bandwidth() {
    let mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);

    assert!(matches!(mock.memory_bandwidth_gbps(), Err(GpuError::NotSupported(_))));
}

#[test]
fn h020_mock_device_refresh() {
    let mut mock = MockDevice::new(0, 0, 0.0, 0.0, 0.0);

    // refresh should succeed
    assert!(mock.refresh().is_ok());
}

// =========================================================================
// H021: CpuDevice Default Derived Metrics
// =========================================================================

#[test]
fn h021_cpu_device_memory_mb_conversion() {
    let cpu = CpuDevice::new();

    // Test memory_used_mb conversion
    if let Ok(used_bytes) = cpu.memory_used_bytes() {
        let used_mb = cpu.memory_used_mb();
        assert!(used_mb.is_ok());
        assert_eq!(used_mb.expect("test"), used_bytes / (1024 * 1024));
    }
}

#[test]
fn h021_cpu_device_memory_total_mb() {
    let cpu = CpuDevice::new();

    if let Ok(total_bytes) = cpu.memory_total_bytes() {
        let total_mb = cpu.memory_total_mb();
        assert!(total_mb.is_ok());
        assert_eq!(total_mb.expect("test"), total_bytes / (1024 * 1024));
    }
}

#[test]
fn h021_cpu_device_memory_total_gb() {
    let cpu = CpuDevice::new();

    if let Ok(total_bytes) = cpu.memory_total_bytes() {
        let total_gb = cpu.memory_total_gb();
        assert!(total_gb.is_ok());
        let expected_gb = total_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        assert!((total_gb.expect("test") - expected_gb).abs() < 0.001);
    }
}

#[test]
fn h021_cpu_device_memory_available() {
    let cpu = CpuDevice::new();

    if let (Ok(used), Ok(total)) = (cpu.memory_used_bytes(), cpu.memory_total_bytes()) {
        let available = cpu.memory_available_bytes();
        assert!(available.is_ok());
        assert_eq!(available.expect("test"), total.saturating_sub(used));
    }
}

mod h022_h030;
