//! Error propagation tests: ErrorMockDevice, PartialErrorMockDevice,
//! ThrottleReason/DeviceType trait coverage, and boundary tests (H031-H044)

use super::*;

// =========================================================================
// MockDevice (shared test fixture for snapshot/field tests)
// =========================================================================

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
// ErrorMockDevice (shared test fixture)
// =========================================================================

/// Mock device that returns errors for testing error propagation in default trait methods
struct ErrorMockDevice {
    return_error: bool,
}

impl ErrorMockDevice {
    fn new(return_error: bool) -> Self {
        Self { return_error }
    }
}

impl ComputeDevice for ErrorMockDevice {
    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }
    fn device_name(&self) -> &str {
        "ErrorMock"
    }
    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }
    fn compute_utilization(&self) -> Result<f64, GpuError> {
        if self.return_error {
            Err(GpuError::NotSupported("test".into()))
        } else {
            Ok(50.0)
        }
    }
    fn compute_clock_mhz(&self) -> Result<u32, GpuError> {
        if self.return_error {
            Err(GpuError::NotSupported("test".into()))
        } else {
            Ok(3000)
        }
    }
    fn compute_temperature_c(&self) -> Result<f64, GpuError> {
        if self.return_error {
            Err(GpuError::NotSupported("test".into()))
        } else {
            Ok(50.0)
        }
    }
    fn compute_power_watts(&self) -> Result<f64, GpuError> {
        if self.return_error {
            Err(GpuError::NotSupported("test".into()))
        } else {
            Ok(100.0)
        }
    }
    fn compute_power_limit_watts(&self) -> Result<f64, GpuError> {
        if self.return_error {
            Err(GpuError::NotSupported("test".into()))
        } else {
            Ok(200.0)
        }
    }
    fn memory_used_bytes(&self) -> Result<u64, GpuError> {
        if self.return_error {
            Err(GpuError::NotSupported("test".into()))
        } else {
            Ok(1024)
        }
    }
    fn memory_total_bytes(&self) -> Result<u64, GpuError> {
        if self.return_error {
            Err(GpuError::NotSupported("test".into()))
        } else {
            Ok(2048)
        }
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
// PartialErrorMockDevice (shared test fixture)
// =========================================================================

/// Mock device that returns errors only for total memory (not used)
/// to test error propagation in memory_usage_percent when second call fails
struct PartialErrorMockDevice {
    error_on_total: bool,
    error_on_limit: bool,
}

impl PartialErrorMockDevice {
    fn with_total_error() -> Self {
        Self { error_on_total: true, error_on_limit: false }
    }

    fn with_limit_error() -> Self {
        Self { error_on_total: false, error_on_limit: true }
    }
}

impl ComputeDevice for PartialErrorMockDevice {
    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }
    fn device_name(&self) -> &str {
        "PartialErrorMock"
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
        Ok(50.0)
    }
    fn compute_power_watts(&self) -> Result<f64, GpuError> {
        Ok(100.0)
    }
    fn compute_power_limit_watts(&self) -> Result<f64, GpuError> {
        if self.error_on_limit {
            Err(GpuError::NotSupported("limit error".into()))
        } else {
            Ok(200.0)
        }
    }
    fn memory_used_bytes(&self) -> Result<u64, GpuError> {
        Ok(1024)
    }
    fn memory_total_bytes(&self) -> Result<u64, GpuError> {
        if self.error_on_total {
            Err(GpuError::NotSupported("total error".into()))
        } else {
            Ok(2048)
        }
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

mod h031_h035;
mod h036_h041;
mod h042_h044;
