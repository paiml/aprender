//! Unified Compute Device Abstraction (TRUENO-SPEC-020)
//!
//! Hardware abstraction layer providing a unified interface for CPU, NVIDIA GPU,
//! and AMD GPU monitoring.
//!
//! # Design Principles (Toyota Way)
//!
//! | Principle | Application |
//! |-----------|-------------|
//! | **Genchi Genbutsu** | Direct hardware sampling via native APIs |
//! | **Poka-Yoke** | Type-safe metrics prevent unit confusion |
//!
//! # References
//!
//! - [Nickolls2008] CUDA programming model
//! - [Jia2018] GPU microarchitecture analysis

mod cpu;
mod types;

pub use cpu::*;
pub use types::*;

use crate::GpuError;

// ============================================================================
// Unified Device Trait (TRUENO-SPEC-020 Section 2.1)
// ============================================================================

/// Unified compute device abstraction
///
/// All compute devices (CPU, NVIDIA GPU, AMD GPU) implement this trait
/// for consistent monitoring across heterogeneous hardware.
///
/// # Example
///
/// ```rust,ignore
/// use trueno_gpu::monitor::{ComputeDevice, CpuDevice};
///
/// let cpu = CpuDevice::new();
/// println!("CPU: {} @ {:.1}%", cpu.device_name(), cpu.compute_utilization()?);
/// ```
pub trait ComputeDevice: Send + Sync {
    /// Get the unique device identifier
    fn device_id(&self) -> DeviceId;

    /// Get the device name (e.g., "NVIDIA GeForce RTX 4090")
    fn device_name(&self) -> &str;

    /// Get the device type
    fn device_type(&self) -> DeviceType;

    /// Get compute utilization (0.0-100.0%)
    fn compute_utilization(&self) -> Result<f64, GpuError>;

    /// Get compute clock speed in MHz
    fn compute_clock_mhz(&self) -> Result<u32, GpuError>;

    /// Get compute temperature in Celsius
    fn compute_temperature_c(&self) -> Result<f64, GpuError>;

    /// Get current power consumption in Watts
    fn compute_power_watts(&self) -> Result<f64, GpuError>;

    /// Get power limit in Watts
    fn compute_power_limit_watts(&self) -> Result<f64, GpuError>;

    /// Get used memory in bytes
    fn memory_used_bytes(&self) -> Result<u64, GpuError>;

    /// Get total memory in bytes
    fn memory_total_bytes(&self) -> Result<u64, GpuError>;

    /// Get memory bandwidth in GB/s (if available)
    fn memory_bandwidth_gbps(&self) -> Result<f64, GpuError>;

    /// Get number of compute units (SMs for NVIDIA, CUs for AMD, cores for CPU)
    fn compute_unit_count(&self) -> u32;

    /// Get number of active compute units
    fn active_compute_units(&self) -> Result<u32, GpuError>;

    /// Get PCIe TX bytes per second (GPU only)
    fn pcie_tx_bytes_per_sec(&self) -> Result<u64, GpuError>;

    /// Get PCIe RX bytes per second (GPU only)
    fn pcie_rx_bytes_per_sec(&self) -> Result<u64, GpuError>;

    /// Get PCIe generation (1, 2, 3, 4, 5)
    fn pcie_generation(&self) -> u8;

    /// Get PCIe width (x1, x4, x8, x16)
    fn pcie_width(&self) -> u8;

    /// Refresh metrics from hardware
    fn refresh(&mut self) -> Result<(), GpuError>;

    // =========================================================================
    // Default implementations for derived metrics
    // =========================================================================

    /// Get memory usage percentage (0.0-100.0)
    fn memory_usage_percent(&self) -> Result<f64, GpuError> {
        let used = self.memory_used_bytes()?;
        let total = self.memory_total_bytes()?;
        if total == 0 {
            return Ok(0.0);
        }
        Ok((used as f64 / total as f64) * 100.0)
    }

    /// Get available memory in bytes
    fn memory_available_bytes(&self) -> Result<u64, GpuError> {
        let used = self.memory_used_bytes()?;
        let total = self.memory_total_bytes()?;
        Ok(total.saturating_sub(used))
    }

    /// Get memory used in MB
    fn memory_used_mb(&self) -> Result<u64, GpuError> {
        Ok(self.memory_used_bytes()? / (1024 * 1024))
    }

    /// Get memory total in MB
    fn memory_total_mb(&self) -> Result<u64, GpuError> {
        Ok(self.memory_total_bytes()? / (1024 * 1024))
    }

    /// Get memory total in GB
    fn memory_total_gb(&self) -> Result<f64, GpuError> {
        Ok(self.memory_total_bytes()? as f64 / (1024.0 * 1024.0 * 1024.0))
    }

    /// Get power usage percentage (current/limit * 100)
    fn power_usage_percent(&self) -> Result<f64, GpuError> {
        let current = self.compute_power_watts()?;
        let limit = self.compute_power_limit_watts()?;
        if limit == 0.0 {
            return Ok(0.0);
        }
        Ok((current / limit) * 100.0)
    }

    /// Check if device is throttling due to temperature
    fn is_thermal_throttling(&self) -> Result<bool, GpuError> {
        let temp = self.compute_temperature_c()?;
        // Conservative threshold - most GPUs throttle around 83-85°C
        Ok(temp > 80.0)
    }

    /// Check if device is throttling due to power
    fn is_power_throttling(&self) -> Result<bool, GpuError> {
        let percent = self.power_usage_percent()?;
        Ok(percent > 95.0)
    }
}

// ============================================================================
// Tests (Extreme TDD - TRUENO-SPEC-020)
// ============================================================================

#[cfg(test)]
mod tests_core;

#[cfg(test)]
mod tests_coverage;

#[cfg(test)]
mod tests_error_propagation;
