//! Device types and data structures (TRUENO-SPEC-020)
//!
//! Pure data types for device identification, throttling, and metric snapshots.

use std::fmt;

use super::ComputeDevice;
use crate::GpuError;

// ============================================================================
// Device Identification (TRUENO-SPEC-020 Section 2.1)
// ============================================================================

/// Unique identifier for a compute device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceId {
    /// Device type discriminant
    pub device_type: DeviceType,
    /// Device index within its type (e.g., GPU 0, GPU 1)
    pub index: u32,
}

impl DeviceId {
    /// Create a new device ID
    #[must_use]
    pub const fn new(device_type: DeviceType, index: u32) -> Self {
        Self { device_type, index }
    }

    /// Create CPU device ID
    #[must_use]
    pub const fn cpu() -> Self {
        Self::new(DeviceType::Cpu, 0)
    }

    /// Create NVIDIA GPU device ID
    #[must_use]
    pub const fn nvidia(index: u32) -> Self {
        Self::new(DeviceType::NvidiaGpu, index)
    }

    /// Create AMD GPU device ID
    #[must_use]
    pub const fn amd(index: u32) -> Self {
        Self::new(DeviceType::AmdGpu, index)
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.device_type {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::NvidiaGpu => write!(f, "NVIDIA:{}", self.index),
            DeviceType::AmdGpu => write!(f, "AMD:{}", self.index),
            DeviceType::IntelGpu => write!(f, "Intel:{}", self.index),
            DeviceType::AppleSilicon => write!(f, "Apple:{}", self.index),
        }
    }
}

/// Type of compute device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU (x86, ARM, etc.)
    Cpu,
    /// NVIDIA GPU (CUDA)
    NvidiaGpu,
    /// AMD GPU (ROCm/HIP)
    AmdGpu,
    /// Intel GPU (oneAPI)
    IntelGpu,
    /// Apple Silicon (Metal)
    AppleSilicon,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::NvidiaGpu => write!(f, "NVIDIA GPU"),
            Self::AmdGpu => write!(f, "AMD GPU"),
            Self::IntelGpu => write!(f, "Intel GPU"),
            Self::AppleSilicon => write!(f, "Apple Silicon"),
        }
    }
}

// ============================================================================
// Throttle Reason (TRUENO-SPEC-020 Section 4.2)
// ============================================================================

/// Reason for compute throttling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThrottleReason {
    /// No throttling
    None,
    /// Thermal throttling (temperature limit)
    Thermal,
    /// Power throttling (power limit)
    Power,
    /// Application-set clock limits
    ApplicationClocks,
    /// Software power cap
    SwPowerCap,
    /// Hardware slowdown (external factors)
    HwSlowdown,
    /// Sync boost throttling
    SyncBoost,
}

impl fmt::Display for ThrottleReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Thermal => write!(f, "Thermal"),
            Self::Power => write!(f, "Power"),
            Self::ApplicationClocks => write!(f, "AppClocks"),
            Self::SwPowerCap => write!(f, "SwPowerCap"),
            Self::HwSlowdown => write!(f, "HwSlowdown"),
            Self::SyncBoost => write!(f, "SyncBoost"),
        }
    }
}

// ============================================================================
// Device Snapshot (for history tracking)
// ============================================================================

/// Point-in-time snapshot of device metrics
#[derive(Debug, Clone)]
pub struct DeviceSnapshot {
    /// Device ID
    pub device_id: DeviceId,
    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
    /// Compute utilization (0.0-100.0)
    pub compute_utilization: f64,
    /// Memory used bytes
    pub memory_used_bytes: u64,
    /// Memory total bytes
    pub memory_total_bytes: u64,
    /// Temperature in Celsius
    pub temperature_c: f64,
    /// Power in Watts
    pub power_watts: f64,
    /// Clock speed in MHz
    pub clock_mhz: u32,
}

impl DeviceSnapshot {
    /// Create snapshot from a compute device
    pub fn capture<D: ComputeDevice>(device: &D) -> Result<Self, GpuError> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Ok(Self {
            device_id: device.device_id(),
            timestamp_ms: now,
            compute_utilization: device.compute_utilization().unwrap_or(0.0),
            memory_used_bytes: device.memory_used_bytes().unwrap_or(0),
            memory_total_bytes: device.memory_total_bytes().unwrap_or(0),
            temperature_c: device.compute_temperature_c().unwrap_or(0.0),
            power_watts: device.compute_power_watts().unwrap_or(0.0),
            clock_mhz: device.compute_clock_mhz().unwrap_or(0),
        })
    }

    /// Get memory usage percentage
    #[must_use]
    pub fn memory_usage_percent(&self) -> f64 {
        if self.memory_total_bytes == 0 {
            return 0.0;
        }
        (self.memory_used_bytes as f64 / self.memory_total_bytes as f64) * 100.0
    }
}
