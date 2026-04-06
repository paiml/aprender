//! CUDA GPU Monitoring (TRUENO-SPEC-010)
//!
//! Provides native CUDA device monitoring via the CUDA Driver API.
//! This module enables accurate device information and real-time memory metrics.
//!
//! # Design Philosophy
//!
//! **Native CUDA**: Direct access via cuDeviceGetName, cuMemGetInfo provides
//! accurate information (e.g., "NVIDIA GeForce RTX 4090") compared to wgpu's
//! generic backend queries.
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno_gpu::monitor::{CudaDeviceInfo, CudaMemoryInfo};
//!
//! // Query device info
//! let info = CudaDeviceInfo::query(0)?;
//! println!("GPU: {} ({} GB)", info.name, info.total_memory_gb());
//!
//! // Query memory usage
//! let mem = CudaMemoryInfo::query()?;
//! println!("Free: {} / {} MB", mem.free_mb(), mem.total_mb());
//! ```
//!
//! # References
//!
//! - NVIDIA CUDA Driver API: cuDeviceGetName, cuDeviceTotalMem, cuMemGetInfo
//! - TRUENO-SPEC-010: GPU Monitoring, Tracing, and Visualization

#[cfg(feature = "cuda")]
use crate::driver::{cuda_available, device_count, CudaContext};
use crate::GpuError;

// ============================================================================
// CUDA Device Information (TRUENO-SPEC-010 Section 3.1)
// ============================================================================

/// CUDA device information from native driver API
///
/// Provides accurate device information including:
/// - Device name (e.g., "NVIDIA GeForce RTX 4090")
/// - Total VRAM in bytes
/// - Device ordinal
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    /// Device ordinal (0-based index)
    pub index: u32,
    /// Device name from cuDeviceGetName
    pub name: String,
    /// Total VRAM in bytes from cuDeviceTotalMem
    pub total_memory: u64,
}

impl CudaDeviceInfo {
    /// Query device information for the specified device index
    ///
    /// # Arguments
    ///
    /// * `device_index` - Device ordinal (0 for first GPU)
    ///
    /// # Errors
    ///
    /// Returns error if device is not found or query fails.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let info = CudaDeviceInfo::query(0)?;
    /// println!("GPU: {}", info.name);
    /// ```
    #[cfg(feature = "cuda")]
    #[allow(clippy::cast_possible_wrap)]
    pub fn query(device_index: u32) -> Result<Self, GpuError> {
        let ctx = CudaContext::new(device_index as i32)?;
        let name = ctx.device_name()?;
        let total_memory = ctx.total_memory()? as u64;

        Ok(Self { index: device_index, name, total_memory })
    }

    /// Query device information (non-CUDA stub)
    #[cfg(not(feature = "cuda"))]
    pub fn query(_device_index: u32) -> Result<Self, GpuError> {
        Err(GpuError::CudaNotAvailable("cuda feature not enabled".to_string()))
    }

    /// Enumerate all available CUDA devices
    ///
    /// # Errors
    ///
    /// Returns error if enumeration fails.
    #[cfg(feature = "cuda")]
    pub fn enumerate() -> Result<Vec<Self>, GpuError> {
        let count = device_count()?;
        let mut devices = Vec::with_capacity(count);

        for i in 0..count {
            devices.push(Self::query(i as u32)?);
        }

        Ok(devices)
    }

    /// Enumerate devices (non-CUDA stub)
    #[cfg(not(feature = "cuda"))]
    pub fn enumerate() -> Result<Vec<Self>, GpuError> {
        Err(GpuError::CudaNotAvailable("cuda feature not enabled".to_string()))
    }

    /// Get total memory in megabytes
    #[must_use]
    pub fn total_memory_mb(&self) -> u64 {
        self.total_memory / (1024 * 1024)
    }

    /// Get total memory in gigabytes
    #[must_use]
    pub fn total_memory_gb(&self) -> f64 {
        self.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

impl std::fmt::Display for CudaDeviceInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {} ({:.1} GB)", self.index, self.name, self.total_memory_gb())
    }
}

// ============================================================================
// CUDA Memory Information (TRUENO-SPEC-010 Section 4.1.2)
// ============================================================================

/// Real-time CUDA memory information from cuMemGetInfo
///
/// Provides current memory usage on the active CUDA context.
#[derive(Debug, Clone, Copy)]
pub struct CudaMemoryInfo {
    /// Free memory in bytes
    pub free: u64,
    /// Total memory in bytes
    pub total: u64,
}

impl CudaMemoryInfo {
    /// Query current memory information
    ///
    /// Requires an active CUDA context.
    ///
    /// # Errors
    ///
    /// Returns error if no context is active or query fails.
    #[cfg(feature = "cuda")]
    pub fn query(ctx: &CudaContext) -> Result<Self, GpuError> {
        let (free, total) = ctx.memory_info()?;
        Ok(Self { free: free as u64, total: total as u64 })
    }

    /// Get used memory in bytes
    #[must_use]
    pub fn used(&self) -> u64 {
        self.total.saturating_sub(self.free)
    }

    /// Get free memory in megabytes
    #[must_use]
    pub fn free_mb(&self) -> u64 {
        self.free / (1024 * 1024)
    }

    /// Get total memory in megabytes
    #[must_use]
    pub fn total_mb(&self) -> u64 {
        self.total / (1024 * 1024)
    }

    /// Get used memory in megabytes
    #[must_use]
    pub fn used_mb(&self) -> u64 {
        self.used() / (1024 * 1024)
    }

    /// Get memory usage percentage (0.0 - 100.0)
    #[must_use]
    pub fn usage_percent(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            (self.used() as f64 / self.total as f64) * 100.0
        }
    }
}

impl std::fmt::Display for CudaMemoryInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} / {} MB ({:.1}% used)", self.used_mb(), self.total_mb(), self.usage_percent())
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Check if CUDA monitoring is available
///
/// Returns `true` if CUDA driver is installed and at least one device exists.
#[must_use]
pub fn cuda_monitoring_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda_available()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get the number of CUDA devices
///
/// # Errors
///
/// Returns error if CUDA is not available.
pub fn cuda_device_count() -> Result<usize, GpuError> {
    #[cfg(feature = "cuda")]
    {
        device_count()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(GpuError::CudaNotAvailable("cuda feature not enabled".to_string()))
    }
}

// ============================================================================
// Tests (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests;
