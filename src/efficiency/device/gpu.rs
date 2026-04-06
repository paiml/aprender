//! GPU information and detection.

use serde::{Deserialize, Serialize};

/// GPU information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU name/model
    pub name: String,
    /// Video RAM in bytes
    pub vram_bytes: u64,
    /// CUDA compute capability (major, minor) if NVIDIA
    pub compute_capability: Option<(u32, u32)>,
    /// GPU index (for multi-GPU systems)
    pub index: u32,
}

impl GpuInfo {
    /// Create new GPU info
    pub fn new(name: impl Into<String>, vram_bytes: u64) -> Self {
        Self { name: name.into(), vram_bytes, compute_capability: None, index: 0 }
    }

    /// Set CUDA compute capability
    pub fn with_compute_capability(mut self, major: u32, minor: u32) -> Self {
        self.compute_capability = Some((major, minor));
        self
    }

    /// Set GPU index
    pub fn with_index(mut self, index: u32) -> Self {
        self.index = index;
        self
    }

    /// Check if GPU supports specific CUDA compute capability
    pub fn supports_compute_capability(&self, major: u32, minor: u32) -> bool {
        self.compute_capability.is_some_and(|(m, n)| m > major || (m == major && n >= minor))
    }

    /// Get VRAM in GB
    pub fn vram_gb(&self) -> f64 {
        self.vram_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}
