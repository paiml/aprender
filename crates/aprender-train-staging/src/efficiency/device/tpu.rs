//! TPU information and detection.

use serde::{Deserialize, Serialize};

/// TPU information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TpuInfo {
    /// TPU version (e.g., "v4", "v5e")
    pub version: String,
    /// Number of TPU cores
    pub cores: u32,
    /// High bandwidth memory in bytes
    pub hbm_bytes: u64,
}

impl TpuInfo {
    /// Create new TPU info
    pub fn new(version: impl Into<String>, cores: u32, hbm_bytes: u64) -> Self {
        Self { version: version.into(), cores, hbm_bytes }
    }

    /// Get HBM in GB
    pub fn hbm_gb(&self) -> f64 {
        self.hbm_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}
