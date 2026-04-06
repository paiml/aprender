//! Apple Silicon information and detection.

use serde::{Deserialize, Serialize};

/// Apple Silicon information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AppleSiliconInfo {
    /// Chip model (e.g., "M1", "M2 Pro", "M3 Max")
    pub chip: String,
    /// Performance cores
    pub p_cores: u32,
    /// Efficiency cores
    pub e_cores: u32,
    /// GPU cores
    pub gpu_cores: u32,
    /// Neural Engine cores
    pub neural_cores: u32,
    /// Unified memory in bytes
    pub unified_memory_bytes: u64,
}

impl AppleSiliconInfo {
    /// Create new Apple Silicon info
    pub fn new(chip: impl Into<String>) -> Self {
        Self {
            chip: chip.into(),
            p_cores: 0,
            e_cores: 0,
            gpu_cores: 0,
            neural_cores: 16, // Default for most Apple Silicon
            unified_memory_bytes: 0,
        }
    }

    /// Set core configuration
    pub fn with_cores(mut self, p_cores: u32, e_cores: u32, gpu_cores: u32) -> Self {
        self.p_cores = p_cores;
        self.e_cores = e_cores;
        self.gpu_cores = gpu_cores;
        self
    }

    /// Set unified memory
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.unified_memory_bytes = bytes;
        self
    }

    /// Get total CPU cores
    pub fn total_cpu_cores(&self) -> u32 {
        self.p_cores + self.e_cores
    }

    /// Get unified memory in GB
    pub fn unified_memory_gb(&self) -> f64 {
        self.unified_memory_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
    }

    /// Detect Apple Silicon information (macOS only)
    #[cfg(target_os = "macos")]
    pub fn detect() -> Option<Self> {
        // Check if we're on Apple Silicon
        let arch = std::env::consts::ARCH;
        if arch != "aarch64" {
            return None;
        }

        // Get chip name from sysctl
        let chip = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Apple Silicon".to_string());

        // Get core counts
        let p_cores = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel0.logicalcpu"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        let e_cores = std::process::Command::new("sysctl")
            .args(["-n", "hw.perflevel1.logicalcpu"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        // Get memory
        let memory = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()
            .and_then(|output| String::from_utf8(output.stdout).ok())
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        Some(
            Self::new(chip)
                .with_cores(p_cores, e_cores, 0) // GPU cores harder to detect
                .with_memory(memory),
        )
    }

    #[cfg(not(target_os = "macos"))]
    pub fn detect() -> Option<Self> {
        None
    }
}
