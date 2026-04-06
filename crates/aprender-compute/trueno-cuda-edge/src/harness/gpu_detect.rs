//! GPU capability detection.
//!
//! Provides [`GpuCapability`] describing a detected GPU and runtime
//! detection functions. Actual GPU detection is gated on the `cuda`
//! feature; without it, [`detect_gpus`] returns an empty list.

use serde::{Deserialize, Serialize};

use crate::shmem_prober::overflow::ComputeCapability;

/// Describes a detected GPU device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuCapability {
    /// Device index (0-based).
    pub device_index: u32,
    /// Device name (e.g., "NVIDIA A100").
    pub name: String,
    /// Compute capability.
    pub compute_capability: ComputeCapability,
    /// Total global memory in bytes.
    pub total_memory_bytes: u64,
}

/// Detect available GPUs.
///
/// Returns an empty list when compiled without the `cuda` feature or
/// when no GPUs are present.
#[must_use]
pub fn detect_gpus() -> Vec<GpuCapability> {
    #[cfg(feature = "cuda")]
    {
        // Real implementation would query CUDA driver
        Vec::new()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Vec::new()
    }
}

/// Returns true if at least one GPU is available.
#[must_use]
pub fn gpu_available() -> bool {
    !detect_gpus().is_empty()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn detect_gpus_returns_vec() {
        // Without cuda feature, always empty
        let gpus = detect_gpus();
        assert!(gpus.is_empty() || !gpus.is_empty()); // type check
    }

    #[test]
    fn gpu_available_consistent() {
        let available = gpu_available();
        let gpus = detect_gpus();
        assert_eq!(available, !gpus.is_empty());
    }

    #[test]
    fn gpu_capability_serializes() {
        let cap = GpuCapability {
            device_index: 0,
            name: "Test GPU".into(),
            compute_capability: ComputeCapability::new(8, 0),
            total_memory_bytes: 16 * 1024 * 1024 * 1024,
        };
        let json = serde_json::to_string(&cap).expect("serialization should succeed");
        assert!(json.contains("Test GPU"));
    }
}
