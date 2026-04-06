//! Kernel checksum types for divergence detection.
//!
//! CORRECTNESS-011: Captures output checksum per kernel invocation.

use std::fmt;

/// Kernel checksum for divergence detection.
///
/// CORRECTNESS-011: Captures output checksum per kernel invocation.
#[derive(Debug, Clone)]
pub struct KernelChecksum {
    /// Kernel/brick name
    pub name: String,
    /// Layer index
    pub layer_idx: usize,
    /// Sequence position
    pub position: u32,
    /// FNV-1a checksum of first 64 output floats
    pub checksum: u64,
}

/// Information about a detected divergence between CPU and GPU.
#[derive(Debug, Clone)]
pub struct DivergenceInfo {
    /// Name of the divergent kernel
    pub kernel_name: String,
    /// Layer where divergence occurred
    pub layer_idx: usize,
    /// Position where divergence occurred
    pub position: u32,
    /// Expected checksum (from CPU/reference)
    pub expected_checksum: u64,
    /// Actual checksum (from GPU/test)
    pub actual_checksum: u64,
}

impl fmt::Display for DivergenceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DIVERGENCE at '{}' (layer {}, pos {}): expected 0x{:016X}, got 0x{:016X}",
            self.kernel_name,
            self.layer_idx,
            self.position,
            self.expected_checksum,
            self.actual_checksum
        )
    }
}

/// FNV-1a hash of f32 slice (first 64 elements for efficiency).
///
/// Used for quick divergence detection between CPU and GPU outputs.
#[inline]
pub fn fnv1a_f32_checksum(data: &[f32]) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;
    let len = data.len().min(64);
    for &val in &data[..len] {
        let bytes = val.to_le_bytes();
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }
    hash
}

/// Macro for convenient brick timing with automatic sync.
///
/// # Usage
///
/// ```rust,ignore
/// time_brick!(profiler, "RmsNorm", 1, {
///     rmsnorm_kernel.launch();
///     stream.synchronize(); // REQUIRED for GPU
/// });
/// ```
#[macro_export]
macro_rules! time_brick {
    ($profiler:expr, $name:expr, $elements:expr, $body:block) => {{
        let timer = $profiler.start($name);
        let result = $body;
        $profiler.stop(timer, $elements);
        result
    }};
}
