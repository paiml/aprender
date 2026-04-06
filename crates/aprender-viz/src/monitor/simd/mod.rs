//! SIMD-accelerated kernels for metric collection.
//!
//! This module provides vectorized operations for high-performance metric parsing
//! and processing. All operations are designed for sub-millisecond latency.
//!
//! ## Design Philosophy
//!
//! Following Toyota Way principles (Kaizen, Jidoka) and Popperian falsificationism,
//! each kernel includes:
//! - Explicit performance targets (falsifiable hypotheses)
//! - Automatic fallback to scalar paths
//! - Zero-allocation hot paths
//!
//! ## References
//!
//! - Lemire & Langdale (2019): simdjson parsing techniques
//! - Polychroniou et al. (2015): SIMD for in-memory databases

pub mod compressed;
pub mod correlation;
pub mod kernels;
pub mod ring_buffer;
pub mod soa;
pub mod timeseries;

#[cfg(test)]
mod integration_tests;

pub use compressed::{CompressedBlock, CompressedMetricStore, Timestamp};
pub use correlation::{
    simd_correlation_matrix, simd_cross_correlation, simd_pearson_correlation, top_correlations,
    CorrelationResult, CorrelationStrength, CorrelationTracker,
};
pub use kernels::*;
pub use ring_buffer::{ReductionOp, SimdRingBuffer};
pub use soa::*;
pub use timeseries::{
    Aggregations, QueryResult, TableStats, TierConfig, TimeSeriesDb, TimeSeriesTable,
};

/// SIMD alignment constant (64 bytes for AVX-512 compatibility).
pub const SIMD_ALIGNMENT: usize = 64;

/// Maximum number of CPU cores supported for SoA structures.
pub const MAX_CORES: usize = 256;

/// Maximum number of network interfaces.
pub const MAX_INTERFACES: usize = 64;

/// Maximum number of disks.
pub const MAX_DISKS: usize = 128;

/// Backend selection for SIMD operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackend {
    /// Scalar fallback (no SIMD).
    Scalar,
    /// SSE2 (128-bit, x86_64 baseline).
    Sse2,
    /// AVX2 (256-bit, Haswell 2013+).
    Avx2,
    /// AVX-512 (512-bit, Skylake-X 2017+).
    Avx512,
    /// ARM NEON (128-bit).
    Neon,
    /// WebAssembly SIMD128.
    WasmSimd128,
}

impl SimdBackend {
    /// Detects the best available SIMD backend for the current platform.
    #[must_use]
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return Self::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return Self::Avx2;
            }
            if is_x86_feature_detected!("sse2") {
                return Self::Sse2;
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is mandatory on AArch64
            return Self::Neon;
        }

        #[cfg(target_arch = "wasm32")]
        {
            #[cfg(target_feature = "simd128")]
            return Self::WasmSimd128;
        }

        Self::Scalar
    }

    /// Returns the register width in bits.
    #[must_use]
    pub const fn register_width_bits(&self) -> usize {
        match self {
            Self::Scalar => 64,
            Self::Sse2 | Self::Neon | Self::WasmSimd128 => 128,
            Self::Avx2 => 256,
            Self::Avx512 => 512,
        }
    }

    /// Returns the number of u64 values processed per SIMD operation.
    #[must_use]
    pub const fn u64_lanes(&self) -> usize {
        self.register_width_bits() / 64
    }

    /// Returns the number of f64 values processed per SIMD operation.
    #[must_use]
    pub const fn f64_lanes(&self) -> usize {
        self.register_width_bits() / 64
    }
}

/// Statistics computed via SIMD operations.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct SimdStats {
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Sum of all values.
    pub sum: f64,
    /// Sum of squared values (for variance).
    pub sum_sq: f64,
    /// Count of values.
    pub count: u64,
    /// Padding to 64 bytes.
    _padding: [u8; 24],
}

impl SimdStats {
    /// Creates new empty statistics.
    #[must_use]
    pub const fn new() -> Self {
        Self { min: f64::MAX, max: f64::MIN, sum: 0.0, sum_sq: 0.0, count: 0, _padding: [0; 24] }
    }

    /// Updates statistics with a single value.
    #[inline]
    pub fn update(&mut self, value: f64) {
        self.min = self.min.min(value);
        self.max = self.max.max(value);
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;
    }

    /// Returns the mean value.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    /// Returns the variance.
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            let mean = self.mean();
            (self.sum_sq / self.count as f64) - (mean * mean)
        }
    }

    /// Returns the standard deviation.
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Resets all statistics.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Merges another SimdStats into this one (for parallel reduction).
    pub fn merge(&mut self, other: &Self) {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.sum += other.sum;
        self.sum_sq += other.sum_sq;
        self.count += other.count;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        let backend = SimdBackend::detect();
        // Should at least return Scalar
        assert!(backend.u64_lanes() >= 1);
    }

    #[test]
    fn test_simd_stats() {
        let mut stats = SimdStats::new();
        for i in 1..=10 {
            stats.update(f64::from(i));
        }
        assert_eq!(stats.count, 10);
        assert!((stats.mean() - 5.5).abs() < 0.001);
        assert!((stats.min - 1.0).abs() < 0.001);
        assert!((stats.max - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_simd_stats_merge() {
        let mut stats1 = SimdStats::new();
        let mut stats2 = SimdStats::new();

        for i in 1..=5 {
            stats1.update(f64::from(i));
        }
        for i in 6..=10 {
            stats2.update(f64::from(i));
        }

        stats1.merge(&stats2);
        assert_eq!(stats1.count, 10);
        assert!((stats1.min - 1.0).abs() < 0.001);
        assert!((stats1.max - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_alignment() {
        assert_eq!(std::mem::align_of::<SimdStats>(), 64);
        assert_eq!(std::mem::size_of::<SimdStats>(), 64);
    }
}
