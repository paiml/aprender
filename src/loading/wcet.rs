//! Worst-Case Execution Time (WCET) Calculator
//!
//! Provides mathematical bounds for model loading time, required for
//! ISO 26262 ASIL-D and DO-178C Level A certification evidence.
//!
//! # References
//!
//! - [Wilhelm et al. 2008] "The worst-case execution-time problem"
//! - [Liu & Layland 1973] Real-time scheduling theory
//! - [Zstd Format Spec] Bounded decompression complexity
//! - [Collet 2016] Zstd compression algorithm
//!
//! # Toyota Way Alignment
//!
//! - **Genchi Genbutsu**: Platform specs must be characterized via hardware profiling
//! - **Jidoka**: Conservative 10% safety margin built into calculations

use std::time::Duration;

/// Platform-specific timing parameters for WCET calculation.
///
/// Must be characterized via hardware profiling (Genchi Genbutsu).
///
/// # Safety Note
///
/// These values represent **worst-case** performance. Actual execution
/// will typically be faster. Use these for real-time deadline guarantees.
#[derive(Debug, Clone, Copy)]
pub struct PlatformSpecs {
    /// Minimum guaranteed read speed from storage (MB/s)
    /// Account for worst-case fragmentation and bus contention
    pub min_read_speed_mbps: f64,

    /// Minimum decompression throughput (MB/s uncompressed output)
    /// Zstd has bounded worst-case
    pub min_decomp_speed_mbps: f64,

    /// CRC32 throughput using hardware acceleration (MB/s)
    /// Most ARM cores: ~2000 MB/s with CRC32 instruction
    pub crc32_throughput_mbps: f64,

    /// Ed25519 signature verification time (microseconds)
    /// Constant time regardless of message size
    pub ed25519_verify_us: f64,

    /// Bincode deserialization overhead (bytes/μs)
    pub deserialize_throughput_bps: f64,
}

impl PlatformSpecs {
    /// Create custom platform specs
    #[must_use]
    pub const fn new(
        min_read_speed_mbps: f64,
        min_decomp_speed_mbps: f64,
        crc32_throughput_mbps: f64,
        ed25519_verify_us: f64,
        deserialize_throughput_bps: f64,
    ) -> Self {
        Self {
            min_read_speed_mbps,
            min_decomp_speed_mbps,
            crc32_throughput_mbps,
            ed25519_verify_us,
            deserialize_throughput_bps,
        }
    }

    /// Get effective throughput (minimum of read and decompress)
    #[must_use]
    pub fn effective_throughput_mbps(&self) -> f64 {
        self.min_read_speed_mbps.min(self.min_decomp_speed_mbps)
    }
}

/// Pre-characterized platform specifications
pub mod platforms {
    use super::PlatformSpecs;

    /// Automotive-grade ECU (NXP S32G, Cortex-A53)
    pub const AUTOMOTIVE_S32G: PlatformSpecs = PlatformSpecs {
        min_read_speed_mbps: 50.0,     // eMMC worst-case
        min_decomp_speed_mbps: 200.0,  // Zstd level 3
        crc32_throughput_mbps: 2000.0, // Hardware CRC
        ed25519_verify_us: 800.0,      // Software implementation
        deserialize_throughput_bps: 500_000_000.0,
    };

    /// Aerospace flight computer (RAD750-class)
    pub const AEROSPACE_RAD750: PlatformSpecs = PlatformSpecs {
        min_read_speed_mbps: 10.0,    // Radiation-hardened flash
        min_decomp_speed_mbps: 50.0,  // Conservative estimate
        crc32_throughput_mbps: 100.0, // Software CRC
        ed25519_verify_us: 5000.0,    // No crypto acceleration
        deserialize_throughput_bps: 50_000_000.0,
    };

    /// Edge device (Raspberry Pi 4, Cortex-A72)
    pub const EDGE_RPI4: PlatformSpecs = PlatformSpecs {
        min_read_speed_mbps: 100.0,
        min_decomp_speed_mbps: 400.0,
        crc32_throughput_mbps: 3000.0,
        ed25519_verify_us: 200.0,
        deserialize_throughput_bps: 1_000_000_000.0,
    };

    /// Desktop/Server (x86-64, `NVMe` SSD)
    pub const DESKTOP_X86: PlatformSpecs = PlatformSpecs {
        min_read_speed_mbps: 500.0,   // NVMe worst-case
        min_decomp_speed_mbps: 800.0, // Zstd with AVX2
        crc32_throughput_mbps: 5000.0,
        ed25519_verify_us: 50.0,
        deserialize_throughput_bps: 2_000_000_000.0,
    };

    /// WebAssembly in browser
    pub const WASM_BROWSER: PlatformSpecs = PlatformSpecs {
        min_read_speed_mbps: 20.0,    // Network fetch
        min_decomp_speed_mbps: 100.0, // JS-based decompression
        crc32_throughput_mbps: 500.0,
        ed25519_verify_us: 1000.0,
        deserialize_throughput_bps: 200_000_000.0,
    };

    /// Industrial PLC (ARM Cortex-M7)
    pub const INDUSTRIAL_PLC: PlatformSpecs = PlatformSpecs {
        min_read_speed_mbps: 25.0,
        min_decomp_speed_mbps: 80.0,
        crc32_throughput_mbps: 800.0,
        ed25519_verify_us: 2000.0,
        deserialize_throughput_bps: 100_000_000.0,
    };
}

/// Header information for WCET calculation
#[derive(Debug, Clone, Copy)]
pub struct HeaderInfo {
    /// Compressed payload size in bytes
    pub compressed_size_bytes: u64,
    /// Uncompressed payload size in bytes
    pub uncompressed_size_bytes: u64,
    /// Total payload size for verification
    pub payload_size_bytes: u64,
    /// Whether the model is signed
    pub is_signed: bool,
    /// Model type identifier
    pub model_type: u16,
}

impl HeaderInfo {
    /// Create a new header info
    #[must_use]
    pub const fn new(
        compressed_size_bytes: u64,
        uncompressed_size_bytes: u64,
        is_signed: bool,
    ) -> Self {
        Self {
            compressed_size_bytes,
            uncompressed_size_bytes,
            payload_size_bytes: compressed_size_bytes,
            is_signed,
            model_type: 0,
        }
    }

    /// Get compression ratio
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        if self.compressed_size_bytes == 0 {
            1.0
        } else {
            self.uncompressed_size_bytes as f64 / self.compressed_size_bytes as f64
        }
    }
}

/// Calculates the theoretical Worst-Case Execution Time (WCET) for model loading.
///
/// Used for ISO 26262 ASIL-D and DO-178C Level A certification evidence.
///
/// # Safety Rationale (Jidoka)
///
/// This function provides a **conservative upper bound**. Actual execution
/// will typically be faster. The WCET is used to guarantee the system
/// meets real-time deadlines under all operating conditions.
///
/// # Formula
///
/// ```text
/// T_load = T_header + (S_comp / R_read) + (S_uncomp / R_decomp) + T_verify + T_deserialize
/// ```
///
/// With 10% safety margin (Toyota Way: conservative design)
///
/// # References
///
/// - [Wilhelm et al. 2008] WCET analysis overview
/// - [Liu & Layland 1973] Real-time scheduling theory
/// - [Zstd Format Spec] Bounded decompression complexity
#[must_use]
pub fn calculate_wcet(header: &HeaderInfo, platform: &PlatformSpecs) -> Duration {
    // 1. Header validation (constant time)
    let header_time_us = 1.0;

    // 2. Storage read latency (worst-case sequential read)
    let compressed_mb = header.compressed_size_bytes as f64 / (1024.0 * 1024.0);
    let read_time_us = (compressed_mb / platform.min_read_speed_mbps) * 1_000_000.0;

    // 3. Decompression latency (Zstd worst-case is strictly bounded)
    let uncompressed_mb = header.uncompressed_size_bytes as f64 / (1024.0 * 1024.0);
    let decomp_time_us = (uncompressed_mb / platform.min_decomp_speed_mbps) * 1_000_000.0;

    // 4. Integrity verification
    // CRC32 is always performed for data integrity
    let payload_mb = header.payload_size_bytes as f64 / (1024.0 * 1024.0);
    let crc32_time_us = (payload_mb / platform.crc32_throughput_mbps) * 1_000_000.0;

    let verify_time_us = if header.is_signed {
        // Signed models: CRC32 + Ed25519 signature verification
        crc32_time_us + platform.ed25519_verify_us
    } else {
        // Unsigned models: CRC32 only
        crc32_time_us
    };

    // 5. Deserialization overhead
    let deserialize_time_us =
        header.uncompressed_size_bytes as f64 / platform.deserialize_throughput_bps * 1_000_000.0;

    // Total WCET with 10% safety margin (Toyota Way: conservative design)
    let total_us =
        (header_time_us + read_time_us + decomp_time_us + verify_time_us + deserialize_time_us)
            * 1.1;

    Duration::from_micros(total_us.ceil() as u64)
}

/// Calculate breakdown of WCET components
#[must_use]
pub fn calculate_wcet_breakdown(header: &HeaderInfo, platform: &PlatformSpecs) -> WcetBreakdown {
    let compressed_mb = header.compressed_size_bytes as f64 / (1024.0 * 1024.0);
    let uncompressed_mb = header.uncompressed_size_bytes as f64 / (1024.0 * 1024.0);
    let payload_mb = header.payload_size_bytes as f64 / (1024.0 * 1024.0);

    let header_us = 1.0;
    let read_us = (compressed_mb / platform.min_read_speed_mbps) * 1_000_000.0;
    let decomp_us = (uncompressed_mb / platform.min_decomp_speed_mbps) * 1_000_000.0;

    // CRC32 is always performed for data integrity
    let crc32_us = (payload_mb / platform.crc32_throughput_mbps) * 1_000_000.0;
    let verify_us = if header.is_signed {
        // Signed models: CRC32 + Ed25519 signature verification
        crc32_us + platform.ed25519_verify_us
    } else {
        // Unsigned models: CRC32 only
        crc32_us
    };
    let deserialize_us =
        header.uncompressed_size_bytes as f64 / platform.deserialize_throughput_bps * 1_000_000.0;

    let total_us = (header_us + read_us + decomp_us + verify_us + deserialize_us) * 1.1;

    WcetBreakdown {
        header: Duration::from_micros(header_us.ceil() as u64),
        read: Duration::from_micros(read_us.ceil() as u64),
        decompress: Duration::from_micros(decomp_us.ceil() as u64),
        verify: Duration::from_micros(verify_us.ceil() as u64),
        deserialize: Duration::from_micros(deserialize_us.ceil() as u64),
        total: Duration::from_micros(total_us.ceil() as u64),
        safety_margin: 0.1,
    }
}

/// Breakdown of WCET components for analysis
#[derive(Debug, Clone)]
pub struct WcetBreakdown {
    /// Header validation time
    pub header: Duration,
    /// Storage read time
    pub read: Duration,
    /// Decompression time
    pub decompress: Duration,
    /// Verification time
    pub verify: Duration,
    /// Deserialization time
    pub deserialize: Duration,
    /// Total time (with safety margin)
    pub total: Duration,
    /// Safety margin applied
    pub safety_margin: f64,
}

impl WcetBreakdown {
    /// Get the dominant component
    #[must_use]
    pub fn dominant_component(&self) -> &'static str {
        let max = self
            .header
            .max(self.read)
            .max(self.decompress)
            .max(self.verify)
            .max(self.deserialize);
        if max == self.read {
            "Storage I/O"
        } else if max == self.decompress {
            "Decompression"
        } else if max == self.verify {
            "Verification"
        } else if max == self.deserialize {
            "Deserialization"
        } else {
            "Header validation"
        }
    }

    /// Get percentage of time spent in each component
    #[must_use]
    pub fn percentages(&self) -> WcetPercentages {
        let base_total = self.total.as_micros() as f64 / 1.1; // Remove safety margin
        WcetPercentages {
            header: self.header.as_micros() as f64 / base_total * 100.0,
            read: self.read.as_micros() as f64 / base_total * 100.0,
            decompress: self.decompress.as_micros() as f64 / base_total * 100.0,
            verify: self.verify.as_micros() as f64 / base_total * 100.0,
            deserialize: self.deserialize.as_micros() as f64 / base_total * 100.0,
        }
    }
}

/// Percentage breakdown of WCET components
#[derive(Debug, Clone)]
pub struct WcetPercentages {
    /// Header validation percentage
    pub header: f64,
    /// Storage read percentage
    pub read: f64,
    /// Decompression percentage
    pub decompress: f64,
    /// Verification percentage
    pub verify: f64,
    /// Deserialization percentage
    pub deserialize: f64,
}

/// Estimate maximum model size that fits within time budget
#[must_use]
pub fn estimate_max_size_for_budget(platform: &PlatformSpecs, budget: Duration) -> usize {
    let budget_us = budget.as_micros() as f64;
    let effective_throughput = platform.effective_throughput_mbps();
    let max_mb = (budget_us / 1_000_000.0) * effective_throughput * 0.8; // 80% utilization
    (max_mb * 1024.0 * 1024.0) as usize
}

/// Calculate minimum ring buffer size for jitter-free streaming
///
/// # Formula
///
/// ```text
/// B >= (R_decomp_max - R_consume_min) × T_window
/// ```
#[must_use]
pub fn min_ring_buffer_size(decomp_max_mbps: f64, consume_min_mbps: f64, window_ms: f64) -> usize {
    let rate_diff_mbps = decomp_max_mbps - consume_min_mbps;
    if rate_diff_mbps <= 0.0 {
        // Consumer is always faster; minimal buffer needed
        return 64 * 1024; // 64KB minimum
    }

    let buffer_mb = rate_diff_mbps * (window_ms / 1000.0);
    let buffer_bytes = (buffer_mb * 1024.0 * 1024.0).ceil() as usize;

    // Round up to page boundary (4KB alignment)
    ((buffer_bytes + 4095) / 4096) * 4096
}

/// Safety error for time/memory budget violations
#[derive(Debug, Clone)]
pub enum SafetyError {
    /// Loading time exceeds budget
    TimeBudgetExceeded {
        /// Allocated time budget
        budget: Duration,
        /// Calculated worst-case time
        worst_case: Duration,
        /// Model type
        model_type: u16,
        /// Compressed size
        compressed_size: u64,
        /// Recommendation
        recommendation: String,
    },
    /// Memory usage exceeds budget
    MemoryBudgetExceeded {
        /// Allocated memory budget
        budget: usize,
        /// Required memory
        required: usize,
    },
    /// Integrity check failed
    IntegrityCheckFailed {
        /// Expected checksum
        expected: u32,
        /// Computed checksum
        computed: u32,
    },
}

impl std::fmt::Display for SafetyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TimeBudgetExceeded {
                budget,
                worst_case,
                recommendation,
                ..
            } => {
                write!(
                    f,
                    "Time budget exceeded: budget={budget:?}, worst_case={worst_case:?}. {recommendation}"
                )
            }
            Self::MemoryBudgetExceeded { budget, required } => {
                write!(
                    f,
                    "Memory budget exceeded: budget={budget} bytes, required={required} bytes"
                )
            }
            Self::IntegrityCheckFailed { expected, computed } => {
                write!(
                    f,
                    "Integrity check failed: expected=0x{expected:08X}, computed=0x{computed:08X}"
                )
            }
        }
    }
}

impl std::error::Error for SafetyError {}

/// Runtime assertion for time budget compliance (Jidoka)
///
/// If the model cannot be loaded within the time budget, this function
/// returns an error immediately (stop-the-line).
pub fn assert_time_budget(
    header: &HeaderInfo,
    platform: &PlatformSpecs,
    budget: Duration,
) -> Result<(), SafetyError> {
    let worst_case = calculate_wcet(header, platform);

    if worst_case > budget {
        return Err(SafetyError::TimeBudgetExceeded {
            budget,
            worst_case,
            model_type: header.model_type,
            compressed_size: header.compressed_size_bytes,
            recommendation: format!(
                "Reduce model size below {} bytes or use faster storage",
                estimate_max_size_for_budget(platform, budget)
            ),
        });
    }

    Ok(())
}

/// Runtime assertion for memory budget compliance (Heijunka)
pub fn assert_memory_budget(required: usize, budget: usize) -> Result<(), SafetyError> {
    if required > budget {
        return Err(SafetyError::MemoryBudgetExceeded { budget, required });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_specs_effective_throughput() {
        let specs = PlatformSpecs::new(100.0, 200.0, 1000.0, 100.0, 1_000_000.0);
        assert!((specs.effective_throughput_mbps() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_header_info_compression_ratio() {
        let header = HeaderInfo::new(100, 400, false);
        assert!((header.compression_ratio() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_header_info_compression_ratio_zero() {
        let header = HeaderInfo::new(0, 400, false);
        assert!((header.compression_ratio() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_calculate_wcet_small_model() {
        let header = HeaderInfo::new(1024 * 1024, 2 * 1024 * 1024, false);
        let wcet = calculate_wcet(&header, &platforms::DESKTOP_X86);

        // Should be in millisecond range for small model on desktop
        assert!(wcet.as_millis() < 100);
    }

    #[test]
    fn test_calculate_wcet_signed_model() {
        let header = HeaderInfo::new(1024 * 1024, 2 * 1024 * 1024, true);
        let wcet_signed = calculate_wcet(&header, &platforms::DESKTOP_X86);

        let header_unsigned = HeaderInfo::new(1024 * 1024, 2 * 1024 * 1024, false);
        let wcet_unsigned = calculate_wcet(&header_unsigned, &platforms::DESKTOP_X86);

        // Signed should take longer due to Ed25519 verification
        assert!(wcet_signed >= wcet_unsigned);
    }

    #[test]
    fn test_calculate_wcet_platform_comparison() {
        let header = HeaderInfo::new(10 * 1024 * 1024, 20 * 1024 * 1024, false);

        let wcet_desktop = calculate_wcet(&header, &platforms::DESKTOP_X86);
        let wcet_aerospace = calculate_wcet(&header, &platforms::AEROSPACE_RAD750);

        // Aerospace should be much slower due to rad-hardened components
        assert!(wcet_aerospace > wcet_desktop * 5);
    }

    #[test]
    fn test_calculate_wcet_breakdown() {
        let header = HeaderInfo::new(1024 * 1024, 2 * 1024 * 1024, false);
        let breakdown = calculate_wcet_breakdown(&header, &platforms::DESKTOP_X86);

        assert!(breakdown.total > Duration::ZERO);
        assert!(!breakdown.dominant_component().is_empty());

        let percentages = breakdown.percentages();
        let total_pct = percentages.header
            + percentages.read
            + percentages.decompress
            + percentages.verify
            + percentages.deserialize;
        assert!((total_pct - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_estimate_max_size_for_budget() {
        let budget = Duration::from_millis(100);
        let max_size = estimate_max_size_for_budget(&platforms::DESKTOP_X86, budget);

        // Desktop should support several MB in 100ms
        assert!(max_size > 1024 * 1024);
    }

    #[test]
    fn test_min_ring_buffer_size() {
        // Consumer faster than producer - minimal buffer
        let size = min_ring_buffer_size(100.0, 200.0, 10.0);
        assert_eq!(size, 64 * 1024);

        // Producer faster - need larger buffer
        let size = min_ring_buffer_size(200.0, 100.0, 10.0);
        assert!(size > 64 * 1024);

        // Should be page-aligned
        assert_eq!(size % 4096, 0);
    }

    #[test]
    fn test_assert_time_budget_success() {
        let header = HeaderInfo::new(1024, 2048, false);
        let result = assert_time_budget(&header, &platforms::DESKTOP_X86, Duration::from_secs(1));
        assert!(result.is_ok());
    }

    #[test]
    fn test_assert_time_budget_failure() {
        let header = HeaderInfo::new(1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024, false);
        let result = assert_time_budget(
            &header,
            &platforms::AEROSPACE_RAD750,
            Duration::from_micros(1),
        );
        assert!(result.is_err());

        if let Err(SafetyError::TimeBudgetExceeded { recommendation, .. }) = result {
            assert!(!recommendation.is_empty());
        }
    }

    #[test]
    fn test_assert_memory_budget() {
        assert!(assert_memory_budget(100, 200).is_ok());
        assert!(assert_memory_budget(300, 200).is_err());
    }

    #[test]
    fn test_safety_error_display() {
        let err = SafetyError::TimeBudgetExceeded {
            budget: Duration::from_millis(10),
            worst_case: Duration::from_millis(100),
            model_type: 1,
            compressed_size: 1024,
            recommendation: "Test recommendation".to_string(),
        };
        let display = format!("{}", err);
        assert!(display.contains("Time budget exceeded"));

        let err = SafetyError::IntegrityCheckFailed {
            expected: 0x12345678,
            computed: 0xABCDEF01,
        };
        let display = format!("{}", err);
        assert!(display.contains("Integrity check failed"));
    }

    #[test]
    fn test_pre_characterized_platforms() {
        // Ensure all platforms have sensible values
        assert!(platforms::AUTOMOTIVE_S32G.min_read_speed_mbps > 0.0);
        assert!(platforms::AEROSPACE_RAD750.min_read_speed_mbps > 0.0);
        assert!(platforms::EDGE_RPI4.min_read_speed_mbps > 0.0);
        assert!(platforms::DESKTOP_X86.min_read_speed_mbps > 0.0);
        assert!(platforms::WASM_BROWSER.min_read_speed_mbps > 0.0);
        assert!(platforms::INDUSTRIAL_PLC.min_read_speed_mbps > 0.0);
    }
}
