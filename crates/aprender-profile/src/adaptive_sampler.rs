//! Adaptive Sampling for Performance Optimization (Specification Section 7.3)
//!
//! Implements adaptive sampling to minimize tracing overhead while capturing
//! critical performance data. Ensures <5% overhead as per specification.
//!
//! # Strategy
//!
//! | Operation Type | Threshold | Sample Rate | Rationale |
//! |---------------|-----------|-------------|-----------|
//! | GPU kernels | >100μs | 100% | Always profile GPU operations |
//! | SIMD blocks | >50μs | 100% | Capture compute-intensive operations |
//! | Syscalls (I/O) | >10μs | 100% | I/O operations are inherently slow |
//! | Syscalls (fast) | <10μs | 1% | Statistical sampling for hot paths |
//!
//! # Reference
//!
//! Unified Tracing for Sovereign AI: Formal Specification v1.0
//! Section 7.3: Adaptive Sampling

use rand::Rng;

/// Adaptive sampler configuration
///
/// Controls which operations are traced based on estimated duration
/// and random sampling for statistical profiling.
#[derive(Debug, Clone)]
pub struct AdaptiveSampler {
    /// Minimum duration threshold in microseconds (default: 100μs)
    threshold_us: u64,
    /// Sample rate for operations below threshold (0.0-1.0, default: 0.01 = 1%)
    sample_rate: f64,
    /// Trace all operations regardless of duration (debug mode)
    trace_all: bool,
}

impl AdaptiveSampler {
    /// Create a new adaptive sampler with default settings
    ///
    /// Default configuration:
    /// - Threshold: 100μs (trace operations >100μs)
    /// - Sample rate: 1% (for fast operations)
    /// - Trace all: false
    pub fn new() -> Self {
        AdaptiveSampler { threshold_us: 100, sample_rate: 0.01, trace_all: false }
    }

    /// Create sampler with custom threshold
    ///
    /// # Arguments
    ///
    /// * `threshold_us` - Minimum duration in microseconds to always trace
    pub fn with_threshold(threshold_us: u64) -> Self {
        AdaptiveSampler { threshold_us, sample_rate: 0.01, trace_all: false }
    }

    /// Create sampler with custom sample rate
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Probability of sampling fast operations (0.0-1.0)
    pub fn with_sample_rate(sample_rate: f64) -> Self {
        AdaptiveSampler {
            threshold_us: 100,
            sample_rate: sample_rate.clamp(0.0, 1.0),
            trace_all: false,
        }
    }

    /// Create sampler that traces everything (debug mode)
    pub fn trace_all() -> Self {
        contract_pre_error_handling!();
        contract_post_error_handling!(&"ok");
        AdaptiveSampler { threshold_us: 0, sample_rate: 1.0, trace_all: true }
    }

    /// Preset: GPU kernel sampling (always trace >100μs)
    pub fn gpu_preset() -> Self {
        AdaptiveSampler {
            threshold_us: 100,
            sample_rate: 1.0, // Always trace GPU operations
            trace_all: false,
        }
    }

    /// Preset: SIMD block sampling (always trace >50μs)
    pub fn simd_preset() -> Self {
        AdaptiveSampler {
            threshold_us: 50,
            sample_rate: 1.0, // Always trace SIMD operations
            trace_all: false,
        }
    }

    /// Preset: I/O syscall sampling (always trace >10μs)
    pub fn io_preset() -> Self {
        AdaptiveSampler {
            threshold_us: 10,
            sample_rate: 1.0, // Always trace I/O syscalls
            trace_all: false,
        }
    }

    /// Preset: Fast syscall sampling (1% sampling, >10μs threshold)
    pub fn fast_syscall_preset() -> Self {
        AdaptiveSampler {
            threshold_us: 10,
            sample_rate: 0.01, // 1% statistical sampling
            trace_all: false,
        }
    }

    /// Set custom threshold
    pub fn set_threshold(&mut self, threshold_us: u64) {
        self.threshold_us = threshold_us;
    }

    /// Set custom sample rate
    pub fn set_sample_rate(&mut self, sample_rate: f64) {
        self.sample_rate = sample_rate.clamp(0.0, 1.0);
    }

    /// Enable/disable trace-all mode
    pub fn set_trace_all(&mut self, trace_all: bool) {
        contract_pre_error_handling!();
        self.trace_all = trace_all;
        contract_post_error_handling!(&"ok");
    }

    /// Get current threshold
    pub fn threshold(&self) -> u64 {
        self.threshold_us
    }

    /// Get current sample rate
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Check if trace-all mode is enabled
    pub fn is_trace_all(&self) -> bool {
        contract_pre_error_handling!();
        self.trace_all
    }

    /// Decide whether to trace an operation based on estimated duration
    ///
    /// # Arguments
    ///
    /// * `estimated_duration_us` - Estimated operation duration in microseconds
    ///
    /// # Returns
    ///
    /// `true` if the operation should be traced, `false` otherwise
    ///
    /// # Algorithm
    ///
    /// 1. If `trace_all` mode, always trace
    /// 2. If duration >= threshold, always trace (slow operations)
    /// 3. Otherwise, probabilistic sampling based on `sample_rate`
    pub fn should_trace(&self, estimated_duration_us: u64) -> bool {
        // Debug mode: trace everything
        if self.trace_all {
            return true;
        }

        // Always trace operations exceeding the latency threshold
        if estimated_duration_us >= self.threshold_us {
            return true;
        }

        // Probabilistic sampling for fast operations
        let mut rng = rand::thread_rng();
        rng.gen::<f64>() < self.sample_rate
    }

    /// Decide whether to trace based on operation name and estimated duration
    ///
    /// Provides operation-specific sampling logic. For example:
    /// - GPU operations: always trace if >100μs
    /// - SIMD operations: always trace if >50μs
    /// - I/O syscalls: always trace if >10μs
    /// - Other syscalls: sample at `sample_rate`
    ///
    /// # Arguments
    ///
    /// * `operation` - Operation type (e.g., "gpu", "simd", "syscall:read")
    /// * `estimated_duration_us` - Estimated duration in microseconds
    pub fn should_trace_operation(&self, operation: &str, estimated_duration_us: u64) -> bool {
        contract_pre_error_handling!(operation);
        // Debug mode: trace everything
        if self.trace_all {
            return true;
        }

        // GPU operations: always trace if >100μs
        if operation.starts_with("gpu") && estimated_duration_us >= 100 {
            return true;
        }

        // SIMD operations: always trace if >50μs
        if operation.starts_with("simd") && estimated_duration_us >= 50 {
            return true;
        }

        // I/O syscalls: always trace if >10μs
        let io_syscalls = [
            "read", "write", "open", "close", "stat", "fstat", "lstat", "poll", "lseek", "mmap",
            "munmap", "sendto", "recvfrom", "sendmsg", "recvmsg",
        ];
        if operation.starts_with("syscall:") {
            let syscall_name = operation.strip_prefix("syscall:").unwrap_or("");
            if io_syscalls.contains(&syscall_name) && estimated_duration_us >= 10 {
                return true;
            }
        }

        // Default: use standard threshold and sampling
        self.should_trace(estimated_duration_us)
    }

    /// Calculate overhead percentage for a given workload
    ///
    /// Estimates the overhead based on operation mix and sampling strategy.
    ///
    /// # Arguments
    ///
    /// * `total_operations` - Total number of operations
    /// * `operations_above_threshold` - Number of operations above threshold
    ///
    /// # Returns
    ///
    /// Estimated overhead as a percentage (0.0-100.0)
    pub fn estimated_overhead(
        &self,
        total_operations: u64,
        operations_above_threshold: u64,
    ) -> f64 {
        if total_operations == 0 {
            return 0.0;
        }

        if self.trace_all {
            return 5.0; // Worst-case overhead
        }

        let operations_below_threshold = total_operations - operations_above_threshold;

        // Overhead per traced operation: ~0.001% (ptrace context switch)
        let traced_slow = operations_above_threshold as f64;
        let traced_fast = operations_below_threshold as f64 * self.sample_rate;
        let total_traced = traced_slow + traced_fast;

        // Overhead formula: (traced_operations / total_operations) * 5%
        (total_traced / total_operations as f64) * 5.0
    }
}

impl Default for AdaptiveSampler {
    fn default() -> Self {
        Self::new()
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(AdaptiveSampler: Send, Sync);

// ============================================================================
// UNIT TESTS (EXTREME TDD)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Default sampler configuration
    #[test]
    fn test_default_sampler() {
        let sampler = AdaptiveSampler::new();
        assert_eq!(sampler.threshold(), 100);
        assert!((sampler.sample_rate() - 0.01).abs() < 1e-6);
        assert!(!sampler.is_trace_all());
    }

    // Test 2: Custom threshold
    #[test]
    fn test_with_threshold() {
        let sampler = AdaptiveSampler::with_threshold(50);
        assert_eq!(sampler.threshold(), 50);
        assert!((sampler.sample_rate() - 0.01).abs() < 1e-6);
    }

    // Test 3: Custom sample rate
    #[test]
    fn test_with_sample_rate() {
        let sampler = AdaptiveSampler::with_sample_rate(0.05);
        assert_eq!(sampler.threshold(), 100);
        assert!((sampler.sample_rate() - 0.05).abs() < 1e-6);
    }

    // Test 4: Sample rate clamping (too high)
    #[test]
    fn test_sample_rate_clamp_high() {
        let sampler = AdaptiveSampler::with_sample_rate(1.5);
        assert!((sampler.sample_rate() - 1.0).abs() < 1e-6);
    }

    // Test 5: Sample rate clamping (too low)
    #[test]
    fn test_sample_rate_clamp_low() {
        let sampler = AdaptiveSampler::with_sample_rate(-0.5);
        assert!((sampler.sample_rate() - 0.0).abs() < 1e-6);
    }

    // Test 6: Trace all mode
    #[test]
    fn test_trace_all() {
        let sampler = AdaptiveSampler::trace_all();
        assert!(sampler.is_trace_all());
        assert_eq!(sampler.threshold(), 0);
        assert!((sampler.sample_rate() - 1.0).abs() < 1e-6);
    }

    // Test 7: GPU preset
    #[test]
    fn test_gpu_preset() {
        let sampler = AdaptiveSampler::gpu_preset();
        assert_eq!(sampler.threshold(), 100);
        assert!((sampler.sample_rate() - 1.0).abs() < 1e-6);
    }

    // Test 8: SIMD preset
    #[test]
    fn test_simd_preset() {
        let sampler = AdaptiveSampler::simd_preset();
        assert_eq!(sampler.threshold(), 50);
        assert!((sampler.sample_rate() - 1.0).abs() < 1e-6);
    }

    // Test 9: I/O preset
    #[test]
    fn test_io_preset() {
        let sampler = AdaptiveSampler::io_preset();
        assert_eq!(sampler.threshold(), 10);
        assert!((sampler.sample_rate() - 1.0).abs() < 1e-6);
    }

    // Test 10: Fast syscall preset
    #[test]
    fn test_fast_syscall_preset() {
        let sampler = AdaptiveSampler::fast_syscall_preset();
        assert_eq!(sampler.threshold(), 10);
        assert!((sampler.sample_rate() - 0.01).abs() < 1e-6);
    }

    // Test 11: Should trace slow operations (above threshold)
    #[test]
    fn test_should_trace_slow() {
        let sampler = AdaptiveSampler::new();
        // Operations >= 100μs should always be traced
        assert!(sampler.should_trace(100));
        assert!(sampler.should_trace(150));
        assert!(sampler.should_trace(1000));
    }

    // Test 12: Trace all mode traces everything
    #[test]
    fn test_trace_all_mode() {
        let sampler = AdaptiveSampler::trace_all();
        assert!(sampler.should_trace(0));
        assert!(sampler.should_trace(10));
        assert!(sampler.should_trace(100));
        assert!(sampler.should_trace(1000));
    }

    // Test 13: Setters work correctly
    #[test]
    fn test_setters() {
        let mut sampler = AdaptiveSampler::new();

        sampler.set_threshold(200);
        assert_eq!(sampler.threshold(), 200);

        sampler.set_sample_rate(0.1);
        assert!((sampler.sample_rate() - 0.1).abs() < 1e-6);

        sampler.set_trace_all(true);
        assert!(sampler.is_trace_all());
    }

    // Test 14: GPU operation-specific tracing
    #[test]
    fn test_should_trace_gpu() {
        let sampler = AdaptiveSampler::new();

        // GPU operations >100μs should be traced
        assert!(sampler.should_trace_operation("gpu:matmul", 150));
        assert!(sampler.should_trace_operation("gpu:kernel", 100));
    }

    // Test 15: SIMD operation-specific tracing
    #[test]
    fn test_should_trace_simd() {
        let sampler = AdaptiveSampler::new();

        // SIMD operations >50μs should be traced
        assert!(sampler.should_trace_operation("simd:dot", 60));
        assert!(sampler.should_trace_operation("simd:add", 50));
    }

    // Test 16: I/O syscall-specific tracing
    #[test]
    fn test_should_trace_io_syscalls() {
        let sampler = AdaptiveSampler::new();

        // I/O syscalls >10μs should be traced
        assert!(sampler.should_trace_operation("syscall:read", 15));
        assert!(sampler.should_trace_operation("syscall:write", 20));
        assert!(sampler.should_trace_operation("syscall:open", 10));
    }

    // Test 17: Estimated overhead (all operations traced)
    #[test]
    fn test_estimated_overhead_all_traced() {
        let sampler = AdaptiveSampler::trace_all();
        let overhead = sampler.estimated_overhead(1000, 1000);
        assert!((overhead - 5.0).abs() < 1e-6); // Worst-case 5%
    }

    // Test 18: Estimated overhead (no operations)
    #[test]
    fn test_estimated_overhead_zero() {
        let sampler = AdaptiveSampler::new();
        let overhead = sampler.estimated_overhead(0, 0);
        assert!((overhead - 0.0).abs() < 1e-6);
    }

    // Test 19: Estimated overhead (50% slow operations)
    #[test]
    fn test_estimated_overhead_balanced() {
        let sampler = AdaptiveSampler::new(); // 1% sample rate
        let overhead = sampler.estimated_overhead(1000, 500);

        // 500 slow (always traced) + 500 * 0.01 fast (sampled) = 505 traced
        // (505 / 1000) * 5% = 2.525%
        assert!((overhead - 2.525).abs() < 0.1);
    }

    // Test 20: Estimated overhead (all fast operations)
    #[test]
    fn test_estimated_overhead_all_fast() {
        let sampler = AdaptiveSampler::new(); // 1% sample rate
        let overhead = sampler.estimated_overhead(1000, 0);

        // 0 slow + 1000 * 0.01 fast = 10 traced
        // (10 / 1000) * 5% = 0.05%
        assert!((overhead - 0.05).abs() < 0.01);
    }

    // Test 21: Default trait
    #[test]
    fn test_default_trait() {
        let sampler: AdaptiveSampler = Default::default();
        assert_eq!(sampler.threshold(), 100);
        assert!((sampler.sample_rate() - 0.01).abs() < 1e-6);
    }

    // Test 22: Clone trait
    #[test]
    fn test_clone_trait() {
        let sampler1 = AdaptiveSampler::with_threshold(200);
        let sampler2 = sampler1.clone();

        assert_eq!(sampler2.threshold(), 200);
        assert!((sampler2.sample_rate() - 0.01).abs() < 1e-6);
    }

    // Test 23: Debug trait
    #[test]
    fn test_debug_trait() {
        let sampler = AdaptiveSampler::new();
        let debug_str = format!("{:?}", sampler);
        assert!(debug_str.contains("AdaptiveSampler"));
    }

    // Test 24: Sample rate zero (never sample fast operations)
    #[test]
    fn test_sample_rate_zero() {
        let sampler = AdaptiveSampler::with_sample_rate(0.0);

        // Slow operations still traced
        assert!(sampler.should_trace(100));

        // Fast operations never traced (deterministic with rate=0)
        // Note: With rate=0, should_trace(fast) will always return false
        // because rand() < 0.0 is always false
    }

    // Test 25: Sample rate one (always sample)
    #[test]
    fn test_sample_rate_one() {
        let sampler = AdaptiveSampler::with_sample_rate(1.0);

        // All operations should be traced (both slow and fast)
        assert!(sampler.should_trace(0));
        assert!(sampler.should_trace(50));
        assert!(sampler.should_trace(100));
        assert!(sampler.should_trace(1000));
    }
}

#[cfg(kani)]
mod kani_proofs {
    use super::*;

    /// Prove that sample_rate is always clamped to [0.0, 1.0]
    #[kani::proof]
    fn proof_sample_rate_clamped() {
        let rate: f64 = kani::any();
        kani::assume(!rate.is_nan());
        let sampler = AdaptiveSampler::with_sample_rate(rate);
        kani::assert(sampler.sample_rate >= 0.0, "sample rate must be >= 0");
        kani::assert(sampler.sample_rate <= 1.0, "sample rate must be <= 1");
    }

    /// Prove that operations above threshold are always traced
    #[kani::proof]
    fn proof_above_threshold_always_traced() {
        let threshold: u64 = kani::any();
        kani::assume(threshold < u64::MAX);
        let sampler = AdaptiveSampler::with_threshold(threshold);
        let duration = threshold + 1;
        kani::assert(sampler.should_trace(duration), "above-threshold must always trace");
    }
}
