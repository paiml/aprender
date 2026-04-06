
/// ARM64-specific optimizations
///
/// Per spec §6.1: Graviton optimization
pub mod arm64 {
    /// Check if running on ARM64 architecture
    #[must_use]
    pub const fn is_arm64() -> bool {
        cfg!(target_arch = "aarch64")
    }

    /// Target architecture string
    #[must_use]
    pub const fn target_arch() -> &'static str {
        #[cfg(target_arch = "aarch64")]
        {
            "aarch64"
        }
        #[cfg(target_arch = "x86_64")]
        {
            "x86_64"
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            "unknown"
        }
    }

    /// Optimal SIMD instruction set for current architecture
    #[must_use]
    pub const fn optimal_simd() -> &'static str {
        #[cfg(target_arch = "aarch64")]
        {
            "NEON"
        }
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            "AVX2"
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
        {
            "SSE2"
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            "Scalar"
        }
    }
}

/// Cold start benchmark utilities
///
/// Per spec §7.1: Cold start mitigation measurement
pub mod benchmark {
    use serde::{Deserialize, Serialize};

    use super::{arm64, Instant, LambdaError, LambdaHandler, LambdaRequest};

    /// Target cold start time (per spec §8.1)
    pub const TARGET_COLD_START_MS: f64 = 50.0;

    /// Target warm inference time (per spec §8.1)
    pub const TARGET_WARM_INFERENCE_MS: f64 = 10.0;

    /// Benchmark result
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct BenchmarkResult {
        /// Cold start latency (ms)
        pub cold_start_ms: f64,
        /// Warm inference latency (ms) - median of N invocations
        pub warm_inference_ms: f64,
        /// Number of warm invocations measured
        pub warm_iterations: usize,
        /// Model size (bytes)
        pub model_size_bytes: usize,
        /// Target architecture
        pub target_arch: String,
        /// SIMD instruction set used
        pub simd_backend: String,
        /// Meets cold start target
        pub meets_cold_start_target: bool,
        /// Meets warm inference target
        pub meets_warm_inference_target: bool,
    }

    impl BenchmarkResult {
        /// Check if all targets are met
        #[must_use]
        pub fn meets_all_targets(&self) -> bool {
            self.meets_cold_start_target && self.meets_warm_inference_target
        }
    }

    /// Run cold start benchmark
    ///
    /// Per spec §7.1: Measure cold start with breakdown
    ///
    /// # Errors
    ///
    /// Returns `LambdaError` if handler invocation fails.
    ///
    /// # Panics
    ///
    /// Panics if latencies contain NaN values (should not happen with valid inputs).
    pub fn benchmark_cold_start(
        handler: &LambdaHandler,
        request: &LambdaRequest,
        warm_iterations: usize,
    ) -> Result<BenchmarkResult, LambdaError> {
        // First invocation is cold start
        let cold_start = Instant::now();
        let _cold_response = handler.handle(request)?;
        let cold_start_ms = cold_start.elapsed().as_secs_f64() * 1000.0;

        // Warm invocations
        let mut warm_latencies = Vec::with_capacity(warm_iterations);
        for _ in 0..warm_iterations {
            let start = Instant::now();
            let _response = handler.handle(request)?;
            warm_latencies.push(start.elapsed().as_secs_f64() * 1000.0);
        }

        // Compute median warm latency
        warm_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let warm_inference_ms = if warm_latencies.is_empty() {
            0.0
        } else {
            warm_latencies[warm_latencies.len() / 2]
        };

        Ok(BenchmarkResult {
            cold_start_ms,
            warm_inference_ms,
            warm_iterations,
            model_size_bytes: handler.model_size_bytes(),
            target_arch: arm64::target_arch().to_string(),
            simd_backend: arm64::optimal_simd().to_string(),
            meets_cold_start_target: cold_start_ms <= TARGET_COLD_START_MS,
            meets_warm_inference_target: warm_inference_ms <= TARGET_WARM_INFERENCE_MS,
        })
    }
}
