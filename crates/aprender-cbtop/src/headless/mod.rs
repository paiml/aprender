//! Headless benchmark mode for CI/CD and AI agent integration
//!
//! Enables cbtop to run without a TTY, outputting machine-readable results.
//!
//! # Example
//!
//! ```bash
//! # Run headless benchmark
//! cbtop --headless --format json --duration 5
//!
//! # Use bench subcommand
//! cbtop bench --backend simd --workload gemm --duration 5
//! ```

mod types;
pub use types::*;

use crate::brick::Scorable;
use crate::bricks::generators::SimdLoadBrick;
use crate::config::{ComputeBackend, WorkloadType};
use crate::error::CbtopError;
use std::time::{Duration, Instant};

/// Headless benchmark runner
pub struct HeadlessBenchmark {
    backend: ComputeBackend,
    workload: WorkloadType,
    size: usize,
    duration: Duration,
}

impl HeadlessBenchmark {
    /// Create a new headless benchmark
    pub fn new(
        backend: ComputeBackend,
        workload: WorkloadType,
        size: usize,
        duration: Duration,
    ) -> Self {
        Self {
            backend,
            workload,
            size,
            duration,
        }
    }

    /// Run the benchmark and return results
    pub fn run(&self) -> Result<BenchmarkResult, CbtopError> {
        let system = SystemInfo::detect();
        let start_time = Instant::now();

        // Create and configure the load brick
        let mut brick = SimdLoadBrick::new(self.size);
        brick.set_workload(self.workload);
        brick.set_intensity(1.0); // Full intensity for benchmarking
        brick.start();

        // OPT-013: Warmup phase with scaled duration
        // Longer warmup for small sizes to ensure stable cache/branch predictor state
        // Small sizes complete quickly, need more warmup time to reach steady state
        let base_warmup_ms = (self.duration.as_millis() / 10).max(100) as u64;
        let warmup_duration = if self.size < 100_000 {
            Duration::from_millis(base_warmup_ms * 2) // 2x warmup for small sizes
        } else {
            Duration::from_millis(base_warmup_ms)
        };
        let warmup_start = Instant::now();
        while warmup_start.elapsed() < warmup_duration {
            brick.run_iteration();
        }

        // Reset metrics after warmup
        let mut brick = SimdLoadBrick::new(self.size);
        brick.set_workload(self.workload);
        brick.set_intensity(1.0);
        brick.start();

        // OPT-014: Sample CPU frequency at start of measurement
        let start_freq_mhz = Self::sample_cpu_freq();

        // OPT-008: Calculate minimum iterations for statistical stability
        // Small workloads complete too quickly, causing high variance (CV > 600%)
        // Require more iterations for smaller sizes to get stable measurements
        let min_iterations: u64 = if self.size < 10_000 {
            5000 // Very small: need many iterations
        } else if self.size < 100_000 {
            1000 // Small: need moderate iterations
        } else if self.size < 1_000_000 {
            100 // Medium: fewer iterations needed
        } else {
            10 // Large: minimal iterations (each takes significant time)
        };

        // Measurement phase
        let mut iterations = 0u64;
        let measure_start = Instant::now();

        // OPT-008: Run until both duration AND minimum iterations are satisfied
        while measure_start.elapsed() < self.duration || iterations < min_iterations {
            brick.run_iteration();
            iterations += 1;

            // Safety: cap at 100K iterations to prevent runaway benchmarks
            if iterations >= 100_000 {
                break;
            }
        }

        let total_duration = start_time.elapsed();
        brick.stop();

        // OPT-014: Sample CPU frequency at end and detect throttling
        let end_freq_mhz = Self::sample_cpu_freq();

        // Calculate statistics using brick's internal latency history (PERF-002)
        // This ensures CV calculation matches what score() uses
        let latencies = brick.latency_history_slice();
        let latency_stats = Self::calculate_latency_stats(&latencies);
        let gflops = brick.gflops();
        let throughput = if latency_stats.mean > 0.0 {
            1000.0 / latency_stats.mean
        } else {
            0.0
        };

        // Get score
        let score = brick.score();

        // PERF-003: Check for benchmark environment warnings
        let mut warnings = system.check_benchmark_readiness();

        // OPT-014: Detect frequency throttling during benchmark
        if let (Some(start), Some(end)) = (start_freq_mhz, end_freq_mhz) {
            if start > 0 {
                let drop_percent = ((start as f64 - end as f64) / start as f64) * 100.0;
                if drop_percent > 5.0 {
                    warnings.push(format!(
                        "CPU frequency dropped {}MHz -> {}MHz ({:.1}% drop) during benchmark. \
                         Possible thermal throttling.",
                        start, end, drop_percent
                    ));
                }
            }
        }

        Ok(BenchmarkResult {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            duration_secs: total_duration.as_secs_f64(),
            system,
            benchmark: BenchmarkConfig {
                backend: format!("{:?}", self.backend),
                workload: format!("{:?}", self.workload),
                size: self.size,
                iterations,
            },
            results: BenchmarkResults {
                gflops,
                throughput_ops_sec: throughput,
                latency_ms: latency_stats,
            },
            score: score.into(),
            warnings,
        })
    }

    /// OPT-014: Sample current CPU frequency for throttling detection
    fn sample_cpu_freq() -> Option<u32> {
        #[cfg(target_os = "linux")]
        {
            let path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq";
            if let Ok(content) = std::fs::read_to_string(path) {
                return content.trim().parse::<u32>().ok().map(|khz| khz / 1000);
            }
        }
        None
    }

    fn calculate_latency_stats(latencies: &[f64]) -> LatencyStats {
        if latencies.is_empty() {
            return LatencyStats {
                mean: 0.0,
                min: 0.0,
                max: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
                cv_percent: 0.0,
            };
        }

        // OPT-015: Filter outliers using IQR method before calculating CV
        // This reduces measurement noise from system interrupts, GC pauses, etc.
        let filtered = Self::filter_outliers_iqr(latencies);
        let data = if filtered.len() >= 10 {
            &filtered
        } else {
            latencies
        };

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Calculate standard deviation on filtered data
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();
        let cv_percent = if mean > 0.0 {
            (std_dev / mean) * 100.0
        } else {
            0.0
        };

        // Calculate percentiles on original data (for accurate p95/p99)
        let mut sorted = latencies.to_vec();
        sorted.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("latency values MUST be comparable (no NaN)")
        });

        let percentile = |p: f64| -> f64 {
            let idx = (p * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        };

        LatencyStats {
            mean,
            min,
            max,
            p50: percentile(0.50),
            p95: percentile(0.95),
            p99: percentile(0.99),
            cv_percent,
        }
    }

    /// OPT-015: Filter outliers using IQR (Interquartile Range) method
    /// Removes values outside Q1 - 1.5*IQR and Q3 + 1.5*IQR
    fn filter_outliers_iqr(data: &[f64]) -> Vec<f64> {
        if data.len() < 4 {
            return data.to_vec();
        }

        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| {
            a.partial_cmp(b)
                .expect("data values MUST be comparable (no NaN)")
        });

        let n = sorted.len();
        let q1_idx = n / 4;
        let q3_idx = (3 * n) / 4;

        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        // Use 1.5*IQR rule (standard for outlier detection)
        let lower_bound = q1 - 1.5 * iqr;
        let upper_bound = q3 + 1.5 * iqr;

        data.iter()
            .cloned()
            .filter(|&x| x >= lower_bound && x <= upper_bound)
            .collect()
    }
}

// ============================================================================
// Library API for Programmatic Access (HL-007)
// ============================================================================

/// Builder for creating benchmarks programmatically
///
/// This provides an ergonomic API for running cbtop benchmarks from Rust code.
///
/// # Example
///
/// ```rust,no_run
/// use cbtop::{Benchmark, BenchmarkResult};
/// use std::time::Duration;
///
/// let result: BenchmarkResult = Benchmark::builder()
///     .workload("gemm")
///     .size(1_000_000)
///     .duration(Duration::from_secs(5))
///     .build()
///     .unwrap()
///     .run()
///     .unwrap();
///
/// println!("GFLOP/s: {}", result.results.gflops);
/// ```
#[derive(Default)]
pub struct BenchmarkBuilder {
    backend: Option<ComputeBackend>,
    workload: Option<WorkloadType>,
    size: Option<usize>,
    duration: Option<Duration>,
}

impl BenchmarkBuilder {
    /// Create a new benchmark builder with defaults
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the compute backend (default: Auto/Simd)
    pub fn backend(mut self, backend: ComputeBackend) -> Self {
        self.backend = Some(backend);
        self
    }

    /// Set the compute backend from string (e.g., "simd", "cuda", "auto")
    pub fn backend_str(mut self, backend: &str) -> Self {
        self.backend = Some(match backend.to_lowercase().as_str() {
            "cuda" => ComputeBackend::Cuda,
            "wgpu" => ComputeBackend::Wgpu,
            "simd" => ComputeBackend::Simd,
            _ => ComputeBackend::Simd, // Default to SIMD
        });
        self
    }

    /// Set the workload type (default: Gemm)
    pub fn workload_type(mut self, workload: WorkloadType) -> Self {
        self.workload = Some(workload);
        self
    }

    /// Set the workload type from string (e.g., "gemm", "dot", "elementwise")
    pub fn workload(mut self, workload: &str) -> Self {
        self.workload = Some(match workload.to_lowercase().as_str() {
            "dot" | "dotproduct" | "dot_product" => WorkloadType::Gemm,
            "elementwise" | "element_wise" => WorkloadType::Elementwise,
            "reduction" | "reduce" => WorkloadType::Reduction,
            "bandwidth" | "memcpy" => WorkloadType::Bandwidth,
            "conv2d" | "conv" | "convolution" => WorkloadType::Conv2d,
            "attention" | "attn" => WorkloadType::Attention,
            "all" => WorkloadType::All,
            _ => WorkloadType::Gemm, // Default to GEMM
        });
        self
    }

    /// Set the problem size (default: 1_000_000)
    pub fn size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    /// Set the benchmark duration (default: 5 seconds)
    pub fn duration(mut self, duration: Duration) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set the benchmark duration in seconds
    pub fn duration_secs(mut self, secs: u64) -> Self {
        self.duration = Some(Duration::from_secs(secs));
        self
    }

    /// Build the benchmark with the configured parameters
    pub fn build(self) -> Result<Benchmark, CbtopError> {
        Ok(Benchmark {
            inner: HeadlessBenchmark::new(
                self.backend.unwrap_or(ComputeBackend::Simd),
                self.workload.unwrap_or(WorkloadType::Gemm),
                self.size.unwrap_or(1_000_000),
                self.duration.unwrap_or(Duration::from_secs(5)),
            ),
        })
    }
}

/// Benchmark runner for programmatic access
///
/// Created via [`Benchmark::builder()`].
pub struct Benchmark {
    inner: HeadlessBenchmark,
}

impl Benchmark {
    /// Create a new benchmark builder
    pub fn builder() -> BenchmarkBuilder {
        BenchmarkBuilder::new()
    }

    /// Run the benchmark and return results
    pub fn run(&self) -> Result<BenchmarkResult, CbtopError> {
        self.inner.run()
    }

    /// Run the benchmark and compare against a baseline
    pub fn run_with_baseline(
        &self,
        baseline: &BenchmarkResult,
        threshold: f64,
    ) -> Result<(BenchmarkResult, RegressionResult), CbtopError> {
        let result = self.inner.run()?;
        let regression = result.check_regression(baseline, threshold);
        Ok((result, regression))
    }
}

#[cfg(test)]
mod tests;
