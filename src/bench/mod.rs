//! Benchmark harness for model runner comparison
//!
//! Implements the benchmark specification v1.1 with Toyota Way engineering principles:
//! - Dynamic CV-based stop-rule (Hoefler & Belli, SC12)
//! - Thermal throttling protocol
//! - ITL variance measurement (Dean & Barroso, "Tail at Scale")
//! - KV-cache fragmentation detection (PagedAttention methodology)
//! - KL-Divergence quality validation (LLM.int8())
//!
//! ## References
//!
//! - [17] Hoefler & Belli, "Scientific Benchmarking of Parallel Computing Systems", SC'15
//! - [11] Dean & Barroso, "The Tail at Scale", CACM 2013
//! - [12] Kwon et al., "PagedAttention", SOSP'23
//! - [13] Dettmers et al., "LLM.int8()", NeurIPS 2022

#![allow(clippy::cast_precision_loss)] // Statistical functions need usize->f64

use std::time::Duration;

use serde::{Deserialize, Serialize};

// PMAT-802: Extracted modules
mod gpu_parity;
mod load_testing;
mod matrix;
mod runtime;
mod statistics;

// RuntimeType always available (matrix.rs needs it)
pub use runtime::RuntimeType;

// HTTP-dependent runtime backends
#[cfg(feature = "bench-http")]
pub use runtime::{
    BackendInfo, BackendRegistry, InferenceRequest, InferenceResponse, LlamaCppBackend,
    LlamaCppConfig, MockBackend, OllamaBackend, OllamaConfig, RuntimeBackend, VllmBackend,
    VllmConfig,
};

pub use statistics::{
    detect_outliers, welch_t_test, BenchmarkMetrics, LatencyStatistics, MeasurementProtocol,
    Regression, RegressionDetector, RegressionReport, WelchTTestResult,
};

pub use load_testing::{LoadTestConfig, LoadTestResult, LoadTestRunner};

pub use matrix::{
    BackendSummary, BenchmarkMatrix, ComputeBackendType, MatrixBenchmarkConfig,
    MatrixBenchmarkEntry, MatrixSummary,
};

pub use gpu_parity::{
    FalsifiableClaim, FlashAttentionConfig, FusedOpSpec, FusedOpType, GapAnalysis,
    GemmPerformanceResult, GpuParityBenchmark, GpuParityResult, Imp900Result, MemoryPoolConfig,
    OptimizedGemmBenchmark, OptimizedGemmConfig,
};

// ============================================================================
// Dynamic Sampler (Section 2.1)
// ============================================================================

/// Dynamic stop-rule based on Coefficient of Variation (CV)
///
/// Per Hoefler & Belli [17], fixed iteration counts mask variance characteristics.
/// This sampler stops when statistical stability is achieved.
#[derive(Debug, Clone)]
pub struct DynamicSampler {
    /// Minimum number of samples before checking CV
    pub min_samples: usize,
    /// Maximum samples (failsafe)
    pub max_samples: usize,
    /// Target CV threshold (default: 0.05 = 5%)
    pub cv_threshold: f64,
    /// Sliding window size for CV calculation
    pub cv_window: usize,
    /// Number of consecutive stable windows required
    pub stability_count: usize,
    /// Current stability streak
    stable_streak: usize,
}

impl Default for DynamicSampler {
    fn default() -> Self {
        Self {
            min_samples: 100,
            max_samples: 10_000,
            cv_threshold: 0.05,
            cv_window: 50,
            stability_count: 3,
            stable_streak: 0,
        }
    }
}

impl DynamicSampler {
    /// Create a new sampler with custom parameters
    #[must_use]
    pub fn new(min_samples: usize, max_samples: usize, cv_threshold: f64) -> Self {
        Self {
            min_samples,
            max_samples,
            cv_threshold,
            cv_window: 50,
            stability_count: 3,
            stable_streak: 0,
        }
    }

    /// Check if sampling should continue
    ///
    /// Returns `true` if more samples are needed, `false` if stable.
    #[must_use]
    pub fn should_continue(&mut self, samples: &[f64]) -> bool {
        let n = samples.len();

        // Always continue until minimum samples
        if n < self.min_samples {
            return true;
        }

        // Stop at maximum (failsafe)
        if n >= self.max_samples {
            return false;
        }

        // Compute CV over sliding window
        let window_start = n.saturating_sub(self.cv_window);
        let window = &samples[window_start..];
        let cv = compute_cv(window);

        if cv < self.cv_threshold {
            self.stable_streak += 1;
            if self.stable_streak >= self.stability_count {
                return false; // Stable - stop sampling
            }
        } else {
            self.stable_streak = 0;
        }

        true // Continue sampling
    }

    /// Get the current CV for the last window
    #[must_use]
    pub fn current_cv(&self, samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return f64::INFINITY;
        }
        let window_start = samples.len().saturating_sub(self.cv_window);
        compute_cv(&samples[window_start..])
    }

    /// Reset the sampler for a new run
    pub fn reset(&mut self) {
        self.stable_streak = 0;
    }
}

/// Compute Coefficient of Variation (CV = std_dev / mean)
fn compute_cv(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::INFINITY;
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;

    if mean.abs() < 1e-10 {
        return f64::INFINITY;
    }

    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();

    std_dev / mean.abs()
}

// ============================================================================
// Thermal Guard (Section 2.2)
// ============================================================================

/// Temperature monitoring for benchmark validity
///
/// Per spec Section 2.2, benchmarks are invalid if temperature variance > 2°C.
#[derive(Debug, Clone)]
pub struct ThermalGuard {
    /// Maximum temperature before cooldown (°C)
    pub max_temp_c: f64,
    /// Temperature to resume at after cooldown (°C)
    pub cooldown_threshold_c: f64,
    /// Cooldown sleep duration (ms)
    pub cooldown_sleep_ms: u64,
    /// Maximum allowed temperature variance (°C)
    pub temp_variance_c: f64,
}

impl Default for ThermalGuard {
    fn default() -> Self {
        Self {
            max_temp_c: 80.0,
            cooldown_threshold_c: 70.0,
            cooldown_sleep_ms: 10_000,
            temp_variance_c: 2.0,
        }
    }
}

/// Result of thermal validation
#[derive(Debug, Clone, PartialEq)]
pub enum ThermalValidity {
    /// Temperature variance within acceptable range
    Valid,
    /// Temperature variance too high
    Invalid(String),
}

impl ThermalGuard {
    /// Create a new ThermalGuard with custom parameters
    #[must_use]
    pub fn new(
        max_temp_c: f64,
        cooldown_threshold_c: f64,
        cooldown_sleep_ms: u64,
        temp_variance_c: f64,
    ) -> Self {
        Self {
            max_temp_c,
            cooldown_threshold_c,
            cooldown_sleep_ms,
            temp_variance_c,
        }
    }

    /// Check if cooldown is needed (without sleeping)
    #[must_use]
    pub fn needs_cooldown(&self, current_temp: f64) -> bool {
        current_temp > self.max_temp_c
    }

    /// Check if benchmark results are thermally valid
    #[must_use]
    pub fn validate_run(&self, temps: &[f64]) -> ThermalValidity {
        if temps.is_empty() {
            return ThermalValidity::Valid;
        }

        let variance = compute_variance(temps);
        let std_dev = variance.sqrt();

        if std_dev > self.temp_variance_c {
            ThermalValidity::Invalid(format!(
                "Temperature variance {std_dev:.2}°C exceeds threshold {:.2}°C",
                self.temp_variance_c
            ))
        } else {
            ThermalValidity::Valid
        }
    }

    /// Check if cooldown is needed and sleep if so
    pub fn cooldown_if_needed(&self, current_temp: f64) {
        if current_temp > self.max_temp_c {
            std::thread::sleep(Duration::from_millis(self.cooldown_sleep_ms));
        }
    }

    /// Get max temperature from readings
    #[must_use]
    pub fn max_temp(&self, temps: &[f64]) -> f64 {
        if temps.is_empty() {
            return 0.0;
        }
        temps.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    }

    /// Get temperature variance
    #[must_use]
    pub fn temp_variance(&self, temps: &[f64]) -> f64 {
        compute_variance(temps).sqrt()
    }
}

/// Compute variance of a dataset
fn compute_variance(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
}

// ============================================================================
// KV-Cache Metrics (Section 4.3)
// ============================================================================

/// KV-cache fragmentation metrics per PagedAttention [12]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KvCacheMetrics {
    /// Total allocated KV-cache memory (bytes)
    pub allocated_bytes: u64,
    /// Actually used KV-cache memory (bytes)
    pub used_bytes: u64,
    /// Fragmentation percentage (waste)
    pub fragmentation_pct: f64,
}

impl KvCacheMetrics {
    /// Create new metrics from allocated and used bytes
    #[must_use]
    pub fn new(allocated_bytes: u64, used_bytes: u64) -> Self {
        let waste = allocated_bytes.saturating_sub(used_bytes);
        let fragmentation_pct = if allocated_bytes > 0 {
            (waste as f64 / allocated_bytes as f64) * 100.0
        } else {
            0.0
        };

        Self {
            allocated_bytes,
            used_bytes,
            fragmentation_pct,
        }
    }

    /// Get allocated memory in MB
    #[must_use]
    pub fn allocated_mb(&self) -> f64 {
        self.allocated_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Get used memory in MB
    #[must_use]
    pub fn used_mb(&self) -> f64 {
        self.used_bytes as f64 / (1024.0 * 1024.0)
    }

    /// Check if fragmentation is acceptable (< threshold)
    #[must_use]
    pub fn is_acceptable(&self, threshold_pct: f64) -> bool {
        self.fragmentation_pct < threshold_pct
    }
}

// ============================================================================
// Energy Metrics (Section 4.4)
// ============================================================================

/// Energy measurement metrics per Garcia-Martin et al. [14]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnergyMetrics {
    /// Total energy consumed (Joules)
    pub total_joules: f64,
    /// Idle power consumption (Watts)
    pub idle_watts: f64,
    /// Average active power consumption (Watts)
    pub active_watts_avg: f64,
    /// Number of tokens generated
    pub tokens_generated: u64,
}

impl EnergyMetrics {
    /// Create new energy metrics
    #[must_use]
    pub fn new(total_joules: f64, idle_watts: f64, active_watts_avg: f64, tokens: u64) -> Self {
        Self {
            total_joules,
            idle_watts,
            active_watts_avg,
            tokens_generated: tokens,
        }
    }

    /// Calculate energy per token (Joules/token)
    #[must_use]
    pub fn joules_per_token(&self) -> f64 {
        if self.tokens_generated == 0 {
            return 0.0;
        }
        self.total_joules / self.tokens_generated as f64
    }

    /// Calculate energy efficiency ratio (tokens per Joule)
    #[must_use]
    pub fn tokens_per_joule(&self) -> f64 {
        if self.total_joules < 1e-10 {
            return 0.0;
        }
        self.tokens_generated as f64 / self.total_joules
    }
}

// ============================================================================
// ITL (Inter-Token Latency) Metrics (Section 4.2)
// ============================================================================

/// Inter-Token Latency metrics per "Tail at Scale" [11]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ItlMetrics {
    /// Median ITL (ms)
    pub median_ms: f64,
    /// Standard deviation (jitter indicator)
    pub std_dev_ms: f64,
    /// p99 ITL (ms)
    pub p99_ms: f64,
    /// p99.9 ITL (ms)
    pub p999_ms: f64,
}

include!("mod_measurements_itl_metrics.rs");
include!("convoy_test_result.rs");
include!("mod_benchmark_full.rs");
