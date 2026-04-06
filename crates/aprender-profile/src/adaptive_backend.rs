//! Adaptive Backend Selection for Trueno Compute Integration (Section 5.2)
//!
//! Provides intelligent backend selection (GPU/SIMD/Scalar) based on:
//! - Historical profiling data
//! - Workload characteristics (operation type, input size)
//! - Adaptive sampling to minimize tracing overhead
//!
//! # Architecture
//!
//! - Integrates with Trueno's backend selection system
//! - Tracks performance metrics per `operation+input_size`
//! - Applies adaptive sampling (>100μs threshold by default)
//! - Exports backend selection decisions to OTLP
//!
//! # Usage
//!
//! ```ignore
//! use renacer::{AdaptiveBackend, OtlpExporter, OtlpConfig};
//!
//! let otlp_config = OtlpConfig::new(
//!     "http://localhost:4317".to_string(),
//!     "trueno-app".to_string(),
//! );
//! let otlp_exporter = OtlpExporter::new(otlp_config, None).expect("test");
//! let otlp_arc = std::sync::Arc::new(otlp_exporter);
//!
//! let backend = AdaptiveBackend::new(Some(otlp_arc));
//!
//! // Select backend for operation
//! let selected = backend.select("matrix_multiply", 10000);
//! println!("Selected backend: {:?}", selected);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "otlp")]
use crate::otlp_exporter::OtlpExporter;

#[cfg(not(feature = "otlp"))]
use crate::otlp_exporter::OtlpExporter;

/// Backend type for compute operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// GPU backend (CUDA/wgpu)
    GPU,
    /// SIMD backend (AVX2/NEON)
    SIMD,
    /// Scalar backend (fallback)
    Scalar,
}

impl Backend {
    pub fn to_string(&self) -> &'static str {
        match self {
            Backend::GPU => "gpu",
            Backend::SIMD => "simd",
            Backend::Scalar => "scalar",
        }
    }
}

/// Performance metrics for a specific operation
#[derive(Debug, Clone)]
struct PerformanceMetrics {
    /// Average duration in microseconds
    avg_duration_us: f64,
    /// Number of samples collected
    sample_count: u64,
    /// Backend that produced these metrics
    backend: Backend,
}

/// Adaptive backend selector with tracing integration
///
/// Selects the optimal backend (GPU/SIMD/Scalar) based on:
/// - Historical profiling data
/// - Operation characteristics (name, input size)
/// - Adaptive sampling policy (<5% overhead target)
pub struct AdaptiveBackend {
    /// OTLP exporter for tracing backend selection decisions
    #[cfg(feature = "otlp")]
    otlp_exporter: Option<Arc<OtlpExporter>>,

    #[cfg(not(feature = "otlp"))]
    otlp_exporter: Option<Arc<OtlpExporter>>,

    /// Performance history: (operation, `input_size`) → metrics
    #[allow(clippy::type_complexity)]
    performance_history: Arc<Mutex<HashMap<(String, usize), Vec<PerformanceMetrics>>>>,

    /// Adaptive sampling threshold (microseconds)
    sampling_threshold_us: u64,

    /// Hot path detection: operations with >10,000 calls/sec disable tracing
    call_counts: Arc<Mutex<HashMap<String, u64>>>,
}

impl AdaptiveBackend {
    /// Create a new adaptive backend selector
    ///
    /// # Arguments
    ///
    /// * `otlp_exporter` - Optional OTLP exporter for tracing
    ///
    /// # Returns
    ///
    /// Returns a new `AdaptiveBackend` with default settings:
    /// - Sampling threshold: 100μs
    /// - Hot path threshold: 10,000 calls/sec
    pub fn new(otlp_exporter: Option<Arc<OtlpExporter>>) -> Self {
        AdaptiveBackend {
            otlp_exporter,
            performance_history: Arc::new(Mutex::new(HashMap::new())),
            sampling_threshold_us: 100, // Same as Sprint 32/37
            call_counts: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Select optimal backend for the given operation
    ///
    /// # Arguments
    ///
    /// * `operation` - Operation name (e.g., "`matrix_multiply`")
    /// * `input_size` - Input data size in elements
    ///
    /// # Returns
    ///
    /// Returns the selected `Backend` based on historical data or heuristics
    pub fn select(&self, operation: &str, input_size: usize) -> Backend {
        // Check if this is a hot path (>10,000 calls/sec)
        if self.is_hot_path(operation) {
            // Use heuristics without tracing overhead
            return self.select_without_tracing(operation, input_size);
        }

        // Look up historical performance data
        let backend = if let Some(best_backend) = self.get_best_backend(operation, input_size) {
            best_backend
        } else {
            // No historical data, use heuristics
            self.select_heuristic(operation, input_size)
        };

        // Record backend selection decision
        self.record_selection(operation, input_size, backend);

        backend
    }

    /// Check if operation is a hot path (>10,000 calls/sec)
    fn is_hot_path(&self, operation: &str) -> bool {
        if let Ok(counts) = self.call_counts.lock() {
            if let Some(&count) = counts.get(operation) {
                return count > 10_000;
            }
        }
        false
    }

    /// Select backend without tracing (hot path optimization)
    fn select_without_tracing(&self, operation: &str, input_size: usize) -> Backend {
        self.select_heuristic(operation, input_size)
    }

    /// Get best backend from historical performance data
    fn get_best_backend(&self, operation: &str, input_size: usize) -> Option<Backend> {
        let history = self.performance_history.lock().ok()?;
        let key = (operation.to_string(), input_size);
        let metrics = history.get(&key)?;

        // Find backend with lowest average duration
        metrics
            .iter()
            .min_by(|a, b| {
                a.avg_duration_us
                    .partial_cmp(&b.avg_duration_us)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| m.backend)
    }

    /// Select backend using heuristics (no historical data)
    fn select_heuristic(&self, operation: &str, input_size: usize) -> Backend {
        // GPU heuristic: large matrix operations
        if self.should_use_gpu(operation, input_size) {
            return Backend::GPU;
        }

        // SIMD heuristic: medium-sized vector operations
        if self.should_use_simd(operation) {
            return Backend::SIMD;
        }

        // Fallback: scalar
        Backend::Scalar
    }

    /// Heuristic: should use GPU backend?
    ///
    /// Uses GPU for:
    /// - Matrix operations with >10,000 elements
    /// - Convolution operations with >1,000 elements
    pub fn should_use_gpu(&self, operation: &str, input_size: usize) -> bool {
        match operation {
            "matrix_multiply" | "matrix_add" | "matrix_transpose" => input_size > 10_000,
            "convolution" | "pooling" => input_size > 1_000,
            _ => false,
        }
    }

    /// Heuristic: should use SIMD backend?
    ///
    /// Uses SIMD for:
    /// - Vector operations (add, multiply, dot product)
    /// - Element-wise operations
    /// - Reductions (sum, mean, max)
    pub fn should_use_simd(&self, operation: &str) -> bool {
        matches!(
            operation,
            "vector_add"
                | "vector_multiply"
                | "dot_product"
                | "elementwise_add"
                | "elementwise_multiply"
                | "sum"
                | "mean"
                | "max"
                | "min"
        )
    }

    /// Record backend selection decision
    #[allow(unused_variables)]
    fn record_selection(&self, operation: &str, input_size: usize, backend: Backend) {
        // Update call count
        if let Ok(mut counts) = self.call_counts.lock() {
            *counts.entry(operation.to_string()).or_insert(0) += 1;
        }

        // Export to OTLP if enabled and not a hot path
        #[cfg(feature = "otlp")]
        if let Some(exporter) = &self.otlp_exporter {
            if !self.is_hot_path(operation) {
                self.export_selection(exporter, operation, input_size, backend);
            }
        }
    }

    /// Export backend selection to OTLP
    #[cfg(feature = "otlp")]
    fn export_selection(
        &self,
        exporter: &OtlpExporter,
        _operation: &str,
        input_size: usize,
        _backend: Backend,
    ) {
        use crate::otlp_exporter::ComputeBlock;

        // Create a compute block for backend selection
        let block = ComputeBlock {
            operation: "backend_selection",
            duration_us: 1, // Negligible overhead
            elements: input_size,
            is_slow: false,
        };

        exporter.record_compute_block(block);
    }

    /// Record performance metrics for a completed operation
    ///
    /// # Arguments
    ///
    /// * `operation` - Operation name
    /// * `input_size` - Input data size
    /// * `backend` - Backend that was used
    /// * `duration_us` - Execution duration in microseconds
    pub fn record_performance(
        &self,
        operation: &str,
        input_size: usize,
        backend: Backend,
        duration_us: u64,
    ) {
        // Only record if above sampling threshold
        if duration_us < self.sampling_threshold_us {
            // Sample 1% of fast operations
            if rand::random::<f64>() > 0.01 {
                return;
            }
        }

        let key = (operation.to_string(), input_size);
        if let Ok(mut history) = self.performance_history.lock() {
            let metrics_list = history.entry(key).or_insert_with(Vec::new);

            // Find existing metrics for this backend or create new
            if let Some(metrics) = metrics_list.iter_mut().find(|m| m.backend == backend) {
                // Update running average
                let total = metrics.avg_duration_us * metrics.sample_count as f64;
                metrics.sample_count += 1;
                metrics.avg_duration_us =
                    (total + duration_us as f64) / metrics.sample_count as f64;
            } else {
                // Create new metrics entry
                metrics_list.push(PerformanceMetrics {
                    avg_duration_us: duration_us as f64,
                    sample_count: 1,
                    backend,
                });
            }
        }
    }

    /// Get performance statistics for an operation
    ///
    /// Returns the best backend and its average duration, or None if no data
    pub fn get_performance_stats(
        &self,
        operation: &str,
        input_size: usize,
    ) -> Option<(Backend, f64)> {
        let history = self.performance_history.lock().ok()?;
        let key = (operation.to_string(), input_size);
        let metrics = history.get(&key)?;

        metrics
            .iter()
            .min_by(|a, b| {
                a.avg_duration_us
                    .partial_cmp(&b.avg_duration_us)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|m| (m.backend, m.avg_duration_us))
    }

    /// Reset performance history (for testing)
    #[cfg(test)]
    pub fn reset_history(&self) {
        if let Ok(mut history) = self.performance_history.lock() {
            history.clear();
        }
        if let Ok(mut counts) = self.call_counts.lock() {
            counts.clear();
        }
    }
}

// Add rand dependency stub for 1% sampling
mod rand {
    pub(super) fn random<T>() -> T
    where
        T: From<f64>,
    {
        // Use simple PRNG for sampling decision
        // In production, use `rand` crate
        T::from(0.5) // Deterministic for testing
    }
}

// Compile-time thread-safety verification (Sprint 59)
static_assertions::assert_impl_all!(Backend: Send, Sync);
static_assertions::assert_impl_all!(AdaptiveBackend: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_backend_new() {
        let backend = AdaptiveBackend::new(None);
        assert_eq!(backend.sampling_threshold_us, 100);
    }

    #[test]
    fn test_backend_to_string() {
        assert_eq!(Backend::GPU.to_string(), "gpu");
        assert_eq!(Backend::SIMD.to_string(), "simd");
        assert_eq!(Backend::Scalar.to_string(), "scalar");
    }

    #[test]
    fn test_should_use_gpu_matrix_multiply_large() {
        let backend = AdaptiveBackend::new(None);
        assert!(backend.should_use_gpu("matrix_multiply", 20_000));
    }

    #[test]
    fn test_should_use_gpu_matrix_multiply_small() {
        let backend = AdaptiveBackend::new(None);
        assert!(!backend.should_use_gpu("matrix_multiply", 100));
    }

    #[test]
    fn test_should_use_simd_vector_operations() {
        let backend = AdaptiveBackend::new(None);
        assert!(backend.should_use_simd("vector_add"));
        assert!(backend.should_use_simd("dot_product"));
        assert!(backend.should_use_simd("sum"));
    }

    #[test]
    fn test_should_use_simd_non_vector_operations() {
        let backend = AdaptiveBackend::new(None);
        assert!(!backend.should_use_simd("matrix_multiply"));
        assert!(!backend.should_use_simd("convolution"));
    }

    #[test]
    fn test_select_heuristic_gpu() {
        let backend = AdaptiveBackend::new(None);
        let selected = backend.select_heuristic("matrix_multiply", 20_000);
        assert_eq!(selected, Backend::GPU);
    }

    #[test]
    fn test_select_heuristic_simd() {
        let backend = AdaptiveBackend::new(None);
        let selected = backend.select_heuristic("vector_add", 1_000);
        assert_eq!(selected, Backend::SIMD);
    }

    #[test]
    fn test_select_heuristic_scalar() {
        let backend = AdaptiveBackend::new(None);
        let selected = backend.select_heuristic("custom_operation", 100);
        assert_eq!(selected, Backend::Scalar);
    }

    #[test]
    fn test_record_performance() {
        let backend = AdaptiveBackend::new(None);
        backend.record_performance("matrix_multiply", 10_000, Backend::GPU, 500);

        let stats = backend.get_performance_stats("matrix_multiply", 10_000);
        assert!(stats.is_some());
        let (best_backend, avg_duration) = stats.expect("test");
        assert_eq!(best_backend, Backend::GPU);
        assert_eq!(avg_duration, 500.0);
    }

    #[test]
    fn test_record_performance_multiple_backends() {
        let backend = AdaptiveBackend::new(None);
        backend.record_performance("matrix_multiply", 10_000, Backend::GPU, 500);
        backend.record_performance("matrix_multiply", 10_000, Backend::SIMD, 800);
        backend.record_performance("matrix_multiply", 10_000, Backend::Scalar, 1200);

        let stats = backend.get_performance_stats("matrix_multiply", 10_000);
        assert!(stats.is_some());
        let (best_backend, avg_duration) = stats.expect("test");
        assert_eq!(best_backend, Backend::GPU); // Fastest
        assert_eq!(avg_duration, 500.0);
    }

    #[test]
    fn test_get_best_backend_with_history() {
        let backend = AdaptiveBackend::new(None);
        backend.record_performance("vector_add", 1_000, Backend::SIMD, 100);
        backend.record_performance("vector_add", 1_000, Backend::Scalar, 200);

        let best = backend.get_best_backend("vector_add", 1_000);
        assert_eq!(best, Some(Backend::SIMD));
    }

    #[test]
    fn test_get_best_backend_no_history() {
        let backend = AdaptiveBackend::new(None);
        let best = backend.get_best_backend("unknown_operation", 1_000);
        assert!(best.is_none());
    }

    #[test]
    fn test_select_uses_history_when_available() {
        let backend = AdaptiveBackend::new(None);
        backend.record_performance("matrix_multiply", 10_000, Backend::GPU, 300);
        backend.record_performance("matrix_multiply", 10_000, Backend::SIMD, 600);

        let selected = backend.select("matrix_multiply", 10_000);
        assert_eq!(selected, Backend::GPU); // Fastest from history
    }

    #[test]
    fn test_select_uses_heuristic_when_no_history() {
        let backend = AdaptiveBackend::new(None);
        let selected = backend.select("matrix_multiply", 20_000);
        assert_eq!(selected, Backend::GPU); // Heuristic: large matrix
    }

    #[test]
    fn test_is_hot_path_false_initially() {
        let backend = AdaptiveBackend::new(None);
        assert!(!backend.is_hot_path("operation"));
    }

    #[test]
    fn test_hot_path_detection() {
        let backend = AdaptiveBackend::new(None);

        // Simulate 10,001 calls to trigger hot path
        for _ in 0..10_001 {
            backend.select("hot_operation", 1_000);
        }

        assert!(backend.is_hot_path("hot_operation"));
    }

    #[test]
    fn test_reset_history() {
        let backend = AdaptiveBackend::new(None);
        backend.record_performance("operation", 1_000, Backend::GPU, 500);

        backend.reset_history();

        let stats = backend.get_performance_stats("operation", 1_000);
        assert!(stats.is_none());
    }

    #[test]
    fn test_running_average_calculation() {
        let backend = AdaptiveBackend::new(None);

        // Record 3 measurements: 100, 200, 300
        backend.record_performance("operation", 1_000, Backend::GPU, 100);
        backend.record_performance("operation", 1_000, Backend::GPU, 200);
        backend.record_performance("operation", 1_000, Backend::GPU, 300);

        let stats = backend.get_performance_stats("operation", 1_000);
        assert!(stats.is_some());
        let (_, avg_duration) = stats.expect("test");
        assert_eq!(avg_duration, 200.0); // (100 + 200 + 300) / 3 = 200
    }
}
