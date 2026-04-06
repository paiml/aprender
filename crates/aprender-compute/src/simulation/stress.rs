//! Stress Testing (Heijunka: Leveled Workload Testing)
//!
//! Provides configurable stress testing infrastructure for validating
//! compute operations under various load conditions.

use crate::Backend;

/// Stress test configuration for trueno operations
#[derive(Debug, Clone)]
pub struct StressTestConfig {
    /// Number of cycles per backend
    pub cycles_per_backend: u32,
    /// Input sizes to test (leveled)
    pub input_sizes: Vec<usize>,
    /// Backends to stress test
    pub backends: Vec<Backend>,
    /// Performance thresholds
    pub thresholds: StressThresholds,
    /// Master seed for RNG
    pub master_seed: u64,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            cycles_per_backend: 100,
            input_sizes: vec![100, 1_000, 10_000, 100_000, 1_000_000],
            backends: vec![Backend::Scalar, Backend::AVX2],
            thresholds: StressThresholds::default(),
            master_seed: 42,
        }
    }
}

impl StressTestConfig {
    /// Create new stress test config
    #[must_use]
    pub fn new(master_seed: u64) -> Self {
        Self { master_seed, ..Default::default() }
    }

    /// Set cycles per backend
    #[must_use]
    pub const fn with_cycles(mut self, cycles: u32) -> Self {
        self.cycles_per_backend = cycles;
        self
    }

    /// Set input sizes
    #[must_use]
    pub fn with_input_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.input_sizes = sizes;
        self
    }

    /// Set backends to test
    #[must_use]
    pub fn with_backends(mut self, backends: Vec<Backend>) -> Self {
        self.backends = backends;
        self
    }

    /// Set performance thresholds
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: StressThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// Calculate total test count
    #[must_use]
    pub fn total_tests(&self) -> usize {
        self.backends.len() * self.input_sizes.len() * self.cycles_per_backend as usize
    }
}

/// Performance thresholds for stress testing
#[derive(Debug, Clone)]
pub struct StressThresholds {
    /// Max time per operation (ms)
    pub max_op_time_ms: u64,
    /// Max memory per operation (bytes)
    pub max_memory_bytes: usize,
    /// Max variance in operation times (coefficient of variation)
    pub max_timing_variance: f64,
    /// Max allowed failure rate (0.0 to 1.0)
    pub max_failure_rate: f64,
}

impl Default for StressThresholds {
    fn default() -> Self {
        Self {
            max_op_time_ms: 1000,                // 1s max per op
            max_memory_bytes: 256 * 1024 * 1024, // 256MB max
            max_timing_variance: 0.5,            // 50% max variance
            max_failure_rate: 0.0,               // Zero failures allowed
        }
    }
}

impl StressThresholds {
    /// Strict thresholds for CI
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            max_op_time_ms: 100,
            max_memory_bytes: 64 * 1024 * 1024,
            max_timing_variance: 0.2,
            max_failure_rate: 0.0,
        }
    }

    /// Maximum operation time for development (5 seconds)
    const RELAXED_MAX_OP_TIME_MS: u64 = 5000;

    /// Relaxed thresholds for development
    #[must_use]
    pub const fn relaxed() -> Self {
        Self {
            max_op_time_ms: Self::RELAXED_MAX_OP_TIME_MS,
            max_memory_bytes: 512 * 1024 * 1024,
            max_timing_variance: 1.0,
            max_failure_rate: 0.01,
        }
    }
}

/// Stress test result for a single operation
#[derive(Debug, Clone)]
pub struct StressResult {
    /// Backend used
    pub backend: Backend,
    /// Input size
    pub input_size: usize,
    /// Cycles completed
    pub cycles_completed: u32,
    /// Total tests passed
    pub tests_passed: u32,
    /// Total tests failed
    pub tests_failed: u32,
    /// Mean operation time (ms)
    pub mean_op_time_ms: f64,
    /// Max operation time (ms)
    pub max_op_time_ms: u64,
    /// Timing variance (coefficient of variation)
    pub timing_variance: f64,
    /// Detected anomalies
    pub anomalies: Vec<StressAnomaly>,
}

impl StressResult {
    /// Check if all tests passed
    #[must_use]
    pub fn passed(&self) -> bool {
        self.tests_failed == 0 && self.anomalies.is_empty()
    }

    /// Calculate pass rate
    #[must_use]
    pub fn pass_rate(&self) -> f64 {
        let total = self.tests_passed + self.tests_failed;
        if total == 0 {
            1.0
        } else {
            self.tests_passed as f64 / total as f64
        }
    }
}

/// Anomaly detected during stress testing
#[derive(Debug, Clone)]
pub struct StressAnomaly {
    /// Cycle where anomaly was detected
    pub cycle: u32,
    /// Type of anomaly
    pub kind: StressAnomalyKind,
    /// Description
    pub description: String,
}

/// Types of stress test anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressAnomalyKind {
    /// Operation too slow
    SlowOperation,
    /// High memory usage
    HighMemory,
    /// Test failure
    TestFailure,
    /// Timing spike
    TimingSpike,
    /// Non-deterministic output
    NonDeterministic,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_test_config_default() {
        let config = StressTestConfig::default();

        assert_eq!(config.cycles_per_backend, 100);
        assert_eq!(config.input_sizes.len(), 5);
        assert_eq!(config.backends.len(), 2);
        assert_eq!(config.master_seed, 42);
    }

    #[test]
    fn test_stress_test_config_builder() {
        let config = StressTestConfig::new(123)
            .with_cycles(50)
            .with_input_sizes(vec![100, 1000])
            .with_backends(vec![Backend::Scalar])
            .with_thresholds(StressThresholds::strict());

        assert_eq!(config.master_seed, 123);
        assert_eq!(config.cycles_per_backend, 50);
        assert_eq!(config.input_sizes.len(), 2);
        assert_eq!(config.backends.len(), 1);
    }

    #[test]
    fn test_stress_test_config_total_tests() {
        let config = StressTestConfig::default()
            .with_cycles(10)
            .with_input_sizes(vec![100, 1000, 10000])
            .with_backends(vec![Backend::Scalar, Backend::AVX2]);

        // 2 backends * 3 sizes * 10 cycles = 60 tests
        assert_eq!(config.total_tests(), 60);
    }

    #[test]
    fn test_stress_thresholds_default() {
        let thresholds = StressThresholds::default();

        assert_eq!(thresholds.max_op_time_ms, 1000);
        assert_eq!(thresholds.max_memory_bytes, 256 * 1024 * 1024);
        assert!((thresholds.max_timing_variance - 0.5).abs() < 0.001);
        assert_eq!(thresholds.max_failure_rate, 0.0);
    }

    #[test]
    fn test_stress_thresholds_strict() {
        let thresholds = StressThresholds::strict();

        assert_eq!(thresholds.max_op_time_ms, 100);
        assert_eq!(thresholds.max_memory_bytes, 64 * 1024 * 1024);
        assert!((thresholds.max_timing_variance - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_stress_thresholds_relaxed() {
        let thresholds = StressThresholds::relaxed();

        assert_eq!(thresholds.max_op_time_ms, 5000);
        assert_eq!(thresholds.max_memory_bytes, 512 * 1024 * 1024);
        assert!((thresholds.max_timing_variance - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_stress_result_passed() {
        let result = StressResult {
            backend: Backend::Scalar,
            input_size: 1000,
            cycles_completed: 10,
            tests_passed: 100,
            tests_failed: 0,
            mean_op_time_ms: 50.0,
            max_op_time_ms: 100,
            timing_variance: 0.1,
            anomalies: vec![],
        };

        assert!(result.passed());
        assert_eq!(result.pass_rate(), 1.0);
    }

    #[test]
    fn test_stress_result_failed() {
        let result = StressResult {
            backend: Backend::AVX2,
            input_size: 10000,
            cycles_completed: 10,
            tests_passed: 95,
            tests_failed: 5,
            mean_op_time_ms: 100.0,
            max_op_time_ms: 500,
            timing_variance: 0.3,
            anomalies: vec![],
        };

        assert!(!result.passed()); // Failed because tests_failed > 0
        assert!((result.pass_rate() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_stress_result_with_anomaly() {
        let result = StressResult {
            backend: Backend::Scalar,
            input_size: 1000,
            cycles_completed: 10,
            tests_passed: 100,
            tests_failed: 0,
            mean_op_time_ms: 50.0,
            max_op_time_ms: 100,
            timing_variance: 0.1,
            anomalies: vec![StressAnomaly {
                cycle: 5,
                kind: StressAnomalyKind::SlowOperation,
                description: "Operation took 200ms".to_string(),
            }],
        };

        assert!(!result.passed()); // Failed because anomalies not empty
    }

    #[test]
    fn test_stress_anomaly_kinds() {
        assert_eq!(StressAnomalyKind::SlowOperation, StressAnomalyKind::SlowOperation);
        assert_ne!(StressAnomalyKind::SlowOperation, StressAnomalyKind::TestFailure);

        // Test all variants exist
        let _ = StressAnomalyKind::SlowOperation;
        let _ = StressAnomalyKind::HighMemory;
        let _ = StressAnomalyKind::TestFailure;
        let _ = StressAnomalyKind::TimingSpike;
        let _ = StressAnomalyKind::NonDeterministic;
    }

    #[test]
    fn test_stress_result_zero_tests() {
        let result = StressResult {
            backend: Backend::Scalar,
            input_size: 0,
            cycles_completed: 0,
            tests_passed: 0,
            tests_failed: 0,
            mean_op_time_ms: 0.0,
            max_op_time_ms: 0,
            timing_variance: 0.0,
            anomalies: vec![],
        };

        // Zero tests should still pass with pass_rate of 1.0
        assert!(result.passed());
        assert_eq!(result.pass_rate(), 1.0);
    }
}
