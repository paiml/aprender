//! Load testing framework for stress testing inference servers
//!
//! Extracted from bench/mod.rs (PMAT-802) to reduce module size.
//! Contains:
//! - LoadTestConfig for wrk2-style load testing configuration
//! - LoadTestResult for test outcomes
//! - LoadTestRunner for executing load tests

#![allow(clippy::cast_precision_loss)]

use serde::{Deserialize, Serialize};

// ============================================================================
// Load Testing (Section 14.1)
// ============================================================================

/// Configuration for load testing
///
/// Per spec §14: Implements wrk2-style load testing with configurable
/// concurrency, duration, and target rates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestConfig {
    /// Number of concurrent connections/threads
    pub concurrency: usize,
    /// Test duration in seconds
    pub duration_secs: u64,
    /// Target requests per second (0 = unlimited)
    pub target_rps: f64,
    /// Request timeout in milliseconds
    pub timeout_ms: u64,
    /// Warm-up period in seconds
    pub warmup_secs: u64,
    /// Target latency threshold (p99) in milliseconds
    pub latency_threshold_ms: f64,
}

impl Default for LoadTestConfig {
    fn default() -> Self {
        Self {
            concurrency: 10,
            duration_secs: 60,
            target_rps: 0.0, // Unlimited
            timeout_ms: 5000,
            warmup_secs: 5,
            latency_threshold_ms: 500.0, // Per spec: <500ms p99 target
        }
    }
}

impl LoadTestConfig {
    /// Create config for stress testing
    #[must_use]
    pub fn for_stress_test() -> Self {
        Self {
            concurrency: 100,
            duration_secs: 300,
            target_rps: 0.0,
            timeout_ms: 10_000,
            warmup_secs: 10,
            latency_threshold_ms: 1000.0,
        }
    }

    /// Create config for latency-focused testing
    #[must_use]
    pub fn for_latency_test() -> Self {
        Self {
            concurrency: 1,
            duration_secs: 60,
            target_rps: 10.0, // Fixed rate
            timeout_ms: 2000,
            warmup_secs: 5,
            latency_threshold_ms: 200.0,
        }
    }

    /// Validate the configuration
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.concurrency > 0
            && self.duration_secs > 0
            && self.timeout_ms > 0
            && self.latency_threshold_ms > 0.0
    }
}

/// Results from a load test run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadTestResult {
    /// Total requests made
    pub total_requests: usize,
    /// Successful requests
    pub successful_requests: usize,
    /// Failed requests
    pub failed_requests: usize,
    /// Requests per second (achieved)
    pub rps_achieved: f64,
    /// Latency percentiles in milliseconds
    pub latency_p50_ms: f64,
    /// Latency p95 in milliseconds
    pub latency_p95_ms: f64,
    /// Latency p99 in milliseconds
    pub latency_p99_ms: f64,
    /// Maximum latency in milliseconds
    pub latency_max_ms: f64,
    /// Total data transferred in bytes
    pub data_transferred_bytes: u64,
    /// Test duration in seconds
    pub duration_secs: f64,
    /// Error rate (0.0-1.0)
    pub error_rate: f64,
    /// Whether the test passed the latency threshold
    pub passed_latency_threshold: bool,
}

impl LoadTestResult {
    /// Check if the load test passed all thresholds
    #[must_use]
    pub fn is_passing(&self) -> bool {
        self.passed_latency_threshold && self.error_rate < 0.01 // <1% error rate
    }

    /// Calculate throughput in MB/s
    #[must_use]
    pub fn throughput_mbps(&self) -> f64 {
        if self.duration_secs > 0.0 {
            (self.data_transferred_bytes as f64 / 1_000_000.0) / self.duration_secs
        } else {
            0.0
        }
    }
}

/// Load test runner
#[derive(Debug)]
pub struct LoadTestRunner {
    config: LoadTestConfig,
}

impl LoadTestRunner {
    /// Create a new load test runner
    #[must_use]
    pub fn new(config: LoadTestConfig) -> Self {
        Self { config }
    }

    /// Get the configuration
    #[must_use]
    pub fn config(&self) -> &LoadTestConfig {
        &self.config
    }

    /// Simulate a load test run (for testing purposes)
    ///
    /// In production, this would make actual HTTP requests.
    #[must_use]
    pub fn simulate_run(&self) -> LoadTestResult {
        // Simulate based on configuration
        let total_requests =
            (self.config.concurrency as f64 * self.config.duration_secs as f64 * 10.0) as usize;
        let error_count = total_requests / 100; // 1% error rate
        let successful = total_requests - error_count;

        // Simulate latencies based on concurrency
        // Higher concurrency = higher latencies
        let base_latency = 20.0; // 20ms base
        let concurrency_factor = (self.config.concurrency as f64).ln();

        let p50 = base_latency + concurrency_factor * 5.0;
        let p95 = p50 * 2.5;
        let p99 = p50 * 4.0;
        let max = p99 * 2.0;

        let duration = self.config.duration_secs as f64;
        let rps = if duration > 0.0 {
            total_requests as f64 / duration
        } else {
            0.0
        };

        LoadTestResult {
            total_requests,
            successful_requests: successful,
            failed_requests: error_count,
            rps_achieved: rps,
            latency_p50_ms: p50,
            latency_p95_ms: p95,
            latency_p99_ms: p99,
            latency_max_ms: max,
            data_transferred_bytes: (total_requests * 1024) as u64, // ~1KB per request
            duration_secs: duration,
            error_rate: error_count as f64 / total_requests as f64,
            passed_latency_threshold: p99 < self.config.latency_threshold_ms,
        }
    }
}

include!("load_testing_config.rs");
