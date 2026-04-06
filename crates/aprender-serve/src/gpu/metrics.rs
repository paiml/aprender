//! Metrics & Health Monitoring (PMAT-802)
//!
//! M28: InferenceMetrics, HealthChecker, ShutdownCoordinator, GpuCompute, HybridScheduler.
#![allow(unreachable_pub)] // Items re-exported as pub from gpu/mod.rs

use super::MatmulOp;
use crate::error::{RealizarError, Result};
use crate::tensor::Tensor;

// =============================================================================
// M28: Metrics & Health Monitoring (Phase 19)
// =============================================================================

/// Inference metrics collector (M28 - IMP-067)
///
/// Collects and aggregates inference performance metrics including
/// latency distribution and throughput.
#[derive(Debug)]
pub struct InferenceMetrics {
    latencies: Vec<std::time::Duration>,
    total_tokens: u64,
    start_time: std::time::Instant,
}

impl InferenceMetrics {
    /// Create a new inference metrics collector
    #[must_use]
    pub fn new() -> Self {
        Self {
            latencies: Vec::new(),
            total_tokens: 0,
            start_time: std::time::Instant::now(),
        }
    }

    /// Get total number of recorded inferences
    #[must_use]
    pub fn total_inferences(&self) -> usize {
        self.latencies.len()
    }

    /// Get total number of tokens processed
    #[must_use]
    pub fn total_tokens(&self) -> u64 {
        self.total_tokens
    }

    /// Record an inference with its latency and token count
    pub fn record_inference(&mut self, latency: std::time::Duration, tokens: usize) {
        self.latencies.push(latency);
        self.total_tokens += tokens as u64;
    }

    /// Get latency at given percentile (0-100)
    ///
    /// Returns None if no inferences recorded.
    #[must_use]
    pub fn latency_percentile(&self, percentile: u8) -> Option<std::time::Duration> {
        if self.latencies.is_empty() {
            return None;
        }

        let mut sorted = self.latencies.clone();
        sorted.sort();

        let idx = ((percentile as usize) * sorted.len() / 100).min(sorted.len() - 1);
        Some(sorted[idx])
    }

    /// Calculate throughput in tokens per second
    #[must_use]
    pub fn throughput(&self) -> f64 {
        let elapsed = self.start_time.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_tokens as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        self.latencies.clear();
        self.total_tokens = 0;
        self.start_time = std::time::Instant::now();
    }
}

impl Default for InferenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for health check function
pub type HealthCheckFn = Box<dyn Fn() -> bool + Send + Sync>;

/// Health checker for system components (M28 - IMP-068)
///
/// Monitors health status of system components via registered check functions.
pub struct HealthChecker {
    checks: Vec<(String, HealthCheckFn)>,
    last_results: std::collections::HashMap<String, bool>,
}

impl std::fmt::Debug for HealthChecker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HealthChecker")
            .field("check_count", &self.checks.len())
            .field("last_results", &self.last_results)
            .finish()
    }
}

impl HealthChecker {
    /// Create a new health checker
    #[must_use]
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            last_results: std::collections::HashMap::new(),
        }
    }

    /// Get number of registered checks
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }

    /// Register a health check function
    pub fn register_check(&mut self, name: &str, check: HealthCheckFn) {
        self.checks.push((name.to_string(), check));
    }

    /// Run all health checks and return results
    pub fn check_all(&mut self) -> std::collections::HashMap<String, bool> {
        let mut results = std::collections::HashMap::new();
        for (name, check) in &self.checks {
            let healthy = check();
            results.insert(name.clone(), healthy);
        }
        self.last_results.clone_from(&results);
        results
    }

    /// Check if system is overall healthy (all checks pass)
    #[must_use]
    pub fn is_healthy(&self) -> bool {
        if self.checks.is_empty() {
            return true;
        }
        self.last_results.values().all(|&v| v)
    }

    /// Clear all registered checks
    pub fn clear(&mut self) {
        self.checks.clear();
        self.last_results.clear();
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for shutdown handler function
pub type ShutdownHandlerFn = Box<dyn Fn() + Send + Sync>;

/// Graceful shutdown coordinator (M28 - IMP-069)
///
/// Coordinates shutdown sequence with request draining and handler callbacks.
pub struct ShutdownCoordinator {
    shutting_down: bool,
    pending_requests: u32,
    handlers: Vec<ShutdownHandlerFn>,
}

impl std::fmt::Debug for ShutdownCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShutdownCoordinator")
            .field("shutting_down", &self.shutting_down)
            .field("pending_requests", &self.pending_requests)
            .field("handler_count", &self.handlers.len())
            .finish()
    }
}

impl ShutdownCoordinator {
    /// Create a new shutdown coordinator
    #[must_use]
    pub fn new() -> Self {
        Self {
            shutting_down: false,
            pending_requests: 0,
            handlers: Vec::new(),
        }
    }

    /// Check if shutdown has been initiated
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.shutting_down
    }

    /// Get number of pending requests
    #[must_use]
    pub fn pending_requests(&self) -> u32 {
        self.pending_requests
    }

    /// Get number of registered handlers
    #[must_use]
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Register a shutdown handler
    pub fn register_handler(&mut self, handler: ShutdownHandlerFn) {
        self.handlers.push(handler);
    }

    /// Mark that a request has started
    pub fn request_started(&mut self) {
        self.pending_requests += 1;
    }

    /// Mark that a request has completed
    pub fn request_completed(&mut self) {
        self.pending_requests = self.pending_requests.saturating_sub(1);
    }

    /// Initiate shutdown sequence
    ///
    /// Calls all registered handlers.
    pub fn initiate_shutdown(&mut self) {
        if self.shutting_down {
            return;
        }
        self.shutting_down = true;

        // Call all handlers
        for handler in &self.handlers {
            handler();
        }
    }

    /// Check if shutdown is complete (initiated + no pending requests)
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.shutting_down && self.pending_requests == 0
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    /// GPU compute via trueno's wgpu backend
    Gpu,
    /// CPU compute (fallback)
    Cpu,
    /// Auto-select best available backend
    #[default]
    Auto,
}

/// GPU compute context
///
/// Provides GPU-accelerated operations with automatic fallback to CPU
/// when GPU is not available.
pub struct GpuCompute {
    backend: ComputeBackend,
    gpu: Option<trueno::backends::gpu::GpuBackend>,
}

include!("gpu_compute.rs");
include!("buffer_pool.rs");
include!("metrics_inference.rs");
