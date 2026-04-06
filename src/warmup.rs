//! Model Warm-up and Pre-loading
//!
//! Reduces cold start latency by pre-loading models and running warm-up inference.
//!
//! ## Features
//!
//! - Pre-load models into memory before serving
//! - Run warm-up inference to JIT compile and optimize
//! - Validate model integrity before accepting traffic
//! - Background model loading for zero-downtime updates
//!
//! ## Example
//!
//! ```rust,ignore
//! use realizar::warmup::{WarmupConfig, ModelWarmer};
//!
//! let config = WarmupConfig::new()
//!     .with_warmup_iterations(3)
//!     .with_timeout(Duration::from_secs(30));
//!
//! let warmer = ModelWarmer::new(config);
//! warmer.warm_up(&model, &tokenizer).await?;
//! ```
//!
//! ## Toyota Way Principles
//!
//! - Heijunka: Level loading by pre-warming
//! - Jidoka: Validate model before serving
//! - Poka-Yoke: Prevent cold start errors

use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use serde::{Deserialize, Serialize};

// ============================================================================
// WARM-001: Configuration
// ============================================================================

/// Configuration for model warm-up
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupConfig {
    /// Number of warm-up inference iterations
    pub warmup_iterations: usize,
    /// Timeout for warm-up process
    pub timeout: Duration,
    /// Sample prompt for warm-up inference
    pub sample_prompt: String,
    /// Maximum tokens for warm-up generation
    pub sample_max_tokens: usize,
    /// Validate model output during warm-up
    pub validate_output: bool,
    /// Run garbage collection after warm-up
    pub gc_after_warmup: bool,
    /// Log warm-up progress
    pub verbose: bool,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            warmup_iterations: 3,
            timeout: Duration::from_secs(60),
            sample_prompt: "Hello, world!".to_string(),
            sample_max_tokens: 10,
            validate_output: true,
            gc_after_warmup: true,
            verbose: false,
        }
    }
}

impl WarmupConfig {
    /// Create a new warm-up configuration
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of warm-up iterations
    #[must_use]
    pub fn with_warmup_iterations(mut self, n: usize) -> Self {
        self.warmup_iterations = n.max(1);
        self
    }

    /// Set warm-up timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set sample prompt for warm-up
    #[must_use]
    pub fn with_sample_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.sample_prompt = prompt.into();
        self
    }

    /// Set maximum tokens for warm-up generation
    #[must_use]
    pub fn with_sample_max_tokens(mut self, n: usize) -> Self {
        self.sample_max_tokens = n;
        self
    }

    /// Enable/disable output validation
    #[must_use]
    pub fn with_validate_output(mut self, validate: bool) -> Self {
        self.validate_output = validate;
        self
    }

    /// Enable/disable garbage collection after warm-up
    #[must_use]
    pub fn with_gc_after_warmup(mut self, gc: bool) -> Self {
        self.gc_after_warmup = gc;
        self
    }

    /// Enable/disable verbose logging
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

// ============================================================================
// WARM-002: Warm-up Status
// ============================================================================

/// Status of warm-up process
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarmupStatus {
    /// Not yet started
    NotStarted,
    /// Currently warming up
    InProgress,
    /// Warm-up completed successfully
    Ready,
    /// Warm-up failed
    Failed,
    /// Warm-up timed out
    TimedOut,
}

impl WarmupStatus {
    /// Check if model is ready to serve
    #[must_use]
    pub fn is_ready(&self) -> bool {
        matches!(self, Self::Ready)
    }

    /// Check if warm-up is still in progress
    #[must_use]
    pub fn is_in_progress(&self) -> bool {
        matches!(self, Self::InProgress)
    }

    /// Check if warm-up failed
    #[must_use]
    pub fn has_failed(&self) -> bool {
        matches!(self, Self::Failed | Self::TimedOut)
    }
}

// ============================================================================
// WARM-003: Warm-up Result
// ============================================================================

/// Result of warm-up process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarmupResult {
    /// Final status
    pub status: WarmupStatus,
    /// Number of iterations completed
    pub iterations_completed: usize,
    /// Total warm-up duration
    pub total_duration: Duration,
    /// Average inference latency during warm-up
    pub avg_latency: Duration,
    /// First inference latency (cold)
    pub first_latency: Duration,
    /// Last inference latency (warm)
    pub last_latency: Duration,
    /// Speedup factor (first / last)
    pub speedup_factor: f64,
    /// Error message if failed
    pub error: Option<String>,
}

impl WarmupResult {
    /// Create a successful result
    #[must_use]
    pub fn success(iterations: usize, duration: Duration, latencies: &[Duration]) -> Self {
        let first = latencies.first().copied().unwrap_or(Duration::ZERO);
        let last = latencies.last().copied().unwrap_or(Duration::ZERO);
        let avg = if latencies.is_empty() {
            Duration::ZERO
        } else {
            Duration::from_nanos(
                latencies.iter().map(|d| d.as_nanos() as u64).sum::<u64>() / latencies.len() as u64,
            )
        };

        let speedup = if last.as_nanos() > 0 {
            first.as_nanos() as f64 / last.as_nanos() as f64
        } else {
            1.0
        };

        Self {
            status: WarmupStatus::Ready,
            iterations_completed: iterations,
            total_duration: duration,
            avg_latency: avg,
            first_latency: first,
            last_latency: last,
            speedup_factor: speedup,
            error: None,
        }
    }

    /// Create a failed result
    #[must_use]
    pub fn failed(error: impl Into<String>, iterations: usize, duration: Duration) -> Self {
        Self {
            status: WarmupStatus::Failed,
            iterations_completed: iterations,
            total_duration: duration,
            avg_latency: Duration::ZERO,
            first_latency: Duration::ZERO,
            last_latency: Duration::ZERO,
            speedup_factor: 0.0,
            error: Some(error.into()),
        }
    }

    /// Create a timed out result
    #[must_use]
    pub fn timed_out(iterations: usize, duration: Duration) -> Self {
        Self {
            status: WarmupStatus::TimedOut,
            iterations_completed: iterations,
            total_duration: duration,
            avg_latency: Duration::ZERO,
            first_latency: Duration::ZERO,
            last_latency: Duration::ZERO,
            speedup_factor: 0.0,
            error: Some("Warm-up timed out".to_string()),
        }
    }
}

// ============================================================================
// WARM-004: Model Health
// ============================================================================

/// Model health status for readiness probes
#[derive(Debug, Clone)]
pub struct ModelHealth {
    /// Whether model is ready
    ready: Arc<AtomicBool>,
    /// Warm-up status
    status: Arc<std::sync::RwLock<WarmupStatus>>,
    /// Total requests served
    requests_served: Arc<AtomicU64>,
    /// Failed requests
    requests_failed: Arc<AtomicU64>,
    /// Last health check timestamp
    last_health_check: Arc<std::sync::RwLock<Instant>>,
    /// Model load timestamp
    loaded_at: Instant,
}

impl Default for ModelHealth {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelHealth {
    /// Create new health tracker
    #[must_use]
    pub fn new() -> Self {
        Self {
            ready: Arc::new(AtomicBool::new(false)),
            status: Arc::new(std::sync::RwLock::new(WarmupStatus::NotStarted)),
            requests_served: Arc::new(AtomicU64::new(0)),
            requests_failed: Arc::new(AtomicU64::new(0)),
            last_health_check: Arc::new(std::sync::RwLock::new(Instant::now())),
            loaded_at: Instant::now(),
        }
    }

    /// Check if model is ready
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }

    /// Set ready status
    pub fn set_ready(&self, ready: bool) {
        self.ready.store(ready, Ordering::Release);
    }

    /// Get current status
    #[must_use]
    pub fn status(&self) -> WarmupStatus {
        *self.status.read().expect("test")
    }

    /// Set status
    pub fn set_status(&self, status: WarmupStatus) {
        *self.status.write().expect("test") = status;
        if status == WarmupStatus::Ready {
            self.set_ready(true);
        }
    }

    /// Record a successful request
    pub fn record_success(&self) {
        self.requests_served.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a failed request
    pub fn record_failure(&self) {
        self.requests_failed.fetch_add(1, Ordering::Relaxed);
    }

    /// Get total requests served
    #[must_use]
    pub fn total_requests(&self) -> u64 {
        self.requests_served.load(Ordering::Relaxed)
    }

    /// Get failed requests
    #[must_use]
    pub fn failed_requests(&self) -> u64 {
        self.requests_failed.load(Ordering::Relaxed)
    }

    /// Get error rate
    #[must_use]
    pub fn error_rate(&self) -> f64 {
        let total = self.total_requests();
        let failed = self.failed_requests();
        if total == 0 {
            0.0
        } else {
            failed as f64 / total as f64
        }
    }

    /// Update health check timestamp
    pub fn touch(&self) {
        *self.last_health_check.write().expect("test") = Instant::now();
    }

    /// Get uptime since model was loaded
    #[must_use]
    pub fn uptime(&self) -> Duration {
        self.loaded_at.elapsed()
    }

    /// Get time since last health check
    #[must_use]
    pub fn time_since_last_check(&self) -> Duration {
        self.last_health_check.read().expect("test").elapsed()
    }

    /// Generate health report
    #[must_use]
    pub fn report(&self) -> HealthReport {
        HealthReport {
            ready: self.is_ready(),
            status: self.status(),
            uptime_secs: self.uptime().as_secs_f64(),
            total_requests: self.total_requests(),
            failed_requests: self.failed_requests(),
            error_rate: self.error_rate(),
            time_since_last_check_secs: self.time_since_last_check().as_secs_f64(),
        }
    }
}

/// Health report for API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Is model ready to serve
    pub ready: bool,
    /// Current warm-up status
    pub status: WarmupStatus,
    /// Uptime in seconds
    pub uptime_secs: f64,
    /// Total requests served
    pub total_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
    /// Error rate (0.0 - 1.0)
    pub error_rate: f64,
    /// Time since last health check in seconds
    pub time_since_last_check_secs: f64,
}

// ============================================================================
// WARM-005: Warm-up Executor
// ============================================================================

/// Executes model warm-up process
#[derive(Debug, Clone)]
pub struct WarmupExecutor {
    config: WarmupConfig,
}

include!("warmup_config_simulate.rs");
include!("warmup_config.rs");
