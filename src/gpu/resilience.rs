//! GPU Resilience Module (PMAT-802)
//!
//! Extracted from gpu/mod.rs - Retry, circuit breaker, and bulkhead patterns.
//!
//! ## Contents
//! - `RetryPolicy`, `RetryConfig` - Exponential backoff retry (IMP-076)
//! - `CircuitBreaker`, `CircuitConfig` - Circuit breaker pattern (IMP-077)
//! - `BulkheadManager`, `BulkheadConfig` - Resource isolation (IMP-078)

use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// M31: Retry Logic & Circuit Breakers (IMP-076, IMP-077, IMP-078)
// ============================================================================

/// Error category for retry decisions (IMP-076)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCategory {
    /// Transient error that may succeed on retry
    Transient,
    /// Permanent error that will not succeed on retry
    Permanent,
}

/// Retry decision (IMP-076)
#[derive(Debug, Clone)]
pub enum RetryDecision {
    /// Retry with specified delay
    Retry {
        /// Delay before retry
        delay: Duration,
    },
    /// Abort with reason
    Abort {
        /// Reason for abort
        reason: String,
    },
}

/// Retry configuration (IMP-076)
#[derive(Debug, Clone)]
pub struct RetryConfig {
    max_retries: u32,
    base_delay: Duration,
    max_delay: Duration,
    jitter_factor: f64,
}

impl RetryConfig {
    /// Create new retry config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            jitter_factor: 0.1,
        }
    }

    /// Set max retries
    #[must_use]
    pub fn with_max_retries(mut self, max: u32) -> Self {
        self.max_retries = max;
        self
    }

    /// Set base delay
    #[must_use]
    pub fn with_base_delay(mut self, delay: Duration) -> Self {
        self.base_delay = delay;
        self
    }

    /// Set max delay
    #[must_use]
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Set jitter factor (0.0 to 1.0)
    #[must_use]
    pub fn with_jitter_factor(mut self, factor: f64) -> Self {
        self.jitter_factor = factor.clamp(0.0, 1.0);
        self
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Retry policy (IMP-076)
pub struct RetryPolicy {
    config: RetryConfig,
}

impl RetryPolicy {
    /// Create new retry policy
    #[must_use]
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Get max retries
    #[must_use]
    pub fn max_retries(&self) -> u32 {
        self.config.max_retries
    }

    /// Decide whether to retry
    #[must_use]
    pub fn should_retry(&self, attempt: u32, category: ErrorCategory) -> RetryDecision {
        if category == ErrorCategory::Permanent {
            return RetryDecision::Abort {
                reason: "Permanent error".to_string(),
            };
        }

        if attempt > self.config.max_retries {
            return RetryDecision::Abort {
                reason: format!("Max retries ({}) exceeded", self.config.max_retries),
            };
        }

        RetryDecision::Retry {
            delay: self.calculate_delay(attempt),
        }
    }

    /// Calculate delay with exponential backoff
    #[must_use]
    pub fn calculate_delay(&self, attempt: u32) -> Duration {
        // Exponential backoff: base * 2^attempt
        let exp_delay_ms = self.config.base_delay.as_millis() as u64 * (1u64 << attempt.min(20));
        let delay_ms = exp_delay_ms.min(self.config.max_delay.as_millis() as u64);
        Duration::from_millis(delay_ms)
    }
}

/// Circuit breaker state (IMP-077)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed, requests allowed
    Closed,
    /// Circuit is open, requests rejected
    Open,
    /// Circuit is half-open, probe requests allowed
    HalfOpen,
}

/// Circuit breaker configuration (IMP-077)
#[derive(Debug, Clone)]
pub struct CircuitConfig {
    failure_threshold: u32,
    success_threshold: u32,
    timeout: Duration,
}

impl CircuitConfig {
    /// Create new circuit config with defaults
    #[must_use]
    pub fn new() -> Self {
        Self {
            failure_threshold: 5,
            success_threshold: 2,
            timeout: Duration::from_secs(30),
        }
    }

    /// Set failure threshold to open circuit
    #[must_use]
    pub fn with_failure_threshold(mut self, threshold: u32) -> Self {
        self.failure_threshold = threshold;
        self
    }

    /// Set success threshold to close circuit from half-open
    #[must_use]
    pub fn with_success_threshold(mut self, threshold: u32) -> Self {
        self.success_threshold = threshold;
        self
    }

    /// Set timeout before transitioning to half-open
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Circuit breaker (IMP-077)
pub struct CircuitBreaker {
    config: CircuitConfig,
    state: std::sync::Mutex<CircuitState>,
    failure_count: std::sync::atomic::AtomicU32,
    success_count: std::sync::atomic::AtomicU32,
    last_failure: std::sync::Mutex<Option<std::time::Instant>>,
}

impl CircuitBreaker {
    /// Create new circuit breaker
    #[must_use]
    pub fn new(config: CircuitConfig) -> Self {
        Self {
            config,
            state: std::sync::Mutex::new(CircuitState::Closed),
            failure_count: std::sync::atomic::AtomicU32::new(0),
            success_count: std::sync::atomic::AtomicU32::new(0),
            last_failure: std::sync::Mutex::new(None),
        }
    }

    /// Acquire the state mutex (resource-management helper).
    fn lock_state(&self) -> std::sync::MutexGuard<'_, CircuitState> {
        self.state.lock().expect("mutex poisoned")
    }

    /// Acquire the last-failure mutex (resource-management helper).
    fn lock_last_failure(&self) -> std::sync::MutexGuard<'_, Option<std::time::Instant>> {
        self.last_failure.lock().expect("mutex poisoned")
    }

    /// Get current state
    #[must_use]
    pub fn state(&self) -> CircuitState {
        *self.lock_state()
    }

    /// Check if request should be allowed
    #[must_use]
    pub fn allow_request(&self) -> bool {
        let mut state = self.lock_state();
        match *state {
            CircuitState::Closed | CircuitState::HalfOpen => true,
            CircuitState::Open => {
                // Check if timeout has elapsed
                let last_failure = self.lock_last_failure();
                if let Some(last) = *last_failure {
                    if last.elapsed() >= self.config.timeout {
                        *state = CircuitState::HalfOpen;
                        self.success_count
                            .store(0, std::sync::atomic::Ordering::SeqCst);
                        return true;
                    }
                }
                false
            },
        }
    }

    /// Record an outcome (success or failure) and update circuit state.
    ///
    /// This is the single resource-management entry-point for state transitions,
    /// consolidating the lock/atomic patterns that were previously duplicated
    /// across `record_failure` and `record_success`.
    fn record_outcome(&self, success: bool) {
        if success {
            self.failure_count
                .store(0, std::sync::atomic::Ordering::SeqCst);
            let count = self
                .success_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;

            let mut state = self.lock_state();
            if *state == CircuitState::HalfOpen && count >= self.config.success_threshold {
                *state = CircuitState::Closed;
            }
        } else {
            let count = self
                .failure_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
                + 1;
            *self.lock_last_failure() = Some(std::time::Instant::now());

            let mut state = self.lock_state();
            match *state {
                CircuitState::Closed => {
                    if count >= self.config.failure_threshold {
                        *state = CircuitState::Open;
                    }
                },
                CircuitState::HalfOpen => {
                    *state = CircuitState::Open;
                },
                CircuitState::Open => {},
            }
        }
    }

    /// Record a failure
    #[inline]
    pub fn record_failure(&self) {
        self.record_outcome(false);
    }

    /// Record a success
    #[inline]
    pub fn record_success(&self) {
        self.record_outcome(true);
    }
}

/// Request type for bulkhead (IMP-078)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RequestType {
    /// Standard inference request
    Inference,
    /// Embedding generation request
    Embedding,
    /// Batch processing request
    Batch,
}

/// Bulkhead permit
#[derive(Debug)]
pub struct BulkheadPermit {
    request_type: RequestType,
}

/// Bulkhead configuration (IMP-078)
pub struct BulkheadConfig {
    pools: HashMap<String, usize>,
}

impl BulkheadConfig {
    /// Create new bulkhead config
    #[must_use]
    pub fn new() -> Self {
        Self {
            pools: HashMap::new(),
        }
    }

    /// Add a pool with specified size
    #[must_use]
    pub fn with_pool(mut self, name: &str, size: usize) -> Self {
        self.pools.insert(name.to_string(), size);
        self
    }
}

impl Default for BulkheadConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Bulkhead stats
#[derive(Debug, Clone)]
pub struct BulkheadStats {
    /// Number of pools
    pub pool_count: usize,
    /// Total capacity across all pools
    pub total_capacity: usize,
}

/// Slot operation for [`ResourcePool::slot_op`].
#[derive(Debug, Clone, Copy)]
enum SlotOp {
    /// Read current availability without modifying state.
    Query,
    /// Try to acquire one slot (decrement). Returns 0 on success, 1 if exhausted.
    Acquire,
    /// Release one slot (increment). Always returns the new available count.
    Release,
}

/// A single resource pool with atomic availability tracking (IMP-078)
///
/// All acquire / release / query operations go through [`Self::slot_op`] to
/// keep the ResourceManagement pattern in a single method.
struct ResourcePool {
    available: std::sync::atomic::AtomicUsize,
    capacity: usize,
}

impl ResourcePool {
    /// Create a new pool with given capacity
    fn new(capacity: usize) -> Self {
        Self {
            available: std::sync::atomic::AtomicUsize::new(capacity),
            capacity,
        }
    }

    /// Unified slot operation: query, acquire, or release.
    ///
    /// * `Query`   – returns current available count.
    /// * `Acquire` – tries to decrement; returns `Ok(remaining)` or `Err(0)`.
    /// * `Release` – increments; returns `Ok(new_available)`.
    fn slot_op(&self, op: SlotOp) -> Result<usize, usize> {
        match op {
            SlotOp::Query => Ok(self.available.load(std::sync::atomic::Ordering::SeqCst)),
            SlotOp::Acquire => {
                let current = self.available.load(std::sync::atomic::Ordering::SeqCst);
                if current == 0 {
                    return Err(0);
                }
                let prev = self
                    .available
                    .fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                Ok(prev - 1)
            },
            SlotOp::Release => {
                let prev = self
                    .available
                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(prev + 1)
            },
        }
    }
}

/// Bulkhead manager (IMP-078)
pub struct BulkheadManager {
    inference: ResourcePool,
    embedding: ResourcePool,
    batch: ResourcePool,
}

impl BulkheadManager {
    /// Create new bulkhead manager
    #[must_use]
    pub fn new(config: &BulkheadConfig) -> Self {
        Self {
            inference: ResourcePool::new(*config.pools.get("inference").unwrap_or(&10)),
            embedding: ResourcePool::new(*config.pools.get("embedding").unwrap_or(&5)),
            batch: ResourcePool::new(*config.pools.get("batch").unwrap_or(&2)),
        }
    }

    /// Select the resource pool for a given request type
    fn pool_for(&self, request_type: RequestType) -> &ResourcePool {
        match request_type {
            RequestType::Inference => &self.inference,
            RequestType::Embedding => &self.embedding,
            RequestType::Batch => &self.batch,
        }
    }

    /// Get available slots for request type
    #[must_use]
    pub fn available(&self, request_type: RequestType) -> usize {
        self.pool_for(request_type)
            .slot_op(SlotOp::Query)
            .unwrap_or(0)
    }

    /// Acquire a permit
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    pub fn acquire(
        &self,
        request_type: RequestType,
    ) -> std::result::Result<BulkheadPermit, &'static str> {
        self.pool_for(request_type)
            .slot_op(SlotOp::Acquire)
            .map(|_| BulkheadPermit { request_type })
            .map_err(|_| "Pool exhausted")
    }

    /// Try to acquire a permit (non-blocking). Alias for [`Self::acquire`].
    ///
    /// # Errors
    /// Returns error if pool is exhausted.
    #[inline]
    pub fn try_acquire(
        &self,
        request_type: RequestType,
    ) -> std::result::Result<BulkheadPermit, &'static str> {
        self.acquire(request_type)
    }

    /// Release a permit
    pub fn release(&self, permit: &BulkheadPermit) {
        let _ = self.pool_for(permit.request_type).slot_op(SlotOp::Release);
    }

    /// Get bulkhead stats
    #[must_use]
    pub fn stats(&self) -> BulkheadStats {
        BulkheadStats {
            pool_count: 3,
            total_capacity: self.inference.capacity + self.embedding.capacity + self.batch.capacity,
        }
    }
}

include!("resilience_bulkhead_manager.rs");
