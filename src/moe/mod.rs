//! Mixture-of-Experts (MOE) routing with Capacity Factor load balancing
//!
//! Implements inference-time load balancing per Fedus et al. (2022) Switch Transformers.
//!
//! ## Features
//!
//! - **Power of Two Choices**: Mitzenmacher (2001) load balancing algorithm
//! - **Capacity Factor Routing**: Fedus et al. (2022) expert capacity limits
//! - **Circuit Breaker**: Nygard (2018) failure isolation pattern
//! - **Heijunka Controller**: Toyota Production System load leveling via Little's Law
//! - **Andon Triggers**: Jidoka (built-in quality) automated quality control

use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    },
    time::{Duration, Instant},
};

use crate::error::{RealizarError, Result};

/// Configuration for capacity factor routing
#[derive(Debug, Clone)]
pub struct CapacityConfig {
    /// Maximum queue depth per expert
    pub capacity: usize,
    /// Number of experts
    pub num_experts: usize,
}

/// Capacity Factor Router for inference-time load balancing
pub struct CapacityFactorRouter {
    config: CapacityConfig,
    queue_depths: Vec<AtomicUsize>,
}

impl CapacityFactorRouter {
    /// Create new router
    #[must_use]
    pub fn new(config: CapacityConfig) -> Self {
        let queue_depths = (0..config.num_experts)
            .map(|_| AtomicUsize::new(0))
            .collect();
        Self {
            config,
            queue_depths,
        }
    }

    /// Route to best expert, falling back if at capacity
    ///
    /// # Errors
    ///
    /// Returns `MoeError` if score count doesn't match expert count.
    /// Returns `ExpertCapacityExceeded` if all top experts are at capacity.
    pub fn route(&self, scores: &[f32]) -> Result<usize> {
        if scores.len() != self.config.num_experts {
            return Err(RealizarError::MoeError(format!(
                "Expected {} scores, got {}",
                self.config.num_experts,
                scores.len()
            )));
        }

        let top2 = Self::top_k_indices(scores, 2);
        let primary = top2[0];

        if self.queue_depths[primary].load(Ordering::Relaxed) < self.config.capacity {
            Ok(primary)
        } else if top2.len() > 1 {
            Ok(top2[1])
        } else {
            Err(RealizarError::ExpertCapacityExceeded {
                expert_id: primary,
                queue_depth: self.queue_depths[primary].load(Ordering::Relaxed),
                capacity: self.config.capacity,
            })
        }
    }

    /// Record expert usage
    pub fn record_start(&self, expert_id: usize) {
        self.queue_depths[expert_id].fetch_add(1, Ordering::Relaxed);
    }

    /// Record expert completion
    pub fn record_end(&self, expert_id: usize) {
        self.queue_depths[expert_id].fetch_sub(1, Ordering::Relaxed);
    }

    /// Get queue depth for expert
    #[must_use]
    pub fn queue_depth(&self, expert_id: usize) -> usize {
        self.queue_depths[expert_id].load(Ordering::Relaxed)
    }

    fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

// ============================================================================
// Power of Two Choices Router (Mitzenmacher 2001)
// ============================================================================

/// Configuration for Power of Two Choices routing
#[derive(Debug, Clone)]
pub struct PowerOfTwoConfig {
    /// Number of experts available
    pub num_experts: usize,
    /// Maximum queue depth per expert
    pub capacity: usize,
}

/// Power of Two Choices Router per Mitzenmacher (2001)
///
/// Instead of always routing to the highest-scoring expert, this router
/// picks the top 2 experts by score and routes to the *least loaded* one.
/// This dramatically improves load balancing compared to simple top-k routing.
///
/// ## Algorithm
///
/// 1. Select top-2 experts by score
/// 2. Compare their current queue depths
/// 3. Route to the one with lower load (breaking ties by score)
///
/// ## Citation
///
/// Mitzenmacher, M. (2001). "The Power of Two Choices in Randomized Load Balancing."
/// IEEE Transactions on Parallel and Distributed Systems.
pub struct PowerOfTwoChoicesRouter {
    config: PowerOfTwoConfig,
    queue_depths: Vec<AtomicUsize>,
}

impl PowerOfTwoChoicesRouter {
    /// Create a new Power of Two Choices router
    #[must_use]
    pub fn new(config: PowerOfTwoConfig) -> Self {
        let queue_depths = (0..config.num_experts)
            .map(|_| AtomicUsize::new(0))
            .collect();
        Self {
            config,
            queue_depths,
        }
    }

    /// Route request using Power of Two Choices algorithm
    ///
    /// # Errors
    ///
    /// Returns error if score count doesn't match expert count or all top experts at capacity.
    pub fn route(&self, scores: &[f32]) -> Result<usize> {
        if scores.len() != self.config.num_experts {
            return Err(RealizarError::MoeError(format!(
                "Expected {} scores, got {}",
                self.config.num_experts,
                scores.len()
            )));
        }

        // Get top 2 experts by score
        let top2 = Self::top_k_indices(scores, 2);

        // Check both for capacity and pick least loaded
        let mut best_choice = None;
        let mut best_load = usize::MAX;

        for &expert_id in &top2 {
            let load = self.queue_depths[expert_id].load(Ordering::Relaxed);
            if load < self.config.capacity && load < best_load {
                best_load = load;
                best_choice = Some(expert_id);
            }
        }

        best_choice.ok_or_else(|| RealizarError::ExpertCapacityExceeded {
            expert_id: top2[0],
            queue_depth: self.queue_depths[top2[0]].load(Ordering::Relaxed),
            capacity: self.config.capacity,
        })
    }

    /// Record that an expert started processing a request
    pub fn record_start(&self, expert_id: usize) {
        self.queue_depths[expert_id].fetch_add(1, Ordering::Relaxed);
    }

    /// Record that an expert finished processing a request
    pub fn record_end(&self, expert_id: usize) {
        self.queue_depths[expert_id].fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current queue depth for an expert
    #[must_use]
    pub fn queue_depth(&self, expert_id: usize) -> usize {
        self.queue_depths[expert_id].load(Ordering::Relaxed)
    }

    fn top_k_indices(scores: &[f32], k: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(k).map(|(i, _)| i).collect()
    }
}

// ============================================================================
// Circuit Breaker (Nygard 2018)
// ============================================================================

/// Circuit breaker states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation - requests flow through
    Closed,
    /// Failure threshold exceeded - requests blocked
    Open,
    /// Testing if service recovered - limited requests allowed
    HalfOpen,
}

/// Configuration for circuit breaker
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Number of consecutive failures before opening
    pub failure_threshold: usize,
    /// Number of successes needed to close from half-open
    pub success_threshold: usize,
    /// Time in milliseconds before transitioning from open to half-open
    pub timeout_ms: u64,
}

/// Circuit Breaker per Nygard (2018) "Release It!"
///
/// Prevents cascading failures by isolating failing components.
///
/// ## State Machine
///
/// ```text
/// CLOSED --[failures >= threshold]--> OPEN
///    ^                                  |
///    |                                  v
///    +--[successes >= threshold]-- HALF_OPEN <--[timeout]--+
/// ```
///
/// ## Citation
///
/// Nygard, M. (2018). "Release It! Design and Deploy Production-Ready Software."
/// Pragmatic Bookshelf, 2nd Edition.
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    /// Protected mutable state
    state: Mutex<CircuitBreakerState>,
}

struct CircuitBreakerState {
    current: CircuitState,
    failure_count: usize,
    success_count: usize,
    last_failure_time: Option<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker
    #[must_use]
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            state: Mutex::new(CircuitBreakerState {
                current: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_time: None,
            }),
        }
    }

    /// Get current circuit state
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        let mut state = self.state.lock().expect("CircuitBreaker mutex poisoned");
        self.maybe_transition_to_half_open(&mut state);
        state.current
    }

    /// Check if request should be allowed
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn allow_request(&self) -> bool {
        let mut state = self.state.lock().expect("CircuitBreaker mutex poisoned");
        self.maybe_transition_to_half_open(&mut state);

        match state.current {
            CircuitState::Open => false,
            CircuitState::Closed | CircuitState::HalfOpen => true,
        }
    }

    /// Record a successful request
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn record_success(&self) {
        let mut state = self.state.lock().expect("CircuitBreaker mutex poisoned");
        self.maybe_transition_to_half_open(&mut state);

        match state.current {
            CircuitState::Closed => {
                state.failure_count = 0; // Reset on success
            },
            CircuitState::HalfOpen => {
                state.success_count += 1;
                if state.success_count >= self.config.success_threshold {
                    state.current = CircuitState::Closed;
                    state.failure_count = 0;
                    state.success_count = 0;
                }
            },
            CircuitState::Open => {}, // Shouldn't happen, but ignore
        }
    }

    /// Record a failed request
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn record_failure(&self) {
        let mut state = self.state.lock().expect("CircuitBreaker mutex poisoned");

        state.failure_count += 1;
        state.last_failure_time = Some(Instant::now());

        if state.failure_count >= self.config.failure_threshold {
            state.current = CircuitState::Open;
            state.success_count = 0;
        }
    }

    fn maybe_transition_to_half_open(&self, state: &mut CircuitBreakerState) {
        if state.current == CircuitState::Open {
            if let Some(last_failure) = state.last_failure_time {
                let timeout = Duration::from_millis(self.config.timeout_ms);
                if last_failure.elapsed() >= timeout {
                    state.current = CircuitState::HalfOpen;
                    state.success_count = 0;
                }
            }
        }
    }
}

// ============================================================================
// Heijunka Controller (Toyota Production System)
// ============================================================================

/// Configuration for Heijunka (load leveling) controller
#[derive(Debug, Clone)]
pub struct HeijunkaConfig {
    /// Target latency in milliseconds
    pub target_latency_ms: f64,
    /// Maximum allowed concurrency
    pub max_concurrency: usize,
}

/// Load shedding decision
#[derive(Debug, Clone)]
pub struct LoadSheddingDecision {
    /// Whether to shed load (reject requests)
    pub shed_load: bool,
    /// Recommended concurrency level
    pub recommended_concurrency: usize,
}

/// Heijunka Controller for load leveling via Little's Law
///
/// Little's Law: L = lambda * W
/// - L = average number of items in system (concurrency)
/// - lambda = arrival rate (requests per second)
/// - W = average wait time (latency)
///
/// Rearranging: `optimal_concurrency = arrival_rate * (latency_ms / 1000)`
///
/// ## Toyota Production System
///
/// Heijunka means "leveling" - smoothing production to avoid overburden.
/// In ML inference, this means maintaining steady throughput without latency spikes.
pub struct HeijunkaController {
    config: HeijunkaConfig,
}

include!("mod_optimal_concurrency_heijunka.rs");
