//! Health Checker - Monitors node health across the federation
//!
//! Tracks latency, throughput, error rates, and GPU utilization.
//! Supports both active probing and passive health updates.

use super::traits::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::RwLock;
use std::time::{Duration, Instant};

// ============================================================================
// Health Status (exported type)
// ============================================================================

/// Health status summary for external consumers
#[derive(Debug, Clone)]
pub struct HealthStatus {
    pub node_id: NodeId,
    pub state: HealthState,
    pub latency_p50: Duration,
    pub latency_p99: Duration,
    pub queue_depth: u32,
    pub last_updated: Instant,
}

impl From<NodeHealth> for HealthStatus {
    fn from(health: NodeHealth) -> Self {
        Self {
            node_id: health.node_id,
            state: health.status,
            latency_p50: health.latency_p50,
            latency_p99: health.latency_p99,
            queue_depth: health.queue_depth,
            last_updated: health.last_check,
        }
    }
}

// ============================================================================
// Health Checker Implementation
// ============================================================================

/// Configuration for health checking
#[derive(Debug, Clone)]
pub struct HealthConfig {
    /// How often to check health (default: 10s)
    pub check_interval: Duration,
    /// Timeout for health probes (default: 5s)
    pub probe_timeout: Duration,
    /// Number of failures before marking unhealthy (default: 3)
    pub failure_threshold: u32,
    /// Number of successes to recover from unhealthy (default: 2)
    pub recovery_threshold: u32,
    /// Latency threshold for degraded state (default: 1s)
    pub degraded_latency: Duration,
}

impl Default for HealthConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(10),
            probe_timeout: Duration::from_secs(5),
            failure_threshold: 3,
            recovery_threshold: 2,
            degraded_latency: Duration::from_secs(1),
        }
    }
}

/// Internal tracking state for a node
#[derive(Debug, Clone)]
struct NodeState {
    health: NodeHealth,
    consecutive_failures: u32,
    consecutive_successes: u32,
}

/// In-memory health checker
pub struct HealthChecker {
    config: HealthConfig,
    states: RwLock<HashMap<NodeId, NodeState>>,
    monitoring: AtomicBool,
}

impl HealthChecker {
    pub fn new(config: HealthConfig) -> Self {
        Self {
            config,
            states: RwLock::new(HashMap::new()),
            monitoring: AtomicBool::new(false),
        }
    }

    /// Register a node for health tracking
    pub fn register_node(&self, node_id: NodeId) {
        let mut states = self.states.write().expect("health lock poisoned");

        let health = NodeHealth {
            node_id: node_id.clone(),
            status: HealthState::Unknown,
            latency_p50: Duration::ZERO,
            latency_p99: Duration::ZERO,
            throughput: 0,
            gpu_utilization: None,
            queue_depth: 0,
            last_check: Instant::now(),
        };

        states.insert(
            node_id,
            NodeState {
                health,
                consecutive_failures: 0,
                consecutive_successes: 0,
            },
        );
    }

    /// Remove a node from health tracking
    pub fn deregister_node(&self, node_id: &NodeId) {
        let mut states = self.states.write().expect("health lock poisoned");
        states.remove(node_id);
    }

    /// Report a successful request (passive health update)
    pub fn report_success(&self, node_id: &NodeId, latency: Duration) {
        let mut states = self.states.write().expect("health lock poisoned");

        if let Some(state) = states.get_mut(node_id) {
            state.consecutive_failures = 0;
            state.consecutive_successes += 1;

            // Update latency (simple moving average)
            let old_latency = state.health.latency_p50;
            state.health.latency_p50 = Duration::from_millis(
                (old_latency.as_millis() as u64 * 9 + latency.as_millis() as u64) / 10,
            );

            state.health.last_check = Instant::now();

            // Update status based on latency
            if latency > self.config.degraded_latency {
                state.health.status = HealthState::Degraded;
            } else if state.consecutive_successes >= self.config.recovery_threshold {
                state.health.status = HealthState::Healthy;
            }
        }
    }

    /// Report a failed request (passive health update)
    pub fn report_failure(&self, node_id: &NodeId) {
        let mut states = self.states.write().expect("health lock poisoned");

        if let Some(state) = states.get_mut(node_id) {
            state.consecutive_successes = 0;
            state.consecutive_failures += 1;
            state.health.last_check = Instant::now();

            if state.consecutive_failures >= self.config.failure_threshold {
                state.health.status = HealthState::Unhealthy;
            } else {
                state.health.status = HealthState::Degraded;
            }
        }
    }

    /// Get all health statuses
    pub fn all_statuses(&self) -> Vec<HealthStatus> {
        let states = self.states.read().expect("health lock poisoned");
        states
            .values()
            .map(|s| HealthStatus::from(s.health.clone()))
            .collect()
    }

    /// Check if monitoring is active
    pub fn is_monitoring(&self) -> bool {
        self.monitoring.load(Ordering::SeqCst)
    }

    /// Get count of healthy nodes
    pub fn healthy_count(&self) -> usize {
        let states = self.states.read().expect("health lock poisoned");
        states
            .values()
            .filter(|s| s.health.status == HealthState::Healthy)
            .count()
    }

    /// Get total node count
    pub fn total_count(&self) -> usize {
        let states = self.states.read().expect("health lock poisoned");
        states.len()
    }
}

impl Default for HealthChecker {
    fn default() -> Self {
        Self::new(HealthConfig::default())
    }
}

impl HealthCheckerTrait for HealthChecker {
    fn check_node(&self, node_id: &NodeId) -> BoxFuture<'_, FederationResult<NodeHealth>> {
        let node_id = node_id.clone();

        Box::pin(async move {
            // In production, this would make an HTTP/gRPC health probe
            // For now, return cached state or Unknown
            let states = self.states.read().expect("health lock poisoned");

            states
                .get(&node_id)
                .map(|s| s.health.clone())
                .ok_or(FederationError::NodeUnreachable(node_id))
        })
    }

    fn get_cached_health(&self, node_id: &NodeId) -> Option<NodeHealth> {
        let states = self.states.read().expect("health lock poisoned");
        states.get(node_id).map(|s| s.health.clone())
    }

    fn start_monitoring(&self, _interval: Duration) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.monitoring.store(true, Ordering::SeqCst);
            // In production, this would spawn a background task
            // that periodically probes all registered nodes
        })
    }

    fn stop_monitoring(&self) -> BoxFuture<'_, ()> {
        Box::pin(async move {
            self.monitoring.store(false, Ordering::SeqCst);
        })
    }
}

// ============================================================================
// Circuit Breaker Implementation
// ============================================================================

/// Circuit breaker configuration
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Failures to open circuit (default: 5)
    pub failure_threshold: u32,
    /// Time before half-open probe (default: 30s)
    pub reset_timeout: Duration,
    /// Successes in half-open to close (default: 3)
    pub half_open_successes: u32,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 5,
            reset_timeout: Duration::from_secs(30),
            half_open_successes: 3,
        }
    }
}

/// Per-node circuit state
#[derive(Debug, Clone)]
struct CircuitBreakerState {
    state: CircuitState,
    failures: u32,
    successes_in_half_open: u32,
    last_failure: Option<Instant>,
}

/// Circuit breaker implementation
pub struct CircuitBreaker {
    config: CircuitBreakerConfig,
    states: RwLock<HashMap<NodeId, CircuitBreakerState>>,
}

impl CircuitBreaker {
    pub fn new(config: CircuitBreakerConfig) -> Self {
        Self {
            config,
            states: RwLock::new(HashMap::new()),
        }
    }

    fn get_or_create_state(&self, node_id: &NodeId) -> CircuitBreakerState {
        let states = self.states.read().expect("circuit breaker lock poisoned");
        states.get(node_id).cloned().unwrap_or(CircuitBreakerState {
            state: CircuitState::Closed,
            failures: 0,
            successes_in_half_open: 0,
            last_failure: None,
        })
    }

    fn update_state(&self, node_id: &NodeId, state: CircuitBreakerState) {
        let mut states = self.states.write().expect("circuit breaker lock poisoned");
        states.insert(node_id.clone(), state);
    }

    /// Get all circuit breaker states
    pub fn all_states(&self) -> Vec<(NodeId, CircuitState)> {
        let states = self.states.read().expect("circuit breaker lock poisoned");
        states
            .iter()
            .map(|(node_id, state)| (node_id.clone(), state.state))
            .collect()
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(CircuitBreakerConfig::default())
    }
}

impl CircuitBreakerTrait for CircuitBreaker {
    fn is_open(&self, node_id: &NodeId) -> bool {
        let state = self.get_or_create_state(node_id);

        match state.state {
            CircuitState::Open => {
                // Check if enough time has passed to try half-open
                if let Some(last_failure) = state.last_failure {
                    if last_failure.elapsed() >= self.config.reset_timeout {
                        // Transition to half-open
                        let mut new_state = state;
                        new_state.state = CircuitState::HalfOpen;
                        new_state.successes_in_half_open = 0;
                        self.update_state(node_id, new_state);
                        return false; // Allow one request through
                    }
                }
                true // Still open
            }
            CircuitState::HalfOpen => false, // Allow probe requests
            CircuitState::Closed => false,
        }
    }

    fn record_success(&self, node_id: &NodeId) {
        let mut state = self.get_or_create_state(node_id);

        match state.state {
            CircuitState::HalfOpen => {
                state.successes_in_half_open += 1;
                if state.successes_in_half_open >= self.config.half_open_successes {
                    // Transition back to closed
                    state.state = CircuitState::Closed;
                    state.failures = 0;
                    state.successes_in_half_open = 0;
                }
            }
            CircuitState::Closed => {
                // Reset failure counter on success
                state.failures = 0;
            }
            CircuitState::Open => {
                // Shouldn't happen, but reset just in case
                state.state = CircuitState::Closed;
                state.failures = 0;
            }
        }

        self.update_state(node_id, state);
    }

    fn record_failure(&self, node_id: &NodeId) {
        let mut state = self.get_or_create_state(node_id);
        state.failures += 1;
        state.last_failure = Some(Instant::now());

        match state.state {
            CircuitState::Closed => {
                if state.failures >= self.config.failure_threshold {
                    state.state = CircuitState::Open;
                }
            }
            CircuitState::HalfOpen => {
                // Any failure in half-open goes back to open
                state.state = CircuitState::Open;
                state.successes_in_half_open = 0;
            }
            CircuitState::Open => {
                // Already open, just update last_failure time
            }
        }

        self.update_state(node_id, state);
    }

    fn state(&self, node_id: &NodeId) -> CircuitState {
        self.get_or_create_state(node_id).state
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_status_transitions() {
        let checker = HealthChecker::default();
        let node = NodeId("test-node".to_string());

        checker.register_node(node.clone());

        // Initially unknown
        let health = checker.get_cached_health(&node).expect("node should exist");
        assert_eq!(health.status, HealthState::Unknown);

        // Report successes to become healthy
        for _ in 0..3 {
            checker.report_success(&node, Duration::from_millis(50));
        }

        let health = checker.get_cached_health(&node).expect("node should exist");
        assert_eq!(health.status, HealthState::Healthy);
    }

    #[test]
    fn test_health_degraded_on_high_latency() {
        let checker = HealthChecker::default();
        let node = NodeId("slow-node".to_string());

        checker.register_node(node.clone());

        // Report high latency
        checker.report_success(&node, Duration::from_secs(2));

        let health = checker.get_cached_health(&node).expect("node should exist");
        assert_eq!(health.status, HealthState::Degraded);
    }

    #[test]
    fn test_health_unhealthy_on_failures() {
        let config = HealthConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let checker = HealthChecker::new(config);
        let node = NodeId("failing-node".to_string());

        checker.register_node(node.clone());

        // Report failures
        for _ in 0..3 {
            checker.report_failure(&node);
        }

        let health = checker.get_cached_health(&node).expect("node should exist");
        assert_eq!(health.status, HealthState::Unhealthy);
    }

    #[test]
    fn test_circuit_breaker_opens_on_failures() {
        let config = CircuitBreakerConfig {
            failure_threshold: 3,
            ..Default::default()
        };
        let breaker = CircuitBreaker::new(config);
        let node = NodeId("failing-node".to_string());

        // Initially closed
        assert!(!breaker.is_open(&node));
        assert_eq!(breaker.state(&node), CircuitState::Closed);

        // Record failures
        for _ in 0..3 {
            breaker.record_failure(&node);
        }

        assert!(breaker.is_open(&node));
        assert_eq!(breaker.state(&node), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_success_resets() {
        let breaker = CircuitBreaker::default();
        let node = NodeId("flaky-node".to_string());

        // Some failures
        breaker.record_failure(&node);
        breaker.record_failure(&node);

        // Success should reset
        breaker.record_success(&node);

        let state = breaker.get_or_create_state(&node);
        assert_eq!(state.failures, 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_recovery() {
        let config = CircuitBreakerConfig {
            failure_threshold: 2,
            half_open_successes: 2,
            reset_timeout: Duration::from_millis(10),
        };
        let breaker = CircuitBreaker::new(config);
        let node = NodeId("recovering-node".to_string());

        // Open the circuit
        breaker.record_failure(&node);
        breaker.record_failure(&node);
        assert_eq!(breaker.state(&node), CircuitState::Open);

        // Wait for reset timeout
        std::thread::sleep(Duration::from_millis(20));

        // Should transition to half-open on next check
        assert!(!breaker.is_open(&node));
        assert_eq!(breaker.state(&node), CircuitState::HalfOpen);

        // Successes in half-open should close
        breaker.record_success(&node);
        breaker.record_success(&node);
        assert_eq!(breaker.state(&node), CircuitState::Closed);
    }
}
