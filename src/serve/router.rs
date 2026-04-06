//! Spillover Router for Hybrid Cloud
//!
//! Implements dynamic routing logic to "spill" excess local traffic
//! to remote APIs when local queue depth exceeds thresholds.
//!
//! Toyota Way: "Heijunka" (Level Loading) across backends.

use crate::serve::backends::{PrivacyTier, ServingBackend};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// SERVE-RTR-001: Queue Metrics
// ============================================================================

/// Queue metrics for a backend
#[derive(Debug, Default)]
pub struct QueueMetrics {
    /// Current queue depth
    depth: AtomicUsize,
    /// Total requests processed
    total_requests: AtomicU64,
    /// Total latency in milliseconds (for averaging)
    total_latency_ms: AtomicU64,
    /// Requests in last window
    recent_requests: AtomicU64,
}

impl QueueMetrics {
    /// Create new metrics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment queue depth (request started)
    pub fn enqueue(&self) {
        self.depth.fetch_add(1, Ordering::SeqCst);
    }

    /// Decrement queue depth (request completed)
    pub fn dequeue(&self, latency_ms: u64) {
        self.depth.fetch_sub(1, Ordering::SeqCst);
        self.total_requests.fetch_add(1, Ordering::SeqCst);
        self.total_latency_ms.fetch_add(latency_ms, Ordering::SeqCst);
        self.recent_requests.fetch_add(1, Ordering::SeqCst);
    }

    /// Get current queue depth
    #[must_use]
    pub fn depth(&self) -> usize {
        self.depth.load(Ordering::SeqCst)
    }

    /// Get average latency in milliseconds
    #[must_use]
    pub fn avg_latency_ms(&self) -> f64 {
        let total = self.total_requests.load(Ordering::SeqCst);
        if total == 0 {
            0.0
        } else {
            self.total_latency_ms.load(Ordering::SeqCst) as f64 / total as f64
        }
    }

    /// Get total requests processed
    #[must_use]
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::SeqCst)
    }

    /// Reset recent request counter (for rate calculation)
    pub fn reset_recent(&self) {
        self.recent_requests.store(0, Ordering::SeqCst);
    }

    /// Get recent requests and reset
    #[must_use]
    pub fn take_recent(&self) -> u64 {
        self.recent_requests.swap(0, Ordering::SeqCst)
    }
}

// ============================================================================
// SERVE-RTR-002: Router Configuration
// ============================================================================

/// Spillover routing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Queue depth threshold before spillover
    pub spillover_threshold: usize,
    /// Maximum queue depth (reject requests)
    pub max_queue_depth: usize,
    /// Target latency SLA in milliseconds
    pub latency_sla_ms: u64,
    /// Privacy tier for routing decisions
    pub privacy: PrivacyTier,
    /// Preferred local backend
    pub local_backend: ServingBackend,
    /// Spillover backends in priority order
    pub spillover_backends: Vec<ServingBackend>,
    /// Enable spillover (can disable for testing)
    pub spillover_enabled: bool,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            spillover_threshold: 10,
            max_queue_depth: 50,
            latency_sla_ms: 1000, // 1 second
            privacy: PrivacyTier::Standard,
            local_backend: ServingBackend::Realizar,
            spillover_backends: vec![
                ServingBackend::Groq,      // Fastest
                ServingBackend::Together,  // Cost-effective
                ServingBackend::Fireworks, // Good balance
            ],
            spillover_enabled: true,
        }
    }
}

impl RouterConfig {
    /// Create sovereign config (no spillover to public APIs)
    #[must_use]
    pub fn sovereign() -> Self {
        Self {
            privacy: PrivacyTier::Sovereign,
            spillover_backends: vec![ServingBackend::Ollama, ServingBackend::LlamaCpp],
            spillover_enabled: true,
            ..Default::default()
        }
    }

    /// Create config with custom threshold
    #[must_use]
    pub fn with_threshold(threshold: usize) -> Self {
        Self { spillover_threshold: threshold, ..Default::default() }
    }
}

// ============================================================================
// SERVE-RTR-003: Routing Decision
// ============================================================================

/// Routing decision result
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingDecision {
    /// Route to primary local backend
    Local(ServingBackend),
    /// Spillover to remote backend
    Spillover(ServingBackend),
    /// Reject request (queue full)
    Reject(RejectReason),
}

/// Reason for rejection
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectReason {
    /// Queue depth exceeded
    QueueFull,
    /// No backends available
    NoBackends,
    /// Privacy constraint
    PrivacyViolation,
}

impl std::fmt::Display for RejectReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::QueueFull => write!(f, "Queue full, try again later"),
            Self::NoBackends => write!(f, "No backends available"),
            Self::PrivacyViolation => write!(f, "Request violates privacy constraints"),
        }
    }
}

// ============================================================================
// SERVE-RTR-004: Spillover Router
// ============================================================================

/// Spillover router for hybrid cloud routing
pub struct SpilloverRouter {
    config: RouterConfig,
    /// Metrics per backend
    metrics: HashMap<ServingBackend, QueueMetrics>,
    /// Last metrics window time
    last_window: std::sync::RwLock<Instant>,
    /// Window duration for rate calculation
    window_duration: Duration,
}

impl SpilloverRouter {
    /// Create a new spillover router
    #[must_use]
    pub fn new(config: RouterConfig) -> Self {
        let mut metrics = HashMap::new();
        metrics.insert(config.local_backend, QueueMetrics::new());
        for backend in &config.spillover_backends {
            metrics.insert(*backend, QueueMetrics::new());
        }

        Self {
            config,
            metrics,
            last_window: std::sync::RwLock::new(Instant::now()),
            window_duration: Duration::from_secs(60),
        }
    }

    /// Create with default config
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(RouterConfig::default())
    }

    /// Route a request
    #[must_use]
    pub fn route(&self) -> RoutingDecision {
        // Get local queue metrics
        let local_depth = self.backend_depth(self.config.local_backend);

        // Check if we should reject
        if local_depth >= self.config.max_queue_depth {
            // Check spillover backends
            if self.config.spillover_enabled {
                if let Some(backend) = self.find_available_spillover() {
                    return RoutingDecision::Spillover(backend);
                }
            }
            return RoutingDecision::Reject(RejectReason::QueueFull);
        }

        // Check if we should spillover
        if self.config.spillover_enabled && local_depth >= self.config.spillover_threshold {
            if let Some(backend) = self.find_available_spillover() {
                return RoutingDecision::Spillover(backend);
            }
        }

        // Route to local
        RoutingDecision::Local(self.config.local_backend)
    }

    /// Get queue depth for a backend, returning 0 if unknown
    fn backend_depth(&self, backend: ServingBackend) -> usize {
        self.metrics.get(&backend).map_or(0, QueueMetrics::depth)
    }

    /// Find an available spillover backend
    fn find_available_spillover(&self) -> Option<ServingBackend> {
        self.config
            .spillover_backends
            .iter()
            .copied()
            .filter(|b| self.config.privacy.allows(*b))
            .find(|b| self.backend_depth(*b) < self.config.max_queue_depth)
    }

    /// Record request start
    pub fn start_request(&self, backend: ServingBackend) {
        if let Some(metrics) = self.metrics.get(&backend) {
            metrics.enqueue();
        }
    }

    /// Record request completion
    pub fn complete_request(&self, backend: ServingBackend, latency_ms: u64) {
        if let Some(metrics) = self.metrics.get(&backend) {
            metrics.dequeue(latency_ms);
        }
    }

    /// Get current queue depth for a backend
    #[must_use]
    pub fn queue_depth(&self, backend: ServingBackend) -> usize {
        self.backend_depth(backend)
    }

    /// Get total local queue depth
    #[must_use]
    pub fn local_queue_depth(&self) -> usize {
        self.queue_depth(self.config.local_backend)
    }

    /// Get router statistics
    #[must_use]
    pub fn stats(&self) -> RouterStats {
        let local_latency =
            self.metrics.get(&self.config.local_backend).map_or(0.0, QueueMetrics::avg_latency_ms);

        let spillover_depth: usize =
            self.config.spillover_backends.iter().map(|b| self.backend_depth(*b)).sum();

        RouterStats {
            local_queue_depth: self.local_queue_depth(),
            local_avg_latency_ms: local_latency,
            spillover_queue_depth: spillover_depth,
            spillover_threshold: self.config.spillover_threshold,
            max_queue_depth: self.config.max_queue_depth,
            spillover_enabled: self.config.spillover_enabled,
        }
    }

    /// Get config
    #[must_use]
    pub fn config(&self) -> &RouterConfig {
        &self.config
    }

    /// Check if currently spilling over
    #[must_use]
    pub fn is_spilling(&self) -> bool {
        self.local_queue_depth() >= self.config.spillover_threshold
    }
}

impl Default for SpilloverRouter {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// Router statistics
#[derive(Debug, Clone, Default)]
pub struct RouterStats {
    pub local_queue_depth: usize,
    pub local_avg_latency_ms: f64,
    pub spillover_queue_depth: usize,
    pub spillover_threshold: usize,
    pub max_queue_depth: usize,
    pub spillover_enabled: bool,
}

impl RouterStats {
    /// Queue utilization as percentage
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.max_queue_depth == 0 {
            0.0
        } else {
            (self.local_queue_depth as f64 / self.max_queue_depth as f64) * 100.0
        }
    }

    /// Check if approaching spillover
    #[must_use]
    pub fn near_spillover(&self) -> bool {
        self.local_queue_depth >= (self.spillover_threshold * 80 / 100)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    /// Create a router with custom threshold and max queue depth.
    /// Spillover is enabled by default.
    fn router_with(threshold: usize, max_depth: usize) -> SpilloverRouter {
        SpilloverRouter::new(RouterConfig {
            spillover_threshold: threshold,
            max_queue_depth: max_depth,
            ..Default::default()
        })
    }

    /// Enqueue `n` local (Realizar) requests on the router.
    fn fill_local_queue(router: &SpilloverRouter, n: usize) {
        for _ in 0..n {
            router.start_request(ServingBackend::Realizar);
        }
    }

    // ========================================================================
    // SERVE-RTR-001: Queue Metrics Tests
    // ========================================================================

    #[test]
    fn test_SERVE_RTR_001_metrics_new() {
        let metrics = QueueMetrics::new();
        assert_eq!(metrics.depth(), 0);
        assert_eq!(metrics.total_requests(), 0);
    }

    #[test]
    fn test_SERVE_RTR_001_enqueue_dequeue() {
        let metrics = QueueMetrics::new();
        metrics.enqueue();
        assert_eq!(metrics.depth(), 1);

        metrics.enqueue();
        assert_eq!(metrics.depth(), 2);

        metrics.dequeue(100);
        assert_eq!(metrics.depth(), 1);
        assert_eq!(metrics.total_requests(), 1);
    }

    #[test]
    fn test_SERVE_RTR_001_avg_latency() {
        let metrics = QueueMetrics::new();
        metrics.enqueue();
        metrics.dequeue(100);
        metrics.enqueue();
        metrics.dequeue(200);

        assert_eq!(metrics.avg_latency_ms(), 150.0);
    }

    #[test]
    fn test_SERVE_RTR_001_avg_latency_empty() {
        let metrics = QueueMetrics::new();
        assert_eq!(metrics.avg_latency_ms(), 0.0);
    }

    // ========================================================================
    // SERVE-RTR-002: Router Config Tests
    // ========================================================================

    #[test]
    fn test_SERVE_RTR_002_default_config() {
        let config = RouterConfig::default();
        assert_eq!(config.spillover_threshold, 10);
        assert_eq!(config.max_queue_depth, 50);
        assert!(config.spillover_enabled);
    }

    #[test]
    fn test_SERVE_RTR_002_sovereign_config() {
        let config = RouterConfig::sovereign();
        assert_eq!(config.privacy, PrivacyTier::Sovereign);
        // Should only have local backends
        for backend in &config.spillover_backends {
            assert!(backend.is_local());
        }
    }

    #[test]
    fn test_SERVE_RTR_002_custom_threshold() {
        let config = RouterConfig::with_threshold(5);
        assert_eq!(config.spillover_threshold, 5);
    }

    // ========================================================================
    // SERVE-RTR-003: Routing Decision Tests
    // ========================================================================

    #[test]
    fn test_SERVE_RTR_003_route_local_empty_queue() {
        let router = SpilloverRouter::with_defaults();
        let decision = router.route();
        assert!(matches!(decision, RoutingDecision::Local(_)));
    }

    #[test]
    fn test_SERVE_RTR_003_route_spillover_when_busy() {
        let router = router_with(2, 10);

        // Fill local queue past threshold
        fill_local_queue(&router, 3);

        let decision = router.route();
        assert!(matches!(decision, RoutingDecision::Spillover(_)));
    }

    #[test]
    fn test_SERVE_RTR_003_route_reject_when_full() {
        let router = SpilloverRouter::new(RouterConfig {
            spillover_threshold: 2,
            max_queue_depth: 3,
            spillover_enabled: false, // Disable spillover
            ..Default::default()
        });

        // Fill queue to max
        fill_local_queue(&router, 3);

        let decision = router.route();
        assert!(matches!(decision, RoutingDecision::Reject(RejectReason::QueueFull)));
    }

    // ========================================================================
    // SERVE-RTR-004: Spillover Router Tests
    // ========================================================================

    #[test]
    fn test_SERVE_RTR_004_queue_depth() {
        let router = SpilloverRouter::with_defaults();
        assert_eq!(router.local_queue_depth(), 0);

        router.start_request(ServingBackend::Realizar);
        assert_eq!(router.local_queue_depth(), 1);

        router.complete_request(ServingBackend::Realizar, 50);
        assert_eq!(router.local_queue_depth(), 0);
    }

    #[test]
    fn test_SERVE_RTR_004_is_spilling() {
        let router = router_with(2, 50);

        assert!(!router.is_spilling());

        fill_local_queue(&router, 2);

        assert!(router.is_spilling());
    }

    // ========================================================================
    // SERVE-RTR-005: Statistics Tests
    // ========================================================================

    #[test]
    fn test_SERVE_RTR_005_stats() {
        let router = SpilloverRouter::with_defaults();
        let stats = router.stats();
        assert_eq!(stats.local_queue_depth, 0);
        assert!(stats.spillover_enabled);
    }

    #[test]
    fn test_SERVE_RTR_005_utilization() {
        let router = router_with(10, 100);

        // Add 25 requests
        fill_local_queue(&router, 25);

        let stats = router.stats();
        assert_eq!(stats.utilization(), 25.0);
    }

    #[test]
    fn test_SERVE_RTR_005_near_spillover() {
        let router = router_with(10, 50);

        // 80% of threshold = 8
        fill_local_queue(&router, 8);

        let stats = router.stats();
        assert!(stats.near_spillover());
    }

    // ========================================================================
    // SERVE-RTR-006: Privacy Tests
    // ========================================================================

    #[test]
    fn test_SERVE_RTR_006_sovereign_no_public_spillover() {
        let router = SpilloverRouter::new(RouterConfig::sovereign());

        // Fill local queue
        fill_local_queue(&router, 15);

        let decision = router.route();
        // Should spillover to local backend only
        match decision {
            RoutingDecision::Spillover(backend) => assert!(backend.is_local()),
            RoutingDecision::Local(_) => {}  // Also acceptable
            RoutingDecision::Reject(_) => {} // If no local backends available
        }
    }

    // ========================================================================
    // SERVE-RTR-007: Reject Reason Display Tests
    // ========================================================================

    #[test]
    fn test_SERVE_RTR_007_reject_reason_display() {
        assert!(RejectReason::QueueFull.to_string().contains("Queue"));
        assert!(RejectReason::NoBackends.to_string().contains("backend"));
        assert!(RejectReason::PrivacyViolation.to_string().contains("privacy"));
    }

    // ========================================================================
    // Additional coverage tests
    // ========================================================================

    #[test]
    fn test_queue_metrics_reset_recent() {
        let metrics = QueueMetrics::new();
        metrics.enqueue();
        metrics.dequeue(50);
        assert_eq!(metrics.total_requests(), 1);
        metrics.reset_recent();
        // total_requests unchanged, but recent reset
        assert_eq!(metrics.total_requests(), 1);
    }

    #[test]
    fn test_queue_metrics_take_recent() {
        let metrics = QueueMetrics::new();
        metrics.enqueue();
        metrics.dequeue(100);
        metrics.enqueue();
        metrics.dequeue(200);
        let recent = metrics.take_recent();
        assert_eq!(recent, 2);
        // After take, recent should be 0
        let recent_after = metrics.take_recent();
        assert_eq!(recent_after, 0);
    }

    #[test]
    fn test_queue_metrics_default() {
        let metrics = QueueMetrics::default();
        assert_eq!(metrics.depth(), 0);
        assert_eq!(metrics.total_requests(), 0);
        assert_eq!(metrics.avg_latency_ms(), 0.0);
    }

    #[test]
    fn test_router_stats_default() {
        let stats = RouterStats::default();
        assert_eq!(stats.local_queue_depth, 0);
        assert_eq!(stats.local_avg_latency_ms, 0.0);
        assert_eq!(stats.spillover_queue_depth, 0);
        assert_eq!(stats.spillover_threshold, 0);
        assert_eq!(stats.max_queue_depth, 0);
        assert!(!stats.spillover_enabled);
    }

    #[test]
    fn test_router_stats_utilization_zero_max() {
        let stats = RouterStats { max_queue_depth: 0, local_queue_depth: 5, ..Default::default() };
        assert_eq!(stats.utilization(), 0.0);
    }

    #[test]
    fn test_router_stats_near_spillover_false() {
        let stats =
            RouterStats { spillover_threshold: 100, local_queue_depth: 10, ..Default::default() };
        assert!(!stats.near_spillover());
    }

    #[test]
    fn test_spillover_router_default() {
        let router = SpilloverRouter::default();
        assert_eq!(router.local_queue_depth(), 0);
        assert!(!router.is_spilling());
    }

    #[test]
    fn test_spillover_router_config_accessor() {
        let config = RouterConfig::with_threshold(42);
        let router = SpilloverRouter::new(config);
        assert_eq!(router.config().spillover_threshold, 42);
    }

    #[test]
    fn test_routing_decision_equality() {
        let local1 = RoutingDecision::Local(ServingBackend::Realizar);
        let local2 = RoutingDecision::Local(ServingBackend::Realizar);
        assert_eq!(local1, local2);

        let spillover1 = RoutingDecision::Spillover(ServingBackend::Groq);
        let spillover2 = RoutingDecision::Spillover(ServingBackend::Groq);
        assert_eq!(spillover1, spillover2);

        let reject1 = RoutingDecision::Reject(RejectReason::QueueFull);
        let reject2 = RoutingDecision::Reject(RejectReason::QueueFull);
        assert_eq!(reject1, reject2);
    }

    #[test]
    fn test_routing_decision_inequality() {
        let local = RoutingDecision::Local(ServingBackend::Realizar);
        let spillover = RoutingDecision::Spillover(ServingBackend::Groq);
        assert_ne!(local, spillover);
    }

    #[test]
    fn test_reject_reason_equality() {
        assert_eq!(RejectReason::QueueFull, RejectReason::QueueFull);
        assert_eq!(RejectReason::NoBackends, RejectReason::NoBackends);
        assert_eq!(RejectReason::PrivacyViolation, RejectReason::PrivacyViolation);
    }

    #[test]
    fn test_reject_reason_inequality() {
        assert_ne!(RejectReason::QueueFull, RejectReason::NoBackends);
        assert_ne!(RejectReason::NoBackends, RejectReason::PrivacyViolation);
    }

    #[test]
    fn test_queue_depth_for_unknown_backend() {
        let router = SpilloverRouter::with_defaults();
        // Query depth for backend not in config
        let depth = router.queue_depth(ServingBackend::Anthropic);
        assert_eq!(depth, 0);
    }

    #[test]
    fn test_start_request_unknown_backend() {
        let router = SpilloverRouter::with_defaults();
        // Should not panic for unknown backend
        router.start_request(ServingBackend::Anthropic);
        assert_eq!(router.queue_depth(ServingBackend::Anthropic), 0);
    }

    #[test]
    fn test_complete_request_unknown_backend() {
        let router = SpilloverRouter::with_defaults();
        // Should not panic for unknown backend
        router.complete_request(ServingBackend::Anthropic, 100);
    }

    #[test]
    fn test_router_stats_local_avg_latency() {
        let router = SpilloverRouter::with_defaults();
        router.start_request(ServingBackend::Realizar);
        router.complete_request(ServingBackend::Realizar, 100);
        router.start_request(ServingBackend::Realizar);
        router.complete_request(ServingBackend::Realizar, 200);

        let stats = router.stats();
        assert_eq!(stats.local_avg_latency_ms, 150.0);
    }

    #[test]
    fn test_router_config_latency_sla() {
        let config = RouterConfig::default();
        assert_eq!(config.latency_sla_ms, 1000);
    }

    #[test]
    fn test_router_config_local_backend() {
        let config = RouterConfig::default();
        assert_eq!(config.local_backend, ServingBackend::Realizar);
    }

    #[test]
    fn test_router_config_spillover_backends() {
        let config = RouterConfig::default();
        assert!(!config.spillover_backends.is_empty());
        assert!(config.spillover_backends.contains(&ServingBackend::Groq));
    }
}
