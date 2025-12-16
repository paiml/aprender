//! Router - Intelligent node selection for inference requests
//!
//! Combines catalog, health, and policy data to select the optimal
//! node for each request.

use super::catalog::ModelCatalog;
use super::health::{CircuitBreaker, HealthChecker};
use super::policy::CompositePolicy;
use super::traits::*;
use std::sync::Arc;
use std::time::Duration;

// ============================================================================
// Route Decision (exported type)
// ============================================================================

/// Final routing decision with reasoning
#[derive(Debug, Clone)]
pub struct RouteDecision {
    pub target: RouteTarget,
    pub alternatives: Vec<RouteTarget>,
    pub reasoning: String,
}

// ============================================================================
// Router Implementation
// ============================================================================

/// Configuration for the router
#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Maximum candidates to evaluate
    pub max_candidates: usize,
    /// Minimum score to be considered
    pub min_score: f64,
    /// Load balance strategy
    pub strategy: LoadBalanceStrategy,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            max_candidates: 10,
            min_score: 0.1,
            strategy: LoadBalanceStrategy::LeastLatency,
        }
    }
}

/// The main router implementation
pub struct Router {
    config: RouterConfig,
    catalog: Arc<ModelCatalog>,
    health: Arc<HealthChecker>,
    circuit_breaker: Arc<CircuitBreaker>,
    policy: CompositePolicy,
}

impl Router {
    pub fn new(
        config: RouterConfig,
        catalog: Arc<ModelCatalog>,
        health: Arc<HealthChecker>,
        circuit_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        Self {
            config,
            catalog,
            health,
            circuit_breaker,
            policy: CompositePolicy::enterprise_default(),
        }
    }

    /// Create router with custom policy
    pub fn with_policy(mut self, policy: CompositePolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Build candidates from catalog and health data
    fn build_candidates(&self, capability: &Capability) -> Vec<RouteCandidate> {
        // This would be async in production, but for simplicity we use sync here
        // The actual routing is async via the trait

        let mut candidates = Vec::new();

        // Get nodes from catalog that support this capability
        let entries = self.catalog.all_entries();

        for entry in entries {
            // Check if model supports the capability
            let has_capability = entry.metadata.capabilities.iter().any(|c| c == capability);
            if !has_capability {
                continue;
            }

            for deployment in &entry.deployments {
                // Check circuit breaker
                if self.circuit_breaker.is_open(&deployment.node_id) {
                    continue;
                }

                // Get health status
                let health = self
                    .health
                    .get_cached_health(&deployment.node_id)
                    .unwrap_or_else(|| NodeHealth {
                        node_id: deployment.node_id.clone(),
                        status: HealthState::Unknown,
                        latency_p50: Duration::from_secs(1),
                        latency_p99: Duration::from_secs(5),
                        throughput: 0,
                        gpu_utilization: None,
                        queue_depth: 0,
                        last_check: std::time::Instant::now(),
                    });

                // Skip unhealthy nodes
                if health.status == HealthState::Unhealthy {
                    continue;
                }

                let target = RouteTarget {
                    node_id: deployment.node_id.clone(),
                    region_id: deployment.region_id.clone(),
                    endpoint: deployment.endpoint.clone(),
                    estimated_latency: health.latency_p50,
                    score: 0.0, // Will be calculated by policy
                };

                let health_score = match health.status {
                    HealthState::Healthy => 1.0,
                    HealthState::Degraded => 0.5,
                    HealthState::Unknown => 0.3,
                    HealthState::Unhealthy => 0.0,
                };

                let scores = RouteScores {
                    latency_score: 1.0 - (health.latency_p50.as_millis() as f64 / 5000.0).min(1.0),
                    throughput_score: (health.throughput as f64 / 1000.0).min(1.0),
                    cost_score: 0.5,     // Would come from region pricing
                    locality_score: 0.5, // Would be calculated based on request origin
                    health_score,
                    total: 0.0,
                };

                candidates.push(RouteCandidate {
                    target,
                    scores,
                    eligible: true,
                    rejection_reason: None,
                });
            }
        }

        candidates
    }

    /// Score and rank candidates
    fn rank_candidates(&self, candidates: &mut [RouteCandidate], request: &InferenceRequest) {
        for candidate in candidates.iter_mut() {
            // Check eligibility
            if !self.policy.is_eligible(candidate, request) {
                candidate.eligible = false;
                candidate.rejection_reason = Some("Policy rejected".to_string());
                continue;
            }

            // Calculate score
            let score = self.policy.score(candidate, request);
            candidate.target.score = score;
            candidate.scores.total = score;
        }

        // Sort by score descending
        candidates.sort_by(|a, b| {
            b.scores
                .total
                .partial_cmp(&a.scores.total)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Select best candidate based on strategy
    fn select_best(&self, candidates: &[RouteCandidate]) -> Option<RouteCandidate> {
        let eligible: Vec<_> = candidates
            .iter()
            .filter(|c| c.eligible && c.scores.total >= self.config.min_score)
            .take(self.config.max_candidates)
            .collect();

        if eligible.is_empty() {
            return None;
        }

        match self.config.strategy {
            LoadBalanceStrategy::LeastLatency => {
                // Already sorted by score (which factors in latency)
                eligible.first().map(|c| (*c).clone())
            }
            LoadBalanceStrategy::LeastConnections => {
                // Would use queue_depth in production
                eligible.first().map(|c| (*c).clone())
            }
            LoadBalanceStrategy::RoundRobin => {
                // Would maintain counter state
                eligible.first().map(|c| (*c).clone())
            }
            LoadBalanceStrategy::WeightedRandom => {
                // Weighted random selection based on scores
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                let total_weight: f64 = eligible.iter().map(|c| c.scores.total).sum();
                if total_weight <= 0.0 {
                    return eligible.first().map(|c| (*c).clone());
                }

                // Simple pseudo-random using current time
                let mut hasher = DefaultHasher::new();
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos()
                    .hash(&mut hasher);
                let random = (hasher.finish() as f64) / (u64::MAX as f64);

                let target = random * total_weight;
                let mut cumulative = 0.0;

                for candidate in &eligible {
                    cumulative += candidate.scores.total;
                    if cumulative >= target {
                        return Some((*candidate).clone());
                    }
                }

                eligible.last().map(|c| (*c).clone())
            }
            LoadBalanceStrategy::ConsistentHash => {
                // Would hash request ID to consistent bucket
                eligible.first().map(|c| (*c).clone())
            }
        }
    }
}

impl RouterTrait for Router {
    fn route(&self, request: &InferenceRequest) -> BoxFuture<'_, FederationResult<RouteTarget>> {
        // Clone request to avoid lifetime issues with async block
        let request = request.clone();

        Box::pin(async move {
            let mut candidates = self.build_candidates(&request.capability);

            if candidates.is_empty() {
                return Err(FederationError::NoCapacity(request.capability.clone()));
            }

            self.rank_candidates(&mut candidates, &request);

            self.select_best(&candidates)
                .map(|c| c.target)
                .ok_or_else(|| FederationError::AllNodesUnhealthy(request.capability.clone()))
        })
    }

    fn get_candidates(
        &self,
        request: &InferenceRequest,
    ) -> BoxFuture<'_, FederationResult<Vec<RouteCandidate>>> {
        // Clone request to avoid lifetime issues with async block
        let request = request.clone();

        Box::pin(async move {
            let mut candidates = self.build_candidates(&request.capability);
            self.rank_candidates(&mut candidates, &request);
            Ok(candidates)
        })
    }
}

// ============================================================================
// Builder for Router
// ============================================================================

/// Builder for creating routers with custom configuration
pub struct RouterBuilder {
    config: RouterConfig,
    catalog: Option<Arc<ModelCatalog>>,
    health: Option<Arc<HealthChecker>>,
    circuit_breaker: Option<Arc<CircuitBreaker>>,
    policy: Option<CompositePolicy>,
}

impl RouterBuilder {
    pub fn new() -> Self {
        Self {
            config: RouterConfig::default(),
            catalog: None,
            health: None,
            circuit_breaker: None,
            policy: None,
        }
    }

    pub fn config(mut self, config: RouterConfig) -> Self {
        self.config = config;
        self
    }

    pub fn catalog(mut self, catalog: Arc<ModelCatalog>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    pub fn health(mut self, health: Arc<HealthChecker>) -> Self {
        self.health = Some(health);
        self
    }

    pub fn circuit_breaker(mut self, cb: Arc<CircuitBreaker>) -> Self {
        self.circuit_breaker = Some(cb);
        self
    }

    pub fn policy(mut self, policy: CompositePolicy) -> Self {
        self.policy = Some(policy);
        self
    }

    pub fn build(self) -> Router {
        let catalog = self
            .catalog
            .unwrap_or_else(|| Arc::new(ModelCatalog::new()));
        let health = self
            .health
            .unwrap_or_else(|| Arc::new(HealthChecker::default()));
        let circuit_breaker = self
            .circuit_breaker
            .unwrap_or_else(|| Arc::new(CircuitBreaker::default()));

        let router = Router::new(self.config, catalog, health, circuit_breaker);

        if let Some(policy) = self.policy {
            router.with_policy(policy)
        } else {
            router
        }
    }
}

impl Default for RouterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_test_router() -> (Router, Arc<ModelCatalog>, Arc<HealthChecker>) {
        let catalog = Arc::new(ModelCatalog::new());
        let health = Arc::new(HealthChecker::default());
        let circuit_breaker = Arc::new(CircuitBreaker::default());

        let router = Router::new(
            RouterConfig::default(),
            Arc::clone(&catalog),
            Arc::clone(&health),
            circuit_breaker,
        );

        (router, catalog, health)
    }

    #[tokio::test]
    async fn test_route_no_nodes() {
        let (router, _, _) = setup_test_router();

        let request = InferenceRequest {
            capability: Capability::Transcribe,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "test-1".to_string(),
            tenant_id: None,
        };

        let result = router.route(&request).await;
        assert!(matches!(result, Err(FederationError::NoCapacity(_))));
    }

    #[tokio::test]
    async fn test_route_single_node() {
        let (router, catalog, health) = setup_test_router();

        // Register a node
        catalog
            .register(
                ModelId("whisper".to_string()),
                NodeId("node-1".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Transcribe],
            )
            .await
            .expect("registration failed");

        health.register_node(NodeId("node-1".to_string()));
        health.report_success(&NodeId("node-1".to_string()), Duration::from_millis(50));

        let request = InferenceRequest {
            capability: Capability::Transcribe,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "test-2".to_string(),
            tenant_id: None,
        };

        let result = router.route(&request).await;
        assert!(result.is_ok());

        let target = result.expect("routing failed");
        assert_eq!(target.node_id, NodeId("node-1".to_string()));
    }

    #[tokio::test]
    async fn test_route_prefers_healthy() {
        let (router, catalog, health) = setup_test_router();

        // Register two nodes
        catalog
            .register(
                ModelId("llama".to_string()),
                NodeId("healthy-node".to_string()),
                RegionId("us-west".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("registration failed");

        catalog
            .register(
                ModelId("llama".to_string()),
                NodeId("degraded-node".to_string()),
                RegionId("us-east".to_string()),
                vec![Capability::Generate],
            )
            .await
            .expect("registration failed");

        // Make one healthy, one degraded
        health.register_node(NodeId("healthy-node".to_string()));
        health.register_node(NodeId("degraded-node".to_string()));

        for _ in 0..5 {
            health.report_success(
                &NodeId("healthy-node".to_string()),
                Duration::from_millis(20),
            );
            health.report_failure(&NodeId("degraded-node".to_string()));
        }

        let request = InferenceRequest {
            capability: Capability::Generate,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "test-3".to_string(),
            tenant_id: None,
        };

        let result = router.route(&request).await;
        assert!(result.is_ok());

        let target = result.expect("routing failed");
        assert_eq!(target.node_id, NodeId("healthy-node".to_string()));
    }

    #[tokio::test]
    async fn test_get_candidates_returns_all() {
        let (router, catalog, health) = setup_test_router();

        // Register multiple nodes
        for i in 0..3 {
            catalog
                .register(
                    ModelId("embed".to_string()),
                    NodeId(format!("node-{}", i)),
                    RegionId("us-west".to_string()),
                    vec![Capability::Embed],
                )
                .await
                .expect("registration failed");

            health.register_node(NodeId(format!("node-{}", i)));
            health.report_success(&NodeId(format!("node-{}", i)), Duration::from_millis(50));
        }

        let request = InferenceRequest {
            capability: Capability::Embed,
            input: vec![],
            qos: QoSRequirements::default(),
            request_id: "test-4".to_string(),
            tenant_id: None,
        };

        let candidates = router
            .get_candidates(&request)
            .await
            .expect("get_candidates failed");

        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn test_router_builder() {
        let router = RouterBuilder::new()
            .config(RouterConfig {
                max_candidates: 5,
                min_score: 0.2,
                strategy: LoadBalanceStrategy::RoundRobin,
            })
            .build();

        assert_eq!(router.config.max_candidates, 5);
        assert_eq!(router.config.min_score, 0.2);
    }
}
