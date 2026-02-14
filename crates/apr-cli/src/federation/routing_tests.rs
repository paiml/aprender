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

// =========================================================================
// RouterConfig tests
// =========================================================================

#[test]
fn test_router_config_default() {
    let config = RouterConfig::default();
    assert_eq!(config.max_candidates, 10);
    assert_eq!(config.min_score, 0.1);
    assert!(matches!(config.strategy, LoadBalanceStrategy::LeastLatency));
}

#[test]
fn test_router_config_clone() {
    let config = RouterConfig {
        max_candidates: 20,
        min_score: 0.5,
        strategy: LoadBalanceStrategy::ConsistentHash,
    };
    let cloned = config.clone();
    assert_eq!(cloned.max_candidates, 20);
    assert_eq!(cloned.min_score, 0.5);
}

// =========================================================================
// RouteDecision tests
// =========================================================================

#[test]
fn test_route_decision_construction() {
    let decision = RouteDecision {
        target: RouteTarget {
            node_id: NodeId("n1".to_string()),
            region_id: RegionId("r1".to_string()),
            endpoint: "http://n1:8080".to_string(),
            estimated_latency: Duration::from_millis(50),
            score: 0.95,
        },
        alternatives: vec![],
        reasoning: "best latency score".to_string(),
    };
    assert_eq!(decision.target.score, 0.95);
    assert!(decision.alternatives.is_empty());
}

#[test]
fn test_route_decision_with_alternatives() {
    let primary = RouteTarget {
        node_id: NodeId("n1".to_string()),
        region_id: RegionId("us-west".to_string()),
        endpoint: "http://n1:8080".to_string(),
        estimated_latency: Duration::from_millis(50),
        score: 0.95,
    };
    let alt = RouteTarget {
        node_id: NodeId("n2".to_string()),
        region_id: RegionId("eu-west".to_string()),
        endpoint: "http://n2:8080".to_string(),
        estimated_latency: Duration::from_millis(120),
        score: 0.7,
    };
    let decision = RouteDecision {
        target: primary,
        alternatives: vec![alt],
        reasoning: "latency-based".to_string(),
    };
    assert_eq!(decision.alternatives.len(), 1);
}

// =========================================================================
// Router with_policy tests
// =========================================================================

#[test]
fn test_router_with_custom_policy() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    let custom_policy =
        CompositePolicy::new().with_policy(super::super::policy::LatencyPolicy::default());

    let router = Router::new(RouterConfig::default(), catalog, health, circuit_breaker)
        .with_policy(custom_policy);

    assert_eq!(router.config.max_candidates, 10);
}

// =========================================================================
// RouterBuilder chaining tests
// =========================================================================

#[test]
fn test_router_builder_default() {
    let builder = RouterBuilder::default();
    assert!(builder.catalog.is_none());
    assert!(builder.health.is_none());
    assert!(builder.circuit_breaker.is_none());
    assert!(builder.policy.is_none());
}

#[test]
fn test_router_builder_with_catalog() {
    let catalog = Arc::new(ModelCatalog::new());
    let builder = RouterBuilder::new().catalog(Arc::clone(&catalog));
    assert!(builder.catalog.is_some());
}

#[test]
fn test_router_builder_with_health() {
    let health = Arc::new(HealthChecker::default());
    let builder = RouterBuilder::new().health(health);
    assert!(builder.health.is_some());
}

#[test]
fn test_router_builder_with_circuit_breaker() {
    let cb = Arc::new(CircuitBreaker::default());
    let builder = RouterBuilder::new().circuit_breaker(cb);
    assert!(builder.circuit_breaker.is_some());
}

#[test]
fn test_router_builder_with_policy() {
    let policy = CompositePolicy::new();
    let builder = RouterBuilder::new().policy(policy);
    assert!(builder.policy.is_some());
}

#[test]
fn test_router_builder_full_chain() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let cb = Arc::new(CircuitBreaker::default());

    let router = RouterBuilder::new()
        .config(RouterConfig {
            max_candidates: 20,
            min_score: 0.05,
            strategy: LoadBalanceStrategy::WeightedRandom,
        })
        .catalog(catalog)
        .health(health)
        .circuit_breaker(cb)
        .policy(CompositePolicy::enterprise_default())
        .build();

    assert_eq!(router.config.max_candidates, 20);
    assert_eq!(router.config.min_score, 0.05);
}

#[test]
fn test_router_builder_build_without_policy() {
    let router = RouterBuilder::new().build();
    // Should use enterprise default policy
    assert_eq!(router.config.max_candidates, 10);
}

// =========================================================================
// build_candidates tests
// =========================================================================

#[tokio::test]
async fn test_build_candidates_skips_circuit_open() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    // Register node
    catalog
        .register(
            ModelId("model".to_string()),
            NodeId("n1".to_string()),
            RegionId("us-west".to_string()),
            vec![Capability::Generate],
        )
        .await
        .expect("registration failed");

    health.register_node(NodeId("n1".to_string()));
    for _ in 0..3 {
        health.report_success(&NodeId("n1".to_string()), Duration::from_millis(50));
    }

    // Open circuit breaker for the node
    for _ in 0..5 {
        circuit_breaker.record_failure(&NodeId("n1".to_string()));
    }

    let router = Router::new(RouterConfig::default(), catalog, health, circuit_breaker);

    let request = InferenceRequest {
        capability: Capability::Generate,
        input: vec![],
        qos: QoSRequirements::default(),
        request_id: "test".to_string(),
        tenant_id: None,
    };

    // Should have no candidates since circuit is open
    let result = router.route(&request).await;
    assert!(matches!(result, Err(FederationError::NoCapacity(_))));
}

#[tokio::test]
async fn test_build_candidates_skips_unhealthy() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::new(super::super::health::HealthConfig {
        failure_threshold: 2,
        ..Default::default()
    }));
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    catalog
        .register(
            ModelId("model".to_string()),
            NodeId("unhealthy-node".to_string()),
            RegionId("us-west".to_string()),
            vec![Capability::Generate],
        )
        .await
        .expect("registration failed");

    health.register_node(NodeId("unhealthy-node".to_string()));
    // Make unhealthy
    for _ in 0..3 {
        health.report_failure(&NodeId("unhealthy-node".to_string()));
    }

    let router = Router::new(RouterConfig::default(), catalog, health, circuit_breaker);

    let request = InferenceRequest {
        capability: Capability::Generate,
        input: vec![],
        qos: QoSRequirements::default(),
        request_id: "test".to_string(),
        tenant_id: None,
    };

    // Unhealthy nodes are skipped
    let result = router.route(&request).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_build_candidates_wrong_capability() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    catalog
        .register(
            ModelId("whisper".to_string()),
            NodeId("n1".to_string()),
            RegionId("us-west".to_string()),
            vec![Capability::Transcribe],
        )
        .await
        .expect("registration failed");

    let router = Router::new(RouterConfig::default(), catalog, health, circuit_breaker);

    let request = InferenceRequest {
        capability: Capability::Generate, // Different capability
        input: vec![],
        qos: QoSRequirements::default(),
        request_id: "test".to_string(),
        tenant_id: None,
    };

    let result = router.route(&request).await;
    assert!(matches!(result, Err(FederationError::NoCapacity(_))));
}

// =========================================================================
// select_best strategy tests
// =========================================================================

#[tokio::test]
async fn test_route_with_round_robin_strategy() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    catalog
        .register(
            ModelId("model".to_string()),
            NodeId("n1".to_string()),
            RegionId("us-west".to_string()),
            vec![Capability::Generate],
        )
        .await
        .expect("registration failed");

    health.register_node(NodeId("n1".to_string()));
    for _ in 0..3 {
        health.report_success(&NodeId("n1".to_string()), Duration::from_millis(50));
    }

    let router = Router::new(
        RouterConfig {
            strategy: LoadBalanceStrategy::RoundRobin,
            ..Default::default()
        },
        catalog,
        health,
        circuit_breaker,
    );

    let request = InferenceRequest {
        capability: Capability::Generate,
        input: vec![],
        qos: QoSRequirements::default(),
        request_id: "test".to_string(),
        tenant_id: None,
    };

    let result = router.route(&request).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_route_with_consistent_hash_strategy() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    catalog
        .register(
            ModelId("model".to_string()),
            NodeId("n1".to_string()),
            RegionId("us-west".to_string()),
            vec![Capability::Embed],
        )
        .await
        .expect("registration failed");

    health.register_node(NodeId("n1".to_string()));
    for _ in 0..3 {
        health.report_success(&NodeId("n1".to_string()), Duration::from_millis(50));
    }

    let router = Router::new(
        RouterConfig {
            strategy: LoadBalanceStrategy::ConsistentHash,
            ..Default::default()
        },
        catalog,
        health,
        circuit_breaker,
    );

    let request = InferenceRequest {
        capability: Capability::Embed,
        input: vec![],
        qos: QoSRequirements::default(),
        request_id: "test".to_string(),
        tenant_id: None,
    };

    let result = router.route(&request).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_route_with_weighted_random_strategy() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    catalog
        .register(
            ModelId("model".to_string()),
            NodeId("n1".to_string()),
            RegionId("us-west".to_string()),
            vec![Capability::Generate],
        )
        .await
        .expect("registration failed");

    health.register_node(NodeId("n1".to_string()));
    for _ in 0..3 {
        health.report_success(&NodeId("n1".to_string()), Duration::from_millis(50));
    }

    let router = Router::new(
        RouterConfig {
            strategy: LoadBalanceStrategy::WeightedRandom,
            ..Default::default()
        },
        catalog,
        health,
        circuit_breaker,
    );

    let request = InferenceRequest {
        capability: Capability::Generate,
        input: vec![],
        qos: QoSRequirements::default(),
        request_id: "test".to_string(),
        tenant_id: None,
    };

    let result = router.route(&request).await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_route_with_least_connections_strategy() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    catalog
        .register(
            ModelId("model".to_string()),
            NodeId("n1".to_string()),
            RegionId("us-west".to_string()),
            vec![Capability::Generate],
        )
        .await
        .expect("registration failed");

    health.register_node(NodeId("n1".to_string()));
    for _ in 0..3 {
        health.report_success(&NodeId("n1".to_string()), Duration::from_millis(50));
    }

    let router = Router::new(
        RouterConfig {
            strategy: LoadBalanceStrategy::LeastConnections,
            ..Default::default()
        },
        catalog,
        health,
        circuit_breaker,
    );

    let request = InferenceRequest {
        capability: Capability::Generate,
        input: vec![],
        qos: QoSRequirements::default(),
        request_id: "test".to_string(),
        tenant_id: None,
    };

    let result = router.route(&request).await;
    assert!(result.is_ok());
}

// =========================================================================
// rank_candidates with policy rejection
// =========================================================================

#[tokio::test]
async fn test_get_candidates_with_rejected() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    // Register node
    catalog
        .register(
            ModelId("model".to_string()),
            NodeId("n1".to_string()),
            RegionId("us-west".to_string()),
            vec![Capability::Generate],
        )
        .await
        .expect("registration failed");

    health.register_node(NodeId("n1".to_string()));
    for _ in 0..3 {
        health.report_success(&NodeId("n1".to_string()), Duration::from_millis(50));
    }

    // Use a policy that rejects based on privacy
    let policy = CompositePolicy::new().with_policy(
        super::super::policy::PrivacyPolicy::default()
            .with_region(RegionId("us-west".to_string()), PrivacyLevel::Public),
    );

    let router = Router::new(RouterConfig::default(), catalog, health, circuit_breaker)
        .with_policy(policy);

    // Request requires Confidential, but us-west is Public
    let request = InferenceRequest {
        capability: Capability::Generate,
        input: vec![],
        qos: QoSRequirements {
            privacy: PrivacyLevel::Confidential,
            ..Default::default()
        },
        request_id: "test".to_string(),
        tenant_id: None,
    };

    let candidates = router
        .get_candidates(&request)
        .await
        .expect("get_candidates failed");
    assert!(!candidates.is_empty());
    // Candidate should be marked as not eligible
    assert!(!candidates[0].eligible);
    assert!(candidates[0].rejection_reason.is_some());
}

// =========================================================================
// min_score filtering
// =========================================================================

#[tokio::test]
async fn test_route_min_score_filters_candidates() {
    let catalog = Arc::new(ModelCatalog::new());
    let health = Arc::new(HealthChecker::default());
    let circuit_breaker = Arc::new(CircuitBreaker::default());

    catalog
        .register(
            ModelId("model".to_string()),
            NodeId("n1".to_string()),
            RegionId("us-west".to_string()),
            vec![Capability::Generate],
        )
        .await
        .expect("registration failed");

    health.register_node(NodeId("n1".to_string()));
    // Leave at Unknown state (low health score)

    let router = Router::new(
        RouterConfig {
            min_score: 0.99, // Very high minimum
            ..Default::default()
        },
        catalog,
        health,
        circuit_breaker,
    );

    let request = InferenceRequest {
        capability: Capability::Generate,
        input: vec![],
        qos: QoSRequirements::default(),
        request_id: "test".to_string(),
        tenant_id: None,
    };

    let result = router.route(&request).await;
    // Might fail if score is below min_score
    // Unknown health = 0.3 score, which is below 0.99
    assert!(result.is_err());
}
