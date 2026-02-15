
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

    let router =
        Router::new(RouterConfig::default(), catalog, health, circuit_breaker).with_policy(policy);

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
